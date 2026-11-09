"""Exercise production node modes through real TE, adapter and Comfy sampling."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "custom_nodes"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("server", None)


def main():
    import torch
    import comfy.sd
    import comfy.sample
    from comfy.hooks import HookGroup, TransformerOptionsHook, EnumHookScope
    from A1111_Prompt_Node.nodes.prompt_node import A1111PromptNode
    from A1111_Prompt_Node.nodes.negative_node import A1111PromptNegative
    from research.anima_emphasis import single_pass_tokens
    from research.anima_attention_bias import temporary_attention_bias
    from research.prepare_silvermoon_artist_sweep import PROMPT

    records = []
    output = Path(__file__).parent / "results/anima-modes-live-2026-11-09.json"

    def check(name, condition, **details):
        records.append({"check": name, "passed": bool(condition), **details})
        output.write_text(json.dumps({"checkpoint": "silvermoonmixAnima_v23.safetensors",
            "size": 256, "steps": 2, "seed": 12345, "checks": records}, indent=2), encoding="utf-8")
        print(f"{'PASS' if condition else 'FAIL'} {name} {details}", flush=True)
        if not condition:
            raise AssertionError(name)

    clip = comfy.sd.load_clip(ckpt_paths=["D:/ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors"],
        embedding_directory=[], model_options={"load_device": torch.device("cuda:0"),
        "offload_device": torch.device("cpu"), "dtype": torch.float32})
    node = A1111PromptNode()
    negative_node = A1111PromptNegative()
    text = PROMPT.replace("@cutesexyrobutts", "(@cutesexyrobutts:3)", 1)

    def encode(text, mode, negative=False):
        return (negative_node if negative else node).encode(clip, text, anima_emphasis=mode)["result"][0]

    def equal(a, b):
        return torch.equal(a[0][0], b[0][0]) and set(a[0][1]) == set(b[0][1]) and all(
            torch.equal(v, b[0][1][k]) if isinstance(v, torch.Tensor) else v == b[0][1][k]
            for k, v in a[0][1].items())

    with torch.inference_mode():
        native = encode(text, "native")
        core = clip.encode_from_tokens(clip.tokenize(text), return_dict=True)
        check("native default parity", equal(native, [[core.pop("cond"), core]]))
        stable = encode(text, "single_pass")
        tokens, _ = single_pass_tokens(clip.tokenizer, text)
        experimental = clip.encode_from_tokens(tokens, return_dict=True)
        check("singlepass research parity", equal(stable, [[experimental.pop("cond"), experimental]]))
        baseline = encode(PROMPT, "native")
        neutral = PROMPT.replace("@cutesexyrobutts", "(@cutesexyrobutts:1)", 1)
        for mode in ("single_pass", "attention"):
            check(f"{mode} neutral identity", equal(encode(neutral, mode), baseline))
        attention = encode(text, "attention")
        check("attention carries scoped hook and neutral adapter weights", "hooks" in attention[0][1] and bool((attention[0][1]["t5xxl_weights"] == 1).all()))
        negative = encode("low quality, blurry, jpeg artifacts", "native", negative=True)
        weighted_negative = encode("(blurry:1.5)", "attention", negative=True)
        model = comfy.sd.load_diffusion_model("D:/ComfyUI/models/diffusion_models/silvermoonmixAnima_v23.safetensors",
            model_options={"dtype": torch.bfloat16}, disable_dynamic=True)
        original_ops = [block.cross_attn.attn_op for block in model.model.diffusion_model.blocks]
        latent = torch.zeros(1, 16, 1, 32, 32)
        noise = comfy.sample.prepare_noise(latent, 12345)

        def sample(positive, neg, target=model):
            return comfy.sample.sample(target, noise.clone(), 2, 4., "euler", "normal", positive, neg,
                                       latent.clone(), disable_pbar=True, seed=12345)

        before = sample(native, negative)
        samples = sample(attention, negative)
        check("production attention sampling finite", bool(torch.isfinite(samples).all()))
        # Separate reference CFG branches exactly as HookedOnly does, avoiding
        # batch-shape rounding differences in the parity comparison.
        reference_meta = attention[0][1].copy()
        reference_hooks = HookGroup()
        reference_hooks.add(TransformerOptionsHook({}, hook_scope=EnumHookScope.HookedOnly))
        reference_meta["hooks"] = reference_hooks
        weights = [t[1] for t in tokens["t5xxl"][0]]
        weights += [1.] * (max(512, len(weights)) - len(weights))
        with temporary_attention_bias(model, weights):
            reference = sample([[attention[0][0], reference_meta]], negative)
        error = float((samples - reference).float().norm() / reference.float().norm())
        from comfy.ldm.modules.attention import optimized_attention
        print(f"Default backend {optimized_attention.__name__}; research relative L2 {error}", flush=True)
        records.append({"diagnostic": "default backend vs research", "backend": optimized_attention.__name__, "relative_l2": error})
        # Controlled comparison: research used raw PyTorch SDPA. Preserve native
        # self-attention, but match that backend for the masked cross-attention.
        matched = model.clone()

        def pytorch_masked(original, q, k, v, heads, **kwargs):
            if kwargs.get("mask") is None:
                return original(q, k, v, heads, **kwargs)
            result = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=kwargs["mask"])
            return result if kwargs.get("skip_output_reshape") else result.transpose(1, 2).flatten(2)

        matched.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = pytorch_masked
        matched_samples = sample(attention, negative, matched)
        matched_error = float((matched_samples - reference).float().norm() / reference.float().norm())
        check("matched-backend production vs research sampling parity", torch.allclose(matched_samples, reference, atol=.005, rtol=.005), relative_l2=matched_error)
        check("attention changes output", not torch.equal(samples, sample(baseline, negative)))
        check("weighted negative attention sampling finite", bool(torch.isfinite(sample(attention, weighted_negative)).all()))
        check("singlepass sampling finite", bool(torch.isfinite(sample(stable, negative)).all()))
        check("native sampling unchanged after attention", torch.equal(before, sample(native, negative)))
        check("model attention objects unchanged", all(block.cross_attn.attn_op is op for block, op in zip(model.model.diffusion_model.blocks, original_ops)))
    output.write_text(json.dumps({"checkpoint": "silvermoonmixAnima_v23.safetensors",
        "size": 256, "steps": 2, "seed": 12345, "checks": records}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
