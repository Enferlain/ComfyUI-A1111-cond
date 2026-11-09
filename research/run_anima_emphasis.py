"""Run bounded real-tokenizer and learned-adapter emphasis experiments.

Example (ComfyUI's Python environment):
  python -X utf8 -u research/run_anima_emphasis.py --stage adapter --output <dir>

No production behavior is changed. Adapter weights are read from the specified
checkpoint; tensor metrics are not measures of visual style quality.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.anima_emphasis import single_pass_tokens


def ids(tokens, stream):
    return [int(item[0]) for batch in tokens[stream] for item in batch]


def token_cases(tokenizer, artists, concepts=()):
    cases = []
    terms = [(name, "artist", "@" + name) for name in artists]
    terms += [(name, "concept", name) for name in concepts]
    for artist, kind, tag in terms:
        for position, prefix, suffix in (
            ("start", "", ", 1girl, smile, garden"),
            ("middle", "1girl, ", ", smile, garden"),
        ):
            plain = prefix + tag + suffix
            baseline = tokenizer.tokenize_with_weights(plain)
            for weight in (1.0, 1.3, 2.0):
                text = prefix + f"({tag}:{weight})" + suffix
                native = tokenizer.tokenize_with_weights(text)
                stable, detail = single_pass_tokens(tokenizer, text)
                cases.append({
                    "artist": artist, "kind": kind, "position": position, "weight": weight,
                    "plain": plain, "text": text, "native": native,
                    "stable": stable, "baseline": baseline, "detail": detail,
                })
    return cases


def token_report(cases):
    records = []
    for case in cases:
        record = {key: case[key] for key in ("artist", "kind", "position", "weight", "plain", "text")}
        record.update(case["detail"])
        for stream in ("qwen3_06b", "t5xxl"):
            baseline = ids(case["baseline"], stream)
            native = ids(case["native"], stream)
            stable = ids(case["stable"], stream)
            record[stream] = {
                "baseline_ids": baseline, "native_ids": native, "stable_ids": stable,
                "native_ids_equal_plain": native == baseline,
                "stable_ids_equal_plain": stable == baseline,
            }
        record["stable_t5_weights"] = [float(item[1]) for item in case["stable"]["t5xxl"][0]]
        records.append(record)
    return records


def load_adapter(checkpoint, device):
    import torch
    import comfy.ops
    from comfy.ldm.anima.model import LLMAdapter
    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device="cpu") as source:
        keys = [key for key in source.keys() if "llm_adapter." in key]
        if not keys:
            raise ValueError("Checkpoint has no complete llm_adapter weights.")
        prefixes = {key.split("llm_adapter.")[0] + "llm_adapter." for key in keys}
        if len(prefixes) != 1:
            raise ValueError("Ambiguous adapter prefix in checkpoint.")
        prefix = prefixes.pop()
        state = {key[len(prefix):]: source.get_tensor(key) for key in keys}
    adapter = LLMAdapter(device="cpu", dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    adapter.load_state_dict(state, strict=True)
    adapter.eval().requires_grad_(False)
    return adapter.to(device=device), sum(value.numel() for value in state.values())


def adapter_report(cases, checkpoint, te_path, device, prepared_path=None):
    import torch
    import torch.nn.functional as F
    import comfy.sd
    from comfy.ldm.anima.model import Anima

    print(f"Loading learned adapter from {checkpoint.name}", flush=True)
    adapter, parameters = load_adapter(checkpoint, device)
    print(f"Adapter parameters: {parameters:,}; loading Qwen TE", flush=True)
    clip = comfy.sd.load_clip(
        ckpt_paths=[str(te_path)], embedding_directory=[],
        model_options={"load_device": device, "offload_device": device, "dtype": torch.float32},
    )
    qwen_cache = {}
    adapter_cache = {}

    def qwen_key(tokens):
        key = tuple((int(item[0]), float(item[1])) for batch in tokens["qwen3_06b"] for item in batch)
        if any(weight != 1.0 for _, weight in key):
            raise ValueError("This Anima experiment requires neutral native Qwen token weights.")
        return key

    def prepare(tokens):
        encoded_key = qwen_key(tokens)
        t5_key = tuple(ids(tokens, "t5xxl"))
        if encoded_key not in qwen_cache:
            output = clip.encode_from_tokens(tokens, return_dict=True)
            qwen_cache[encoded_key] = output["cond"].to(device=device, dtype=torch.float32)
        key = (encoded_key, t5_key)
        if key not in adapter_cache:
            cond = qwen_cache[encoded_key]
            text_ids = torch.tensor([t5_key], device=device, dtype=torch.long)
            adapter_cache[key] = adapter(cond, text_ids)
        base = adapter_cache[key]
        weights = torch.tensor(
            [float(item[1]) for item in tokens["t5xxl"][0]], device=device, dtype=base.dtype
        ).reshape(1, -1, 1)
        return base, base * weights, weights

    def pad(tensor):
        return F.pad(tensor, (0, 0, 0, max(0, 512 - tensor.shape[1])))

    def metrics(value, reference):
        a, b = pad(value).flatten().float(), pad(reference).flatten().float()
        return {
            "equal": bool(torch.equal(a, b)),
            "relative_l2": float((a - b).norm() / b.norm().clamp_min(1e-12)),
            "cosine": float(F.cosine_similarity(a, b, dim=0)),
            "rms_ratio": float(a.square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)),
        }

    records = []
    render_cases = []
    with torch.inference_mode():
        for index, case in enumerate(cases):
            _, plain, _ = prepare(case["baseline"])
            _, native, _ = prepare(case["native"])
            stable_base, stable, weights = prepare(case["stable"])
            rms_scale = stable_base.square().mean().sqrt() / stable.square().mean().sqrt().clamp_min(1e-12)
            mean_before = stable_base.mean()
            mean_after = stable.mean()
            mean_scale = float(mean_before / mean_after) if abs(float(mean_after)) > 1e-8 else None
            mask = (weights.squeeze(-1) == 1).unsqueeze(-1).expand_as(stable)
            # Verify our prepared native result against the actual model method,
            # without allocating the diffusion backbone.
            from types import SimpleNamespace
            tokens = case["native"]
            actual = Anima.preprocess_text_embeds(
                SimpleNamespace(llm_adapter=adapter),
                qwen_cache[qwen_key(tokens)],
                torch.tensor([ids(tokens, "t5xxl")], device=device),
                torch.tensor([item[1] for item in tokens["t5xxl"][0]], device=device).reshape(1, -1, 1),
            )
            if not torch.equal(actual, pad(native)):
                raise AssertionError("Prepared output does not match actual Anima preprocessing.")
            records.append({
                "artist": case["artist"], "kind": case["kind"], "position": case["position"], "weight": case["weight"],
                "native_vs_plain": metrics(native, plain),
                "stable_vs_plain": metrics(stable, plain),
                "stable_unweighted_tokens_unchanged": bool(torch.equal(stable[mask], stable_base[mask])),
                "rms_normalized_vs_plain": metrics(stable * rms_scale, plain),
                "rms_normalization_scale": float(rms_scale),
                "mean_preserving_scale": mean_scale,
                "mean_preserving_scale_flips_sign": mean_scale is not None and mean_scale < 0,
            })
            if prepared_path and case["position"] == "middle" and case["artist"] in ["as109", "fkey"]:
                variants = []
                if case["weight"] == 1:
                    variants = [("plain", plain), ("native_1.0", native)]
                elif case["weight"] == 2:
                    variants = [("native_2.0", native), ("singlepass_2.0", stable),
                                ("singlepass_2.0_rms", stable * rms_scale)]
                for name, tensor in variants:
                    render_cases.append({"artist": case["artist"], "variant": name,
                                         "prompt": case["plain"] if name == "plain" else case["text"],
                                         "cond": pad(tensor).cpu()})
                if case["weight"] == 2:
                    for saved in render_cases:
                        if saved["artist"] == case["artist"] and saved["variant"] == "plain":
                            values = weights.flatten().tolist()
                            saved["attention_bias_weights"] = values + [1.] * (saved["cond"].shape[1] - len(values))
            print(f"Adapter case {index + 1}/{len(cases)}: {case['artist']} {case['position']} {case['weight']}", flush=True)
        if prepared_path:
            negative_text = "low quality, blurry, jpeg artifacts"
            negative_tokens = clip.tokenizer.tokenize_with_weights(negative_text)
            _, negative, _ = prepare(negative_tokens)
            torch.save({"checkpoint": str(checkpoint), "cases": render_cases,
                        "negative_prompt": negative_text, "negative": pad(negative).cpu()}, prepared_path)
    return {
        "checkpoint": str(checkpoint), "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "adapter_parameters": parameters, "te": str(te_path),
        "device": str(device), "dtype": "float32", "qwen_encodes": len(qwen_cache),
        "adapter_encodes": len(adapter_cache), "records": records,
        "native_preprocess_parity": True,
        "native_preprocess_verified_indices": list(range(len(cases))),
        "qwen_weights_asserted_neutral": True,
        "limits": "Learned-adapter tensor diagnostics only; not image/style quality evidence.",
    }


def schedule_probe():
    import torch
    from A1111_Prompt_Node.hooks import A1111StepConditioningHook

    results = []
    for steps in (2, 3, 6, 30):
        embeddings = [(torch.full((1, 1, 1), float(i)), None) for i in range(steps)]
        hook = A1111StepConditioningHook(embeddings, default_steps=steps)
        sigmas = torch.arange(steps, -1, -1, dtype=torch.float32)
        observed = []
        for step in range(steps):
            output = hook.model_function_wrapper(
                lambda x, timestep, **c: c["c_crossattn"],
                {"input": torch.zeros(1), "timestep": sigmas[step],
                 "c": {"c_crossattn": torch.full((1, 1, 1), -1.0),
                       "transformer_options": {"sample_sigmas": sigmas}},
                 "cond_or_uncond": [0]},
            )
            observed.append(int(output.item()))
        expected = list(range(steps))
        results.append({"steps": steps, "expected": expected, "observed": observed,
                        "matches": observed == expected})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("tokens", "adapter"), default="tokens")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artists", nargs="+", default=["as109", "fkey", "mignon", "nnn yryr"])
    parser.add_argument("--concepts", nargs="*", default=[])
    parser.add_argument("--checkpoint", type=Path, default=Path("D:/ComfyUI/models/diffusion_models/animayume_v15Base.safetensors"))
    parser.add_argument("--te", type=Path, default=Path("D:/ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--save-prepared", action="store_true", help="Save a bounded image-comparison conditioning bundle")
    args = parser.parse_args()
    comfy_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(comfy_root / "custom_nodes"))
    sys.modules.setdefault("server", None)
    from comfy.text_encoders.anima import AnimaTokenizer

    args.output.mkdir(parents=True, exist_ok=True)
    cases = token_cases(AnimaTokenizer(), args.artists, args.concepts)
    records = token_report(cases)
    report = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256((Path(__file__).parent / "anima_emphasis.py").read_bytes()).hexdigest(),
        "artists": args.artists, "concepts": args.concepts, "records": records,
        "single_pass_ids_equal_plain": all(
            record[stream]["stable_ids_equal_plain"] for record in records
            for stream in ("qwen3_06b", "t5xxl")
        ),
        "native_neutral_syntax_changes": {
            stream: sum(not record[stream]["native_ids_equal_plain"] for record in records if record["weight"] == 1)
            for stream in ("qwen3_06b", "t5xxl")
        },
    }
    (args.output / "tokenization.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["single_pass_ids_equal_plain"]:
        raise AssertionError("Single-pass candidate changed plain-text token IDs.")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}), flush=True)
    if args.stage == "adapter":
        import torch
        prepared_path = args.output / "prepared.pt" if args.save_prepared else None
        report = adapter_report(cases, args.checkpoint, args.te, torch.device(args.device), prepared_path)
        report["legacy_schedule_probe"] = schedule_probe()
        (args.output / "adapter.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("Adapter and legacy schedule diagnostics saved.", flush=True)


if __name__ == "__main__":
    main()
