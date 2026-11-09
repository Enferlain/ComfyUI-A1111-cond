"""Prepare the user's exact three-artist prompt; isolated research only.

Run with ComfyUI's Python, then pass prepared.pt to render_anima_emphasis.py.
Each row changes just one artist. All modes use the same FP32 adapter baseline.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ARTISTS = ("cutesexyrobutts", "quasarcake", "ssambatea")
PROMPT = r"@cutesexyrobutts, @quasarcake, @ssambatea, best quality, newest, 1girl, solo, umika \(blue archive\), grey hair, long hair, brown eyes, dog ears, floppy ears, blue halo, blue kimono, white apron, wa maid, maid headdress, holding takoyaki, eating, open mouth, hands up, small sweatdrop, upper body, from side, looking at viewer, outdoors"
NEGATIVE = "low quality, blurry, jpeg artifacts"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("D:/ComfyUI/models/diffusion_models/silvermoonmixAnima_v23.safetensors"))
    parser.add_argument("--te", type=Path, default=Path("D:/ComfyUI/models/text_encoders/qwen_3_06b_base.safetensors"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.modules.setdefault("server", None)
    import torch
    import torch.nn.functional as F
    import comfy.sd
    from comfy.ldm.anima.model import Anima
    from research.run_anima_emphasis import load_adapter, ids
    from research.anima_emphasis import single_pass_tokens
    from types import SimpleNamespace

    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    adapter, parameters = load_adapter(args.checkpoint, device)
    clip = comfy.sd.load_clip(ckpt_paths=[str(args.te)], embedding_directory=[],
                              model_options={"load_device": device, "offload_device": device, "dtype": torch.float32})
    tokenizer = clip.tokenizer
    qwen_cache, adapter_cache = {}, {}

    def prepare(tokens):
        key = tuple((int(x[0]), float(x[1])) for batch in tokens["qwen3_06b"] for x in batch)
        assert all(w == 1 for _, w in key), "Expected neutral Qwen weights"
        if key not in qwen_cache:
            qwen_cache[key] = clip.encode_from_tokens(tokens, return_dict=True)["cond"].to(device=device, dtype=torch.float32)
        t5_ids = torch.tensor([ids(tokens, "t5xxl")], device=device)
        cache_key = (key, tuple(ids(tokens, "t5xxl")))
        if cache_key not in adapter_cache:
            adapter_cache[cache_key] = adapter(qwen_cache[key], t5_ids)
        weights = torch.tensor([x[1] for x in tokens["t5xxl"][0]], device=device).reshape(1, -1, 1)
        weighted = adapter_cache[cache_key] * weights
        result = F.pad(weighted, (0, 0, 0, max(0, 512 - weighted.shape[1])))
        actual = Anima.preprocess_text_embeds(SimpleNamespace(llm_adapter=adapter), qwen_cache[key], t5_ids, weights)
        assert torch.equal(actual, result), "Native preprocessing parity failed"
        return result.cpu(), weights.flatten().tolist()

    cases, checks = [], []
    with torch.inference_mode():
        baseline_tokens = tokenizer.tokenize_with_weights(PROMPT)
        baseline, _ = prepare(baseline_tokens)
        negative, _ = prepare(tokenizer.tokenize_with_weights(NEGATIVE))
        for artist in ARTISTS:
            cases.append({"artist": artist, "variant": "plain", "prompt": PROMPT, "cond": baseline})
            for weight in (1, 2, 3):
                text = PROMPT.replace("@" + artist, f"(@{artist}:{weight})", 1)
                native_tokens = tokenizer.tokenize_with_weights(text)
                native, _ = prepare(native_tokens)
                stable_tokens, detail = single_pass_tokens(tokenizer, text)
                for stream in ("qwen3_06b", "t5xxl"):
                    assert ids(stable_tokens, stream) == ids(baseline_tokens, stream), "Stable IDs changed"
                stable, weights = prepare(stable_tokens)
                if weight == 1:
                    assert torch.equal(stable, baseline), "Stable weight1 identity failed"
                weights += [1.] * (baseline.shape[1] - len(weights))
                checks.append({"artist": artist, "weight": weight, "prompt": text,
                               "native_equals_plain": torch.equal(native, baseline),
                               "stable_equals_plain": torch.equal(stable, baseline),
                               "stable_ids_equal_plain": True, "native_preprocess_parity": True,
                               "weighted_token_indices": [i for i, w in enumerate(weights) if w != 1],
                               "boundary_crossings": detail["boundary_crossings"]})
                cases.append({"artist": artist, "variant": f"native_{weight}", "prompt": text, "cond": native})
                if weight != 1:
                    cases.extend([
                        {"artist": artist, "variant": f"singlepass_{weight}", "prompt": text, "cond": stable},
                        {"artist": artist, "variant": f"attention_bias_{weight}", "prompt": text,
                         "cond": baseline, "attention_weights": weights},
                    ])
                print(f"Prepared {artist} weight {weight}", flush=True)
        # Consistent column order for all artists.
        order = ["plain", "native_1", "native_2", "native_3", "singlepass_2", "singlepass_3", "attention_bias_2", "attention_bias_3"]
        cases.sort(key=lambda c: (ARTISTS.index(c["artist"]), order.index(c["variant"])))
        torch.save({"checkpoint": str(args.checkpoint), "cases": cases,
                    "negative_prompt": NEGATIVE, "negative": negative}, args.output / "prepared.pt")
    report = {"prompt": PROMPT, "negative_prompt": NEGATIVE, "checkpoint": str(args.checkpoint),
              "checkpoint_size": args.checkpoint.stat().st_size,
              "checkpoint_mtime_ns": args.checkpoint.stat().st_mtime_ns,
              "checkpoint_sha256": hashlib.file_digest(args.checkpoint.open("rb"), "sha256").hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "adapter_parameters": parameters, "dtype": "float32", "te": str(args.te),
              "records": checks, "image_cases": len(cases),
              "limits": "One seed; blends with other artist tags held at 1. No artist-reference fidelity scoring."}
    (args.output / "preparation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {len(cases)} cases", flush=True)


if __name__ == "__main__":
    main()
