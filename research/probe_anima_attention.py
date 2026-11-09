"""Measure weighting through a real Anima cross-attention block's Q/K/V path."""

import argparse
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    import comfy.ops
    from comfy.ldm.cosmos.predict2 import Attention
    from safetensors import safe_open

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    with safe_open(bundle["checkpoint"], framework="pt", device="cpu") as source:
        matches = [key for key in source.keys() if key.endswith("blocks.0.cross_attn.q_proj.weight")]
        if len(matches) != 1:
            raise ValueError("Expected one unambiguous first cross-attention block.")
        key = matches[0]
        prefix = key[:-len("q_proj.weight")]
        state = {key[len(prefix):]: source.get_tensor(key) for key in source.keys() if key.startswith(prefix)}
    query_dim = state["q_proj.weight"].shape[1]
    context_dim = state["k_proj.weight"].shape[1]
    head_dim = state["k_norm.weight"].numel()
    heads = state["k_proj.weight"].shape[0] // head_dim
    attention = Attention(query_dim, context_dim, n_heads=heads, head_dim=head_dim,
                          device="cpu", dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    attention.load_state_dict(state, strict=True)
    attention.eval().requires_grad_(False).to(args.device)
    context = next(case["cond"] for case in bundle["cases"] if case["artist"] == "as109" and case["variant"] == "plain").to(args.device)
    report = json.loads((args.bundle.parent / "tokenization.json").read_text())
    case = next(case for case in report["records"] if case["artist"] == "as109" and case["position"] == "middle" and case["weight"] == 2)
    values = case["stable_t5_weights"]
    weights = torch.tensor(values + [1.] * (context.shape[1] - len(values)), device=args.device).reshape(1, -1, 1)
    mask = weights[0, :, 0] != 1
    generator = torch.Generator(device=args.device).manual_seed(321)
    queries = torch.randn((1, 8, query_dim), generator=generator, device=args.device)
    with torch.inference_mode():
        q, k, v = attention.compute_qkv(queries, context)
        qw, kw, vw = attention.compute_qkv(queries, context * weights)
        logits = torch.einsum("bqhd,bkhd->bhqk", q, k) / head_dim ** .5
        weighted_logits = torch.einsum("bqhd,bkhd->bhqk", qw, kw) / head_dim ** .5
        probabilities = logits.softmax(dim=-1)
        native_probabilities = weighted_logits.softmax(dim=-1)
        log_bias_probabilities = (logits + weights[0, :, 0].log()).softmax(dim=-1)
        mass = probabilities[..., mask].sum(-1)
        native_mass = native_probabilities[..., mask].sum(-1)
        biased_mass = log_bias_probabilities[..., mask].sum(-1)
        record = {
            "checkpoint": bundle["checkpoint"], "block": prefix, "dtype": "float32",
            "device": args.device, "heads": heads, "head_dim": head_dim,
            "weighted_token_indices": mask.nonzero().flatten().tolist(),
            "key_relative_l2": float((kw - k).norm() / k.norm()),
            "value_weighting_relative_error": float((vw - v * weights.unsqueeze(-1)).norm() / vw.norm()),
            "attention_probability_max_delta_native": float((native_probabilities - probabilities).abs().max()),
            "native_artist_attention_mass_ratio_mean": float((native_mass / mass).mean()),
            "log_bias_artist_attention_mass_ratio_mean": float((biased_mass / mass).mean()),
            "log_bias_artist_attention_mass_ratio_min": float((biased_mass / mass).min()),
            "limits": "Actual learned first-block projections with synthetic image queries; not a full denoising/style quality measurement. Log-bias is a mathematical candidate, not production behavior.",
        }
    args.output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
