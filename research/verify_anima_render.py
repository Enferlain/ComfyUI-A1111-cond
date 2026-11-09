"""Verify VAE range and bind legacy attention weights to stored conditioning."""

import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    sys.modules.setdefault("server", None)
    import numpy as np
    from PIL import Image
    import torch
    import comfy.sd
    import comfy.utils

    bundle_path = args.directory / "prepared.pt"
    token_path = args.directory / "tokenization.json"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    tokens = json.loads(token_path.read_text())
    binding = []
    for case in bundle["cases"]:
        if case["variant"] != "plain":
            continue
        token_case = next(item for item in tokens["records"] if item["artist"] == case["artist"]
                          and item["position"] == "middle" and item["weight"] == 2)
        weights = token_case["stable_t5_weights"]
        weights = torch.tensor(weights + [1.] * (case["cond"].shape[1] - len(weights))).reshape(1, -1, 1)
        weighted = next(item for item in bundle["cases"] if item["artist"] == case["artist"]
                        and item["variant"] == "singlepass_2.0")
        equal = torch.equal(case["cond"] * weights, weighted["cond"])
        if not equal:
            raise AssertionError("Legacy report weights do not match stored weighted conditioning.")
        binding.append({"artist": case["artist"], "weighted_conditioning_equal": equal})

    manifest = json.loads((args.directory / "images/images.json").read_text())
    first = manifest["records"][0]["file"]
    samples = torch.load(args.directory / "images" / (first + ".latent.pt"), map_location="cpu", weights_only=True)
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(manifest["vae"]))
    mapping = vae.process_output(torch.tensor([-1., 0., 1.])).tolist()
    with torch.inference_mode():
        decoded = vae.decode(samples)
    if not float(decoded.min()) >= 0 or not float(decoded.max()) <= 1:
        raise AssertionError("Decoded image is not in the expected [0,1] range.")
    pixels = decoded[0, 0] if decoded.ndim == 5 else decoded[0]
    expected_png = np.clip(pixels.cpu().numpy() * 255, 0, 255).astype(np.uint8)
    actual_png = np.asarray(Image.open(args.directory / "images" / first))
    report = {
        "bundle_sha256": hashlib.sha256(bundle_path.read_bytes()).hexdigest(),
        "tokenization_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
        "legacy_attention_weight_binding": binding,
        "vae_process_output_minus_one_zero_one": mapping,
        "decoded_shape": list(decoded.shape), "decoded_min": float(decoded.min()),
        "decoded_max": float(decoded.max()), "saved_png_matches_redecoded_latent": bool(np.array_equal(expected_png, actual_png)),
    }
    (args.directory / "render_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
