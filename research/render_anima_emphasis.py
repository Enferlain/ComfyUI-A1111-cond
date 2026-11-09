"""Render a fixed-seed diagnostic grid from learned-adapter prepared tensors.

Uses the actual local diffusion model; no production nodes are modified.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vae", type=Path, default=Path("D:/ComfyUI/models/vae/qwen_image_vae.safetensors"))
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--attention-bias-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.modules.setdefault("server", None)
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    import comfy.sd
    import comfy.sample
    import comfy.utils
    from research.anima_attention_bias import temporary_attention_bias

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    if args.attention_bias_only:
        token_path = args.bundle.parent / "tokenization.json"
        tokens = json.loads(token_path.read_text())
        cases = []
        for case in bundle["cases"]:
            if case["variant"] != "plain":
                continue
            token_case = next(item for item in tokens["records"] if item["artist"] == case["artist"]
                              and item["position"] == "middle" and item["weight"] == 2)
            weights = case.get("attention_bias_weights")
            if weights is None:
                # Legacy bundles derive weights from the adjacent report, but
                # verify them against the weighted conditioning already stored.
                weights = token_case["stable_t5_weights"]
                weights = weights + [1.] * (case["cond"].shape[1] - len(weights))
                weighted = next(item for item in bundle["cases"]
                                if item["artist"] == case["artist"] and item["variant"] == "singlepass_2.0")
                expected = case["cond"] * torch.tensor(weights).reshape(1, -1, 1)
                if not torch.equal(expected, weighted["cond"]):
                    raise ValueError("Token-report weights do not match the prepared bundle.")
            cases.append(case | {"variant": "attention_bias_2.0", "prompt": token_case["text"],
                                 "attention_weights": weights})
        bundle["cases"] = cases
    args.output.mkdir(parents=True, exist_ok=True)
    plan = {"bundle_sha256": hashlib.sha256(args.bundle.read_bytes()).hexdigest(),
            "seed": args.seed, "steps": args.steps, "size": args.size, "cfg": args.cfg,
            "vae": str(args.vae), "sampler": "euler", "scheduler": "normal",
            "attention_bias_only": args.attention_bias_only}
    if args.attention_bias_only:
        plan["tokenization_sha256"] = hashlib.sha256(token_path.read_bytes()).hexdigest()
    plan_path = args.output / "plan.json"
    if plan_path.exists():
        previous_plan = json.loads(plan_path.read_text())
        previous_plan.setdefault("attention_bias_only", False)
        if previous_plan != plan:
            raise ValueError("Output directory belongs to a different experiment; use a new output path.")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"Loading diffusion checkpoint {bundle['checkpoint']}", flush=True)
    model = comfy.sd.load_diffusion_model(
        bundle["checkpoint"], model_options={"dtype": torch.bfloat16}, disable_dynamic=True
    )
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(args.vae)))
    # Anima's Cosmos backbone expects a single-frame video latent (B,C,T,H,W).
    latent = torch.zeros(1, 16, 1, args.size // 8, args.size // 8)
    noise = comfy.sample.prepare_noise(latent, args.seed)
    records = []
    with torch.inference_mode():
        for index, case in enumerate(bundle["cases"]):
            start = time.monotonic()
            filename = f"{index:02d}_{case['artist']}_{case['variant']}.png"
            print(f"Rendering {index + 1}/{len(bundle['cases'])}: {filename}", flush=True)
            latent_path = args.output / (filename + ".latent.pt")
            if latent_path.exists():
                samples = torch.load(latent_path, map_location="cpu", weights_only=True)
            else:
                override = temporary_attention_bias(model, case["attention_weights"]) if "attention_weights" in case else nullcontext()
                with override:
                    samples = comfy.sample.sample(
                        model, noise.clone(), args.steps, args.cfg, "euler", "normal",
                        [[case["cond"], {}]], [[bundle["negative"], {}]], latent.clone(),
                        disable_pbar=True, seed=args.seed,
                    )
                torch.save(samples.cpu(), latent_path)
            decoded = vae.decode(samples)
            if decoded.ndim == 5 and decoded.shape[:2] == (1, 1):
                pixels = decoded[0, 0].detach().cpu().numpy()
            elif decoded.ndim == 4 and decoded.shape[0] == 1:
                pixels = decoded[0].detach().cpu().numpy()
            else:
                raise ValueError(f"Unexpected single-frame VAE output shape: {tuple(decoded.shape)}")
            image = Image.fromarray(np.clip(pixels * 255, 0, 255).astype(np.uint8))
            image.save(args.output / filename)
            records.append({key: value for key, value in case.items() if key != "cond"} |
                           {"file": filename, "seconds": time.monotonic() - start})
            (args.output / "images.json").write_text(json.dumps({
                "checkpoint": bundle["checkpoint"], "vae": str(args.vae),
                "seed": args.seed, "steps": args.steps, "size": args.size,
                "cfg": args.cfg, "sampler": "euler", "scheduler": "normal",
                "negative_prompt": bundle["negative_prompt"], "records": records,
                "limits": "One checkpoint and seed; diagnostic differences, not artist/style quality ranking.",
            }, indent=2), encoding="utf-8")
    # Standard plotting-style contact sheet, without modifying source images.
    variants = list(dict.fromkeys(record["variant"] for record in records))
    artists = list(dict.fromkeys(record["artist"] for record in records))
    cell = 384
    grid = Image.new("RGB", (cell * len(variants), (cell + 48) * len(artists)), "white")
    draw = ImageDraw.Draw(grid)
    for record in records:
        row = artists.index(record["artist"])
        col = variants.index(record["variant"])
        image = Image.open(args.output / record["file"]).convert("RGB")
        image.thumbnail((cell, cell))
        grid.paste(image, (col * cell, row * (cell + 48) + 48))
        draw.text((col * cell + 5, row * (cell + 48) + 5),
                  f"@{record['artist']}\n{record['variant']}", fill="black")
    grid.save(args.output / "comparison.png")
    print(f"Saved {len(records)} images and comparison.png", flush=True)


if __name__ == "__main__":
    main()
