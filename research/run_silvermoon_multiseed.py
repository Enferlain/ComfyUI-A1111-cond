"""Repeat the preselected plain/value3/bias3 cases; retain all fixed seeds."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", action="store_true", help="Validate completed evidence and rebuild sheets without sampling")
    args = parser.parse_args()
    import torch
    from PIL import Image, ImageDraw

    root = Path(__file__).resolve().parent
    source = root / "results/silvermoon-v23-artists-2026-11-09"
    output = root / "results/silvermoon-v23-multiseed-2026-11-09"
    output.mkdir(parents=True, exist_ok=True)
    seeds = [12345, 12346, 12347, 12348, 12349]
    variants = ["plain", "singlepass_3", "attention_bias_3"]
    original = torch.load(source / "prepared.pt", map_location="cpu", weights_only=True)
    bundle = original | {"cases": [c for c in original["cases"] if c["variant"] in variants]}
    if len(bundle["cases"]) != 9:
        raise ValueError("Expected exactly three cases for each of three artists")
    expected = {(a, v) for a in ("cutesexyrobutts", "quasarcake", "ssambatea") for v in variants}
    if {(c["artist"], c["variant"]) for c in bundle["cases"]} != expected:
        raise ValueError("Unexpected artist/variant composition")
    plan = {"source_sha256": hashlib.sha256((source / "prepared.pt").read_bytes()).hexdigest(),
            "seeds": seeds, "variants": variants, "original_seed_directory": str(source / "images"),
            "settings": {"steps": 24, "size": 768, "cfg": 4, "sampler": "euler", "scheduler": "normal"}}
    plan_path = output / "experiment.json"
    source_plan = json.loads((source / "images/plan.json").read_text())
    if source_plan["bundle_sha256"] != plan["source_sha256"] or source_plan["seed"] != seeds[0]:
        raise ValueError("Reused images belong to a different bundle or seed")
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing experiment has a different plan")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    prepared = output / "prepared.pt"
    if not prepared.exists():
        temporary = output / "prepared.tmp.pt"
        torch.save(bundle, temporary)
        temporary.replace(prepared)
    else:
        saved = torch.load(prepared, map_location="cpu", weights_only=True)
        if saved["checkpoint"] != bundle["checkpoint"] or len(saved["cases"]) != len(bundle["cases"]):
            raise ValueError("Prepared subset mismatch")
        for a, b in zip(saved["cases"], bundle["cases"]):
            if not torch.equal(a["cond"], b["cond"]) or {k: v for k, v in a.items() if k != "cond"} != {k: v for k, v in b.items() if k != "cond"}:
                raise ValueError("Prepared subset contents changed")
        if not torch.equal(saved["negative"], bundle["negative"]):
            raise ValueError("Negative conditioning mismatch")
    for seed in ([] if args.summarize_only else seeds[1:]):
        subprocess.run([sys.executable, "-X", "utf8", "-u", str(root / "render_anima_emphasis.py"),
                        str(prepared), "--output", str(output / str(seed)), "--seed", str(seed),
                        "--steps", str(plan["settings"]["steps"]), "--size", str(plan["settings"]["size"]),
                        "--cfg", str(plan["settings"]["cfg"])], check=True)
    artists = list(dict.fromkeys(c["artist"] for c in bundle["cases"]))
    records = []
    for seed in seeds:
        directory = source / "images" if seed == seeds[0] else output / str(seed)
        manifest = json.loads((directory / "images.json").read_text())
        if manifest["seed"] != seed or any(manifest[k] != v for k, v in plan["settings"].items()):
            raise ValueError("Render settings mismatch")
        if manifest["checkpoint"] != bundle["checkpoint"] or manifest["vae"] != source_plan["vae"] or manifest["negative_prompt"] != bundle["negative_prompt"]:
            raise ValueError("Checkpoint, VAE or negative prompt mismatch")
        render_plan = json.loads((directory / "plan.json").read_text())
        expected_hash = plan["source_sha256"] if seed == seeds[0] else hashlib.sha256(prepared.read_bytes()).hexdigest()
        if render_plan["bundle_sha256"] != expected_hash:
            raise ValueError("Render plan conditioning mismatch")
        baseline_hashes = set()
        for artist in artists:
            for variant in variants:
                matches = [r for r in manifest["records"] if r["artist"] == artist and r["variant"] == variant]
                if len(matches) != 1:
                    raise ValueError("Missing or duplicated case")
                path = directory / matches[0]["file"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if variant == "plain":
                    baseline_hashes.add(digest)
                records.append({"seed": seed, "artist": artist, "variant": variant,
                                "file": str(path.relative_to(root.parent)), "sha256": digest})
        if len(baseline_hashes) != 1:
            raise ValueError("Repeated baseline renders differ")
    for artist in artists:
        cell, header = 384, 40
        sheet = Image.new("RGB", (cell * 3, (cell + header) * len(seeds)), "white")
        draw = ImageDraw.Draw(sheet)
        for record in records:
            if record["artist"] != artist:
                continue
            x, y = variants.index(record["variant"]), seeds.index(record["seed"])
            with Image.open(root.parent / record["file"]) as image:
                image = image.convert("RGB")
                image.thumbnail((cell, cell))
                sheet.paste(image, (x * cell, y * (cell + header) + header))
            draw.text((x * cell + 4, y * (cell + header) + 4), f"{record['seed']} @{artist}\n{record['variant']}", fill="black")
        sheet.save(output / f"comparison_{artist}.png")
        # Full-resolution top/bottom strips help inspect small generated text.
        strips = Image.new("RGB", (768 * 3, 296 * len(seeds)), "white")
        draw = ImageDraw.Draw(strips)
        for record in records:
            if record["artist"] != artist:
                continue
            x, y = variants.index(record["variant"]), seeds.index(record["seed"])
            with Image.open(root.parent / record["file"]) as image:
                if image.size != (768, 768):
                    raise ValueError("Unexpected image size for margin inspection")
                strips.paste(image.crop((0, 0, 768, 128)), (x * 768, y * 296 + 40))
                strips.paste(image.crop((0, 640, 768, 768)), (x * 768, y * 296 + 168))
            draw.text((x * 768 + 4, y * 296 + 4), f"{record['seed']} @{artist} {record['variant']} / top128 then bottom128", fill="black")
        strips.save(output / f"margins_{artist}.png")
    (output / "images.json").write_text(json.dumps({"plan": plan, "records": records,
        "baseline_repeats_identical_per_seed": True,
        "limits": "Five fixed seeds, one prompt/checkpoint. Shared plain images are not independent artist trials."}, indent=2), encoding="utf-8")
    print("Completed four additional seeds and three five-seed sheets", flush=True)


if __name__ == "__main__":
    main()
