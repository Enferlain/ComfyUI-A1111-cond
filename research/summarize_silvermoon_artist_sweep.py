"""Create per-artist 3x3 sheets and image-difference diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    import numpy as np
    from PIL import Image, ImageDraw

    manifest = json.loads((args.directory / "images.json").read_text())
    records = manifest["records"]
    artists = list(dict.fromkeys(record["artist"] for record in records))
    if len(artists) != 3 or len(records) != 24:
        raise ValueError("Expected the completed three-artist, 24-case sweep")
    metrics = []
    for artist in artists:
        selected = {r["variant"]: r for r in records if r["artist"] == artist}

        def pixels(variant):
            return np.asarray(Image.open(args.directory / selected[variant]["file"]).convert("RGB"), dtype=np.float32) / 255

        plain, native1 = pixels("plain"), pixels("native_1")
        for variant, record in selected.items():
            value = pixels(variant)
            metrics.append({"artist": artist, "variant": variant,
                            "rmse_vs_plain": float(np.sqrt(np.mean((value - plain) ** 2))),
                            "rmse_vs_native_1": float(np.sqrt(np.mean((value - native1) ** 2))),
                            "sha256": hashlib.sha256((args.directory / record["file"]).read_bytes()).hexdigest()})
        cell, header = 384, 44
        grid = Image.new("RGB", (cell * 3, (cell + header) * 3), "white")
        draw = ImageDraw.Draw(grid)
        rows = (("native_1", "native_2", "native_3"),
                ("plain", "singlepass_2", "singlepass_3"),
                ("plain", "attention_bias_2", "attention_bias_3"))
        for y, row in enumerate(rows):
            for x, variant in enumerate(row):
                image = Image.open(args.directory / selected[variant]["file"]).convert("RGB")
                image.thumbnail((cell, cell))
                grid.paste(image, (x * cell, y * (cell + header) + header))
                draw.text((x * cell + 5, y * (cell + header) + 5), f"@{artist}\n{variant}", fill="black")
        grid.save(args.directory / f"comparison_{artist}.png")
    # All duplicated baseline renders should be exactly identical.
    baseline_hashes = {m["sha256"] for m in metrics if m["variant"] == "plain"}
    if len(baseline_hashes) != 1:
        raise AssertionError("Identical baseline cases produced different images")
    (args.directory / "differences.json").write_text(json.dumps({
        "records": metrics, "baseline_repeats_identical": True,
        "limits": "Pixel differences measure change, not artist/style fidelity or strength calibration.",
    }, indent=2), encoding="utf-8")
    print("Saved three per-artist grids; repeated baselines identical")


if __name__ == "__main__":
    main()
