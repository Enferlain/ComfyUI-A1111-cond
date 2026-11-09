"""Build a contact sheet and pixel-difference diagnostics from completed runs."""

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

    sources = [args.directory / "images", args.directory / "images-attention-bias"]
    if any(not (folder / "images.json").exists() for folder in sources):
        raise ValueError("Complete both the value-scaling and attention-bias image runs first.")
    records = [(folder, record) for folder in sources
               for record in json.loads((folder / "images.json").read_text())["records"]]
    variants = ["plain", "native_1.0", "native_2.0", "singlepass_2.0",
                "singlepass_2.0_rms", "attention_bias_2.0"]
    artists = list(dict.fromkeys(record["artist"] for _, record in records))
    cell = 384
    grid = Image.new("RGB", (cell * len(variants), (cell + 48) * len(artists)), "white")
    draw = ImageDraw.Draw(grid)
    baselines = {}
    differences = []
    for folder, record in records:
        if record["variant"] == "plain":
            baselines[record["artist"]] = np.asarray(Image.open(folder / record["file"])).astype(float) / 255
    for folder, record in records:
        image = Image.open(folder / record["file"]).convert("RGB")
        values = np.asarray(image).astype(float) / 255
        delta = values - baselines[record["artist"]]
        differences.append({"artist": record["artist"], "variant": record["variant"],
                            "rmse_from_plain": float(np.sqrt(np.mean(delta ** 2))),
                            "equal_to_plain": bool(np.array_equal(values, baselines[record["artist"]])),
                            "file_sha256": hashlib.sha256((folder / record["file"]).read_bytes()).hexdigest()})
        col = variants.index(record["variant"])
        row = artists.index(record["artist"])
        image.thumbnail((cell, cell))
        grid.paste(image, (col * cell, row * (cell + 48) + 48))
        draw.text((col * cell + 5, row * (cell + 48) + 5),
                  f"@{record['artist']}\n{record['variant']}", fill="black")
    grid.save(args.directory / "comparison-all.png")
    (args.directory / "image_differences.json").write_text(json.dumps({
        "records": differences,
        "limits": "Pixel differences measure change, not fidelity to an artist or image quality.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(differences, indent=2), flush=True)


if __name__ == "__main__":
    main()
