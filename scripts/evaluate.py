"""Evaluate cached probability maps on a manifest of aligned, binary PNG masks."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
from PIL import Image
from clipseg_project.metrics import measure, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--output", type=Path, default=Path("outputs/evaluation"))
    args = parser.parse_args()
    rows = []
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            probability = np.load(args.manifest.parent / row["probability"])
            mask = np.asarray(Image.open(args.manifest.parent / row["mask"]))
            if mask.ndim != 2 or not set(np.unique(mask)).issubset({0, 1, 255}):
                raise ValueError("Expected single-channel binary PNG mask, not semantic labels.")
            rows.append({"id": row["id"], **measure(probability, mask > 0, args.threshold)})
    result = {"manifest": str(args.manifest), "threshold": args.threshold, **summarize(rows)}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "per_query.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
