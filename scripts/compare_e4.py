"""Compare E4 arms on identical queries: paper-protocol metrics plus image-level paired bootstrap.

Each RESULT is an extracted `clipseg_e4_<arm>.zip` (it contains splits/<split>/paper_threshold_stats.npz and
sample_metrics.csv). The first arm is the reference (B1); the decision is read on val, test is only reported.

    python scripts/compare_e4.py --out outputs/e4-comparison B1=path/to/b1 E4c=path/to/e4c

An E5 (CLIPSeg + SAM) result stores one stats array per variant; select it with `#variant`, e.g.
    E1_SAM=outputs/kaggle-e5-sam-20261028#sam_boxpoint_mean  E1=outputs/kaggle-e5-sam-20261028#clipseg
"""
import argparse
import ast
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import simpson

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = np.linspace(0, 1, 51)[1:-1]
T03 = int(np.argmin(np.abs(THRESHOLDS - 0.3)))


def paper_metrics_function():
    """paper_metrics exactly as evaluated in the E1/E4 notebooks."""
    cells = json.loads((ROOT / "source" / "e1_train_eval_phrasecut.ipynb").read_text(encoding="utf-8"))["cells"]
    for cell in cells:
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if cell["cell_type"] == "code" and "def paper_metrics" in source:
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "paper_metrics")
            namespace = {"np": np, "simpson": simpson, "PAPER_THRESHOLDS": THRESHOLDS}
            exec(compile(ast.Module(body=[node], type_ignores=[]), "paper_metrics", "exec"), namespace)
            return namespace["paper_metrics"]
    raise ValueError("paper_metrics not found")


def load(folder, split):
    folder, _, key = str(folder).partition("#")
    key = key or "stats"
    folder = Path(folder) / "splits" / split
    data = np.load(folder / "paper_threshold_stats.npz")
    with (folder / "sample_metrics.csv").open(encoding="utf-8") as f:
        image_of = {row["task_id"]: int(row["image_id"]) for row in csv.DictReader(f)}
    task_ids = [str(t) for t in data["task_ids"]]
    assert len(task_ids) == len(set(task_ids)) == len(data[key])
    return dict(zip(task_ids, data[key])), image_of


def paired(reference, other, image_of, reps=5000, seed=42):
    """Per-query paper IoU at t=0.3, Δ = other − reference, bootstrap over query images."""
    ids = sorted(reference)
    iou = lambda s: s[0, T03] / (1 + s[0, T03] + s[1, T03] + s[2, T03])
    delta = np.array([iou(other[q]) - iou(reference[q]) for q in ids])
    images = sorted({image_of[q] for q in ids})
    index = {image: i for i, image in enumerate(images)}
    groups = np.array([index[image_of[q]] for q in ids])
    sums, counts = np.bincount(groups, delta, len(images)), np.bincount(groups, minlength=len(images))
    draws = np.random.default_rng(seed).integers(0, len(images), size=(reps, len(images)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return {"delta_miou_0.3": float(delta.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": int((delta > 1e-9).sum()), "ties": int((np.abs(delta) <= 1e-9).sum()), "losses": int((delta < -1e-9).sum()),
            "n_queries": len(ids), "n_images": len(images)}


def main():
    # Windows child processes can inherit cp1258 even when stdout is redirected.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--splits", default="val,test", help="comma-separated")
    parser.add_argument("arms", nargs="+", help="name=extracted_result_folder; the first is the reference")
    args = parser.parse_args()
    arms = dict(a.split("=", 1) for a in args.arms)
    names = list(arms)
    paper_metrics = paper_metrics_function()
    result, lines = {}, ["# E4 comparison", "", "Paper protocol at 352×352, threshold 0.3; Δ is paired per query vs "
                         f"{names[0]}, 95% CI from an image-level bootstrap.", ""]
    for split in args.splits.split(","):
        loaded = {name: load(folder, split) for name, folder in arms.items()}
        reference, image_of = loaded[names[0]]
        for name, (stats, _) in loaded.items():
            if set(stats) != set(reference):
                raise ValueError(f"{name} and {names[0]} evaluated different {split} queries")
        result[split] = {name: {"paper_protocol": paper_metrics(np.stack([stats[q] for q in sorted(stats)]))}
                         for name, (stats, _) in loaded.items()}
        for name in names[1:]:
            result[split][name]["paired_vs_" + names[0]] = paired(reference, loaded[name][0], image_of)
        lines += [f"## {split} ({len(reference):,} queries)", "",
                  "| Arm | mIoU | IoU_FG | AP | Δ mIoU (95% CI) | wins / losses |", "|---|---:|---:|---:|---:|---:|"]
        for name in names:
            m = result[split][name]["paper_protocol"]
            p = result[split][name].get("paired_vs_" + names[0])
            extra = (f"{100 * p['delta_miou_0.3']:+.2f} [{100 * p['ci95'][0]:+.2f}, {100 * p['ci95'][1]:+.2f}] | "
                     f"{p['wins']} / {p['losses']}") if p else "reference | —"
            lines.append(f"| {name} | {100 * m['miou_0.3']:.2f} | {100 * m['fgiou_0.3']:.2f} | {100 * m['ap']:.2f} | {extra} |")
        lines.append("")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "e4_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "e4_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
