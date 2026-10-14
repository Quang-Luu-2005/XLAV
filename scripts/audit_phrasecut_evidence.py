"""Audit local Kaggle evidence and render a contact sheet for PhraseCut GT review."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
SMOKE = OUTPUTS / "kaggle-e1-20261013" / "job_e8a58e956a45472cb6e5b726944ff131" / "artifacts"
FULL = OUTPUTS / "kaggle-e1-20261013" / "job_e77c7852eec44814b8d068008f1ded81"
E2 = OUTPUTS / "kaggle-e2-20261014" / "job_e45526a2db6b42398e1d0e33434de55c" / "artifacts" / "e1_decoder"
EVIDENCE = ROOT / "docs" / "evidence"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def make_gt_preview(path: Path) -> Image.Image:
    """Overlay the saved polygon mask on its original cached query image."""
    image_id = path.stem.split("__", 1)[0]
    image_path = OUTPUTS / "baseline-notebook-check" / "phrasecut_image_cache" / f"{image_id}.jpg"
    with Image.open(image_path) as image, Image.open(path) as ground_truth:
        query = image.convert("RGB")
        if query.size != ground_truth.size:
            raise ValueError(f"Image/GT size mismatch: {path}")
        gt_array = np.asarray(ground_truth.convert("L"))
        mask = Image.fromarray(np.uint8(gt_array > 127) * 255)
        if not mask.getbbox():
            raise ValueError(f"Empty GT mask: {path}")
        tint = Image.new("RGB", query.size, (240, 45, 45))
        overlay = Image.composite(Image.blend(query, tint, 0.50), query, mask)
        pair = Image.new("RGB", (query.width * 2, query.height), "white")
        pair.paste(query, (0, 0))
        pair.paste(overlay, (query.width, 0))
        return pair


def main() -> None:
    train = read_json(SMOKE / "train_tasks.json")
    miniv = read_json(SMOKE / "selected_tasks.json")
    errors = read_json(SMOKE / "image_errors.json")
    pairs = read_json(E2 / "e2_pairs.json")
    run = read_json(FULL / "run.json")
    history = (FULL / "artifacts" / "training_history.csv").read_text(encoding="utf-8")

    train_ids = {row["image_id"] for row in train}
    miniv_ids = {row["image_id"] for row in miniv}
    assert len(train) == 512 and len(miniv) == 821
    assert len(miniv_ids) == 100 and not train_ids.intersection(miniv_ids)
    assert not errors
    assert all(row["image_id"] != support["image_id"] for row in pairs for support in row["supports"])
    assert len(pairs) == 30 and all(len(row["supports"]) == 3 for row in pairs)
    assert run["training"]["full_valid_train_queries"] == 310742
    assert run["training"]["excluded_train_queries"] == 74
    assert run["training"]["completed_epochs"] == 5
    assert len(history.strip().splitlines()) == 6

    samples = sorted((OUTPUTS / "baseline-notebook-check" / "clipseg_baseline_miniv" / "ground_truth").glob("*.png"))
    assert len(samples) >= 10
    task_ids = [path.stem for path in samples]
    assert len(set(task_ids)) == len(task_ids)

    tile_width, tile_height = 520, 235
    canvas = Image.new("RGB", (tile_width * 3, tile_height * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for index, path in enumerate(samples):
        preview = make_gt_preview(path)
        preview.thumbnail((tile_width - 16, tile_height - 34))
        x = (index % 3) * tile_width + 8
        y = (index // 3) * tile_height + 24
        canvas.paste(preview, (x, y))
        draw.text((x, y - 18), f"{index+1:02d}  {task_ids[index]}  original | GT overlay", fill="black")

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    canvas.save(EVIDENCE / "phrasecut_gt_overlays.jpg", quality=88, optimize=True)
    report = {
        "dataset": "poudelsandeep/phrasecut",
        "smoke_train_queries": len(train),
        "smoke_train_images": len(train_ids),
        "miniv_queries": len(miniv),
        "miniv_images": len(miniv_ids),
        "train_miniv_image_overlap": 0,
        "image_errors": len(errors),
        "full_valid_train_queries": run["training"]["full_valid_train_queries"],
        "full_excluded_train_queries": run["training"]["excluded_train_queries"],
        "full_train_annotation_sha256": run["training"]["train_annotation_sha256"],
        "full_train_manifest_sha256": run["training"]["train_manifest_sha256"],
        "full_test_annotation_sha256": run["annotation_sha256"],
        "full_test_image_metadata_sha256": run["image_metadata_sha256"],
        "gt_preview_task_ids": task_ids,
        "e2_query_support_pairs": len(pairs) * 3,
        "e2_support_same_image_violations": 0,
        "note": "GT contact sheet is for visual QA; miniv is not used for training or checkpoint selection.",
    }
    (EVIDENCE / "phrasecut_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
