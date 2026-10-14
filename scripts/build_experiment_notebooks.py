"""Build the E2 and E3 Kaggle notebooks.

Data loading, image download, model loading, GT rasterization and metrics are copied
from the E1 notebook so the three experiments share one protocol. Rerun after editing E1:

    python scripts/build_experiment_notebooks.py
"""
import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "source"
E1 = json.loads((SOURCE / "e1_train_eval_phrasecut.ipynb").read_text(encoding="utf-8"))
E1_CELLS = {cell["id"]: cell["source"] for cell in E1["cells"]}


def e1(cell_id, start=None, end=None):
    """Source of an E1 cell, optionally cut to the part from `start` up to `end`."""
    source = E1_CELLS[cell_id]
    for marker in [start, end]:
        if marker is not None and source.count(marker) != 1:
            raise ValueError(f"Marker not unique in {cell_id}: {marker!r}")
    if start is not None:
        source = source[source.index(start):]
    if end is not None:
        source = source[:source.index(end)]
    return source.strip() + "\n"


def notebook(cells):
    built = []
    for kind, source, *tag in cells:
        value = {"cell_type": kind, "id": f"cell-{len(built):02d}", "metadata": {}}
        if tag:
            value["metadata"]["tags"] = tag
        value["source"] = textwrap.dedent(source).strip() + "\n"
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        built.append(value)
    return {"cells": built, "metadata": E1["metadata"], "nbformat": 4, "nbformat_minor": 5}


COMMON_CONFIG = """
# None = decoder pretrained. Hoặc decoder đã train trong notebook E1 theo EPOCHS + BATCH_SIZE, ví dụ
# "/kaggle/input/<output-e1>/clipseg_e1/checkpoints/decoder_epoch_004.pt"
DECODER_CHECKPOINT = None
DATA_ROOT = None                 # ví dụ: "/kaggle/input/phrasecut-miniv/VGPhraseCut_v0"
EVAL_SPLIT = "miniv"             # "miniv" | "val" | "test"; support của E2 lấy trong cùng split
MODEL_PATH = None                # thư mục checkpoint offline, hoặc None
IMAGE_ROOT = None                # thư mục chứa ảnh <image_id>.jpg; None = tìm trong mọi Kaggle Input
WORK_ROOT = Path("/kaggle/working")
REQUIRE_GPU = True               # False chỉ khi kiểm tra trên máy CPU
USE_AMP = False                  # mặc định FP32 như E1
SEED = 42
THRESHOLD = 0.3
CALIBRATION_FRACTION = 0.2
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
N_BOOTSTRAP = 2000               # bootstrap theo ảnh cho CI 95% của ΔIoU
TIE_TOLERANCE = 0.01             # |ΔIoU| ≤ 0,01 tính là hòa
SAVE_MASKS = True
N_VISUALIZATIONS = 6
DOWNLOAD_WORKERS = 4

MODEL_ID = "CIDAS/clipseg-rd64-refined"
MODEL_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
PHRASECUT_API_COMMIT = "276470b7773d8a230aee5b5536876fbc485fb04b"
"""

SETUP = """
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import gdown
import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm
from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
from IPython.display import display

assert 0 < CALIBRATION_FRACTION < 1
assert 0 <= THRESHOLD <= 1 and all(0 <= t <= 1 for t in THRESHOLD_GRID)
assert N_BOOTSTRAP >= 100 and TIE_TOLERANCE >= 0
random.seed(SEED)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("Chưa có CUDA. Bật GPU trong Kaggle Settings rồi restart session.")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    try:
        torch.ones(1, device=device).add_(1).item()
    except RuntimeError as exc:
        raise RuntimeError("PyTorch của Kaggle không chạy được trên GPU này (thường gặp với P100). "
                           "Chọn Accelerator GPU T4 x2 rồi restart session.") from exc
assert not USE_AMP or device.type == "cuda", "FP16 AMP yêu cầu CUDA."
WORK_ROOT.mkdir(parents=True, exist_ok=True)
OUT = WORK_ROOT / OUTPUT_NAME
OUT.mkdir(parents=True, exist_ok=True)
# Xóa kết quả lượt cũ để không trộn artifact vào ZIP.
for folder in ["masks", "visualizations"]:
    shutil.rmtree(OUT / folder, ignore_errors=True)
for filename in ["summary.json", "run.json"]:
    (OUT / filename).unlink(missing_ok=True)
(WORK_ROOT / f"{OUTPUT_NAME}.zip").unlink(missing_ok=True)

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

versions = {name: importlib.metadata.version(name) for name in
            ["torch", "transformers", "numpy", "Pillow", "scikit-learn", "matplotlib", "gdown"]}
print("Device:", device, "| GPU:", torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")
print(json.dumps(versions, indent=2))
"""

MODEL_MD = """
## 5. Nạp model, inference và metric

Nếu `DECODER_CHECKPOINT` được đặt, decoder được thay bằng decoder đã train trong notebook E1
(CLIP vẫn là backbone đóng băng của checkpoint gốc). Epoch, batch size và lịch sử loss đọc từ checkpoint
được ghi vào `summary.json` và `run.json`. E1 chỉ train trên split `train`, nên miniv vẫn là dữ liệu chưa thấy.

Model, rasterize GT và metric giống hệt E1. `predict` nhận **hoặc** danh sách text prompt
**hoặc** danh sách ảnh support (truyền vào `conditional_pixel_values` của CLIPSeg),
luôn dự đoán trên cùng một ảnh query. Logits nội suy bilinear về kích thước gốc rồi sigmoid.

`paired_comparison` so từng query với biến thể gốc: ΔIoU = biến thể − gốc, CI 95% bằng
bootstrap theo **ảnh** (các query cùng ảnh không độc lập), kèm số query thắng/hòa/thua.
"""

LOAD_DECODER = """
decoder_info = {"source": "pretrained", "checkpoint": None}
if DECODER_CHECKPOINT:
    decoder_path = Path(DECODER_CHECKPOINT)
    if not decoder_path.is_file():
        raise FileNotFoundError(f"DECODER_CHECKPOINT không tồn tại: {decoder_path}")
    checkpoint = torch.load(decoder_path, map_location="cpu", weights_only=True)
    if checkpoint.get("base_model_id") != MODEL_ID or checkpoint.get("base_model_revision") != MODEL_REVISION:
        raise ValueError("Checkpoint được train trên base model/revision khác MODEL_ID/MODEL_REVISION.")
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()
    training = checkpoint.get("training") or {}
    decoder_info = {"source": "e1_trained_decoder", "checkpoint": str(decoder_path),
        "checkpoint_sha256": sha256(decoder_path), "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"], "epochs_planned": training.get("epochs"),
        "batch_size": training.get("batch_size"), "initialization": training.get("initialization"),
        "selected_train_queries": training.get("selected_train_queries"),
        # Checkpoint của bản E1 cũ không ghi train_mode: bản đó chỉ train bằng text.
        "train_mode": training.get("train_mode", "text_only" if training else None),
        "history": checkpoint.get("history")}
    if training.get("epochs") is not None and checkpoint["epoch"] != training["epochs"]:
        print(f"Cảnh báo: checkpoint epoch {checkpoint['epoch']}/{training['epochs']}, không phải epoch cuối.")
    del checkpoint
print("Decoder:", json.dumps({k: v for k, v in decoder_info.items() if k != "history"}, ensure_ascii=False))
"""

PREDICT = """
@torch.inference_mode()
def predict(image, prompts=None, supports=None):
    \"\"\"Return [N,H,W] probabilities for N text prompts or N support images on one query image.\"\"\"
    if (prompts is None) == (supports is None):
        raise ValueError("Truyền text prompts hoặc support images, không cả hai.")
    count = len(prompts) if prompts is not None else len(supports)
    inputs = dict(processor(images=[image] * count, return_tensors="pt"))
    if prompts is not None:
        inputs.update(processor.tokenizer(text=list(prompts), padding=True, truncation=True, return_tensors="pt"))
    else:
        inputs["conditional_pixel_values"] = processor(images=list(supports), return_tensors="pt")["pixel_values"]
    inputs = {k: v.to(device) for k, v in inputs.items()}
    synchronize()
    started = time.perf_counter()
    context = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
    with context:
        logits = model(**inputs).logits
    if logits.ndim == 2:
        logits = logits.unsqueeze(0)
    probabilities = F.interpolate(logits.float().unsqueeze(1), size=(image.height, image.width),
        mode="bilinear", align_corners=False).squeeze(1).sigmoid()
    synchronize()
    seconds = time.perf_counter() - started
    values = probabilities.cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("Prediction chứa NaN/Inf; thử USE_AMP=False.")
    return values, seconds

def evaluate(probability, target, base):
    \"\"\"Metric tại THRESHOLD (kèm pixel AP) và các dòng quét ngưỡng cho calibration.\"\"\"
    row = {**base, "threshold": THRESHOLD, **binary_metrics(probability, target, THRESHOLD),
           "pixel_ap": float(average_precision_score(target.ravel(), probability.ravel()))}
    sweep = [{**base, "threshold": t, **binary_metrics(probability, target, t)} for t in THRESHOLD_GRID]
    return row, sweep

def variant_report(rows, sweep_rows):
    \"\"\"Ngưỡng cố định trên mọi query; ngưỡng bổ sung chọn chỉ trên calibration.\"\"\"
    table = []
    for threshold in THRESHOLD_GRID:
        subset = [r for r in sweep_rows if r["group"] == "calibration" and r["threshold"] == threshold]
        table.append({"threshold": threshold, **summarize(subset)})
    best = max(table, key=lambda r: (r["mean_sample_iou"], -r["threshold"]))["threshold"]
    return {
        "all_fixed_threshold": summarize(rows),
        "calibration_fixed_threshold": summarize([r for r in rows if r["group"] == "calibration"]),
        "evaluation_fixed_threshold": summarize([r for r in rows if r["group"] == "evaluation"]),
        "calibration_selected_threshold": best,
        "evaluation_calibrated_threshold": summarize([r for r in sweep_rows
            if r["group"] == "evaluation" and r["threshold"] == best]),
    }, table

def paired_comparison(base_iou, other_iou, image_of, n_bootstrap, tie_tolerance, seed):
    \"\"\"ΔIoU theo query (other − base), CI 95% bootstrap theo ảnh, số query thắng/hòa/thua.\"\"\"
    query_ids = sorted(base_iou)
    if set(query_ids) != set(other_iou):
        raise ValueError("Hai biến thể phải có cùng tập query.")
    deltas = np.array([other_iou[q] - base_iou[q] for q in query_ids])
    groups = defaultdict(list)
    for index, query_id in enumerate(query_ids):
        groups[image_of[query_id]].append(index)
    clusters = [np.array(indices) for _, indices in sorted(groups.items())]
    rng = np.random.default_rng(seed)
    boot = [deltas[np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), len(clusters))])].mean()
            for _ in range(n_bootstrap)]
    return {"n_queries": len(deltas), "delta_mean_iou": float(deltas.mean()),
            "ci95_low": float(np.percentile(boot, 2.5)), "ci95_high": float(np.percentile(boot, 97.5)),
            "wins": int((deltas > tie_tolerance).sum()), "ties": int((np.abs(deltas) <= tie_tolerance).sum()),
            "losses": int((deltas < -tie_tolerance).sum())}

def overlay(image, probability):
    rgb = np.asarray(image).astype(np.float32).copy()
    mask = probability >= THRESHOLD
    rgb[mask] = 0.55 * rgb[mask] + 0.45 * np.array([255, 60, 60])
    return rgb.astype(np.uint8)

def load_rgb(image_id):
    with Image.open(image_paths[image_id]) as raw:
        return raw.convert("RGB")

def save_mask(probability, variant, stem):
    if SAVE_MASKS:
        folder = OUT / "masks" / variant
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray((probability >= THRESHOLD).astype(np.uint8) * 255).save(folder / f"{stem}.png")

def table_row(name, report, comparison):
    fixed = report["all_fixed_threshold"]
    return {"variant": name, "rows": fixed["n_queries"], "mean_iou": fixed["mean_sample_iou"],
        "cumulative_iou": fixed["cumulative_iou"], "mean_dice": fixed["mean_dice"],
        "mean_pixel_ap": fixed["mean_sample_pixel_ap"], "precision_at_0.5": fixed["precision_at_0.5"],
        "calibrated_threshold": report["calibration_selected_threshold"],
        "evaluation_iou_fixed": report["evaluation_fixed_threshold"]["mean_sample_iou"],
        "evaluation_iou_calibrated": report["evaluation_calibrated_threshold"]["mean_sample_iou"],
        **{key: (comparison or {}).get(key) for key in
           ["delta_mean_iou", "ci95_low", "ci95_high", "wins", "ties", "losses"]}}

def print_table(rows):
    columns = ["variant", "rows", "mean_iou", "cumulative_iou", "mean_pixel_ap", "delta_mean_iou", "ci95_low", "ci95_high", "wins", "losses"]
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in columns))

def run_metadata(extra):
    inference_seconds = sum(b["forward_resize_seconds"] for b in batch_times)
    prompts = sum(b["prompts"] for b in batch_times)
    return {**extra, "decoder": decoder_info,
        "model_id": MODEL_ID, "requested_revision": MODEL_REVISION if not MODEL_PATH else None,
        "model_source": model_source, "resolved_commit": getattr(model.config, "_commit_hash", None),
        "phrasecut_api_commit": PHRASECUT_API_COMMIT,
        "annotation_sha256": sha256(annotation_path), "image_metadata_sha256": sha256(info_path),
        "seed": SEED, "threshold": THRESHOLD, "threshold_grid": THRESHOLD_GRID,
        "calibration_fraction": CALIBRATION_FRACTION, "n_bootstrap": N_BOOTSTRAP, "tie_tolerance": TIE_TOLERANCE,
        "precision": "fp16_autocast" if USE_AMP else "fp32", "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_cuda_version": torch.version.cuda, "python": platform.python_version(), "versions": versions,
        "image_processor": processor.image_processor.to_dict(),
        "forward_resize_seconds_total": inference_seconds,
        "forward_resize_ms_per_prompt": 1000 * inference_seconds / prompts,
        "wall_seconds_inference_metrics_saving": elapsed_all,
        "peak_allocated_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "timing_note": "Two warmup batches; forward+resize+sigmoid timed with CUDA sync; excludes preprocessing and metric/save.",
    }
"""

MODEL_CELL = (e1("cell-11", end="@torch.inference_mode()") + LOAD_DECODER + "\n"
              + e1("cell-11", start="def binary_metrics") + PREDICT)

PACKAGE = """
for path in sorted((OUT / "visualizations").glob("*.png"))[:N_VISUALIZATIONS]:
    with Image.open(path) as image:
        display(image.copy())
archive = WORK_ROOT / f"{OUTPUT_NAME}.zip"
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
    for path in sorted(OUT.rglob("*")):
        if path.is_file():
            package.write(path, arcname=path.relative_to(OUT).as_posix())
print("Result folder:", OUT)
print("Download:", archive)
"""

SOURCES_MD = """
## Nguồn và giới hạn

- [CLIPSeg / CVPR 2022](https://github.com/timojl/clipseg), [checkpoint HF](https://huggingface.co/CIDAS/clipseg-rd64-refined).
- [PhraseCutDataset](https://github.com/ChenyunWu/PhraseCutDataset): annotation, URL ảnh, split và quy ước polygon/mask.
- [CLIPSeg trong Transformers 4.55.4](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/clipseg).

Checkpoint pretrained, không train. PhraseCut liên quan tới dữ liệu huấn luyện CLIPSeg nên không gọi là zero-shot.
Google Drive/Visual Genome có thể gián đoạn; Add Input dataset chứa hai file JSON (và ảnh) là đường dự phòng.
"""

# --------------------------------------------------------------------------- E3

E3_INTRO = """
# XLAV — E3: prompt engineering cho text prompt (CLIPSeg, PhraseCut miniv)

Câu hỏi: **cách viết câu lệnh** có làm thay đổi chất lượng mask không? Mỗi query được chạy với
4 template cố định từ trước: `{phrase}`, `a photo of {phrase}`, `the {phrase}`, `{phrase} in the image`.
Ảnh, GT, ngưỡng 0,3 và metric giữ nguyên như E1; chỉ chuỗi text thay đổi.

1. Bật **GPU T4 x2** và **Internet** trong Settings của notebook Kaggle.
2. **Run All**. Kết quả ở `/kaggle/working/clipseg_e3_prompt_templates`, kèm file ZIP.

Báo cáo **mọi template**, cả tốt lẫn kém. `{phrase}` là gốc để so sánh từng cặp (ΔIoU, CI 95% bootstrap theo ảnh).
Không chọn template "tốt nhất" trên evaluation rồi coi là kết quả độc lập; ngưỡng bổ sung chỉ chọn trên calibration.
Chọn 100 query bằng seed 42 và chia calibration/evaluation theo image_id giống hệt E1.
"""

E3_CONFIG_MD = """
## 2. Cấu hình

`MAX_QUERIES=100` theo kế hoạch; đặt `8` để kiểm tra nhanh. `BATCH_SIZE` là số prompt mỗi lần forward
(query × template trên cùng ảnh); giảm nếu thiếu VRAM. `DATA_ROOT=None` tự tìm trong `/kaggle/input`
rồi tải từ nguồn chính thức nếu chưa có.

`DECODER_CHECKPOINT=None` dùng decoder pretrained. Để đánh giá decoder đã train theo `EPOCHS` + `BATCH_SIZE`:
chạy notebook E1 với `TRAIN_DECODER=True`, lưu version để Output thành dataset, Add Input dataset đó vào notebook này,
rồi trỏ `DECODER_CHECKPOINT` tới `checkpoints/decoder_epoch_XXX.pt` của **epoch cuối**.
Kết quả ghi đè thư mục output; tải ZIP của lượt pretrained trước khi chạy lượt có checkpoint.
"""

E3_CONFIG = """
from pathlib import Path

MAX_QUERIES = 100                # kế hoạch E3: 100 query; 8 = smoke
TEMPLATES = ["{phrase}", "a photo of {phrase}", "the {phrase}", "{phrase} in the image"]
BASELINE_TEMPLATE = "{phrase}"
BATCH_SIZE = 16                  # số prompt mỗi forward
OUTPUT_NAME = "clipseg_e3_prompt_templates"
""" + COMMON_CONFIG

E3_RUN_MD = """
## 6. Chạy 4 template trên mọi query

Mỗi ảnh gom (query × template) thành batch. Mask lưu tại `masks/<t0..t3>/<task_id>.png`
(bảng ánh xạ key → template ở `run.json`). Hình minh họa đặt 4 template cạnh nhau cho cùng query.
"""

E3_RUN = """
assert BATCH_SIZE >= 1
assert all("{phrase}" in t for t in TEMPLATES) and len(set(TEMPLATES)) == len(TEMPLATES)
assert BASELINE_TEMPLATE in TEMPLATES
template_keys = {template: f"t{i}" for i, template in enumerate(TEMPLATES)}
(OUT / "visualizations").mkdir(exist_ok=True)

tasks_by_image = defaultdict(list)
for task in selected:
    tasks_by_image[int(task["image_id"])].append(task)
warmup_image = load_rgb(image_ids[0])
warmup_prompts = [TEMPLATES[0].format(phrase=t["phrase"]) for t in tasks_by_image[image_ids[0]]][:BATCH_SIZE]
for _ in range(2):
    predict(warmup_image, prompts=warmup_prompts)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

def save_template_figure(image, target, task, probabilities, ious, index):
    fig, axes = plt.subplots(1, 2 + len(TEMPLATES), figsize=(4 * (2 + len(TEMPLATES)), 4))
    axes[0].imshow(image); axes[0].set_title("Ảnh")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth")
    for ax, template in zip(axes[2:], TEMPLATES):
        ax.imshow(overlay(image, probabilities[template]))
        ax.set_title(f"{template_keys[template]}: IoU={ious[template]:.3f}\\n{template.format(phrase='…')}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{task['task_id']} — {task['phrase']}")
    fig.tight_layout()
    fig.savefig(OUT / "visualizations" / f"{index:02d}_{task['task_id']}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

rows, sweep_rows, batch_times = [], [], []
visual_count = 0
started_all = time.perf_counter()
for image_id in tqdm(image_ids, desc="E3 images"):
    image = load_rgb(image_id)
    current = tasks_by_image[image_id]
    targets = {str(t["task_id"]): rasterize_gt(t, image.width, image.height) for t in current}
    jobs = [(task, template) for task in current for template in TEMPLATES]
    kept = defaultdict(dict)      # xác suất của các query được vẽ hình
    ious = defaultdict(dict)
    for start in range(0, len(jobs), BATCH_SIZE):
        chunk = jobs[start:start + BATCH_SIZE]
        probabilities, seconds = predict(image, prompts=[tpl.format(phrase=t["phrase"]) for t, tpl in chunk])
        batch_times.append({"prompts": len(chunk), "forward_resize_seconds": seconds})
        for (task, template), probability in zip(chunk, probabilities):
            task_id = str(task["task_id"])
            base = {"task_id": task_id, "image_id": image_id, "phrase": task["phrase"],
                "variant": template, "variant_key": template_keys[template],
                "prompt": template.format(phrase=task["phrase"]), "group": group_for_image[image_id],
                "width": image.width, "height": image.height}
            row, sweep = evaluate(probability, targets[task_id], base)
            rows.append(row)
            sweep_rows.extend(sweep)
            save_mask(probability, template_keys[template], task_id)
            ious[task_id][template] = row["iou"]
            kept[task_id][template] = probability
    for task in current:
        task_id = str(task["task_id"])
        if visual_count < N_VISUALIZATIONS:
            save_template_figure(image, targets[task_id], task, kept[task_id], ious[task_id], visual_count)
            visual_count += 1
    write_csv(OUT / "sample_metrics.csv", rows)
elapsed_all = time.perf_counter() - started_all
assert len(rows) == len(selected) * len(TEMPLATES), "Chưa đánh giá đủ query × template."
print(f"Completed {len(selected)} queries × {len(TEMPLATES)} templates; wall time {elapsed_all:.1f}s")
"""

E3_SUMMARY_MD = """
## 7. Bảng E3

`e3_template_table.csv` là bảng chính: mỗi template một dòng, metric tại t=0,3 trên toàn bộ query đã chọn,
ΔIoU so với `{phrase}` kèm CI 95% và số query thắng/hòa/thua. CI chứa 0 nghĩa là chưa đủ bằng chứng
template đó khác `{phrase}`. Cột calibrated chỉ để tham khảo, ngưỡng được chọn trên calibration.
"""

E3_SUMMARY = """
image_of = {str(t["task_id"]): int(t["image_id"]) for t in selected}
reports, calibration_rows, table = {}, [], []
iou_by_template = {}
for template in TEMPLATES:
    report, calibration_table = variant_report([r for r in rows if r["variant"] == template],
                                               [r for r in sweep_rows if r["variant"] == template])
    reports[template] = report
    calibration_rows += [{"variant": template, **r} for r in calibration_table]
    iou_by_template[template] = {r["task_id"]: r["iou"] for r in rows if r["variant"] == template}
for template in TEMPLATES:
    comparison = None if template == BASELINE_TEMPLATE else paired_comparison(
        iou_by_template[BASELINE_TEMPLATE], iou_by_template[template], image_of, N_BOOTSTRAP, TIE_TOLERANCE, SEED)
    reports[template]["paired_vs_baseline"] = comparison
    table.append({"variant_key": template_keys[template], **table_row(template, reports[template], comparison)})
write_csv(OUT / "calibration_thresholds.csv", calibration_rows)
write_csv(OUT / "e3_template_table.csv", table)
summary = {"experiment": "E3_prompt_templates", "decoder": decoder_info, "dataset": "VGPhraseCut", "split": EVAL_SPLIT,
    "n_queries": len(selected), "complete_official_split": len(selected) == len(tasks),
    "baseline_template": BASELINE_TEMPLATE, "templates": reports,
    "metric_note": "IoU averaged per query at t=0.3; ΔIoU paired per query; CI is image-level bootstrap."}
write_json(OUT / "summary.json", summary)
write_json(OUT / "run.json", run_metadata({"experiment": "E3_prompt_templates", "max_queries": MAX_QUERIES,
    "batch_size": BATCH_SIZE, "templates": template_keys, "baseline_template": BASELINE_TEMPLATE,
    "selected_tasks_sha256": sha256(OUT / "selected_tasks.json"),
    "calibration_image_ids": sorted(calibration_ids),
    "evaluation_image_ids": sorted(set(image_ids) - calibration_ids)}))
print_table(table)
"""

e3 = notebook([
    ("markdown", E3_INTRO),
    ("markdown", e1("cell-01")),
    ("code", e1("cell-02"), "install"),
    ("markdown", E3_CONFIG_MD),
    ("code", E3_CONFIG, "config"),
    ("code", SETUP),
    ("markdown", E1_CELLS["cell-06"]),
    ("code", e1("cell-07")),
    ("markdown", E1_CELLS["cell-08"]),
    ("code", e1("cell-09")),
    ("markdown", MODEL_MD),
    ("code", MODEL_CELL),
    ("markdown", E3_RUN_MD),
    ("code", E3_RUN),
    ("markdown", E3_SUMMARY_MD),
    ("code", E3_SUMMARY),
    ("markdown", "## 8. Xem hình và đóng gói\n\nTải `clipseg_e3_prompt_templates.zip` trong Kaggle Output."),
    ("code", PACKAGE),
    ("markdown", SOURCES_MD),
])

# --------------------------------------------------------------------------- E2

E2_INTRO = """
# XLAV — E2: text prompt và visual prompt (CLIPSeg, PhraseCut miniv)

Câu hỏi: chỉ cho CLIPSeg biết vật cần tìm **bằng chữ** hay **bằng một ảnh mẫu (support)** thì tốt hơn,
và ảnh mẫu nên được xử lý thế nào? Mỗi query miniv được dự đoán với 5 biến thể prompt:

| Biến thể | Prompt |
|---|---|
| `text_phrase` | Cụm từ gốc, ví dụ `red truck` (giống E1, là gốc để so sánh) |
| `text_name` | Chỉ danh từ chính `truck`, cùng mức thông tin với ảnh support |
| `visual_crop` | Ảnh support crop theo bbox vật (nới 10%) |
| `visual_dark` | Cả ảnh support, nền ngoài vật giữ 10% độ sáng |
| `visual_crop_dark` | Kết hợp crop và tối nền |

**Support** lấy từ **ảnh miniv khác** có cùng danh từ chính (`phrase_structure.name`). Mask dùng để crop/tối nền
là GT **của ảnh support**; mask GT của ảnh query không bao giờ được dùng để tạo prompt. Mỗi query dùng đúng
`SUPPORTS_PER_QUERY` support từ các ảnh khác nhau; metric của biến thể visual là trung bình trên các support.
Danh sách cặp lưu ở `e2_pairs.json`.

1. Bật **GPU T4 x2** và **Internet** trong Settings của notebook Kaggle.
2. **Run All**. Kết quả ở `/kaggle/working/clipseg_e2_text_vs_visual`, kèm file ZIP.

Đây là phiên bản rút gọn ý tưởng Bảng 2 của paper (paper đo ΔP trên LVIS, có thêm blur); số liệu không so trực tiếp.
"""

E2_CONFIG_MD = """
## 2. Cấu hình

`MAX_QUERIES=30` theo kế hoạch; nếu không đủ query có support hợp lệ, notebook chạy với số có được và ghi rõ N.
`MIN_SUPPORT_BOX_FRACTION` loại support quá nhỏ (tổng diện tích box của vật / diện tích ảnh support),
vì crop từ vật vài pixel không còn là ảnh mẫu có nghĩa.

`DECODER_CHECKPOINT=None` dùng decoder pretrained. Để đánh giá decoder đã train theo `EPOCHS` + `BATCH_SIZE`:
chạy notebook E1 với `TRAIN_DECODER=True`, lưu version để Output thành dataset, Add Input dataset đó vào notebook này,
rồi trỏ `DECODER_CHECKPOINT` tới `checkpoints/decoder_epoch_XXX.pt` của **epoch cuối**.
Kết quả ghi đè thư mục output; tải ZIP của lượt pretrained trước khi chạy lượt có checkpoint.
"""

E2_CONFIG = """
from pathlib import Path

MAX_QUERIES = 30                 # kế hoạch E2: 30 query; 8 = smoke
SUPPORTS_PER_QUERY = 3           # mỗi query dùng đúng K support từ K ảnh khác nhau
MIN_SUPPORT_BOX_FRACTION = 0.02  # bỏ support có vật quá nhỏ
CROP_PADDING = 0.1               # nới bbox support 10% mỗi phía trước khi crop
DARK_FACTOR = 0.1                # nền giữ 10% độ sáng, như Bảng 2 của paper
BATCH_SIZE = 9                   # số visual prompt mỗi forward (support × biến thể); giảm nếu thiếu VRAM
OUTPUT_NAME = "clipseg_e2_text_vs_visual"
""" + COMMON_CONFIG

E2_SELECT_MD = """
## 3. Lấy annotation chính thức và ghép query với support

Annotation tải như E1. Query hợp lệ khi có ít nhất `SUPPORTS_PER_QUERY` **ảnh khác** chứa task cùng danh từ chính
và đủ lớn. Chọn query và support bằng seed 42, rồi chia calibration/evaluation theo image_id của query.
Ảnh cần tải gồm ảnh query và ảnh support.
"""

E2_SELECT = e1("cell-07", end="selected = sorted(tasks") + """
assert all(str(t.get("phrase_structure", {}).get("name", "")).strip() for t in tasks), "Thiếu phrase_structure.name."
assert SUPPORTS_PER_QUERY >= 1 and MAX_QUERIES >= 1 and 0 <= MIN_SUPPORT_BOX_FRACTION < 1

def target_key(task):
    return task["phrase_structure"]["name"].strip().lower()

def box_fraction(task):
    info = info_by_id[int(task["image_id"])]
    area = sum(max(w, 0) * max(h, 0) for _, _, w, h in task["instance_boxes"])
    return area / (int(info["width"]) * int(info["height"]))

ordered = sorted(tasks, key=lambda t: str(t["task_id"]))
supports_by_key = defaultdict(dict)   # key -> image_id -> một task support (task_id nhỏ nhất)
for task in ordered:
    if box_fraction(task) >= MIN_SUPPORT_BOX_FRACTION:
        supports_by_key[target_key(task)].setdefault(int(task["image_id"]), task)
eligible = []
for task in ordered:
    candidates = [s for image_id, s in sorted(supports_by_key[target_key(task)].items())
                  if image_id != int(task["image_id"])]
    if len(candidates) >= SUPPORTS_PER_QUERY:
        eligible.append((task, candidates))
rng = random.Random(SEED)
chosen = rng.sample(eligible, min(MAX_QUERIES, len(eligible)))
chosen.sort(key=lambda pair: (int(pair[0]["image_id"]), str(pair[0]["task_id"])))
pairs = [{"query": task, "supports": rng.sample(candidates, SUPPORTS_PER_QUERY)} for task, candidates in chosen]
if len(pairs) < MAX_QUERIES:
    print(f"Chỉ có {len(pairs)} query có đủ support hợp lệ (yêu cầu {MAX_QUERIES}); báo cáo đúng N này.")
if len(pairs) < 2:
    raise ValueError("Không đủ query có support hợp lệ; giảm SUPPORTS_PER_QUERY hoặc MIN_SUPPORT_BOX_FRACTION.")
for pair in pairs:
    assert all(int(s["image_id"]) != int(pair["query"]["image_id"]) for s in pair["supports"]), "Support trùng ảnh query."
    assert len({int(s["image_id"]) for s in pair["supports"]}) == SUPPORTS_PER_QUERY

selected = [pair["query"] for pair in pairs]
query_image_ids = sorted({int(t["image_id"]) for t in selected})
if len(query_image_ids) < 2:
    raise ValueError("Cần ít nhất 2 ảnh query để chia calibration/evaluation; tăng MAX_QUERIES.")
shuffled_ids = query_image_ids.copy()
random.Random(SEED).shuffle(shuffled_ids)
n_cal = min(len(query_image_ids) - 1, max(1, round(len(query_image_ids) * CALIBRATION_FRACTION)))
calibration_ids = set(shuffled_ids[:n_cal])
group_for_image = {i: "calibration" if i in calibration_ids else "evaluation" for i in query_image_ids}
image_ids = sorted(set(query_image_ids) | {int(s["image_id"]) for p in pairs for s in p["supports"]})
write_json(OUT / "e2_pairs.json", [{"task_id": str(p["query"]["task_id"]), "image_id": int(p["query"]["image_id"]),
    "phrase": p["query"]["phrase"], "target_key": target_key(p["query"]),
    "group": group_for_image[int(p["query"]["image_id"])],
    "supports": [{"task_id": str(s["task_id"]), "image_id": int(s["image_id"]), "phrase": s["phrase"],
                  "box_fraction": round(box_fraction(s), 4)} for s in p["supports"]]} for p in pairs])
print(f"Annotation: {len(tasks)} queries / {len(info_by_id)} miniv images; eligible queries: {len(eligible)}")
print(f"Selected: {len(pairs)} queries / {len(query_image_ids)} query images; calibration {n_cal} images")
print(f"Images to load (query + support): {len(image_ids)}")
"""

E2_RUN_MD = """
## 6. Tạo visual prompt và chạy 5 biến thể

`support_views` tạo 3 visual prompt từ ảnh support và mask GT **của ảnh support**.
Text: một forward cho `text_phrase` và `text_name`. Visual: gom (support × 3 biến thể) thành batch `BATCH_SIZE`.
Mask lưu tại `masks/<variant>/<task_id>.png` (text) và `masks/<variant>/<task_id>__<support_task_id>.png` (visual).
"""

E2_RUN = """
TEXT_VARIANTS = ["text_phrase", "text_name"]
VISUAL_VARIANTS = ["visual_crop", "visual_dark", "visual_crop_dark"]
VARIANTS = TEXT_VARIANTS + VISUAL_VARIANTS
BASELINE_VARIANT = "text_phrase"
assert BATCH_SIZE >= 1
if decoder_info.get("train_mode") == "text_only":
    print("CẢNH BÁO: decoder chỉ được train bằng text, chưa từng thấy embedding ảnh support. "
          "Kết quả visual chỉ dùng làm ablation, không so ngang với decoder train PhraseCut+.")
(OUT / "visualizations").mkdir(exist_ok=True)

def support_views(image, mask, crop_padding, dark_factor):
    \"\"\"Crop / tối nền / kết hợp, chỉ dùng mask của chính ảnh support.\"\"\"
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    pad_x, pad_y = round((x1 - x0) * crop_padding), round((y1 - y0) * crop_padding)
    box = (max(0, x0 - pad_x), max(0, y0 - pad_y), min(image.width, x1 + pad_x), min(image.height, y1 + pad_y))
    rgb = np.asarray(image, dtype=np.float32)
    dark = Image.fromarray(np.where(mask[..., None], rgb, rgb * dark_factor).round().astype(np.uint8))
    return {"visual_crop": image.crop(box), "visual_dark": dark, "visual_crop_dark": dark.crop(box)}

views_cache = {}
def views_for(support):
    key = str(support["task_id"])
    if key not in views_cache:
        image = load_rgb(int(support["image_id"]))
        mask = rasterize_gt(support, image.width, image.height)
        views_cache[key] = support_views(image, mask, CROP_PADDING, DARK_FACTOR)
    return views_cache[key]

def save_pair_figure(image, target, pair, probabilities, ious, index):
    task, support = pair["query"], pair["supports"][0]
    panels = [("Ảnh query", image), ("Ground truth", target),
              (f"text_phrase IoU={ious['text_phrase']:.3f}", overlay(image, probabilities["text_phrase"])),
              (f"Support: {support['phrase']}", views_for(support)["visual_crop_dark"])]
    panels += [(f"{v} IoU={ious[v]:.3f}", overlay(image, probabilities[v])) for v in VISUAL_VARIANTS]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, value) in zip(axes, panels):
        ax.imshow(value, cmap="gray" if title == "Ground truth" else None)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{task['task_id']} — {task['phrase']} (visual dùng support đầu tiên)")
    fig.tight_layout()
    fig.savefig(OUT / "visualizations" / f"{index:02d}_{task['task_id']}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)

warmup_image = load_rgb(int(pairs[0]["query"]["image_id"]))
for _ in range(2):
    predict(warmup_image, prompts=[pairs[0]["query"]["phrase"], target_key(pairs[0]["query"])])
    predict(warmup_image, supports=[views_for(pairs[0]["supports"][0])[v] for v in VISUAL_VARIANTS])
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

rows, sweep_rows, batch_times = [], [], []
started_all = time.perf_counter()
for index, pair in enumerate(tqdm(pairs, desc="E2 queries")):
    task = pair["query"]
    task_id, image_id = str(task["task_id"]), int(task["image_id"])
    image = load_rgb(image_id)
    target = rasterize_gt(task, image.width, image.height)
    base = {"task_id": task_id, "image_id": image_id, "phrase": task["phrase"], "target_key": target_key(task),
            "group": group_for_image[image_id], "width": image.width, "height": image.height}
    first_probabilities, first_ious = {}, {}
    text_prompts = {"text_phrase": task["phrase"], "text_name": target_key(task)}
    probabilities, seconds = predict(image, prompts=list(text_prompts.values()))
    batch_times.append({"prompts": len(text_prompts), "forward_resize_seconds": seconds})
    for (variant, prompt), probability in zip(text_prompts.items(), probabilities):
        row, sweep = evaluate(probability, target, {**base, "variant": variant, "prompt": prompt,
                                                    "support_task_id": "", "support_image_id": ""})
        rows.append(row)
        sweep_rows.extend(sweep)
        save_mask(probability, variant, task_id)
        first_probabilities[variant], first_ious[variant] = probability, row["iou"]
    jobs = [(support, variant) for support in pair["supports"] for variant in VISUAL_VARIANTS]
    for start in range(0, len(jobs), BATCH_SIZE):
        chunk = jobs[start:start + BATCH_SIZE]
        probabilities, seconds = predict(image, supports=[views_for(s)[v] for s, v in chunk])
        batch_times.append({"prompts": len(chunk), "forward_resize_seconds": seconds})
        for (support, variant), probability in zip(chunk, probabilities):
            row, sweep = evaluate(probability, target, {**base, "variant": variant,
                "prompt": f"support:{support['task_id']}", "support_task_id": str(support["task_id"]),
                "support_image_id": int(support["image_id"])})
            rows.append(row)
            sweep_rows.extend(sweep)
            save_mask(probability, variant, f"{task_id}__{support['task_id']}")
            if support is pair["supports"][0]:
                first_probabilities[variant], first_ious[variant] = probability, row["iou"]
    if index < N_VISUALIZATIONS:
        save_pair_figure(image, target, pair, first_probabilities, first_ious, index)
    write_csv(OUT / "sample_metrics.csv", rows)
elapsed_all = time.perf_counter() - started_all
assert len(rows) == len(pairs) * (len(TEXT_VARIANTS) + len(VISUAL_VARIANTS) * SUPPORTS_PER_QUERY)
print(f"Completed {len(pairs)} queries × {len(VARIANTS)} variants; wall time {elapsed_all:.1f}s")
"""

E2_SUMMARY_MD = """
## 7. Bảng E2

`e2_variant_table.csv` là bảng chính. Cột `rows` là số dòng metric: N query với text,
N × K (query × support) với visual; mean IoU, cumulative IoU, AP và precision tính trên các dòng đó.
ΔIoU so với `text_phrase` tính **theo query**: IoU visual được trung bình trên K support trước khi trừ.
CI 95% bootstrap theo ảnh query. Visual prompt thường cần ngưỡng khác text, nên xem thêm cột
calibrated (ngưỡng chọn trên calibration) và AP (không phụ thuộc ngưỡng).
"""

E2_SUMMARY = """
image_of = {str(t["task_id"]): int(t["image_id"]) for t in selected}
reports, calibration_rows, table = {}, [], []
iou_by_variant = {}
for variant in VARIANTS:
    variant_rows = [r for r in rows if r["variant"] == variant]
    report, calibration_table = variant_report(variant_rows, [r for r in sweep_rows if r["variant"] == variant])
    reports[variant] = report
    calibration_rows += [{"variant": variant, **r} for r in calibration_table]
    per_query = defaultdict(list)
    for r in variant_rows:
        per_query[r["task_id"]].append(r["iou"])
    iou_by_variant[variant] = {task_id: float(np.mean(values)) for task_id, values in per_query.items()}
    reports[variant]["mean_iou_per_query"] = float(np.mean(list(iou_by_variant[variant].values())))
for variant in VARIANTS:
    comparison = None if variant == BASELINE_VARIANT else paired_comparison(
        iou_by_variant[BASELINE_VARIANT], iou_by_variant[variant], image_of, N_BOOTSTRAP, TIE_TOLERANCE, SEED)
    reports[variant]["paired_vs_baseline"] = comparison
    table.append(table_row(variant, reports[variant], comparison))
write_csv(OUT / "calibration_thresholds.csv", calibration_rows)
write_csv(OUT / "e2_variant_table.csv", table)
summary = {"experiment": "E2_text_vs_visual", "decoder": decoder_info, "dataset": "VGPhraseCut", "split": EVAL_SPLIT,
    "n_queries": len(pairs), "supports_per_query": SUPPORTS_PER_QUERY, "baseline_variant": BASELINE_VARIANT,
    "variants": reports,
    "metric_note": ("Text rows: one per query; visual rows: one per query×support. "
                    "ΔIoU is per query after averaging visual IoU over supports; CI is image-level bootstrap.")}
write_json(OUT / "summary.json", summary)
write_json(OUT / "run.json", run_metadata({"experiment": "E2_text_vs_visual", "max_queries": MAX_QUERIES,
    "supports_per_query": SUPPORTS_PER_QUERY, "min_support_box_fraction": MIN_SUPPORT_BOX_FRACTION,
    "crop_padding": CROP_PADDING, "dark_factor": DARK_FACTOR, "variants": VARIANTS, "batch_size": BATCH_SIZE,
    "pairs_sha256": sha256(OUT / "e2_pairs.json"),
    "calibration_image_ids": sorted(calibration_ids),
    "evaluation_image_ids": sorted(set(query_image_ids) - calibration_ids)}))
print_table(table)
"""

e2 = notebook([
    ("markdown", E2_INTRO),
    ("markdown", e1("cell-01")),
    ("code", e1("cell-02"), "install"),
    ("markdown", E2_CONFIG_MD),
    ("code", E2_CONFIG, "config"),
    ("code", SETUP),
    ("markdown", E2_SELECT_MD),
    ("code", E2_SELECT),
    ("markdown", E1_CELLS["cell-08"]),
    ("code", e1("cell-09")),
    ("markdown", MODEL_MD),
    ("code", MODEL_CELL),
    ("markdown", E2_RUN_MD),
    ("code", E2_RUN),
    ("markdown", E2_SUMMARY_MD),
    ("code", E2_SUMMARY),
    ("markdown", "## 8. Xem hình và đóng gói\n\nTải `clipseg_e2_text_vs_visual.zip` trong Kaggle Output."),
    ("code", PACKAGE),
    ("markdown", SOURCES_MD),
])

for name, value in [("e2_text_vs_visual.ipynb", e2), ("e3_prompt_templates.ipynb", e3)]:
    path = SOURCE / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(path)
