import json
import textwrap
from pathlib import Path

cells = []


def cell(kind, source, tag=None):
    value = {"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {}}
    if tag:
        value["metadata"]["tags"] = [tag]
    value["source"] = textwrap.dedent(source).strip() + "\n"
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    cells.append(value)


cell("markdown", """
# XLAV — CLIPSeg baseline trên PhraseCut miniv

Tái tạo quá trình train CLIPSeg: CLIP đóng băng + decoder, train trên **PhraseCut+** (text + ảnh support,
mẫu âm, augmentation, tạo ngẫu nhiên từ PhraseCut) theo **epoch và batch size**, rồi đánh giá text prompt
trên PhraseCut miniv. `TRAIN_DECODER=False` chỉ đánh giá checkpoint pretrained của tác giả (mốc tham chiếu).
Notebook tự chứa toàn bộ code, chỉ cần upload file `.ipynb` lên Kaggle.

1. Bật **GPU** và **Internet** trong Settings của notebook Kaggle.
2. Chạy lần lượt các cell hoặc **Run All**. Lượt đầu tải checkpoint và dữ liệu.
3. Đặt `EPOCHS` và `BATCH_SIZE`, chọn dữ liệu train tại `TRAIN_ROOT` nếu đã Add Input.
4. Kết quả được ghi vào `/kaggle/working/clipseg_baseline_miniv`, kèm file ZIP để tải về.

Theo [PhraseCutDataset chính thức](https://github.com/ChenyunWu/PhraseCutDataset),
`miniv` có **100 ảnh / 821 query**. `MAX_QUERIES=None` chạy toàn bộ query hợp lệ;
đặt `200` để chạy tập con theo kế hoạch E1, hoặc `8` để kiểm tra nhanh.

Protocol: resize logits bilinear về kích thước ảnh gốc → sigmoid → threshold 0,3;
không min-max normalize. GT là hợp của polygon của mọi instance được phrase nhắc tới,
rasterize theo PIL như API PhraseCut. Mean IoU được tính **theo query**, cumulative IoU
là tổng intersection / tổng union. AP là AP pixel theo query rồi trung bình,
**chưa khẳng định tương đương AP trong paper CLIPSeg**.

Đánh giá trên miniv, không phải tập test của paper, nên chưa so trực tiếp với bảng paper.
Không gọi là zero-shot vì decoder được huấn luyện trên PhraseCut. Decoder train ở đây dùng tiếp cho E2/E3
qua `DECODER_CHECKPOINT`.
"""
)
cell("markdown", """
## 1. Dependency

Giữ PyTorch/CUDA, NumPy, Pillow, scikit-learn và matplotlib sẵn có của Kaggle.
Cell này chỉ pin Transformers và gdown; không cài wheel PyTorch CPU.
Nếu kernel đã import Transformers phiên bản khác trước đó, restart session rồi chạy lại từ đầu.
"""
)
cell("code", """
import subprocess
import sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--quiet",
    "transformers==4.55.4", "gdown==5.2.0",
])
""", "install")
cell("markdown", """
## 2. Cấu hình

Mặc định tải dữ liệu online. Nếu đã Add Input một Kaggle Dataset chứa dữ liệu,
đặt `DATA_ROOT` tới thư mục chứa `refer_miniv.json`, `image_data_split.json`
(hoặc `image_data_split3000.json`) và `images/`. Để `None` sẽ tự tìm trong `/kaggle/input`
rồi tải từ nguồn chính thức nếu chưa có. `MODEL_PATH=None` tải Hugging Face;
muốn offline thì trỏ tới thư mục snapshot có config, processor, tokenizer và weights.

Baseline mặc định FP32, dùng một GPU; T4 x2 vẫn chạy trên GPU đầu tiên.
`USE_AMP=True` là tùy chọn FP16, phải ghi riêng khi so sánh số liệu.
"""
)
cell("code", """
from pathlib import Path

DATA_ROOT = None                 # ví dụ: "/kaggle/input/phrasecut-miniv/VGPhraseCut_v0"
MODEL_PATH = None                # thư mục checkpoint offline, hoặc None
WORK_ROOT = Path("/kaggle/working")
REQUIRE_GPU = True               # False chỉ khi kiểm tra trên máy CPU
MAX_QUERIES = None               # None = toàn bộ miniv; 200 = E1 gọn; 8 = smoke
BATCH_SIZE = 8
EPOCHS = 1
TRAIN_DECODER = False            # E1 = False: chỉ đánh giá checkpoint pretrained
TRAIN_ROOT = None                # thư mục refer_train.json + images/ trong Kaggle Input
MAX_TRAIN_QUERIES = None         # None = toàn bộ train; 8 = smoke train
TRAIN_INIT = "random_decoder"   # "pretrained_decoder" = fine-tune decoder đã học
TRAIN_MODE = "phrasecut_plus"    # "phrasecut_plus" = text + ảnh support như paper; "text_only" = ablation rd64-uni-novis
NEGATIVE_PROB = 0.2              # tỉ lệ mẫu âm: cụm từ không có trong ảnh, mask rỗng
MIX_TEXT_MAX = 0.5               # trọng số text w ~ U(0, 0,5); mẫu không có support dùng w = 1
PROMPT_AUGMENT = True            # train với 9 template "shuffle+" của tác giả
AUG_CROP = True                  # crop vuông ngẫu nhiên, giữ vật trong khung
LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
NUM_WORKERS = 0                  # có thể tăng lên 2 sau khi cache đầy đủ ảnh
RESUME_CHECKPOINT = None         # checkpoint epoch của chính notebook, hoặc None
USE_AMP = False                  # mặc định FP32 cho baseline
SEED = 42
THRESHOLD = 0.3
CALIBRATION_FRACTION = 0.2
THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
PROMPT_TEMPLATE = "{phrase}"      # giữ nguyên phrase, cố định trước evaluation
SAVE_PROBABILITIES = False       # True lưu float32 .npy (tốn thêm dung lượng)
N_VISUALIZATIONS = 6
DOWNLOAD_WORKERS = 4

MODEL_ID = "CIDAS/clipseg-rd64-refined"
MODEL_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
PHRASECUT_API_COMMIT = "276470b7773d8a230aee5b5536876fbc485fb04b"
""", "config")
cell("code", """
import csv
import hashlib
import importlib.metadata
import json
import math
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

assert BATCH_SIZE >= 1
assert isinstance(EPOCHS, int) and EPOCHS >= 1
assert TRAIN_INIT in ["random_decoder", "pretrained_decoder"]
assert TRAIN_MODE in ["phrasecut_plus", "text_only"]
assert 0 <= NEGATIVE_PROB < 1 and 0 <= MIX_TEXT_MAX <= 1
assert MAX_TRAIN_QUERIES is None or MAX_TRAIN_QUERIES >= 1
assert 0 <= MIN_LEARNING_RATE <= LEARNING_RATE and LEARNING_RATE > 0
assert MAX_QUERIES is None or MAX_QUERIES >= 1
assert 0 < CALIBRATION_FRACTION < 1
assert 0 <= THRESHOLD <= 1 and all(0 <= t <= 1 for t in THRESHOLD_GRID)
assert "{phrase}" in PROMPT_TEMPLATE
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
OUT = WORK_ROOT / "clipseg_baseline_miniv"
OUT.mkdir(parents=True, exist_ok=True)
# Xóa dấu hoàn thành cũ trước khi bắt đầu lượt mới.
(OUT / "summary.json").unlink(missing_ok=True)
(OUT / "run.json").unlink(missing_ok=True)
(WORK_ROOT / "clipseg_baseline_miniv.zip").unlink(missing_ok=True)

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
)
cell("markdown", """
## 3. Lấy annotation chính thức và chọn query

Chỉ tải `refer_miniv.json` và metadata ảnh từ Drive ID trong
[downloader tác giả](https://github.com/ChenyunWu/PhraseCutDataset/blob/276470b7773d8a230aee5b5536876fbc485fb04b/download_dataset.py).
Chỉ các ảnh miniv được chọn mới được tải, không tải toàn bộ Visual Genome.
Nếu Google Drive hết quota, Add Input dữ liệu đã tải rồi đặt `DATA_ROOT`.

Chia calibration/evaluation theo **image_id**, seed 42, trước khi đọc prediction.
Nếu giới hạn số query, sampling bằng seed, lưu đúng danh sách vào `selected_tasks.json`.
"""
)
cell("code", """
def metadata_path(root):
    for name in ["image_data_split.json", "image_data_split3000.json"]:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return root / "image_data_split.json"

if DATA_ROOT is None:
    candidates = sorted(Path("/kaggle/input").rglob("refer_miniv.json")) if Path("/kaggle/input").exists() else []
    candidates = [p.parent for p in candidates if metadata_path(p.parent).is_file()]
    if len(candidates) > 1:
        raise ValueError("Có nhiều dataset miniv; đặt DATA_ROOT rõ ràng: " + str(candidates))
    data_root = candidates[0] if candidates else WORK_ROOT / "phrasecut_miniv"
else:
    data_root = Path(DATA_ROOT)
    if not data_root.is_dir():
        raise FileNotFoundError(f"DATA_ROOT không tồn tại: {data_root}")
data_root.mkdir(parents=True, exist_ok=True)
annotation_path = data_root / "refer_miniv.json"
info_path = metadata_path(data_root)

def ensure_json(path, file_id):
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not value:
            raise ValueError(f"JSON không phải danh sách annotation: {path}")
        return value
    if str(path).startswith("/kaggle/input/"):
        raise FileNotFoundError(f"Thiếu {path.name} trong input read-only; bổ sung Kaggle Dataset.")
    temp = path.with_suffix(".download")
    try:
        result = gdown.download(id=file_id, output=str(temp), quiet=False)
        if result is None:
            raise RuntimeError("Google Drive không trả về file.")
        value = json.loads(temp.read_text(encoding="utf-8"))
        if not isinstance(value, list) or not value:
            raise ValueError("Nội dung download không phải annotation JSON.")
        temp.replace(path)
        return value
    except Exception as exc:
        raise RuntimeError(f"Không tải được {path.name}. Bật Internet, hoặc Add Input dataset và đặt DATA_ROOT.") from exc

tasks = ensure_json(annotation_path, "1oLcTQ1blTIQyu5ZMelQN2HSniuQaxM4E")
image_info = ensure_json(info_path, "1xB9eExJo35K3DQu8PnBWZ1q_OMoMl8-i")
info_by_id = {int(row["image_id"]): row for row in image_info if row["split"] == "miniv"}
assert len({str(t["task_id"]) for t in tasks}) == len(tasks), "Task ID bị trùng."
assert all(int(t["image_id"]) in info_by_id for t in tasks), "Annotation không thuộc split miniv."
assert all(t.get("Polygons") and t.get("phrase", "").strip() for t in tasks), "Thiếu polygon GT hoặc phrase."
assert all(re.fullmatch(r"[A-Za-z0-9_-]+", str(t["task_id"])) for t in tasks), "Task ID không hợp lệ."

selected = sorted(tasks, key=lambda t: str(t["task_id"]))
if MAX_QUERIES is not None and MAX_QUERIES < len(selected):
    selected = random.Random(SEED).sample(selected, MAX_QUERIES)
selected.sort(key=lambda t: (int(t["image_id"]), str(t["task_id"])))
image_ids = sorted({int(t["image_id"]) for t in selected})
if len(image_ids) < 2:
    raise ValueError("Cần ít nhất 2 image_id để chia calibration/evaluation; tăng MAX_QUERIES.")
shuffled_ids = image_ids.copy()
random.Random(SEED).shuffle(shuffled_ids)
n_cal = min(len(image_ids) - 1, max(1, round(len(image_ids) * CALIBRATION_FRACTION)))
calibration_ids = set(shuffled_ids[:n_cal])
group_for_image = {i: "calibration" if i in calibration_ids else "evaluation" for i in image_ids}
assert not calibration_ids.intersection(set(image_ids) - calibration_ids)
write_json(OUT / "selected_tasks.json", [{"task_id": str(t["task_id"]), "image_id": int(t["image_id"]),
    "phrase": t["phrase"], "group": group_for_image[int(t["image_id"])]} for t in selected])
print(f"Annotation: {len(tasks)} queries / {len(info_by_id)} miniv images")
print(f"Selected: {len(selected)} queries / {len(image_ids)} images; calibration {n_cal} images")
"""
)
cell("markdown", """
## 4. Tải và kiểm tra ảnh

Ưu tiên ảnh đã có trong dataset input. Ảnh thiếu sẽ tải từ URL Visual Genome trong metadata
bằng HTTPS vào working cache, retry có giới hạn. Lỗi tải hoặc kích thước khác metadata làm
dừng lượt chạy: không âm thầm bỏ query để tạo số liệu đẹp hơn.
"""
)
cell("code", """
image_cache = WORK_ROOT / "phrasecut_image_cache"
image_cache.mkdir(parents=True, exist_ok=True)
local_images = {}
for suffix in ["*.jpg", "*.jpeg", "*.png"]:
    for path in sorted(data_root.rglob(suffix)):
        if path.stem.isdigit() and int(path.stem) in image_ids:
            local_images.setdefault(int(path.stem), path)

def valid_image(path, info):
    with Image.open(path) as image:
        image.load()
        if image.size != (int(info["width"]), int(info["height"])):
            raise ValueError(f"{path.name}: ảnh {image.size} != metadata {(info['width'], info['height'])}")

def prepare_image(image_id):
    info = info_by_id[image_id]
    path = local_images.get(image_id, image_cache / f"{image_id}.jpg")
    if path.is_file():
        valid_image(path, info)
        return image_id, path
    url = info["url"].replace("http://", "https://", 1)
    temp = path.with_suffix(".download")
    for attempt in range(3):
        try:
            with requests.get(url, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                with temp.open("wb") as f:
                    for chunk in response.iter_content(1024 * 1024):
                        f.write(chunk)
            valid_image(temp, info)
            temp.replace(path)
            return image_id, path
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)

image_paths, image_errors = {}, []
def prepare_safe(image_id):
    try:
        return prepare_image(image_id), None
    except Exception as exc:
        return None, {"image_id": image_id, "error": str(exc)}

with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
    for result, error in tqdm(pool.map(prepare_safe, image_ids), total=len(image_ids), desc="Images"):
        if error:
            image_errors.append(error)
        else:
            image_paths[result[0]] = result[1]
write_json(OUT / "image_errors.json", image_errors)
if image_errors:
    raise RuntimeError(f"{len(image_errors)} ảnh lỗi; xem image_errors.json. Sửa dữ liệu rồi chạy lại.")
print("Validated images:", len(image_paths))
"""
)
cell("markdown", """
## 5. Nạp model và định nghĩa inference/metric

Dùng [triển khai CLIPSeg của Transformers](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/clipseg),
revision checkpoint được cố định. GT được hợp polygon theo
[API PhraseCut](https://github.com/ChenyunWu/PhraseCutDataset/blob/276470b7773d8a230aee5b5536876fbc485fb04b/utils/data_transfer.py).
Không dùng GT để tạo text prompt hoặc thay đổi đầu vào ảnh.
"""
)
cell("code", """
load_kwargs = {"local_files_only": True} if MODEL_PATH else {"revision": MODEL_REVISION}
model_source = str(MODEL_PATH) if MODEL_PATH else MODEL_ID
processor = CLIPSegProcessor.from_pretrained(model_source, use_fast=False, **load_kwargs)
model = CLIPSegForImageSegmentation.from_pretrained(model_source, **load_kwargs).to(device).eval()
assert processor.image_processor.size == {"height": 352, "width": 352}, "Processor không có input 352 như baseline."
assert model.config.vision_config.patch_size == 16, "Baseline yêu cầu ViT-B/16."
print("Model:", model_source, "| parameters:", sum(p.numel() for p in model.parameters()))

def rasterize_gt(task, width, height):
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for instance in task["Polygons"]:
        for polygon in instance:
            if len(polygon) >= 2:
                draw.polygon([(int(x), int(y)) for x, y in polygon], outline=1, fill=1)
    target = np.asarray(mask, dtype=bool)
    if not target.any():
        raise ValueError(f"GT rỗng: task {task['task_id']}")
    return target

def synchronize():
    if device.type == "cuda":
        torch.cuda.synchronize(device)

@torch.inference_mode()
def predict_batch(image, batch_tasks):
    prompts = [PROMPT_TEMPLATE.format(phrase=t["phrase"]) for t in batch_tasks]
    inputs = dict(processor.tokenizer(text=prompts, padding=True, truncation=True, return_tensors="pt"))
    inputs.update(processor(images=[image] * len(prompts), return_tensors="pt"))
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

def binary_metrics(probability, target, threshold):
    if probability.shape != target.shape or not target.any():
        raise ValueError("Shape không khớp hoặc GT rỗng.")
    prediction = probability >= threshold
    intersection = int(np.logical_and(prediction, target).sum())
    union = int(np.logical_or(prediction, target).sum())
    return {"intersection": intersection, "union": union, "iou": intersection / union,
            "dice": 2 * intersection / int(prediction.sum() + target.sum())}

def summarize(rows):
    if not rows:
        raise ValueError("Không có query để tính metric.")
    result = {"n_queries": len(rows), "n_images": len({r["image_id"] for r in rows}),
        "mean_sample_iou": float(np.mean([r["iou"] for r in rows])),
        "cumulative_iou": sum(r["intersection"] for r in rows) / sum(r["union"] for r in rows),
        "mean_dice": float(np.mean([r["dice"] for r in rows]))}
    for cutoff in [0.3, 0.5, 0.7, 0.9]:
        result[f"precision_at_{cutoff}"] = float(np.mean([r["iou"] >= cutoff for r in rows]))
    if "pixel_ap" in rows[0]:
        result["mean_sample_pixel_ap"] = float(np.mean([r["pixel_ap"] for r in rows]))
    return result
"""
)
cell("markdown", """
## 6. Biến PhraseCut train thành PhraseCut+ và tính số bước từ epoch

PhraseCut+ không phải file dữ liệu riêng: nó được tạo **ngẫu nhiên mỗi lần lấy mẫu** từ PhraseCut,
theo `PhraseCutPlus` trong [datasets/phrasecut.py](https://github.com/timojl/clipseg/blob/master/datasets/phrasecut.py)
và vòng train trong [training.py](https://github.com/timojl/clipseg/blob/master/training.py) của tác giả:

1. **Crop vuông ngẫu nhiên** cạnh min(H, W), ưu tiên crop chứa vật (`find_crop`), resize 352.
2. **Mẫu âm** với xác suất `NEGATIVE_PROB`: thay cụm từ bằng cụm khác, mask đáp án rỗng.
3. **Support** (`TRAIN_MODE="phrasecut_plus"`): chọn ngẫu nhiên một mẫu train **ở ảnh khác** có cùng cụm từ,
   dùng mask GT của chính ảnh support để làm mờ nền, giữ nền 10%, crop quanh vật (nới 10%), pad vuông, resize 352
   (`crop_blur_highlight352`). Cụm từ không xuất hiện ở ảnh khác thì không có support.
4. **Template** ngẫu nhiên trong 9 template `shuffle+`.
5. Trong vòng train: `cond = w·text + (1−w)·ảnh`, `w ~ U(0, MIX_TEXT_MAX)`; mẫu không có support dùng `w = 1`.

`TRAIN_MODE="text_only"` bỏ bước 3 và 5, tương ứng cấu hình `rd64-uni-novis` của tác giả (ablation chỉ học text).
Khác bản gốc: support luôn lấy từ ảnh khác (bản gốc có thể chọn trúng chính mẫu đó); GT rasterize bằng PIL như phần đánh giá.

`steps_per_epoch = ceil(N_train / BATCH_SIZE)`, `total_steps = EPOCHS × steps_per_epoch`.
Mỗi epoch duyệt hết tập train đã chọn một lần, shuffle, **không bỏ batch cuối**.
Tác giả train batch 64, 20.000 bước (`rd64-uni`) hoặc 50.000 bước (`rd64-uni-refined`), AMP;
với 310.816 query train, 20.000 × 64 tương đương khoảng 4 epoch.
Mặc định reset decoder ngẫu nhiên, giữ CLIP pretrained; `TRAIN_INIT="pretrained_decoder"`
fine-tune decoder của checkpoint refined.

Train full cần annotation ~344 MB và nhiều GB ảnh; nên Add Input dataset đã cache.
Nếu `TRAIN_ROOT=None`, tự tìm `refer_train.json` trong Input; nếu chưa có thì tải annotation
chính thức và tải ảnh train theo nhu cầu vào working cache. `MAX_TRAIN_QUERIES=8` để smoke.
Miniv chỉ dùng đánh giá sau train, không dùng cập nhật trọng số hoặc chọn checkpoint.
"""
)
cell("code", """
from torch.utils.data import Dataset, DataLoader

# 9 template "shuffle+" trong models/clipseg.py của repo CLIPSeg gốc.
SHUFFLE_PLUS_TEMPLATES = ["a photo of a {}.", "a photograph of a {}.", "an image of a {}.", "{}.",
    "a cropped photo of a {}.", "a good photo of a {}.", "a photo of one {}.", "a bad photo of a {}.",
    "a photo of the {}."]

def random_square_crop(target, iterations=50, min_frac=0.05):
    \"\"\"Crop vuông cạnh min(H, W); nhận crop đầu tiên có số pixel vật > min_frac × diện tích ảnh (find_crop gốc).\"\"\"
    height, width = target.shape
    side = min(height, width)
    min_sum = height * width * min_frac
    best = (-1, 0, 0)
    for _ in range(iterations):
        y = int(torch.randint(0, height - side + 1, (1,)))
        x = int(torch.randint(0, width - side + 1, (1,)))
        total = int(target[y:y + side, x:x + side].sum())
        if total > min_sum:
            return y, x, side
        if total > best[0]:
            best = (total, y, x)
    return best[1], best[2], side

def gaussian_blur(image, kernel_size=15, sigma=3.0):
    \"\"\"Gaussian blur [C,H,W] biên reflect, tương đương cv2.GaussianBlur(img, (15, 15), 3) của repo gốc.\"\"\"
    coords = torch.arange(kernel_size, dtype=image.dtype) - (kernel_size - 1) / 2
    kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    channels, pad = image.shape[0], kernel_size // 2
    blurred = F.pad(image.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    blurred = F.conv2d(blurred, kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1), groups=channels)
    blurred = F.conv2d(blurred, kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1), groups=channels)
    return blurred.squeeze(0)

def visual_prompt(image, mask, size=352, blur_sigma=3.0, background_factor=0.1, context=0.1):
    \"\"\"crop_blur_highlight352: nền làm mờ còn 10% (trên ảnh đã chuẩn hóa), crop quanh vật nới 10%, pad vuông, resize.\"\"\"
    mask = mask.to(image.dtype)
    blended = image * mask + background_factor * gaussian_blur(image, sigma=blur_sigma) * (1 - mask)
    ys, xs = torch.nonzero(mask > 0, as_tuple=True)
    if len(ys):
        y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    else:
        y0, y1, x0, x1 = 0, mask.shape[0], 0, mask.shape[1]
    dy, dx = context * (y1 - y0), context * (x1 - x0)
    y0, y1 = int(max(0, y0 - dy)), int(min(mask.shape[0], y1 + dy))
    x0, x1 = int(max(0, x0 - dx)), int(min(mask.shape[1], x1 + dx))
    crop = blended[:, y0:y1, x0:x1]
    height, width = crop.shape[1:]
    side = max(height, width)
    top, left = (side - height) // 2, (side - width) // 2
    crop = F.pad(crop, (left, side - width - left, top, side - height - top))
    return F.interpolate(crop.unsqueeze(0), size=(size, size), mode="nearest").squeeze(0)

def mix_conditionals(text_cond, visual_cond, support_valid, max_text_weight):
    \"\"\"cond = w·text + (1−w)·ảnh, w ~ U(0, max) theo từng mẫu; mẫu không có support dùng w = 1.\"\"\"
    weights = torch.rand(text_cond.shape[0], 1) * max_text_weight
    weights = torch.maximum(weights, (~support_valid.cpu()).float()[:, None])
    weights = weights.to(text_cond.device, text_cond.dtype)
    return text_cond * weights + visual_cond * (1 - weights)

training_metadata = {"enabled": TRAIN_DECODER}
if TRAIN_DECODER:
    if TRAIN_ROOT is None:
        train_candidates = sorted(Path("/kaggle/input").rglob("refer_train.json")) if Path("/kaggle/input").exists() else []
        if len(train_candidates) > 1:
            raise ValueError("Có nhiều dataset train; đặt TRAIN_ROOT rõ ràng.")
        train_root = train_candidates[0].parent if train_candidates else WORK_ROOT / "phrasecut_train"
    else:
        train_root = Path(TRAIN_ROOT)
        if not train_root.is_dir():
            raise FileNotFoundError(f"TRAIN_ROOT không tồn tại: {train_root}")
    train_root.mkdir(parents=True, exist_ok=True)
    train_annotation_path = train_root / "refer_train.json"
    train_raw = ensure_json(train_annotation_path, "1qx-0q6r9r0YUGpoyT0B8HJKmUFWQDSu7")
    train_info_by_id = {int(row["image_id"]): row for row in image_info if row["split"] == "train"}
    # Các image_id sai kích thước được loại trong wrapper CLIPSeg gốc.
    bad_image_ids = {150417, 285665, 498246, 61564, 285743, 498269, 498010, 150516,
                     150344, 286093, 61530, 150333, 286065, 285814, 498187, 285761, 498042}
    train_tasks = []
    excluded_train = []
    for task in train_raw:
        image_id = int(task["image_id"])
        if image_id not in train_info_by_id:
            raise ValueError(f"Task {task['task_id']} không thuộc split train chính thức.")
        if image_id in bad_image_ids:
            excluded_train.append({"task_id": str(task["task_id"]), "image_id": image_id,
                                   "reason": "known_dimension_mismatch_in_original_clipseg_wrapper"})
            continue
        if not task.get("Polygons") or not task.get("phrase", "").strip():
            raise ValueError(f"Train task thiếu GT hoặc phrase: {task['task_id']}")
        train_tasks.append({key: task[key] for key in ["task_id", "image_id", "phrase", "Polygons"]})
    del train_raw
    train_tasks.sort(key=lambda t: str(t["task_id"]))
    full_train_queries = len(train_tasks)
    if MAX_TRAIN_QUERIES is not None and MAX_TRAIN_QUERIES < len(train_tasks):
        train_tasks = random.Random(SEED).sample(train_tasks, MAX_TRAIN_QUERIES)
    if not train_tasks:
        raise ValueError("Train dataset rỗng.")
    train_image_ids = {int(t["image_id"]) for t in train_tasks}
    assert train_image_ids.isdisjoint(info_by_id), "Train/miniv bị trùng image_id."
    assert len({str(t["task_id"]) for t in train_tasks}) == len(train_tasks), "Train task ID bị trùng."
    write_json(OUT / "excluded_train_tasks.json", excluded_train)
    write_json(OUT / "train_tasks.json", [{"task_id": str(t["task_id"]), "image_id": int(t["image_id"]),
                                          "phrase": t["phrase"]} for t in train_tasks])
    train_local_images = {}
    for suffix in ["*.jpg", "*.jpeg", "*.png"]:
        for path in sorted(train_root.rglob(suffix)):
            if path.stem.isdigit() and int(path.stem) in train_image_ids:
                train_local_images.setdefault(int(path.stem), path)
    train_image_cache = WORK_ROOT / "phrasecut_train_image_cache"
    train_image_cache.mkdir(parents=True, exist_ok=True)

    def get_train_image_path(image_id):
        path = train_local_images.get(image_id, train_image_cache / f"{image_id}.jpg")
        info = train_info_by_id[image_id]
        if path.is_file():
            return path
        # Temp riêng cho từng worker, tránh ghi đè nếu nhiều query có cùng ảnh.
        temp = path.with_name(f"{image_id}.{os.getpid()}.download")
        url = info["url"].replace("http://", "https://", 1)
        for attempt in range(3):
            try:
                with requests.get(url, stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    with temp.open("wb") as f:
                        for chunk in response.iter_content(1024 * 1024):
                            f.write(chunk)
                valid_image(temp, info)
                temp.replace(path)
                return path
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    # Chỉ mục cụm từ -> các mẫu train, dùng để chọn support và mẫu âm.
    tasks_by_phrase = defaultdict(list)
    for index, task in enumerate(train_tasks):
        tasks_by_phrase[task["phrase"]].append(index)
    all_train_phrases = sorted(tasks_by_phrase)
    queries_with_support = sum(1 for task in train_tasks if any(
        train_tasks[i]["image_id"] != task["image_id"] for i in tasks_by_phrase[task["phrase"]]))
    image_mean = torch.tensor(processor.image_processor.image_mean, dtype=torch.float32).view(3, 1, 1)
    image_std = torch.tensor(processor.image_processor.image_std, dtype=torch.float32).view(3, 1, 1)

    def load_train_sample(task):
        \"\"\"Ảnh chuẩn hóa [3,352,352] và mask [352,352], crop vuông ngẫu nhiên như load_sample gốc.\"\"\"
        image_id = int(task["image_id"])
        with Image.open(get_train_image_path(image_id)) as raw:
            image = raw.convert("RGB")
        info = train_info_by_id[image_id]
        if image.size != (int(info["width"]), int(info["height"])):
            raise ValueError(f"Train ảnh {image_id} sai kích thước metadata.")
        target = rasterize_gt(task, image.width, image.height)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1)
        if AUG_CROP:
            y, x, side = random_square_crop(target)
            target, pixels = target[y:y + side, x:x + side], pixels[:, y:y + side, x:x + side]
        pixels = F.interpolate(pixels.unsqueeze(0), size=(352, 352), mode="bilinear", align_corners=True).squeeze(0)
        mask = F.interpolate(torch.from_numpy(target.astype(np.float32))[None, None], size=(352, 352), mode="nearest")[0, 0]
        return (pixels / 255 - image_mean) / image_std, mask

    class PhraseCutPlusDataset(Dataset):
        \"\"\"PhraseCut+ tạo lại ở mỗi lần lấy mẫu: mẫu âm, support cùng cụm từ từ ảnh khác, template ngẫu nhiên.\"\"\"
        def __len__(self):
            return len(train_tasks)

        def __getitem__(self, index):
            task = train_tasks[index]
            image, target = load_train_sample(task)
            phrase = task["phrase"]
            if len(all_train_phrases) > 1 and float(torch.rand(1)) < NEGATIVE_PROB:
                while phrase == task["phrase"]:
                    phrase = all_train_phrases[int(torch.randint(0, len(all_train_phrases), (1,)))]
                target = torch.zeros_like(target)
            support, support_valid = torch.zeros_like(image), False
            if TRAIN_MODE == "phrasecut_plus":
                candidates = [i for i in tasks_by_phrase[phrase] if train_tasks[i]["image_id"] != task["image_id"]]
                if candidates:
                    other = train_tasks[candidates[int(torch.randint(0, len(candidates), (1,)))]]
                    support, support_valid = visual_prompt(*load_train_sample(other)), True
            if PROMPT_AUGMENT:
                prompt = SHUFFLE_PLUS_TEMPLATES[int(torch.randint(0, len(SHUFFLE_PLUS_TEMPLATES), (1,)))].format(phrase)
            else:
                prompt = PROMPT_TEMPLATE.format(phrase=phrase)
            return image, prompt, support, support_valid, target

    def collate_train(samples):
        images, prompts, supports, valid, masks = zip(*samples)
        inputs = dict(processor.tokenizer(text=list(prompts), padding=True, truncation=True, return_tensors="pt"))
        inputs.update(pixel_values=torch.stack(images), support_pixel_values=torch.stack(supports),
                      support_valid=torch.tensor(valid), labels=torch.stack(masks))
        return inputs

    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(PhraseCutPlusDataset(), batch_size=BATCH_SIZE, shuffle=True,
        drop_last=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
        collate_fn=collate_train, generator=generator)
    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch
    assert steps_per_epoch == math.ceil(len(train_tasks) / BATCH_SIZE)
    training_metadata = {"enabled": True, "split": "train", "initialization": TRAIN_INIT,
        "seed": SEED, "precision": "fp16_autocast" if USE_AMP else "fp32", "prompt_template": PROMPT_TEMPLATE,
        "base_model_source": model_source, "base_model_revision": MODEL_REVISION if not MODEL_PATH else None,
        "full_valid_train_queries": full_train_queries, "selected_train_queries": len(train_tasks),
        "train_images": len(train_image_ids), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "steps_per_epoch": steps_per_epoch, "total_planned_steps": total_steps,
        "drop_last": False, "lr": LEARNING_RATE, "min_lr": MIN_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "num_workers": NUM_WORKERS,
        "train_annotation_sha256": sha256(train_annotation_path),
        "train_manifest_sha256": sha256(OUT / "train_tasks.json"),
        "excluded_train_queries": len(excluded_train), "loss": "BCEWithLogitsLoss",
        "train_mode": TRAIN_MODE, "negative_prob": NEGATIVE_PROB,
        "mix_text_weight": [0.0, MIX_TEXT_MAX] if TRAIN_MODE == "phrasecut_plus" else None,
        "prompt_augment": "shuffle+" if PROMPT_AUGMENT else None, "aug_crop": AUG_CROP,
        "visual_prompt": "crop_blur_highlight352" if TRAIN_MODE == "phrasecut_plus" else None,
        "unique_train_phrases": len(all_train_phrases), "queries_with_support": queries_with_support,
        "protocol": f"{TRAIN_MODE}; frozen CLIP; final epoch checkpoint; miniv held out"}
    write_json(OUT / "training_config.json", training_metadata)
    # Hình kiểm tra 4 mẫu PhraseCut+; fork_rng để không làm đổi RNG của lượt train.
    with torch.random.fork_rng():
        torch.manual_seed(SEED)
        preview = [PhraseCutPlusDataset()[i] for i in torch.randperm(len(train_tasks))[:4].tolist()]
    fig, axes = plt.subplots(len(preview), 3, figsize=(9, 3 * len(preview)), squeeze=False)
    for row, (image, prompt, support, valid, target) in zip(axes, preview):
        for ax, value, title in zip(row, [image, support, target], [f"Query: {prompt}", "Support" if valid else "Không có support", "Mask đáp án"]):
            shown = (value * image_std + image_mean).clamp(0, 1).permute(1, 2, 0) if value.ndim == 3 else value
            ax.imshow(shown, cmap="gray" if value.ndim == 2 else None, vmin=0, vmax=1)
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "phrasecut_plus_samples.png", dpi=100)
    plt.close(fig)
    print(f"Mode={TRAIN_MODE}; N_train={len(train_tasks)}; phrases={len(all_train_phrases)}; "
          f"queries with support from another image={queries_with_support}")
    print(f"N_train={len(train_tasks)}, epochs={EPOCHS}, batch={BATCH_SIZE}")
    print(f"steps/epoch={steps_per_epoch}, total optimizer steps={total_steps}")
else:
    print("TRAIN_DECODER=False: đánh giá checkpoint pretrained, không train.")
"""
)
cell("markdown", """
## 7. Train decoder theo epoch và lưu checkpoint

AdamW + cosine LR từ `LEARNING_RATE` tới `MIN_LEARNING_RATE` trong `total_steps`.
Lưu loss theo epoch và checkpoint sau mỗi epoch, gồm decoder, optimizer, scheduler,
AMP scaler và RNG để resume tại ranh giới epoch. `RESUME_CHECKPOINT` phải có cùng
manifest và cấu hình (kể cả tổng EPOCHS); không tự đổi lịch LR khi tiếp tục.

Checkpoint chỉ lưu decoder vì CLIP đóng băng; ghép với base model/revision cố định để chạy lại.
Không dùng miniv để chọn epoch tốt nhất. Kết quả cuối notebook là model ở **epoch cuối**.
"""
)
cell("code", """
if TRAIN_DECODER:
    if TRAIN_INIT == "random_decoder":
        model.decoder = type(model.decoder)(model.config)
        model.decoder.post_init()
        model.decoder = model.decoder.to(device)
    model.clip.requires_grad_(False)
    model.decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=MIN_LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    signature = hashlib.sha256(json.dumps(training_metadata, sort_keys=True).encode()).hexdigest()
    start_epoch, global_step, history = 0, 0, []
    checkpoint_dir = OUT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    if not RESUME_CHECKPOINT:
        for old_checkpoint in checkpoint_dir.glob("decoder_epoch_*.pt"):
            old_checkpoint.unlink()
    if RESUME_CHECKPOINT:
        checkpoint = torch.load(RESUME_CHECKPOINT, map_location="cpu", weights_only=True)
        if checkpoint["config_signature"] != signature:
            raise ValueError("Checkpoint khác train manifest/config; giữ nguyên cấu hình khi resume.")
        model.decoder.load_state_dict(checkpoint["decoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, global_step, history = checkpoint["epoch"], checkpoint["global_step"], checkpoint["history"]
        generator.set_state(checkpoint["loader_rng"].cpu())
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        random.setstate(checkpoint["python_rng"])
        state = checkpoint["numpy_rng"]
        np.random.set_state((state[0], state[1].numpy().astype(np.uint32), state[2], state[3], state[4]))
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in checkpoint["cuda_rng"]])
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        model.clip.eval()
        model.decoder.train()
        loss_sum, seen = 0.0, 0
        epoch_started = time.perf_counter()
        for batch in tqdm(train_loader, desc=f"Train epoch {epoch + 1}/{EPOCHS}"):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()
            with context:
                size = batch["labels"].shape[0]
                cond = model.get_conditional_embeddings(batch_size=size, input_ids=batch["input_ids"],
                                                        attention_mask=batch["attention_mask"])
                if TRAIN_MODE == "phrasecut_plus":
                    visual_cond = model.get_conditional_embeddings(
                        batch_size=size, conditional_pixel_values=batch["support_pixel_values"])
                    cond = mix_conditionals(cond, visual_cond, batch["support_valid"], MIX_TEXT_MAX)
                logits = model(pixel_values=batch["pixel_values"], conditional_embeddings=cond).logits
                if logits.ndim == 2:
                    logits = logits.unsqueeze(0)
                loss = F.binary_cross_entropy_with_logits(logits.float(), batch["labels"])
            if not torch.isfinite(loss):
                raise ValueError("Training loss NaN/Inf; dừng để kiểm tra dữ liệu/AMP.")
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # AMP có thể bỏ optimizer step khi overflow; không tăng lịch LR lúc đó.
            if not USE_AMP or scaler.get_scale() >= old_scale:
                scheduler.step()
                global_step += 1
            count = batch["labels"].shape[0]
            loss_sum += loss.detach().item() * count
            seen += count
        assert seen == len(train_tasks), "Epoch chưa duyệt đủ train queries."
        synchronize()
        history.append({"epoch": epoch + 1, "train_loss": loss_sum / seen, "n_queries": seen,
            "optimizer_steps_total": global_step, "learning_rate_next": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_started})
        np_state = np.random.get_state()
        checkpoint = {"epoch": epoch + 1, "global_step": global_step, "config_signature": signature,
            "decoder": model.decoder.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "history": history,
            "loader_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "python_rng": random.getstate(),
            "numpy_rng": (np_state[0], torch.from_numpy(np_state[1].astype(np.int64)), np_state[2], np_state[3], np_state[4]),
            "base_model_id": MODEL_ID, "base_model_revision": MODEL_REVISION,
            "training": training_metadata}  # E2/E3 đọc epoch, batch size từ đây
        checkpoint_path = checkpoint_dir / f"decoder_epoch_{epoch + 1:03d}.pt"
        torch.save(checkpoint, checkpoint_path)
        write_csv(OUT / "training_history.csv", history)
        print(history[-1])
    assert start_epoch <= EPOCHS, "Checkpoint vượt EPOCHS được yêu cầu."
    model.eval()
    training_metadata.update({"completed_epochs": EPOCHS, "optimizer_steps_completed": global_step,
        "final_checkpoint": str(RESUME_CHECKPOINT) if start_epoch == EPOCHS else str(checkpoint_path)})
    write_json(OUT / "training_config.json", training_metadata)
    # Backbone không nhận gradient; chỉ decoder có tham số trainable.
    assert all(p.grad is None for p in model.clip.parameters())
    print("Training complete. Evaluate final epoch on miniv next.")
"""
)
cell("markdown", """
## 8. Inference toàn bộ query miniv đã chọn

Warm-up trước khi đo. Batch các phrase trên cùng ảnh để resize output đúng kích thước.
Lưu mask theo `task_id.png`, đúng quy ước PhraseCut; GT, ảnh phân tích và probability tùy chọn
được ghi riêng. CSV được checkpoint sau mỗi ảnh; nếu lỗi, kết quả một phần giữ lại nhưng
chưa có `summary.json` hoàn chỉnh. Chạy lại cell sẽ tính lại toàn bộ để tránh ghép nhầm lượt chạy.
"""
)
cell("code", """
for name in ["masks", "ground_truth", "visualizations", "probabilities"]:
    (OUT / name).mkdir(exist_ok=True)
# Tách từng run bằng config/manifest: không trộn artifact cũ vào ZIP.
for folder in ["masks", "ground_truth", "visualizations", "probabilities"]:
    for path in (OUT / folder).iterdir():
        if path.is_file():
            path.unlink()
for filename in ["summary.json", "sample_metrics.csv", "calibration_thresholds.csv", "run.json"]:
    (OUT / filename).unlink(missing_ok=True)

tasks_by_image = defaultdict(list)
for task in selected:
    tasks_by_image[int(task["image_id"])].append(task)
first_image_id = image_ids[0]
with Image.open(image_paths[first_image_id]) as raw:
    warmup_image = raw.convert("RGB")
warmup_tasks = tasks_by_image[first_image_id][:BATCH_SIZE]
for _ in range(2):
    predict_batch(warmup_image, warmup_tasks)
if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats(device)

def save_visualization(image, target, probability, task, metrics, index):
    prediction = probability >= THRESHOLD
    overlay = np.asarray(image).astype(np.float32).copy()
    overlay[prediction] = 0.55 * overlay[prediction] + 0.45 * np.array([255, 60, 60])
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image); axes[0].set_title("Ảnh")
    axes[1].imshow(target, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth")
    axes[2].imshow(probability, cmap="magma", vmin=0, vmax=1); axes[2].set_title("Sigmoid probability")
    axes[3].imshow(overlay.astype(np.uint8)); axes[3].set_title(f"t={THRESHOLD}, IoU={metrics['iou']:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{task['task_id']} — {task['phrase']}")
    fig.tight_layout()
    fig.savefig(OUT / "visualizations" / f"{index:02d}_{task['task_id']}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

rows, sweep_rows, batch_times = [], [], []
visual_count = 0
started_all = time.perf_counter()
for image_id in tqdm(image_ids, desc="CLIPSeg images"):
    with Image.open(image_paths[image_id]) as raw:
        image = raw.convert("RGB")
    current_tasks = tasks_by_image[image_id]
    for start in range(0, len(current_tasks), BATCH_SIZE):
        batch = current_tasks[start:start + BATCH_SIZE]
        probabilities, seconds = predict_batch(image, batch)
        batch_times.append({"queries": len(batch), "forward_resize_seconds": seconds})
        for task, probability in zip(batch, probabilities):
            target = rasterize_gt(task, image.width, image.height)
            base = {"task_id": str(task["task_id"]), "image_id": image_id,
                "phrase": task["phrase"], "prompt": PROMPT_TEMPLATE.format(phrase=task["phrase"]),
                "group": group_for_image[image_id], "width": image.width, "height": image.height}
            metrics = binary_metrics(probability, target, THRESHOLD)
            rows.append({**base, "threshold": THRESHOLD, **metrics,
                "pixel_ap": float(average_precision_score(target.ravel(), probability.ravel()))})
            for threshold in THRESHOLD_GRID:
                sweep_rows.append({**base, "threshold": threshold, **binary_metrics(probability, target, threshold)})
            Image.fromarray((probability >= THRESHOLD).astype(np.uint8) * 255).save(OUT / "masks" / f"{task['task_id']}.png")
            Image.fromarray(target.astype(np.uint8) * 255).save(OUT / "ground_truth" / f"{task['task_id']}.png")
            if SAVE_PROBABILITIES:
                np.save(OUT / "probabilities" / f"{task['task_id']}.npy", probability)
            if visual_count < N_VISUALIZATIONS:
                save_visualization(image, target, probability, task, metrics, visual_count)
                visual_count += 1
    write_csv(OUT / "sample_metrics.csv", rows)
elapsed_all = time.perf_counter() - started_all
assert len(rows) == len(selected), "Chưa đánh giá đủ query."
print(f"Completed {len(rows)} queries; wall time {elapsed_all:.1f}s")
"""
)
cell("markdown", """
## 9. Tổng hợp và chọn ngưỡng trên calibration

Báo cáo chính `selected_miniv_fixed_threshold` dùng t=0,3 cho toàn bộ query đã chọn.
Ngưỡng bổ sung được chọn bằng mean IoU **chỉ trên calibration**, rồi báo riêng trên evaluation.
Không chọn template hoặc threshold theo evaluation. Nếu `MAX_QUERIES` khác None,
ghi đây là tập con, không gắn nhãn kết quả full miniv.
"""
)
cell("code", """
calibration_table = []
for threshold in THRESHOLD_GRID:
    subset = [r for r in sweep_rows if r["group"] == "calibration" and r["threshold"] == threshold]
    calibration_table.append({"threshold": threshold, **summarize(subset)})
best_threshold = max(calibration_table, key=lambda r: (r["mean_sample_iou"], -r["threshold"]))["threshold"]
write_csv(OUT / "calibration_thresholds.csv", calibration_table)
evaluation_fixed = [r for r in rows if r["group"] == "evaluation"]
evaluation_tuned = [r for r in sweep_rows if r["group"] == "evaluation" and r["threshold"] == best_threshold]
summary = {
    "training": training_metadata,
    "dataset": "VGPhraseCut", "split": "miniv", "complete_official_miniv": len(selected) == len(tasks),
    "threshold": THRESHOLD, "calibration_selected_threshold": best_threshold,
    "selected_miniv_fixed_threshold": summarize(rows),
    "calibration_fixed_threshold": summarize([r for r in rows if r["group"] == "calibration"]),
    "evaluation_fixed_threshold": summarize(evaluation_fixed),
    "evaluation_calibrated_threshold": summarize(evaluation_tuned),
    "metric_note": "IoU averaged per query; pixel AP averaged per query, not claimed to match paper AP.",
}
write_json(OUT / "summary.json", summary)
inference_seconds = sum(r["forward_resize_seconds"] for r in batch_times)
run = {
    "training": training_metadata,
    "model_id": MODEL_ID, "requested_revision": MODEL_REVISION if not MODEL_PATH else None,
    "model_source": model_source, "resolved_commit": getattr(model.config, "_commit_hash", None),
    "phrasecut_api_commit": PHRASECUT_API_COMMIT,
    "annotation_sha256": sha256(annotation_path), "image_metadata_sha256": sha256(info_path),
    "selected_tasks_sha256": sha256(OUT / "selected_tasks.json"),
    "seed": SEED, "max_queries": MAX_QUERIES, "batch_size": BATCH_SIZE,
    "prompt_template": PROMPT_TEMPLATE, "threshold": THRESHOLD, "threshold_grid": THRESHOLD_GRID,
    "calibration_fraction": CALIBRATION_FRACTION, "calibration_image_ids": sorted(calibration_ids),
    "evaluation_image_ids": sorted(set(image_ids) - calibration_ids),
    "precision": "fp16_autocast" if USE_AMP else "fp32", "device": str(device),
    "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    "torch_cuda_version": torch.version.cuda, "python": platform.python_version(), "versions": versions,
    "image_processor": processor.image_processor.to_dict(), "model_config": model.config.to_dict(),
    "parameter_count": sum(p.numel() for p in model.parameters()),
    "decoder_parameter_count": sum(p.numel() for p in model.decoder.parameters()),
    "forward_resize_seconds_total": inference_seconds,
    "forward_resize_ms_per_query": 1000 * inference_seconds / len(rows),
    "wall_seconds_inference_metrics_saving": elapsed_all,
    "peak_allocated_gpu_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
    "timing_note": "Two warmup batches; forward+resize+sigmoid timed with CUDA sync; excludes preprocessing, transfer and metric/save. GPU peak includes resizing.",
    "save_probabilities": SAVE_PROBABILITIES,
}
write_json(OUT / "run.json", run)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"Forward+resize+sigmoid: {run['forward_resize_ms_per_query']:.2f} ms/query")
"""
)
cell("markdown", """
## 10. Xem hình và đóng gói

Trong Kaggle Output, tải `clipseg_baseline_miniv.zip`. Giữ `run.json`, `selected_tasks.json`
và `sample_metrics.csv` cùng bảng số liệu để chạy lại đúng protocol.
Masks dùng ngưỡng cố định 0,3; ngưỡng calibration bổ sung chỉ được báo trong summary.
"""
)
cell("code", """
for path in sorted((OUT / "visualizations").glob("*.png"))[:N_VISUALIZATIONS]:
    with Image.open(path) as image:
        display(image.copy())
archive = WORK_ROOT / "clipseg_baseline_miniv.zip"
training_artifacts = {"training_config.json", "training_history.csv", "train_tasks.json", "excluded_train_tasks.json", "checkpoints"}
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
    for path in sorted(OUT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(OUT)
        if not TRAIN_DECODER and relative.parts[0] in training_artifacts:
            continue
        package.write(path, arcname=relative.as_posix())
print("Result folder:", OUT)
print("Download:", archive)
"""
)
cell("markdown", """
## Nguồn và giới hạn

- [CLIPSeg / CVPR 2022](https://github.com/timojl/clipseg), [checkpoint HF](https://huggingface.co/CIDAS/clipseg-rd64-refined).
- [PhraseCutDataset](https://github.com/ChenyunWu/PhraseCutDataset): annotation, URL ảnh, split và quy ước polygon/mask.
- [CLIPSeg trong Transformers 4.55.4](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/clipseg).

Kích thước mask được đánh giá ở độ phân giải ảnh gốc, dùng rasterization PIL của PhraseCut API.
Repo CLIPSeg gốc dùng `skimage.draw.polygon2mask` và có preprocessing/evaluation khác;
số liệu notebook này cần ghi kèm checkpoint và protocol, không so trực tiếp như cùng một thiết lập.
Khi `TRAIN_DECODER=True`: decoder khởi tạo ngẫu nhiên, CLIP đóng băng, train trên PhraseCut+ (`TRAIN_MODE`).
Chọn `TRAIN_MODE="text_only"` cho ablation chỉ học text; chọn `pretrained_decoder` để fine-tune.
Google Drive/Visual Genome có thể gián đoạn; dataset input và checkpoint offline là đường dự phòng.
Khi Internet tắt, còn cần các dependency đã cài hoặc wheel đưa vào input.
"""
)

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"}},
    "nbformat": 4, "nbformat_minor": 5,
}
output = Path(__file__).resolve().parents[1] / "source" / "baseline_clipseg_phrasecut_miniv.ipynb"
output.parent.mkdir(exist_ok=True)
output.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
print(output)
