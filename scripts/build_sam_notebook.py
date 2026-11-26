"""Build source/e5_clipseg_sam.ipynb: training-free SAM refinement of CLIPSeg masks, paper protocol.

The notebook reuses the E1 cells for data, CLIPSeg, decoder loading and paper metrics, then:
  * derives SAM prompts only from the CLIPSeg prediction (connected components of p >= THRESHOLD:
    one box per component, optionally its highest-probability point); ground truth is never used;
  * scores CLIPSeg and four SAM variants (box / box+point × replace / mean fusion) on identical queries;
  * selects one variant on the full val split with a rule fixed in advance, then reports test.
The CLIPSeg row must reproduce the E1 test numbers, which checks that the pipeline did not drift.
Edit this builder, not the generated notebook.
"""
import ast
import copy
import json
import re
from pathlib import Path

from build_e4_notebook import EVAL_LOAD_OLD, EVAL_LOAD_NEW, MODULE as E4_MODULE

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "source" / "e1_train_eval_phrasecut.ipynb"
OUTPUT = ROOT / "source" / "e5_clipseg_sam.ipynb"
E1_EPOCH7_SHA256 = "b9bc28509a6a6f36495bc4826a511d182bdee8885a819190997da56fd67947df"

TITLE = """
# XLAV — E5: CLIPSeg + SAM (tinh chỉnh mask không cần train)

Nâng cấp **pipeline**: mask của CLIPSeg (decoder E1 epoch 7) được dùng làm prompt cho **SAM** (ViT-B) để vẽ lại biên.
Không train lại gì. Prompt chỉ lấy từ dự đoán của CLIPSeg — ground truth chỉ dùng để chấm.

- Prompt: các vùng liên thông của `p ≥ THRESHOLD` (tối đa `SAM_MAX_COMPONENTS` vùng, bỏ vùng nhỏ hơn
  `SAM_MIN_COMPONENT_RATIO` × vùng lớn nhất); mỗi vùng một box, tuỳ biến thể thêm điểm có xác suất cao nhất.
  Phrase nhiều instance → nhiều box, gộp bằng max logit.
- Biến thể: `clipseg` (gốc), `sam_box_replace`, `sam_box_mean`, `sam_boxpoint_replace`, `sam_boxpoint_mean`.
  `replace` = xác suất SAM; `mean` = trung bình xác suất CLIPSeg và SAM. Query không có vùng nào ≥ ngưỡng giữ nguyên CLIPSeg.
- Chấm theo protocol paper (352×352, 49 ngưỡng, FP32) cho mọi biến thể trên **cùng query**.
- **Chọn trên val, báo trên test:** biến thể SAM có mIoU@0,3 trên val cao nhất được chốt trước khi chấm test.
  Dòng `clipseg` trên test phải trùng số test đã biết của decoder (`CLIPSEG_REFERENCE_TEST`; E1: 46,66 / 54,81 / 77,95)
  — kiểm tra pipeline không lệch.
- Decoder chọn bằng `DECODER_LABEL` + `DECODER_SHA256`: E1 epoch 7 (mặc định), B1, hoặc E4c (checkpoint có LoRA trên
  CLIP vision — notebook tự gắn LoRA, embedding điều kiện vẫn dùng CLIP gốc như lúc train).

**Chạy trên Kaggle:** Add Input `poudelsandeep/phrasecut` và dataset chứa checkpoint decoder (tìm theo SHA256);
GPU T4 (x1 là đủ), Internet on; **Save & Run All**.
Một phiên chấm hết val rồi test (ước tính 3–4 giờ). Kết quả ở `/kaggle/working/clipseg_e5_sam.zip`.
"""

SAM_CONFIG = f'''
SAM_MODEL = "facebook/sam-vit-base"   # SAM ViT-B trong Transformers 4.55.4
SAM_MAX_COMPONENTS = 4                # số vùng liên thông tối đa làm prompt cho mỗi query
SAM_MIN_COMPONENT_RATIO = 0.1         # bỏ vùng nhỏ hơn tỉ lệ này × vùng lớn nhất
DECODER_SHA256 = "{E1_EPOCH7_SHA256}"  # E1 epoch 7; DECODER_CHECKPOINT=None thì tự tìm trong Kaggle Input
DECODER_LABEL = "e1"                  # "e1" | "b1" | "e4c_lr003": tên decoder ghi vào kết quả
CLIPSEG_REFERENCE_TEST = {{"miou_0.3": 0.4666, "fgiou_0.3": 0.5481, "ap": 0.7795}}  # số test đã biết của decoder này
LORA_RANK = 8                         # chỉ dùng khi checkpoint có LoRA nhưng không ghi cấu hình
LORA_ALPHA = 16
LORA_GRAD_CHECKPOINT = False
VARIANTS = ["clipseg", "sam_box_replace", "sam_box_mean", "sam_boxpoint_replace", "sam_boxpoint_mean"]
'''

CONFIG_VALUES = {"EVAL_SPLIT": '"val"', "MAX_QUERIES": "None", "TRAIN_DECODER": "False", "EVAL_AMP": "False",
                 "REQUIRE_GPU": "True", "BATCH_SIZE": "8"}

LOCATE = '''
# Checkpoint decoder trong Kaggle Input, tìm theo SHA256 (tên file có thể là decoder_epoch_002/007.pt).
if DECODER_CHECKPOINT is None:
    found = sorted(Path("/kaggle/input").rglob("*.pt")) if Path("/kaggle/input").exists() else []
    found = [p for p in found if p.stat().st_size < 500 * 1024**2 and sha256(p) == DECODER_SHA256]
    if not found:
        raise FileNotFoundError(f"Không tìm thấy checkpoint {DECODER_LABEL} (SHA256 {DECODER_SHA256[:8]}…) trong Kaggle Input.")
    DECODER_CHECKPOINT = str(found[0])
assert sha256(Path(DECODER_CHECKPOINT)) == DECODER_SHA256, f"Checkpoint khác {DECODER_LABEL}."
print("Decoder:", DECODER_LABEL, DECODER_CHECKPOINT)
'''

# Hàm LoRA và forward dùng chung của notebook E4, để nạp được checkpoint E4c.
E4_CODE = ast.unparse(ast.Module(body=[node for node in ast.parse(E4_MODULE).body
    if isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef))], type_ignores=[])) + "\n"

LORA_CHECK = '''
if has_lora(model):
    # Forward của Transformers (dùng trong predict_batch) phải trùng segment của lúc train/chấm E4c.
    with torch.no_grad():
        probe = processor.tokenizer(["a red car", "dog"], padding=True, return_tensors="pt").to(device)
        probe_pixels = torch.randn(2, 3, PAPER_SIZE, PAPER_SIZE, device=device)
        assert torch.allclose(model(pixel_values=probe_pixels, **probe).logits,
                              segment(model, probe_pixels, probe["input_ids"], probe["attention_mask"]), atol=1e-4)
    print(f"LoRA: {len(lora_modules(model))} lớp, {sum(p.numel() for p in lora_parameters(model)):,} tham số; "
          "forward khớp segment")
else:
    print("Decoder không có LoRA")
'''

SAM_SETUP_MD = """
## E5. SAM và các hàm tinh chỉnh

Prompt tính từ xác suất CLIPSeg ở độ phân giải gốc. Toạ độ được đổi sang khung 1024 của SAM
(cạnh dài = 1024, như `SamProcessor`). Logit SAM được đưa về kích thước gốc bằng `post_process_masks`,
gộp các vùng bằng max, rồi resize bilinear về 352×352 để chấm theo protocol paper.
"""

SAM_SETUP = '''
from scipy import ndimage
from transformers import SamModel, SamProcessor

sam_processor = SamProcessor.from_pretrained(SAM_MODEL)
sam_model = SamModel.from_pretrained(SAM_MODEL).to(device).eval()
sam_commit = getattr(sam_model.config, "_commit_hash", None)
print("SAM:", SAM_MODEL, sam_commit, "| tham số:", sum(p.numel() for p in sam_model.parameters()))
PAPER_T03 = int(np.argmin(np.abs(PAPER_THRESHOLDS - THRESHOLD)))

def prompt_components(probability, threshold=THRESHOLD):
    """Box [x0, y0, x1, y1] và điểm [x, y] (tâm pixel có xác suất cao nhất) cho từng vùng liên thông p >= threshold."""
    labels, count = ndimage.label(probability >= threshold)
    if count == 0:
        return []
    areas = np.bincount(labels.ravel())[1:]
    order = np.argsort(-areas, kind="stable")
    prompts = []
    for index in order[:SAM_MAX_COMPONENTS]:
        if areas[index] < SAM_MIN_COMPONENT_RATIO * areas[order[0]]:
            break
        region = labels == index + 1
        ys, xs = np.nonzero(region)
        y, x = np.unravel_index(np.where(region, probability, -1.0).argmax(), probability.shape)
        prompts.append(([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                        [float(x) + 0.5, float(y) + 0.5]))
    return prompts

def sam_coordinate_scale(height, width, longest_edge=1024):
    """Hệ số đổi toạ độ (x, y) sang khung SAM, làm tròn kích thước như SamProcessor._normalize_coordinates."""
    scale = longest_edge / max(height, width)
    new_height, new_width = int(height * scale + 0.5), int(width * scale + 0.5)
    return new_width / width, new_height / height

@torch.inference_mode()
def sam_image(image):
    """Embedding SAM một lần cho mỗi ảnh, kèm kích thước để đổi toạ độ và hậu xử lý mask."""
    inputs = sam_processor(images=image, return_tensors="pt")
    embeddings = sam_model.get_image_embeddings(inputs["pixel_values"].to(device))
    return {"embeddings": embeddings, "original_sizes": inputs["original_sizes"],
            "reshaped_input_sizes": inputs["reshaped_input_sizes"]}

@torch.inference_mode()
def sam_logits(sam_state, prompts, use_points):
    """Logit SAM ở kích thước gốc [H, W]: mỗi vùng một mask (multimask_output=False), gộp bằng max."""
    height, width = (int(v) for v in sam_state["original_sizes"][0])
    sx, sy = sam_coordinate_scale(height, width)
    boxes = torch.tensor([[box for box, _ in prompts]], dtype=torch.float32) * torch.tensor([sx, sy, sx, sy])
    kwargs = {"input_boxes": boxes.to(device)}
    if use_points:
        points = torch.tensor([[[point] for _, point in prompts]], dtype=torch.float32) * torch.tensor([sx, sy])
        kwargs["input_points"] = points.to(device)
        kwargs["input_labels"] = torch.ones((1, len(prompts), 1), dtype=torch.long, device=device)
    outputs = sam_model(image_embeddings=sam_state["embeddings"], multimask_output=False, **kwargs)
    masks = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), sam_state["original_sizes"],
        sam_state["reshaped_input_sizes"], binarize=False)[0]          # [n_prompts, 1, H, W] logits
    return masks[:, 0].amax(0).float()

def to_paper_size(logits):
    """Logit [H, W] ở kích thước gốc → xác suất 352×352 (bilinear trên logit rồi sigmoid)."""
    resized = F.interpolate(logits[None, None], size=(PAPER_SIZE, PAPER_SIZE), mode="bilinear", align_corners=False)
    return resized[0, 0].sigmoid().numpy()

def variant_probabilities(clip_352, sam_352_box, sam_352_point):
    """Năm biến thể ở 352×352. Không có prompt (sam_* = None) thì giữ nguyên CLIPSeg."""
    box = clip_352 if sam_352_box is None else sam_352_box
    point = clip_352 if sam_352_point is None else sam_352_point
    return {"clipseg": clip_352, "sam_box_replace": box, "sam_box_mean": 0.5 * (clip_352 + box),
            "sam_boxpoint_replace": point, "sam_boxpoint_mean": 0.5 * (clip_352 + point)}

def query_iou(stats):
    tp, fp, fn = stats[0, PAPER_T03], stats[1, PAPER_T03], stats[2, PAPER_T03]
    return float(tp / (1 + tp + fp + fn))

def paired_bootstrap(base, other, image_of, reps=5000, seed=SEED):
    """Δ mIoU@0,3 (other − base) ghép theo query, CI 95% bootstrap theo ảnh."""
    ids = sorted(base)
    delta = np.array([other[q] - base[q] for q in ids])
    images = sorted({image_of[q] for q in ids})
    position = {image: i for i, image in enumerate(images)}
    groups = np.array([position[image_of[q]] for q in ids])
    sums, counts = np.bincount(groups, delta, len(images)), np.bincount(groups, minlength=len(images))
    draws = np.random.default_rng(seed).integers(0, len(images), size=(reps, len(images)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return {"delta_miou_0.3": float(delta.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "wins": int((delta > 1e-9).sum()), "losses": int((delta < -1e-9).sum()), "n_queries": len(ids)}

def run_split():
    """Chấm mọi biến thể trên split hiện tại (EVAL_SPLIT); lưu stats, metric từng query và summary."""
    split_dir = OUT / "splits" / EVAL_SPLIT
    split_dir.mkdir(parents=True, exist_ok=True)
    tasks_by_image = defaultdict(list)
    for task in selected:
        tasks_by_image[int(task["image_id"])].append(task)
    stats = {v: [] for v in VARIANTS}
    rows, fallback, started = [], 0, time.perf_counter()
    for image_id in tqdm(image_ids, desc=f"CLIPSeg+SAM {EVAL_SPLIT}"):
        with Image.open(image_paths[image_id]) as raw:
            image = raw.convert("RGB")
        state = sam_image(image)
        current = tasks_by_image[image_id]
        for start in range(0, len(current), BATCH_SIZE):
            batch = current[start:start + BATCH_SIZE]
            probabilities, probabilities_352, _ = predict_batch(image, batch, with_paper=True)
            for task, probability, clip_352 in zip(batch, probabilities, probabilities_352):
                prompts = prompt_components(probability)
                if prompts:
                    sam_box = to_paper_size(sam_logits(state, prompts, use_points=False))
                    sam_point = to_paper_size(sam_logits(state, prompts, use_points=True))
                else:
                    sam_box = sam_point = None
                    fallback += 1
                target = paper_target(rasterize_gt(task, image.width, image.height))
                row = {"task_id": str(task["task_id"]), "image_id": image_id, "n_prompts": len(prompts)}
                for variant, probability_352 in variant_probabilities(clip_352, sam_box, sam_point).items():
                    query_stats = paper_threshold_stats(probability_352, target)
                    stats[variant].append(query_stats)
                    row[f"iou_0.3_{variant}"] = query_iou(query_stats)
                rows.append(row)
        if len(rows) % 2000 < len(current):
            print(f"{EVAL_SPLIT}: {len(rows)}/{len(selected)} query, {time.perf_counter() - started:.0f}s", flush=True)
    assert len(rows) == len(selected), "Chưa chấm đủ query."
    write_csv(split_dir / "sample_metrics.csv", rows)
    np.savez_compressed(split_dir / "paper_threshold_stats.npz", thresholds=PAPER_THRESHOLDS,
                        task_ids=np.array([r["task_id"] for r in rows]),
                        **{variant: np.stack(values) for variant, values in stats.items()})
    image_of = {r["task_id"]: r["image_id"] for r in rows}
    iou = {v: {r["task_id"]: r[f"iou_0.3_{v}"] for r in rows} for v in VARIANTS}
    result = {"split": EVAL_SPLIT, "n_queries": len(rows), "complete_official_split": len(selected) == len(tasks),
              "fallback_no_prompt": fallback, "seconds": time.perf_counter() - started,
              "paper_protocol": {v: paper_metrics(np.stack(stats[v])) for v in VARIANTS},
              "paired_vs_clipseg": {v: paired_bootstrap(iou["clipseg"], iou[v], image_of) for v in VARIANTS[1:]}}
    write_json(split_dir / "summary.json", result)
    for v in VARIANTS:
        m = result["paper_protocol"][v]
        extra = "" if v == "clipseg" else (f" | Δ {100 * result['paired_vs_clipseg'][v]['delta_miou_0.3']:+.2f} "
            f"[{100 * result['paired_vs_clipseg'][v]['ci95'][0]:+.2f}, {100 * result['paired_vs_clipseg'][v]['ci95'][1]:+.2f}]")
        print(f"{EVAL_SPLIT} {v:22s} mIoU {100 * m['miou_0.3']:.2f} | IoU_FG {100 * m['fgiou_0.3']:.2f} | AP {100 * m['ap']:.2f}{extra}")
    print(f"Query không có vùng p ≥ {THRESHOLD} (giữ CLIPSeg): {fallback}")
    return result
'''

RUN_VAL = '''
val_result = run_split()
# Quy tắc chọn, cố định trước khi chấm test: biến thể SAM có mIoU@0,3 trên val cao nhất.
candidates = VARIANTS[1:]
SELECTED_VARIANT = max(candidates, key=lambda v: (val_result["paper_protocol"][v]["miou_0.3"], -candidates.index(v)))
selection = {"rule": "highest val paper mIoU@0.3 among SAM variants; ties -> earlier in VARIANTS",
             "selected_variant": SELECTED_VARIANT, "val_paired_vs_clipseg": val_result["paired_vs_clipseg"][SELECTED_VARIANT],
             "improves_on_val": val_result["paired_vs_clipseg"][SELECTED_VARIANT]["ci95"][0] > 0}
write_json(OUT / "selection.json", selection)
print("Chốt trên val:", json.dumps(selection, ensure_ascii=False))
'''

SWITCH_TEST = '''
EVAL_SPLIT = "test"
print("Chấm test: mọi biến thể, biến thể chính đã chốt trên val =", SELECTED_VARIANT)
'''

RUN_TEST = '''
test_result = run_split()
clipseg_test = test_result["paper_protocol"]["clipseg"]
reproduces = (all(abs(clipseg_test[k] - v) <= 0.0002 for k, v in CLIPSEG_REFERENCE_TEST.items())
              if MAX_QUERIES is None else None)
final = {"decoder_label": DECODER_LABEL, "decoder_checkpoint_sha256": DECODER_SHA256, "has_lora": has_lora(model),
         "sam_model": SAM_MODEL, "sam_commit": sam_commit,
         "settings": {"threshold": THRESHOLD, "max_components": SAM_MAX_COMPONENTS,
                      "min_component_ratio": SAM_MIN_COMPONENT_RATIO, "fusions": ["replace", "mean"]},
         "selection": selection, "val": val_result, "test": test_result,
         "clipseg_test_reproduces_reference": reproduces, "clipseg_reference_test": CLIPSEG_REFERENCE_TEST,
         "clipseg_test_reproduces_e1": reproduces if DECODER_LABEL == "e1" else None}
write_json(OUT / "summary.json", final)
lines = ["| Biến thể | val mIoU | val Δ (95% CI) | test mIoU | test IoU_FG | test AP | test Δ (95% CI) |",
         "|---|---:|---:|---:|---:|---:|---:|"]
for v in VARIANTS:
    val_m, test_m = val_result["paper_protocol"][v], test_result["paper_protocol"][v]
    def delta(result):
        if v == "clipseg":
            return "—"
        d = result["paired_vs_clipseg"][v]
        return f"{100 * d['delta_miou_0.3']:+.2f} [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}]"
    mark = " (chốt)" if v == SELECTED_VARIANT else ""
    lines.append(f"| {v}{mark} | {100 * val_m['miou_0.3']:.2f} | {delta(val_result)} | {100 * test_m['miou_0.3']:.2f} | "
                 f"{100 * test_m['fgiou_0.3']:.2f} | {100 * test_m['ap']:.2f} | {delta(test_result)} |")
(OUT / "e5_comparison.md").write_text("\\n".join(lines) + "\\n", encoding="utf-8")
print("\\n".join(lines))
print(f"Dòng clipseg trên test trùng số đã biết của {DECODER_LABEL}:", reproduces)
shutil.copy2(OUT / "summary.json", OUT / "splits" / "summary_all.json")
'''

PACKAGE = '''
archive = WORK_ROOT / "clipseg_e5_sam.zip"
archive.unlink(missing_ok=True)
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as package:
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.suffix != ".png":
            package.write(path, arcname=path.relative_to(OUT).as_posix())
print("Download:", archive)
'''


def code(cell_id, source, tag=None):
    return {"cell_type": "code", "id": cell_id, "metadata": {"tags": [tag]} if tag else {},
            "execution_count": None, "outputs": [], "source": source.strip() + "\n"}


def markdown(cell_id, source):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.strip() + "\n"}


def replace(source, old, new):
    assert source.count(old) == 1, (source.count(old), old[:80])
    return source.replace(old, new)


def build():
    notebook = json.loads(E1.read_text(encoding="utf-8"))
    cells_by_id = {c["id"]: c for c in notebook["cells"]}
    source = {k: "".join(c["source"]) if isinstance(c["source"], list) else c["source"] for k, c in cells_by_id.items()}
    config = source["cell-04"]
    for key, value in CONFIG_VALUES.items():
        config, n = re.subn(rf"(?m)^{key} = \S+", f"{key} = {value}", config)
        assert n == 1, key
    config = replace(config, "from pathlib import Path\n", "from pathlib import Path\n" + SAM_CONFIG)
    setup = replace(source["cell-05"], 'OUT = WORK_ROOT / "clipseg_e1"', 'OUT = WORK_ROOT / "clipseg_e5_sam"')
    setup = replace(setup, '(WORK_ROOT / "clipseg_e1.zip")', '(WORK_ROOT / "clipseg_e5_sam.zip")')
    cells = [markdown("cell-00", TITLE)]
    for cid in ["cell-01", "cell-02", "cell-03"]:
        cells.append(copy.deepcopy(cells_by_id[cid]))
    cells.append(code("cell-04", config, "config"))
    cells.append(code("cell-05", setup))
    cells.append(code("e5-locate-decoder", LOCATE))
    for cid in ["cell-06", "cell-07", "cell-08", "cell-09", "cell-10", "cell-11", "cell-13"]:
        cells.append(copy.deepcopy(cells_by_id[cid]))
    cells.append(code("e5-lora-helpers", "LORA_LR_SCALE = 1.0\n" + E4_CODE))
    cells.append(code("cell-15", replace(source["cell-15"], EVAL_LOAD_OLD, EVAL_LOAD_NEW)))
    cells.append(code("e5-lora-check", LORA_CHECK))
    cells += [markdown("e5-sam-md", SAM_SETUP_MD), code("e5-sam", SAM_SETUP),
              markdown("e5-val-md", "## E5. Chấm val và chốt biến thể (trước khi chấm test)"), code("e5-run-val", RUN_VAL),
              markdown("e5-test-md", "## E5. Chấm test bằng biến thể đã chốt (kèm mọi biến thể để minh bạch)"),
              code("e5-switch-test", SWITCH_TEST),
              code("cell-07-test", source["cell-07"]), code("cell-09-test", source["cell-09"]),
              code("e5-run-test", RUN_TEST), code("e5-package", PACKAGE)]
    for cell in cells:
        if cell["cell_type"] == "code":
            compile(cell["source"], cell["id"], "exec")
        cell.setdefault("metadata", {})
    assert len({c["id"] for c in cells}) == len(cells)
    notebook["cells"] = cells
    OUTPUT.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return OUTPUT


if __name__ == "__main__":
    print(build())
