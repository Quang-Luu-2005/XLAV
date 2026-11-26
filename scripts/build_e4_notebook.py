"""Build source/e4_lora_clip.ipynb from the E1 notebook.

E4 keeps every E1 cell (data, PhraseCut+, BCE loss, training loop, paper-protocol evaluation) and changes only:
  * the config: continue from the E1 decoder for EXTRA epochs with a new cosine schedule;
  * ARM "B1": the E1 model unchanged (control: same data, steps, seed and schedule);
  * ARM "E4c": the E1 model plus LoRA on the q/v projections of the CLIP vision layers that feed the decoder;
  * FROM_SCRATCH=True (default): E4c starts from a random decoder with LoRA from epoch 1 and repeats, over
    9 epochs, the exact schedule B1 received counting from scratch (E1 5+2 epochs, then B1's 2 epochs);
    STOP_AFTER_EPOCH splits it into Kaggle sessions that resume from the previous session's checkpoint;
  * one shared forward (`segment`) used for training and inference by both arms.
Edit this builder, not the generated notebook.
"""
import ast
import copy
import json
import textwrap
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
E1 = ROOT / "source" / "e1_train_eval_phrasecut.ipynb"
OUTPUT = ROOT / "source" / "e4_lora_clip.ipynb"

TITLE = """
# XLAV — E4: LoRA cho CLIP vision encoder của CLIPSeg

Các thử nghiệm thay decoder (thêm cross-attention theo từ, tăng `reduce_dim`) không vượt đối chứng,
nên E4 chuyển sang **module encoder**: gắn **LoRA** vào `q_proj`/`v_proj` của các layer CLIP vision
có output đi vào decoder (layer 0–9). Trọng số CLIP gốc vẫn đóng băng; chỉ LoRA (~0,25M tham số) và decoder được học.
Loss (BCE), dữ liệu PhraseCut+ và protocol chấm giữ nguyên như E1 để so được với tác giả.

| `ARM` | Model | Vai trò |
|---|---|---|
| `"B1"` | E1 giữ nguyên kiến trúc | Đối chứng: tách tác động của việc "train thêm" |
| `"E4c"` | E1 + LoRA (rank 8) trên CLIP vision | Cải tiến |

**Mặc định (`FROM_SCRATCH = True`): E4c train lại từ đầu** — decoder ngẫu nhiên (cùng seed với E1) và LoRA có mặt
ngay từ epoch 1 — trong **9 epoch**, đúng lịch mà B1 đã nhận nếu tính từ đầu:

| Chặng | Epoch | Learning rate (cosine) | Như |
|---|---|---|---|
| 1 | 1–5 | 1e-3 → 1e-4 | E1 lượt đầu |
| 2 | 6–7 | 1e-4 → 5e-5, giữ optimizer | E1 train tiếp |
| 3 | 8–9 | optimizer mới, 3e-4 → 1e-5 | B1 |

Vì một phiên Kaggle tối đa 12 giờ, chia thành nhiều phiên bằng `STOP_AFTER_EPOCH` (ví dụ 3 → 6 → hết).
Phiên sau tự tìm checkpoint mới nhất của phiên trước trong Input (`RESUME_CHECKPOINT = "auto"`), kiểm tra
cùng cấu hình rồi chạy tiếp, giữ optimizer, scheduler, AMP scaler và RNG. Chỉ phiên cuối (epoch 9) chấm val/test.

`FROM_SCRATCH = False` (cách nhanh): khởi tạo từ E1 epoch 7 và train thêm `EPOCHS` epoch như B1;
LoRA khởi tạo `B = 0` nên lúc bắt đầu cho **đúng** logits của E1 (notebook kiểm tra).
Embedding điều kiện (text và ảnh support) luôn tính bằng CLIP gốc (tắt LoRA), nên LoRA chỉ tinh chỉnh
đặc trưng dày của ảnh query — visual prompt ở E2 không bị đổi nghĩa.

Cả hai cách đều so E4c với **B1** (cùng tổng 9 epoch). Chọn trên **val**; test chỉ để báo cáo.

**Chạy trên Kaggle**
1. Add Input: dataset PhraseCut (`poudelsandeep/phrasecut`). Phiên 2 trở đi: thêm Output của phiên trước.
   (Chỉ `FROM_SCRATCH = False` mới cần dataset checkpoint E1 epoch 7, SHA256 `b9bc2850…`.)
2. Settings: **Accelerator GPU T4 x2**, **Internet on**. Đặt `STOP_AFTER_EPOCH`, rồi **Save & Run All**.
3. Kết quả ở `/kaggle/working/clipseg_e4_<arm>` và file `clipseg_e4_<arm>.zip` (có checkpoint từng epoch).
"""

E4_CONFIG = '''
ARM = "E4c"                      # "E4c" = LoRA trên CLIP vision (cải tiến) | "B1" = đối chứng train tiếp kiến trúc gốc
INIT_CHECKPOINT = None           # checkpoint E1 (decoder_epoch_007.pt/resume_latest.pt); None = tự tìm trong Kaggle Input
INIT_CHECKPOINT_SHA256 = "b9bc28509a6a6f36495bc4826a511d182bdee8885a819190997da56fd67947df"  # E1 epoch 7; None = không kiểm
LORA_RANK = 8                    # E4c: hạng LoRA trên q_proj/v_proj của các layer CLIP vision mà decoder dùng
LORA_ALPHA = 16                  # E4c: hệ số LoRA, scale = alpha / rank
LORA_GRAD_CHECKPOINT = False     # E4c: True nếu báo thiếu VRAM (CUDA out of memory); chậm hơn ~30%
FROM_SCRATCH = True              # True = train lại từ đầu 9 epoch theo lịch của B1 | False = train tiếp từ E1 epoch 7
STOP_AFTER_EPOCH = None          # FROM_SCRATCH: dừng phiên sau epoch này để nối phiên (vd 3, rồi 6); None = chạy tới epoch 9
LORA_LR_SCALE = 1.0              # LR của LoRA = LORA_LR_SCALE × LR decoder, cùng lịch cosine (vd 0.1, 0.03)
TEST_TIME_BUDGET_HOURS = None    # bỏ chấm test nếu phiên đã chạy quá số giờ này khi chấm xong val (vd 10.0); None = luôn chấm
'''

SCRATCH_CONFIG = '''
import time as _time
SESSION_STARTED = _time.time()   # để canh giới hạn 12 giờ của một phiên Kaggle
STOPPED_EARLY = False
RUN_TEST = True
if FROM_SCRATCH:
    # Lịch B1 tính từ đầu: E1 (epoch 1-5 cosine 1e-3 -> 1e-4; epoch 6-7 cosine 1e-4 -> 5e-5, giữ optimizer)
    # rồi B1 (epoch 8-9: optimizer, AMP scaler, thứ tự dữ liệu mới, cosine 3e-4 -> 1e-5).
    SCRATCH_STAGES = [{"epochs": 5, "lr": 1e-3, "min_lr": 1e-4, "start": "random decoder + LoRA"},
                      {"epochs": 2, "lr": 1e-4, "min_lr": 5e-5, "start": "keep optimizer, new cosine"},
                      {"epochs": 2, "lr": 3e-4, "min_lr": 1e-5, "start": "new optimizer, scaler, loader seed"}]
    TRAIN_INIT = "random_decoder"
    EPOCHS = sum(stage["epochs"] for stage in SCRATCH_STAGES)
    LEARNING_RATE, MIN_LEARNING_RATE = SCRATCH_STAGES[0]["lr"], SCRATCH_STAGES[0]["min_lr"]
    if RESUME_CHECKPOINT is None:
        RESUME_CHECKPOINT = "auto"   # tìm checkpoint của phiên trước trong Kaggle Input
'''

CONFIG_VALUES = {
    "EVAL_SPLIT": '"val"', "BATCH_SIZE": "64", "EPOCHS": "2", "TRAIN_DECODER": "True",
    "TRAIN_INIT": '"e1_checkpoint"', "LEARNING_RATE": "3e-4", "MIN_LEARNING_RATE": "1e-5", "USE_AMP": "True",
    "REQUIRE_GPU": "True", "MULTI_GPU": "True", "EVAL_AMP": "False",
}

MODULE_MD = """
## 5b. LoRA cho CLIP vision và forward dùng chung

`segment` thay cho `model(...)`: tính activation CLIP, conditional embedding (text, hoặc trộn text–ảnh khi train
PhraseCut+), rồi gọi decoder. Với model gốc, `segment` cho đúng logits của `CLIPSegForImageSegmentation.forward`
(kiểm tra ngay bên dưới), nên B1 và E1 dùng cùng một phép tính. Khi có LoRA, activation được tính lại với gradient
(chỉ tới layer cuối decoder dùng), còn embedding điều kiện vẫn tính bằng CLIP gốc.
"""

MODULE = '''
import math
from contextlib import contextmanager
import torch.utils.checkpoint

class LoRALinear(torch.nn.Module):
    """W·x + (alpha/r)·B·A·x; B khởi tạo 0 nên lúc đầu đúng bằng lớp gốc. Lớp gốc giữ đóng băng."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.lora_a = torch.nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = torch.nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        torch.nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, x):
        out = self.base(x)
        if self.enabled:
            out = out + (x @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype) * self.scale
        return out

def lora_layers(model):
    """Các layer CLIP vision có output đi vào decoder (0..max(extract_layers))."""
    return list(model.clip.vision_model.encoder.layers[:max(model.extract_layers) + 1])

def lora_modules(model):
    return [m for m in model.clip.vision_model.modules() if isinstance(m, LoRALinear)]

def has_lora(model):
    return bool(lora_modules(model))

def attach_lora(model, rank, alpha):
    assert not has_lora(model), "LoRA đã được gắn"
    for layer in lora_layers(model):
        for name in ["q_proj", "v_proj"]:
            setattr(layer.self_attn, name, LoRALinear(getattr(layer.self_attn, name), rank, alpha))

def lora_parameters(model):
    return [p for m in lora_modules(model) for p in (m.lora_a, m.lora_b)]

def trainable_groups(model, lr):
    """Nhóm 0 = decoder (lr), nhóm 1 = LoRA (lr × LORA_LR_SCALE). Không có LoRA thì chỉ một nhóm, như B1."""
    groups = [{"params": list(model.decoder.parameters()), "lr": lr}]
    if lora_parameters(model):
        groups.append({"params": lora_parameters(model), "lr": lr * LORA_LR_SCALE})
    return groups

def sync_lora_lr(optimizer):
    """Giữ LR của LoRA đúng bằng LORA_LR_SCALE × LR decoder sau mỗi bước của lịch cosine."""
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]["lr"] = optimizer.param_groups[0]["lr"] * LORA_LR_SCALE

def lora_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.clip.vision_model.state_dict().items() if "lora_" in k}

def load_lora_state(model, state):
    missing, unexpected = model.clip.vision_model.load_state_dict(state, strict=False)
    assert not unexpected and not [k for k in missing if "lora_" in k], (missing, unexpected)

@contextmanager
def lora_disabled(model):
    """Embedding điều kiện (text, ảnh support) luôn tính bằng CLIP gốc, kể cả ở E4c."""
    modules = lora_modules(model)
    for m in modules:
        m.enabled = False
    try:
        yield
    finally:
        for m in modules:
            m.enabled = True

def vision_activations(model, pixel_values):
    """Output các layer extract_layers của CLIP vision, chỉ chạy tới layer cuối cần dùng."""
    vision = model.clip.vision_model
    hidden = vision.pre_layrnorm(vision.embeddings(pixel_values, interpolate_pos_encoding=True))
    activations = []
    for index, layer in enumerate(lora_layers(model)):
        if LORA_GRAD_CHECKPOINT and torch.is_grad_enabled():
            hidden = torch.utils.checkpoint.checkpoint(layer, hidden, None, None, use_reentrant=False)[0]
        else:
            hidden = layer(hidden, None, None)[0]
        if index in model.extract_layers:
            activations.append(hidden)
    return activations

def mix_conditionals(text_cond, visual_cond, text_weights):
    """cond = w·text + (1−w)·ảnh (giống cell PhraseCut+; định nghĩa sớm để segment dùng được khi chỉ đánh giá)."""
    weights = text_weights.to(text_cond.device, text_cond.dtype)
    return text_cond * weights + visual_cond * (1 - weights)

def segment(model, pixel_values, input_ids, attention_mask, support_pixel_values=None, text_weights=None):
    """Logits [B,352,352] cho train lẫn inference, model gốc (B1) hoặc CLIP vision có LoRA (E4c)."""
    size = pixel_values.shape[0]
    lora = has_lora(model)
    with torch.no_grad():
        if not lora:
            hidden = model.clip.vision_model(pixel_values=pixel_values, output_hidden_states=True,
                                             interpolate_pos_encoding=True).hidden_states
            activations = [hidden[i + 1] for i in model.extract_layers]
        with lora_disabled(model):
            cond = model.get_conditional_embeddings(batch_size=size, input_ids=input_ids, attention_mask=attention_mask)
            if support_pixel_values is not None:
                visual = model.get_conditional_embeddings(batch_size=size, conditional_pixel_values=support_pixel_values)
                cond = mix_conditionals(cond, visual, text_weights)
    if lora:
        activations = vision_activations(model, pixel_values)  # gradient chỉ tới tham số LoRA
    logits = model.decoder(activations, cond).logits
    return logits.reshape(size, *logits.shape[-2:])

def experiment_logits(model, pixel_values, input_ids=None, attention_mask=None, conditional_pixel_values=None):
    """E2/E3 forward: text hoặc ảnh support; với LoRA, embedding điều kiện vẫn dùng CLIP gốc."""
    if not has_lora(model):
        return model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask,
                     conditional_pixel_values=conditional_pixel_values).logits
    if (input_ids is None) == (conditional_pixel_values is None):
        raise ValueError("Supply text or support images, not both")
    if input_ids is not None:
        return segment(model, pixel_values, input_ids, attention_mask)
    with torch.no_grad():
        activations = vision_activations(model, pixel_values)
        with lora_disabled(model):
            cond = model.get_conditional_embeddings(batch_size=pixel_values.shape[0],
                                                    conditional_pixel_values=conditional_pixel_values)
    return model.decoder(activations, cond).logits

# segment với model gốc phải trùng forward của Transformers.
with torch.no_grad():
    check = processor.tokenizer(["a red car", "dog"], padding=True, return_tensors="pt").to(device)
    check_pixels = torch.randn(2, 3, PAPER_SIZE, PAPER_SIZE, device=device)
    reference = model(pixel_values=check_pixels, **check).logits
    assert torch.allclose(segment(model, check_pixels, check["input_ids"], check["attention_mask"]),
                          reference, atol=1e-4), "segment lệch forward của Transformers"
    hidden = model.clip.vision_model(pixel_values=check_pixels, output_hidden_states=True).hidden_states
    for ours, theirs in zip(vision_activations(model, check_pixels), [hidden[i + 1] for i in model.extract_layers]):
        assert torch.allclose(ours, theirs, atol=1e-4), "vision_activations lệch hidden_states của Transformers"
print("segment khớp CLIPSegForImageSegmentation.forward")
'''

INIT_BLOCK = '''    if TRAIN_INIT == "e1_checkpoint":
        if INIT_CHECKPOINT is None:
            found = sorted(p for name in ["decoder_epoch_007.pt", "resume_latest.pt"]
                           for p in Path("/kaggle/input").rglob(name)) if Path("/kaggle/input").exists() else []
            found = [p for p in found if INIT_CHECKPOINT_SHA256 is None or sha256(p) == INIT_CHECKPOINT_SHA256]
            if not found:
                raise FileNotFoundError("Không tìm thấy checkpoint E1 epoch 7 (khớp SHA256) trong Kaggle Input.")
            INIT_CHECKPOINT = str(found[0])  # cùng SHA256 nên bản nào cũng như nhau
        init_sha256 = sha256(Path(INIT_CHECKPOINT))
        if INIT_CHECKPOINT_SHA256 and init_sha256 != INIT_CHECKPOINT_SHA256:
            raise ValueError("Checkpoint E1 khác SHA256 đã chốt.")
        init_state = torch.load(INIT_CHECKPOINT, map_location="cpu", weights_only=True)
        if init_state.get("base_model_id") != MODEL_ID or init_state.get("base_model_revision") != MODEL_REVISION:
            raise ValueError("Checkpoint E1 được train trên base model/revision khác.")
        model.decoder.load_state_dict(init_state["decoder"])
        if ARM == "E4c":
            with torch.no_grad(), torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
                probe = processor.tokenizer(["the man on the left", "red car"], padding=True, return_tensors="pt").to(device)
                probe_pixels = torch.randn(2, 3, PAPER_SIZE, PAPER_SIZE, device=device)
                before = segment(model, probe_pixels, probe["input_ids"], probe["attention_mask"])
            # Khởi tạo LoRA không được làm đổi RNG lấy mẫu dữ liệu so với B1.
            with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
                attach_lora(model, LORA_RANK, LORA_ALPHA)
            with torch.no_grad():
                after = segment(model, probe_pixels, probe["input_ids"], probe["attention_mask"])
            assert torch.allclose(before, after, atol=1e-4), "LoRA B = 0 nhưng E4c khác E1 lúc khởi tạo"
            print("E4c lúc khởi tạo cho đúng logits của E1 (LoRA B = 0)")
        elif ARM != "B1":
            raise ValueError(f"ARM không hợp lệ: {ARM}")
        new_parameters = sum(p.numel() for p in lora_parameters(model))
        training_metadata.update(arm=ARM, init_checkpoint_sha256=init_sha256, init_epoch=init_state["epoch"],
            init_global_step=init_state["global_step"], decoder_arch="clipseg",
            lora={"rank": LORA_RANK, "alpha": LORA_ALPHA, "targets": ["q_proj", "v_proj"],
                  "layers": f"CLIP vision 0-{max(model.extract_layers)}", "init": "B = 0",
                  "conditioning": "text and support embeddings use frozen CLIP (LoRA disabled)",
                  "grad_checkpoint": LORA_GRAD_CHECKPOINT, "new_parameters": new_parameters,
                  # chỉ ghi khi khác 1.0 để chữ ký cấu hình của các lượt cũ không đổi
                  **({"lr_scale": LORA_LR_SCALE} if LORA_LR_SCALE != 1.0 else {})} if ARM == "E4c" else None)
        write_json(OUT / "training_config.json", training_metadata)
        print(f"ARM={ARM}: từ E1 epoch {init_state['epoch']} (step {init_state['global_step']}), "
              f"+{EPOCHS} epoch, tham số mới {new_parameters:,}")
        del init_state
'''

SCRATCH_INIT = '''    if FROM_SCRATCH:
        # Decoder vừa khởi tạo ngẫu nhiên như E1; gắn LoRA (B = 0) mà không đổi RNG lấy mẫu dữ liệu.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            attach_lora(model, LORA_RANK, LORA_ALPHA)
        new_parameters = sum(p.numel() for p in lora_parameters(model))
        training_metadata.update(arm=ARM, from_scratch=True, decoder_arch="clipseg", schedule=SCRATCH_STAGES,
            lora={"rank": LORA_RANK, "alpha": LORA_ALPHA, "targets": ["q_proj", "v_proj"],
                  "layers": f"CLIP vision 0-{max(model.extract_layers)}", "init": "B = 0",
                  "conditioning": "text and support embeddings use frozen CLIP (LoRA disabled)",
                  "grad_checkpoint": LORA_GRAD_CHECKPOINT, "new_parameters": new_parameters,
                  **({"lr_scale": LORA_LR_SCALE} if LORA_LR_SCALE != 1.0 else {})})
        write_json(OUT / "training_config.json", training_metadata)
        print(f"ARM={ARM} từ đầu: decoder ngẫu nhiên + LoRA ({new_parameters:,} tham số), {EPOCHS} epoch, "
              f"lịch {[(s['epochs'], s['lr'], s['min_lr']) for s in SCRATCH_STAGES]}")
'''

SCRATCH_RESUME = '''    if RESUME_CHECKPOINT == "auto":
        # Checkpoint mới nhất của các phiên trước có cùng chữ ký cấu hình (cùng ARM, lịch, dữ liệu, seed).
        found = []
        for path in (Path("/kaggle/input").rglob("decoder_epoch_*.pt") if Path("/kaggle/input").exists() else []):
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("config_signature") == signature:
                found.append((int(state["epoch"]), str(path)))
            del state
        RESUME_CHECKPOINT = max(found)[1] if found else None
        print("Tiếp tục từ:", RESUME_CHECKPOINT or "không có — bắt đầu epoch 1")
'''

SCRATCH_LOOP = '''    last_epoch = EPOCHS if STOP_AFTER_EPOCH is None else min(EPOCHS, STOP_AFTER_EPOCH)
    STOPPED_EARLY = last_epoch < EPOCHS
    boundaries = [sum(s["epochs"] for s in SCRATCH_STAGES[:i]) for i in range(1, len(SCRATCH_STAGES))] if FROM_SCRATCH else []
    for epoch in range(start_epoch, last_epoch):
        if epoch in boundaries:
            stage = SCRATCH_STAGES[boundaries.index(epoch) + 1]
            stage_steps = stage["epochs"] * steps_per_epoch
            if stage["start"].startswith("new optimizer"):
                # Như lúc bắt đầu B1: optimizer, AMP scaler mới và thứ tự dữ liệu từ SEED.
                optimizer = torch.optim.AdamW(trainable_groups(model, stage["lr"]), weight_decay=WEIGHT_DECAY)
                scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
                generator.manual_seed(SEED)
            else:
                # Như lượt train tiếp E1 epoch 6-7: giữ trạng thái AdamW, đặt lại lr và lịch cosine.
                for index, group in enumerate(optimizer.param_groups):
                    group["lr"] = group["initial_lr"] = stage["lr"] * (LORA_LR_SCALE if index else 1.0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=stage_steps, eta_min=stage["min_lr"])
            sync_lora_lr(optimizer)
            print(f"Epoch {epoch + 1}: chặng mới {stage}", flush=True)
'''

TRAINABLE_OLD = '''    model.decoder.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)'''

TRAINABLE_NEW = '''    model.decoder.requires_grad_(True)
    for parameter in lora_parameters(model):  # E4c: chỉ LoRA trong CLIP được học, trọng số CLIP gốc giữ đóng băng
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(trainable_groups(model, LEARNING_RATE), weight_decay=WEIGHT_DECAY)'''

TRAIN_FORWARD_OLD = '''            with context:
                cond = self.model.get_conditional_embeddings(batch_size=size, input_ids=input_ids,
                                                             attention_mask=attention_mask)
                if support_pixel_values is not None:
                    visual_cond = self.model.get_conditional_embeddings(
                        batch_size=size, conditional_pixel_values=support_pixel_values)
                    cond = mix_conditionals(cond, visual_cond, text_weights)
                logits = self.model(pixel_values=pixel_values, conditional_embeddings=cond).logits
            # HF bỏ chiều batch khi batch = 1; giữ [B,H,W] để DataParallel gộp đúng.
            return logits.reshape(size, *logits.shape[-2:])'''

TRAIN_FORWARD_NEW = '''            with context:
                # segment giữ [B,H,W] để DataParallel gộp đúng; E4c tính activation CLIP có LoRA với gradient.
                return segment(self.model, pixel_values, input_ids, attention_mask, support_pixel_values, text_weights)'''

EVAL_LOAD_OLD = '''    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()'''

EVAL_LOAD_NEW = '''    if checkpoint.get("lora"):
        adapter = (checkpoint.get("training") or {}).get("lora") or {}
        attach_lora(model, adapter.get("rank", LORA_RANK), adapter.get("alpha", LORA_ALPHA))
        load_lora_state(model, checkpoint["lora"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    model.eval()'''


SWITCH_TEST = '''
elapsed_hours = (_time.time() - SESSION_STARTED) / 3600
RUN_TEST = TEST_TIME_BUDGET_HOURS is None or elapsed_hours <= TEST_TIME_BUDGET_HOURS
write_json(OUT / "session_time.json", {"elapsed_hours_after_val": elapsed_hours,
                                         "test_time_budget_hours": TEST_TIME_BUDGET_HOURS, "run_test": RUN_TEST})
if RUN_TEST:
    EVAL_SPLIT = "test"
    print(f"Chấm test bằng checkpoint epoch cuối (đã chạy {elapsed_hours:.2f} giờ)")
else:
    print(f"BỎ CHẤM TEST: đã chạy {elapsed_hours:.2f} giờ > {TEST_TIME_BUDGET_HOURS} giờ. "
          "Kết quả val và checkpoint vẫn được lưu; chấm test sau bằng chế độ chỉ đánh giá.")
'''

TEST_CELLS = {"cell-07-test", "cell-09-test", "cell-17-test", "cell-19-test", "e4-save-test"}

SAVE_SPLIT_MD = """
## 9b. Lưu kết quả theo split, rồi chấm test

Kết quả của split vừa chấm được chép vào `splits/<split>/` (summary, metric từng query, stats theo protocol paper)
để `scripts/compare_e4.py` so ghép cặp với B1. Sau val, notebook chấm tiếp **test** bằng cùng checkpoint epoch cuối:
test chỉ để báo cáo, không dùng để chọn gì.
"""

SAVE_SPLIT = '''
split_dir = OUT / "splits" / EVAL_SPLIT
split_dir.mkdir(parents=True, exist_ok=True)
for name in ["summary.json", "run.json", "sample_metrics.csv", "calibration_thresholds.csv",
             "paper_threshold_stats.npz", "selected_tasks.json", "image_errors.json"]:
    if (OUT / name).is_file():
        shutil.copy2(OUT / name, split_dir / name)
if MAX_QUERIES is None:
    assert summary["complete_official_split"], "Chưa chấm đủ split chính thức."
paper = summary["paper_protocol"]
print(f"Đã lưu {split_dir}: {paper['n_queries']} query, mIoU {100 * paper['miou_0.3']:.2f} | "
      f"IoU_FG {100 * paper['fgiou_0.3']:.2f} | AP {100 * paper['ap']:.2f}")
'''


def replace(source, old, new, count=1):
    assert source.count(old) == count, (source.count(old), old[:80])
    return source.replace(old, new)


def code(cell_id, source, tag=None):
    return {"cell_type": "code", "id": cell_id, "metadata": {"tags": [tag]} if tag else {},
            "execution_count": None, "outputs": [], "source": source.strip() + "\n"}


EVALUATION_CELLS = {"cell-17", "cell-19", "e4-save-val", "e4-switch-test", "cell-07-test", "cell-09-test",
                    "cell-17-test", "cell-19-test", "e4-save-test"}


def skip_when_stopped(source, cell_id):
    """Bọc cell chấm điểm trong `if not STOPPED_EARLY:` (cell test thêm `and RUN_TEST`); kiểm tra AST không đổi."""
    condition = "not STOPPED_EARLY and RUN_TEST" if cell_id in TEST_CELLS else "not STOPPED_EARLY"
    wrapped = f"if {condition}:\n" + textwrap.indent(source, "    ", lambda line: True)
    inner = ast.parse(wrapped).body[0].body
    assert ast.dump(ast.Module(body=inner, type_ignores=[])) == ast.dump(ast.parse(source)), cell_id
    return wrapped


def build():
    notebook = json.loads(E1.read_text(encoding="utf-8"))
    cells = []
    originals = {c["id"]: "".join(c["source"]) if isinstance(c["source"], list) else c["source"]
                 for c in notebook["cells"]}
    for original in notebook["cells"]:
        cell = copy.deepcopy(original)
        source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        cid = cell["id"]
        if cid == "cell-00":
            source = TITLE.strip() + "\n"
        elif cid == "cell-04":
            for key, value in CONFIG_VALUES.items():
                source, n = re.subn(rf"(?m)^{key} = \S+", f"{key} = {value}", source)
                assert n == 1, key
            source = replace(source, "from pathlib import Path\n", "from pathlib import Path\n" + E4_CONFIG)
            source = source.rstrip("\n") + "\n" + SCRATCH_CONFIG
        elif cid == "cell-05":
            source = replace(source, 'assert TRAIN_INIT in ["random_decoder", "pretrained_decoder"]',
                             'assert TRAIN_INIT in ["random_decoder", "pretrained_decoder", "e1_checkpoint"]\n'
                             'assert ARM in ["B1", "E4c"] and LORA_RANK >= 1\n'
                             'assert not FROM_SCRATCH or ARM == "E4c", "B1 từ đầu chính là E1 + B1, đã có"\n'
                             'assert STOP_AFTER_EPOCH is None or (FROM_SCRATCH and 1 <= STOP_AFTER_EPOCH <= EPOCHS)')
            source = replace(source, 'OUT = WORK_ROOT / "clipseg_e1"', 'OUT = WORK_ROOT / f"clipseg_e4_{ARM.lower()}"')
            source = replace(source, '(WORK_ROOT / "clipseg_e1.zip")', '(WORK_ROOT / f"clipseg_e4_{ARM.lower()}.zip")')
        elif cid == "cell-11":
            source = replace(source, "        logits = model(**inputs).logits",
                             '        logits = segment(model, inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])')
        elif cid == "cell-15":
            source = replace(source, '    if TRAIN_INIT == "random_decoder":', INIT_BLOCK + '    elif TRAIN_INIT == "random_decoder":')
            source = replace(source, TRAIN_FORWARD_OLD, TRAIN_FORWARD_NEW)
            source = replace(source, EVAL_LOAD_OLD, EVAL_LOAD_NEW)
            source = replace(source, TRAINABLE_OLD, TRAINABLE_NEW)
            source = replace(source, "    model.clip.requires_grad_(False)\n", SCRATCH_INIT + "    model.clip.requires_grad_(False)\n")
            source = replace(source, "    assert all(p.grad is None for p in model.clip.parameters())",
                             "    assert all(p.grad is None for p in model.clip.parameters())\n"
                             "    print(\"LR cuối:\", [round(g[\"lr\"], 10) for g in optimizer.param_groups], flush=True)")
            source = replace(source, "                scheduler.step()\n",
                             "                scheduler.step()\n                sync_lora_lr(optimizer)\n")
            source = replace(source, "    scaler = torch.amp.GradScaler(\"cuda\", enabled=USE_AMP)\n",
                             "    sync_lora_lr(optimizer)\n    scaler = torch.amp.GradScaler(\"cuda\", enabled=USE_AMP)\n")
            source = replace(source, "T_max=total_steps,",
                             'T_max=SCRATCH_STAGES[0]["epochs"] * steps_per_epoch if FROM_SCRATCH else total_steps,')
            source = replace(source, "    if not RESUME_CHECKPOINT:\n", SCRATCH_RESUME + "    if not RESUME_CHECKPOINT:\n")
            source = replace(source, "    for epoch in range(start_epoch, EPOCHS):\n", SCRATCH_LOOP)
            source = replace(source, '"completed_epochs": EPOCHS', '"completed_epochs": last_epoch')
            source = replace(source, "start_epoch == EPOCHS", "start_epoch >= last_epoch")
            source = replace(source, '    print(f"Training complete',
                             '    if STOPPED_EARLY:\n'
                             '        print(f"Dừng sau epoch {last_epoch}/{EPOCHS}. Thêm Output phiên này làm Input rồi chạy phiên tiếp; "\n'
                             '              "chưa chấm val/test.")\n'
                             '    print(f"Training complete')
            source = replace(source, '        model.decoder.load_state_dict(checkpoint["decoder"])\n        optimizer.load_state_dict',
                             '        model.decoder.load_state_dict(checkpoint["decoder"])\n'
                             '        if checkpoint.get("lora"):\n            load_lora_state(model, checkpoint["lora"])\n'
                             '        optimizer.load_state_dict')
            source = replace(source, '"decoder": model.decoder.state_dict(), "optimizer"',
                             '"decoder": model.decoder.state_dict(), "lora": lora_state_dict(model), "optimizer"')
            source = replace(source, 'support_pixel_values=None, text_weights=None):',
                             'support_pixel_values=None, text_weights=None, force_fp32=False):')
            source = replace(source, 'context = torch.autocast(device_type="cuda", dtype=torch.float16) if USE_AMP else nullcontext()',
                             'context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=USE_AMP and not force_fp32)')
            source = replace(source, '''            logits = train_forward(**inputs)
            loss = F.binary_cross_entropy_with_logits(logits.float(), batch["labels"].to(logits.device, non_blocking=True))
            if not torch.isfinite(loss):
                raise ValueError("Training loss NaN/Inf; dừng để kiểm tra dữ liệu/AMP.")
            old_scale = scaler.get_scale()
            scaler.scale(loss).backward()''', '''            old_scale = scaler.get_scale()
            loss = loss_backward_with_recovery(train_forward, model, inputs, batch["labels"], scaler,
                OUT, epoch + 1, seen // BATCH_SIZE, global_step)''')
            source = replace(source, '"decoder": model.decoder.state_dict(), "lora": lora_state_dict(model), "optimizer"',
                             '"numerical_policy": NUMERICAL_POLICY, '
                             '"numerical_recovery": json.loads((OUT / "numerical_recovery.json").read_text()) if (OUT / "numerical_recovery.json").exists() else None, '
                             '"decoder": model.decoder.state_dict(), "lora": lora_state_dict(model), "optimizer"')
            source = replace(source, '        scaler.load_state_dict(checkpoint["scaler"])',
                             '        scaler.load_state_dict(checkpoint["scaler"])\n'
                             '        if checkpoint.get("numerical_recovery"):\n'
                             '            save_numerical_report(OUT, checkpoint["numerical_recovery"])')
            source = replace(source, "    assert all(p.grad is None for p in model.clip.parameters())",
                             '    assert all(p.grad is None for n, p in model.clip.named_parameters() if "lora_" not in n)')
        elif cid == "cell-21":
            source = replace(source, 'archive = WORK_ROOT / "clipseg_e1.zip"', 'archive = WORK_ROOT / f"clipseg_e4_{ARM.lower()}.zip"')
        cell["source"] = source
        cells.append(cell)
        if cid == "cell-11":
            cells.append({"cell_type": "markdown", "id": "e4-module-md", "metadata": {}, "source": MODULE_MD.strip() + "\n"})
            cells.append(code("e4-module", MODULE))
            cells.append(code("e4-amp-recovery", (Path(__file__).parent / "e4_amp_recovery.py").read_text(encoding="utf-8") +
                              '\nwrite_json(OUT / "numerical_policy.json", {"policy": NUMERICAL_POLICY, '
                              '"effective_batch_size": BATCH_SIZE, "fp32_retry_microbatch_size": 16, '
                              '"on_nonfinite_fp32": "stop_before_optimizer_update", "samples_skipped": 0})\n'
                              'print("Numerical policy:", NUMERICAL_POLICY, flush=True)\n'))
        if cid == "cell-19":
            # Chấm val (EVAL_SPLIT) rồi test bằng cùng checkpoint epoch cuối, mỗi split lưu riêng.
            cells.append({"cell_type": "markdown", "id": "e4-save-val-md", "metadata": {},
                          "source": SAVE_SPLIT_MD.strip() + "\n"})
            cells.append(code("e4-save-val", SAVE_SPLIT))
            cells.append(code("e4-switch-test", SWITCH_TEST))
            for source_id in ["cell-07", "cell-09", "cell-17", "cell-19"]:
                cells.append(code(f"{source_id}-test", originals[source_id]))
            cells.append(code("e4-save-test", SAVE_SPLIT))
    for cell in cells:
        if cell["cell_type"] == "code" and cell["id"] in EVALUATION_CELLS:
            cell["source"] = skip_when_stopped(cell["source"], cell["id"])
    for cell in cells:
        if cell["cell_type"] == "code":
            compile(cell["source"], cell["id"], "exec")
    notebook["cells"] = cells
    OUTPUT.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
    return OUTPUT


if __name__ == "__main__":
    print(build())
