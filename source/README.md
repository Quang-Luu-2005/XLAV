# Baseline CLIPSeg — PhraseCut miniv trên Kaggle GPU

File chạy chính: [e1_train_eval_phrasecut.ipynb](e1_train_eval_phrasecut.ipynb). Notebook tự chứa code; upload **một file `.ipynb`** là đủ, không cần clone repo XLAV hoặc submodule CLIPSeg. **E1** tái tạo quá trình train CLIPSeg: `TRAIN_DECODER=True` train decoder theo **epoch + batch size** trên **PhraseCut+**, tạo ngẫu nhiên từ PhraseCut `train` (text + ảnh support, mẫu âm, augmentation) như tác giả, rồi đánh giá trên `miniv`. Mặc định `TRAIN_DECODER=False` chỉ đánh giá checkpoint pretrained của tác giả, dùng làm mốc tham chiếu.

## Chạy trên Kaggle

1. Tạo notebook Kaggle, dùng **Import Notebook** để upload file trên.
2. Trong Settings, chọn Accelerator **GPU T4 x2** (tránh P100: PyTorch mới của Kaggle có thể không hỗ trợ) và bật Internet. Baseline dùng GPU đầu tiên.
3. Đặt `EPOCHS`, `BATCH_SIZE`, và `TRAIN_ROOT` nếu đã Add Input tập train. Chạy **Run All**. Mặc định tải annotation từ nguồn PhraseCut chính thức, ảnh theo URL trong metadata và base checkpoint từ Hugging Face.
4. Tải `clipseg_e1.zip` trong Output. Các file gốc nằm tại `/kaggle/working/clipseg_e1/`.

Ví dụ cấu hình huấn luyện:

```python
TRAIN_DECODER = True
EPOCHS = 5
BATCH_SIZE = 8
TRAIN_INIT = "random_decoder"
MAX_TRAIN_QUERIES = None
MAX_QUERIES = None
```

`steps_per_epoch = ceil(N_train / BATCH_SIZE)`; `total_steps = EPOCHS * steps_per_epoch`. Mỗi epoch duyệt hết các train query đã chọn; không bỏ batch cuối. Không còn tham số `max_iterations`. Scheduler cosine theo số bước thực tế được tính từ epoch/batch, không giữ mốc 20.000 steps của tác giả.

Khi bật `TRAIN_DECODER=True`, notebook đặt `EPOCHS=1`, `BATCH_SIZE=8`, huấn luyện decoder ngẫu nhiên với CLIP pretrained đóng băng. Chọn `TRAIN_INIT="pretrained_decoder"` để fine-tune decoder sẵn có; mặc định `TRAIN_DECODER=False` chỉ chạy baseline pretrained. Đây là các thiết lập khác nhau, cần ghi rõ khi báo số liệu.

### PhraseCut+

Tác giả không công bố PhraseCut+ thành file. Nó được tạo **ngẫu nhiên mỗi lần lấy mẫu** từ PhraseCut theo `PhraseCutPlus` ([datasets/phrasecut.py](../third_party/clipseg/datasets/phrasecut.py)) và vòng train ([training.py](../third_party/clipseg/training.py)), cấu hình ở [experiments/phrasecut.yaml](../third_party/clipseg/experiments/phrasecut.yaml). Notebook tái tạo các bước đó:

| Bước | Tham số | Theo bản gốc |
|---|---|---|
| Crop vuông ngẫu nhiên giữ vật, resize 352 | `AUG_CROP=True` | `find_crop`, `aug_crop` |
| Mẫu âm: đổi cụm từ, mask rỗng | `NEGATIVE_PROB=0.2` | `negative_prob: 0.2` |
| Support: mẫu ở **ảnh khác** cùng cụm từ, làm mờ nền, nền còn 10%, crop nới 10%, pad vuông | `TRAIN_MODE="phrasecut_plus"` | `with_visual`, `crop_blur_highlight352` |
| Template ngẫu nhiên trong 9 mẫu | `PROMPT_AUGMENT=True` | `prompt: shuffle+` |
| `cond = w·text + (1−w)·ảnh`, `w ~ U(0; 0,5)`, không có support thì `w = 1` | `MIX_TEXT_MAX=0.5` | `mix: True`, `mix_text_max: 0.5` |

`TRAIN_MODE="text_only"` bỏ support và trộn embedding, tương ứng ablation `rd64-uni-novis` của tác giả. Hai điểm khác bản gốc: support luôn lấy từ ảnh khác, và GT rasterize bằng PIL như phần đánh giá. Notebook lưu `phrasecut_plus_samples.png` (4 mẫu: query, support, mask) để kiểm tra trước khi train.

Tác giả train batch 64, AdamW lr 1e-3 → 1e-4 cosine, AMP; `rd64-uni` 20.000 bước, `rd64-uni-refined` (checkpoint HF) 50.000 bước. Với 310.816 query train, 20.000 × 64 tương đương khoảng 4 epoch (`EPOCHS=4`, `BATCH_SIZE=64`). Trên Kaggle nên bật `USE_AMP=True`, đặt `NUM_WORKERS=4` khi ảnh train đã có trong Input, và chia lượt train qua nhiều session bằng `RESUME_CHECKPOINT`.

Cell cấu hình có `MAX_QUERIES=None` để chạy toàn bộ 821 query / 100 ảnh của miniv. Đặt `8` để kiểm tra nhanh, hoặc `200` để chạy E1 gọn theo kế hoạch. `MAX_TRAIN_QUERIES=8` chạy thử train trên 8 query **từ split train**, không lấy miniv để train. Danh sách query được lưu, sampling và chia ảnh dùng seed 42. Mặc định FP32, batch 8; nếu thiếu VRAM, giảm batch xuống 4 hoặc 1. Có thể bật `USE_AMP=True` cho FP16, nhưng phải ghi cấu hình này khi so sánh kết quả.

Cell cài đặt pin `transformers==4.55.4` và `gdown==5.2.0`. PyTorch/CUDA và các thư viện phân tích dùng bản sẵn có của Kaggle, được ghi vào `run.json`; không cài `requirements.txt` dành cho Windows lên Kaggle. Chạy notebook trong session mới để tránh import nhầm Transformers đã được nạp trước khi cài.

## Đánh giá toàn bộ PhraseCut và train trên 2 GPU

**Split đánh giá.** `EVAL_SPLIT` chọn split của PhraseCut. Mặc định `MAX_QUERIES=None` chạy toàn bộ query của split đó.

| `EVAL_SPLIT` | Ảnh / query | Dùng khi |
|---|---|---|
| `"miniv"` | 100 / 821 | Kiểm tra nhanh |
| `"val"` | ~2.900 ảnh | Chọn cấu hình, ngưỡng |
| `"test"` | 2.545 / 14.354 | **Báo số cuối**, cùng split với bảng paper |

Notebook tự tìm `refer_<split>.json` trong Input. Nếu không có, nó tải từ Google Drive chính thức của PhraseCut. Ảnh tải theo URL Visual Genome. Với `test`/`val`, nên dùng dataset có sẵn ảnh để khỏi tải lại khoảng 2.500 ảnh mỗi lượt. Có thể train và đánh giá trên `test` trong cùng một lượt: split train không trùng ảnh với split đánh giá, và notebook kiểm tra điều này.

**Hai GPU.** Với `MULTI_GPU=True` (mặc định), phần train bọc model bằng `torch.nn.DataParallel`. Mỗi batch `BATCH_SIZE` được chia đều cho mọi GPU, ví dụ batch 64 chia 32 + 32 trên T4 x2, nên ý nghĩa `EPOCHS`/`BATCH_SIZE` không đổi. Số GPU và VRAM đỉnh của từng GPU ghi trong `training_history.csv` (cột `gpus`, `peak_gpu_mb`); cả hai GPU phải có VRAM > 0. Phần đánh giá vẫn chạy trên một GPU, vì nhẹ và bị giới hạn bởi tính metric trên CPU.

**CPU.** Mỗi mẫu PhraseCut+ đọc 2 ảnh, rasterize polygon và blur trên CPU. Vì vậy `NUM_WORKERS=4` là mặc định, bằng số CPU của Kaggle. Nếu GPU vẫn rảnh (xem `nvidia-smi` hoặc thời gian mỗi epoch), nút cổ chai là CPU chứ không phải GPU. Chạy trên Windows thì đặt `NUM_WORKERS=0`.

Từ bản này, thư mục kết quả đổi thành `/kaggle/working/clipseg_e1`, và `summary.json` dùng khóa `split`, `complete_official_split`, `selected_fixed_threshold` thay cho các khóa có chữ `miniv`.

## Dùng dữ liệu đã Add Input

Cấu trúc hỗ trợ:

```text
VGPhraseCut_v0/
├── refer_miniv.json
├── image_data_split.json          # hoặc image_data_split3000.json
└── images/
    ├── 4744.jpg
    └── ...
```

Đặt `DATA_ROOT="/kaggle/input/<dataset>/VGPhraseCut_v0"`, hoặc để `None` cho notebook tự tìm. Nếu có nhiều dataset miniv thì cần chọn đường dẫn cụ thể. Notebook kiểm tra split, task ID, polygon và kích thước ảnh; lỗi dữ liệu làm dừng lượt chạy và ghi `image_errors.json`, không âm thầm bỏ query.

Tập train cần `refer_train.json` và ảnh có tên `<image_id>.jpg`; đặt `TRAIN_ROOT` đến thư mục này. Dùng metadata của PhraseCut để xác nhận từng image_id thuộc `train` và không trùng miniv. Train full cần annotation khoảng 344 MB và nhiều GB ảnh; nên Add Input dataset đã cache. Nếu chưa có input, notebook tải annotation train và tải ảnh theo nhu cầu trong epoch đầu; các epoch sau dùng cache. Những ảnh sai kích thước đã được wrapper CLIPSeg gốc liệt kê được loại với danh sách `excluded_train_tasks.json`.

Nếu Google Drive/Visual Genome không tải được, dùng Add Input với hai JSON và ảnh đã tải. Để chạy offline hoàn toàn, còn cần dependency đã cài (hoặc wheel trong Input) và `MODEL_PATH` trỏ tới một snapshot Hugging Face đầy đủ gồm weights, config, tokenizer và processor.

## Model và protocol

- Checkpoint: [`CIDAS/clipseg-rd64-refined`](https://huggingface.co/CIDAS/clipseg-rd64-refined), revision `999e0328d9e10b484360c477313983f9afdd7050`.
- Đánh giá dùng text prompt giữ nguyên phrase. Train chỉ cập nhật decoder (BCE, AdamW); support lấy mask GT của chính ảnh support trong split train; miniv không dùng để cập nhật trọng số hoặc chọn checkpoint.
- Nội suy logits bilinear về kích thước ảnh gốc, sigmoid rồi ngưỡng **0,3**; không min-max normalize.
- GT là hợp polygon của tất cả instance, rasterize bằng PIL theo [API PhraseCut](https://github.com/ChenyunWu/PhraseCutDataset/blob/276470b7773d8a230aee5b5536876fbc485fb04b/utils/data_transfer.py).
- Báo mean IoU theo query, cumulative IoU, Dice, mean AP pixel theo query và precision tại IoU 0,3/0,5/0,7/0,9. AP này chưa khẳng định cùng protocol với paper CLIPSeg.
- Báo kết quả cố định t=0,3 cho toàn bộ query được chọn; báo riêng calibration/evaluation, chia theo image_id 20%/80%. Ngưỡng bổ sung chỉ được chọn trên calibration rồi áp dụng lên evaluation.

Miniv chính thức có [100 ảnh và 821 phrase-region pairs](https://github.com/ChenyunWu/PhraseCutDataset). Paper báo số trên tập test, nên kết quả miniv không so trực tiếp với bảng paper; không gọi là zero-shot. Kết quả miniv dùng checkpoint **epoch cuối**, không chọn epoch theo miniv.

## Kết quả xuất ra

| File / thư mục | Nội dung |
|---|---|
| `summary.json` | Metric tổng hợp, kết quả ngưỡng cố định và ngưỡng calibration |
| `sample_metrics.csv` | Metric từng query tại ngưỡng cố định |
| `calibration_thresholds.csv` | Quét 0,1–0,7 chỉ trên calibration |
| `selected_tasks.json` | Task ID, image ID, phrase và nhóm calibration/evaluation |
| `run.json` | Revision, hash annotation/manifest, config, seed, phiên bản, GPU, timing và VRAM |
| `masks/<task_id>.png` | Mask nhị phân 0/255 ở độ phân giải ảnh gốc, dùng t=0,3 |
| `ground_truth/` | Mask GT để kiểm tra |
| `visualizations/` | Ảnh query, GT, heatmap và overlay |
| `probabilities/` | `.npy` float32 nếu bật `SAVE_PROBABILITIES` |
| `image_errors.json` | Lỗi tải / kích thước ảnh, rỗng khi kiểm tra thành công |
| `training_config.json` | Số epoch, batch, số mẫu train, steps/epoch, tổng steps, cấu hình optimizer và PhraseCut+ |
| `phrasecut_plus_samples.png` | 4 mẫu train: query + template, support, mask đáp án |
| `training_history.csv` | Train loss, số query và số bước sau mỗi epoch |
| `train_tasks.json` | Manifest train, tách biệt với miniv |
| `excluded_train_tasks.json` | Query bị loại do image_id sai kích thước theo wrapper tác giả |
| `checkpoints/decoder_epoch_XXX.pt` | Decoder, optimizer, scheduler, scaler và RNG để resume |

Để resume, đặt `RESUME_CHECKPOINT` đến file epoch đã lưu và giữ nguyên cấu hình/manifest, kể cả tổng `EPOCHS`. Notebook khôi phục trạng thái optimizer, scheduler và RNG, tiếp tục ở epoch kế tiếp. Checkpoint chỉ chứa decoder; backbone lấy từ `MODEL_ID`/`MODEL_REVISION` trong notebook. Nếu chọn `MODEL_PATH` offline, cần giữ chính snapshot đó.

Timing forward/resize/sigmoid được đo sau warm-up, đồng bộ CUDA; không gồm tải model, preprocessing, transfer, metric hoặc ghi file. Thời gian toàn vòng lặp được ghi riêng. Chạy lại sẽ thay kết quả trong thư mục output baseline, vì vậy tải ZIP hoặc đổi `WORK_ROOT` trước khi chạy một cấu hình mới cần giữ lại.

## Kiểm tra trước bàn giao

Notebook được kiểm tra schema/cú pháp và chạy từ đầu tới cuối bằng CPU tại máy phát triển: train trên 8 query PhraseCut train thật với 2 epoch, batch 3; sau đó đánh giá 12 query miniv thật. Có 3 bước mỗi epoch, tổng 6 bước, giữ đủ 8 mẫu kể cả batch cuối. Resume từ epoch 1 cho ra trọng số decoder và loss epoch 2 khớp chính xác với lượt train liên tục; scheduler cũng khớp. Bộ kiểm tra tự động có 8 test đã pass.

Các file kết quả kiểm tra ở `outputs/baseline-notebook-check/` và `outputs/baseline-notebook-resume/` (không đưa vào Git). Chưa thực thi trên session GPU Kaggle; `run.json` của lượt Kaggle mới là bằng chứng cho tốc độ và VRAM trên GPU.

## E2 và E3

Hai notebook riêng, cùng cách chạy trên Kaggle như E1 (upload một file `.ipynb`, GPU T4 x2, bật Internet, Run All). Phần tải dữ liệu, tải ảnh, model, rasterize GT và metric được chép từ notebook E1 bằng [`scripts/build_experiment_notebooks.py`](../scripts/build_experiment_notebooks.py), nên ba thí nghiệm dùng chung protocol. Notebook E1 cũng được sinh từ [`scripts/build_baseline_notebook.py`](../scripts/build_baseline_notebook.py). Sửa E1 trong script đó, rồi chạy lại cả hai script theo thứ tự E1 → E2/E3. Không sửa tay các file `.ipynb`.

| | [E2: text và visual prompt](e2_text_vs_visual.ipynb) | [E3: template câu](e3_prompt_templates.ipynb) |
|---|---|---|
| Query | 30 query miniv có ít nhất 3 ảnh support hợp lệ (seed 42) | 100 query miniv (seed 42, cách chọn giống E1) |
| Biến thể | `text_phrase` (gốc), `text_name`, `visual_crop`, `visual_dark`, `visual_crop_dark` | `{phrase}` (gốc), `a photo of {phrase}`, `the {phrase}`, `{phrase} in the image` |
| Support | Ảnh miniv **khác** có cùng `phrase_structure.name`, dùng mask GT của chính ảnh support; danh sách trong `e2_pairs.json` | Không dùng |
| Bảng chính | `e2_variant_table.csv` | `e3_template_table.csv` |
| Output | `/kaggle/working/clipseg_e2_text_vs_visual(.zip)` | `/kaggle/working/clipseg_e3_prompt_templates(.zip)` |

Cả hai báo metric tại t=0,3 cho mọi biến thể, kèm ΔIoU theo query so với biến thể gốc, CI 95% bootstrap theo ảnh và số query thắng/hòa/thua. Ngưỡng bổ sung chỉ chọn trên calibration. Ở E2, các dòng visual là query × support. ΔIoU lấy trung bình IoU trên các support của mỗi query trước khi so.

### Đánh giá decoder đã train theo epoch + batch size

E2/E3 không tự train. Epoch và batch size được đặt **một lần** trong notebook E1, rồi E2/E3 nạp decoder đó, để cả ba thí nghiệm dùng cùng một model:

1. Chạy notebook E1 với `TRAIN_DECODER=True`, đặt `EPOCHS`, `BATCH_SIZE` (và `TRAIN_ROOT`). Checkpoint mỗi epoch nằm ở `clipseg_e1/checkpoints/decoder_epoch_XXX.pt`. Từ bản này, checkpoint lưu kèm cấu hình train.
2. Trên Kaggle, **Save Version** notebook E1 để Output thành dataset, rồi **Add Input** output đó vào notebook E2/E3.
3. Đặt `DECODER_CHECKPOINT` tới checkpoint **epoch cuối**, ví dụ `"/kaggle/input/<output-e1>/clipseg_e1/checkpoints/decoder_epoch_004.pt"`. Notebook dừng nếu checkpoint được train trên base model/revision khác. Nếu checkpoint không phải epoch cuối, notebook in cảnh báo. E2 cảnh báo khi decoder train `text_only`, vì decoder đó chưa từng thấy embedding ảnh support.
4. `summary.json` và `run.json` ghi `decoder`: đường dẫn, sha256, epoch, số bước, batch size, cách khởi tạo và lịch sử loss.

`DECODER_CHECKPOINT=None` (mặc định) dùng decoder pretrained như trước. Hai lượt ghi cùng thư mục output, nên tải ZIP lượt trước khi chạy lượt sau. `BATCH_SIZE` trong E2/E3 là số prompt mỗi lần forward khi inference, không phải batch size train.

Kiểm tra trên CPU tại máy phát triển: E3 với 12 query (`{phrase}` cho mean IoU 0,5967, trùng E1 trên cùng 12 query) và E2 với 5 query × 2 support. Cả hai chạy hết các cell, với decoder pretrained và với decoder train thử trong E1 (8 query, 2 epoch, batch 3). Lượt dùng decoder train thử: E3 `{phrase}` cho 0,0932, trùng E1 đánh giá chính checkpoint đó. Chưa chạy trên GPU Kaggle.
