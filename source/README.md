# Baseline CLIPSeg — PhraseCut miniv trên Kaggle GPU

File chạy chính: [baseline_clipseg_phrasecut_miniv.ipynb](baseline_clipseg_phrasecut_miniv.ipynb). Notebook tự chứa code; upload **một file `.ipynb`** là đủ, không cần clone repo XLAV hoặc submodule CLIPSeg. Mặc định train decoder theo **epoch + batch size** trên PhraseCut `train`, rồi đánh giá trên `miniv`.

## Chạy trên Kaggle

1. Tạo notebook Kaggle, dùng **Import Notebook** để upload file trên.
2. Trong Settings, chọn GPU (ví dụ T4) và bật Internet. Baseline dùng GPU đầu tiên.
3. Đặt `EPOCHS`, `BATCH_SIZE`, và `TRAIN_ROOT` nếu đã Add Input tập train. Chạy **Run All**. Mặc định tải annotation từ nguồn PhraseCut chính thức, ảnh theo URL trong metadata và base checkpoint từ Hugging Face.
4. Tải `clipseg_baseline_miniv.zip` trong Output. Các file gốc nằm tại `/kaggle/working/clipseg_baseline_miniv/`.

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

Mặc định notebook đặt `EPOCHS=1`, `BATCH_SIZE=8`, huấn luyện decoder ngẫu nhiên với CLIP pretrained đóng băng. Chọn `TRAIN_INIT="pretrained_decoder"` để fine-tune decoder sẵn có, hoặc `TRAIN_DECODER=False` để chỉ chạy baseline pretrained. Đây là ba thiết lập khác nhau, cần ghi rõ khi báo số liệu.

Cell cấu hình có `MAX_QUERIES=None` để chạy toàn bộ 821 query / 100 ảnh của miniv. Đặt `8` để kiểm tra nhanh, hoặc `200` để chạy E1 gọn theo kế hoạch. `MAX_TRAIN_QUERIES=8` chạy thử train trên 8 query **từ split train**, không lấy miniv để train. Danh sách query được lưu, sampling và chia ảnh dùng seed 42. Mặc định FP32, batch 8; nếu thiếu VRAM, giảm batch xuống 4 hoặc 1. Có thể bật `USE_AMP=True` cho FP16, nhưng phải ghi cấu hình này khi so sánh kết quả.

Cell cài đặt pin `transformers==4.55.4` và `gdown==5.2.0`. PyTorch/CUDA và các thư viện phân tích dùng bản sẵn có của Kaggle, được ghi vào `run.json`; không cài `requirements.txt` dành cho Windows lên Kaggle. Chạy notebook trong session mới để tránh import nhầm Transformers đã được nạp trước khi cài.

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
- Text prompt mặc định giữ nguyên phrase. Chỉ train decoder với BCE và AdamW; không dùng GT để tạo prompt, không dùng miniv để cập nhật trọng số hoặc chọn checkpoint.
- Nội suy logits bilinear về kích thước ảnh gốc, sigmoid rồi ngưỡng **0,3**; không min-max normalize.
- GT là hợp polygon của tất cả instance, rasterize bằng PIL theo [API PhraseCut](https://github.com/ChenyunWu/PhraseCutDataset/blob/276470b7773d8a230aee5b5536876fbc485fb04b/utils/data_transfer.py).
- Báo mean IoU theo query, cumulative IoU, Dice, mean AP pixel theo query và precision tại IoU 0,3/0,5/0,7/0,9. AP này chưa khẳng định cùng protocol với paper CLIPSeg.
- Báo kết quả cố định t=0,3 cho toàn bộ query được chọn; báo riêng calibration/evaluation, chia theo image_id 20%/80%. Ngưỡng bổ sung chỉ được chọn trên calibration rồi áp dụng lên evaluation.

Miniv chính thức có [100 ảnh và 821 phrase-region pairs](https://github.com/ChenyunWu/PhraseCutDataset). Đây là baseline text-only với kiến trúc refined; không gọi là tái lập chính xác bảng paper hoặc zero-shot. Repo CLIPSeg gốc có rasterization, preprocessing, visual/text mixing, negative sampling và augmentation khác. Kết quả miniv dùng checkpoint **epoch cuối**, không chọn epoch theo miniv.

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
| `training_config.json` | Số epoch, batch, số mẫu train, steps/epoch, tổng steps và cấu hình optimizer |
| `training_history.csv` | Train loss, số query và số bước sau mỗi epoch |
| `train_tasks.json` | Manifest train, tách biệt với miniv |
| `excluded_train_tasks.json` | Query bị loại do image_id sai kích thước theo wrapper tác giả |
| `checkpoints/decoder_epoch_XXX.pt` | Decoder, optimizer, scheduler, scaler và RNG để resume |

Để resume, đặt `RESUME_CHECKPOINT` đến file epoch đã lưu và giữ nguyên cấu hình/manifest, kể cả tổng `EPOCHS`. Notebook khôi phục trạng thái optimizer, scheduler và RNG, tiếp tục ở epoch kế tiếp. Checkpoint chỉ chứa decoder; backbone lấy từ `MODEL_ID`/`MODEL_REVISION` trong notebook. Nếu chọn `MODEL_PATH` offline, cần giữ chính snapshot đó.

Timing forward/resize/sigmoid được đo sau warm-up, đồng bộ CUDA; không gồm tải model, preprocessing, transfer, metric hoặc ghi file. Thời gian toàn vòng lặp được ghi riêng. Chạy lại sẽ thay kết quả trong thư mục output baseline, vì vậy tải ZIP hoặc đổi `WORK_ROOT` trước khi chạy một cấu hình mới cần giữ lại.

## Kiểm tra trước bàn giao

Notebook được kiểm tra schema/cú pháp và chạy từ đầu tới cuối bằng CPU tại máy phát triển: train trên 8 query PhraseCut train thật với 2 epoch, batch 3; sau đó đánh giá 12 query miniv thật. Có 3 bước mỗi epoch, tổng 6 bước, giữ đủ 8 mẫu kể cả batch cuối. Resume từ epoch 1 cho ra trọng số decoder và loss epoch 2 khớp chính xác với lượt train liên tục; scheduler cũng khớp. Bộ kiểm tra tự động có 8 test đã pass.

Các file kết quả kiểm tra ở `outputs/baseline-notebook-check/` và `outputs/baseline-notebook-resume/` (không đưa vào Git). Chưa thực thi trên session GPU Kaggle; `run.json` của lượt Kaggle mới là bằng chứng cho tốc độ và VRAM trên GPU.
