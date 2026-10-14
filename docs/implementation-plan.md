# Kế hoạch triển khai CLIPSeg: tái tạo quá trình train rồi cải tiến

**Cập nhật 09/10/2026.** Bản này thay kế hoạch ngày 07/10. Thay đổi lớn nhất: đồ án **không chỉ đánh giá checkpoint có sẵn** nữa. Mục tiêu mới là **tự train lại CLIPSeg theo công thức của tác giả (PhraseCut+), rồi cải tiến một module ở E4**. Checkpoint `CIDAS/clipseg-rd64-refined` của tác giả chuyển thành mốc tham chiếu.

Quy trình vẫn là [Scrum cho một người](agile/scrum.md): 3 sprint trong 07–27/10/2026, 5 giờ/ngày, WIP 1. Hướng dẫn chạy notebook ở [source/README.md](../source/README.md). Báo cáo nghiên cứu cho E4 ở [reports/Cải tiến CLIPSeg cho E4.md](../reports/Cải%20tiến%20CLIPSeg%20cho%20E4.md).

## 1. Những gì đã thay đổi so với kế hoạch 07/10

| Hạng mục | Kế hoạch 07/10 | Hiện tại (09/10) | Lý do |
|---|---|---|---|
| Hướng đồ án | Chỉ inference trên checkpoint có sẵn, không train | **Tái tạo quá trình train**, sau đó **cải tiến** | Muốn tự dựng lại kết quả thay vì dùng mô hình của tác giả |
| E1 | Đánh giá checkpoint trên tối đa 200 query miniv | **Train decoder trên PhraseCut+** theo epoch + batch size, đánh giá trên miniv; checkpoint tác giả là mốc tham chiếu | Như trên |
| Dữ liệu train | Không dùng | PhraseCut `train` (310.816 query, 71.746 ảnh). PhraseCut+ được **tạo ngẫu nhiên lúc train**, không có file riêng | Tác giả không công bố PhraseCut+; nó là code `PhraseCutPlus` sinh từ PhraseCut |
| E2 | Chạy trên checkpoint tác giả | Chạy trên **decoder E1** (qua `DECODER_CHECKPOINT`) | Cả ba thí nghiệm dùng cùng một model |
| E3 | Như E2 | Như E2 | Như E2 |
| E4 | Tùy chọn: so CLIPSeg với Grounding DINO + SAM 2 | **Cải tiến một module CLIPSeg**: word-level cross-attention trong decoder (dự phòng: `reduce_dim` 128) | Trọng tâm chuyển sang cải tiến CLIPSeg |
| Ablation | Không có | **E1b**: train `text_only` (giống `rd64-uni-novis` của tác giả) | Đo xem học ảnh support có giúp visual prompt và có làm giảm text hay không |
| File | Một notebook baseline | Ba notebook E1/E2/E3, sinh bằng script; 16 test | Giữ protocol thống nhất giữa các thí nghiệm |

Vẫn **ngoài phạm vi**: SAM 3, benchmark RefCOCO đầy đủ, PASCAL-5i, so sánh Grounding DINO + SAM, LoRA hoặc đổi backbone CLIP (vượt ngân sách GPU, xem báo cáo E4).

## 2. Hiện trạng (09/10)

**Đã xong**, nằm trong PR [Quang-Luu-2005/XLAV#1](https://github.com/Quang-Luu-2005/XLAV/pull/1):

| Thành phần | File |
|---|---|
| E1: train PhraseCut+ / text-only, đánh giá miniv | [source/e1_train_eval_phrasecut.ipynb](../source/e1_train_eval_phrasecut.ipynb) |
| E2: text và visual prompt | [source/e2_text_vs_visual.ipynb](../source/e2_text_vs_visual.ipynb) |
| E3: template câu | [source/e3_prompt_templates.ipynb](../source/e3_prompt_templates.ipynb) |
| Script sinh notebook (sửa ở đây, không sửa `.ipynb`) | [scripts/build_baseline_notebook.py](../scripts/build_baseline_notebook.py), [scripts/build_experiment_notebooks.py](../scripts/build_experiment_notebooks.py) |
| Test (16 test, đều qua) | [tests/](../tests/) |

Cả ba notebook đã chạy hết các cell trên CPU ở máy cá nhân với mẫu nhỏ. Lượt train thử PhraseCut+ gồm 12 query, 11 query có support; hình kiểm tra mẫu đúng như mong đợi.

**Chưa có:**
- Lượt chạy nào trên **Kaggle GPU**: chưa đo thời gian mỗi epoch và VRAM ở batch 64.
- **Số liệu thật** cho E1–E4. Mọi con số trong các lượt chạy thử chỉ để kiểm tra code, không dùng cho báo cáo.
- Đánh giá trên **tập test** PhraseCut. Paper báo số trên test, miniv chỉ là tập kiểm tra nhỏ.
- Code cho E4.

## 3. Các thí nghiệm

### E1: tái tạo quá trình train

- **Model**: CLIP ViT-B/16 đóng băng + decoder khởi tạo ngẫu nhiên (`TRAIN_INIT="random_decoder"`), kiến trúc theo config của `CIDAS/clipseg-rd64-refined`.
- **PhraseCut+** (`TRAIN_MODE="phrasecut_plus"`). Mỗi lần lấy mẫu:
  - crop vuông ngẫu nhiên;
  - 20% mẫu âm (`NEGATIVE_PROB=0.2`);
  - ảnh support lấy từ ảnh khác có cùng cụm từ, xử lý crop + blur + nền 10%;
  - một trong 9 template `shuffle+`;
  - trộn `w·text + (1−w)·ảnh`, `w ~ U(0; 0,5)`.
- **Cấu hình gần paper**: `EPOCHS=4`, `BATCH_SIZE=64`, `USE_AMP=True`, AdamW lr 1e-3 → 1e-4 cosine. 20.000 bước × 64 ≈ 4 epoch.
- **Mốc tham chiếu**: chạy notebook với `TRAIN_DECODER=False` để đánh giá checkpoint tác giả bằng cùng protocol.
- **Kết quả cần có**: bảng mIoU / cumulative IoU / AP / Precision@0,5/0,7/0,9 của (a) checkpoint tác giả, (b) E1 PhraseCut+, (c) E1b text-only.

### E1b: ablation chỉ học text

`TRAIN_MODE="text_only"`, giữ nguyên mọi thứ khác. Nếu thiếu GPU, train ít epoch hơn và ghi rõ trong báo cáo.

### E2: text prompt và visual prompt

- **Dữ liệu**: 30 query miniv, mỗi query 3 support từ ảnh miniv khác cùng danh từ chính.
- **Biến thể**: `text_phrase` (gốc), `text_name`, `visual_crop`, `visual_dark`, `visual_crop_dark`.
- **Decoder**: chạy trên decoder E1. Chạy thêm decoder E1b để đối chứng. Notebook cảnh báo khi decoder chỉ học text.
- **Kết quả**: ΔIoU so với `text_phrase`, CI 95% bootstrap theo ảnh, số query thắng/hòa/thua.

### E3: template câu

100 query miniv × 4 template (`{phrase}`, `a photo of {phrase}`, `the {phrase}`, `{phrase} in the image`), chạy trên decoder E1. Báo cáo đủ cả template tốt lẫn kém. Không chọn template trên tập evaluation.

### E4: cải tiến một module

Theo [báo cáo nghiên cứu E4](../reports/Cải%20tiến%20CLIPSeg%20cho%20E4.md):

- **Phương án chính**: giữ FiLM, thêm **word-level cross-attention** vào 3 block decoder.
  - Pixel token làm query, token chữ của CLIP text encoder làm key/value.
  - Có tanh gate khởi tạo bằng 0, khoảng 0,15M tham số.
  - Fine-tune tiếp từ checkpoint E1, khoảng 2–3 giờ.
- **Dự phòng**: `reduce_dim` 64 → 128, train lại từ đầu. Đây là thay đổi duy nhất từng được đo trên PhraseCut với chính CLIPSeg (+2,1 mIoU).
- **Thiết kế công bằng**, tất cả cùng công thức PC+:
  - **B0**: E1.
  - **B1**: E1 train tiếp K bước, kiến trúc gốc. Đây là đối chứng chính.
  - **E4a**: E1 + cross-attention, cùng K bước. **So E4a với B1, không so với B0.**
  - **E4b**: D=128 train từ đầu, cùng số bước với E1.
- **Đánh giá**: t=0,3 cố định và ngưỡng chọn trên calibration; bootstrap ghép cặp theo query; chia theo kích thước vật. Chạy lại E2 để kiểm tra visual prompt không bị kém đi.
- **Kỳ vọng trung thực**: 0 đến +2 mIoU, xấp xỉ mức nhiễu của miniv. Kết quả âm hoặc không đổi vẫn đáng báo cáo nếu thiết kế chặt.

## 4. Protocol đánh giá (áp dụng cho mọi thí nghiệm)

- **Từ logits đến mask**: nội suy bilinear logits về kích thước gốc, rồi sigmoid; không min-max.
- **Ngưỡng**: báo chính ở 0,3. Ngưỡng bổ sung chỉ chọn trên calibration (20% ảnh, chia theo image_id, seed 42), rồi báo trên evaluation.
- **Metric**: mean IoU theo query, cumulative IoU, Dice, mean AP pixel theo query. AP này chưa đối chiếu với protocol AP của paper.
- **Ground truth**: hợp polygon của mọi instance, rasterize bằng PIL.
- **Không gọi là zero-shot**: decoder được train trên PhraseCut.
- **Tách dữ liệu**: miniv không dùng để cập nhật trọng số hay chọn checkpoint. Kết quả dùng checkpoint **epoch cuối**.

## 5. Những điều phải ghi rõ khi viết báo cáo

1. **PC+ cho điểm text thấp hơn PC.** Paper báo trên tập test, t=0,3: PC+ đạt 43,4 / 54,7 / 76,7 (mIoU / IoU_FG / AP), còn PC chỉ text đạt 46,1 / 56,2 / 78,2. E1 train kiểu PC+ nên phải so với 43,4. E1 thấp hơn 46 không phải lỗi tái tạo.
2. **Layer CLIP**: config HF dùng `extract_layers` [3, 6, 9], paper ghi [3, 7, 9]. E1 dùng [3, 6, 9].
3. **Khác bản gốc**: support luôn lấy từ ảnh khác; GT rasterize bằng PIL; đánh giá trên miniv thay vì test. Vì vậy không so trực tiếp với bảng paper.
4. **Số bước train**: tác giả train 20.000 bước (`rd64-uni`) hoặc 50.000 bước (`rd64-uni-refined`, chính là checkpoint HF). Nếu E1 train ít hơn, phải ghi rõ.

## 6. Ngân sách GPU và rủi ro

Kaggle cho khoảng 30 giờ GPU/tuần, mỗi phiên tối đa 12 giờ. Ước tính chưa đo:

| Lượt | Ước tính |
|---|---|
| Lượt đo thử (`MAX_TRAIN_QUERIES=2000`, `EPOCHS=1`) | < 1 giờ |
| E1 đầy đủ (4 epoch, batch 64, AMP) | 6–10 giờ, chia nhiều phiên bằng `RESUME_CHECKPOINT` |
| E1b text-only | 4–8 giờ, nhanh hơn vì không tính embedding ảnh support |
| E2, E3 | < 1 giờ mỗi cái |
| E4 (B1, E4a × 2 seed, E4b) | khoảng 16–20 giờ |

| Rủi ro | Cách xử lý |
|---|---|
| Thời gian train dài hơn ước tính | Giảm E1b xuống 1–2 epoch; E4 chỉ làm E4a và B1 |
| Batch 64 không vừa VRAM T4 | Giảm batch 32, ghi rõ trong báo cáo |
| Tải ảnh train chậm hoặc lỗi | Dùng Kaggle Dataset có sẵn ảnh Visual Genome, đặt `TRAIN_ROOT`, `NUM_WORKERS=4` |
| Notebook chỉ resume ở ranh giới epoch | Mỗi epoch phải dưới 12 giờ; nếu không thì giảm `MAX_TRAIN_QUERIES` |
| Chênh lệch E4 nằm trong nhiễu | Bootstrap ghép cặp; nếu kịp, đánh giá thêm trên một phần PhraseCut val/test |

## 7. Lịch

| Sprint | Việc | Mốc nghiệm thu |
|---|---|---|
| **S1** (đến 13/10) | Merge PR #1. Tạo Kaggle Dataset PhraseCut (train + ảnh). Lượt đo thử. Bắt đầu E1 đầy đủ. Đánh giá checkpoint tác giả trên miniv | Biết thời gian/epoch và VRAM; có bảng mốc tham chiếu |
| **S2** (14–20/10) | Train xong E1 và E1b. Chạy E2, E3. Phân tích lỗi. Code và chạy E4. **Khóa số liệu 20/10** | Bảng E1/E1b, E2, E3, E4; hình minh họa |
| **S3** (21–27/10) | Báo cáo, slide, demo, video dự phòng; luyện bảo vệ | Gói nộp hoàn chỉnh |

## 8. Việc cần làm tiếp, theo thứ tự

1. Merge PR #1 sau khi review.
2. Chọn hoặc tạo Kaggle Dataset PhraseCut có `refer_train.json`, `refer_miniv.json`, `image_data_split.json` và thư mục ảnh. Kiểm tra notebook tự tìm được đường dẫn.
3. Chạy E1 với `TRAIN_DECODER=False` trên miniv đầy đủ (`MAX_QUERIES=None`) để có mốc tham chiếu. Kiểm tra `run.json` có `device = cuda`.
4. Chạy lượt đo thử: `MAX_TRAIN_QUERIES=2000`, `EPOCHS=1`, `BATCH_SIZE=64`, `USE_AMP=True`. Ghi thời gian và VRAM vào daily log, rồi chốt số epoch cho E1/E1b.
5. Train E1 đầy đủ, rồi E1b. Lưu Output thành Dataset để E2/E3/E4 nạp checkpoint.
6. Chạy E2, E3 với `DECODER_CHECKPOINT` trỏ tới epoch cuối của E1. E2 chạy thêm với E1b.
7. Viết code E4a vào script sinh E1: module cross-attention, nạp state dict E1 với `strict=False`, tùy chọn train tiếp K bước. Thêm test cho module, rồi chạy B1, E4a và E4b nếu đủ GPU.
8. Cập nhật Trello/backlog theo các bước trên.
