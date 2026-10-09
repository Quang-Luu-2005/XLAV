# XLAV — Đồ án CLIPSeg, 3 tuần, một người

Quản lý theo [Scrum cho một người](docs/agile/scrum.md): 3 sprint, 5h/ngày, WIP 1, Review/Test trước Done. Theo dõi tại [daily log](docs/agile/daily-log.md) và [review/retro](docs/agile/review-retro.md).

Mẫu LaTeX theo đồ án ADIVP: [báo cáo](manuscript/README.md) và [slide Beamer](slides/README.md). Sửa thông tin chung tại `manuscript/metadata.tex`; build cả hai bằng `powershell -ExecutionPolicy Bypass -File scripts/build_latex.ps1`.

Kế hoạch hiện tại: [docs/implementation-plan.md](docs/implementation-plan.md). Paper và lộ trình đọc: [docs/paper-reading.md](docs/paper-reading.md). Kế hoạch khảo sát ban đầu: [Kế hoạch đồ án CLIPSeg.md](Kế%20hoạch%20đồ%20án%20CLIPSeg.md).

Mô hình chính: `CIDAS/clipseg-rd64-refined`. Code nhận text hoặc ảnh support, xuất probability `.npy`, mask, heatmap, overlay và metadata. Đây là triển khai ban đầu; E1–E3 chưa có số liệu benchmark.

Baseline Kaggle GPU: train decoder trên PhraseCut train theo epoch + batch size, đánh giá trên miniv. Xem [source/README.md](source/README.md) và [notebook độc lập](source/baseline_clipseg_phrasecut_miniv.ipynb). Upload notebook lên Kaggle, bật GPU và Internet, đặt `EPOCHS`/`BATCH_SIZE` rồi Run All. Đặt `TRAIN_DECODER=False` nếu chỉ đánh giá checkpoint pretrained.

## Mã nguồn CLIPSeg gốc

Repo tác giả [timojl/clipseg](https://github.com/timojl/clipseg) đã được clone vào [third_party/clipseg](third_party/clipseg), tại commit `77c4e2dc853880df46413d4a870a725e961f2f73`. Xem [hướng dẫn mã nguồn và chuẩn bị thực nghiệm](docs/clipseg-source.md) để tìm kiến trúc, notebook và cấu hình bài báo.

Clone repo đồ án kèm mã nguồn gốc (Git submodule):

```powershell
git clone --recurse-submodules https://github.com/Quang-Luu-2005/XLAV.git
cd XLAV
```

Nếu đã clone mà chưa có mã CLIPSeg, chạy `git submodule update --init --recursive`.

## Cài và chạy trên Windows

Python 3.12. Dùng môi trường riêng:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/download_demo.py
.\.venv\Scripts\python.exe scripts/infer.py --image data/demo/example_image.jpg --text "a spoon" "a bowl" --output outputs/smoke-text
.\.venv\Scripts\python.exe scripts/infer.py --image data/demo/example_image.jpg --support-image data/demo/support.jpg --output outputs/smoke-visual
```

Lượt đầu tải checkpoint từ Hugging Face. `--device auto` chọn CUDA nếu có, CPU nếu không. Môi trường Windows ở đây dùng CPU. Để chạy Colab/Kaggle GPU, cài PyTorch theo môi trường CUDA của notebook trước, rồi cài các dependency còn lại; không thay wheel CUDA bằng wheel CPU Windows.

Ảnh demo lấy từ [repo tác giả](https://github.com/timojl/clipseg/blob/master/example_image.jpg); chỉ dùng kiểm tra inference, không dùng tính IoU vì chưa có GT. `support.jpg` là crop minh họa từ cùng ảnh demo, nên **không phải protocol support ảnh khác của E2**. Với ảnh cá nhân, thay hai đường dẫn bằng ảnh query và ảnh support của bạn.

## Tính metric từ probability đã lưu

CSV manifest gồm `id,probability,mask`; đường dẫn tương đối với thư mục chứa CSV. Probability là `.npy` hai chiều, mask PNG một kênh có giá trị 0/1 hoặc 0/255, cùng kích thước; không truyền trực tiếp semantic-label PNG.

```powershell
.\.venv\Scripts\python.exe scripts/evaluate.py --manifest path/to/manifest.csv --threshold 0.3 --output outputs/e1
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Mean IoU là trung bình theo query, cumulative IoU là tổng giao / tổng hợp. AP là AP pixel theo query rồi trung bình; không khẳng định tương đương AP bài báo. Manifest có GT trống sẽ bị từ chối trong protocol này. Xem [protocol](docs/implementation-plan.md) trước khi báo số liệu.

`requirements.txt` pin dependency trực tiếp. Sau kiểm tra môi trường, `requirements-lock-windows.txt` ghi các phiên bản đầy đủ dùng tại máy này. `run.json` lưu commit checkpoint, phiên bản, số tham số và cấu hình. Thời gian smoke là một lượt chưa warm-up, không dùng làm benchmark tốc độ.

Dataset và output bỏ qua trong Git. Không đưa token vào repo. Nguồn dataset chính: [PhraseCutDataset](https://github.com/ChenyunWu/PhraseCutDataset); chưa tải benchmark trong bước khởi tạo này.
