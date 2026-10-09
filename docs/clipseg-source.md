# Mã nguồn CLIPSeg và chuẩn bị thực nghiệm

Đã lấy mã nguồn chính thức từ <https://github.com/timojl/clipseg.git> vào `third_party/clipseg` ngày 09/10/2026. Mã nguồn được quản lý dưới dạng Git submodule trong repo XLAV, giữ nguyên mã nguồn tác giả và lịch sử commit. Khi clone XLAV, dùng `--recurse-submodules` hoặc chạy `git submodule update --init --recursive`.

- Nhánh: `master`.
- Commit: `77c4e2dc853880df46413d4a870a725e961f2f73`.
- Kiểm tra phiên bản: `git -C third_party/clipseg rev-parse HEAD`.
- Kiểm tra thay đổi: `git -C third_party/clipseg status --short`.

## Các file nên đọc

| Mục đích | File |
|---|---|
| Hướng dẫn tác giả, dependency, checkpoint | [README.md](../third_party/clipseg/README.md) |
| Kiến trúc CLIP, FiLM, decoder | [models/clipseg.py](../third_party/clipseg/models/clipseg.py) |
| Inference mẫu | [Quickstart.ipynb](../third_party/clipseg/Quickstart.ipynb) |
| Visual prompt engineering | [Visual_Feature_Engineering.ipynb](../third_party/clipseg/Visual_Feature_Engineering.ipynb) |
| Bảng kết quả bài báo | [Tables.ipynb](../third_party/clipseg/Tables.ipynb) |
| Cấu hình thực nghiệm gốc | [experiments](../third_party/clipseg/experiments) |
| Huấn luyện và đánh giá | [training.py](../third_party/clipseg/training.py), [score.py](../third_party/clipseg/score.py) |
| Dataset wrapper | [datasets](../third_party/clipseg/datasets) |

## Chạy thử bằng môi trường hiện có

Các script của đồ án dùng bản CLIPSeg trong Transformers, checkpoint `CIDAS/clipseg-rd64-refined`, revision cố định `999e0328d9e10b484360c477313983f9afdd7050`. Chạy từ thư mục `D:\XLAV\Project`:

```powershell
.\.venv\Scripts\python.exe scripts/infer.py --image data/demo/example_image.jpg --text "a spoon" "a bowl" --output outputs/source-prep-text
.\.venv\Scripts\python.exe scripts/infer.py --image data/demo/example_image.jpg --support-image data/demo/support.jpg --output outputs/source-prep-visual
```

Mỗi prompt xuất probability `.npy`, mask, heatmap, overlay; `run.json` ghi phiên bản và cấu hình. Đây là kiểm tra inference; ảnh demo chưa có ground truth để đánh giá IoU. Ảnh support hiện tại là crop từ query, nên cần thay bằng ảnh support khác khi làm E2 theo kế hoạch.

Kiểm tra ngày 09/10/2026: cả text prompt (2 prompt) và visual prompt đều chạy thành công trên CPU trong `.venv` hiện có. Kết quả nằm tại `outputs/source-prep-text` và `outputs/source-prep-visual`. Kiểm tra này chạy triển khai Transformers; mã gốc của tác giả chưa được chạy.

## Chạy mã gốc của tác giả

Repo gốc đã tải đủ mã nguồn; checkpoint `.pth` và dataset benchmark chưa được tải trong bước này. Dependency gốc dùng OpenAI CLIP và torchvision; `environment.yml` pin `torch==1.12.1+cpu`, `torchvision==0.13.1+cpu`. Khi chạy notebook hoặc script gốc, tạo môi trường riêng tương thích thay vì cài file môi trường cũ vào `.venv` Python 3.12 của đồ án.

Hướng dẫn tải checkpoint nằm trong README tác giả. Bản refined dùng `CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)` và `weights/rd64-uni-refined.pth`. Cần chạy notebook từ thư mục `third_party/clipseg` để các import và đường dẫn tương đối đúng. Mã nguồn có giấy phép MIT; theo README tác giả, MIT không áp dụng cho trọng số.

Trước khi chạy E1–E3, chuẩn bị split dữ liệu, ảnh và mask GT theo [protocol hiện tại](implementation-plan.md). Giữ commit mã nguồn, revision checkpoint, prompt, threshold và seed trong metadata của mỗi lượt chạy để có thể tái lập kết quả.
