# Nhật ký khởi tạo — 07/10/2026

Đã chuyển quản lý sang [Scrum cho một người](agile/scrum.md): ba sprint, backlog có ưu tiên/phụ thuộc/tiêu chí, WIP 1, Review/Test, Blocked và Done. Demo chuyển lên S2; mẫu LaTeX được ghi nhận L01 Done. Các sự kiện tương lai và thí nghiệm chưa được đánh dấu hoàn thành.

Đã bổ sung bộ mẫu LaTeX từ ADIVP theo yêu cầu: [báo cáo và slide](latex-templates.md), dùng chung metadata, khung 5 chương và 15 slide. Đã biên dịch PDF và kiểm tra bố cục. Đây là khung soạn thảo, chưa phải báo cáo/slide hoàn chỉnh.

Đã lập kế hoạch 07–27/10/2026 cho một người, 5 giờ/ngày; 54,5 giờ công việc tập trung đã ước lượng (bao gồm 6 giờ dự phòng), cộng đối chứng tùy chọn 4 giờ. Phần còn lại trong quỹ tối đa 105 giờ dành cho học, kiểm tra dữ liệu, chạy lại và sửa bài.

Đã tạo bảng riêng trên Trello, 19 thẻ với hạn 22:00 giờ Việt Nam, mô tả phụ thuộc và tiêu chí hoàn thành. Liên kết thẻ có trong `trello.json`.

Đã tải paper chính thức về `references/CLIPSeg_CVPR2022.pdf`, có hướng dẫn đọc 3 giờ. Việc bạn đọc và viết ghi chú chưa hoàn thành.

Đã cài `.venv` Python 3.12, PyTorch 2.6.0, Transformers 4.55.4. Khóa phiên bản đầy đủ tại `requirements-lock-windows.txt`. Máy hiện tại không có CUDA khả dụng trong môi trường đã kiểm tra.

Đã chạy checkpoint `CIDAS/clipseg-rd64-refined` commit `999e0328d9e10b484360c477313983f9afdd7050` trên CPU:

- Text: ảnh ví dụ repo tác giả, `a spoon` và `a bowl`, output tại `outputs/smoke-text/`.
- Visual: crop chứa thìa từ cùng ảnh ví dụ, output tại `outputs/smoke-visual/`. Đây là kiểm tra API; không phải E2 với support ảnh khác.
- Cả hai xuất probability, mask, heatmap, overlay ở kích thước ảnh gốc 365×365 và `run.json`.
- Đã xem overlay: text thìa còn bao gồm phần xung quanh; support crop kích hoạt cả vùng bát. Đây là ví dụ cho thấy cần phân tích prompt/nền và ngưỡng, chưa có GT để định lượng chất lượng.

Đã kiểm tra 3 tình huống metric: giao/hợp tính tay, khác biệt mean với cumulative IoU, đầu vào không hợp lệ. Tất cả qua. Số tham số đo của checkpoint refined: 150.747.746 tổng và 1.127.009 decoder; không thay số biến thể paper bằng số này mà không nêu checkpoint.

Chưa tải PhraseCut, chưa chạy benchmark E1–E3, chưa làm demo UI hoặc viết báo cáo/slide hoàn chỉnh. Những việc này đã có thẻ. Bước tiếp theo: bạn đọc paper; chuẩn bị PhraseCut miniv, khóa split theo image_id và xem 10 mask GT trước khi chạy baseline.

## Đồng bộ kế hoạch và Trello — 09/10/2026

Phần 07/10 phía trên là lịch sử khởi tạo. Phạm vi hiện tại theo [kế hoạch triển khai](implementation-plan.md): train E1 PhraseCut+ với epoch/batch, E1b text-only, E2/E3 dùng decoder E1 và E4a giữ FiLM + cross-attention gate=0. E4a phải so B1 train tiếp cùng K bước PC+. Miniv đánh giá độc lập; không train/chọn checkpoint. E4b D=128 tùy chọn; Grounding DINO + SAM đã hoãn.

Đã cập nhật [Trello](https://trello.com/b/85LHoRTG): 18 thẻ sửa, 6 thẻ thêm (P18 review PR, P19 pilot GPU, P20 E1b, P21 code E4, P22 B1/E4a, O02 E4b), tổng 30 thẻ. Đã đọc lại xác nhận tên/mô tả/hạn và vị trí. P01 giữ Done theo bạn cập nhật; P03 là In Progress duy nhất; P04/P18 Review / Test. Đồng bộ tasks/trello JSON, backlog và hướng dẫn Scrum. Công bắt buộc 71,5h, Done 7,5h, còn 64h ước lượng; đây không phải thời gian thực tế.

Ba notebook E1/E2/E3 và script sinh đã có. Chạy lại 16 test: tất cả qua. Log E1/E2/E3 hiện tại là CPU smoke; chưa có benchmark Kaggle GPU. [PR #1](https://github.com/Quang-Luu-2005/XLAV/pull/1) OPEN, chưa merge; E4 mới có nghiên cứu. Bước tiếp theo là input Kaggle/GT và review PR, sau đó pilot P19 để xác nhận VRAM/thời gian/ngân sách E1/E1b/E4 trước khi train full.
