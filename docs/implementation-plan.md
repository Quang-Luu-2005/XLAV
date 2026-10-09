# Kế hoạch triển khai CLIPSeg trong 3 tuần, một người

**Quy trình thực hiện hiện tại:** [Scrum cho một người](agile/scrum.md). Trello dùng Product Backlog, Sprint Backlog, In Progress, Review/Test, Blocked và Done. Các thay đổi dưới đây thay thế lịch tuần cũ: demo P12 chuyển lên S2 (hạn 20/10); P13 dành 8h, P14 4h; P17 là kiểm tra bàn giao 2h, thời gian dự phòng giữ trong capacity. S2/S3 là dự kiến và được chốt lại sau mỗi Sprint Review.

Giai đoạn **07–27/10/2026**, giờ Việt Nam. Ngày 27 là mốc kết thúc kế hoạch; deadline môn chưa được cung cấp. Bạn xác nhận dành **5 giờ/ngày**, tối đa 105 giờ trong 21 ngày. Các thẻ là ước lượng công việc tập trung, không chiếm hết quỹ 105 giờ; thời gian còn lại dành cho học, kiểm tra dữ liệu, chờ chạy, sửa và dự phòng. Lịch chi tiết, giờ ước lượng và tiêu chí nghiệm thu ở `tasks.json` và Trello. Chỉ chuyển tối đa 1 thẻ vào Đang làm.

Kế hoạch gốc đặt mục tiêu cho 3 người / 10 tuần. Bản này là kế hoạch thực thi hiện tại; tài liệu gốc giữ vai trò danh sách hướng nghiên cứu dài hạn.

Sản phẩm cần hoàn thành: code inference có thể chạy lại; 3 bảng E1–E3 với nguồn số liệu; báo cáo khoảng 15–20 trang (điều chỉnh theo yêu cầu môn); 12–15 slide; demo text/visual và video dự phòng. Tất cả dùng checkpoint sẵn có, không huấn luyện lại. Đóng góp dự kiến là thực nghiệm có kiểm soát và phân tích lỗi; chưa tuyên bố tính mới học thuật.

| Tuần | Công việc | Mốc nghiệm thu |
|---|---|---|
| S1 07–13/10 | Đọc 1 paper, môi trường, dữ liệu, metric, E1; viết kiến trúc | CLI text/visual; manifest tách theo ảnh; bảng baseline; nháp kiến trúc và mẫu LaTeX |
| S2 14–20/10 | E2/E3, phân tích lỗi, protocol/khảo sát, demo | 3 bảng, 12 hình, demo, báo cáo nháp; khóa số liệu ngày 20/10 |
| S3 21–27/10 | Hoàn thiện báo cáo/slide, luyện bảo vệ, đóng gói | Gói nộp có code/README, báo cáo, slide, video |

E1 dùng PhraseCut miniv từ nguồn chính thức, tối đa 200 query. Tách calibration/evaluation theo image_id, seed 42; không để các query của cùng ảnh nằm ở hai phần. E1 trên tập con là đánh giá checkpoint, không gọi là tái lập chính xác bảng paper. Số đo hiện chưa có.

E2 tuyển 30 query có support từ **ảnh khác**, cùng loại mục tiêu. So text với crop support và support làm tối nền. Không dùng mask GT query để tạo prompt; ghi riêng danh sách support. Nếu không đủ support hợp lệ, giảm số cặp và báo rõ N.

E3 dùng tối đa 100 query cố định, bốn template: `{phrase}`, `a photo of {phrase}`, `the {phrase}`, `{phrase} in the image`. Giữ nguyên ảnh, GT, ngưỡng và ý nghĩa mục tiêu. Báo cáo cả template tốt lẫn kém; không chọn template dựa trên evaluation rồi coi đó là đánh giá độc lập.

Protocol: nội suy bilinear logits về kích thước gốc rồi sigmoid, không min-max. Báo cáo ngưỡng 0.3; chỉ chọn ngưỡng bổ sung trên calibration. Đo mean IoU theo query, cumulative IoU, Dice, mean AP pixel theo query; ghi rõ AP này chưa đối chiếu với protocol AP của paper. Không dùng tên class-averaged mIoU hay PASCAL-5i cho đánh giá này. PhraseCut là dữ liệu có liên hệ với huấn luyện CLIPSeg; không gọi E1 là zero-shot.

Giới hạn phạm vi: SAM 3, fine-tuning, benchmark RefCOCO đầy đủ, PASCAL-5i đầy đủ và SAM refinement chuyển sang sau đồ án. Grounding DINO + SAM 2 chỉ là tùy chọn 4 giờ trước ngày 20/10 nếu phần bắt buộc đã ổn. Khảo sát liên quan gọn 5–6 phương pháp, dùng nguồn chính thức; chỉ CLIPSeg cần đọc sâu.

Nếu tải dữ liệu không giải quyết trong 2 giờ, chuyển sang tập nhỏ có GT kiểm chứng và ghi đúng tên/split. Nếu chỉ có CPU, chạy mẫu nhỏ để kiểm tra, dùng Colab/Kaggle cho batch đánh giá. Nếu ít hơn 2 giờ/ngày, giảm E1 còn 100 query, E2 còn 15 cặp, E3 còn 50 query, bỏ đối chứng tùy chọn.

Nhịp 5 giờ mỗi ngày theo Scrum: 10 phút daily, 50 phút đọc/ghi chú hoặc viết báo cáo, 3 giờ làm thẻ chính, 1 giờ kiểm tra và ghi bằng chứng. Planning/Review/Retro lấy thời gian trong các phiên này. S3 dành phần lớn phiên chính cho báo cáo/slide và luyện bảo vệ. Xem [backlog có liên kết thẻ](agile/backlog.md).

Hôm nay: 3 giờ đọc paper, 1 giờ xem kết quả smoke và hiểu lệnh chạy, 1 giờ chuẩn bị bước dataset. Không đánh dấu thẻ đọc paper hoàn thành thay bạn.
