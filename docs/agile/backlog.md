# Backlog hiện tại — 09/10/2026

Đã đối chiếu [Trello](https://trello.com/b/85LHoRTG) với [kế hoạch triển khai](../implementation-plan.md), [hướng dẫn notebook](../../source/README.md) và [nghiên cứu E4](../../reports/Cải%20tiến%20CLIPSeg%20cho%20E4.md). Phạm vi mới: train E1 PhraseCut+, E1b text-only, E2/E3 trên decoder tự train và E4a đối chiếu B1 cùng ngân sách. Nhập epoch và batch size; ghi số bước thực tế. Miniv dùng đánh giá, không train/chọn checkpoint.

S1 hiện tại; S2/S3 dự kiến và điều chỉnh sau pilot. Giờ trên thẻ là công tập trung ước lượng, khác GPU-giờ. Hạn 22:00 giờ Việt Nam. Trạng thái là ảnh chụp ngày 09/10; Trello là trạng thái làm việc trực tiếp.

| Thẻ | Sprint | Ưu tiên | Công việc | Ước lượng | Mốc dự kiến | Trạng thái |
|---|---|---|---|---:|---|---|
| [L01](https://trello.com/c/cti1fblT) | S1 | P0 | Bộ mẫu báo cáo và slide LaTeX | 2h | 07/10 | Done |
| [P00](https://trello.com/c/cFR5opn9) | S1 | P0 | Chốt phạm vi 3 tuần và lập Trello | 0.5h | 07/10 | Done |
| [P01](https://trello.com/c/1Ta30fI3) | S1 | P0 | Đọc paper CLIPSeg và ghi chú 1 trang | 3h | 08/10 | Done |
| [P02](https://trello.com/c/yKsupW9k) | S1 | P0 | Dựng môi trường và chạy inference đầu tiên | 2h | 08/10 | Done |
| [P03](https://trello.com/c/LL3Anca0) | S1 | P0 | Chuẩn bị PhraseCut train + miniv trên Kaggle | 4h | 10/10 | In Progress |
| [P04](https://trello.com/c/5AXDehBd) | S1 | P0 | Chốt protocol và kiểm tra metric | 2h | 10/10 | Review / Test |
| [P05](https://trello.com/c/mEU7tsFG) | S1→S2 | P0 | E1: train PhraseCut+ và mốc pretrained | 4h | 14/10 | Sprint Backlog |
| [P06](https://trello.com/c/ugyvb5P6) | S1 | P1 | Viết nháp nền tảng và kiến trúc CLIPSeg | 3h | 13/10 | Sprint Backlog |
| [P18](https://trello.com/c/vD4T8rYD) | S1 | P0 | Review và merge PR #1 | 1h | 10/10 | Review / Test |
| [P19](https://trello.com/c/ZESTZeTl) | S1 | P0 | Đo pilot Kaggle GPU và chốt epoch/batch | 2h | 12/10 | Sprint Backlog |
| [P07](https://trello.com/c/VSPMiLyH) | S2 | P0 | E2: text/visual trên decoder E1 và E1b | 4h | 17/10 | Product Backlog |
| [P08](https://trello.com/c/Kw6ax0TG) | S2 | P0 | E3: thử 4 cách viết prompt | 4h | 17/10 | Product Backlog |
| [P09](https://trello.com/c/gDxkXjlJ) | S2 | P1 | Phân tích lỗi và chọn hình minh họa | 3h | 20/10 | Product Backlog |
| [P10](https://trello.com/c/2SY1q2aS) | S2 | P1 | Viết dữ liệu, protocol và khảo sát gọn | 3h | 19/10 | Product Backlog |
| [P11](https://trello.com/c/P8qjb5vH) | S2 | P0 | Khóa E1/E1b/E2–E4 và báo cáo nháp | 2h | 20/10 | Product Backlog |
| [P12](https://trello.com/c/e0BaAbKp) | S2 | P0 | Làm demo text và ảnh mẫu | 4h | 20/10 | Product Backlog |
| [P20](https://trello.com/c/ViLmaf3X) | S2 | P0 | E1b: ablation train text_only | 3h | 16/10 | Product Backlog |
| [P21](https://trello.com/c/iolocchm) | S2 | P0 | E4a: triển khai cross-attention gate=0 | 4h | 18/10 | Product Backlog |
| [P22](https://trello.com/c/g7NUn3qV) | S2 | P0 | E4: chạy B1/E4a và đối chiếu công bằng | 3h | 19/10 | Product Backlog |
| [P13](https://trello.com/c/G0P8lpHU) | S3 | P0 | Hoàn thiện báo cáo | 8h | 23/10 | Product Backlog |
| [P14](https://trello.com/c/jmRyT4tr) | S3 | P0 | Làm slide và chuẩn bị câu hỏi bảo vệ | 4h | 24/10 | Product Backlog |
| [P15](https://trello.com/c/m0nNTAXf) | S3 | P1 | Tập trình bày và thử demo offline | 2h | 25/10 | Product Backlog |
| [P16](https://trello.com/c/2Tq8fciW) | S3 | P0 | Đóng gói và chạy lại từ README | 2h | 26/10 | Product Backlog |
| [P17](https://trello.com/c/H95n1hDw) | S3 | P0 | Kiểm tra gói nộp và chốt bàn giao | 2h | 27/10 | Product Backlog |
| [O01](https://trello.com/c/6zqZm0P1) | Hoãn | P2 | Grounding DINO + SAM 2 | 4h | 20/10 | Hoãn ngoài scope |
| [O02](https://trello.com/c/s5LUM5MA) | Tùy chọn | P2 | E4b dự phòng: reduce_dim 64 → 128 | 4h | 20/10 | Product Backlog |

Tổng bắt buộc: **71,5h**, trong đó **7,5h** thuộc thẻ Done; còn **64h** ước lượng. Quỹ tối đa 105h trừ khoảng 9h quy trình còn 24,5h học/sửa/dự phòng. S2 có 30h công tập trung, chỉ còn khoảng 2h dự phòng sau quy trình, chưa tính phần theo dõi E1 kéo từ S1; cần xác nhận GPU-giờ bằng P19. O01 hoãn ngoài scope; O02 4h chưa được chọn. Xem [quy trình và Sprint Goal](scrum.md).

P01 giữ Done theo cập nhật của bạn. P03 là việc duy nhất In Progress; P04/P18 đang Review / Test; P05/P06/P19 ở Sprint Backlog. Ba notebook và 16 test đã có, CPU smoke đã chạy; PR #1 còn OPEN, chưa merge, chưa có benchmark Kaggle GPU. E4 mới có nghiên cứu, chưa triển khai.
