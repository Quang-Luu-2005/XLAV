# Scrum cho đồ án CLIPSeg một người

Áp dụng từ **07 đến 27/10/2026**, 5 giờ/ngày, giờ Việt Nam. Đây là cách áp dụng nhịp Scrum cho cá nhân: bạn đảm nhiệm việc ưu tiên sản phẩm, giữ quy trình và thực hiện kỹ thuật. Giảng viên là người nhận kết quả và góp ý khi có điều kiện. Tham chiếu [Scrum Guide chính thức](https://scrumguides.org/scrum-guide.html) cho Product Goal, Sprint Goal, Increment và Definition of Done; thời lượng và quy ước Trello dưới đây được điều chỉnh cho đồ án này.

## Product Goal

Đến cuối ngày 27/10 có đồ án CLIPSeg chạy lại được: inference text/visual, ba thí nghiệm E1–E3 với protocol và bằng chứng, demo, báo cáo LaTeX, 15 slide và video dự phòng. Số liệu tự đo được phân biệt với số bài báo. Ngày 27 là mốc kế hoạch, chưa phải deadline môn đã xác nhận.

## Vai trò của bạn

| Vai trò | Việc quyết định |
|---|---|
| Product Owner | Ưu tiên backlog theo yêu cầu môn, giá trị và thời gian còn lại |
| Scrum Master | Giữ nhịp review, làm rõ blocker và chọn bước tháo gỡ |
| Developer | Đọc paper, viết code, kiểm tra dữ liệu, chạy thí nghiệm và viết báo cáo |

Codex hỗ trợ theo yêu cầu trong phiên làm việc. Không mặc định có một thành viên khác theo dõi, họp hoặc thực hiện các sprint tương lai tự động.

## Ba sprint và sản phẩm dùng được

| Sprint | Thời gian | Sprint Goal | Increment để review |
|---|---|---|---|
| S1 | 07–13/10 | Có baseline CLIPSeg kiểm chứng và giải thích được | CLI text/visual, dữ liệu/manifest sạch, protocol, E1, nháp nền tảng/kiến trúc và mẫu LaTeX |
| S2 | 14–20/10 | Có bằng chứng trả lời hai câu hỏi về visual prompt và cách viết text | E1–E3 đã khóa, 12 hình phân tích lỗi, demo text/visual, báo cáo nháp có bảng kết quả |
| S3 | 21–27/10 | Người khác đọc, xem và chạy lại được đồ án | Báo cáo và slide hoàn chỉnh, demo/video, gói code/README, kiểm tra gói nộp |

S1 là sprint hiện tại. S2/S3 là dự kiến, được điều chỉnh tại Planning theo kết quả sprint trước. Demo P12 chuyển từ S3 lên S2 để phát hiện lỗi sớm. Viết báo cáo và thử pipeline diễn ra trong từng sprint. Mẫu LaTeX đã làm được ghi nhận là L01 Done; nội dung cuối cùng của báo cáo và slide vẫn chưa hoàn thành.

## Capacity và ước lượng

Mỗi sprint có tối đa 35 giờ. Dành khoảng 3 giờ cho Planning, daily, refinement, Review và Retrospective. Các giờ trên thẻ là ước lượng ban đầu, không phải lịch cố định hoặc thời gian thực tế.

| Sprint | Ước lượng các thẻ bắt buộc | Đã Done khi thiết lập Scrum | Phần còn lại | Quỹ học/sửa/dự phòng sau 3h quy trình |
|---|---:|---:|---:|---:|
| S1 | 19,5h | 4,5h | 15h | 12,5h |
| S2 | 20h | 0h | 20h | 12h |
| S3 | 18h | 0h | 18h | 14h |

L01 2h là ước lượng hồi cứu. P13 tăng lên 8h, chia hai lượt 4h để hoàn thiện báo cáo; P14 4h. P17 là kiểm tra bàn giao 2h; thời gian dự phòng được giữ trong capacity, không đóng vai trò một task 6h cần làm hết. O01 đối chứng 4h ở backlog tùy chọn và chỉ đưa vào S2 nếu đạt Sprint Goal sớm, còn capacity và GPU phù hợp. Không mở thí nghiệm mới trong S3.

## Trello

Xem [backlog và liên kết từng thẻ](backlog.md); dữ liệu máy đọc được ở `../tasks.json` và `../trello.json`.

| Cột | Quy tắc sử dụng |
|---|---|
| Scrum • Mục tiêu và quy ước | Product Goal, Sprint Goal, nhịp làm việc và quy tắc nghiệm thu |
| Product Backlog • S2/S3 dự kiến | Việc tương lai và tùy chọn, xếp theo giá trị và phụ thuộc |
| Sprint Backlog • S1 07–13/10 | Việc chọn cho sprint hiện tại, đã có đầu vào và tiêu chí rõ |
| In Progress • WIP 1 | Tối đa một thẻ đang trực tiếp thực hiện |
| Review / Test | Sản phẩm đã làm xong, đang kiểm tra chất lượng/bằng chứng |
| Blocked | Ghi nguyên nhân, bước tháo gỡ tiếp theo và mốc kiểm tra lại |
| Done | Đạt Acceptance Criteria và Definition of Done, có bằng chứng |

Tên thẻ: `[S1][P0][P03] Chuẩn bị dữ liệu • 4h`. P0 phục vụ mục tiêu bắt buộc, P1 tăng chất lượng và phân tích, P2 là mở rộng. Mức ưu tiên là quy ước riêng của dự án. Mỗi thẻ có user story, phụ thuộc, giờ ước lượng, tiêu chí riêng và chỗ ghi bằng chứng. Một thẻ có thể gồm checklist các bước ngắn; mỗi bước nên làm được trong một phiên tối đa 4–5 giờ.

Khi sang sprint mới, hoàn tất Review/Retro, đưa việc chưa Done trở lại Product Backlog để xét lại, đổi tên cột Sprint Backlog, rồi kéo những thẻ đã chọn vào đó. Không kéo toàn bộ S2/S3 vào sprint chỉ vì đã có tên sprint trên thẻ.

## Điều kiện sẵn sàng của thẻ

Đây là quy ước dự án: biết giá trị cần tạo, có đầu vào/link nguồn, tiêu chí nghiệm thu rõ, phụ thuộc đã xử lý hoặc có phương án, có ước lượng và bước nhỏ đầu tiên. Khi chưa đủ điều kiện, làm rõ trong refinement trước khi bắt đầu.

## Definition of Done

Mọi thẻ cần đạt tiêu chí riêng, có đường dẫn bằng chứng, được kiểm tra phù hợp và cập nhật phần tài liệu liên quan. Dữ liệu phải có nguồn/split; số liệu phải có manifest, checkpoint, prompt, ngưỡng và log; code phải chạy lại được bằng lệnh đã ghi. Với LaTeX, PDF phải biên dịch được, xem kiểm tra bố cục, dẫn nguồn đầy đủ và không còn placeholder trong phần được bàn giao là hoàn chỉnh. Với paper, phải có ghi chú bằng lời của bạn. Với demo, thử ít nhất ba ảnh ở cả text và visual, có video dự phòng.

Kéo vào Review/Test khi có sản phẩm. Tự chạy lại hoặc đọc lại theo checklist, rồi chuyển Done và đánh dấu complete. Việc chưa đạt chuẩn không được tính vào Increment. Các mẫu có placeholder được nghiệm thu dưới tên **mẫu**, không dùng để đánh dấu báo cáo hoàn chỉnh.

## Nhịp làm việc

Không áp đặt giờ bắt đầu trong ngày; các hoạt động tính trong 5 giờ/ngày của bạn.

| Hoạt động | Thời lượng | Khi nào | Đầu ra |
|---|---:|---|---|
| Sprint Planning | 30 phút | Đầu phiên 07, 14, 21/10 | Goal, backlog được chọn, bước thực hiện và capacity |
| Daily cá nhân | 10 phút | Đầu mỗi phiên làm việc | Tiến độ tới Goal, việc quan trọng hôm nay, blocker và bước xử lý |
| Backlog refinement | 20 phút | 10, 17, 24/10 | Làm rõ thẻ tiếp theo, cập nhật ước lượng/phụ thuộc |
| Sprint Review | 30 phút | Cuối phiên 13, 20, 27/10 | Chạy/show Increment, so với Goal, ghi thiếu sót và điều chỉnh backlog |
| Retrospective | 20 phút | Sau Review 13, 20, 27/10 | Một thay đổi quy trình cụ thể cho phiên/sprint tiếp theo |

Daily ghi trong `daily-log.md`, Review/Retro ghi trong `review-retro.md`. Thử tự trình bày kết quả như trình bày cho giảng viên; nếu có góp ý thật thì ghi lại, không đánh dấu stakeholder đã review khi chưa diễn ra.

Nhịp ngày thường: 10 phút daily, 50 phút đọc/viết, 3 giờ làm thẻ chính, 1 giờ kiểm tra và ghi bằng chứng. Ngày Planning/Review lấy thời gian từ các phiên này. Nếu đã chạy inference dài, có thể viết báo cáo trong thời gian chờ; chỉ giữ một việc cần tập trung trực tiếp trong In Progress.

## Xử lý thay đổi và rủi ro

Thẻ mới vào Product Backlog trước. Khi đổi phạm vi trong sprint, ghi lý do và đánh đổi, giữ Sprint Goal và chất lượng nghiệm thu. Nếu có nguy cơ trễ: bỏ O01 trước; giảm N thí nghiệm có ghi rõ; ưu tiên pipeline/GT đúng; giữ thời gian cho báo cáo và demo. Blocker dữ liệu quá 2 giờ cần chọn phương án tải/nguồn khác đã kiểm chứng; không lặp cùng một cách thử. Ngưỡng chỉ chọn trên calibration, support E2 khác ảnh query.

## Theo dõi tiến độ

Theo dõi Goal đạt/chưa đạt, số thẻ Done và tổng **giờ còn lại ước lượng** mỗi ngày. Ghi giờ thực tế riêng nếu có, không coi số ước lượng là số đã đo. Chưa có sprint hoàn tất để tính velocity. Cuối sprint dùng công việc thực sự Done và các blocker đã gặp để điều chỉnh dự kiến sprint sau.

Việc tiếp theo hiện tại: **P01 đọc paper và ghi chú**, sau đó **P03 chuẩn bị dữ liệu**. P00, P02 và L01 đã Done. E1–E3 chưa chạy benchmark.
