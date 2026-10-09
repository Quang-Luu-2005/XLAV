# Bản dịch tiếng Việt paper CLIPSeg

Paper: Timo Lüddecke và Alexander Ecker, *Image Segmentation Using Text and Image Prompts*, CVPR 2022.

- Nguồn: `../CLIPSeg_CVPR2022.pdf`.
- [PDF tiếng Việt](../../output/pdf/clipseg-vietnamese/CLIPSeg_CVPR2022-vi.pdf).
- Công cụ: [`pdf-translate`](D:/KLTN/VI-Translate/SKILL.md), core đi kèm phiên bản `1.9.11`, ruleset `code4life-preservation-v1`.
- Chế độ: handoff; Codex trực tiếp dịch các đoạn, công cụ ghép lại PDF. Không gửi văn bản sang Google Translate. Không cần OCR.
- Ngày tạo: 07/10/2026. Đây là bản dịch hỗ trợ đọc, không phải bản dịch chính thức của tác giả.

## Phạm vi và kiểm tra

Nội dung chính từ Tóm tắt đến Kết luận, Hạn chế, Tác động rộng hơn và chú thích hình/bảng đã được dịch. Giữ thuật ngữ kỹ thuật cần đối chiếu, tên phương pháp, số liệu, công thức và trích dẫn. Các nhãn nằm trong hình, một số bảng được engine bảo vệ, dấu bản quyền và danh mục tham khảo vẫn bằng tiếng Anh. Do đó không gọi đây là bản Việt hóa mọi chữ trong PDF.

Engine trích xuất 121 đoạn: 107 đoạn đầu gồm nội dung, chú thích, nhãn bảng và thông tin tác giả; 14 đoạn cuối là tài liệu tham khảo và được giữ tiếng Anh. Đã cung cấp đủ 121 bản ghi hợp lệ; bước ghép báo 0 đoạn thiếu trong tập đoạn được trích xuất. Con số này không tính chữ trong vùng được bảo vệ.

Đã xác nhận cùng 11 trang, cùng kích thước trang, SHA-256 file gốc không đổi; không có placeholder bị lộ, U+0000, U+FFFD hoặc đoạn vượt trang theo phép kiểm tra text của công cụ. Đã render và xem từng trang bằng Poppler; kiểm tra thêm công thức ở trang 3, kiến trúc trang 4, bảng trang 5-8 và tài liệu tham khảo trang 9-11. Bốn cảnh báo giao nhau của bounding box thuộc dấu umlaut của tên riêng ở trang 1 và phân số ở trang 3, không phải đoạn văn chồng nhau.

Kiểm tra công cụ: 354 unittest chạy xong (35 skipped), `pip check`, Ruff E9/F63/F7/F82 và `git diff --check` đạt. GUI smoke-test không đạt vì môi trường thiếu phụ thuộc OCR; bản dịch dùng CLI với OCR tắt và không phụ thuộc GUI. Không sửa mã nguồn của VI-Translate.

## Thuật ngữ

| Tiếng Anh | Cách dùng trong bản dịch |
|---|---|
| prompt | lời nhắc |
| visual prompt engineering | thiết kế lời nhắc thị giác |
| query image / support image | ảnh truy vấn / ảnh hỗ trợ |
| backbone | mạng trích xuất đặc trưng, mạng nền |
| decoder | bộ giải mã |
| embedding | biểu diễn nhúng |
| conditioning | điều kiện hóa |
| affordance | khả năng sử dụng của đối tượng |
| referring expression segmentation | phân đoạn theo biểu thức tham chiếu |
| zero-shot / one-shot | giữ thuật ngữ gốc |
| ground truth | nhãn chuẩn |
| intersection over union (IoU) | tỷ số giao trên hợp |

## Tái tạo

`translations.jsonl` lưu nguyên văn khóa `src`, bản dịch `dst` và các marker công thức/kiểu chữ. `qa.json` lưu kết quả kiểm tra cấu trúc và ghi nhận xem trang. Chạy từ `D:\KLTN\VI-Translate`, chọn thư mục kết quả mới để tránh ghi đè:

```powershell
& .\.venv\Scripts\python.exe scripts\translate_pdf.py `
  D:\XLAV\Project\references\CLIPSeg_CVPR2022.pdf `
  --engine handoff --source-language en --target-language vi `
  --segments D:\XLAV\Project\references\clipseg-vi\translations.jsonl `
  --output-dir D:\XLAV\Project\output\pdf\clipseg-vietnamese-rerun
```

Paper nêu lớp trích xuất `[3,7,9]` và 1,122,305 tham số học được cho biến thể mô tả. Checkpoint đang triển khai có `extract_layers=[3,6,9]` và số tham số đo thực tế khác; bản dịch giữ số liệu của paper. Khi viết báo cáo, đối chiếu cấu hình triển khai như hướng dẫn trong `docs/paper-reading.md`.

Việc có bản dịch chưa đồng nghĩa đã hoàn thành task đọc paper P01 trên Trello.
