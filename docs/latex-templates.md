# Bộ mẫu LaTeX cho CLIPSeg

Định dạng lấy từ `D:\AIDP\ADIVP`, đã chuyển sang dự án hiện tại và một sinh viên. Dự án nguồn không bị chỉnh sửa.

- Báo cáo: `manuscript/main.tex`; nội dung 5 chương trong `manuscript/sections/`.
- Slide: `slides/clipseg_slides.tex`; 15 slide, Beamer Madrid 16:9.
- Thông tin dùng chung: `manuscript/metadata.tex`.
- Hướng dẫn riêng: `manuscript/README.md`, `slides/README.md`.
- PDF xem trước: `output/pdf/clipseg_report_template.pdf`, `output/pdf/clipseg_slides_template.pdf`.
- Gói upload/chép: `output/latex/CLIPSeg_LaTeX_templates.zip`.

Build cả hai từ thư mục gốc:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_latex.ps1
```

Yêu cầu XeLaTeX và Biber. Máy hiện tại đã có MiKTeX và đã biên dịch cả hai. Các trang PDF được render và xem kiểm tra; trích dẫn và mục lục đã cập nhật. Các ô `[...]`, hình trống và số `---` là phần chủ đích để bạn điền nội dung, không phải kết quả thực nghiệm đã hoàn tất.

Tên/MSSV giữ theo mẫu người dùng. Thông tin học kỳ cũ và sinh viên thứ hai đã gỡ; kiểm tra và điền lớp, giảng viên, mã môn, học kỳ và năm học trong metadata trước khi nộp.
