# Báo cáo CLIPSeg bằng LaTeX

Mẫu lấy từ `D:\AIDP\ADIVP\manuscript`: A4, report 10pt, Latin Modern, giãn dòng 1.15, lề trái/phải 0.75in và trên/dưới 1in. Giữ bìa khung, logo HCMUS, mục lục, danh sách hình/bảng và bibliography IEEE. Logo trích từ PDF tham chiếu vì thư mục nguồn thiếu file assets.

Sửa trước: `metadata.tex` dùng chung với slide. Tên và MSSV lấy từ mẫu cũ; kiểm tra lại. Điền mã môn, lớp, giảng viên, học kỳ và năm học. Báo cáo đã chuyển sang một sinh viên.

Nội dung chính chia theo `sections/01_introduction.tex` đến `05_conclusion.tex`. Các đoạn trong dấu `[...]` và các giá trị `---` là ô cần hoàn thiện, chưa phải kết quả thực nghiệm. File `references.bib` là thư viện trích dẫn. `figures/` chứa ví dụ smoke và là nơi đặt hình thực nghiệm.

Từ thư mục gốc dự án chạy:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_latex.ps1 -Target report
```

PDF xem trước: `output/pdf/clipseg_report_template.pdf`. Trên Overleaf, upload thư mục manuscript, chọn `main.tex` và compiler **XeLaTeX**. Bibliography dùng **Biber** theo cấu hình biblatex; README mẫu gốc ghi BibTeX không khớp với preamble nên đã sửa hướng dẫn.

Biên dịch thủ công trong thư mục manuscript: `xelatex main.tex`, `biber main`, rồi `xelatex main.tex` hai lần.
