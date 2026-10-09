# Slide CLIPSeg bằng Beamer

Mẫu định dạng lấy từ `D:\AIDP\ADIVP\slides\lvit_full_report_slides.tex`: Beamer Madrid, 16:9, 11pt, font Latin Modern, tiêu đề xanh đậm và các bố cục hai cột. Preamble cùng các macro hình được giữ trong `style.tex`.

Sửa nội dung tại `clipseg_slides.tex`, thông tin sinh viên/môn học tại `../manuscript/metadata.tex`. Khung có **15 slide**, phù hợp kế hoạch trình bày khoảng 10--12 phút. Placeholder và bảng `---` cần thay bằng nội dung/số thực đo.

Từ thư mục gốc:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_latex.ps1 -Target slides
```

PDF xem trước: `output/pdf/clipseg_slides_template.pdf`. Biên dịch thủ công trong thư mục slides: `xelatex clipseg_slides.tex` hai lần.

Đưa lên Overleaf: upload cả hai thư mục `slides` và `manuscript` để giữ đường dẫn metadata và hình, chọn main file `slides/clipseg_slides.tex` và compiler XeLaTeX.
