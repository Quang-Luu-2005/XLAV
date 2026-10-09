"""Adapt the user's local ADIVP LaTeX styles into CLIPSeg starter templates."""
from pathlib import Path
import shutil

import fitz

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\AIDP\ADIVP")
report = ROOT / "manuscript"
slides = ROOT / "slides"
for folder in [report / "assets", report / "figures", report / "sections", slides]:
    folder.mkdir(parents=True, exist_ok=True)

original = (SOURCE / "manuscript/main.tex").read_text(encoding="utf-8")
preamble = original[original.index(r"\usepackage[left="):original.index(r"\begin{document}")]
preamble = preamble.replace(r"Figure placeholder\\\texttt{#2}", r"Chèn hình\\\texttt{\detokenize{#2}}")
(report / "main.tex").write_text(
    "% !TEX TS-program = xelatex\n"
    + "\\documentclass[a4paper,10pt]{report}\n\\input{metadata}\n"
    + preamble
    + r"""
\begin{document}
\include{cover}
\fontsize{10pt}{12.5pt}\selectfont
\include{preface}
\include{info}
\include{sections/01_introduction}
\include{sections/02_related_work}
\include{sections/03_method}
\include{sections/04_experiments}
\include{sections/05_conclusion}
\chapter*{Tài liệu tham khảo}
\addcontentsline{toc}{chapter}{Tài liệu tham khảo}
\printbibliography[heading=none]
\end{document}
""", encoding="utf-8")

cover = (SOURCE / "manuscript/cover.tex").read_text(encoding="utf-8")
cover = cover.replace(r"Nhóm: \groupname", r"Hình thức: \groupname")
cover = cover.replace("                           & \\studenttwo{} -- \\studenttwoid \\\\[6pt]\n", "")
cover = cover.replace("                           & \\advisortwo \\\\[6pt]\n", "")
cover = cover.replace("                           & \\advisorthree \\\\[12pt]\n", "")
(report / "cover.tex").write_text(cover, encoding="utf-8")

# Restore the original embedded logo, which is absent from the source assets folder.
source_pdf = fitz.open(SOURCE / "manuscript/Bao_cao_do_an_LViT_QaTaLViT.pdf")
images = source_pdf[0].get_images(full=True)
if not images:
    raise RuntimeError("The reference cover does not contain an extractable logo")
logo_info = max(images, key=lambda image: image[2] * image[3])
pix = fitz.Pixmap(source_pdf, logo_info[0])
if logo_info[1]:
    pix = fitz.Pixmap(pix, fitz.Pixmap(source_pdf, logo_info[1]))
pix.save(report / "assets/HCMUS.png")

for src, target in [
    ("data/demo/example_image.jpg", "demo_query.jpg"),
    ("outputs/smoke-text/prompt_00_overlay.png", "demo_text.png"),
    ("outputs/smoke-visual/prompt_00_overlay.png", "demo_visual.png"),
]:
    if (ROOT / src).exists():
        shutil.copy2(ROOT / src, report / "figures" / target)

original_slides = (SOURCE / "slides/lvit_full_report_slides.tex").read_text(encoding="utf-8")
slide_style = original_slides[:original_slides.index(r"\title[LViT")]
(slides / "style.tex").write_text(slide_style, encoding="utf-8")
print("Copied report preamble, cover, embedded logo, slide style and smoke figures.")
