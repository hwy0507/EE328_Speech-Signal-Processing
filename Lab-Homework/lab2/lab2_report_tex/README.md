# LaTeX Report Package

## Contents
- `main.tex`: complete report source
- `figures/`: all report figures (renamed to `fig01.png` ... `fig17.png`)

## Compile
`main.tex` now supports both `pdfLaTeX` and `XeLaTeX`:

```bash
cd lab2_report_tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

If your platform can select compiler, `XeLaTeX` is also fine:

```bash
cd lab2_report_tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

If you use Overleaf/OpenAI Prism:
1. Upload the whole folder.
2. Prefer **pdfLaTeX** first (current template supports it).
3. If your environment lacks CJK package support, switch to **XeLaTeX**.
4. Compile `main.tex`.

## Note
In the Praat section, each phoneme (AA/IY/UW) is arranged as 3 images in one row to save space.
