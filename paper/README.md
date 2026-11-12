# Manuscript package

`main.tex`, `references.tex`, `sections/` and `figures/` contain the complete article, with no TeX inputs outside this folder. `main.pdf` is the review copy. From the repository root, run `python workflows/build_paper.py` with Tectonic installed to rebuild it and create `source.zip`.

Numerical inputs, reference tables and executable reproduction workflows are at repository level, rather than duplicated inside the manuscript. Run `make results` before building if you want to regenerate figures. Regeneration uses frozen evidence; native reevaluation and optimization are explicit separate commands.
