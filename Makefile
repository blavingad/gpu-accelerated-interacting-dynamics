PYTHON ?= python
TECTONIC ?= tectonic

.PHONY: results verify paper
results:
	$(PYTHON) workflows/assess_prints.py
	$(PYTHON) workflows/reproduce_figures.py
	$(PYTHON) workflows/draw_methodology.py
verify:
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) workflows/verify_uniform_field.py --device cpu
	$(PYTHON) workflows/verify_kernel.py
paper:
	$(PYTHON) workflows/build_paper.py --engine $(TECTONIC)
