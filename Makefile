PROJECT_NAME = kilograms
VE = ve_$(PROJECT_NAME)
install: clean_ve
	virtualenv $(VE)
	$(VE)/bin/pip install IPython pytest
	$(VE)/bin/pip install -e .
clean_ve:
	rm -rf $(VE) || true
py:
	$(VE)/bin/ipython
