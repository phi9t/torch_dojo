PYTHON ?= python

.PHONY: install eval test

install:
	$(PYTHON) -m pip install -e .

eval:
	@set -e; for script in dojo/*_eval.py; do PYTHONPATH=. $(PYTHON) "$$script"; done

test:
	$(PYTHON) -m unittest discover -s tests -v
