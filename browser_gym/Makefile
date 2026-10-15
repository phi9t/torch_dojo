PYTHON := .venv/bin/python
JUPYTER := .venv/bin/jupyter

.PHONY: setup build offline-build serve test

setup:
	python -m venv .venv
	.venv/bin/pip install -r requirements-build.txt
	npm ci --prefix extension --no-audit --no-fund

build:
	$(PYTHON) build_catalog.py
	PATH="$(CURDIR)/.venv/bin:$$PATH" npm --prefix extension run build
	mkdir -p .venv/share/jupyter/labextensions/@torch-dojo
	rm -rf .venv/share/jupyter/labextensions/@torch-dojo/browser-gym
	cp -R extension/labextension .venv/share/jupyter/labextensions/@torch-dojo/browser-gym
	# doit skips the jupyter-lite.json merge when only dist/ changed, which drops disabledExtensions.
	rm -f .jupyterlite.doit.db
	$(JUPYTER) lite build --force --lite-dir . --contents build --output-dir dist
	mkdir -p dist/gym-assets/checkers
	cp build/checkers/*.json dist/gym-assets/checkers/
	cp -R build/lessons dist/gym-assets/
	cp build/catalog.json build/motifs.json dist/gym-assets/
	cp check_runtime.py checker-worker.js dist/gym-assets/

offline-build:
	$(PYTHON) offline_build.py
	$(PYTHON) offline_build.py --verify

serve:
	$(PYTHON) -m http.server 8000 --directory dist

test:
	PYTHONPATH=.. $(PYTHON) -m pytest tests labs/mlp/test_checks.py -q
	npm --prefix extension run typecheck
	npm --prefix extension test
	npm --prefix extension run test:e2e
