.PHONY: help install dev run test lint fmt typecheck clean build-macos build-appimage samples

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

help:
	@echo "PlotaViz make targets:"
	@echo "  install         create venv and install runtime deps"
	@echo "  dev             install with dev extras + pre-commit hooks"
	@echo "  run             launch the GUI"
	@echo "  test            run pytest"
	@echo "  lint            ruff check + format check + mypy on core/"
	@echo "  fmt             ruff format (writes)"
	@echo "  samples         regenerate the sample datasets"
	@echo "  build-macos     build the .app bundle (macOS only)"
	@echo "  build-appimage  build the AppImage (Linux only)"
	@echo "  clean           remove build artifacts and caches"

$(BIN)/python:
	$(PY) -m venv $(VENV)

install: $(BIN)/python
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e .

dev: $(BIN)/python
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev,llm]"
	-$(BIN)/pre-commit install

run:
	$(BIN)/python -m plotaviz.main

test:
	$(BIN)/python -m pytest -v

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .
	$(BIN)/mypy plotaviz/core

fmt:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

samples:
	$(BIN)/python scripts/make_samples.py

build-macos:
	$(BIN)/pip install -e ".[build]"
	bash scripts/build_macos.sh

build-appimage:
	$(BIN)/pip install -e ".[build]"
	bash scripts/build_appimage.sh

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
