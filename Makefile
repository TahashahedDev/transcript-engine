PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF   := .venv/bin/ruff
MYPY   := .venv/bin/mypy

.PHONY: install test lint typecheck format check all

install:
	python3.12 -m venv .venv
	$(PIP) install -e ".[dev]" --quiet

test:
	$(PYTEST) tests/unit/ -v

test-all:
	RUN_SLOW_TESTS=1 $(PYTEST) tests/ -v

lint:
	$(RUFF) check transcript_engine/ tests/

format:
	$(RUFF) format transcript_engine/ tests/

typecheck:
	$(MYPY) transcript_engine/

check: lint typecheck test

all: install check
