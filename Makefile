PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
PYTEST := .venv/bin/pytest
RUFF   := .venv/bin/ruff
MYPY   := .venv/bin/mypy

# Every Python package in the repo. Kept in one place so `make lint` and
# `make typecheck` cover the same ground as a release check — an earlier version
# checked only transcript_engine/ and so reported clean while api/ and scripts/
# were never looked at.
SOURCES   := transcript_engine api db scripts
LINT_DIRS := $(SOURCES) tests

# mypy runs in strict mode (see pyproject). That is right for shipped code but
# not for tests, where fixtures and monkeypatching are deliberately untyped —
# so tests are linted but not type-checked.

.PHONY: install test test-all lint format typecheck check all

# python3 rather than a pinned minor version: the project supports 3.12+, and
# hardcoding one that happens not to be installed makes the target simply fail.
install:
	python3 -m venv .venv
	$(PIP) install --upgrade pip --quiet
	$(PIP) install -e ".[dev]" --quiet

test:
	$(PYTEST) tests/unit/ -q

test-all:
	RUN_SLOW_TESTS=1 $(PYTEST) tests/ -q

lint:
	$(RUFF) check $(LINT_DIRS)

format:
	$(RUFF) format $(LINT_DIRS)

typecheck:
	$(MYPY) $(SOURCES)

check: lint typecheck test

all: install check
