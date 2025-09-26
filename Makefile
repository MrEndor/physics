SHELL:=/usr/bin/env bash

.PHONY: serve
serve:
	uv run streamlit run ${PWD}/src/main.py

.PHONY: lint
lint:
	uv run ruff check --exit-non-zero-on-fix
	uv run ruff format --check --diff
	uv run mypy ${PWD}

.PHONY: format
format:
	uv run ruff format

.PHONY: unit
unit:
	uv run pytest

.PHONY: package
package:
	uv run pip check

.PHONY: test
test: lint package unit
