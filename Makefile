SHELL:=/usr/bin/env bash

.PHONY: lint
lint:
	uv run ruff check --exit-non-zero-on-fix
	uv run ruff format --check --diff
	uv run mypy .

.PHONY: unit
unit:
	uv run pytest

.PHONY: package
package:
	uv run pip check

.PHONY: test
test: lint package unit
