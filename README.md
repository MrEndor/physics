# Physics

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![workflow](https://github.com/MrEndor/physics/actions/workflows/test.yaml/badge.svg)

### Here are presented projects on the course "Digitalization of Physical Processes"

## Prerequisites

You will need:

- `python^3.12` (see `pyproject.toml` for full version)
- `make`

## Quick start

Сheck your python version (should be 3.12)

1) Install uv
```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

or

```shell
pip install uv
```
2) Choose uv python version 3.12
```shell
uv python pin 3.12
```
3) Download all dependencies
```shell
uv sync --no-dev
```

4) Run streamlit server
```shell
make serve
```

or

```shell
uv run streamlit run ./src/main.py
```
