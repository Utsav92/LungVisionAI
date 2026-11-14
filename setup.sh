#!/usr/bin/env bash
# Reproducible environment setup (macOS, Apple Silicon). Requires uv (https://docs.astral.sh/uv/).
set -euo pipefail
cd "$(dirname "$0")"
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
uv pip install --python .venv/bin/python -e . --no-deps
.venv/bin/python scripts/env_report.py
