#!/usr/bin/env bash
# Create a reproducible Python environment and run only deterministic checks.
# This script does NOT send any request to Ollama.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

cd "$REPO_DIR"
if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest tests/ -q
python -m py_compile scripts/*.py $(find src -name '*.py' -print)

echo "Setup complete. Configure OLLAMA_HOST on the remote server before running 01_run_full_experiment.sh."
