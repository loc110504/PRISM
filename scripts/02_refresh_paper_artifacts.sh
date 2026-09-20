#!/usr/bin/env bash
# Recompute deterministic tables, figures, audit, and report from existing
# raw predictions. This makes no Ollama calls.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/10_leakage_audit.py
python scripts/08_make_tables.py
python scripts/09_make_figures.py
python scripts/11_generate_report.py
