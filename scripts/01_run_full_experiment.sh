#!/usr/bin/env bash
# Full reproducible experiment workflow. Run this on the server that can reach
# Ollama; it intentionally performs the real model calls required for results.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

if [[ -z "${OLLAMA_HOST:-}" ]]; then
  echo "Set OLLAMA_HOST to the reachable Ollama endpoint, e.g. http://host:11434" >&2
  exit 2
fi

# The only stage that downloads/rebuilds the official corpus. Omit it only if
# data/processed is already known to match the desired SARA archive/seed.
python scripts/00_prepare_sara.py
python scripts/01_build_index.py

# Development-only selection, then evolution on evolution_train only.
python scripts/02_tune_prompts_dev.py
python scripts/03_run_evolution.py
python scripts/03_run_evolution.py --no-regression-gate
python scripts/04_freeze_checkpoint.py

# Preflight must pass before the official held-out split is evaluated.
python scripts/10_leakage_audit.py
python scripts/06_run_baselines.py
python scripts/05_run_test.py
python scripts/07_run_ablations.py

# Derived metrics, figures, final leakage check, and paper-facing report.
python scripts/10_leakage_audit.py
python scripts/08_make_tables.py
python scripts/09_make_figures.py
python scripts/11_generate_report.py

echo "Complete. Read outputs/PAPER_RESULTS_REPORT.md; do not copy numbers from console logs."
