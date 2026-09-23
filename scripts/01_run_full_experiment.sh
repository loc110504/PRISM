#!/usr/bin/env bash
# Full reproducible experiment workflow. Run this on the server that can reach
# Ollama; it intentionally performs the real model calls required for results.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Python loads .env for every stage. Source it here as well so this shell can
# decide whether an Ollama endpoint is required.
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

# Optional fast path for a previously prepared corpus and an existing DEV
# prompt-tuning grid.  Defaults preserve the original exhaustive workflow.
# Example:
#   SKIP_DATA_PREPARATION=1 USE_EXISTING_PROMPT_BEST=1 EVOLUTION_MAX_CASES=50 \
#     OLLAMA_HOST=http://host:11434 bash scripts/01_run_full_experiment.sh
SKIP_DATA_PREPARATION="${SKIP_DATA_PREPARATION:-0}"
USE_EXISTING_PROMPT_BEST="${USE_EXISTING_PROMPT_BEST:-0}"
EVOLUTION_MAX_CASES="${EVOLUTION_MAX_CASES:-}"
EVOLUTION_ARGS=()
if [[ -n "$EVOLUTION_MAX_CASES" ]]; then
  EVOLUTION_ARGS=(--max-evolution-cases "$EVOLUTION_MAX_CASES")
fi

if [[ "${LLM_PROVIDER:-ollama}" != "openai" && -z "${OLLAMA_HOST:-}" ]]; then
  echo "Set OLLAMA_HOST to the reachable Ollama endpoint, e.g. http://host:11434" >&2
  exit 2
fi

# The only stage that downloads/rebuilds the official corpus. Set
# SKIP_DATA_PREPARATION=1 only when data/processed is already known to match
# the desired SARA archive/seed.
if [[ "$SKIP_DATA_PREPARATION" != "1" ]]; then
  python scripts/00_prepare_sara.py
fi
python scripts/01_build_index.py

# Development-only selection, then evolution on evolution_train only.
if [[ "$USE_EXISTING_PROMPT_BEST" == "1" ]]; then
  python scripts/02_freeze_existing_prompt_winner.py --allow-incomplete-grid
else
  python scripts/02_tune_prompts_dev.py
fi
python scripts/03_run_evolution.py "${EVOLUTION_ARGS[@]}"
python scripts/03_run_evolution.py --no-regression-gate "${EVOLUTION_ARGS[@]}"
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
