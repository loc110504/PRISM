#!/usr/bin/env bash
# Self-evolve EvoDef-RAG (M4/evodef_full) on a budgeted evolution-train
# subset, freeze the resulting checkpoint, then evaluate it on the official
# test split. Assumes data/processed and outputs/checkpoints/frozen_prompt_config.json
# already exist (scripts/00_prepare_sara.py + scripts/02_*) - it does not
# redo those stages, unlike scripts/01_run_full_experiment.sh.
#
# Override EVOLUTION_MAX_CASES to change the evolution-train subset size;
# the replay-gate batch size comes from configs/base.yaml's
# memory.gate_batch_size (currently 20).
#
# Example:
#   EVOLUTION_MAX_CASES=50 bash scripts/03_run_proposed_method.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
source .venv/bin/activate
export PYTHONUNBUFFERED=1

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

EVOLUTION_MAX_CASES="${EVOLUTION_MAX_CASES:-50}"

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

echo "== Step 1/3: self-evolving on ${EVOLUTION_MAX_CASES} evolution-train cases =="
python scripts/03_run_evolution.py --max-evolution-cases "$EVOLUTION_MAX_CASES" \
  |& tee "$LOG_DIR/03_run_evolution_${STAMP}.log"

echo "== Step 2/3: freezing the 100% checkpoint =="
python scripts/04_freeze_checkpoint.py \
  |& tee "$LOG_DIR/04_freeze_checkpoint_${STAMP}.log"

echo "== Step 3/3: evaluating evodef_full on the official test split =="
python scripts/05_run_test.py \
  |& tee "$LOG_DIR/05_run_test_${STAMP}.log"

echo "Done. Results appended to outputs/results/main_results.csv (method=evodef_full, split=test)."
