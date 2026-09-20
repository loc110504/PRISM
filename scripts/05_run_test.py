#!/usr/bin/env python3
"""Milestone F: run EvoDef-RAG (M4) on the official, untouched test split
using the frozen checkpoint + frozen prompts. Memory is read-only here
(`write_memory=False`) - see 08_AGENT_EXECUTION_CHECKLIST.md Milestone F's
pre-flight checks, which `scripts/10_leakage_audit.py` verifies.

Run scripts/04_freeze_checkpoint.py first.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    build_resources,
    cases_by_split,
    load_corpus_and_cases,
    load_frozen_prompt_variants,
    run_method_on_cases,
    save_predictions,
    summarize_main_result,
    append_csv_row,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.memory.gap_memory import GapMemory  # noqa: E402
from evodef.memory.norm_memory import NormMemory  # noqa: E402
from evodef.utils import load_config, resolve_path, set_global_seed  # noqa: E402


def main() -> None:
    config = load_config()
    set_global_seed(config["seed"])
    chunks, cases = load_corpus_and_cases(config)
    prompt_variants = load_frozen_prompt_variants(config, require_frozen=True)

    checkpoints_dir = resolve_path(config, "checkpoints_dir")
    norm_path = checkpoints_dir / "norm_memory_100pct.json"
    gap_path = checkpoints_dir / "gap_memory_100pct.json"
    if not norm_path.exists():
        raise FileNotFoundError(f"{norm_path} missing - run scripts/03_run_evolution.py and scripts/04_freeze_checkpoint.py first.")

    norm_memory = NormMemory.load(norm_path)
    gap_memory = GapMemory.load(gap_path)
    print(f"Loaded frozen checkpoint: {len(norm_memory.entries)} norm entries, {len(gap_memory.entries)} gap entries")

    test_cases = cases_by_split(cases, "test")
    print(f"Running evodef_full on {len(test_cases)} official test cases (READ-ONLY memory)")

    resources = build_resources(
        config, chunks,
        norm_memory=norm_memory, gap_memory=gap_memory,
        formalization_variant=prompt_variants["formalization_variant"],
        grounding_variant=prompt_variants["grounding_variant"],
        write_memory=False,  # frozen: never write during test
    )

    predictions = run_method_on_cases(resources, "evodef_full", test_cases)
    save_predictions(config, "evodef_full", "test", predictions)

    cases_by_id = {c.case_id: c.model_dump() for c in test_cases}
    row = summarize_main_result("evodef_full", "test", predictions, cases_by_id, config)
    append_csv_row(resolve_path(config, "results_dir") / "main_results.csv", row)
    print(row)


if __name__ == "__main__":
    main()
