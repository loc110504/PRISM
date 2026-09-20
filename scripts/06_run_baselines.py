#!/usr/bin/env python3
"""Milestone F: run the four baselines (Direct-LLM, Vanilla-RAG, Prompted-RAG,
Static-NeSy-RAG) on the official test split, using the SAME generator/initial
retriever as EvoDef-RAG (00_MASTER_SPEC.md #8) and the frozen prompts, but no
persistent evolution memory (Static-NeSy-RAG's "deterministic cache disabled
for fairness" per 05_EXPERIMENT_PLAN.md M3 - `use_norm_memory=False` in
configs/experiments.yaml already encodes this).
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
from evodef.utils import load_config, resolve_path, set_global_seed  # noqa: E402

BASELINE_METHODS = ["direct_llm", "vanilla_rag", "prompted_rag", "static_nesy_rag"]


def main() -> None:
    config = load_config()
    set_global_seed(config["seed"])
    chunks, cases = load_corpus_and_cases(config)
    prompt_variants = load_frozen_prompt_variants(config, require_frozen=True)
    test_cases = cases_by_split(cases, "test")
    cases_by_id = {c.case_id: c.model_dump() for c in test_cases}

    resources = build_resources(
        config, chunks,
        formalization_variant=prompt_variants["formalization_variant"],
        grounding_variant=prompt_variants["grounding_variant"],
        write_memory=False,
    )

    results_path = resolve_path(config, "results_dir") / "main_results.csv"
    for method_name in BASELINE_METHODS:
        print(f"Running {method_name} on {len(test_cases)} official test cases ...")
        predictions = run_method_on_cases(resources, method_name, test_cases)
        save_predictions(config, method_name, "test", predictions)
        row = summarize_main_result(method_name, "test", predictions, cases_by_id, config)
        append_csv_row(results_path, row)
        print(row)


if __name__ == "__main__":
    main()
