#!/usr/bin/env python3
"""Milestone G: 05_EXPERIMENT_PLAN.md Experiment 2 - the five-row core
ablation, on the official test split.

A0 evodef_full, A1 no_proof_gap, A2 no_exception_query, A3 no_memory all
reuse the SAME gated 100% checkpoint (they only toggle inference-time flags
per configs/experiments.yaml). A4 no_regression_gate needs its OWN checkpoint,
evolved without the gate:

    python scripts/03_run_evolution.py --no-regression-gate
    python scripts/07_run_ablations.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    append_csv_row,
    build_resources,
    cases_by_split,
    load_corpus_and_cases,
    load_frozen_prompt_variants,
    result_metadata,
    run_method_on_cases,
    save_predictions,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.evaluation.error_analysis import classify_error  # noqa: E402
from evodef.evaluation.metrics import accuracy, retrieval_metrics_for_cases  # noqa: E402
from evodef.memory.gap_memory import GapMemory  # noqa: E402
from evodef.memory.norm_memory import NormMemory  # noqa: E402
from evodef.utils import load_config, resolve_path, set_global_seed  # noqa: E402

ABLATIONS = ["evodef_full", "no_proof_gap", "no_exception_query", "no_memory", "no_regression_gate"]


def _load_checkpoint(checkpoints_dir: Path, suffix: str) -> tuple[NormMemory, GapMemory]:
    norm = NormMemory.load(checkpoints_dir / f"norm_memory_100pct{suffix}.json")
    gap = GapMemory.load(checkpoints_dir / f"gap_memory_100pct{suffix}.json")
    return norm, gap


def main() -> None:
    config = load_config()
    set_global_seed(config["seed"])
    chunks, cases = load_corpus_and_cases(config)
    prompt_variants = load_frozen_prompt_variants(config, require_frozen=True)
    test_cases = cases_by_split(cases, "test")
    cases_by_id = {c.case_id: c.model_dump() for c in test_cases}

    checkpoints_dir = resolve_path(config, "checkpoints_dir")
    gated_norm, gated_gap = _load_checkpoint(checkpoints_dir, "")

    prior_dev_predictions_path = checkpoints_dir / "dev_predictions_100pct.jsonl"
    ungated_dev_predictions_path = checkpoints_dir / "dev_predictions_100pct_no_gate.jsonl"

    rows = []
    for ablation_name in ABLATIONS:
        if ablation_name == "no_regression_gate":
            norm_path = checkpoints_dir / "norm_memory_100pct_no_gate.json"
            if not norm_path.exists():
                print(f"SKIP no_regression_gate: {norm_path} missing - run `python scripts/03_run_evolution.py --no-regression-gate` first.")
                continue
            norm_memory, gap_memory = _load_checkpoint(checkpoints_dir, "_no_gate")
        else:
            norm_memory, gap_memory = gated_norm, gated_gap

        print(f"Running ablation '{ablation_name}' on {len(test_cases)} test cases ...")
        resources = build_resources(
            config, chunks,
            norm_memory=norm_memory, gap_memory=gap_memory,
            formalization_variant=prompt_variants["formalization_variant"],
            grounding_variant=prompt_variants["grounding_variant"],
            write_memory=False,
        )
        predictions = run_method_on_cases(resources, ablation_name, test_cases)
        save_predictions(config, ablation_name, "test", predictions)

        retrieval = retrieval_metrics_for_cases(cases_by_id, predictions)
        exception_error_count = sum(
            1 for p in predictions if classify_error(p, cases_by_id.get(p["case_id"], {})) == "EXCEPTION_ERROR"
        )

        regression_rate = float("nan")
        if ablation_name == "no_regression_gate" and ungated_dev_predictions_path.exists() and prior_dev_predictions_path.exists():
            from evodef.utils import read_jsonl

            gated_dev = {p["case_id"]: p["prediction"] == p["gold"] for p in read_jsonl(prior_dev_predictions_path)}
            ungated_dev = {p["case_id"]: p["prediction"] == p["gold"] for p in read_jsonl(ungated_dev_predictions_path)}
            previously_correct = [cid for cid, ok in gated_dev.items() if ok]
            if previously_correct:
                now_wrong = sum(1 for cid in previously_correct if not ungated_dev.get(cid, False))
                regression_rate = now_wrong / len(previously_correct)

        row = {
            "ablation": ablation_name,
            "split": "test",
            "n_cases": len(predictions),
            "accuracy": accuracy(predictions),
            "recall_at_5": retrieval["recall_at_5"],
            "exception_error_count": exception_error_count,
            "regression_rate_vs_gated_dev": regression_rate,
            **result_metadata(config, predictions, checkpoint_kind="ungated" if ablation_name == "no_regression_gate" else "gated"),
        }
        rows.append(row)
        print(row)
        append_csv_row(resolve_path(config, "results_dir") / "ablation_results.csv", {**row, "method": ablation_name})

    print(f"Wrote {resolve_path(config, 'results_dir') / 'ablation_results.csv'}")


if __name__ == "__main__":
    main()
