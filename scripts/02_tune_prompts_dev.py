#!/usr/bin/env python3
"""Experiment 0 (05_EXPERIMENT_PLAN.md #2): grid-search F0-F3 x Q0-Q2 x G0-G1
on DEV ONLY, score by the composite objective in 04_PROMPT_ENGINEERING_SPEC.md #6,
and freeze the winning combination + prompt hashes for every later script.

Never reads test labels (00_MASTER_SPEC.md #9 / 02_DATA_AND_SPLITS_SPEC.md #9).
"""
from __future__ import annotations

import sys
from itertools import product
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, build_resources, cases_by_split, load_corpus_and_cases  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.evaluation.metrics import accuracy, retrieval_metrics_for_cases  # noqa: E402
from evodef.pipeline import run_case  # noqa: E402
from evodef.prompt_templates import load_prompt_dir  # noqa: E402
from evodef.utils import load_config, resolve_path, set_global_seed, write_json  # noqa: E402


def _method_cfg_for(query_id: str, exception_focused: bool) -> dict:
    return {
        "retrieval_query_variant": "raw" if query_id == "Q0" else "issue_query",
        "exception_focused_query": exception_focused,
        "llm_rerank": query_id != "Q0",
        "symbolic_reasoning": True,
        "targeted_retrieval": False,
        "use_norm_memory": False,
        "use_gap_memory": False,
        "regression_gate": False,
    }


def main() -> None:
    config = load_config()
    set_global_seed(config["seed"])
    chunks, cases = load_corpus_and_cases(config)
    dev_cases = cases_by_split(cases, "dev")
    cases_by_id = {c.case_id: c.model_dump() for c in dev_cases}

    variants = config["_experiments"]["prompt_variants"]
    objective_weights = config["prompt_tuning"]["objective"]

    grid = list(
        product(config["prompt_tuning"]["formalization"], config["prompt_tuning"]["query"], config["prompt_tuning"]["grounding"])
    )

    results_path = resolve_path(config, "results_dir") / "prompt_tuning_grid.csv"

    # Resume support: a full grid run issues thousands of sequential LLM
    # calls, so even with network retries (see ollama_client.py) a very
    # long run can still be interrupted. Re-reading already-completed
    # combinations from a prior partial run means a restart never repeats
    # (and re-pays for) work that already succeeded. Delete
    # results_path to force a full re-run.
    rows: list[dict] = []
    done: set[tuple[str, str, str]] = set()
    if results_path.exists():
        rows = pd.read_csv(results_path).to_dict("records")
        done = {(r["formalization_id"], r["query_id"], r["grounding_id"]) for r in rows}
        print(f"Resuming from {results_path}: {len(done)}/{len(grid)} combinations already completed")

    remaining = [combo for combo in grid if combo not in done]
    print(f"Tuning grid: {len(grid)} combinations x {len(dev_cases)} dev cases ({len(remaining)} remaining)")

    best = max(rows, key=lambda r: r["objective"]) if rows else None

    for f_id, q_id, g_id in remaining:
        formalization_variant = variants["formalization"][f_id]
        grounding_variant = variants["grounding"][g_id]
        exception_focused = variants["query"][q_id].get("exception_focused_query", False)
        method_cfg = _method_cfg_for(q_id, exception_focused)

        diagnostics: dict = {}
        resources = build_resources(
            config, chunks,
            formalization_variant=formalization_variant,
            grounding_variant=grounding_variant,
        )
        resources.diagnostics = diagnostics

        predictions = []
        for case in dev_cases:
            row = run_case(resources, case, method_name=f"tune_{f_id}_{q_id}_{g_id}", method_cfg_override=method_cfg)
            predictions.append(row.model_dump())

        acc = accuracy(predictions)
        retrieval = retrieval_metrics_for_cases(cases_by_id, predictions)
        schema_valid_rate = diagnostics.get("schema_valid", 0) / diagnostics["chunks_formalized"] if diagnostics.get("chunks_formalized") else float("nan")
        total_rules = diagnostics.get("rules_accepted", 0) + diagnostics.get("rules_rejected", 0)
        source_quote_validity = diagnostics.get("rules_accepted", 0) / total_rules if total_rules else float("nan")

        objective = (
            objective_weights["downstream_accuracy"] * (acc if acc == acc else 0.0)
            + objective_weights["retrieval_recall_at_5"] * (retrieval["recall_at_5"] if retrieval["recall_at_5"] == retrieval["recall_at_5"] else 0.0)
            + objective_weights["schema_valid_rate"] * (schema_valid_rate if schema_valid_rate == schema_valid_rate else 0.0)
            + objective_weights["source_quote_validity"] * (source_quote_validity if source_quote_validity == source_quote_validity else 0.0)
        )

        row = {
            "formalization_id": f_id, "query_id": q_id, "grounding_id": g_id,
            "dev_accuracy": acc, "dev_recall_at_5": retrieval["recall_at_5"],
            "schema_valid_rate": schema_valid_rate, "source_quote_validity": source_quote_validity,
            "objective": objective,
        }
        rows.append(row)
        print(row)

        if best is None or objective > best["objective"]:
            best = row

        # Checkpoint after every combination so a failure later in the grid
        # (e.g. a case that exhausts every network retry) never discards
        # already-completed work.
        pd.DataFrame(rows).to_csv(results_path, index=False)

    print(f"Wrote {results_path}")

    frozen = {
        "formalization_id": best["formalization_id"],
        "formalization_variant": variants["formalization"][best["formalization_id"]],
        "query_id": best["query_id"],
        "grounding_id": best["grounding_id"],
        "grounding_variant": variants["grounding"][best["grounding_id"]],
        "objective_score": best["objective"],
        "prompt_hashes": {
            name: template.hash
            for name, template in load_prompt_dir(REPO_ROOT / config["paths"]["prompts_dir"]).items()
        },
    }
    frozen_path = resolve_path(config, "checkpoints_dir") / "frozen_prompt_config.json"
    write_json(frozen_path, frozen)
    print(f"Froze winning combination and prompt hashes to {frozen_path}: {frozen}")


if __name__ == "__main__":
    main()
