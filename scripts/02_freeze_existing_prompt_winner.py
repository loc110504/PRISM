#!/usr/bin/env python3
"""Freeze the highest-scoring row already present in the DEV tuning grid.

This helper performs no model calls.  It is intended for resuming a run whose
prompt grid has already been evaluated, including a deliberately stopped grid
when ``--allow-incomplete-grid`` is supplied.  The resulting provenance file
records the grid coverage so a paper does not imply an exhaustive search.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.prompt_templates import load_prompt_dir  # noqa: E402
from evodef.utils import load_config, resolve_path, write_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-incomplete-grid", action="store_true",
        help="Freeze the best completed DEV row even when grid combinations are missing.",
    )
    args = parser.parse_args()

    config = load_config()
    variants = config["_experiments"]["prompt_variants"]
    grid = [
        (formalization_id, query_id, grounding_id)
        for formalization_id in config["prompt_tuning"]["formalization"]
        for query_id in config["prompt_tuning"]["query"]
        for grounding_id in config["prompt_tuning"]["grounding"]
    ]
    results_path = resolve_path(config, "results_dir") / "prompt_tuning_grid.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"{results_path} missing; run scripts/02_tune_prompts_dev.py first.")

    rows = pd.read_csv(results_path)
    required = {"formalization_id", "query_id", "grounding_id", "objective"}
    missing_columns = required - set(rows.columns)
    if missing_columns:
        raise ValueError(f"Prompt-tuning grid is missing columns: {sorted(missing_columns)}")
    rows = rows.dropna(subset=["objective"])
    if rows.empty:
        raise ValueError("Prompt-tuning grid contains no scored rows")

    completed = set(rows[["formalization_id", "query_id", "grounding_id"]].itertuples(index=False, name=None))
    missing = sorted(set(grid) - completed)
    if missing and not args.allow_incomplete_grid:
        raise RuntimeError(
            f"Prompt grid is incomplete ({len(completed)}/{len(grid)}); missing={missing}. "
            "Run scripts/02_tune_prompts_dev.py or pass --allow-incomplete-grid explicitly."
        )

    best = rows.loc[rows["objective"].idxmax()].to_dict()
    formalization_id = str(best["formalization_id"])
    query_id = str(best["query_id"])
    grounding_id = str(best["grounding_id"])
    frozen = {
        "formalization_id": formalization_id,
        "formalization_variant": variants["formalization"][formalization_id],
        "query_id": query_id,
        "grounding_id": grounding_id,
        "grounding_variant": variants["grounding"][grounding_id],
        "objective_score": float(best["objective"]),
        "selection_source": "highest completed DEV-only row in outputs/results/prompt_tuning_grid.csv",
        "prompt_grid_completed": len(completed),
        "prompt_grid_total": len(grid),
        "prompt_grid_missing": [list(combo) for combo in missing],
        "prompt_hashes": {
            name: template.hash
            for name, template in load_prompt_dir(REPO_ROOT / config["paths"]["prompts_dir"]).items()
        },
    }
    path = resolve_path(config, "checkpoints_dir") / "frozen_prompt_config.json"
    write_json(path, frozen)
    print(f"Froze {formalization_id}/{query_id}/{grounding_id} (objective={frozen['objective_score']:.6f}) to {path}")
    if missing:
        print(f"WARNING: selected from incomplete grid ({len(completed)}/{len(grid)}); missing={missing}")


if __name__ == "__main__":
    main()
