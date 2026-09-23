#!/usr/bin/env python3
"""Milestone E: process evolution-train sequentially, growing Norm/Gap memory
under a regression gate, saving 0/25/50/75/100% checkpoints and evaluating
each checkpoint on DEV ONLY (05_EXPERIMENT_PLAN.md #5, 01_METHOD_SPEC.md #5).

Pass --no-regression-gate to produce the ungated checkpoint used by the
`no_regression_gate` ablation (05_EXPERIMENT_PLAN.md Experiment 2, A4).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_resources, cases_by_split, load_corpus_and_cases, load_frozen_prompt_variants, result_metadata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.evaluation.metrics import accuracy, retrieval_metrics_for_cases  # noqa: E402
from evodef.memory.gap_memory import GapMemory  # noqa: E402
from evodef.memory.norm_memory import NormMemory  # noqa: E402
from evodef.memory.regression_gate import accept_without_gate, run_regression_gate  # noqa: E402
from evodef.pipeline import run_case  # noqa: E402
from evodef.utils import load_config, resolve_path, set_global_seed, write_json, write_jsonl  # noqa: E402

EVODEF_METHOD_CFG = {
    "retrieval_query_variant": "issue_query",
    "exception_focused_query": True,
    "llm_rerank": True,
    "symbolic_reasoning": True,
    "targeted_retrieval": True,
    "use_norm_memory": True,
    "use_gap_memory": True,
    "regression_gate": True,
}


def evaluate_on_dev(config, chunks, prompt_variants, norm_memory: NormMemory, gap_memory: GapMemory, dev_cases) -> tuple[dict, list[dict]]:
    """Returns (summary_metrics, raw_predictions). The raw predictions are
    kept so callers can diff consecutive checkpoints case-by-case for the
    Regression Rate metric (06_EVALUATION_AND_STATS_SPEC.md #5), which needs
    per-case correctness, not just an aggregate accuracy number.
    """
    resources = build_resources(
        config, chunks,
        norm_memory=norm_memory, gap_memory=gap_memory,
        formalization_variant=prompt_variants["formalization_variant"],
        grounding_variant=prompt_variants["grounding_variant"],
        write_memory=False,
    )
    predictions = []
    diagnostics = {"formalization_calls": 0}
    for case in dev_cases:
        row = run_case(resources, case, "evodef_full", method_cfg_override=EVODEF_METHOD_CFG)
        predictions.append(row.model_dump())
        diagnostics["formalization_calls"] += row.formalization_calls
    cases_by_id = {c.case_id: c.model_dump() for c in dev_cases}
    retrieval = retrieval_metrics_for_cases(cases_by_id, predictions)
    summary = {
        "dev_accuracy": accuracy(predictions),
        "dev_recall_at_5": retrieval["recall_at_5"],
        "avg_formalization_calls": diagnostics["formalization_calls"] / len(dev_cases) if dev_cases else float("nan"),
        "norm_memory_size": len(norm_memory.entries),
        "gap_memory_size": len(gap_memory.entries),
    }
    return summary, predictions


def replay_score_fn(config, chunks, prompt_variants, replay_cases):
    def score(norm_memory: NormMemory, gap_memory: GapMemory) -> float:
        summary, _ = evaluate_on_dev(config, chunks, prompt_variants, norm_memory, gap_memory, replay_cases)
        return summary["dev_accuracy"]

    return score


def process_batch(config, chunks, prompt_variants, case_batch, current_norm, current_gap):
    """Runs one batch of evolution-train cases against CLONES of the current
    memories (so a rejected batch leaves current_norm/current_gap untouched),
    returning the proposed memories and the gap outcomes observed.
    """
    proposed_norm = current_norm.clone()
    proposed_gap = current_gap.clone()
    resources = build_resources(
        config, chunks,
        norm_memory=proposed_norm, gap_memory=proposed_gap,
        formalization_variant=prompt_variants["formalization_variant"],
        grounding_variant=prompt_variants["grounding_variant"],
        write_memory=True,
    )
    for case in case_batch:
        gap_trace: list[dict] = []
        row = run_case(resources, case, "evodef_full", method_cfg_override=EVODEF_METHOD_CFG, gap_trace=gap_trace)
        for chunk_id in [*row.retrieved_initial, *row.retrieved_targeted]:
            chunk = resources.corpus.get(chunk_id)
            if chunk is not None:
                proposed_norm.record_use(chunk, success=row.prediction == row.gold)
        gold_hit_in_targeted = any(cid in case.gold_chunk_ids for cid in row.retrieved_targeted)
        success = gold_hit_in_targeted or (row.prediction == row.gold and bool(row.retrieved_targeted))
        recall_gain = 1.0 if gold_hit_in_targeted else 0.0
        for gap_use in gap_trace:
            for template in gap_use["queries"]:
                proposed_gap.record_outcome(gap_use["signature"], template, success=success, recall_gain=recall_gain)
    proposed_gap.prune(
        min_trials=config["memory"]["gap_prune_min_trials"],
        min_success_rate=config["memory"]["gap_prune_min_success_rate"],
    )
    return proposed_norm, proposed_gap


def select_evolution_cases(cases, max_cases: int | None, seed: int):
    """Return a deterministic, label-balanced subset of evolution-train.

    This is a compute-budget control for evolution only.  DEV and the
    official TEST split remain unchanged; case identifiers are written to a
    manifest so the learned checkpoint is reproducible and auditable.
    """
    if max_cases is None or max_cases >= len(cases):
        return cases
    if max_cases < 2:
        raise ValueError("--max-evolution-cases must be at least 2")

    by_label: dict[str, list] = {}
    for case in cases:
        by_label.setdefault(case.gold_label, []).append(case)
    if len(by_label) != 2:
        raise ValueError("Expected a binary, label-balanced evolution split")

    rng = random.Random(seed)
    selected = []
    labels = sorted(by_label)
    base, remainder = divmod(max_cases, len(labels))
    for index, label in enumerate(labels):
        pool = list(by_label[label])
        rng.shuffle(pool)
        selected.extend(pool[: base + (1 if index < remainder else 0)])
    rng.shuffle(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-regression-gate", action="store_true", help="Accept every batch unconditionally (no_regression_gate ablation).")
    parser.add_argument(
        "--max-evolution-cases", type=int, default=None,
        help="Deterministically use this many label-balanced evolution-train cases; DEV and TEST are untouched.",
    )
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["seed"])
    chunks, cases = load_corpus_and_cases(config)
    prompt_variants = load_frozen_prompt_variants(config, require_frozen=True)

    all_evolution_cases = cases_by_split(cases, "evolution_train")
    evolution_cases = select_evolution_cases(all_evolution_cases, args.max_evolution_cases, config["seed"])
    dev_cases = cases_by_split(cases, "dev")
    print(f"Evolving over {len(evolution_cases)}/{len(all_evolution_cases)} evolution-train cases, replaying on {len(dev_cases)} dev cases")
    print(f"Frozen prompt variants: {prompt_variants}")
    print(f"Regression gate: {'DISABLED (ablation)' if args.no_regression_gate else 'enabled'}")

    checkpoint_pcts = config["evolution_checkpoints"]
    batch_size = config["memory"]["gate_batch_size"]
    epsilon = config["memory"]["regression_epsilon"]

    current_norm, current_gap = NormMemory(), GapMemory()
    checkpoints_dir = resolve_path(config, "checkpoints_dir")
    suffix = "_no_gate" if args.no_regression_gate else ""
    write_json(
        checkpoints_dir / f"evolution_sample_manifest{suffix}.json",
        {
            "selection": "deterministic_label_balanced_subset" if args.max_evolution_cases else "full_evolution_train",
            "seed": config["seed"],
            "available_evolution_train_cases": len(all_evolution_cases),
            "selected_evolution_train_cases": len(evolution_cases),
            "case_ids": [case.case_id for case in evolution_cases],
            "labels": {label: sum(case.gold_label == label for case in evolution_cases) for label in sorted({case.gold_label for case in evolution_cases})},
        },
    )

    def save_checkpoint(pct: int) -> None:
        current_norm.save(checkpoints_dir / f"norm_memory_{pct}pct{suffix}.json")
        current_gap.save(checkpoints_dir / f"gap_memory_{pct}pct{suffix}.json")

    curve_rows = []
    gate_log: list[dict] = []

    checkpoint_dev_predictions: dict[int, list[dict]] = {}

    def regression_rate(prev_predictions: list[dict], new_predictions: list[dict]) -> float:
        """RR = previously_correct_now_wrong / previously_correct (06_EVALUATION_AND_STATS_SPEC.md #5)."""
        prev_by_id = {p["case_id"]: p["prediction"] == p["gold"] for p in prev_predictions}
        new_by_id = {p["case_id"]: p["prediction"] == p["gold"] for p in new_predictions}
        previously_correct = [cid for cid, correct in prev_by_id.items() if correct]
        if not previously_correct:
            return float("nan")
        now_wrong = sum(1 for cid in previously_correct if not new_by_id.get(cid, False))
        return now_wrong / len(previously_correct)

    def record_curve_point(pct: int) -> None:
        summary, predictions = evaluate_on_dev(config, chunks, prompt_variants, current_norm, current_gap, dev_cases)
        prev_pcts = sorted(checkpoint_dev_predictions.keys())
        rr = regression_rate(checkpoint_dev_predictions[prev_pcts[-1]], predictions) if prev_pcts else float("nan")
        checkpoint_dev_predictions[pct] = predictions
        write_jsonl(checkpoints_dir / f"dev_predictions_{pct}pct{suffix}.jsonl", predictions)
        row = {
            "pct_evolution_train_processed": pct,
            "regression_rate_vs_prev_checkpoint": rr,
            **summary,
            **result_metadata(config, predictions, checkpoint_kind="ungated" if args.no_regression_gate else "gated"),
        }
        curve_rows.append(row)
        print(row)

    save_checkpoint(0)
    record_curve_point(0)

    total = len(evolution_cases)
    next_checkpoint_idx = 1  # index into checkpoint_pcts, 0% already done

    for start in range(0, total, batch_size):
        batch = evolution_cases[start : start + batch_size]
        proposed_norm, proposed_gap = process_batch(config, chunks, prompt_variants, batch, current_norm, current_gap)

        if args.no_regression_gate:
            gate_result = accept_without_gate(proposed_norm, proposed_gap)
        else:
            gate_result = run_regression_gate(
                current_norm, current_gap, proposed_norm, proposed_gap,
                replay_score_fn=replay_score_fn(config, chunks, prompt_variants, dev_cases),
                epsilon=epsilon,
            )
        print(f"batch [{start}:{start+len(batch)}] -> {gate_result.reason}")
        if gate_result.accepted:
            gate_result.norm_memory.mark_unlabeled_entries(
                f"evolution_batch_{start}_{start + len(batch)}"
            )
        gate_log.append(
            {
                "batch_start": start,
                "batch_end": start + len(batch),
                "accepted": gate_result.accepted,
                "score_before": gate_result.score_before,
                "score_after": gate_result.score_after,
                "reason": gate_result.reason,
                "candidate_norm_memory_size": len(proposed_norm.entries),
                "accepted_norm_memory_size": len(gate_result.norm_memory.entries),
            }
        )
        current_norm, current_gap = gate_result.norm_memory, gate_result.gap_memory

        processed = start + len(batch)
        pct_done = round(100 * processed / total)
        while next_checkpoint_idx < len(checkpoint_pcts) and pct_done >= checkpoint_pcts[next_checkpoint_idx]:
            pct = checkpoint_pcts[next_checkpoint_idx]
            save_checkpoint(pct)
            record_curve_point(pct)
            next_checkpoint_idx += 1

    # Ensure the 100% checkpoint is always written even if the loop's
    # rounding didn't land exactly on it.
    if next_checkpoint_idx < len(checkpoint_pcts):
        save_checkpoint(100)
        record_curve_point(100)

    curve_path = resolve_path(config, "results_dir") / f"evolution_curve{suffix}.csv"
    pd.DataFrame(curve_rows).to_csv(curve_path, index=False)
    write_json(resolve_path(config, "results_dir") / f"evolution_gate_log{suffix}.json", gate_log)
    print(f"Wrote {curve_path}")


if __name__ == "__main__":
    main()
