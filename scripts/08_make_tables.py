#!/usr/bin/env python3
"""Milestone H: build the paper tables + statistical tests from the raw
JSONL predictions written by scripts/05-07. Never hand-types a number
(03_IMPLEMENTATION_SPEC.md #10: "Figures and paper tables must be generated
from these files, never manually typed.").
"""
from __future__ import annotations

import sys
import re
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import cases_by_split, load_corpus_and_cases, result_metadata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.evaluation.error_analysis import build_error_breakdown  # noqa: E402
from evodef.evaluation.metrics import (  # noqa: E402
    counterfactual_validity,
    exception_coverage,
    proof_executable_rate,
    recall_at_k,
    retrieval_metrics_for_cases,
    source_link_coverage,
    undetermined_rate,
)
from evodef.evaluation.stats import mcnemar_test, paired_bootstrap_ci  # noqa: E402
from evodef.utils import load_config, read_jsonl, resolve_path, write_json  # noqa: E402

MAIN_METHODS = ["direct_llm", "vanilla_rag", "prompted_rag", "static_nesy_rag", "evodef_full"]


def _combined_evidence(prediction: dict) -> list[str]:
    """Stable initial-then-targeted evidence order, with no duplicate chunk."""
    return list(dict.fromkeys([
        *prediction.get("retrieved_initial", []),
        *prediction.get("retrieved_targeted", []),
    ]))


def _subgroup(case: dict, chunks_by_id: dict) -> str:
    gold_chunks = [chunks_by_id[cid] for cid in case.get("gold_chunk_ids", []) if cid in chunks_by_id]
    text = " ".join(chunk.text for chunk in gold_chunks).lower()
    if any(chunk.references for chunk in gold_chunks):
        return "cross_reference"
    if re.search(r"\b(unless|except|exception|other than)\b", text):
        return "exception_or_defeater"
    if len(gold_chunks) > 1:
        return "multi_hop"
    return "single_rule_direct"


def _load_predictions(config, method_name: str, split: str = "test") -> list[dict] | None:
    path = resolve_path(config, "predictions_dir") / f"{method_name}_{split}.jsonl"
    if not path.exists():
        print(f"  (missing {path}, skipping)")
        return None
    return read_jsonl(path)


def _paired(predictions_by_method: dict[str, list[dict]], a: str, b: str):
    """Aligns two methods' predictions by case_id, returns None if either is missing."""
    if a not in predictions_by_method or b not in predictions_by_method:
        return None
    by_id_a = {p["case_id"]: p for p in predictions_by_method[a]}
    by_id_b = {p["case_id"]: p for p in predictions_by_method[b]}
    common_ids = sorted(set(by_id_a) & set(by_id_b))
    if not common_ids:
        return None
    return [by_id_a[cid] for cid in common_ids], [by_id_b[cid] for cid in common_ids]


def main() -> None:
    config = load_config()
    chunks, cases = load_corpus_and_cases(config)
    test_cases = cases_by_split(cases, "test")
    cases_by_id = {c.case_id: c.model_dump() for c in test_cases}
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    predictions_by_method: dict[str, list[dict]] = {}
    for method in MAIN_METHODS:
        preds = _load_predictions(config, method)
        if preds is not None:
            predictions_by_method[method] = preds

    tables_dir = resolve_path(config, "tables_dir")
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tables 1 and 2: generated directly from run-result CSVs ----
    results_dir = resolve_path(config, "results_dir")
    main_results_path = results_dir / "main_results.csv"
    if main_results_path.exists():
        main = pd.read_csv(main_results_path)
        main = main[main["split"] == "test"]
        columns = [
            "method", "accuracy", "macro_f1", "recall_at_5", "citation_coverage",
            "avg_llm_calls", "avg_latency_s", "n_cases", "model", "embedder", "seed",
            "dataset_sha256", "prompt_hashes_sha256", "config_hashes_sha256",
            "checkpoint_norm_memory_sha256", "run_timestamp_utc",
        ]
        main.reindex(columns=[col for col in columns if col in main.columns]).to_csv(
            tables_dir / "table1_main_results.csv", index=False
        )
        print("Wrote table1_main_results.csv")

    ablations_path = results_dir / "ablation_results.csv"
    if ablations_path.exists():
        ablations = pd.read_csv(ablations_path)
        columns = [
            "ablation", "accuracy", "recall_at_5", "exception_error_count",
            "regression_rate_vs_gated_dev", "n_cases", "dataset_sha256",
            "prompt_hashes_sha256", "config_hashes_sha256",
            "checkpoint_norm_memory_sha256", "run_timestamp_utc",
        ]
        ablations.reindex(columns=[col for col in columns if col in ablations.columns]).to_csv(
            tables_dir / "table2_ablations.csv", index=False
        )
        print("Wrote table2_ablations.csv")

    # ---- Table 3: XAI (computed for static_nesy_rag and evodef_full) ----
    xai_rows = []
    for method in ["static_nesy_rag", "evodef_full"]:
        if method not in predictions_by_method:
            continue
        preds = predictions_by_method[method]
        xai_rows.append({
                "method": method,
                "proof_executable_rate": proof_executable_rate(preds),
                "source_link_coverage": source_link_coverage(preds),
                "counterfactual_validity": counterfactual_validity(preds),
                "exception_coverage": exception_coverage(preds),
                "undetermined_rate": undetermined_rate(preds),
                **result_metadata(config, preds),
            })
    if xai_rows:
        pd.DataFrame(xai_rows).to_csv(resolve_path(config, "results_dir") / "xai_results.csv", index=False)
        pd.DataFrame(xai_rows).to_csv(tables_dir / "table3_xai.csv", index=False)
        print(f"Wrote table3_xai.csv ({len(xai_rows)} rows)")

    # ---- Error breakdown (M2 vs M3 vs M4) ----
    error_rows = []
    for method in ["prompted_rag", "static_nesy_rag", "evodef_full"]:
        if method not in predictions_by_method:
            continue
        breakdown = build_error_breakdown(predictions_by_method[method], cases_by_id)
        error_rows.append({"method": method, **breakdown, **result_metadata(config, predictions_by_method[method])})
    if error_rows:
        pd.DataFrame(error_rows).to_csv(resolve_path(config, "results_dir") / "error_breakdown.csv", index=False)
        print(f"Wrote error_breakdown.csv ({len(error_rows)} rows)")

    # ---- Efficiency and targeted-retrieval evidence accounting ----
    efficiency_rows = []
    for method, preds in predictions_by_method.items():
        n = len(preds)
        if not n:
            continue
        final_evidence = [
            {**prediction, "retrieved_final_evidence": _combined_evidence(prediction)}
            for prediction in preds
        ]
        final_retrieval = retrieval_metrics_for_cases(
            cases_by_id, final_evidence, retrieved_field="retrieved_final_evidence"
        )
        final_recall_at_8 = [
            recall_at_k(prediction["retrieved_final_evidence"], cases_by_id[prediction["case_id"]]["gold_chunk_ids"], 8)
            for prediction in final_evidence
            if prediction["case_id"] in cases_by_id and cases_by_id[prediction["case_id"]].get("gold_chunk_ids")
        ]
        efficiency_rows.append({
            "method": method,
            "n_cases": n,
            "avg_llm_calls": sum(p.get("llm_calls", 0) for p in preds) / n,
            "avg_formalization_calls": sum(p.get("formalization_calls", 0) for p in preds) / n,
            "avg_latency_s": sum(p.get("latency_s", 0.0) for p in preds) / n,
            "memory_hit_rate": sum(p.get("memory_hits", 0) for p in preds) / n,
            "final_evidence_recall_at_5": final_retrieval["recall_at_5"],
            "final_evidence_recall_at_8": sum(final_recall_at_8) / len(final_recall_at_8) if final_recall_at_8 else float("nan"),
            **result_metadata(config, preds),
        })
    if efficiency_rows:
        pd.DataFrame(efficiency_rows).to_csv(results_dir / "efficiency_results.csv", index=False)
        print(f"Wrote efficiency_results.csv ({len(efficiency_rows)} rows)")

    # ---- Deterministic subgroup analysis: no LLM judge ----
    subgroup_rows = []
    groups = {case_id: _subgroup(case, chunks_by_id) for case_id, case in cases_by_id.items()}
    for subgroup in sorted(set(groups.values())):
        ids = {case_id for case_id, label in groups.items() if label == subgroup}
        row = {"subgroup": subgroup, "n_cases": len(ids)}
        for method in ["prompted_rag", "static_nesy_rag", "evodef_full"]:
            preds = [p for p in predictions_by_method.get(method, []) if p["case_id"] in ids]
            row[f"{method}_accuracy"] = sum(p["prediction"] == p["gold"] for p in preds) / len(preds) if preds else float("nan")
        subgroup_rows.append(row)
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(results_dir / "subgroup_results.csv", index=False)
        print(f"Wrote subgroup_results.csv ({len(subgroup_rows)} rows)")

    # ---- Statistical tests: M4 vs M3, M4 vs M2 ----
    stats_out = {}
    for baseline in ["static_nesy_rag", "prompted_rag"]:
        paired = _paired(predictions_by_method, "evodef_full", baseline)
        if paired is None:
            continue
        evodef_preds, baseline_preds = paired
        correct_evodef = [p["prediction"] == p["gold"] for p in evodef_preds]
        correct_baseline = [p["prediction"] == p["gold"] for p in baseline_preds]

        mcnemar = mcnemar_test(correct_baseline, correct_evodef)  # A=baseline, B=evodef_full

        cases_subset = {p["case_id"]: cases_by_id[p["case_id"]] for p in evodef_preds if p["case_id"] in cases_by_id}
        recall_evodef = [
            retrieval_metrics_for_cases({p["case_id"]: cases_by_id[p["case_id"]]}, [p])["recall_at_5"]
            for p in evodef_preds if p["case_id"] in cases_by_id
        ]
        recall_baseline = [
            retrieval_metrics_for_cases({p["case_id"]: cases_by_id[p["case_id"]]}, [p])["recall_at_5"]
            for p in baseline_preds if p["case_id"] in cases_by_id
        ]
        acc_ci = paired_bootstrap_ci(
            [float(c) for c in correct_baseline], [float(c) for c in correct_evodef],
            n_resamples=config["statistics"]["paired_bootstrap_resamples"],
            confidence_level=config["statistics"]["confidence_level"],
            seed=config["seed"],
        )
        recall_ci = None
        if len(recall_evodef) == len(recall_baseline) and recall_evodef:
            import math

            pairs = [(a, b) for a, b in zip(recall_baseline, recall_evodef) if not (math.isnan(a) or math.isnan(b))]
            if pairs:
                a_vals, b_vals = zip(*pairs)
                recall_ci = paired_bootstrap_ci(
                    list(a_vals), list(b_vals),
                    n_resamples=config["statistics"]["paired_bootstrap_resamples"],
                    confidence_level=config["statistics"]["confidence_level"],
                    seed=config["seed"],
                )

        stats_out[f"evodef_full_vs_{baseline}"] = {
            "mcnemar": mcnemar,
            "accuracy_diff_bootstrap_ci": acc_ci,
            "recall_at_5_diff_bootstrap_ci": recall_ci,
        }

    if stats_out:
        stats_path = resolve_path(config, "results_dir") / "statistical_tests.json"
        write_json(stats_path, stats_out)
        print(f"Wrote {stats_path}")

    print("Done. See outputs/results/*.csv and outputs/tables/*.csv for everything computed.")


if __name__ == "__main__":
    main()
