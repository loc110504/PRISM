#!/usr/bin/env python3
"""Milestone A (cont'd): build BM25 + dense indices, report raw-assertion
retrieval metrics on dev as a sanity baseline before any prompt engineering.

Requires a reachable Ollama server for the embedder (set OLLAMA_HOST or edit
configs/base.yaml's ollama.base_url) - see README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import build_resources, cases_by_split, load_corpus_and_cases, result_metadata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.evaluation.metrics import retrieval_metrics_for_cases  # noqa: E402
from evodef.utils import load_config, resolve_path, set_global_seed  # noqa: E402


def main() -> None:
    config = load_config()
    set_global_seed(config["seed"])
    chunks, cases = load_corpus_and_cases(config)

    print(f"Building BM25 + dense index over {len(chunks)} statute chunks (embedder={config['models']['embedder']}) ...")
    resources = build_resources(config, chunks, use_dense=True)

    dev_cases = cases_by_split(cases, "dev")
    print(f"Evaluating raw-assertion retrieval on {len(dev_cases)} dev cases ...")

    predictions = []
    for case in dev_cases:
        query = f"{case.assertion} {case.facts_text}"
        fused = resources.bm25_index.search(query, top_k=config["retrieval"]["bm25_topk"])
        if resources.dense_index is not None:
            from evodef.retrieval.hybrid import rrf_fuse

            query_vec = resources.client.embed(config["models"]["embedder"], [query])[0]
            dense = resources.dense_index.search(query_vec, top_k=config["retrieval"]["dense_topk"])
            fused = rrf_fuse([fused, dense], k_rrf=config["retrieval"]["rrf_k"])
        predictions.append({"case_id": case.case_id, "retrieved_initial": [cid for cid, _ in fused[:20]]})

    cases_by_id = {c.case_id: c.model_dump() for c in dev_cases}
    metrics = retrieval_metrics_for_cases(cases_by_id, predictions)
    metrics_row = {
        "method": "raw_assertion_baseline",
        "split": "dev",
        "model": config["models"]["generator"],
        "embedder": config["models"]["embedder"],
        "seed": config["seed"],
        "n_cases": len(dev_cases),
        **metrics,
        **result_metadata(config, predictions),
    }

    results_path = resolve_path(config, "results_dir") / "retrieval_results.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics_row])
    if results_path.exists():
        df = pd.concat([pd.read_csv(results_path), df], ignore_index=True)
    df.to_csv(results_path, index=False)
    print(f"Wrote {results_path}")
    print(metrics_row)


if __name__ == "__main__":
    main()
