"""Shared setup helpers for scripts/01-11. Not a script itself.

Centralizing this avoids re-deriving the same PipelineResources wiring in
every script (corpus/case loading, index building, prompt loading).
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evodef.data.corpus import build_corpus  # noqa: E402
from evodef.memory.gap_memory import GapMemory  # noqa: E402
from evodef.memory.norm_memory import NormMemory  # noqa: E402
from evodef.ollama_client import OllamaClient  # noqa: E402
from evodef.pipeline import PipelineResources  # noqa: E402
from evodef.prompt_templates import load_prompt_dir  # noqa: E402
from evodef.retrieval.bm25 import BM25Index  # noqa: E402
from evodef.retrieval.dense import DenseIndex  # noqa: E402
from evodef.schemas import Case, StatuteChunk  # noqa: E402
from evodef.utils import load_config, read_json, read_jsonl, resolve_path, set_global_seed, sha256_file, sha256_json, write_jsonl  # noqa: E402


def load_corpus_and_cases(config: dict) -> tuple[list[StatuteChunk], list[Case]]:
    processed_dir = resolve_path(config, "processed_dir")
    chunks_raw = read_jsonl(processed_dir / "statutes.jsonl")
    cases_raw = read_jsonl(processed_dir / "cases.jsonl")
    chunks = [StatuteChunk.model_validate(c) for c in chunks_raw]
    cases = [Case.model_validate(c) for c in cases_raw]
    if not chunks or not cases:
        raise FileNotFoundError(
            "data/processed/{statutes,cases}.jsonl not found or empty - run scripts/00_prepare_sara.py first."
        )
    return chunks, cases


def build_resources(
    config: dict,
    chunks: list[StatuteChunk],
    *,
    generator_model: str | None = None,
    use_dense: bool = True,
    norm_memory: NormMemory | None = None,
    gap_memory: GapMemory | None = None,
    formalization_variant: dict | None = None,
    grounding_variant: dict | None = None,
    write_memory: bool = False,
) -> PipelineResources:
    prompts = load_prompt_dir(REPO_ROOT / config["paths"]["prompts_dir"])
    client = OllamaClient(
        # An explicit config value would otherwise suppress OllamaClient's
        # environment default, so resolve OLLAMA_HOST here as documented.
        base_url=os.environ.get("OLLAMA_HOST", config["ollama"]["base_url"]),
        max_schema_retries=config["ollama"]["max_schema_retries"],
        request_timeout_s=config["ollama"]["request_timeout_s"],
        max_network_retries=config["ollama"].get("max_network_retries", 2),
        network_retry_backoff_s=config["ollama"].get("network_retry_backoff_s", 5.0),
        provider=config.get("llm", {}).get("provider", "ollama"),
        embed_provider=config.get("llm", {}).get("embed_provider"),
    )
    corpus = {c.chunk_id: c for c in chunks}
    bm25 = BM25Index.build(chunks)

    dense_index = None
    if use_dense:
        embedder = config["models"]["embedder"]
        cache_path = resolve_path(config, "processed_dir") / "dense_cache.npy"
        dense_index = DenseIndex.build(
            chunks,
            embed_fn=lambda texts: client.embed(embedder, texts),
            cache_path=cache_path,
            cache_key=embedder,
        )

    return PipelineResources(
        corpus=corpus,
        bm25_index=bm25,
        dense_index=dense_index,
        client=client,
        generator_model=generator_model or config["models"]["generator"],
        prompts=prompts,
        config=config,
        norm_memory=norm_memory or NormMemory(),
        gap_memory=gap_memory or GapMemory(),
        formalization_variant=formalization_variant or {},
        grounding_variant=grounding_variant or {},
        write_memory=write_memory,
    )


def cases_by_split(cases: list[Case], split: str) -> list[Case]:
    return [c for c in cases if c.split == split]


def run_method_on_cases(resources: PipelineResources, method_name: str, cases: list[Case], method_cfg_override: dict | None = None) -> list[dict]:
    from evodef.pipeline import run_case

    predictions = []
    for i, case in enumerate(cases, start=1):
        row = run_case(resources, case, method_name, method_cfg_override=method_cfg_override)
        predictions.append(row.model_dump())
        if i % 10 == 0 or i == len(cases):
            print(f"  [{method_name}] {i}/{len(cases)} cases")
    return predictions


def summarize_main_result(method_name: str, split: str, predictions: list[dict], cases_by_id: dict[str, dict], config: dict) -> dict:
    from evodef.evaluation.metrics import accuracy, citation_coverage, macro_f1, retrieval_metrics_for_cases

    retrieval = retrieval_metrics_for_cases(cases_by_id, predictions, retrieved_field="retrieved_initial")
    avg_llm_calls = sum(p["llm_calls"] for p in predictions) / len(predictions) if predictions else float("nan")
    avg_latency = sum(p["latency_s"] for p in predictions) / len(predictions) if predictions else float("nan")

    metadata = result_metadata(config, predictions)
    return {
        "method": method_name,
        "split": split,
        "model": config["models"]["generator"],
        "embedder": config["models"]["embedder"],
        "seed": config["seed"],
        "n_cases": len(predictions),
        "accuracy": accuracy(predictions),
        "macro_f1": macro_f1(predictions, ["ENTAILMENT", "CONTRADICTION"]),
        "recall_at_5": retrieval["recall_at_5"],
        "citation_coverage": citation_coverage(predictions),
        "avg_llm_calls": avg_llm_calls,
        "avg_latency_s": avg_latency,
        **metadata,
    }


def result_metadata(config: dict, predictions: list[dict] | None = None, checkpoint_kind: str = "gated") -> dict:
    """Stable provenance fields required on derived result rows.

    The data hash is the processed-data manifest's source archive hash when
    available, otherwise a hash of the two processed JSONL inputs. Prompt and
    config hashes are read from the actual prediction rows so reports never
    silently describe a different run.
    """
    from datetime import datetime, timezone

    processed = resolve_path(config, "processed_dir")
    manifest_path = processed / "data_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        dataset_hash = manifest.get("source_sha256") or sha256_json(manifest)
    else:
        data_files = [processed / "statutes.jsonl", processed / "cases.jsonl"]
        dataset_hash = sha256_json([sha256_file(p) if p.exists() else None for p in data_files])

    prompt_hashes = sorted({sha256_json(p.get("prompt_hashes", {})) for p in (predictions or [])})
    config_hashes = sorted({p.get("config_hash") for p in (predictions or []) if p.get("config_hash")})
    checkpoint_manifest = resolve_path(config, "checkpoints_dir") / "frozen_checkpoint_manifest.json"
    checkpoint_hash = None
    if checkpoint_manifest.exists():
        frozen = read_json(checkpoint_manifest)
        manifest_key = "ungated_checkpoint" if checkpoint_kind == "ungated" else "gated_checkpoint"
        checkpoint_hash = frozen.get(manifest_key, {}).get("norm_memory_sha256")

    return {
        "model": config["models"]["generator"],
        "embedder": config["models"]["embedder"],
        "seed": config["seed"],
        "dataset_sha256": dataset_hash,
        "prompt_hashes_sha256": sha256_json(prompt_hashes) if prompt_hashes else None,
        "config_hashes_sha256": sha256_json(config_hashes) if config_hashes else None,
        "checkpoint_norm_memory_sha256": checkpoint_hash,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def save_predictions(config: dict, method_name: str, split: str, predictions: list[dict]) -> Path:
    path = resolve_path(config, "predictions_dir") / f"{method_name}_{split}.jsonl"
    write_jsonl(path, predictions)
    return path


def append_csv_row(path: Path, row: dict) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        existing = pd.read_csv(path)
        # replace any prior row for the same (method, split) pair so re-runs don't duplicate
        key_cols = [c for c in ("method", "split") if c in row and c in existing.columns]
        if key_cols:
            mask = pd.Series(True, index=existing.index)
            for c in key_cols:
                mask &= existing[c] == row[c]
            existing = existing[~mask]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(path, index=False)


def load_frozen_prompt_variants(config: dict, *, require_frozen: bool = False) -> dict:
    """Reads outputs/checkpoints/frozen_prompt_config.json written by
    scripts/02_tune_prompts_dev.py, falling back to the first grid entry
    (F1/Q2/G1) if tuning has not been run yet.
    """
    path = resolve_path(config, "checkpoints_dir") / "frozen_prompt_config.json"
    if path.exists():
        import json

        frozen = json.loads(path.read_text())
        expected_hashes = frozen.get("prompt_hashes")
        if expected_hashes:
            current_hashes = {
                name: template.hash
                for name, template in load_prompt_dir(REPO_ROOT / config["paths"]["prompts_dir"]).items()
            }
            if current_hashes != expected_hashes:
                raise RuntimeError(
                    "Prompt files differ from the frozen prompt configuration. "
                    "Re-run development tuning and freeze again; do not run evolution/test with altered prompts."
                )
        return frozen
    if require_frozen:
        raise FileNotFoundError(
            f"{path} is missing - run scripts/02_tune_prompts_dev.py before evolution or held-out evaluation."
        )
    variants = config["_experiments"]["prompt_variants"]
    return {
        "formalization_id": "F1",
        "formalization_variant": variants["formalization"]["F1"],
        "query_id": "Q2",
        "grounding_id": "G1",
        "grounding_variant": variants["grounding"]["G1"],
    }
