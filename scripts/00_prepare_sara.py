#!/usr/bin/env python3
"""Milestone A: download SARA, build the statute corpus and case splits.

Produces:
  data/processed/statutes.jsonl
  data/processed/cases.jsonl
  data/processed/data_manifest.json   (URL, sha256, counts - 02_DATA_AND_SPLITS_SPEC.md #1)

Does not call Ollama. Safe to run on any machine with network access.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evodef.data.corpus import build_corpus
from evodef.data.sara import build_cases, config_hash, download_sara
from evodef.utils import load_config, resolve_path, set_global_seed, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    config = load_config()
    set_global_seed(config["seed"])

    raw_dir = resolve_path(config, "raw_dir")
    processed_dir = resolve_path(config, "processed_dir")

    print(f"Downloading SARA to {raw_dir} ...")
    info = download_sara(raw_dir, force=args.force_download)
    print(f"  url={info.url}")
    print(f"  sha256={info.sha256}")

    statutes_dir = info.extracted_dir / "statutes" / "source"
    print(f"Parsing statute corpus from {statutes_dir} ...")
    chunks = build_corpus(statutes_dir)
    print(f"  {len(chunks)} statute chunks from {len(list(statutes_dir.glob('section*')))} source files")

    seed = config["seed"]
    evo_fraction = config["splits"]["evolution_train_fraction"]
    print(f"Building case splits (seed={seed}, evolution_train_fraction={evo_fraction}) ...")
    cases = build_cases(info.extracted_dir, seed=seed, evolution_train_fraction=evo_fraction)

    split_counts = Counter(c.split for c in cases)
    label_counts = Counter(c.gold_label for c in cases)

    chunk_ids = {c.chunk_id for c in chunks}
    unresolved_gold = sorted(
        {gid for c in cases for gid in c.gold_chunk_ids if gid not in chunk_ids}
    )
    if unresolved_gold:
        print(f"  WARNING: {len(unresolved_gold)} gold_chunk_ids do not match any corpus chunk: {unresolved_gold}")

    statutes_path = processed_dir / "statutes.jsonl"
    cases_path = processed_dir / "cases.jsonl"
    manifest_path = processed_dir / "data_manifest.json"

    write_jsonl(statutes_path, [c.model_dump() for c in chunks])
    write_jsonl(cases_path, [c.model_dump() for c in cases])
    write_json(
        manifest_path,
        {
            "source_url": info.url,
            "source_sha256": info.sha256,
            "seed": seed,
            "evolution_train_fraction": evo_fraction,
            "config_hash": config_hash(seed, evo_fraction),
            "n_statute_chunks": len(chunks),
            "n_statute_source_files": len(list(statutes_dir.glob("section*"))),
            "n_cases_total": len(cases),
            "split_counts": dict(split_counts),
            "label_counts": dict(label_counts),
            "n_unresolved_gold_chunk_ids": len(unresolved_gold),
            "unresolved_gold_chunk_ids": unresolved_gold,
        },
    )

    print(f"Wrote {statutes_path}")
    print(f"Wrote {cases_path}")
    print(f"Wrote {manifest_path}")
    print(f"Split counts: {dict(split_counts)}")
    print(f"Label counts: {dict(label_counts)}")


if __name__ == "__main__":
    main()
