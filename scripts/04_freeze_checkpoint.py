#!/usr/bin/env python3
"""Milestone E/F boundary: freeze the 100% evolution checkpoint (and, if
present, the ungated one) for use by scripts/05-07. Records file hashes so
every later script can prove it used exactly this checkpoint
(03_IMPLEMENTATION_SPEC.md #7 "config_hash"/"prompt_hashes";
02_DATA_AND_SPLITS_SPEC.md #9 leakage audit consumes this file too).

Selection is unconditional (the 100% checkpoint), never chosen by looking at
test performance (00_MASTER_SPEC.md #9).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_frozen_prompt_variants  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evodef.utils import load_config, resolve_path, sha256_file, write_json  # noqa: E402


def _hash_if_exists(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def main() -> None:
    config = load_config()
    checkpoints_dir = resolve_path(config, "checkpoints_dir")
    prompt_variants = load_frozen_prompt_variants(config, require_frozen=True)

    gated_norm = checkpoints_dir / "norm_memory_100pct.json"
    gated_gap = checkpoints_dir / "gap_memory_100pct.json"
    ungated_norm = checkpoints_dir / "norm_memory_100pct_no_gate.json"
    ungated_gap = checkpoints_dir / "gap_memory_100pct_no_gate.json"

    for path in [gated_norm, gated_gap]:
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run scripts/03_run_evolution.py first.")

    manifest = {
        "gated_checkpoint": {
            "norm_memory_path": str(gated_norm.relative_to(resolve_path(config, "checkpoints_dir").parent.parent)),
            "norm_memory_sha256": _hash_if_exists(gated_norm),
            "gap_memory_path": str(gated_gap.relative_to(resolve_path(config, "checkpoints_dir").parent.parent)),
            "gap_memory_sha256": _hash_if_exists(gated_gap),
        },
        "ungated_checkpoint": {
            "norm_memory_sha256": _hash_if_exists(ungated_norm),
            "gap_memory_sha256": _hash_if_exists(ungated_gap),
            "available": ungated_norm.exists() and ungated_gap.exists(),
        },
        "frozen_prompt_variants": prompt_variants,
        "note": "Selected unconditionally as the 100% evolution checkpoint; never chosen by test performance.",
    }

    out_path = checkpoints_dir / "frozen_checkpoint_manifest.json"
    write_json(out_path, manifest)
    print(f"Wrote {out_path}")
    print(manifest)


if __name__ == "__main__":
    main()
