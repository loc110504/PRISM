"""Gap Policy Memory `M_G` (01_METHOD_SPEC.md #4.2): reusable gap-to-query strategies.

Frozen and read-only at test time; only `scripts/03_run_evolution.py` writes
to it, and only on evolution-train.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas import GapPolicyEntry, GapRecord
from ..utils import read_json, write_json

_ARGS_RE = re.compile(r"\([^)]*\)")


def _stem(predicate: str) -> str:
    """Strip call arguments and normalize a predicate to a stable stem,
    e.g. `qualifying_child_of_other_taxpayer(Person,Year)` -> `qualifying_child_of_other_taxpayer`.
    """
    return _ARGS_RE.sub("", predicate).strip().lower()


def generalize_signature(gap: GapRecord, source_family: str | None = None) -> str:
    """01_METHOD_SPEC.md #4.2 example: `UNRESOLVED_EXCEPTION|qualifying_child|tax_dependency`."""
    parts = [gap.gap_type, _stem(gap.target_predicate)]
    if source_family:
        parts.append(source_family)
    return "|".join(parts)


@dataclass
class GapMemory:
    entries: dict[str, GapPolicyEntry] = field(default_factory=dict)

    def get(self, signature: str) -> GapPolicyEntry | None:
        return self.entries.get(signature)

    def record_outcome(
        self,
        signature: str,
        query_template: str,
        success: bool,
        recall_gain: float = 0.0,
    ) -> None:
        entry = self.entries.get(signature)
        if entry is None:
            entry = GapPolicyEntry(signature=signature)
            self.entries[signature] = entry
        if query_template not in entry.query_templates:
            entry.query_templates.append(query_template)
        if success:
            entry.success_count += 1
        else:
            entry.failure_count += 1
        trials = entry.success_count + entry.failure_count
        # running mean of recall gain across all trials (success or not)
        entry.avg_recall_gain = ((entry.avg_recall_gain * (trials - 1)) + recall_gain) / trials

    def prune(self, min_trials: int = 3, min_success_rate: float = 0.25) -> list[str]:
        """01_METHOD_SPEC.md #4.2: "Prune after n>=3 trials if success rate < 0.25."

        Returns the list of pruned signatures.
        """
        pruned = []
        for signature, entry in list(self.entries.items()):
            trials = entry.success_count + entry.failure_count
            if trials >= min_trials and (entry.success_count / trials) < min_success_rate:
                del self.entries[signature]
                pruned.append(signature)
        return pruned

    def clone(self) -> "GapMemory":
        return copy.deepcopy(self)

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {k: v.model_dump() for k, v in self.entries.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "GapMemory":
        return cls(entries={k: GapPolicyEntry.model_validate(v) for k, v in data.items()})

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "GapMemory":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(read_json(p))
