"""Norm Memory `M_N` (01_METHOD_SPEC.md #4.1): validated statute -> rule cache.

Keyed on `chunk_id` + a hash of the chunk text, so a corpus edit invalidates
stale entries automatically instead of silently reusing rules formalized
from different text.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

from ..schemas import DefeasibleRule, NormMemoryEntry, StatuteChunk
from ..symbolic.pythen_adapter import build_evaluator
from ..utils import read_json, sha256_text, write_json
from ..formalization.validators import is_exact_quote


@dataclass
class ValidationOutcome:
    valid: bool
    errors: list[str] = field(default_factory=list)


def key_for(chunk: StatuteChunk) -> str:
    return f"{chunk.chunk_id}:{sha256_text(chunk.text)}"


def validate_candidate(
    chunk: StatuteChunk,
    rules: list[DefeasibleRule],
    known_section_ids: set[str],
) -> ValidationOutcome:
    """Admission checks 1-4 of 01_METHOD_SPEC.md #4.1 (the dev/replay-score
    check and optional structural validation are batch-level concerns
    handled by memory/regression_gate.py).
    """
    errors: list[str] = []
    for rule in rules:
        if not is_exact_quote(rule.source_quote, chunk.text):
            errors.append(f"{rule.rule_id}: source_quote not exact in {chunk.chunk_id}")
        for ref in rule.references:
            if ref not in known_section_ids:
                errors.append(f"{rule.rule_id}: referenced section {ref!r} not known")
        if not rule.needs_review:
            try:
                build_evaluator([rule])
            except Exception as e:  # pythen.RuleValidationError or similar
                errors.append(f"{rule.rule_id}: not executable ({e})")
    return ValidationOutcome(valid=not errors, errors=errors)


@dataclass
class NormMemory:
    entries: dict[str, NormMemoryEntry] = field(default_factory=dict)

    def get(self, chunk: StatuteChunk) -> NormMemoryEntry | None:
        return self.entries.get(key_for(chunk))

    def make_entry(self, chunk: StatuteChunk, rules: list[DefeasibleRule], checkpoint: str | None = None) -> NormMemoryEntry:
        return NormMemoryEntry(
            key=key_for(chunk),
            chunk_id=chunk.chunk_id,
            text_hash=sha256_text(chunk.text),
            rules=rules,
            accepted_checkpoint=checkpoint,
        )

    def commit(self, entry: NormMemoryEntry) -> None:
        self.entries[entry.key] = entry

    def record_use(self, chunk: StatuteChunk, success: bool) -> None:
        entry = self.get(chunk)
        if entry is None:
            return
        if success:
            entry.success_count += 1
        else:
            entry.failure_count += 1

    def mark_unlabeled_entries(self, checkpoint: str) -> None:
        """Attach a provenance label only after a batch passes the gate.

        Candidate entries are created with ``accepted_checkpoint=None`` while
        replay is pending. A rejected proposal is discarded wholesale, so no
        rejected memory can acquire an accepted label.
        """
        for entry in self.entries.values():
            if entry.accepted_checkpoint is None:
                entry.accepted_checkpoint = checkpoint

    def clone(self) -> "NormMemory":
        return copy.deepcopy(self)

    def all_rules(self) -> list[DefeasibleRule]:
        return [r for entry in self.entries.values() for r in entry.rules]

    # -- persistence ----------------------------------------------------
    def to_dict(self) -> dict:
        return {k: v.model_dump() for k, v in self.entries.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "NormMemory":
        return cls(entries={k: NormMemoryEntry.model_validate(v) for k, v in data.items()})

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: str | Path) -> "NormMemory":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(read_json(p))
