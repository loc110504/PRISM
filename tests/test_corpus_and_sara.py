"""Milestone A acceptance tests, run against the REAL downloaded SARA archive
under data/raw/ (03_IMPLEMENTATION_SPEC.md #9 "statute chunk parser").

These are skipped (not failed) if `scripts/00_prepare_sara.py` has not been
run yet, so the suite still passes on a machine with no network access.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from evodef.data.corpus import build_corpus, parse_statute_file
from evodef.data.sara import (
    _predicate_to_chunk_id,
    load_raw_binary_cases,
    make_evolution_dev_split,
    parse_case_file,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SARA_DIR = REPO_ROOT / "data" / "raw" / "sara_extracted" / "sara"

pytestmark = pytest.mark.skipif(not SARA_DIR.exists(), reason="run scripts/00_prepare_sara.py first")


class TestStatuteParser:
    def test_builds_nested_section_ids(self):
        chunks = build_corpus(SARA_DIR / "statutes" / "source")
        by_id = {c.section_id: c for c in chunks}
        assert "152(d)(1)" in by_id
        assert "152(d)(1)(A)" in by_id
        assert by_id["152(d)(1)(A)"].parent_id == by_id["152(d)(1)"].chunk_id

    def test_consecutive_siblings_without_blank_line_separator(self):
        # section1(a)'s rate brackets (i)-(v) are consecutive lines with no
        # blank line between them - a real bug caught during development.
        chunks = build_corpus(SARA_DIR / "statutes" / "source")
        ids = {c.section_id for c in chunks}
        for label in ["i", "ii", "iii", "iv", "v"]:
            assert f"1(a)({label})" in ids

    def test_header_line_with_no_trailing_text(self):
        # section3306(c)(10) is a bare "    (10)" header line with nothing
        # after it - another real bug caught during development.
        chunks = build_corpus(SARA_DIR / "statutes" / "source")
        ids = {c.section_id for c in chunks}
        assert "3306(c)(10)" in ids
        assert "3306(c)(10)(A)" in ids
        assert "3306(c)(10)(A)(i)" in ids

    def test_no_subsections_whole_file_is_one_chunk(self):
        _, _, chunks = parse_statute_file(SARA_DIR / "statutes" / "source" / "section3301")
        assert len(chunks) == 1
        assert chunks[0].section_id == "3301"

    def test_references_extracted(self):
        chunks = build_corpus(SARA_DIR / "statutes" / "source")
        by_id = {c.section_id: c for c in chunks}
        assert "7703" in by_id["1(a)(1)"].references

    def test_every_chunk_id_unique(self):
        chunks = build_corpus(SARA_DIR / "statutes" / "source")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestCaseParser:
    def test_parses_entailment_case(self):
        raw = parse_case_file(SARA_DIR / "cases" / "s151_a_pos.pl")
        assert raw["gold_label"] == "ENTAILMENT"
        assert "4000" in raw["assertion"]
        assert raw["gold_chunk_ids"] == ["sec_151_a"]

    def test_parses_contradiction_case_with_negated_goal(self):
        raw = parse_case_file(SARA_DIR / "cases" / "s3306_a_2_B_neg.pl")
        assert raw["gold_label"] == "CONTRADICTION"
        assert raw["gold_chunk_ids"] == ["sec_3306_a_2_B"]

    def test_numeric_case_returns_none(self):
        assert parse_case_file(SARA_DIR / "cases" / "tax_case_1.pl") is None

    def test_predicate_to_chunk_id(self):
        assert _predicate_to_chunk_id("s151_a") == "sec_151_a"
        assert _predicate_to_chunk_id("s3306_a_2_B") == "sec_3306_a_2_B"


class TestSplits:
    def test_all_gold_chunk_ids_resolve_in_corpus(self):
        train, test = load_raw_binary_cases(SARA_DIR)
        chunk_ids = {c.chunk_id for c in build_corpus(SARA_DIR / "statutes" / "source")}
        for raw in list(train.values()) + list(test.values()):
            for gid in raw["gold_chunk_ids"]:
                assert gid in chunk_ids, gid

    def test_official_test_split_untouched_and_balanced(self):
        _, test = load_raw_binary_cases(SARA_DIR)
        labels = [raw["gold_label"] for raw in test.values()]
        assert len(labels) == 100
        assert labels.count("ENTAILMENT") == labels.count("CONTRADICTION")

    def test_evolution_dev_split_keeps_families_together(self):
        train, _ = load_raw_binary_cases(SARA_DIR)
        assignment = make_evolution_dev_split(train, seed=20260921)
        from evodef.data.sara import _family_of

        family_splits: dict[str, set[str]] = {}
        for case_id, raw in train.items():
            family_splits.setdefault(_family_of(raw), set()).add(assignment[case_id])
        for family, splits in family_splits.items():
            assert len(splits) == 1, f"family {family} straddles {splits}"

    def test_split_is_deterministic_across_calls(self):
        train, _ = load_raw_binary_cases(SARA_DIR)
        a = make_evolution_dev_split(train, seed=20260921)
        b = make_evolution_dev_split(train, seed=20260921)
        assert a == b

    def test_different_seed_can_change_assignment(self):
        train, _ = load_raw_binary_cases(SARA_DIR)
        a = make_evolution_dev_split(train, seed=20260921)
        b = make_evolution_dev_split(train, seed=1)
        assert a != b
