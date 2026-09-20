from __future__ import annotations

from evodef.memory.gap_memory import GapMemory, generalize_signature
from evodef.memory.norm_memory import NormMemory, key_for, validate_candidate
from evodef.memory.regression_gate import accept_without_gate, run_regression_gate
from evodef.schemas import DefeasibleRule, GapRecord, StatuteChunk


def _rule(rule_id="r1", p="p", conditions=("a",), exceptions=(), source_id="s1", quote="the source text", refs=()):
    return DefeasibleRule(
        rule_id=rule_id, p=p, op="ALL", conditions=list(conditions), exceptions=list(exceptions),
        references=list(refs), source_id=source_id, source_quote=quote, confidence=0.9, needs_review=False,
    )


CHUNK = StatuteChunk(chunk_id="sec_1", section_id="1", text="the source text", source_file="f")


class TestNormMemory:
    def test_validate_candidate_accepts_good_rule(self):
        result = validate_candidate(CHUNK, [_rule()], known_section_ids=set())
        assert result.valid

    def test_validate_candidate_rejects_bad_quote(self):
        bad = _rule(quote="not in chunk")
        result = validate_candidate(CHUNK, [bad], known_section_ids=set())
        assert not result.valid

    def test_validate_candidate_rejects_unknown_reference(self):
        rule = _rule(refs=["999"])
        result = validate_candidate(CHUNK, [rule], known_section_ids={"1"})
        assert not result.valid
        result_ok = validate_candidate(CHUNK, [rule], known_section_ids={"1", "999"})
        assert result_ok.valid

    def test_commit_and_get_roundtrip(self):
        mem = NormMemory()
        entry = mem.make_entry(CHUNK, [_rule()])
        mem.commit(entry)
        fetched = mem.get(CHUNK)
        assert fetched is not None
        assert fetched.key == key_for(CHUNK)

    def test_stale_key_on_text_change(self):
        mem = NormMemory()
        mem.commit(mem.make_entry(CHUNK, [_rule()]))
        changed_chunk = StatuteChunk(chunk_id="sec_1", section_id="1", text="different text now", source_file="f")
        assert mem.get(changed_chunk) is None

    def test_save_load_roundtrip(self, tmp_path):
        mem = NormMemory()
        mem.commit(mem.make_entry(CHUNK, [_rule()]))
        path = tmp_path / "norm.json"
        mem.save(path)
        loaded = NormMemory.load(path)
        assert loaded.get(CHUNK) is not None

    def test_load_missing_file_returns_empty(self, tmp_path):
        assert NormMemory.load(tmp_path / "missing.json").entries == {}

    def test_clone_is_independent(self):
        mem = NormMemory()
        mem.commit(mem.make_entry(CHUNK, [_rule()]))
        clone = mem.clone()
        clone.record_use(CHUNK, success=True)
        assert mem.get(CHUNK).success_count == 0
        assert clone.get(CHUNK).success_count == 1

    def test_checkpoint_label_is_added_only_to_pending_entries(self):
        mem = NormMemory()
        mem.commit(mem.make_entry(CHUNK, [_rule()]))
        mem.mark_unlabeled_entries("evolution_batch_0_5")
        assert mem.get(CHUNK).accepted_checkpoint == "evolution_batch_0_5"


class TestGapMemory:
    def test_generalize_signature_strips_arguments(self):
        gap = GapRecord(gap_type="UNRESOLVED_EXCEPTION", target_predicate="qualifying_child(Person,Year)")
        sig = generalize_signature(gap, source_family="152")
        assert sig == "UNRESOLVED_EXCEPTION|qualifying_child|152"

    def test_record_outcome_accumulates_counts(self):
        mem = GapMemory()
        mem.record_outcome("sig1", "template A", success=True, recall_gain=0.5)
        mem.record_outcome("sig1", "template B", success=False, recall_gain=0.0)
        entry = mem.get("sig1")
        assert entry.success_count == 1
        assert entry.failure_count == 1
        assert entry.query_templates == ["template A", "template B"]

    def test_prune_removes_low_success_entries(self):
        mem = GapMemory()
        for _ in range(4):
            mem.record_outcome("bad_sig", "t", success=False)
        mem.record_outcome("good_sig", "t", success=True)
        pruned = mem.prune(min_trials=3, min_success_rate=0.25)
        assert "bad_sig" in pruned
        assert mem.get("bad_sig") is None
        assert mem.get("good_sig") is not None

    def test_prune_keeps_entries_below_min_trials(self):
        mem = GapMemory()
        mem.record_outcome("sig1", "t", success=False)
        mem.record_outcome("sig1", "t", success=False)
        assert mem.prune(min_trials=3) == []

    def test_save_load_roundtrip(self, tmp_path):
        mem = GapMemory()
        mem.record_outcome("sig1", "t", success=True)
        path = tmp_path / "gap.json"
        mem.save(path)
        loaded = GapMemory.load(path)
        assert loaded.get("sig1").success_count == 1


class TestRegressionGate:
    def test_accepts_when_no_regression(self):
        result = run_regression_gate(
            NormMemory(), GapMemory(), NormMemory(), GapMemory(),
            replay_score_fn=lambda n, g: 0.8, epsilon=0.0,
        )
        assert result.accepted

    def test_rolls_back_on_regression(self):
        current_norm, current_gap = NormMemory(), GapMemory()
        proposed_norm, proposed_gap = NormMemory(), GapMemory()
        scores = iter([0.8, 0.5])
        result = run_regression_gate(
            current_norm, current_gap, proposed_norm, proposed_gap,
            replay_score_fn=lambda n, g: next(scores), epsilon=0.0,
        )
        assert not result.accepted
        assert result.norm_memory is current_norm

    def test_epsilon_tolerance(self):
        scores = iter([0.80, 0.79])
        result = run_regression_gate(
            NormMemory(), GapMemory(), NormMemory(), GapMemory(),
            replay_score_fn=lambda n, g: next(scores), epsilon=0.02,
        )
        assert result.accepted

    def test_accept_without_gate_always_accepts(self):
        result = accept_without_gate(NormMemory(), GapMemory())
        assert result.accepted
