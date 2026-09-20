"""Pydantic data contracts shared across the pipeline.

These mirror `01_METHOD_SPEC.md` (core representations), `02_DATA_AND_SPLITS_SPEC.md`
(corpus/case objects) and `schemas/*.json` (LLM structured-output contracts).
Pydantic models are used both to validate Ollama structured output and as the
in-memory representation passed between pipeline stages.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Corpus / case objects (02_DATA_AND_SPLITS_SPEC.md)
# --------------------------------------------------------------------------


class StatuteChunk(BaseModel):
    chunk_id: str
    section_id: str
    text: str
    parent_id: Optional[str] = None
    references: list[str] = Field(default_factory=list)
    source_file: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None


class Case(BaseModel):
    case_id: str
    split: Literal["evolution_train", "dev", "test"]
    facts_text: str
    assertion: str
    gold_label: Literal["ENTAILMENT", "CONTRADICTION"]
    gold_chunk_ids: list[str] = Field(default_factory=list)
    gold_prolog_test: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Defeasible rule representation (01_METHOD_SPEC.md #2.2, schemas/rule_schema.json)
# --------------------------------------------------------------------------


class DefeasibleRule(BaseModel):
    rule_id: str
    p: str
    op: Literal["ALL", "ANY"]
    conditions: list[str] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    source_id: str
    source_quote: str
    confidence: float = Field(ge=0.0, le=1.0)
    needs_review: bool = False

    def to_pythen_dict(self) -> dict[str, Any]:
        """Strip provenance metadata; PYTHEN's evaluator only consumes p/op/conditions/exceptions.

        Predicate names are normalized (arguments dropped) - see
        `utils.normalize_predicate` for why.
        """
        from .utils import normalize_predicate

        return {
            "p": normalize_predicate(self.p),
            "op": self.op,
            "conditions": [normalize_predicate(c) for c in self.conditions],
            "exceptions": [normalize_predicate(e) for e in self.exceptions],
        }


class RuleExtractionOutput(BaseModel):
    """Structured output of prompts/formalize_rule.txt (schemas/rule_schema.json)."""

    rules: list[DefeasibleRule] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Grounded facts (01_METHOD_SPEC.md #2.3, schemas/fact_schema.json)
# --------------------------------------------------------------------------


class GroundedFact(BaseModel):
    predicate: str
    status: Literal["TRUE", "FALSE", "UNKNOWN"]
    source_quote: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class FactGroundingOutput(BaseModel):
    facts: list[GroundedFact] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Retrieval query / rerank outputs (schemas/query_schema.json)
# --------------------------------------------------------------------------


class QueryOutput(BaseModel):
    queries: list[str] = Field(default_factory=list, max_length=3)


RerankLabel = Literal["DIRECT", "EXCEPTION", "DEFINITION", "REFERENCE", "IRRELEVANT"]

RERANK_LABEL_PRIORITY: dict[str, int] = {
    "DIRECT": 0,
    "EXCEPTION": 1,
    "DEFINITION": 2,
    "REFERENCE": 3,
    "IRRELEVANT": 4,
}


class RerankItem(BaseModel):
    chunk_id: str
    label: RerankLabel


class RerankOutput(BaseModel):
    items: list[RerankItem] = Field(default_factory=list)


class DirectAnswerOutput(BaseModel):
    """Structured output of prompts/direct_answer.txt (M0/M1/M2 baselines)."""

    decision: Literal["ENTAILMENT", "CONTRADICTION"]
    rationale: str = ""


# --------------------------------------------------------------------------
# Proof trace (01_METHOD_SPEC.md #2.4)
# --------------------------------------------------------------------------

ProofStatus = Literal["true", "false", "unknown", "defeated"]


class ProofNode(BaseModel):
    predicate: str
    status: ProofStatus
    via_rule_id: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)
    children: list["ProofNode"] = Field(default_factory=list)
    defeating_exception: Optional[str] = None


ProofNode.model_rebuild()


# --------------------------------------------------------------------------
# Typed proof gap (01_METHOD_SPEC.md #3 Phase E)
# --------------------------------------------------------------------------

GapType = Literal[
    "MISSING_SUPPORT",
    "UNRESOLVED_EXCEPTION",
    "MISSING_REFERENCE",
    "FORMALIZATION_UNCERTAIN",
    "CONFLICT",
]


class GapRecord(BaseModel):
    gap_type: GapType
    target_predicate: str
    current_rule_id: Optional[str] = None
    source_id: Optional[str] = None
    query_terms: list[str] = Field(default_factory=list)
    priority: int = 1


# --------------------------------------------------------------------------
# Proof certificate (01_METHOD_SPEC.md #6)
# --------------------------------------------------------------------------


class ExceptionCheck(BaseModel):
    predicate: str
    status: ProofStatus


class Counterfactual(BaseModel):
    intervention: str
    predicted_effect: str
    verified_by_reexecution: bool


class ProofCertificate(BaseModel):
    decision: Literal["ENTAILED", "NOT_ENTAILED", "UNDETERMINED"]
    target: str
    supporting_rules: list[str] = Field(default_factory=list)
    supporting_sources: list[str] = Field(default_factory=list)
    satisfied_conditions: list[str] = Field(default_factory=list)
    missing_conditions: list[str] = Field(default_factory=list)
    exceptions_checked: list[ExceptionCheck] = Field(default_factory=list)
    proof_valid: bool = False
    counterfactuals: list[Counterfactual] = Field(default_factory=list)
    root: Optional[ProofNode] = None


# --------------------------------------------------------------------------
# Run-state prediction row (03_IMPLEMENTATION_SPEC.md #7)
# --------------------------------------------------------------------------


class PredictionRow(BaseModel):
    case_id: str
    method: str
    config_hash: str
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    model: str
    retrieved_initial: list[str] = Field(default_factory=list)
    retrieved_targeted: list[str] = Field(default_factory=list)
    rules_used: list[str] = Field(default_factory=list)
    proof: Optional[dict[str, Any]] = None
    prediction: str
    gold: str
    latency_s: float = 0.0
    llm_calls: int = 0
    formalization_calls: int = 0
    memory_hits: int = 0
    undetermined: bool = False
    errors: list[str] = Field(default_factory=list)
    # Compact, chronological audit trail for inference figures. Raw LLM
    # prompts/responses are intentionally excluded to keep JSONL manageable.
    inference_trace: list[dict[str, Any]] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Memory value objects (01_METHOD_SPEC.md #4)
# --------------------------------------------------------------------------


class NormMemoryEntry(BaseModel):
    key: str  # chunk_id + text hash
    chunk_id: str
    text_hash: str
    rules: list[DefeasibleRule]
    schema_version: str = "1.0"
    success_count: int = 0
    failure_count: int = 0
    accepted_checkpoint: Optional[str] = None
    structurally_validated: bool = False


class GapPolicyEntry(BaseModel):
    signature: str  # e.g. "UNRESOLVED_EXCEPTION|qualifying_child|tax_dependency"
    query_templates: list[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    avg_recall_gain: float = 0.0
