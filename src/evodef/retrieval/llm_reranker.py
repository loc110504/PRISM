"""LLM reranker: labels top-10 fused candidates, keeps top-5 by label priority.

(03_IMPLEMENTATION_SPEC.md #4 "LLM reranker", 04_PROMPT_ENGINEERING_SPEC.md #8)
"""
from __future__ import annotations

from ..ollama_client import LLMCallRecord, OllamaClient
from ..prompt_templates import PromptTemplate
from ..schemas import RERANK_LABEL_PRIORITY, RerankOutput, StatuteChunk


def rerank(
    client: OllamaClient,
    model: str,
    template: PromptTemplate,
    facts_text: str,
    assertion: str,
    candidates: list[StatuteChunk],
    keep_top_k: int = 5,
    temperature: float = 0,
) -> tuple[list[str], RerankOutput | None, LLMCallRecord]:
    """Returns (kept_chunk_ids_in_priority_order, parsed_output_or_None, call_record).

    On schema failure, falls back to the original fused order truncated to
    `keep_top_k` (never crash a run on one malformed output, per
    03_IMPLEMENTATION_SPEC.md #8).
    """
    passages = "\n\n".join(f"[{c.chunk_id}] ({c.section_id}) {c.text}" for c in candidates)
    user = template.render_user(facts_text=facts_text, assertion=assertion, candidate_passages=passages)
    parsed, record = client.chat_json(model, template.system, user, RerankOutput, temperature=temperature)

    if parsed is None or not parsed.items:
        return [c.chunk_id for c in candidates[:keep_top_k]], parsed, record

    label_by_id = {item.chunk_id: item.label for item in parsed.items}
    known_ids = {c.chunk_id for c in candidates}
    ordered = sorted(
        (cid for cid in label_by_id if cid in known_ids),
        key=lambda cid: (RERANK_LABEL_PRIORITY.get(label_by_id[cid], 99), cid),
    )
    # Any candidate the model failed to label is treated as IRRELEVANT and
    # appended after all labeled ones, preserving fused order among unlabeled ties.
    unlabeled = [c.chunk_id for c in candidates if c.chunk_id not in label_by_id]
    return (ordered + unlabeled)[:keep_top_k], parsed, record
