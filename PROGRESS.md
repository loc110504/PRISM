# EvoDef-RAG build progress

Tracks `EvoDefRAG_AGENT_SPECS/08_AGENT_EXECUTION_CHECKLIST.md` so any agent
picking this up (human or Claude) can see exactly what's done, what's next,
and what decisions were already made. Update this file every time a task
below is finished. See `README.md` for how to actually run things (needs a
remote Ollama server - not available in the dev sandbox this was built in).

## Milestone A - Data and deterministic baselines
- [x] SARA loader implemented and run for real (`scripts/00_prepare_sara.py`) - 186 statute chunks, 276 binary cases (156 evolution_train / 20 dev / 100 test), all gold_chunk_ids resolve.
- [x] Official test split preserved untouched.
- [x] Evolution-train/dev split from official train (grouped by statute family, seed 20260921).
- [x] Subsection-level statute corpus builder (`src/evodef/data/corpus.py`) - two real parser bugs found/fixed against actual SARA files (see PROGRESS notes below).
- [x] `gold_chunk_ids` derived from the Prolog `% Test` goal, never exposed to inference code.
- [x] BM25 implemented (`src/evodef/retrieval/bm25.py`).
- [x] Dense embedding cache via Ollama implemented (`src/evodef/retrieval/dense.py`) - untested against a real server (no Ollama in this sandbox).
- [x] RRF hybrid retrieval implemented (`src/evodef/retrieval/hybrid.py`).
- [x] `scripts/01_build_index.py` written (produces `retrieval_results.csv` raw-assertion baseline) - **not run**, needs a real Ollama server for embeddings.

## Milestone B - Ollama structured prompts
- [x] `OllamaClient` implemented with retries/hashing/latency (`src/evodef/ollama_client.py`), tested against a scripted fake backend.
- [x] Pydantic schemas implemented (`src/evodef/schemas.py`).
- [x] Prompt files versioned under `prompts/` (spec's 5 + `rerank_chunks.txt` + `direct_answer.txt`, added for the reranker and M0-M2 baselines which the spec didn't ship templates for).
- [x] Formalization + fact grounding implemented and unit-tested with a fake backend (`src/evodef/formalization/`).
- [ ] **Not done**: the real smoke test ("run formalization prompt on 10 dev statute chunks, verify >=90% schema-valid") requires a live Ollama server - only fake-backend tests exist so far.

## Milestone C - Symbolic reasoning
- [x] PYTHEN installed and used for real (`pythen==0.1.0`).
- [x] Metadata-stripping adapter (`src/evodef/symbolic/pythen_adapter.py`).
- [x] Traced 3-valued evaluator (`src/evodef/symbolic/traced_evaluator.py`).
- [x] Boolean equivalence verified by unit test across all fact-combinations for two rule sets (`tests/test_symbolic.py::TestPythenEquivalence`) - real PYTHEN, not mocked.
- [x] Typed proof-gap analyzer (`src/evodef/symbolic/gap_analyzer.py`).
- [x] Deterministic counterfactual re-execution (`src/evodef/symbolic/counterfactual.py`).
- [x] Unit tests pass; a hand-built qualifying-relative/qualifying-child example produces the expected proof certificate end-to-end (`tests/test_pipeline.py::TestEvoDefFull`).

## Milestone D - Static pipeline
- [x] `static_nesy_rag` implemented in `src/evodef/pipeline.py` and integration-tested with a fake backend.
- [ ] **Not run** on real dev data (needs Ollama).
- [x] `scripts/02_tune_prompts_dev.py` written (F0-F3 x Q0-Q2 x G0-G1 grid, composite objective, freezes winner to `outputs/checkpoints/frozen_prompt_config.json`) - **not run**.

## Milestone E - EvoDef memory
- [x] Norm Memory implemented (`src/evodef/memory/norm_memory.py`).
- [x] Gap Policy Memory implemented (`src/evodef/memory/gap_memory.py`).
- [x] Targeted gap-query retrieval implemented in `pipeline.py` (`_targeted_retrieval`).
- [x] Regression/replay gate implemented (`src/evodef/memory/regression_gate.py`).
- [x] `scripts/03_run_evolution.py` written, smoke-tested end-to-end with a fake backend (checkpoints at 0/25/50/75/100%, regression gate, gap-memory outcome recording all confirmed working) - **not run for real**, needs Ollama. `--no-regression-gate` flag implemented for the ablation checkpoint.
- [x] 0/25/50/75/100% checkpoints + `evolution_curve.csv` generation logic confirmed working in the smoke test above; real files not committed (smoke-test artifacts were deleted, not real results).

## Milestone F - Test execution
- [x] `scripts/10_leakage_audit.py` implemented (prompt/test-memory/config checks; run it before and after held-out evaluation).
- [x] `scripts/05_run_test.py` implemented (EvoDef-RAG on official test with read-only frozen memory).
- [x] `scripts/06_run_baselines.py` implemented (Direct/Vanilla/Prompted/Static on official test).
- [ ] No real held-out run has been performed in this environment; it requires the remote Ollama server.

## Milestone G - Ablations
- [x] `scripts/07_run_ablations.py` implemented. `no_regression_gate` needs its own evolution run (`03_run_evolution.py --no-regression-gate`) producing a separate checkpoint before evaluation.
- [ ] No real ablation run has been performed in this environment.

## Milestone H - XAI/stats/figures
- [x] PER, SLC, CFV, exception-coverage implemented and unit-tested (`src/evodef/evaluation/metrics.py`).
- [x] McNemar + paired bootstrap CI implemented and unit-tested (`src/evodef/evaluation/stats.py`).
- [x] Error taxonomy implemented and unit-tested (`src/evodef/evaluation/error_analysis.py`).
- [x] `scripts/08_make_tables.py` implemented (Tables 1--3, XAI, errors, efficiency, subgroups, and statistics from raw predictions).
- [x] `scripts/09_make_figures.py` implemented (architecture Mermaid, evolution plot, and proof case-study artifacts).
- [ ] Derived artifacts await real prediction files.

## Milestone I - Paper-ready report
- [x] `scripts/11_generate_report.py` implemented. It creates `outputs/PAPER_RESULTS_REPORT.md` from artifacts only and explicitly marks unavailable results; it does not insert placeholder numbers.
- [ ] Run it after the remote experiment workflow creates real artifacts.

## Test suite
- `cd evodef_rag && source .venv/bin/activate && python -m pytest tests/ -q` → **130 passed** after the implementation/review pass. Re-run before trusting this number.

## Key decisions already made (don't re-litigate these without reason)
1. **Predicate matching is argument-insensitive.** PYTHEN does exact string matching, no unification. See `utils.normalize_predicate` and its callers. This is documented in `pipeline.py`'s module docstring.
2. **Proof target selection**: the proposition of the rule from the highest-ranked retrieved+formalized chunk (`pipeline._select_target`). Not derived from gold labels.
3. **UNDETERMINED maps to CONTRADICTION** for the binary SARA metric (fixed rule per 01_METHOD_SPEC.md #8); `PredictionRow.undetermined` tracks the true abstention rate separately.
4. **Exception coverage (EC) metric** is computed over "reasoned cases" (>=1 rule applied) as the denominator, not gold-annotated exception presence (would need SARA-v2 structure parsing, out of scope - see `metrics.py::exception_coverage` docstring).
5. Two prompts not in the original spec bundle were added: `prompts/rerank_chunks.txt` (LLM reranker) and `prompts/direct_answer.txt` (M0/M1/M2 baselines need some prompt to actually answer with).
6. No real Ollama server is available in the dev sandbox (per user instruction: "just implement, do not test ollama server, I use it in my remote server"). Everything LLM-touching is tested with a scripted fake backend in `tests/conftest.py::FakeOllamaBackend`. Point `OLLAMA_HOST` (or edit `configs/base.yaml`) at your server to actually run scripts 01+.
7. `scripts/01_run_full_experiment.sh` is the required remote execution order; `scripts/02_refresh_paper_artifacts.sh` regenerates only deterministic reports/figures and makes no Ollama calls.
8. Each prediction now includes a compact `inference_trace`. Use `scripts/12_export_inference_trace.py --case-id <ID>` after a run to export its JSON timeline and Mermaid inference diagram without another model call.
