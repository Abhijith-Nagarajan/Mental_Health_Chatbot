# Improvements — Making this RAG 2026-Ready

A phased roadmap. Each phase is sized to ship in a weekend (or an evening) and has something concrete to show at the end. Order matters — later phases build on earlier ones.

---

## Phase 0 — Fix what's broken (1 evening)

Before adding anything new, get the existing pipeline producing defensible numbers.

- **Replace the RAGAS judge.** Phi-2 can't produce parseable JSON for the correctness prompt — that's why the current run yields `NaN`. Swap in Claude Haiku 4.5 or GPT-4o-mini via API just for the judge. The generation model can stay local.
- **Actually compute BERTScore.** Five lines on `BioASQ_Test_Responses.csv`. Commit the CSV with score columns so the numbers are reproducible from the repo.
- **Persist the FAISS index to disk.** Right now it rebuilds on every notebook run. Add `storage_context.persist()` + a `load_or_build_index()` helper. Tiny change, large credibility signal — shows the system is thought through beyond a notebook.

---

## Phase 1 — Retrieval quality (1 weekend)

The single most-asked-about area in interviews. This is the real RAG upgrade.

- **Hybrid retrieval**: BM25 + dense via LlamaIndex `QueryFusionRetriever`. PubMed has rare tokens (receptor names, drug codes) where BM25 wins.
- **Reranker**: `BAAI/bge-reranker-base` as a second stage. Retrieve top-20, rerank to top-3. Typically moves answer correctness 10–20 points on its own.
- **Metadata filtering**: `(title, year)` are already stored. Add `disorder` and `neurotransmitter` at ingestion (they are derivable from the JSON filenames in `pubmed_outputs/`). Wire pre-filters into the query path so "2023 studies on acetylcholine and PTSD" actually uses the year filter.
- **One query transformation**: HyDE via `HyDEQueryTransform`. Generates a hypothetical answer, embeds *that*, retrieves on it. Cheap, well-known, gives a concrete talking point.

**Run the eval after each addition.** Build an ablation table:

| Config | Answer Correctness | BERTScore | Context Recall |
|---|---|---|---|
| Dense only | x | x | x |
| + Hybrid | x | x | x |
| + Reranker | x | x | x |
| + HyDE | x | x | x |

This table is the single most valuable artifact a recruiter or interviewer will see.

---

## Phase 2 — Model + shippable UX (1 weekend)

- **Upgrade the generation model.** Phi-2 is a 2023 model and reads like one. Options:
  - **Local**: Llama 3.2 3B Instruct (Q4_K_M) or Qwen 2.5 3B Instruct — both massively better than Phi-2 at the same memory footprint.
  - **Hosted**: Claude Haiku 4.5 via API. Faster, better, cheap enough for a portfolio project.
- **Add citations to responses.** Return `(title, year, score)` per chunk in the answer payload. Render them in the UI. This separates "I built a RAG" from "I built a useful RAG."
- **Ship a Streamlit app.** ~100 lines. A recruiter clicks a link and tries the system in 30 seconds — more signal than the rest of the repo combined. Host on Streamlit Community Cloud or Hugging Face Spaces (both free).
- **Guardrails**: hard refusal on diagnosis/dosing questions, persistent "not medical advice" banner. Domain-appropriate and signals judgment.

---

## Phase 3 — One real differentiator (1 weekend, pick exactly one)

Do **not** do all three. Pick the one that fits a story you can tell.

- **CRAG (Corrective RAG)** — grade retrieved chunks; if scores are low, rewrite the query and retry. Honest, defensible, measurable. Best fit for this corpus.
- **GraphRAG-lite** — extract `(neurotransmitter, disorder, relationship)` triples from abstracts with the LLM, build a small Neo4j or NetworkX graph, query it for multi-hop questions. Highest "wow" factor; most work.
- **ColBERT late-interaction** — token-level retrieval via `ragatouille`. Strongest retrieval upgrade but requires the most explanation in an interview.

**Recommendation: CRAG.** Lowest risk, fits the corpus (PubMed retrieval can be noisy), and gives a clean story: *"I added a retrieval grader because the eval table showed my failures clustered in queries where the top-3 chunks didn't actually contain the answer."* That sentence wins interviews.

---

## Phase 4 — Polish for recruiters (1 evening)

- **Architecture decision record** (`Documentation/decisions.md`): one page, bullet-form. "Chose BioBERT because X. Chose FAISS FlatL2 because Y. Chose hybrid + reranker because the ablation showed Z." Recruiters skim it; interviewers love it.
- **A 60-second demo GIF** at the top of the README, showing the Streamlit app answering one of the harder inference questions with citations.
- **Update the README** to lead with the **eval table** and the **deployed demo link**, not the architecture diagram. Numbers and a clickable demo convert a 10-second scroll into a 2-minute read.

---

## What NOT to do

- **Don't add an agent.** Single corpus, no tools, no actions to take — agency would be cosplay. The judgment to skip it is itself a positive signal.
- **Don't add 10 retrieval techniques.** Two well-explained beats five bolted-on.
- **Don't write a 5,000-word README.** The current one is the right length.
- **Don't switch to LangGraph just because everyone is.** LlamaIndex is fine here; switching costs a week.
- **Don't add multi-modal, voice, or "long-term memory" features.** They don't fit the use case and look like padding.

---

## Total time to "2026-ready"

Realistically: **3 weekends + 2 evenings**, at portfolio-project pace. The Phase 1 ablation table alone — done properly — puts this project ahead of 80% of portfolio RAGs.

---

## Interview-ready talking points (after Phases 0–2)

- *"I started with a single-vector dense RAG and an evaluation harness. The first eval surfaced a judge-model failure mode — Phi-2 couldn't produce parseable JSON for RAGAS — so I separated generation from evaluation and used a stronger judge."*
- *"The biggest retrieval gain came from a `bge-reranker-base` second stage, which moved BERTScore from X to Y on the BioASQ subset."*
- *"I deliberately did not make it agentic. Single corpus, no tools, no actions. Agency would have added latency and failure modes without fixing the actual quality gap, which was in retrieval, not planning."*
- *"Metadata-aware retrieval matters here because the questions span six disorders, seven neurotransmitters, and several decades — filtering by `disorder` or `year` before dense search avoids dragging in irrelevant chunks."*
