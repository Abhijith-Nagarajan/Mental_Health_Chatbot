# Interview Talking Points

Each section follows the same arc: **what the POC did → what I realized and why → what I changed → the takeaway.** These are framed as honest gap analysis — limitations I found and addressed, which is the story to tell.

---

## Data Collection

- **Initial version** capped at **25 abstracts** per disorder×neurotransmitter MeSH pair — deliberately simple, just to validate the end-to-end pipeline for a POC.
- **Then I realized a flat cap was wrong**, because the literature is power-law distributed. I checked live PubMed counts and the imbalance was stark:
  - Schizophrenia + Dopamine → **2,302** abstracts exist (25 = ~1% coverage)
  - OCD + Acetylcholine → only **5** exist (a cap of 25 over-samples it)
- So I moved to a smarter sampling strategy for three reasons:
  - **(X) Cell-size scaling** — big associations need more abstracts; tiny ones should just take everything available, instead of one fixed number for all.
  - **(Y) Review-weighting** — prioritize review articles, which capture field consensus, rather than a random slice of primary studies.
  - **(Z) Recency** — sort by publication date so the corpus reflects current understanding, not whatever PubMed returned by default.
- I also caught a **correctness bug**: PTSD silently returned zero results because the code used `"Post-Traumatic Stress Disorder"[MeSH]`, which isn't a valid MeSH descriptor — the correct heading is `"Stress Disorders, Post-Traumatic"[MeSH]`. My corpus had 5 disorders, not the intended 6.
- **Takeaway:** This moved me from an arbitrary ~760-abstract flat sample (5 disorders) to a balanced, review-weighted, recency-aware corpus: **2,022 abstracts scraped across a full 6×7 grid → 1,625 unique after deduping by PMID** (~2.1× the original), **~34% review articles**, plus disorder/neurotransmitter association metadata on every row. A *defensible* data store rather than an arbitrary one — and I can justify every sampling decision with the PubMed numbers.

### Coverage (saved ÷ what exists on PubMed)

- **Coverage is intentionally inverse to cell size**: cap the large literatures (a ~150 review-weighted sample captures consensus), exhaust the rare ones.
- Per-cell coverage went from **median 16% → 21%** (mean 28% → 33%); the gain concentrates where it matters most — the major associations:

  | Association | Old (flat 25) | New (size-scaled) |
  |---|---|---|
  | Schizophrenia + Dopamine | 1.1% | **6.1%** (~5.5×) |
  | Depression + Serotonin | 1.2% | **7.2%** (~6×) |
  | Depression + Norepinephrine | 3.0% | **11.2%** |
  | Schizophrenia + Serotonin | 3.1% | **11.2%** |

- Rare associations stay near-complete (OCD+Acetylcholine, PTSD+Acetylcholine, OCD+Endorphins ≈ 80–100%).
- **Nuance to state:** raw *document* coverage on the biggest literatures is still single-digit % **by design** — but since ~34% are reviews/meta-analyses (each summarizing hundreds of primary studies), the effective *knowledge* coverage is far higher than the document percentage.
- **Interview line:** *"I don't target uniform coverage — indexing all 2,300 dopamine papers would be near-duplicate findings. I cap large associations at ~150 review-weighted abstracts to capture consensus and exhaust the rare ones, so coverage is deliberately inverse to cell size."*

---

## Knowledge Representation

- **Initially I described the project as** "I have the disorder ↔ neurotransmitter associations."
- **Then I realized that's imprecise** — the associations aren't *modeled* anywhere. There's no graph, no edges. I have a flat corpus of abstracts plus dense retrieval; the associations are **implicit in the documents** and emerge from retrieval + LLM synthesis.
- The accurate framing: I **curated the corpus around those pairs** so retrieval stays grounded in relevant literature — but the relationships themselves are emergent, not structured.
- **Takeaway:** I can speak precisely about what's *modeled* vs. what's *emergent*. Making associations explicit and queryable (e.g. GraphRAG extracting `(disorder, neurotransmitter, relationship)` triples) is a clear, justified roadmap item — not something I overclaim today.

---

## Capability Envelope (Scope)

- **Initial version** included a stress-test question set across 7 categories (treatment mechanisms, gut-brain axis, comorbidity, historical trends, safety, emerging research, core links).
- **Then I mapped each category against the corpus** and realized only the core questions are well-supported:
  - **Sweet spot:** "What is the relationship between [one disorder] and [one neurotransmitter]?" — one MeSH pair, dedicated abstracts.
  - **Degrades:** broad aggregation like "what neurotransmitters are associated with anxiety?" — needs all 7, but top-k=3 only retrieves 3 chunks.
  - **Out of scope:** treatment mechanisms (corpus isn't drug-curated), comorbidity (no graph for cross-talk), lifestyle (not yet in corpus), historical (too shallow per pair).
- **Takeaway:** The corpus defines the capability envelope. I measure my own system's limits and let that gap analysis drive the roadmap — that's more valuable than claiming it answers everything.

---

## Retrieval

- **Initial version** used single-stage **dense retrieval, top-k = 3**, with no reranking, no hybrid search, and metadata (title, year) stored but unused at query time.
- **Then I realized the limits:**
  - top-k = 3 **starves aggregation queries** that need many sources.
  - dense-only retrieval **misses rare biomedical terms** (drug codes, receptor names) where keyword/BM25 matching wins.
  - I had useful metadata (year, disorder) that I wasn't filtering on.
- The plan: **two-stage retrieval** (retrieve top-20 → cross-encoder rerank → top-3), **hybrid BM25 + dense**, **metadata pre-filtering**, and one **query transformation** (HyDE).
- **Takeaway:** I identified that **retrieval — not corpus size or model choice — is the dominant quality lever** past a density threshold, and prioritized work accordingly.

### Diagnosing the 0.41 before reaching for the reranker

I refused to add a reranker on a hunch — I decomposed the RAGAS metrics to localize the failure first:

| Metric | Score | Tells me |
|---|---|---|
| Faithfulness | **0.93** | answers *are* grounded in context — **not** a generation/hallucination problem (this refuted the "quantized model hallucinates" prior) |
| Context Precision | 0.75 | retrieved chunks are mostly relevant |
| Context Recall | **0.567** | retrieved chunks miss ~40% of the facts the references need → **retrieval-coverage** bottleneck |

Then I checked the three causes of low recall, cheapest first:
- **Facts not in corpus?** No — grep showed parvalbumin (26), interneuron (73), mesolimbic (46), NMDA (230) all present.
- **Chunking fragmenting facts?** No — chunk:doc ratio 1.10; only 10% of abstracts split.
- **Retrieval not surfacing them?** **Yes.** For GABA/schizophrenia, "parvalbumin" appears in 26 abstracts but **0** reached the top-3.

Finally I probed *where* the answer-bearing chunks rank under dense retrieval: **rank 4–10** for the failing questions (just outside top-3, well inside a top-20 fetch). That is precisely the band a fetch-20 → rerank → top-3 pipeline captures — so the reranker is **evidence-justified, not a guess**.

- **Honest ceiling:** one failure (acetylcholine/depression) had its chunk at rank 1 yet recall 0 — a *reference-strictness* issue a reranker can't fix.
- **Takeaway:** decompose → localize → confirm the chunks live in the fetch window → *then* build. "Confirm it's the actual problem before adding one."

---

## Evaluation

- **Initial version** used the BioASQ benchmark for ground truth and RAGAS `answer_correctness`, but the metric returned **NaN**.
- **Then I diagnosed why:** I was using the same small local model (Phi-2) for *both* generation and as the RAGAS judge, and it couldn't produce the parseable JSON the metric requires. I also realized I hadn't actually computed BERTScore anywhere.
- The fix: **separate generation from evaluation**, use a judge that's reliable at structured output (hosted, or `qwen2.5:7b` which is strong at JSON), add BERTScore, and build an **ablation table** (dense vs. reranked) so improvements are quantified.
- **I also rejected BioASQ as the test set after inspecting it:** only **1 of 100** BioASQ questions overlaps this corpus (and that one isn't even a mental disorder), so it would only trigger correct abstentions and score ~0. Instead I built a **15-question domain set** of disorder×neurotransmitter Q&A with reference answers, mirroring BioASQ's `(question, ideal answer)` format.

### Baseline results (dense retrieval, top-3, no reranker; n=15)

| Metric | Score |
|---|---|
| RAGAS Answer Correctness (qwen2.5:7b judge) | **0.41** |
| BERTScore F1 | **0.86** |
| Abstentions | 1/15 |

- **What the numbers mean:** Answer Correctness (0.41) is the meaningful, harsher metric — LLM-judge factual F1 (0.75) + embedding similarity (0.25). BERTScore (0.86) is high but "soft" — it rewards topical overlap and barely discriminates between same-domain texts, so it's not the headline.
- **Honesty note:** the portfolio previously published BERTScore 0.67 / Answer Correctness 0.60 — neither reproducible. These measured numbers (0.86 / 0.41) replace them; the lowest-scoring questions (GABA in schizophrenia 0.14, norepinephrine in depression 0.32) are *retrieval* failures, which is the motivation for the reranker.
- **Takeaway:** I run the full loop — **measure → diagnose → fix → re-measure** — and I keep numbers reproducible from the repo. I won't put a metric on a resume that the code can't reproduce.

---

## Model Serving & Environment

- **Initial version** ran Phi-2 as a GGUF file in-process via `llama-cpp-python`, with a hardcoded Windows path (`E:\RAG_Models\...`), and no reproducible environment.
- **Then I realized** this wasn't portable: compiling `llama-cpp-python` on a new machine is friction, the path was machine-specific, and there was no lockfile.
- The change: a **uv-managed environment** plus **Ollama** serving `qwen2.5:7b` — a warm local server, no compilation, automatic Metal/GPU use. Swapping the backend was a **one-line change** (`LlamaCPP(...)` → `Ollama(...)`); everything downstream stayed the same.
- **Context I can speak to:** GGUF is the model *file format*, llama.cpp is the *engine*, and Ollama is a *server + manager* built on llama.cpp that runs GGUF files — so this was a serving-layer change, not a model-architecture change.
- **Takeaway:** I separated model serving from application logic and made the environment reproducible — portfolio-grade infrastructure hygiene, not just a notebook.

---

## Design Judgment: Why NOT Agentic

- **I considered making it agentic** — the popular default right now.
- **Then I reasoned through whether it was warranted:** single corpus, no external tools, no actions to take. Agency would add latency and new failure modes (agent loops, tool-call hallucinations) without fixing the real gap, which is in retrieval, not planning. Agency is only justified with multiple sources to route between, real tool use, or genuine multi-hop retrieve→reason→retrieve loops.
- **Takeaway:** I match the architecture to the problem instead of reaching for buzzwords. The one case where it *would* become justified is adding a second, distinct corpus (e.g. lifestyle factors) — then a router (branched RAG) earns its place. Naming that tradeoff unprompted is itself a strong signal.

---

## One-line project summary (for the opening)

> *"I built an educational, research-grounded RAG over PubMed abstracts that explains how neurotransmitters relate to mood disorders. I curated the corpus around disorder×neurotransmitter MeSH pairs so retrieval stays grounded, evaluated it on the BioASQ benchmark with RAGAS, and the limitations I found — sampling, retrieval depth, scope — are what drive my roadmap."*
