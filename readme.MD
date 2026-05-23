# Mental Health Chatbot — Domain-Specific RAG over PubMed Literature

An end-to-end **Retrieval-Augmented Generation (RAG)** system that answers questions on the neurochemistry of mental health by grounding a local LLM in peer-reviewed PubMed abstracts. Built with **LlamaIndex**, **FAISS**, **BioBERT** embeddings, and a locally-served **GGUF** model via `llama.cpp`. Quality is measured against the **BioASQ-13b** benchmark using **RAGAS**.

---

## Motivation

I wanted to build a mental-health chatbot because it sits at the intersection of three areas I care about — AI, psychology, and neuroscience. The system is designed to link neurotransmitters (dopamine, serotonin, GABA, glutamate, norepinephrine, acetylcholine, endorphins) to mood disorders (depression, anxiety, bipolar, schizophrenia, OCD, PTSD), and to surface how nutrition, exercise, and lifestyle affect well-being. The interdisciplinary scope also gave me a sufficiently large and noisy corpus to stress-test a RAG pipeline end-to-end.

---

## What this project demonstrates

| Skill area | Where it shows up |
|---|---|
| **Domain-aware data acquisition** | Programmatic PubMed scraping with MeSH-term queries via Biopython Entrez (`data_scraping.ipynb`) |
| **Document ingestion & metadata design** | JSON → DataFrame → LlamaIndex `Document` objects with title/year metadata (`extract_documents.ipynb`) |
| **RAG architecture & component selection** | BioBERT embeddings, sentence-level chunking, FAISS L2 vector store, local GGUF LLM, condense-question chat with memory (`build_application.ipynb`) |
| **Hyperparameter tuning** | Comparative experiments with chunk size/overlap, top-k, embedding models, and base LLMs (`POC/llamaindex_tuning.py`, `Documentation/hyperparameters.txt`) |
| **Evaluation against a public benchmark** | BioASQ-13b ground truth → RAGAS `AnswerCorrectness` with a custom correctness prompt and a LlamaIndex LLM wrapper (`POC/llamaindex_eval.ipynb`) |
| **Cost-conscious engineering** | Fully local stack — no paid APIs at inference time; quantized models (Q4_K_M / Q5_K_M / Q8_0) chosen for the available hardware |
| **Test design** | 19 hand-crafted inference questions across 7 dimensions (recall, factual grounding, edge cases, multi-hop, long-context stress, safety, novelty) — see `Mental_Health_Chatbot/Inference_Questions.txt` |

---

## Architecture

```
   ┌──────────────────────────┐
   │  PubMed (NCBI Entrez)    │   MeSH queries: 6 disorders × 7 neurotransmitters
   └──────────────┬───────────┘
                  │  Biopython • esearch + efetch (XML)
                  ▼
   ┌──────────────────────────┐
   │  pubmed_outputs/*.json   │   ~835 abstracts, cached on disk
   └──────────────┬───────────┘
                  │  flatten + add (title, year) metadata
                  ▼
   ┌──────────────────────────┐
   │  LlamaIndex Documents    │
   └──────────────┬───────────┘
                  │  SentenceSplitter (chunk_size=512, overlap=50)
                  ▼
   ┌──────────────────────────┐
   │  BioBERT embeddings      │   pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb
   └──────────────┬───────────┘
                  │  IndexFlatL2
                  ▼
   ┌──────────────────────────┐         ┌──────────────────────────┐
   │  FAISS vector store      │ ───────►│  VectorStoreIndex (top-k=3)│
   └──────────────────────────┘         └──────────────┬───────────┘
                                                       │
                                          ChatMemoryBuffer (600 tok)
                                                       │
                                        CondenseQuestionChatEngine
                                                       │
                                                       ▼
                                        ┌──────────────────────────┐
                                        │  Local LLM (llama.cpp)   │  Phi-2 Orange Q4_K_M
                                        │  ctx=2048, T=0.2         │  (also tested: Mistral-7B-Q5, TinyLlama-Q8)
                                        └──────────────────────────┘
```

---

## Tech stack

- **Orchestration**: `llama-index` (core, settings, node_parser, memory, chat_engine)
- **Embeddings**: `BAAI/bge-small-en`, `all-MiniLM-L6-v2`, and the domain-specific `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` (final choice)
- **Vector store**: `faiss-cpu` (`IndexFlatL2`) wrapped via `FaissVectorStore`
- **LLM inference**: `llama-cpp-python` against quantized GGUFs — Phi-2 Orange (Q4_K_M), Mistral-7B-Instruct (Q5_K_M), TinyLlama 1.1B (Q8_0)
- **Data acquisition**: `biopython` (`Bio.Entrez`), `xmltodict`
- **Evaluation**: `ragas` (`AnswerCorrectness`, `LlamaIndexLLMWrapper`), `datasets`, BioASQ-13b training set
- **Data wrangling**: `pandas`, `numpy`

---

## Repository structure

```
.
├── Mental_Health_Chatbot/        # Production-style RAG pipeline
│   ├── data_scraping.ipynb       # PubMed → JSON via MeSH queries
│   ├── extract_documents.ipynb   # JSON → CSV → LlamaIndex Documents
│   ├── build_application.ipynb   # Full RAG: embed → index → retrieve → generate
│   ├── Inference_Questions.txt   # Hand-crafted eval prompts (7 dimensions)
│   ├── pubmed_outputs/           # Raw scraped abstracts (35 JSON files)
│   └── pubmed_abstracts/         # Consolidated CSV (~835 abstracts)
│
├── POC/                          # Experimentation & evaluation harness
│   ├── llamaindex_starter.py     # Vanilla RAG with default settings
│   ├── llamaindex_tuning.py      # Chunking / embedding / model sweeps + latency check
│   ├── llamaindex_eval.ipynb     # BioASQ ground-truth + RAGAS scoring
│   ├── Abstracts/                # PMID-keyed abstract cache (1,131 files)
│   ├── bioasq_ground_truth.csv   # Questions + ideal answers from BioASQ-13b
│   ├── BioASQ_Test_Responses.csv # Model responses + retrieved contexts
│   └── pubmed_abstracts.csv      # POC abstract corpus
│
├── Documentation/                # Design notes
│   ├── rag_architectures.txt     # Simple / memory / branched / adaptive / agentic
│   ├── hyperparameters.txt       # LlamaIndex tuning surface
│   └── Ragas.txt                 # RAGAS internals (prompt, weights, embeddings)
│
├── Data/                         # Reference papers used during POC
│   ├── REBEL.pdf
│   └── RE_Survey_Paper.pdf
│
└── readme.MD
```

---

## Pipeline walkthrough

### 1. Data acquisition — `Mental_Health_Chatbot/data_scraping.ipynb`
Iterates over the Cartesian product of 6 disorders × 7 neurotransmitters and builds MeSH queries of the form `"Depressive Disorder"[MeSH] AND "Serotonin"[MeSH]`. Pulls up to 25 abstracts per pair via `Entrez.esearch` + `Entrez.efetch`, parses XML, and persists one JSON per pair under `pubmed_outputs/`. Rate-limits to stay within NCBI guidance.

### 2. Document preparation — `extract_documents.ipynb`
Loads all JSON files, drops empty abstracts, and constructs LlamaIndex `Document` objects with `(title, year)` metadata — this metadata is later visible to the retriever for filtering and to the user as provenance.

### 3. Indexing — `build_application.ipynb`
- **Chunking**: `SentenceSplitter(chunk_size=512, chunk_overlap=50)` — large enough to preserve a full claim, small enough to keep top-k retrieval precise.
- **Embeddings**: switched from generic `BAAI/bge-small-en` to **BioBERT** fine-tuned on biomedical NLI/STS data, because domain match outweighed model size for this corpus.
- **Vector store**: `faiss.IndexFlatL2` for exact L2 search — small enough to skip ANN approximations, and gives a deterministic baseline.

### 4. Retrieval + generation
- `as_query_engine(similarity_top_k=3)` keeps context lean for Phi-2's 2,048-token window.
- `CondenseQuestionChatEngine` + `ChatMemoryBuffer(token_limit=600)` enables multi-turn follow-ups by rewriting questions against history.
- LLM: Phi-2 Orange Q4_K_M (`T=0.2`, `max_new_tokens=256`) — chosen as a balance between fluency and the size/latency budget on consumer hardware after Mistral-7B proved too slow.

### 5. Evaluation — `POC/llamaindex_eval.ipynb`
- **Ground truth**: parses the first 100 BioASQ-13b training questions, dedupes their cited PubMed documents (\~1,131 unique PMIDs), and batch-fetches abstracts with an on-disk cache.
- **Test set**: questions, retrieved contexts, model responses, and ideal answers shaped into a HuggingFace `Dataset` with columns `question / retrieved_contexts / answer / ground_truth`.
- **Metric**: RAGAS `AnswerCorrectness` driven by the same local Phi-2 model via `LlamaIndexLLMWrapper`, with a custom JSON-output correctness prompt and BioBERT as the similarity embedding. The current run highlights a real-world failure mode — small quantized judges produce unparseable JSON, yielding `NaN` scores — which is itself a useful finding about evaluator-model selection in low-resource setups.

### 6. Probing questions — `Mental_Health_Chatbot/Inference_Questions.txt`
Hand-designed across seven axes so the system is stressed beyond simple recall:
1. Core neurotransmitter–disorder links (recall & precision)
2. Treatment-mechanism questions (factual grounding)
3. Gut–brain axis (edge-case retrieval)
4. Comorbidity & cross-talk (multi-hop)
5. Historical / trend questions (context-window stress)
6. Safety & side-effects (grounding + specificity)
7. Emerging research (novelty)

---

## Key design decisions

- **Why a local GGUF model?** Reproducibility and zero API cost on a personal machine. The trade-off — a small context window — directly shaped `top_k`, chunk size, and memory-buffer choices.
- **Why BioBERT over a generic sentence-transformer?** PubMed abstracts are dense with domain jargon; a biomedical embedding consistently surfaced more relevant chunks during qualitative inspection.
- **Why FAISS `IndexFlatL2` and not HNSW/IVF?** At sub-1k vectors, exact search is fast and removes recall variance as a confound when tuning the rest of the pipeline.
- **Why BioASQ for evaluation?** It is a peer-reviewed biomedical QA benchmark with curated ideal answers and source documents — a stronger signal than self-generated ground truth.

---

## Getting started

```bash
# 1. Install dependencies (Python 3.10+)
pip install llama-index llama-index-vector-stores-faiss llama-index-embeddings-huggingface \
            llama-index-llms-llama-cpp llama-cpp-python faiss-cpu sentence-transformers \
            biopython xmltodict ragas datasets pandas

# 2. Download a quantized GGUF model (e.g. Phi-2 Orange Q4_K_M) and update `model_path`
#    in Mental_Health_Chatbot/build_application.ipynb

# 3. Scrape PubMed (or reuse the cached pubmed_outputs/)
#    Run: Mental_Health_Chatbot/data_scraping.ipynb

# 4. Build the index and chat
#    Run: Mental_Health_Chatbot/build_application.ipynb
```

---

## Roadmap

- **Branched / adaptive RAG** — route lifestyle vs. mechanism questions to different sub-indexes (see `Documentation/rag_architectures.txt`).
- **Hybrid retrieval** — combine BM25 with dense retrieval for rare-term queries (specific drug names, gene IDs).
- **Stronger evaluator** — re-run RAGAS with a larger judge model (or hosted Claude/GPT-4) to fix the JSON-parsing failure mode seen with Phi-2.
- **Reranker** — add a cross-encoder reranker (e.g. `ms-marco-MiniLM-L-6-v2` or a biomedical equivalent) between retrieval and generation.
- **Guardrails** — explicit "this is not medical advice" framing and refusal on diagnosis-style queries.
