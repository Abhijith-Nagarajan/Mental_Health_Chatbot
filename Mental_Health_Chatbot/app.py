"""
Streamlit chat UI for the mental-health RAG.

Landing page documents the system (domain stats + architecture + implementation
spec); the chat below is grounded in PubMed abstracts with source citations.

Run:
    ollama serve                       # if not already running
    .venv/bin/streamlit run app.py
"""

import os

import pandas as pd
import streamlit as st

from build_application import get_index, build_chat_engine, CSV_PATH, OLLAMA_MODEL, EMBED_MODEL
from data_scraping import DISORDERS, NEUROTRANSMITTERS

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Mental Health RAG", page_icon="🧠", layout="wide")


@st.cache_data
def corpus_stats():
    df = pd.read_csv(CSV_PATH).dropna(subset=["abstract"])
    yr = pd.to_numeric(df["year"], errors="coerce")
    return {
        "abstracts": len(df),
        "reviews": int(df["is_review"].sum()),
        "review_pct": round(df["is_review"].mean() * 100),
        "multi_disorder": int(df["disorders"].astype(str).str.contains(";").sum()),
        "year_min": int(yr.min()),
        "year_max": int(yr.max()),
        "year_median": int(yr.median()),
    }


@st.cache_data
def papers_per_decade():
    df = pd.read_csv(CSV_PATH).dropna(subset=["abstract"])
    yr = pd.to_numeric(df["year"], errors="coerce").dropna()
    counts = (yr // 10 * 10).astype(int).value_counts().sort_index()
    counts.index = counts.index.astype(str) + "s"
    return counts.rename_axis("decade").rename("papers")


@st.cache_data
def eval_metrics(filename):
    path = os.path.join(HERE, filename)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    out = {
        "n": len(df),
        "answer_correctness": df["answer_correctness"].mean(),
        "bert_f1": df["bert_f1"].mean(),
        "abstentions": int(df["answer"].str.contains("don't have enough").sum()),
    }
    # diagnostic metrics (added by diagnose.py), shown if present
    for key, col in [("faithfulness", "faithfulness"),
                     ("context_precision", "llm_context_precision_with_reference"),
                     ("context_recall", "context_recall")]:
        if col in df.columns:
            out[key] = df[col].mean()
    return out


@st.cache_resource(show_spinner="Loading index and model…")
def get_engine():
    return build_chat_engine(get_index())


def render_overview(stats):
    with st.expander("📖 System overview — domain, architecture & implementation", expanded=True):
        st.markdown("##### Domain")
        c = st.columns(4)
        c[0].metric("Disorders", len(DISORDERS))
        c[1].metric("Neurotransmitters", len(NEUROTRANSMITTERS))
        c[2].metric("Abstracts", f"{stats['abstracts']:,}")
        c[3].metric("Reviews", f"{stats['reviews']} ({stats['review_pct']}%)")
        c = st.columns(4)
        c[0].metric("Grid", f"{len(DISORDERS)}×{len(NEUROTRANSMITTERS)} cells")
        c[1].metric("Multi-disorder papers", stats["multi_disorder"])
        c[2].metric("Year range", f"{stats['year_min']}–{stats['year_max']}")
        c[3].metric("Median year", stats["year_median"])
        disorders_str = ", ".join(d.replace("_", " ") for d in DISORDERS)
        nts_str = ", ".join(n.replace("_", " ") for n in NEUROTRANSMITTERS)
        st.markdown(
            f"<p style='font-size:1.1rem;margin-bottom:0.25rem'><b>Disorders:</b> {disorders_str}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p style='font-size:1.1rem'><b>Neurotransmitters:</b> {nts_str}</p>",
            unsafe_allow_html=True,
        )

        st.markdown("**Papers per decade**")
        st.bar_chart(papers_per_decade(), height=220)

        st.markdown("##### Architecture")
        lanes = [
            ("Data Acquisition", [
                ("Source", "PubMed · NCBI Entrez"),
                ("Queries", "MeSH · 6×7 cells"),
                ("Sampling", "review-weighted · size-scaled · recent"),
                ("Corpus", f"{stats['abstracts']:,} abstracts · {stats['review_pct']}% reviews"),
            ]),
            ("Indexing & Embedding", [
                ("Chunking", "SentenceSplitter 512 / 50"),
                ("Embeddings", "BioBERT · 768-d · normalized"),
                ("Vector store", "FAISS IndexFlatL2"),
                ("Persistence", "saved to ./storage"),
            ]),
            ("Retrieval", [
                ("Top-k", "3 nearest chunks"),
                ("Metric", "L2 ≡ cosine"),
                ("Metadata", "disorder · NT · is_review"),
                ("Planned", "reranker · hybrid · filters"),
            ]),
            ("Generation", [
                ("Model", "qwen2.5:7b (Ollama)"),
                ("Chat", "Condense + memory"),
                ("Grounding", "abstains if no context"),
                ("Output", "answer + PubMed cites"),
            ]),
        ]
        cols = st.columns(len(lanes))
        for col, (title, items) in zip(cols, lanes):
            with col:
                st.markdown(
                    f"<div style='background:#4f46e5;color:#fff;padding:8px;border-radius:8px;"
                    f"text-align:center;font-weight:700;font-size:0.85rem'>{title}</div>",
                    unsafe_allow_html=True,
                )
                for label, val in items:
                    st.markdown(
                        f"<div style='background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px;"
                        f"padding:6px 8px;margin-top:8px;font-size:0.78rem;min-height:46px'>"
                        f"<b>{label}</b><br>{val}</div>",
                        unsafe_allow_html=True,
                    )
        st.caption("Pipeline flows left → right.")
        st.caption(f"Embedding model: `{EMBED_MODEL}` · Generator: `{OLLAMA_MODEL}` (local via Ollama)")
        st.caption("Embeddings are unit-normalized, so L2 ranking ≡ cosine. Grounding: abstains when context is insufficient.")

        st.markdown("##### Evaluation")
        base = eval_metrics("eval_results.csv")
        if base:
            st.markdown(
                f"Domain set: 15 disorder × neurotransmitter Q&A, scored with RAGAS "
                f"(qwen2.5:7b judge) + BERTScore. *(BioASQ rejected — 1/100 overlap.)*\n\n"
                "| Config | Answer Correctness | BERTScore F1 | Abstentions |\n"
                "|---|---|---|---|\n"
                f"| Dense, top-3 (no reranker) | {base['answer_correctness']:.2f} | "
                f"{base['bert_f1']:.2f} | {base['abstentions']}/{base['n']} |"
            )
            st.caption("Answer Correctness is the meaningful metric; BERTScore is high but soft (rewards topical overlap).")

            if "faithfulness" in base:
                st.markdown("**Diagnosis — localizing the score (RAGAS decomposition)**")
                d = st.columns(3)
                d[0].metric("Faithfulness", f"{base['faithfulness']:.2f}",
                            help="answer grounded in retrieved context (rules out hallucination)")
                d[1].metric("Context Precision", f"{base['context_precision']:.2f}",
                            help="are the retrieved chunks relevant")
                d[2].metric("Context Recall", f"{base['context_recall']:.2f}",
                            help="do retrieved chunks cover the reference facts")
                st.caption(
                    "High faithfulness rules out generator hallucination; the bottleneck is context recall — "
                    "answer-bearing chunks rank just below top-3 (rank 4–10), which is what the reranker targets."
                )
        else:
            st.caption("Evaluation not run yet — `python run_eval.py`.")


def render_sources(source_nodes):
    if not source_nodes:
        return
    with st.expander(f"Sources ({len(source_nodes)})"):
        for n in source_nodes:
            md = n.node.metadata
            title = md.get("title", "Untitled")
            year = md.get("year", "")
            pmid = md.get("pmid", "")
            tag = " · review" if md.get("is_review") else ""
            link = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            label = f"[{title}]({link})" if link else title
            # FAISS IndexFlatL2 returns squared-L2 distance; vectors are unit-normalized
            # so cosine = 1 - distance/2 (higher = more similar).
            cosine = 1 - n.score / 2
            st.markdown(f"- {label} ({year}{tag}) — cosine {cosine:.2f}")


st.title("🧠 Neurotransmitters & Mental Health — Research RAG")
st.caption("Answers grounded in PubMed abstracts. Educational use only — not medical advice.")

render_overview(corpus_stats())

st.divider()
st.subheader("💬 Chat")

chat_engine = get_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about a disorder and a neurotransmitter…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            response = chat_engine.chat(prompt)
        st.markdown(response.response)
        render_sources(response.source_nodes)

    st.session_state.messages.append({"role": "assistant", "content": response.response})
