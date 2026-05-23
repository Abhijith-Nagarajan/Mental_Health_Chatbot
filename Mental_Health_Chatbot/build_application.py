"""
Mental-health RAG: load corpus -> chunk -> embed (BioBERT) -> FAISS -> retrieve -> chat.

Converted from build_application.ipynb. Documents carry
pmid / is_review / disorders / neurotransmitters metadata (see
dataframe_to_documents), which enables metadata-filtered retrieval later.

Generation runs on a local Ollama server (qwen2.5:7b). Index-building is split
out from the chat engine so chunk/embed/FAISS can run without the LLM.

Run:
    ollama serve            # if not already running
    .venv/bin/python build_application.py
"""

import os

import pandas as pd

from llama_index.core import Document, VectorStoreIndex, StorageContext, load_index_from_storage, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.settings import Settings
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import faiss

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "pubmed_abstracts", "pubmed_abstracts.csv")
PERSIST_DIR = os.path.join(HERE, "storage")   # saved FAISS index + docstore

OLLAMA_MODEL = "qwen2.5:7b"   # generation via local Ollama server
EMBED_MODEL = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
RERANK_MODEL = "BAAI/bge-reranker-base"   # cross-encoder for optional 2nd-stage reranking

TOP_K = 3            # chunks passed to the LLM
RERANK_FETCH_K = 20  # wider candidate set pulled before reranking down to TOP_K

# Grounding prompt with an explicit, non-negotiable abstention instruction.
ABSTAIN_MSG = "I don't have enough information in the provided documents to answer this."
QA_PROMPT = PromptTemplate(
    "You are a careful biomedical research assistant. Answer the question using ONLY the "
    "context below, and do not use prior knowledge.\n"
    "If the provided context does not contain sufficient information to answer accurately, "
    f'respond with exactly: "{ABSTAIN_MSG}"\n'
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Question: {query_str}\n"
    "Answer: "
)

def dataframe_to_documents(df):
    docs = []
    for _, row in df.iterrows():
        text = row["abstract"]
        metadata = {
            "pmid": str(row.get("pmid", "")),
            "title": row.get("title", ""),
            "year": str(row.get("year", "")),
            "is_review": bool(row.get("is_review", False)),
            "disorders": [d for d in str(row.get("disorders", "")).split("; ") if d],
            "neurotransmitters": [n for n in str(row.get("neurotransmitters", "")).split("; ") if n],
        }
        # pmid / is_review are for filtering, not semantic content — keep them
        # out of the text the embedder and LLM see.
        docs.append(Document(
            text=text,
            metadata=metadata,
            excluded_embed_metadata_keys=["pmid", "is_review"],
            excluded_llm_metadata_keys=["pmid", "is_review"],
        ))
    return docs


def _configure_embedding():
    """Embed model + chunker — needed both for building and for querying."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)


def build_index(docs, persist=True):
    """Chunk -> embed (BioBERT) -> FAISS, then persist to disk. No LLM needed."""
    _configure_embedding()
    dims = len(Settings.embed_model.get_text_embedding("sample text"))
    faiss_db = FaissVectorStore(faiss_index=faiss.IndexFlatL2(dims))
    storage_context = StorageContext.from_defaults(vector_store=faiss_db)
    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
    if persist:
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    return index


def load_index():
    """Load the persisted index from disk (fast — no re-embedding)."""
    _configure_embedding()
    vector_store = FaissVectorStore.from_persist_dir(PERSIST_DIR)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=PERSIST_DIR)
    return load_index_from_storage(storage_context)


def get_index(rebuild=False):
    """Load the index if it exists on disk, otherwise build it from the CSV."""
    if not rebuild and os.path.isdir(PERSIST_DIR):
        return load_index()
    df = pd.read_csv(CSV_PATH).dropna(subset=["abstract"])
    docs = dataframe_to_documents(df)
    print(f"Building index from {len(docs)} documents...")
    return build_index(docs, persist=True)


def make_query_engine(index, rerank=False):
    """Query engine, optionally with a cross-encoder second stage.

    Without rerank: dense retrieve TOP_K chunks.
    With rerank:    dense retrieve a wider RERANK_FETCH_K, then bge-reranker
                    re-orders and keeps the best TOP_K. Both use the abstention prompt.

    Diagnosis showed answer-bearing chunks sit at rank ~4-10 under dense retrieval
    (just outside top-3), which is exactly what fetch-20 -> rerank -> top-3 captures.
    """
    postprocessors = []
    top_k = TOP_K
    if rerank:
        from llama_index.core.postprocessor import SentenceTransformerRerank
        postprocessors = [SentenceTransformerRerank(model=RERANK_MODEL, top_n=TOP_K)]
        top_k = RERANK_FETCH_K
    return index.as_query_engine(
        similarity_top_k=top_k,
        node_postprocessors=postprocessors,
        text_qa_template=QA_PROMPT,
    )


def build_chat_engine(index, rerank=False):
    """Wrap the index in a chat engine. Generation runs on Ollama."""
    Settings.llm = Ollama(
        model=OLLAMA_MODEL,
        temperature=0.2,
        request_timeout=120.0,
        context_window=8192,
    )
    memory = ChatMemoryBuffer.from_defaults(token_limit=600)
    query_engine = make_query_engine(index, rerank=rerank)
    return CondenseQuestionChatEngine.from_defaults(query_engine=query_engine, memory=memory)


def main():
    # one-time ingest: (re)build the index and persist it to ./storage
    index = get_index(rebuild=True)
    print(f"Index built and persisted to {PERSIST_DIR}")

    chat_engine = build_chat_engine(index)
    query = "Is elevated dopamine consistently observed in patients with schizophrenia?"
    response = chat_engine.chat(query)
    print("\nQ:", query)
    print("A:", response.response)


if __name__ == "__main__":
    main()
