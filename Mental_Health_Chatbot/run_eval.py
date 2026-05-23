"""
Domain eval for the mental-health RAG.

Runs each question in eval_questions.csv through the persisted index, then scores:
  - RAGAS AnswerCorrectness  (judge = qwen2.5:7b via Ollama; BioBERT similarity)
  - BERTScore F1             (answer vs reference)

Why a custom set instead of BioASQ: only 1 of 100 BioASQ questions overlaps this
corpus, so BioASQ would just trigger correct abstentions and score ~0. This set
mirrors BioASQ's (question, ideal answer) format but is scoped to the
disorder x neurotransmitter domain the corpus actually covers.

Run:
    ollama serve
    .venv/bin/python run_eval.py
"""

import os

import pandas as pd
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

from build_application import get_index, make_query_engine, OLLAMA_MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
QUESTIONS_CSV = os.path.join(HERE, "eval_questions.csv")


def generate_answers(qdf, rerank=False):
    """Run each question through the RAG; collect answer + retrieved contexts."""
    index = get_index()  # loads persisted index, sets Settings.embed_model
    Settings.llm = Ollama(model=OLLAMA_MODEL, temperature=0.2,
                          request_timeout=180.0, context_window=8192)
    qe = make_query_engine(index, rerank=rerank)

    rows = []
    for i, r in qdf.iterrows():
        resp = qe.query(r["Question"])
        rows.append({
            "question": r["Question"],
            "answer": str(resp).strip(),
            "contexts": [n.node.get_content() for n in resp.source_nodes],
            "reference": str(r["Ground Truth"]),
        })
        print(f"  [{i+1}/{len(qdf)}] answered")
    return pd.DataFrame(rows)


def score(df):
    from ragas import evaluate, EvaluationDataset, RunConfig
    from ragas.metrics import AnswerCorrectness
    from ragas.llms import LlamaIndexLLMWrapper
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper

    judge = LlamaIndexLLMWrapper(Ollama(model=OLLAMA_MODEL, temperature=0.0,
                                        request_timeout=180.0, context_window=8192))
    embed = LlamaIndexEmbeddingsWrapper(Settings.embed_model)

    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"], "reference": r["reference"]}
        for _, r in df.iterrows()
    ])
    # Ollama serves sequentially, so keep workers low and timeout generous
    cfg = RunConfig(timeout=300, max_workers=1, max_retries=2)
    result = evaluate(ds, metrics=[AnswerCorrectness()], llm=judge, embeddings=embed, run_config=cfg)
    df["answer_correctness"] = result.to_pandas()["answer_correctness"].values

    from bert_score import score as bert_score
    _, _, f1 = bert_score(df["answer"].tolist(), df["reference"].tolist(), lang="en")
    df["bert_f1"] = f1.numpy()
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true",
                    help="enable fetch-20 -> bge-reranker -> top-3")
    args = ap.parse_args()

    config = "reranked (fetch-20 -> top-3)" if args.rerank else "dense, top-3"
    out_csv = os.path.join(HERE, "eval_results_rerank.csv" if args.rerank else "eval_results.csv")

    qdf = pd.read_csv(QUESTIONS_CSV)
    print(f"Running {len(qdf)} questions through the RAG  [config: {config}]…")
    adf = generate_answers(qdf, rerank=args.rerank)
    print("Scoring (RAGAS AnswerCorrectness + BERTScore)…")
    scored = score(adf)
    scored.to_csv(out_csv, index=False)

    abstentions = int(scored["answer"].str.contains("don't have enough").sum())
    print(f"\n=== RESULTS (config: {config}) ===")
    print(f"Answer Correctness (mean): {scored['answer_correctness'].mean():.3f}")
    print(f"BERTScore F1 (mean):       {scored['bert_f1'].mean():.3f}")
    print(f"Abstentions:               {abstentions}/{len(scored)}")
    print(f"\nPer-question results -> {out_csv}")


if __name__ == "__main__":
    main()
