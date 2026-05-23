"""
Localize the low Answer Correctness (0.41) by decomposing the RAGAS metrics,
BEFORE deciding whether a reranker is even the right fix.

Runs on the existing eval_results.csv — no regeneration:
  - Context Precision / Recall  -> retrieval quality (are the right chunks reaching the LLM?)
  - Faithfulness                -> generation grounding (does the answer stay in the context?)

Reading the result:
  * Context P/R low                       -> retrieval bottleneck    -> reranker may help
  * Context P/R high + Faithfulness low   -> generator hallucinates  -> reranker irrelevant
  * Context P/R high + Faithfulness high  -> grounded but incomplete -> generator/coverage, not ranking

Run:
    ollama serve
    .venv/bin/python diagnose.py
"""

import os
import ast

import pandas as pd
from llama_index.llms.ollama import Ollama
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
from ragas.llms import LlamaIndexLLMWrapper

from build_application import OLLAMA_MODEL

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(HERE, "eval_results.csv")


def to_list(val):
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return [str(val)]


def main():
    df = pd.read_csv(RESULTS_CSV)
    judge = LlamaIndexLLMWrapper(Ollama(model=OLLAMA_MODEL, temperature=0.0,
                                        request_timeout=180.0, context_window=8192))
    ds = EvaluationDataset.from_list([
        {"user_input": r["question"], "response": r["answer"],
         "retrieved_contexts": to_list(r["contexts"]), "reference": r["reference"]}
        for _, r in df.iterrows()
    ])
    metrics = [Faithfulness(), LLMContextPrecisionWithReference(), LLMContextRecall()]
    res = evaluate(ds, metrics=metrics, llm=judge,
                   run_config=RunConfig(timeout=300, max_workers=1, max_retries=2))

    rdf = res.to_pandas()
    inputs = {"user_input", "response", "retrieved_contexts", "reference"}
    metric_cols = [c for c in rdf.columns if c not in inputs]
    for c in metric_cols:
        df[c] = rdf[c].values
    df.to_csv(RESULTS_CSV, index=False)

    means = {c: df[c].mean() for c in metric_cols}
    ac = df["answer_correctness"].mean()

    print("\n=== METRIC DECOMPOSITION (n=%d) ===" % len(df))
    print(f"Answer Correctness : {ac:.3f}")
    for c in metric_cols:
        print(f"{c:35s}: {means[c]:.3f}")

    # locate the relevant means by keyword (column names vary by ragas version)
    prec = next((v for k, v in means.items() if "precision" in k), None)
    rec = next((v for k, v in means.items() if "recall" in k), None)
    faith = next((v for k, v in means.items() if "faith" in k), None)
    retrieval = [v for v in (prec, rec) if v is not None]
    retrieval_ok = retrieval and (sum(retrieval) / len(retrieval)) >= 0.6

    print("\n=== VERDICT ===")
    if retrieval and not retrieval_ok:
        print("Retrieval bottleneck: the right chunks aren't reaching the LLM -> a reranker MAY help.")
    elif faith is not None and faith < 0.6:
        print("Generator problem: answers aren't grounded in the (good) context -> reranker WON'T help;")
        print("look at the generation model / prompt (quantization, grounding).")
    else:
        print("Retrieval is fine and answers are grounded, but still don't match the references ->")
        print("generation coverage / reference strictness, not ranking. Reranker unlikely to move the needle.")


if __name__ == "__main__":
    main()
