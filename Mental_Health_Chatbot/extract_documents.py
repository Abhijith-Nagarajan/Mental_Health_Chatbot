"""
Consolidate the per-cell PubMed JSON files into a single CSV for the RAG.

Replaces extract_documents.ipynb. Improvements:
  - carries the new fields (pmid, is_review, publication_types) alongside
    title/abstract/year, so the RAG can later weight or filter on review articles;
  - dedupes by PMID (the same paper can match several disorder/neurotransmitter
    cells) so the vector index isn't polluted with duplicate chunks;
  - records every disorder and neurotransmitter a paper is associated with, which
    is exactly the metadata needed for filtered retrieval down the line.

Run:
    .venv/bin/python extract_documents.py
"""

import os
import json

import pandas as pd

from data_scraping import DISORDERS, NEUROTRANSMITTERS

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(HERE, "pubmed_outputs")
OUT_DIR = os.path.join(HERE, "pubmed_abstracts")
OUT_CSV = os.path.join(OUT_DIR, "pubmed_abstracts.csv")


def parse_cell(filename: str):
    """Recover (disorder, neurotransmitter) labels from a `<disorder>_<nt>.json` name."""
    stem = filename[:-5] if filename.endswith(".json") else filename
    for disorder in DISORDERS:
        if stem.startswith(disorder + "_"):
            nt = stem[len(disorder) + 1:]
            if nt in NEUROTRANSMITTERS:
                return disorder, nt
    return None, None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_pmid = {}

    for filename in sorted(os.listdir(SOURCE_DIR)):
        if not filename.endswith(".json"):
            continue
        disorder, nt = parse_cell(filename)
        if disorder is None:
            print(f"  could not map file to a cell, skipping: {filename}")
            continue

        with open(os.path.join(SOURCE_DIR, filename)) as f:
            articles = json.load(f)

        for art in articles:
            abstract = (art.get("abstract") or "").strip()
            if not abstract:
                continue
            pmid = str(art.get("pmid", "")).strip()
            key = pmid or abstract[:200]  # fall back to text if a pmid is missing

            if key not in by_pmid:
                by_pmid[key] = {
                    "pmid": pmid,
                    "title": (art.get("title") or "").strip(),
                    "abstract": abstract,
                    "year": art.get("year", "Unknown"),
                    "is_review": bool(art.get("is_review", False)),
                    "publication_types": art.get("publication_types", []),
                    "disorders": set(),
                    "neurotransmitters": set(),
                }
            by_pmid[key]["disorders"].add(disorder)
            by_pmid[key]["neurotransmitters"].add(nt)

    rows = []
    for rec in by_pmid.values():
        rows.append({
            "pmid": rec["pmid"],
            "title": rec["title"],
            "abstract": rec["abstract"],
            "year": rec["year"],
            "is_review": rec["is_review"],
            "disorders": "; ".join(sorted(rec["disorders"])),
            "neurotransmitters": "; ".join(sorted(rec["neurotransmitters"])),
            "publication_types": "; ".join(rec["publication_types"]),
        })

    df = pd.DataFrame(rows, columns=[
        "pmid", "title", "abstract", "year", "is_review",
        "disorders", "neurotransmitters", "publication_types",
    ])
    df.to_csv(OUT_CSV, index=False)

    print(f"Unique abstracts: {len(df)}")
    print(f"Review articles:  {int(df['is_review'].sum())} ({df['is_review'].mean():.0%})")
    print(f"Multi-disorder papers: {(df['disorders'].str.contains(';')).sum()}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
