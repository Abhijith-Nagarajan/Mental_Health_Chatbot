"""
PubMed scraper for the mental-health RAG corpus.

Replaces the flat "25 abstracts per pair" logic in data_scraping.ipynb with a
sampling strategy that accounts for how the literature is actually distributed:

  - Cell-size scaling : big associations (e.g. Schizophrenia+Dopamine ~2300 papers)
                        get more abstracts; tiny ones (e.g. OCD+Acetylcholine ~5)
                        just take everything available.
  - Review-weighting  : a share of each cell is filled from review articles, which
                        capture field consensus rather than a random primary study.
  - Recency           : results are sorted by publication date so the corpus
                        reflects current understanding.

Also fixes the PTSD bug: the old code used "Post-Traumatic Stress Disorder"[MeSH],
which is not a valid MeSH descriptor and silently returned 0 results. The correct
heading is "Stress Disorders, Post-Traumatic"[MeSH].

Run:
    export NCBI_EMAIL="you@example.com"      # required by Entrez
    export NCBI_API_KEY="..."                # optional, raises rate limit to 10/s
    .venv/bin/python data_scraping.py

Writes one JSON file per (disorder, neurotransmitter) cell into pubmed_outputs/,
overwriting existing files in place. Each record carries an `is_review` flag so the
downstream pipeline can weight or filter on it.
"""

import os
import json
import math
import time

from Bio import Entrez

# ---- the answerable grid: clean label -> exact MeSH descriptor ----
DISORDERS = {
    "Anxiety_Disorders": "Anxiety Disorders",
    "Depressive_Disorder": "Depressive Disorder",
    "Bipolar_Disorder": "Bipolar Disorder",
    "Schizophrenia": "Schizophrenia",
    "Obsessive-Compulsive_Disorder": "Obsessive-Compulsive Disorder",
    "PTSD": "Stress Disorders, Post-Traumatic",   # fixed MeSH heading
}

NEUROTRANSMITTERS = {
    "Dopamine": "Dopamine",
    "Serotonin": "Serotonin",
    "Gamma-Aminobutyric_Acid": "gamma-Aminobutyric Acid",
    "Norepinephrine": "Norepinephrine",
    "Glutamic_Acid": "Glutamic Acid",
    "Acetylcholine": "Acetylcholine",
    "Endorphins": "Endorphins",
}

# ---- sampling knobs ----
MIN_PER_CELL = 30        # floor for cells that have enough literature
MAX_PER_CELL = 150       # ceiling: past this, more abstracts are mostly redundant for QA
SCALE_K = 3.35           # target ~= SCALE_K * sqrt(count); tuned so ~2000 -> ~150
REVIEW_FRACTION = 0.4    # share of each cell drawn from review articles

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "pubmed_outputs")

def target_for_cell(count: int) -> int:
    """How many abstracts to pull for a cell, given how many exist on PubMed."""
    if count <= 0:
        return 0
    scaled = int(round(SCALE_K * math.sqrt(count)))
    return min(count, max(MIN_PER_CELL, min(MAX_PER_CELL, scaled)))


def esearch_count(term: str) -> int:
    handle = Entrez.esearch(db="pubmed", term=term, retmax=0)
    record = Entrez.read(handle)
    handle.close()
    return int(record["Count"])


def esearch_ids(term: str, retmax: int, sort: str = "pub_date") -> list:
    """Return PMIDs for a query, most recent first. Falls back if sort is rejected."""
    if retmax <= 0:
        return []
    try:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, sort=sort)
        record = Entrez.read(handle)
    except Exception:
        handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax)
        record = Entrez.read(handle)
    handle.close()
    return list(record["IdList"])


def fetch_articles(pmids: list) -> list:
    """Fetch and parse abstracts; skips records with no abstract text."""
    if not pmids:
        return []
    handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
    records = Entrez.read(handle)
    handle.close()

    out = []
    for art in records.get("PubmedArticle", []):
        try:
            citation = art["MedlineCitation"]
            article = citation["Article"]

            raw_abstract = article.get("Abstract", {}).get("AbstractText", [])
            if isinstance(raw_abstract, list):
                abstract = " ".join(str(part).strip() for part in raw_abstract).strip()
            else:
                abstract = str(raw_abstract).strip()
            if not abstract:
                continue  # no abstract is useless for retrieval

            pub_date = article["Journal"]["JournalIssue"]["PubDate"]
            year = pub_date.get("Year")
            if not year:
                medline_date = pub_date.get("MedlineDate", "")
                year = medline_date[:4] if medline_date[:4].isdigit() else "Unknown"

            pub_types = [str(pt) for pt in article.get("PublicationTypeList", [])]

            out.append({
                "pmid": str(citation["PMID"]),
                "title": str(article.get("ArticleTitle", "")).strip(),
                "abstract": abstract,
                "year": year,
                "is_review": "Review" in pub_types,
                "publication_types": pub_types,
            })
        except Exception as exc:
            print(f"    skipped an article: {exc}")
    return out


def main():
    Entrez.email = os.environ.get("NCBI_EMAIL") or input("NCBI email (required): ").strip()
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key
    pause = 0.12 if api_key else 0.34   # stay under NCBI's 3/s (10/s with a key)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_saved = 0
    total_reviews = 0

    for d_label, d_mesh in DISORDERS.items():
        for n_label, n_mesh in NEUROTRANSMITTERS.items():
            cell = f'"{d_mesh}"[MeSH] AND "{n_mesh}"[MeSH]'

            count = esearch_count(cell)
            time.sleep(pause)
            target = target_for_cell(count)
            if target == 0:
                print(f"{d_label} x {n_label}: 0 results — skipping")
                continue

            # split the target between reviews and recent primary studies;
            # backfill from primary if the cell has few reviews
            n_reviews = round(target * REVIEW_FRACTION)
            review_ids = esearch_ids(f"{cell} AND review[pt]", n_reviews)
            time.sleep(pause)
            primary_ids = esearch_ids(f"{cell} NOT review[pt]", target - len(review_ids))
            time.sleep(pause)

            ids = list(dict.fromkeys(review_ids + primary_ids))  # dedupe, keep order
            articles = fetch_articles(ids)
            time.sleep(pause)

            out_path = os.path.join(OUTPUT_DIR, f"{d_label}_{n_label}.json")
            with open(out_path, "w") as f:
                json.dump(articles, f, indent=2)

            reviews_kept = sum(a["is_review"] for a in articles)
            total_saved += len(articles)
            total_reviews += reviews_kept
            print(f"{d_label} x {n_label}: pool={count}, target={target}, "
                  f"saved={len(articles)} ({reviews_kept} reviews)")

    print(f"\nDone. {total_saved} abstracts across {len(DISORDERS)}x{len(NEUROTRANSMITTERS)} "
          f"cells ({total_reviews} reviews) -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
