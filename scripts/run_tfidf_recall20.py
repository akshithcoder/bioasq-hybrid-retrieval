"""Recall@20 with L2-normalized TF-IDF cosine similarity.

No knobs are chosen here. The same formula is scored on train, validation,
and test. A term weight is (term count) times (idf). Each article vector and
each question vector is divided by its length, and the score is their dot
product, which is the cosine.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_bm25 as bm

TOP_K = 20
OUT_PATH = bm.ROOT / "data" / "processed" / "tfidf_length_recall20.json"


def gold_rows(question, pmid_to_row) -> list[int]:
    rows = []
    for url in question["documents"]:
        pmid = bm.pmid_from_url(url)
        row = pmid_to_row.get(pmid) if pmid else None
        if row is not None and row not in rows:
            rows.append(row)
    return rows


def recall_at_k(scores: np.ndarray, gold: set[int], top_k: int) -> float:
    top = np.argpartition(scores, -top_k)[-top_k:]
    found = {int(row) for row in top if scores[row] > 0}
    return len(gold.intersection(found)) / len(gold)


def build_normalized_weights(posting_docs, posting_tfs, lengths, n_docs):
    idf = {}
    for token, rows in posting_docs.items():
        df = len(rows)
        idf[token] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
    squared = np.zeros(n_docs, dtype=np.float64)
    raw_weights = {}
    for token, rows in posting_docs.items():
        # tf = count / number of words in the article
        weights = (posting_tfs[token] / lengths[rows]) * idf[token]
        raw_weights[token] = weights
        np.add.at(squared, rows, weights * weights)
    norms = np.sqrt(squared)
    norms[norms == 0] = 1.0
    normalized = {}
    for token, rows in posting_docs.items():
        normalized[token] = (raw_weights[token] / norms[rows]).astype(np.float32)
    return idf, normalized


def score_split(questions, pmid_to_row, idf, posting_docs, normalized):
    recalls = []
    gold_counts = []
    skipped = 0
    n_docs = len(pmid_to_row)
    for question in questions:
        gold = gold_rows(question, pmid_to_row)
        if not gold:
            skipped += 1
            continue
        tokens = [token for token in bm.tokenize(question["question"]) if token in idf]
        counts = Counter(tokens)
        scores = np.zeros(n_docs, dtype=np.float64)
        if counts:
            query_length = float(len(tokens))
            query_weights = {
                token: (tf / query_length) * idf[token] for token, tf in counts.items()
            }
            query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
            for token, weight in query_weights.items():
                scores[posting_docs[token]] += (weight / query_norm) * normalized[token]
        recalls.append(recall_at_k(scores, set(gold), TOP_K))
        gold_counts.append(len(gold))
        if len(recalls) % 500 == 0:
            print(f"scored {len(recalls)}", flush=True)
    ordered = sorted(gold_counts)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        median_gold = float(ordered[mid])
    else:
        median_gold = (ordered[mid - 1] + ordered[mid]) / 2
    return {
        "questions_scored": len(recalls),
        "questions_skipped_no_gold_in_corpus": skipped,
        "median_gold_articles": median_gold,
        "recall_at_20": float(sum(recalls) / len(recalls)),
    }


def main() -> None:
    questions = bm.load_questions()
    grouped = {
        "train": [q for q in questions if int(q["asq_challenge"]) <= 11],
        "validation": [q for q in questions if int(q["asq_challenge"]) == 12],
        "test": [q for q in questions if int(q["asq_challenge"]) == 13],
    }
    pmids, docs = bm.load_corpus()
    pmid_to_row = {pmid: row for row, pmid in enumerate(pmids)}
    print(f"indexed {len(pmids)} articles", flush=True)
    _idf_bm25, posting_docs, posting_tfs, lengths, _avgdl = bm.build_index(docs)
    idf, normalized = build_normalized_weights(
        posting_docs, posting_tfs, lengths, len(pmids)
    )
    split_scores = {}
    for name in ("train", "validation", "test"):
        print(f"scoring {name}", flush=True)
        split_scores[name] = score_split(
            grouped[name], pmid_to_row, idf, posting_docs, normalized
        )
        print(name, split_scores[name], flush=True)
    payload = {
        "metric": "mean Recall@20",
        "method": "length-normalized tf-idf, then cosine",
        "tf": "count(word) / number of words",
        "idf": "ln((1+N)/(1+df))+1",
        "normalization": "after tf-idf, divide each vector by its L2 norm so the score is the cosine",
        "tuned": False,
        "splits": split_scores,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
