"""Pick k1 and b for Recall@20 on validation, then score every split once."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_bm25 as bm

TOP_K = 20
OUT_PATH = bm.ROOT / "data" / "processed" / "bm25_recall20.json"


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


def grid_on_validation(questions, pmid_to_row, idf, posting_docs, posting_tfs, norms):
    settings = [(k1, b) for k1 in bm.K1_VALUES for b in bm.B_VALUES]
    totals = np.zeros(len(settings), dtype=np.float64)
    used = 0
    skipped = 0
    n_docs = len(pmid_to_row)
    for question in questions:
        gold = gold_rows(question, pmid_to_row)
        if not gold:
            skipped += 1
            continue
        query = [token for token in dict.fromkeys(bm.tokenize(question["question"])) if token in idf]
        scores = np.zeros((len(settings), n_docs), dtype=np.float64)
        for token in query:
            rows = posting_docs[token]
            tfs = posting_tfs[token]
            weight = idf[token]
            for index, (k1, b) in enumerate(settings):
                ln = norms[b][rows]
                scores[index, rows] += weight * (tfs * (k1 + 1.0)) / (tfs + k1 * ln)
        gold_set = set(gold)
        for index in range(len(settings)):
            totals[index] += recall_at_k(scores[index], gold_set, TOP_K)
        used += 1
        if used % 100 == 0:
            print(f"validation grid {used}", flush=True)
    return settings, totals / used, used, skipped


def score_split(questions, k1, b, pmid_to_row, idf, posting_docs, posting_tfs, length_norm):
    recalls = []
    gold_counts = []
    skipped = 0
    n_docs = len(pmid_to_row)
    for question in questions:
        gold = gold_rows(question, pmid_to_row)
        if not gold:
            skipped += 1
            continue
        query = [token for token in dict.fromkeys(bm.tokenize(question["question"])) if token in idf]
        scores = np.zeros(n_docs, dtype=np.float64)
        for token in query:
            rows = posting_docs[token]
            tfs = posting_tfs[token]
            ln = length_norm[rows]
            scores[rows] += idf[token] * (tfs * (k1 + 1.0)) / (tfs + k1 * ln)
        recalls.append(recall_at_k(scores, set(gold), TOP_K))
        gold_counts.append(len(gold))
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
    idf, posting_docs, posting_tfs, lengths, avgdl = bm.build_index(docs)
    norms = bm.length_norms(lengths, avgdl)
    settings, recalls, used, skipped = grid_on_validation(
        grouped["validation"], pmid_to_row, idf, posting_docs, posting_tfs, norms
    )
    rows = [
        {"k1": k1, "b": b, "recall_at_20": recall}
        for (k1, b), recall in zip(settings, recalls.tolist())
    ]
    rows.sort(key=lambda row: (-row["recall_at_20"], 0 if (row["k1"], row["b"]) == bm.DEFAULT else 1))
    winner = rows[0]
    print("validation grid, Recall@20")
    for row in rows:
        mark = "  <-- chosen" if row is winner else ""
        print(f"k1={row['k1']:.1f} b={row['b']:.2f}  {row['recall_at_20']:.4f}{mark}")
    k1, b = winner["k1"], winner["b"]
    length_norm = norms[b]
    split_scores = {}
    for name in ("train", "validation", "test"):
        print(f"scoring {name}", flush=True)
        split_scores[name] = score_split(
            grouped[name], k1, b, pmid_to_row, idf, posting_docs, posting_tfs, length_norm
        )
    payload = {
        "metric": "mean Recall@20",
        "rule": "highest validation Recall@20 wins; exact ties go to k1=1.2, b=0.75",
        "validation_questions_in_grid": used,
        "validation_skipped_in_grid": skipped,
        "winner": {"k1": k1, "b": b, "validation_recall_at_20": winner["recall_at_20"]},
        "grid": rows,
        "splits": split_scores,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["splits"], indent=2))


if __name__ == "__main__":
    main()
