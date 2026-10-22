"""Score edition 13 once, with the k1 and b chosen on validation."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_bm25 as bm

K1 = 0.9
B = 0.4
TOP_K = 10
OUT_PATH = bm.ROOT / "data" / "processed" / "bm25_test.json"


def score_questions(questions, pmid_to_row, idf, posting_docs, posting_tfs, length_norm):
    per_question = []
    skipped = 0
    for question in questions:
        gold = []
        for url in question["documents"]:
            pmid = bm.pmid_from_url(url)
            row = pmid_to_row.get(pmid) if pmid else None
            if row is not None and row not in gold:
                gold.append(row)
        if not gold:
            skipped += 1
            continue
        query = [token for token in dict.fromkeys(bm.tokenize(question["question"])) if token in idf]
        scores = np.zeros(len(pmid_to_row), dtype=np.float64)
        for token in query:
            rows = posting_docs[token]
            tfs = posting_tfs[token]
            ln = length_norm[rows]
            scores[rows] += idf[token] * (tfs * (K1 + 1.0)) / (tfs + K1 * ln)
        top = np.argpartition(scores, -TOP_K)[-TOP_K:]
        found_rows = {int(row) for row in top if scores[row] > 0}
        hits = len(set(gold).intersection(found_rows))
        per_question.append(
            {
                "id": question["id"],
                "type": question["type"],
                "n_gold": len(gold),
                "hits_at_10": hits,
                "recall_at_10": hits / len(gold),
                "precision_at_10": hits / TOP_K,
            }
        )
    return per_question, skipped


def summarize(rows: list[dict]) -> dict:
    recalls = [row["recall_at_10"] for row in rows]
    precisions = [row["precision_at_10"] for row in rows]
    golds = sorted(row["n_gold"] for row in rows)
    by_type = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["type"]].append(row)
    for name, items in sorted(grouped.items()):
        by_type[name] = {
            "questions": len(items),
            "median_gold_articles": median([item["n_gold"] for item in items]),
            "recall_at_10": mean([item["recall_at_10"] for item in items]),
            "precision_at_10": mean([item["precision_at_10"] for item in items]),
        }
    return {
        "questions": len(rows),
        "median_gold_articles": median(golds),
        "recall_at_10": mean(recalls),
        "precision_at_10": mean(precisions),
        "by_type": by_type,
    }


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def median(values: list[int]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


def main() -> None:
    questions = bm.load_questions()
    test = [question for question in questions if int(question["asq_challenge"]) == 13]
    pmids, docs = bm.load_corpus()
    pmid_to_row = {pmid: row for row, pmid in enumerate(pmids)}
    idf, posting_docs, posting_tfs, lengths, avgdl = bm.build_index(docs)
    length_norm = (1 - B) + B * (lengths / avgdl)
    rows, skipped = score_questions(test, pmid_to_row, idf, posting_docs, posting_tfs, length_norm)
    summary = summarize(rows)
    payload = {
        "split": "test",
        "edition": 13,
        "k1": K1,
        "b": B,
        "chosen_on": "validation edition 12",
        "questions_skipped_no_gold_in_corpus": skipped,
        **summary,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
