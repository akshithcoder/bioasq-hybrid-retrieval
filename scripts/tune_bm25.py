"""Tune BM25 k1 and b on edition-12 validation questions."""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "raw" / "bioasq-14b" / "training14b.json"
ABSTRACTS_PATH = ROOT / "data" / "processed" / "abstracts.jsonl"
SPLIT_PATH = ROOT / "data" / "processed" / "splits.json"
GRID_PATH = ROOT / "data" / "processed" / "bm25_val_grid.json"

K1_VALUES = (0.9, 1.2, 1.5, 2.0)
B_VALUES = (0.4, 0.75, 1.0)
DEFAULT = (1.2, 0.75)
TOP_K = 10
TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def pmid_from_url(url: str) -> str | None:
    tail = url.rstrip("/").split("/")[-1]
    return tail if tail.isdigit() else None


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text())["questions"]


def assign_split(edition: int) -> str:
    if edition <= 11:
        return "train"
    if edition == 12:
        return "validation"
    if edition == 13:
        return "test"
    raise ValueError(f"unexpected edition {edition}")


def save_splits(questions: list[dict]) -> dict[str, list[dict]]:
    grouped = {"train": [], "validation": [], "test": []}
    for question in questions:
        split = assign_split(int(question["asq_challenge"]))
        grouped[split].append(question)
    payload = {
        "rule": "train: editions 1-11; validation: edition 12; test: edition 13",
        "counts": {name: len(items) for name, items in grouped.items()},
        "ids": {name: [item["id"] for item in items] for name, items in grouped.items()},
    }
    SPLIT_PATH.write_text(json.dumps(payload))
    return grouped


def load_corpus() -> tuple[list[str], list[list[str]]]:
    pmids = []
    docs = []
    with ABSTRACTS_PATH.open() as handle:
        for line in handle:
            record = json.loads(line)
            text = f"{record.get('title') or ''} {record.get('abstract') or ''}".strip()
            if not text:
                continue
            pmids.append(record["pmid"])
            docs.append(tokenize(text))
    return pmids, docs


def build_index(docs: list[list[str]]):
    df = defaultdict(int)
    postings = defaultdict(list)
    lengths = np.zeros(len(docs), dtype=np.float64)
    for doc_id, tokens in enumerate(docs):
        lengths[doc_id] = len(tokens)
        counts = defaultdict(int)
        for token in tokens:
            counts[token] += 1
        for token, tf in counts.items():
            df[token] += 1
            postings[token].append((doc_id, tf))
    n_docs = len(docs)
    avgdl = float(lengths.mean())
    idf = {
        token: math.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in df.items()
    }
    posting_docs = {}
    posting_tfs = {}
    for token, pairs in postings.items():
        posting_docs[token] = np.fromiter((doc_id for doc_id, _ in pairs), dtype=np.int32)
        posting_tfs[token] = np.fromiter((tf for _, tf in pairs), dtype=np.float64)
    return idf, posting_docs, posting_tfs, lengths, avgdl


def length_norms(lengths: np.ndarray, avgdl: float) -> dict[float, np.ndarray]:
    norms = {}
    for b in B_VALUES:
        norms[b] = (1 - b) + b * (lengths / avgdl)
    return norms


def validation_recall(questions, pmid_to_row, idf, posting_docs, posting_tfs, norms):
    settings = [(k1, b) for k1 in K1_VALUES for b in B_VALUES]
    totals = np.zeros(len(settings), dtype=np.float64)
    used = 0
    skipped = 0
    for question in questions:
        gold = []
        for url in question["documents"]:
            pmid = pmid_from_url(url)
            row = pmid_to_row.get(pmid) if pmid else None
            if row is not None and row not in gold:
                gold.append(row)
        if not gold:
            skipped += 1
            continue
        query = [token for token in dict.fromkeys(tokenize(question["question"])) if token in idf]
        if not query:
            totals += 0.0
            used += 1
            continue
        scores = np.zeros((len(settings), len(pmid_to_row)), dtype=np.float64)
        for token in query:
            rows = posting_docs[token]
            tfs = posting_tfs[token]
            weight = idf[token]
            for index, (k1, b) in enumerate(settings):
                ln = norms[b][rows]
                scores[index, rows] += weight * (tfs * (k1 + 1.0)) / (tfs + k1 * ln)
        gold_set = set(gold)
        for index in range(len(settings)):
            top = np.argpartition(scores[index], -TOP_K)[-TOP_K:]
            # A score of zero means the article shared no query word.
            # Those articles are not part of the top 10.
            found = {int(row) for row in top if scores[index, row] > 0}
            totals[index] += len(gold_set.intersection(found)) / len(gold_set)
        used += 1
        if used % 50 == 0:
            print(f"scored {used} validation questions", flush=True)
    return settings, totals / used, used, skipped


def main() -> None:
    questions = load_questions()
    grouped = save_splits(questions)
    print(
        "splits",
        {name: len(items) for name, items in grouped.items()},
        flush=True,
    )
    print("training questions are saved and not used to pick k1 or b", flush=True)
    pmids, docs = load_corpus()
    pmid_to_row = {pmid: row for row, pmid in enumerate(pmids)}
    print(f"indexed {len(pmids)} articles", flush=True)
    idf, posting_docs, posting_tfs, lengths, avgdl = build_index(docs)
    print(f"average article length {avgdl:.1f} words", flush=True)
    settings, recalls, used, skipped = validation_recall(
        grouped["validation"],
        pmid_to_row,
        idf,
        posting_docs,
        posting_tfs,
        length_norms(lengths, avgdl),
    )
    rows = [
        {"k1": k1, "b": b, "recall_at_10": recall}
        for (k1, b), recall in zip(settings, recalls.tolist())
    ]
    rows.sort(key=lambda row: (-row["recall_at_10"], 0 if (row["k1"], row["b"]) == DEFAULT else 1))
    winner = rows[0]
    payload = {
        "scored_split": "validation",
        "edition": 12,
        "questions_scored": used,
        "questions_skipped_no_gold_in_corpus": skipped,
        "metric": "mean Recall@10",
        "rule": "highest validation Recall@10 wins; exact ties go to k1=1.2, b=0.75",
        "test_was_scored": False,
        "winner": winner,
        "grid": rows,
    }
    GRID_PATH.write_text(json.dumps(payload, indent=2))
    print(f"validation questions scored {used}, skipped {skipped}", flush=True)
    print(f"{'k1':>6} {'b':>6} {'Recall@10':>10}")
    for row in rows:
        mark = "  <-- chosen" if row is winner else ""
        print(f"{row['k1']:6.1f} {row['b']:6.2f} {row['recall_at_10']:10.4f}{mark}")


if __name__ == "__main__":
    main()
