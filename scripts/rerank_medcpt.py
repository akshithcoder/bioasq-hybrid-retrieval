"""Rerank BM25 candidates with the frozen MedCPT cross-encoder."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_cache as cache
import tune_bm25 as bm

K1 = 0.9
B = 0.4
MODEL_NAME = "ncbi/MedCPT-Cross-Encoder"
MODEL_REVISION = "71caf65d4927987813984f54c284405a13fcca49"
CANDIDATE_DEPTHS = (50, 100, 200)
MAX_DEPTH = 200
RECALL_K = 20
BATCH_SIZE = 16
MAX_LENGTH = 512
SCORE_DIR = bm.ROOT / "data" / "processed" / "rerank"
OUT_PATH = bm.ROOT / "data" / "processed" / "medcpt_rerank_recall20.json"


def article_text(record: dict) -> str:
    title = (record.get("title") or "").strip()
    abstract = (record.get("abstract") or "").strip()
    return f"{title} {abstract}".strip()


def load_texts() -> dict[str, str]:
    texts = {}
    with bm.ABSTRACTS_PATH.open() as handle:
        for line in handle:
            record = json.loads(line)
            text = article_text(record)
            if text:
                texts[record["pmid"]] = text
    return texts


def gold_pmids(question) -> list[str]:
    found = []
    for url in question["documents"]:
        pmid = bm.pmid_from_url(url)
        if pmid and pmid not in found:
            found.append(pmid)
    return found


def bm25_top(question, pmid_to_row, row_to_pmid, idf, posting_docs, posting_tfs, length_norm, depth):
    query = [token for token in dict.fromkeys(bm.tokenize(question["question"])) if token in idf]
    scores = np.zeros(len(pmid_to_row), dtype=np.float64)
    for token in query:
        rows = posting_docs[token]
        tfs = posting_tfs[token]
        scores[rows] += idf[token] * (tfs * (K1 + 1.0)) / (tfs + K1 * length_norm[rows])
    chosen = np.argpartition(scores, -depth)[-depth:]
    order = chosen[np.argsort(-scores[chosen])]
    return [row_to_pmid[int(row)] for row in order]


def recall_at_20(ranked_pmids: list[str], gold: set[str]) -> float:
    return len(gold.intersection(ranked_pmids[:RECALL_K])) / len(gold)


def questions_for(split: str) -> list[dict]:
    questions = bm.load_questions()
    if split == "validation":
        return [q for q in questions if int(q["asq_challenge"]) == 12]
    if split == "test":
        return [q for q in questions if int(q["asq_challenge"]) == 13]
    raise ValueError(split)


def score_path(split: str) -> Path:
    return SCORE_DIR / f"{split}_top{MAX_DEPTH}.jsonl"


def load_scored(path: Path) -> dict[str, dict]:
    return cache.load_rankings(
        path,
        min_candidates=MAX_DEPTH,
        max_candidates=MAX_DEPTH,
        require_logits=True,
    )


def score_pairs(pairs, tokenizer, model, device):
    logits = []
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = pairs[start : start + BATCH_SIZE]
        encoded = tokenizer(
            batch,
            truncation=True,
            padding=True,
            return_tensors="pt",
            max_length=MAX_LENGTH,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            batch_logits = model(**encoded).logits.view(-1).detach().cpu().tolist()
        logits.extend(batch_logits)
    return logits


def ensure_scores(split: str, depth: int, texts, pmid_to_row, row_to_pmid, idf, posting_docs, posting_tfs, length_norm, tokenizer, model, device):
    path = score_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    questions = []
    for question in questions_for(split):
        gold = [pmid for pmid in gold_pmids(question) if pmid in pmid_to_row]
        if gold:
            questions.append((question, gold))
    expected_ids = [question["id"] for question, _ in questions]
    scored = load_scored(path)
    missing = set(cache.missing_question_ids(scored, expected_ids, source=str(path)))
    expected_gold = {question["id"]: gold for question, gold in questions}
    for qid, saved in scored.items():
        if saved["gold"] != expected_gold[qid]:
            raise ValueError(f"{path}: stale gold PMIDs for {qid}")
    pending = [(question, gold) for question, gold in questions if question["id"] in missing]
    print(f"{split}: {len(scored)} already scored, {len(pending)} left", flush=True)
    with path.open("a") as handle:
        for index, (question, gold) in enumerate(pending, start=1):
            pmids = bm25_top(
                question, pmid_to_row, row_to_pmid, idf, posting_docs, posting_tfs, length_norm, depth
            )
            pairs = [[question["question"], texts[pmid]] for pmid in pmids]
            logits = score_pairs(pairs, tokenizer, model, device)
            row = {"id": question["id"], "gold": gold, "pmids": pmids, "logits": logits}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            if index % 10 == 0:
                print(f"{split}: scored {index}/{len(pending)}", flush=True)
    scored = load_scored(path)
    cache.require_complete(scored, expected_ids, source=str(path))
    return scored


def mean_recall(rows: list[dict], depth: int) -> float:
    recalls = []
    for row in rows:
        order = np.argsort(-np.array(row["logits"][:depth], dtype=np.float64))
        ranked = [row["pmids"][int(i)] for i in order]
        recalls.append(recall_at_20(ranked, set(row["gold"])))
    return float(sum(recalls) / len(recalls))


def bm25_mean_recall(rows: list[dict]) -> float:
    recalls = [recall_at_20(row["pmids"], set(row["gold"])) for row in rows]
    return float(sum(recalls) / len(recalls))


def build_search():
    texts = load_texts()
    pmids, docs = bm.load_corpus()
    pmid_to_row = {pmid: row for row, pmid in enumerate(pmids)}
    row_to_pmid = {row: pmid for pmid, row in pmid_to_row.items()}
    idf, posting_docs, posting_tfs, lengths, avgdl = bm.build_index(docs)
    length_norm = (1 - B) + B * (lengths / avgdl)
    return texts, pmid_to_row, row_to_pmid, idf, posting_docs, posting_tfs, length_norm


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model():
    device = best_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model.to(device)
    model.eval()
    print(f"model on {device}", flush=True)
    return tokenizer, model, device


def main() -> None:
    texts, pmid_to_row, row_to_pmid, idf, posting_docs, posting_tfs, length_norm = build_search()
    tokenizer, model, device = load_model()
    validation = ensure_scores(
        "validation",
        MAX_DEPTH,
        texts,
        pmid_to_row,
        row_to_pmid,
        idf,
        posting_docs,
        posting_tfs,
        length_norm,
        tokenizer,
        model,
        device,
    )
    rows = list(validation.values())
    grid = {depth: mean_recall(rows, depth) for depth in CANDIDATE_DEPTHS}
    winner = max(grid, key=grid.get)
    print("validation Recall@20", flush=True)
    print(f"bm25 top 20: {bm25_mean_recall(rows):.4f}", flush=True)
    for depth, recall in grid.items():
        mark = "  <-- chosen" if depth == winner else ""
        print(f"rerank top {depth}: {recall:.4f}{mark}", flush=True)

    test = ensure_scores(
        "test",
        winner,
        texts,
        pmid_to_row,
        row_to_pmid,
        idf,
        posting_docs,
        posting_tfs,
        length_norm,
        tokenizer,
        model,
        device,
    )
    test_rows = list(test.values())
    payload = {
        "model": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "bm25": {"k1": K1, "b": B},
        "metric": "mean Recall@20",
        "rule": "highest validation Recall@20 over candidate depths 50, 100, 200",
        "validation_bm25_recall_at_20": bm25_mean_recall(rows),
        "validation_grid": {str(depth): recall for depth, recall in grid.items()},
        "chosen_depth": winner,
        "validation_recall_at_20": grid[winner],
        "test_bm25_recall_at_20": bm25_mean_recall(test_rows),
        "test_recall_at_20": mean_recall(test_rows, winner),
        "validation_questions": len(rows),
        "test_questions": len(test_rows),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
