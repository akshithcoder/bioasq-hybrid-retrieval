"""Rerank the union of BM25's top 200 and the bi-encoder's top 200.

The adapter is the one trained on BM25's list. No new training.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank_medcpt as rr
import rerank_medcpt_bi as bi
import train_lora_reranker as lora
import tune_bm25 as bm

OUT_DIR = bm.ROOT / "data" / "processed" / "union_lora"
SUMMARY_PATH = bm.ROOT / "data" / "processed" / "union_lora_recall20.json"


def fmt(seconds: float) -> str:
    return lora.fmt(seconds)


def load_jsonl(path: Path) -> dict[str, dict]:
    return lora.load_jsonl(path)


def union_rows(split: str) -> list[dict]:
    bm25_rows = load_jsonl(rr.score_path(split))
    bi_rows = load_jsonl(bi.candidate_path(split))
    questions = {q["id"]: q["question"] for q in bm.load_questions()}
    rows = []
    for qid, bm25 in bm25_rows.items():
        other = bi_rows[qid]
        if bm25["gold"] != other["gold"]:
            raise ValueError(f"gold mismatch for {qid}")
        seen = []
        placed = set()
        for pmid in list(bm25["pmids"]) + list(other["pmids"]):
            if pmid not in placed:
                placed.add(pmid)
                seen.append(pmid)
        rows.append(
            {
                "id": qid,
                "question": questions[qid],
                "gold": bm25["gold"],
                "pmids": seen,
            }
        )
    return rows


def score_split(split: str, texts: dict[str, str], tokenizer, model, device) -> list[dict]:
    source = union_rows(split)
    out_path = OUT_DIR / f"{split}_union.jsonl"
    done = load_jsonl(out_path)
    pending = [row for row in source if row["id"] not in done]
    print(
        f"{split}: {len(done)} already scored, {len(pending)} left, "
        f"mean list {sum(len(row['pmids']) for row in source) / len(source):.1f}",
        flush=True,
    )
    lora.enable_lora_dropout(model, False)
    with out_path.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            pairs = [[row["question"], texts[pmid]] for pmid in row["pmids"]]
            logits = rr.score_pairs(pairs, tokenizer, model, device)
            handle.write(
                json.dumps(
                    {"id": row["id"], "gold": row["gold"], "pmids": row["pmids"], "logits": logits}
                )
                + "\n"
            )
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"{split}: scored {index}/{len(pending)}", flush=True)
    saved = load_jsonl(out_path)
    return [saved[row["id"]] for row in source]


def mean_recall(rows: list[dict], k: int = 20) -> float:
    scores = []
    for row in rows:
        gold = set(row["gold"])
        order = np.argsort(-np.array(row["logits"], dtype=np.float64))
        ranked = [row["pmids"][int(index)] for index in order]
        scores.append(len(gold.intersection(ranked[:k])) / len(gold))
    return float(sum(scores) / len(scores))


def full_coverage(rows: list[dict], k: int = 20) -> float:
    hits = 0
    for row in rows:
        gold = set(row["gold"])
        order = np.argsort(-np.array(row["logits"], dtype=np.float64))
        ranked = [row["pmids"][int(index)] for index in order]
        hits += gold.issubset(ranked[:k])
    return hits / len(rows)


def main() -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    texts = rr.load_texts()
    tokenizer, model, device = lora.prepare_model()
    saved = torch.load(lora.ADAPTER_PATH, map_location="cpu", weights_only=False)
    lora.load_lora_state(model, saved["lora"], device)
    lora.enable_lora_dropout(model, False)
    print(f"loaded BM25 LoRA adapter from {lora.ADAPTER_PATH}", flush=True)
    payload = {
        "candidates": "union of BM25 top 200 and MedCPT bi-encoder top 200",
        "reranker": "LoRA trained on BM25 top 200",
        "adapter": str(lora.ADAPTER_PATH),
    }
    for split in ("validation", "test"):
        rows = score_split(split, texts, tokenizer, model, device)
        payload[split] = {
            "questions": len(rows),
            "mean_list_size": sum(len(row["pmids"]) for row in rows) / len(rows),
            "recall_at_20": mean_recall(rows, 20),
            "full_coverage_at_20": full_coverage(rows, 20),
        }
        print(
            f"{split} Recall@20 {payload[split]['recall_at_20']:.4f}  "
            f"full coverage {payload[split]['full_coverage_at_20']:.3%}",
            flush=True,
        )
    payload["seconds"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    print(f"TOTAL {fmt(payload['seconds'])}", flush=True)


if __name__ == "__main__":
    main()
