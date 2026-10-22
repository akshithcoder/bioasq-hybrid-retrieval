"""First-stage MedCPT bi-encoder, then the trained LoRA cross-encoder.

The bi-encoder embeds every indexed article once and each question once.
Dot product keeps the top 200. The existing LoRA adapter reranks that list.
No new training. Query length 64 and article length 512 are MedCPT's settings.
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
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank_medcpt as rr
import train_lora_reranker as lora
import tune_bm25 as bm

QUERY_MODEL = "ncbi/MedCPT-Query-Encoder"
ARTICLE_MODEL = "ncbi/MedCPT-Article-Encoder"
QUERY_MAX_LENGTH = 64
ARTICLE_MAX_LENGTH = 512
DEPTH = 200
ARTICLE_BATCH = 32
OUT_DIR = bm.ROOT / "data" / "processed" / "medcpt_bi"
EMBED_PATH = OUT_DIR / "article_embeds.npy"
PMID_PATH = OUT_DIR / "pmids.json"
SUMMARY_PATH = bm.ROOT / "data" / "processed" / "medcpt_bi_recall.json"


def fmt(seconds: float) -> str:
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def load_articles() -> list[dict]:
    articles = []
    with bm.ABSTRACTS_PATH.open() as handle:
        for line in handle:
            record = json.loads(line)
            title = (record.get("title") or "").strip()
            abstract = (record.get("abstract") or "").strip()
            if not f"{title} {abstract}".strip():
                continue
            articles.append({"pmid": record["pmid"], "title": title, "abstract": abstract})
    return articles


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows[row["id"]] = row
    return rows


def questions_for(edition: int) -> list[dict]:
    return [q for q in bm.load_questions() if int(q["asq_challenge"]) == edition]


def embed_articles(articles: list[dict]) -> np.ndarray:
    if EMBED_PATH.exists() and PMID_PATH.exists():
        saved = json.loads(PMID_PATH.read_text())
        if saved == [article["pmid"] for article in articles]:
            print(f"article embeddings already saved ({len(saved)})", flush=True)
            return np.load(EMBED_PATH)
    device = rr.best_device()
    tokenizer = AutoTokenizer.from_pretrained(ARTICLE_MODEL)
    model = AutoModel.from_pretrained(ARTICLE_MODEL).to(device).eval()
    print(f"article encoder on {device}", flush=True)
    pieces = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(articles), ARTICLE_BATCH):
            batch = articles[start : start + ARTICLE_BATCH]
            encoded = tokenizer(
                [article["title"] for article in batch],
                [article["abstract"] for article in batch],
                truncation=True,
                padding=True,
                max_length=ARTICLE_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            cls = model(**encoded).last_hidden_state[:, 0, :].float().cpu()
            pieces.append(cls)
            done = min(start + ARTICLE_BATCH, len(articles))
            if done % 3200 == 0 or done == len(articles):
                print(f"embedded {done}/{len(articles)}  {fmt(time.perf_counter() - started)}", flush=True)
    embeds = torch.cat(pieces, dim=0).numpy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBED_PATH, embeds)
    PMID_PATH.write_text(json.dumps([article["pmid"] for article in articles]))
    print(f"saved embeddings {embeds.shape} in {fmt(time.perf_counter() - started)}", flush=True)
    return embeds


def embed_queries(questions: list[dict]) -> np.ndarray:
    device = rr.best_device()
    tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL)
    model = AutoModel.from_pretrained(QUERY_MODEL).to(device).eval()
    print(f"query encoder on {device}", flush=True)
    texts = [question["question"] for question in questions]
    pieces = []
    with torch.inference_mode():
        for start in range(0, len(texts), 64):
            encoded = tokenizer(
                texts[start : start + 64],
                truncation=True,
                padding=True,
                max_length=QUERY_MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            pieces.append(model(**encoded).last_hidden_state[:, 0, :].float().cpu())
    return torch.cat(pieces, dim=0).numpy()


def candidate_path(split: str) -> Path:
    return OUT_DIR / f"{split}_top200.jsonl"


def build_candidates(split: str, edition: int, articles: list[dict], embeds: np.ndarray) -> None:
    path = candidate_path(split)
    if path.exists() and len(load_jsonl(path)) >= 300:
        print(f"{split} candidates already built", flush=True)
        return
    pmid_to_row = {article["pmid"]: index for index, article in enumerate(articles)}
    questions = []
    for question in questions_for(edition):
        gold = [pmid for pmid in rr.gold_pmids(question) if pmid in pmid_to_row]
        if gold:
            questions.append((question, gold))
    started = time.perf_counter()
    query_embeds = embed_queries([question for question, _ in questions])
    scores = query_embeds @ embeds.T
    pmids = [article["pmid"] for article in articles]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for index, ((question, gold), row) in enumerate(zip(questions, scores), start=1):
            chosen = np.argpartition(row, -DEPTH)[-DEPTH:]
            order = chosen[np.argsort(-row[chosen])]
            payload = {
                "id": question["id"],
                "question": question["question"],
                "gold": gold,
                "pmids": [pmids[int(item)] for item in order],
            }
            handle.write(json.dumps(payload) + "\n")
            if index % 50 == 0 or index == len(questions):
                print(f"{split} candidates {index}/{len(questions)}", flush=True)
    print(f"{split} candidates in {fmt(time.perf_counter() - started)}", flush=True)


def score_split(split: str, texts: dict[str, str], tokenizer, model, device) -> None:
    path = candidate_path(split)
    rows = list(load_jsonl(path).values())
    pending = [row for row in rows if "logits" not in row]
    done = {row["id"]: row for row in rows if "logits" in row}
    print(f"{split}: {len(done)} already reranked, {len(pending)} left", flush=True)
    lora.enable_lora_dropout(model, False)
    scored = dict(done)
    for index, row in enumerate(pending, start=1):
        pairs = [[row["question"], texts[pmid]] for pmid in row["pmids"]]
        updated = dict(row)
        updated["logits"] = rr.score_pairs(pairs, tokenizer, model, device)
        scored[row["id"]] = updated
        if index % 10 == 0 or index == len(pending):
            with path.open("w") as handle:
                for item in rows:
                    handle.write(json.dumps(scored.get(item["id"], item)) + "\n")
            print(f"{split}: reranked {index}/{len(pending)}", flush=True)


def mean_recall(rows: list[dict], ranked_of, k: int) -> float:
    scores = []
    for row in rows:
        gold = set(row["gold"])
        ranked = ranked_of(row)
        scores.append(len(gold.intersection(ranked[:k])) / len(gold))
    return float(sum(scores) / len(scores))


def lora_order(row: dict) -> list[str]:
    order = np.argsort(-np.array(row["logits"], dtype=np.float64))
    return [row["pmids"][int(index)] for index in order]


def recovered(split: str, rows: list[dict]) -> dict:
    frozen = load_jsonl(rr.score_path(split))
    outside = 0
    brought_in = 0
    for row in rows:
        previous = set(frozen[row["id"]]["pmids"])
        missing = set(row["gold"]) - previous
        outside += len(missing)
        brought_in += len(missing.intersection(row["pmids"]))
    return {"bm25_outside_top200": outside, "now_inside_bi_top200": brought_in}


def main() -> None:
    started = time.perf_counter()
    articles = load_articles()
    texts = {article["pmid"]: rr.article_text(article) for article in articles}
    embeds = embed_articles(articles)
    build_candidates("validation", 12, articles, embeds)
    build_candidates("test", 13, articles, embeds)
    del embeds
    tokenizer, model, device = lora.prepare_model()
    saved = torch.load(lora.ADAPTER_PATH, map_location="cpu", weights_only=False)
    lora.load_lora_state(model, saved["lora"], device)
    lora.enable_lora_dropout(model, False)
    print("loaded LoRA adapter", flush=True)
    score_split("validation", texts, tokenizer, model, device)
    score_split("test", texts, tokenizer, model, device)

    payload = {
        "first_stage": QUERY_MODEL,
        "article_encoder": ARTICLE_MODEL,
        "reranker": rr.MODEL_NAME,
        "query_max_length": QUERY_MAX_LENGTH,
        "article_max_length": ARTICLE_MAX_LENGTH,
        "depth": DEPTH,
        "lora_adapter": str(lora.ADAPTER_PATH),
        "metric": "mean recall",
    }
    for split in ("validation", "test"):
        rows = list(load_jsonl(candidate_path(split)).values())
        payload[split] = {
            "questions": len(rows),
            "bi_encoder_recall_at_20": mean_recall(rows, lambda row: row["pmids"], 20),
            "bi_encoder_recall_at_200": mean_recall(rows, lambda row: row["pmids"], 200),
            "lora_recall_at_20": mean_recall(rows, lora_order, 20),
            "lora_recall_at_200": mean_recall(rows, lora_order, 200),
            "previously_missed_by_bm25": recovered(split, rows),
        }
    payload["seconds"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    print(f"TOTAL {fmt(payload['seconds'])}", flush=True)


if __name__ == "__main__":
    main()
