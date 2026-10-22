"""Train a new LoRA adapter on the bi-encoder's top 200.

Positives are gold articles inside that list, at most 8 per question.
Each positive gets a different non-gold article from the same list.
Editions 1-11, one pass, same LoRA settings as the BM25 adapter.
The BM25 adapter is left untouched. Edition 12 and 13 are reranked
on the bi-encoder lists already saved.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rerank_medcpt as rr
import rerank_medcpt_bi as bi
import train_lora_reranker as lora
import tune_bm25 as bm

DEPTH = 200
SEED = 0
OUT_DIR = bm.ROOT / "data" / "processed" / "lora_bi"
PAIRS_PATH = OUT_DIR / "train_pairs.jsonl"
META_PATH = OUT_DIR / "train_pairs_meta.json"
ADAPTER_PATH = OUT_DIR / "adapter.pt"
CHECKPOINT_PATH = OUT_DIR / "checkpoint.pt"
SUMMARY_PATH = bm.ROOT / "data" / "processed" / "lora_bi_recall20.json"
TIMES_PATH = OUT_DIR / "times.json"


def fmt(seconds: float) -> str:
    return lora.fmt(seconds)


def save_times(times: dict) -> None:
    TIMES_PATH.write_text(json.dumps(times, indent=2))


def load_jsonl(path: Path) -> dict[str, dict]:
    return lora.load_jsonl(path)


def build_pairs() -> dict:
    if PAIRS_PATH.exists() and META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        print(f"pairs already built: {meta['pairs']} from {meta['questions_used']} questions", flush=True)
        return meta
    started = time.perf_counter()
    pmids = json.loads(bi.PMID_PATH.read_text())
    embeds = np.load(bi.EMBED_PATH)
    pmid_to_row = {pmid: index for index, pmid in enumerate(pmids)}
    questions = []
    for question in bm.load_questions():
        if int(question["asq_challenge"]) > 11:
            continue
        gold = [pmid for pmid in rr.gold_pmids(question) if pmid in pmid_to_row]
        if gold:
            questions.append((question, gold))
    query_embeds = bi.embed_queries([question for question, _ in questions])
    rng = random.Random(SEED)
    pairs = []
    used = 0
    skipped = 0
    repeated = 0
    chunk = 256
    for start in range(0, len(questions), chunk):
        block = questions[start : start + chunk]
        scores = query_embeds[start : start + chunk] @ embeds.T
        for (question, gold), row in zip(block, scores):
            gold_set = set(gold)
            chosen_idx = np.argpartition(row, -DEPTH)[-DEPTH:]
            order = chosen_idx[np.argsort(-row[chosen_idx])]
            top = [pmids[int(index)] for index in order]
            gold_in = [pmid for pmid in top if pmid in gold_set]
            non_gold = [pmid for pmid in top if pmid not in gold_set]
            if not gold_in or not non_gold:
                skipped += 1
                continue
            positives = gold_in if len(gold_in) <= lora.MAX_POSITIVES else rng.sample(gold_in, lora.MAX_POSITIVES)
            take = min(len(positives), len(non_gold))
            positives = positives[:take]
            negatives = rng.sample(non_gold, take)
            if len(set(negatives)) != len(negatives):
                repeated += 1
            for positive, negative in zip(positives, negatives):
                pairs.append({"q": question["question"], "pos": positive, "neg": negative})
            used += 1
        print(f"pairs: scanned {min(start + chunk, len(questions))}/{len(questions)}", flush=True)
    rng.shuffle(pairs)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with PAIRS_PATH.open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    meta = {
        "seed": SEED,
        "max_positives": lora.MAX_POSITIVES,
        "negatives": "unique within each question",
        "depth": DEPTH,
        "first_stage": "medcpt bi-encoder",
        "questions_used": used,
        "questions_skipped": skipped,
        "pairs": len(pairs),
        "questions_with_repeated_negative": repeated,
        "seconds": time.perf_counter() - started,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(
        f"pairs: {len(pairs)} from {used} questions, skipped {skipped}, repeated negatives {repeated}, {fmt(meta['seconds'])}",
        flush=True,
    )
    return meta


def load_pairs() -> list[dict]:
    pairs = []
    with PAIRS_PATH.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def train(tokenizer, model, device, texts: dict[str, str], batch_pairs: int, times: dict) -> None:
    pairs = load_pairs()
    start_step = 0
    optimizer = AdamW(lora.lora_parameters(model), lr=lora.LEARNING_RATE, weight_decay=lora.WEIGHT_DECAY)
    if CHECKPOINT_PATH.exists() and not ADAPTER_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        lora.load_lora_state(model, checkpoint["lora"], device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        lora.move_optimizer_state(optimizer, device)
        torch.set_rng_state(checkpoint["torch_rng"])
        start_step = int(checkpoint["step"])
        print(f"resuming at batch {start_step}", flush=True)
    if ADAPTER_PATH.exists():
        saved = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
        lora.load_lora_state(model, saved["lora"], device)
        print("loaded finished adapter", flush=True)
        return

    remaining = pairs[start_step * batch_pairs :]
    steps_total = start_step + math.ceil(len(remaining) / batch_pairs)
    lora.enable_lora_dropout(model, True)
    started = time.perf_counter()
    running = 0.0
    seen = 0
    for offset in range(0, len(remaining), batch_pairs):
        batch = remaining[offset : offset + batch_pairs]
        step = start_step + offset // batch_pairs + 1
        pos_pairs = [[row["q"], texts[row["pos"]]] for row in batch]
        neg_pairs = [[row["q"], texts[row["neg"]]] for row in batch]
        optimizer.zero_grad(set_to_none=True)
        loss = lora.pairwise_loss(model, tokenizer, device, pos_pairs, neg_pairs)
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu())
        running += value * len(batch)
        seen += len(batch)
        if step == 1 or step % 50 == 0 or step == steps_total:
            elapsed = time.perf_counter() - started
            done = start_step * batch_pairs + seen
            rate = done / elapsed if elapsed else 0
            left = (len(pairs) - done) / rate if rate else 0
            print(
                f"train batch {step}/{steps_total}  loss {value:.4f}  mean {running / seen:.4f}  {fmt(elapsed)}  left {fmt(left)}",
                flush=True,
            )
        if step % lora.CHECKPOINT_EVERY == 0:
            torch.save(
                {
                    "step": step,
                    "batch_pairs": batch_pairs,
                    "lora": lora.lora_state(model),
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                },
                CHECKPOINT_PATH,
            )
    times["train_seconds"] = time.perf_counter() - started
    times["train_pairs"] = len(pairs)
    times["batch_pairs"] = batch_pairs
    save_times(times)
    torch.save({"lora": lora.lora_state(model), "first_stage": "medcpt bi-encoder"}, ADAPTER_PATH)
    lora.enable_lora_dropout(model, False)
    print(f"train done in {fmt(times['train_seconds'])}", flush=True)


def score_split(split: str, texts: dict[str, str], tokenizer, model, device) -> list[dict]:
    source = list(load_jsonl(bi.candidate_path(split)).values())
    out_path = OUT_DIR / f"{split}_top200.jsonl"
    done = load_jsonl(out_path)
    pending = [row for row in source if row["id"] not in done]
    print(f"{split}: {len(done)} already scored, {len(pending)} left", flush=True)
    lora.enable_lora_dropout(model, False)
    with out_path.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            pairs = [[row["question"], texts[pmid]] for pmid in row["pmids"]]
            logits = rr.score_pairs(pairs, tokenizer, model, device)
            handle.write(
                json.dumps({"id": row["id"], "gold": row["gold"], "pmids": row["pmids"], "logits": logits})
                + "\n"
            )
            handle.flush()
            if index % 10 == 0 or index == len(pending):
                print(f"{split}: scored {index}/{len(pending)}", flush=True)
    return list(load_jsonl(out_path).values())


def mean_recall(rows: list[dict], rerank: bool, k: int) -> float:
    scores = []
    for row in rows:
        gold = set(row["gold"])
        if rerank:
            order = np.argsort(-np.array(row["logits"], dtype=np.float64))
            ranked = [row["pmids"][int(index)] for index in order]
        else:
            ranked = row["pmids"]
        scores.append(len(gold.intersection(ranked[:k])) / len(gold))
    return float(sum(scores) / len(scores))


def full_coverage(rows: list[dict], rerank: bool, k: int) -> float:
    hits = 0
    for row in rows:
        gold = set(row["gold"])
        if rerank:
            order = np.argsort(-np.array(row["logits"], dtype=np.float64))
            ranked = [row["pmids"][int(index)] for index in order]
        else:
            ranked = row["pmids"]
        hits += gold.issubset(ranked[:k])
    return hits / len(rows)


def main() -> None:
    started = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    times = json.loads(TIMES_PATH.read_text()) if TIMES_PATH.exists() else {}
    meta = build_pairs()
    times["pairs_seconds"] = meta["seconds"]
    times["pairs"] = meta["pairs"]
    save_times(times)

    texts = rr.load_texts()
    tokenizer, model, device = lora.prepare_model()
    if not ADAPTER_PATH.exists():
        if CHECKPOINT_PATH.exists():
            batch_pairs = int(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)["batch_pairs"])
            print(f"resuming with batch {batch_pairs}", flush=True)
        else:
            batch_pairs = lora.choose_batch_size(tokenizer, model, device)
        train(tokenizer, model, device, texts, batch_pairs, times)
    else:
        saved = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
        lora.load_lora_state(model, saved["lora"], device)
        print("loaded finished adapter, skipping training", flush=True)

    payload = {
        "first_stage": "medcpt bi-encoder top 200",
        "negatives": "unique within each question",
        "train_questions": meta["questions_used"],
        "train_pairs": meta["pairs"],
        "passes": 1,
        "rank": lora.RANK,
        "alpha": lora.ALPHA,
        "learning_rate": lora.LEARNING_RATE,
        "max_positives": lora.MAX_POSITIVES,
    }
    for split in ("validation", "test"):
        rows = score_split(split, texts, tokenizer, model, device)
        previous = list(load_jsonl(bi.candidate_path(split)).values())
        payload[split] = {
            "questions": len(rows),
            "bi_encoder_recall_at_20": mean_recall(rows, False, 20),
            "previous_lora_recall_at_20": mean_recall(previous, True, 20),
            "new_lora_recall_at_20": mean_recall(rows, True, 20),
            "new_lora_recall_at_200": mean_recall(rows, True, 200),
            "new_lora_full_coverage_at_20": full_coverage(rows, True, 20),
        }
    payload["seconds"] = time.perf_counter() - started
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    print(f"TOTAL {fmt(payload['seconds'])}", flush=True)


if __name__ == "__main__":
    main()
