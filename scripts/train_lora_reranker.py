"""Train one pairwise LoRA pass on BM25 top-200 candidates."""

from __future__ import annotations

import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_cache as cache
import rerank_medcpt as rr
import tune_bm25 as bm

RANK = 8
ALPHA = 16
DROPOUT = 0.1
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
MAX_POSITIVES = 8
NEGATIVES_PER_POSITIVE = 1
PASSES = 1
SEED = 0
REQUESTED_BATCH_PAIRS = 8
CHECKPOINT_EVERY = 200

LORA_DIR = bm.ROOT / "data" / "processed" / "lora"
PAIRS_PATH = LORA_DIR / "train_pairs.jsonl"
META_PATH = LORA_DIR / "train_pairs_meta.json"
CHECKPOINT_PATH = LORA_DIR / "checkpoint.pt"
ADAPTER_PATH = bm.ROOT / "weights" / "medcpt-cross-encoder-lora.pt"
SUMMARY_PATH = bm.ROOT / "data" / "processed" / "lora_recall20.json"
TIMES_PATH = LORA_DIR / "times.json"


def fmt(seconds: float) -> str:
    whole = int(round(seconds))
    hours, rem = divmod(whole, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    return f"{minutes}m {secs}s"


def save_times(times: dict) -> None:
    TIMES_PATH.write_text(json.dumps(times, indent=2))


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = self.B(self.A(self.drop(hidden))) * self.scale
        return self.base(hidden) + update


def attach_lora(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for layer in model.bert.encoder.layer:
        attention = layer.attention.self
        attention.query = LoRALinear(attention.query, RANK, ALPHA, DROPOUT)
        attention.value = LoRALinear(attention.value, RANK, ALPHA, DROPOUT)


def lora_modules(model: nn.Module) -> list[LoRALinear]:
    return [module for module in model.modules() if isinstance(module, LoRALinear)]


def enable_lora_dropout(model: nn.Module, enabled: bool) -> None:
    model.eval()
    for module in lora_modules(model):
        module.drop.train(enabled)


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_lora_state(model: nn.Module, state: dict[str, torch.Tensor], device) -> None:
    named = dict(model.named_parameters())
    for name, tensor in state.items():
        named[name].data.copy_(tensor.to(device))


def encode(tokenizer, pairs, device, pad_to_max: bool = False):
    encoded = tokenizer(
        pairs,
        truncation=True,
        padding="max_length" if pad_to_max else True,
        max_length=rr.MAX_LENGTH,
        return_tensors="pt",
    )
    return {key: value.to(device) for key, value in encoded.items()}


def pairwise_loss(model, tokenizer, device, pos_pairs, neg_pairs, pad_to_max: bool = False):
    logits = model(**encode(tokenizer, pos_pairs + neg_pairs, device, pad_to_max)).logits.view(-1)
    half = logits.shape[0] // 2
    return torch.nn.functional.softplus(logits[half:] - logits[:half]).mean()


def build_pairs() -> dict:
    started = time.perf_counter()
    texts, pmid_to_row, row_to_pmid, idf, posting_docs, posting_tfs, length_norm = rr.build_search()
    del texts
    questions = [q for q in bm.load_questions() if int(q["asq_challenge"]) <= 11]
    rng = random.Random(SEED)
    pairs = []
    used = 0
    skipped = 0
    for index, question in enumerate(questions, start=1):
        gold = {pmid for pmid in rr.gold_pmids(question) if pmid in pmid_to_row}
        if not gold:
            skipped += 1
            continue
        top = rr.bm25_top(
            question,
            pmid_to_row,
            row_to_pmid,
            idf,
            posting_docs,
            posting_tfs,
            length_norm,
            rr.MAX_DEPTH,
        )
        gold_in = [pmid for pmid in top if pmid in gold]
        non_gold = [pmid for pmid in top if pmid not in gold]
        if not gold_in or not non_gold:
            skipped += 1
            continue
        chosen = gold_in if len(gold_in) <= MAX_POSITIVES else rng.sample(gold_in, MAX_POSITIVES)
        for positive in chosen:
            for _ in range(NEGATIVES_PER_POSITIVE):
                pairs.append(
                    {
                        "q": question["question"],
                        "pos": positive,
                        "neg": rng.choice(non_gold),
                    }
                )
        used += 1
        if index % 500 == 0:
            print(f"pairs: scanned {index}/{len(questions)}", flush=True)
    rng.shuffle(pairs)
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    with PAIRS_PATH.open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair) + "\n")
    meta = {
        "seed": SEED,
        "max_positives": MAX_POSITIVES,
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "depth": rr.MAX_DEPTH,
        "k1": rr.K1,
        "b": rr.B,
        "questions_used": used,
        "questions_skipped": skipped,
        "pairs": len(pairs),
        "seconds": time.perf_counter() - started,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(
        f"pairs: {len(pairs)} from {used} questions, skipped {skipped}, {fmt(meta['seconds'])}",
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


def prepare_model():
    tokenizer, model, device = rr.load_model()
    attach_lora(model)
    model.to(device)
    model.config.use_cache = False
    enable_lora_dropout(model, False)
    trainable = sum(parameter.numel() for parameter in lora_parameters(model))
    print(f"trainable LoRA parameters: {trainable}", flush=True)
    return tokenizer, model, device


def move_optimizer_state(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def smoke(tokenizer, model, device, batch_pairs: int) -> None:
    saved = lora_state(model)
    enable_lora_dropout(model, True)
    question = "What is a coligo?"
    article = "Circularized oligodeoxynucleotides template small RNA. " * 40
    pos_pairs = [[question, article] for _ in range(batch_pairs)]
    neg_pairs = [[question, "Heart failure treatment uses several clinical drugs. " * 40] for _ in range(batch_pairs)]
    loss = pairwise_loss(tokenizer=tokenizer, model=model, device=device, pos_pairs=pos_pairs, neg_pairs=neg_pairs, pad_to_max=True)
    loss.backward()
    model.zero_grad(set_to_none=True)
    load_lora_state(model, saved, device)
    enable_lora_dropout(model, False)
    if device.type == "mps":
        torch.mps.empty_cache()
    print(f"memory check ok at batch {batch_pairs}", flush=True)


def choose_batch_size(tokenizer, model, device) -> int:
    batch_pairs = REQUESTED_BATCH_PAIRS
    while True:
        try:
            smoke(tokenizer, model, device, batch_pairs)
            return batch_pairs
        except RuntimeError as exc:
            if "memory" not in str(exc).lower() or batch_pairs == 1:
                raise
            print(f"batch {batch_pairs} did not fit ({exc}). trying {batch_pairs // 2}", flush=True)
            batch_pairs //= 2
            model.zero_grad(set_to_none=True)
            if device.type == "mps":
                torch.mps.empty_cache()
            gc.collect()


def train(tokenizer, model, device, texts: dict[str, str], batch_pairs: int, times: dict) -> None:
    pairs = load_pairs()
    start_step = 0
    optimizer = AdamW(lora_parameters(model), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if CHECKPOINT_PATH.exists() and not ADAPTER_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        load_lora_state(model, checkpoint["lora"], device)
        optimizer.load_state_dict(checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
        torch.set_rng_state(checkpoint["torch_rng"])
        start_step = int(checkpoint["step"])
        print(f"resuming at batch {start_step}", flush=True)
    if ADAPTER_PATH.exists():
        saved = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
        load_lora_state(model, saved["lora"], device)
        print("loaded finished adapter", flush=True)
        return

    remaining = pairs[start_step * batch_pairs :]
    steps_total = start_step + math.ceil(len(remaining) / batch_pairs)
    enable_lora_dropout(model, True)
    started = time.perf_counter()
    running = 0.0
    seen = 0
    for offset in range(0, len(remaining), batch_pairs):
        batch = remaining[offset : offset + batch_pairs]
        step = start_step + offset // batch_pairs + 1
        pos_pairs = [[row["q"], texts[row["pos"]]] for row in batch]
        neg_pairs = [[row["q"], texts[row["neg"]]] for row in batch]
        optimizer.zero_grad(set_to_none=True)
        loss = pairwise_loss(model, tokenizer, device, pos_pairs, neg_pairs)
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
        if step % CHECKPOINT_EVERY == 0:
            torch.save(
                {
                    "step": step,
                    "batch_pairs": batch_pairs,
                    "lora": lora_state(model),
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                },
                CHECKPOINT_PATH,
            )
    times["train_seconds"] = time.perf_counter() - started
    times["train_pairs"] = len(pairs)
    times["batch_pairs"] = batch_pairs
    save_times(times)
    torch.save(
        {
            "settings": {
                "model": rr.MODEL_NAME,
                "model_revision": rr.MODEL_REVISION,
                "rank": RANK,
                "alpha": ALPHA,
                "dropout": DROPOUT,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "max_length": rr.MAX_LENGTH,
                "max_positives": MAX_POSITIVES,
                "negatives_per_positive": NEGATIVES_PER_POSITIVE,
                "passes": PASSES,
                "batch_pairs": batch_pairs,
                "seed": SEED,
            },
            "lora": lora_state(model),
        },
        ADAPTER_PATH,
    )
    enable_lora_dropout(model, False)
    print(f"train done in {fmt(times['train_seconds'])}", flush=True)


def load_jsonl(path: Path) -> dict[str, dict]:
    return cache.load_rankings(
        path,
        min_candidates=rr.MAX_DEPTH,
        max_candidates=rr.MAX_DEPTH,
        require_logits=True,
    )


def evaluate(split: str, texts, questions, tokenizer, model, device) -> dict:
    expected_ids = [question["id"] for question in rr.questions_for(split)]
    frozen_rows = rr.load_scored(rr.score_path(split))
    cache.require_complete(frozen_rows, expected_ids, source=str(rr.score_path(split)))
    out_path = LORA_DIR / f"{split}_top200.jsonl"
    done = load_jsonl(out_path)
    cache.missing_question_ids(done, expected_ids, source=str(out_path))
    for qid, saved in done.items():
        source = frozen_rows[qid]
        if saved["gold"] != source["gold"] or saved["pmids"] != source["pmids"]:
            raise ValueError(f"{out_path}: stale candidates for {qid}")
    pending = [frozen_rows[qid] for qid in expected_ids if qid not in done]
    print(f"{split}: {len(done)} already scored, {len(pending)} left", flush=True)
    enable_lora_dropout(model, False)
    started = time.perf_counter()
    with out_path.open("a") as handle:
        for index, row in enumerate(pending, start=1):
            question = questions[row["id"]]
            pairs = [[question, texts[pmid]] for pmid in row["pmids"]]
            logits = rr.score_pairs(pairs, tokenizer, model, device)
            handle.write(
                json.dumps(
                    {"id": row["id"], "gold": row["gold"], "pmids": row["pmids"], "logits": logits}
                )
                + "\n"
            )
            handle.flush()
            if index % 10 == 0:
                print(f"{split}: scored {index}/{len(pending)}", flush=True)
    scored = load_jsonl(out_path)
    cache.require_complete(scored, expected_ids, source=str(out_path))
    rows = [scored[qid] for qid in expected_ids]
    frozen = [frozen_rows[qid] for qid in expected_ids]
    return {
        "questions": len(rows),
        "frozen_recall_at_20": rr.mean_recall(frozen, rr.MAX_DEPTH),
        "lora_recall_at_20": rr.mean_recall(rows, rr.MAX_DEPTH),
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    started = time.perf_counter()
    LORA_DIR.mkdir(parents=True, exist_ok=True)
    ADAPTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    times = json.loads(TIMES_PATH.read_text()) if TIMES_PATH.exists() else {}
    if not PAIRS_PATH.exists():
        meta = build_pairs()
        times["pairs_seconds"] = meta["seconds"]
        times["questions_used"] = meta["questions_used"]
        times["pairs"] = meta["pairs"]
        save_times(times)
        gc.collect()
    else:
        meta = json.loads(META_PATH.read_text())
        print(f"pairs already built: {meta['pairs']} from {meta['questions_used']} questions", flush=True)

    texts = rr.load_texts()
    tokenizer, model, device = prepare_model()
    if not ADAPTER_PATH.exists():
        if CHECKPOINT_PATH.exists():
            batch_pairs = int(torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)["batch_pairs"])
            print(f"resuming with batch {batch_pairs}", flush=True)
        else:
            batch_pairs = choose_batch_size(tokenizer, model, device)
        train(tokenizer, model, device, texts, batch_pairs, times)
    else:
        saved = torch.load(ADAPTER_PATH, map_location="cpu", weights_only=False)
        load_lora_state(model, saved["lora"], device)
        print("loaded finished adapter, skipping training", flush=True)

    questions = {q["id"]: q["question"] for q in bm.load_questions()}
    validation = evaluate("validation", texts, questions, tokenizer, model, device)
    times["validation_seconds"] = validation["seconds"]
    save_times(times)
    print(
        f"validation Recall@20 frozen {validation['frozen_recall_at_20']:.4f}  lora {validation['lora_recall_at_20']:.4f}  {fmt(validation['seconds'])}",
        flush=True,
    )
    test = evaluate("test", texts, questions, tokenizer, model, device)
    times["test_seconds"] = test["seconds"]
    times["total_seconds"] = time.perf_counter() - started
    save_times(times)
    payload = {
        "model": rr.MODEL_NAME,
        "model_revision": rr.MODEL_REVISION,
        "passes": PASSES,
        "rank": RANK,
        "alpha": ALPHA,
        "dropout": DROPOUT,
        "learning_rate": LEARNING_RATE,
        "max_length": rr.MAX_LENGTH,
        "max_positives": MAX_POSITIVES,
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "candidate_depth": rr.MAX_DEPTH,
        "bm25": {"k1": rr.K1, "b": rr.B},
        "metric": "mean Recall@20",
        "train_questions": meta["questions_used"],
        "train_pairs": meta["pairs"],
        "validation_frozen_recall_at_20": validation["frozen_recall_at_20"],
        "validation_lora_recall_at_20": validation["lora_recall_at_20"],
        "test_frozen_recall_at_20": test["frozen_recall_at_20"],
        "test_lora_recall_at_20": test["lora_recall_at_20"],
        "seconds": {
            "pairs": times.get("pairs_seconds"),
            "train": times.get("train_seconds"),
            "validation": times["validation_seconds"],
            "test": times["test_seconds"],
            "total_this_process": times["total_seconds"],
        },
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2), flush=True)
    print(f"TOTAL {fmt(times['total_seconds'])}", flush=True)


if __name__ == "__main__":
    main()
