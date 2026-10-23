"""Measure Recall@20 and Recall@100 from saved ranking artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ranking_cache as cache

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = ROOT / "data" / "raw" / "bioasq-14b" / "training14b.json"
BM25_DIR = ROOT / "data" / "processed" / "rerank"
BI_DIR = ROOT / "data" / "processed" / "medcpt_bi"
LORA_DIR = ROOT / "data" / "processed" / "lora"
LORA_BI_DIR = ROOT / "data" / "processed" / "lora_bi"
UNION_DIR = ROOT / "data" / "processed" / "union_lora"
OUTPUT_PATH = ROOT / "results" / "recall_ceiling.json"
TOP_KS = (20, 100)
SPLIT_EDITIONS = {"validation": 12, "test": 13}


def ranked_pmids(row: dict) -> list[str]:
    order = np.argsort(-np.asarray(row["logits"], dtype=np.float64))
    return [row["pmids"][int(index)] for index in order]


def analyze_split(bm25_rows: list[dict], hybrid_rows: list[dict]) -> dict:
    bm25_by_id = {row["id"]: row for row in bm25_rows}
    hybrid_by_id = {row["id"]: row for row in hybrid_rows}
    if bm25_by_id.keys() != hybrid_by_id.keys():
        raise ValueError("BM25 and hybrid rankings contain different question IDs")

    pairs = []
    for qid, hybrid in hybrid_by_id.items():
        bm25 = bm25_by_id[qid]
        gold = set(hybrid["gold"])
        if gold != set(bm25["gold"]):
            raise ValueError(f"gold PMID mismatch for {qid}")
        pairs.append((bm25, hybrid, gold))

    gold_counts = [len(gold) for _, _, gold in pairs]
    cutoffs = {top_k: analyze_cutoff(pairs, top_k) for top_k in TOP_KS}
    at_20 = cutoffs[20]
    at_100 = cutoffs[100]
    return {
        "questions": len(gold_counts),
        "gold_articles_per_question": {
            "median": median(gold_counts),
            "mean": mean(gold_counts),
            "maximum": max(gold_counts),
            "over_20_questions": at_20["over_limit"],
            "over_20_fraction": at_20["over_limit_fraction"],
            "full_recall_possible_questions": at_20["possible"],
            "full_recall_possible_fraction": at_20["possible_fraction"],
            "over_100_questions": at_100["over_limit"],
            "over_100_fraction": at_100["over_limit_fraction"],
            "full_recall_possible_at_100_questions": at_100["possible"],
            "full_recall_possible_at_100_fraction": at_100["possible_fraction"],
        },
        "ceilings": {
            "top_20_capacity": at_20["capacity"],
            "hybrid_candidate_restricted": at_20["candidate"],
            "candidate_exclusion_gap": at_20["candidate_loss"],
            "top_100_capacity": at_100["capacity"],
            "hybrid_candidate_restricted_at_100": at_100["candidate"],
            "candidate_exclusion_gap_at_100": at_100["candidate_loss"],
        },
        "systems": {
            "bm25": {
                **format_system_summary(at_20["systems"]["bm25"], 20),
                **format_system_summary(at_100["systems"]["bm25"], 100),
            },
            "hybrid": {
                **format_system_summary(at_20["systems"]["hybrid"], 20, candidate=True),
                **format_system_summary(at_100["systems"]["hybrid"], 100, candidate=True),
            },
        },
        "hybrid_gap_attribution": format_gap_attribution(at_20, 20),
        "hybrid_gap_attribution_at_100": format_gap_attribution(at_100, 100),
    }


def analyze_cutoff(pairs: list[tuple[dict, dict, set[str]]], top_k: int) -> dict:
    capacity_ceilings = []
    candidate_ceilings = []
    bm25_recalls = []
    hybrid_recalls = []
    bm25_full = 0
    hybrid_full = 0
    bm25_full_possible = 0
    hybrid_full_possible = 0

    for bm25, hybrid, gold in pairs:
        gold_count = len(gold)
        capacity_ceilings.append(min(top_k, gold_count) / gold_count)
        gold_in_candidates = len(gold.intersection(hybrid["pmids"]))
        candidate_ceilings.append(min(top_k, gold_in_candidates) / gold_count)

        bm25_top = set(bm25["pmids"][:top_k])
        hybrid_top = set(ranked_pmids(hybrid)[:top_k])
        bm25_recalls.append(len(gold.intersection(bm25_top)) / gold_count)
        hybrid_recalls.append(len(gold.intersection(hybrid_top)) / gold_count)

        bm25_complete = gold.issubset(bm25_top)
        hybrid_complete = gold.issubset(hybrid_top)
        bm25_full += bm25_complete
        hybrid_full += hybrid_complete
        if gold_count <= top_k:
            bm25_full_possible += bm25_complete
            hybrid_full_possible += hybrid_complete

    questions = len(pairs)
    possible = sum(len(gold) <= top_k for _, _, gold in pairs)
    over_limit = questions - possible
    capacity = mean(capacity_ceilings)
    candidate = mean(candidate_ceilings)
    bm25_recall = mean(bm25_recalls)
    hybrid_recall = mean(hybrid_recalls)
    if not capacity + 1e-12 >= candidate >= hybrid_recall - 1e-12:
        raise ValueError("expected top-k capacity >= candidate oracle >= observed hybrid recall")

    candidate_loss = capacity - candidate
    ranking_gap = candidate - hybrid_recall
    total_gap = capacity - hybrid_recall
    return {
        "over_limit": over_limit,
        "over_limit_fraction": over_limit / questions,
        "possible": possible,
        "possible_fraction": possible / questions,
        "capacity": capacity,
        "candidate": candidate,
        "candidate_loss": candidate_loss,
        "systems": {
            "bm25": system_summary(
                bm25_recall,
                capacity,
                None,
                bm25_full,
                bm25_full_possible,
                questions,
                possible,
            ),
            "hybrid": system_summary(
                hybrid_recall,
                capacity,
                candidate,
                hybrid_full,
                hybrid_full_possible,
                questions,
                possible,
            ),
        },
        "total_gap": total_gap,
        "ranking_gap": ranking_gap,
    }


def system_summary(
    recall: float,
    capacity: float,
    candidate_ceiling: float | None,
    full: int,
    full_possible: int,
    questions: int,
    possible: int,
) -> dict:
    summary = {
        "recall": recall,
        "capacity_gap": capacity - recall,
        "capacity_fraction": recall / capacity,
        "full_questions": full,
        "full_fraction": full / questions,
        "full_among_possible": safe_ratio(full_possible, possible),
    }
    if candidate_ceiling is not None:
        summary["candidate_gap"] = candidate_ceiling - recall
        summary["candidate_fraction"] = safe_ratio(recall, candidate_ceiling)
    return summary


def format_system_summary(summary: dict, top_k: int, *, candidate: bool = False) -> dict:
    output = {
        f"recall_at_{top_k}": summary["recall"],
        f"absolute_gap_to_top_{top_k}_capacity": summary["capacity_gap"],
        f"fraction_of_top_{top_k}_capacity_achieved": summary["capacity_fraction"],
        f"full_coverage_at_{top_k}_questions": summary["full_questions"],
        f"full_coverage_at_{top_k}": summary["full_fraction"],
        f"full_coverage_at_{top_k}_among_possible_questions": summary["full_among_possible"],
    }
    if candidate:
        suffix = "" if top_k == 20 else f"_at_{top_k}"
        output[f"absolute_gap_to_candidate_restricted_ceiling{suffix}"] = summary["candidate_gap"]
        output[f"fraction_of_candidate_restricted_ceiling_achieved{suffix}"] = summary[
            "candidate_fraction"
        ]
    return output


def format_gap_attribution(analysis: dict, top_k: int) -> dict:
    return {
        f"total_gap_to_top_{top_k}_capacity": analysis["total_gap"],
        "candidate_exclusion_gap": analysis["candidate_loss"],
        "ranking_gap_to_candidate_oracle": analysis["ranking_gap"],
        "candidate_exclusion_share_of_total_gap": safe_ratio(
            analysis["candidate_loss"], analysis["total_gap"]
        ),
        "ranking_share_of_total_gap": safe_ratio(
            analysis["ranking_gap"], analysis["total_gap"]
        ),
    }


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def expected_ids(questions_path: Path, edition: int) -> list[str]:
    questions = json.loads(questions_path.read_text())["questions"]
    return [question["id"] for question in questions if int(question["asq_challenge"]) == edition]


def load_complete(path: Path, expected_ids: list[str], minimum: int, maximum: int) -> dict[str, dict]:
    rows = cache.load_rankings(
        path,
        min_candidates=minimum,
        max_candidates=maximum,
        require_logits=True,
    )
    cache.require_complete(rows, expected_ids, source=str(path))
    return rows


def evaluate_system(rows: dict[str, dict], expected_ids: list[str], *, rerank: bool) -> dict:
    metrics = {}
    for top_k in TOP_KS:
        recalls = []
        for qid in expected_ids:
            row = rows[qid]
            ranked = ranked_pmids(row) if rerank else row["pmids"]
            gold = set(row["gold"])
            recalls.append(len(gold.intersection(ranked[:top_k])) / len(gold))
        metrics[f"recall_at_{top_k}"] = mean(recalls)
    return metrics


def validate_gold(reference: dict[str, dict], other: dict[str, dict], expected_ids: list[str], name: str) -> None:
    for qid in expected_ids:
        if set(reference[qid]["gold"]) != set(other[qid]["gold"]):
            raise ValueError(f"{name}: gold PMID mismatch for {qid}")


def run(
    questions_path: Path,
    bm25_dir: Path,
    bi_dir: Path,
    lora_dir: Path,
    lora_bi_dir: Path,
    union_dir: Path,
) -> dict:
    payload = {
        "metric": "macro Recall@20",
        "additional_metric": "macro Recall@100",
        "top_k_capacity_definition": "mean(min(20, gold_count) / gold_count)",
        "candidate_restricted_definition": "mean(min(20, gold_in_union_candidates) / gold_count)",
        "top_100_capacity_definition": "mean(min(100, gold_count) / gold_count)",
        "candidate_restricted_at_100_definition": "mean(min(100, gold_in_union_candidates) / gold_count)",
        "gold_definition": "Unique annotated gold PMIDs available in the searchable corpus, as saved with each ranking.",
        "source_artifacts": {
            "questions": "training14b.json",
            "bm25": "rerank/{split}_top200.jsonl",
            "dense": "medcpt_bi/{split}_top200.jsonl",
            "bm25_lora": "lora/{split}_top200.jsonl",
            "dense_lora": "lora_bi/{split}_top200.jsonl",
            "hybrid": "union_lora/{split}_union.jsonl",
        },
        "splits": {},
    }
    for split, edition in SPLIT_EDITIONS.items():
        ids = expected_ids(questions_path, edition)
        bm25_path = bm25_dir / f"{split}_top200.jsonl"
        union_path = union_dir / f"{split}_union.jsonl"
        bm25 = load_complete(bm25_path, ids, 200, 200)
        dense = load_complete(bi_dir / f"{split}_top200.jsonl", ids, 200, 200)
        bm25_lora = load_complete(lora_dir / f"{split}_top200.jsonl", ids, 200, 200)
        dense_lora = load_complete(lora_bi_dir / f"{split}_top200.jsonl", ids, 200, 200)
        hybrid = load_complete(union_path, ids, 200, 400)
        for name, rows in (
            ("dense", dense),
            ("BM25 LoRA", bm25_lora),
            ("dense LoRA", dense_lora),
            ("hybrid", hybrid),
        ):
            validate_gold(bm25, rows, ids, name)

        analysis = analyze_split(
            [bm25[qid] for qid in ids],
            [hybrid[qid] for qid in ids],
        )
        analysis["retrieval_systems"] = [
            {"name": "BM25", **evaluate_system(bm25, ids, rerank=False)},
            {
                "name": "MedCPT bi-encoder",
                **evaluate_system(dense, ids, rerank=False),
            },
            {
                "name": "BM25 + frozen MedCPT reranker",
                **evaluate_system(bm25, ids, rerank=True),
            },
            {
                "name": "MedCPT candidates + BM25-trained LoRA reranker",
                **evaluate_system(dense, ids, rerank=True),
            },
            {
                "name": "MedCPT candidates + matching LoRA reranker",
                **evaluate_system(dense_lora, ids, rerank=True),
            },
            {
                "name": "BM25 + LoRA MedCPT reranker",
                **evaluate_system(bm25_lora, ids, rerank=True),
            },
            {
                "name": "BM25/MedCPT union + LoRA reranker",
                **evaluate_system(hybrid, ids, rerank=True),
            },
        ]
        payload["splits"][split] = analysis
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--bm25-dir", type=Path, default=BM25_DIR)
    parser.add_argument("--bi-dir", type=Path, default=BI_DIR)
    parser.add_argument("--lora-dir", type=Path, default=LORA_DIR)
    parser.add_argument("--lora-bi-dir", type=Path, default=LORA_BI_DIR)
    parser.add_argument("--union-dir", type=Path, default=UNION_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    payload = run(
        args.questions,
        args.bm25_dir,
        args.bi_dir,
        args.lora_dir,
        args.lora_bi_dir,
        args.union_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
