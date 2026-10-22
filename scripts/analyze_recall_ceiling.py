"""Measure Recall@20 capacity and candidate-coverage ceilings from saved rankings."""

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
UNION_DIR = ROOT / "data" / "processed" / "union_lora"
OUTPUT_PATH = ROOT / "results" / "recall_ceiling.json"
TOP_K = 20
SPLIT_EDITIONS = {"validation": 12, "test": 13}


def ranked_pmids(row: dict) -> list[str]:
    order = np.argsort(-np.asarray(row["logits"], dtype=np.float64))
    return [row["pmids"][int(index)] for index in order]


def analyze_split(bm25_rows: list[dict], hybrid_rows: list[dict], top_k: int = TOP_K) -> dict:
    bm25_by_id = {row["id"]: row for row in bm25_rows}
    hybrid_by_id = {row["id"]: row for row in hybrid_rows}
    if bm25_by_id.keys() != hybrid_by_id.keys():
        raise ValueError("BM25 and hybrid rankings contain different question IDs")

    gold_counts = []
    capacity_ceilings = []
    candidate_ceilings = []
    bm25_recalls = []
    hybrid_recalls = []
    bm25_full = 0
    hybrid_full = 0
    bm25_full_possible = 0
    hybrid_full_possible = 0

    for qid, hybrid in hybrid_by_id.items():
        bm25 = bm25_by_id[qid]
        gold = set(hybrid["gold"])
        if gold != set(bm25["gold"]):
            raise ValueError(f"gold PMID mismatch for {qid}")

        gold_count = len(gold)
        gold_counts.append(gold_count)
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

    questions = len(gold_counts)
    possible = sum(count <= top_k for count in gold_counts)
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
        "questions": questions,
        "gold_articles_per_question": {
            "median": median(gold_counts),
            "mean": mean(gold_counts),
            "maximum": max(gold_counts),
            "over_20_questions": over_limit,
            "over_20_fraction": over_limit / questions,
            "full_recall_possible_questions": possible,
            "full_recall_possible_fraction": possible / questions,
        },
        "ceilings": {
            "top_20_capacity": capacity,
            "hybrid_candidate_restricted": candidate,
            "candidate_exclusion_gap": candidate_loss,
        },
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
        "hybrid_gap_attribution": {
            "total_gap_to_top_20_capacity": total_gap,
            "candidate_exclusion_gap": candidate_loss,
            "ranking_gap_to_candidate_oracle": ranking_gap,
            "candidate_exclusion_share_of_total_gap": safe_ratio(candidate_loss, total_gap),
            "ranking_share_of_total_gap": safe_ratio(ranking_gap, total_gap),
        },
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
        "recall_at_20": recall,
        "absolute_gap_to_top_20_capacity": capacity - recall,
        "fraction_of_top_20_capacity_achieved": recall / capacity,
        "full_coverage_at_20_questions": full,
        "full_coverage_at_20": full / questions,
        "full_coverage_at_20_among_possible_questions": safe_ratio(full_possible, possible),
    }
    if candidate_ceiling is not None:
        summary["absolute_gap_to_candidate_restricted_ceiling"] = candidate_ceiling - recall
        summary["fraction_of_candidate_restricted_ceiling_achieved"] = safe_ratio(
            recall, candidate_ceiling
        )
    return summary


def safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def expected_ids(questions_path: Path, edition: int) -> list[str]:
    questions = json.loads(questions_path.read_text())["questions"]
    return [question["id"] for question in questions if int(question["asq_challenge"]) == edition]


def run(questions_path: Path, bm25_dir: Path, union_dir: Path) -> dict:
    payload = {
        "metric": "macro Recall@20",
        "top_k_capacity_definition": "mean(min(20, gold_count) / gold_count)",
        "candidate_restricted_definition": "mean(min(20, gold_in_union_candidates) / gold_count)",
        "gold_definition": "Unique annotated gold PMIDs available in the searchable corpus, as saved with each ranking.",
        "source_artifacts": {
            "questions": "training14b.json",
            "bm25": "rerank/{split}_top200.jsonl",
            "hybrid": "union_lora/{split}_union.jsonl",
        },
        "splits": {},
    }
    for split, edition in SPLIT_EDITIONS.items():
        ids = expected_ids(questions_path, edition)
        bm25_path = bm25_dir / f"{split}_top200.jsonl"
        union_path = union_dir / f"{split}_union.jsonl"
        bm25 = cache.load_rankings(
            bm25_path,
            min_candidates=200,
            max_candidates=200,
            require_logits=True,
        )
        hybrid = cache.load_rankings(
            union_path,
            min_candidates=200,
            max_candidates=400,
            require_logits=True,
        )
        cache.require_complete(bm25, ids, source=str(bm25_path))
        cache.require_complete(hybrid, ids, source=str(union_path))
        payload["splits"][split] = analyze_split(
            [bm25[qid] for qid in ids],
            [hybrid[qid] for qid in ids],
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--bm25-dir", type=Path, default=BM25_DIR)
    parser.add_argument("--union-dir", type=Path, default=UNION_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    payload = run(args.questions, args.bm25_dir, args.union_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
