import numpy as np

from scripts import run_recall20, run_tfidf_recall20


def test_recall_at_k_counts_only_positive_scores() -> None:
    scores = np.array([3.0, 2.0, 0.0, 0.0])
    gold = {0, 2}
    assert run_recall20.recall_at_k(scores, gold, 2) == 0.5


def test_tfidf_and_bm25_use_same_recall_definition() -> None:
    scores = np.array([0.1, 0.8, 0.5, 0.0])
    gold = {1, 2}
    assert run_recall20.recall_at_k(scores, gold, 2) == 1.0
    assert run_tfidf_recall20.recall_at_k(scores, gold, 2) == 1.0


def test_gold_rows_deduplicates_annotations() -> None:
    question = {
        "documents": [
            "https://pubmed.ncbi.nlm.nih.gov/10/",
            "https://pubmed.ncbi.nlm.nih.gov/10/",
            "https://pubmed.ncbi.nlm.nih.gov/20/",
        ]
    }
    assert run_recall20.gold_rows(question, {"10": 0, "20": 1}) == [0, 1]
