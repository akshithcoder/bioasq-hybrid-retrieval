import pytest

from scripts import analyze_recall_ceiling


def row(qid: str, gold: list[str], pmids: list[str]) -> dict:
    return {
        "id": qid,
        "gold": gold,
        "pmids": pmids,
        "logits": list(reversed(range(len(pmids)))),
    }


def test_capacity_and_candidate_ceilings_use_per_question_counts() -> None:
    gold_10 = [f"a{index}" for index in range(10)]
    gold_30 = [f"b{index}" for index in range(30)]
    gold_5 = [f"c{index}" for index in range(5)]

    bm25 = [
        row("q1", gold_10, gold_10[:5] + [f"x{index}" for index in range(20)]),
        row("q2", gold_30, gold_30[:10] + [f"y{index}" for index in range(20)]),
        row("q3", gold_5, gold_5[:2] + [f"z{index}" for index in range(23)]),
    ]
    hybrid = [
        row("q1", gold_10, gold_10 + [f"h1-{index}" for index in range(15)]),
        row("q2", gold_30, gold_30[:15] + [f"h2-{index}" for index in range(10)]),
        row(
            "q3",
            gold_5,
            gold_5[:3] + [f"h3-{index}" for index in range(18)] + gold_5[3:],
        ),
    ]

    result = analyze_recall_ceiling.analyze_split(bm25, hybrid)

    assert result["gold_articles_per_question"]["median"] == 10
    assert result["gold_articles_per_question"]["over_20_questions"] == 1
    assert result["gold_articles_per_question"]["full_recall_possible_questions"] == 2
    assert result["ceilings"]["top_20_capacity"] == pytest.approx((1 + 20 / 30 + 1) / 3)
    assert result["ceilings"]["hybrid_candidate_restricted"] == pytest.approx((1 + 15 / 30 + 1) / 3)
    assert result["systems"]["hybrid"]["recall_at_20"] == pytest.approx((1 + 15 / 30 + 3 / 5) / 3)
    assert result["ceilings"]["top_20_capacity"] >= result["ceilings"]["hybrid_candidate_restricted"]
    assert result["systems"]["hybrid"]["full_coverage_at_20_among_possible_questions"] == 0.5


def test_mismatched_gold_sets_are_rejected() -> None:
    bm25 = [row("q1", ["one"], ["one"])]
    hybrid = [row("q1", ["two"], ["two"])]
    with pytest.raises(ValueError, match="gold PMID mismatch"):
        analyze_recall_ceiling.analyze_split(bm25, hybrid)
