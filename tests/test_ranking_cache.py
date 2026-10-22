import json

import pytest

from scripts import ranking_cache


def ranking(qid: str) -> dict:
    return {
        "id": qid,
        "gold": [f"gold-{qid}"],
        "pmids": [f"candidate-{qid}"],
        "logits": [0.5],
    }


def test_300_of_340_questions_is_not_complete(tmp_path) -> None:
    path = tmp_path / "partial.jsonl"
    path.write_text("\n".join(json.dumps(ranking(str(index))) for index in range(300)) + "\n")
    rows = ranking_cache.load_rankings(path, require_logits=True)
    expected = [str(index) for index in range(340)]

    assert ranking_cache.missing_question_ids(rows, expected, source=str(path)) == expected[300:]
    with pytest.raises(ValueError, match="missing 40 expected questions"):
        ranking_cache.require_complete(rows, expected, source=str(path))


def test_duplicate_question_id_is_rejected(tmp_path) -> None:
    path = tmp_path / "duplicate.jsonl"
    path.write_text(json.dumps(ranking("same")) + "\n" + json.dumps(ranking("same")) + "\n")
    with pytest.raises(ValueError, match="duplicate question ID"):
        ranking_cache.load_rankings(path)


def test_candidate_logit_length_mismatch_is_rejected(tmp_path) -> None:
    path = tmp_path / "mismatch.jsonl"
    row = ranking("q1")
    row["pmids"].append("second-candidate")
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="candidate/logit length mismatch"):
        ranking_cache.load_rankings(path, require_logits=True)


def test_missing_and_empty_cache_are_valid_resume_states(tmp_path) -> None:
    missing = tmp_path / "missing.jsonl"
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")

    assert ranking_cache.load_rankings(missing) == {}
    assert ranking_cache.load_rankings(empty) == {}


def test_malformed_row_is_rejected(tmp_path) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text('{"id": "unfinished"\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        ranking_cache.load_rankings(path)


def test_short_candidate_list_is_rejected(tmp_path) -> None:
    path = tmp_path / "short.jsonl"
    path.write_text(json.dumps(ranking("q1")) + "\n")
    with pytest.raises(ValueError, match="only 1 candidates"):
        ranking_cache.load_rankings(path, min_candidates=2)


def test_empty_required_question_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty-question.jsonl"
    row = ranking("q1")
    row["question"] = "  "
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="missing question text"):
        ranking_cache.load_rankings(path, require_question=True)
