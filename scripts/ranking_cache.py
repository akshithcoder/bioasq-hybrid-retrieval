"""Validation helpers for saved per-question ranking files."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_rankings(
    path: Path,
    *,
    min_candidates: int = 1,
    max_candidates: int | None = None,
    require_question: bool = False,
    require_logits: bool = False,
) -> dict[str, dict]:
    if not path.exists():
        return {}

    rows = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")

        qid = row.get("id")
        if not isinstance(qid, str) or not qid:
            raise ValueError(f"{path}:{line_number}: missing question ID")
        if qid in rows:
            raise ValueError(f"{path}:{line_number}: duplicate question ID {qid}")

        gold = _string_list(row.get("gold"), path, line_number, "gold")
        candidates = _string_list(row.get("pmids"), path, line_number, "pmids")
        if not gold:
            raise ValueError(f"{path}:{line_number}: empty gold list")
        if len(gold) != len(set(gold)):
            raise ValueError(f"{path}:{line_number}: duplicate gold PMIDs")
        if len(candidates) != len(set(candidates)):
            raise ValueError(f"{path}:{line_number}: duplicate candidate PMIDs")
        if len(candidates) < min_candidates:
            raise ValueError(f"{path}:{line_number}: only {len(candidates)} candidates")
        if max_candidates is not None and len(candidates) > max_candidates:
            raise ValueError(f"{path}:{line_number}: {len(candidates)} candidates exceeds {max_candidates}")

        if require_question and (
            not isinstance(row.get("question"), str) or not row["question"].strip()
        ):
            raise ValueError(f"{path}:{line_number}: missing question text")

        logits = row.get("logits")
        if require_logits and logits is None:
            raise ValueError(f"{path}:{line_number}: missing logits")
        if logits is not None:
            if not isinstance(logits, list) or len(logits) != len(candidates):
                raise ValueError(f"{path}:{line_number}: candidate/logit length mismatch")
            if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in logits):
                raise ValueError(f"{path}:{line_number}: non-numeric logit")
            if any(not math.isfinite(value) for value in logits):
                raise ValueError(f"{path}:{line_number}: non-finite logit")

        rows[qid] = row
    return rows


def missing_question_ids(
    rows: dict[str, dict], expected_ids: list[str], *, source: str
) -> list[str]:
    expected = set(expected_ids)
    unexpected = sorted(set(rows) - expected)
    if unexpected:
        preview = ", ".join(unexpected[:3])
        raise ValueError(f"{source}: {len(unexpected)} unexpected question IDs ({preview})")
    return [qid for qid in expected_ids if qid not in rows]


def require_complete(rows: dict[str, dict], expected_ids: list[str], *, source: str) -> None:
    missing = missing_question_ids(rows, expected_ids, source=source)
    if missing:
        preview = ", ".join(missing[:3])
        raise ValueError(f"{source}: missing {len(missing)} expected questions ({preview})")


def _string_list(value, path: Path, line_number: int, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{path}:{line_number}: invalid {field} list")
    return value
