"""Download the pinned BioASQ 14b repackaging and write the expected JSON file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw" / "bioasq-14b" / "training14b.json"
DATASET = "JJBauer/bioasq-14b"
REVISION = "451e95d26c5baadcb825dde145afb1f88c2d21b8"
QUESTION_TYPES = ("factoid", "list", "summary", "yesno")
EXPECTED_QUESTIONS = 5729
EXPECTED_SPLITS = {"train": 5049, "validation": 340, "test": 340}


def temporal_split(edition: int) -> str:
    if edition <= 11:
        return "train"
    if edition == 12:
        return "validation"
    if edition == 13:
        return "test"
    raise ValueError(f"unexpected BioASQ edition: {edition}")


def validate(rows: list[dict]) -> dict[str, int]:
    if len(rows) != EXPECTED_QUESTIONS:
        raise ValueError(f"expected {EXPECTED_QUESTIONS} questions, found {len(rows)}")
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("question IDs are not unique")
    counts = {name: 0 for name in EXPECTED_SPLITS}
    for row in rows:
        counts[temporal_split(int(row["asq_challenge"]))] += 1
    if counts != EXPECTED_SPLITS:
        raise ValueError(f"unexpected temporal split counts: {counts}")
    return counts


def download(output: Path = OUTPUT) -> None:
    from datasets import load_dataset

    dataset = load_dataset(DATASET, revision=REVISION)
    rows = [dict(row) for name in QUESTION_TYPES for row in dataset[name]]
    counts = validate(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"questions": rows}, ensure_ascii=False))
    print(f"wrote {len(rows)} questions to {output}")
    print(f"temporal splits: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
