"""Plot the saved validation and test Recall@20 values."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "results" / "metrics.json"
OUTPUT_PATH = ROOT / "docs" / "retrieval-results.png"


def main() -> None:
    import matplotlib.pyplot as plt

    payload = json.loads(METRICS_PATH.read_text())
    systems = payload["systems"]
    names = [row["name"] for row in systems]
    validation = [100 * row["validation_recall_at_20"] for row in systems]
    test = [100 * row["test_recall_at_20"] for row in systems]

    y = range(len(systems))
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.barh([value + 0.2 for value in y], validation, height=0.38, label="Validation")
    axis.barh([value - 0.2 for value in y], test, height=0.38, label="Test")
    axis.set_yticks(list(y), names)
    axis.set_xlim(0, 82)
    axis.set_xlabel("Mean Recall@20 (%)")
    axis.set_title("Biomedical article retrieval on the temporal BioASQ split")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(loc="lower right")
    axis.invert_yaxis()
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
