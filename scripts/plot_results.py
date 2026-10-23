"""Plot the saved validation and test Recall@20 and Recall@100 values."""

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
    y = range(len(systems))
    figure, axes = plt.subplots(1, 2, figsize=(14, 6.8), sharey=True)
    for axis, top_k in zip(axes, (20, 100), strict=True):
        validation = [100 * row[f"validation_recall_at_{top_k}"] for row in systems]
        test = [100 * row[f"test_recall_at_{top_k}"] for row in systems]
        validation_bars = axis.barh(
            [value + 0.2 for value in y], validation, height=0.38, label="Validation"
        )
        test_bars = axis.barh(
            [value - 0.2 for value in y], test, height=0.38, label="Test"
        )
        axis.bar_label(validation_bars, fmt="%.1f", padding=2, fontsize=8)
        axis.bar_label(test_bars, fmt="%.1f", padding=2, fontsize=8)
        axis.set_yticks(list(y), names)
        axis.set_xlim(0, 105)
        axis.set_xlabel(f"Macro Recall@{top_k} (%)")
        axis.set_title(f"Recall@{top_k}")
        axis.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncols=2)
    figure.suptitle("Biomedical article retrieval on the temporal BioASQ split")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
