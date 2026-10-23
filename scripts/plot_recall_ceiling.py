"""Plot observed recall against candidate and top-k ceilings."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "recall_ceiling.json"
OUTPUT_PATH = ROOT / "docs" / "recall-ceilings.png"


def main() -> None:
    import matplotlib.pyplot as plt

    payload = json.loads(RESULTS_PATH.read_text())
    splits = payload["splits"]
    labels = ["Validation", "Test"]
    keys = ["validation", "test"]
    x = list(range(len(labels)))
    width = 0.19
    offsets = (-1.5, -0.5, 0.5, 1.5)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for axis, top_k in zip(axes, (20, 100), strict=True):
        candidate_key = (
            "hybrid_candidate_restricted"
            if top_k == 20
            else "hybrid_candidate_restricted_at_100"
        )
        series = {
            "BM25": [
                100 * splits[key]["systems"]["bm25"][f"recall_at_{top_k}"] for key in keys
            ],
            "Hybrid": [
                100 * splits[key]["systems"]["hybrid"][f"recall_at_{top_k}"] for key in keys
            ],
            "Hybrid candidate oracle": [
                100 * splits[key]["ceilings"][candidate_key] for key in keys
            ],
            "Top-k capacity": [
                100 * splits[key]["ceilings"][f"top_{top_k}_capacity"] for key in keys
            ],
        }
        for (name, values), offset in zip(series.items(), offsets, strict=True):
            bars = axis.bar([value + offset * width for value in x], values, width, label=name)
            axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=8)
        axis.set_xticks(x, labels)
        axis.set_ylim(0, 106)
        axis.set_title(f"Recall@{top_k}")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Macro recall (%)")
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("Observed retrieval recall and attainable ceilings")
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
