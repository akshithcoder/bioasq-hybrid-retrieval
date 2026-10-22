"""Plot observed Recall@20 against its candidate and top-k ceilings."""

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
    series = {
        "BM25": [100 * splits[key]["systems"]["bm25"]["recall_at_20"] for key in keys],
        "Hybrid": [100 * splits[key]["systems"]["hybrid"]["recall_at_20"] for key in keys],
        "Hybrid candidate oracle": [
            100 * splits[key]["ceilings"]["hybrid_candidate_restricted"] for key in keys
        ],
        "Top-20 capacity": [100 * splits[key]["ceilings"]["top_20_capacity"] for key in keys],
    }

    x = list(range(len(labels)))
    width = 0.19
    offsets = (-1.5, -0.5, 0.5, 1.5)
    figure, axis = plt.subplots(figsize=(9, 5.4))
    for (name, values), offset in zip(series.items(), offsets, strict=True):
        bars = axis.bar([value + offset * width for value in x], values, width, label=name)
        axis.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=3, fontsize=9)

    axis.set_xticks(x, labels)
    axis.set_ylim(0, 92)
    axis.set_ylabel("Macro Recall@20 (%)")
    axis.set_title("Observed retrieval recall and attainable ceilings")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower left", ncols=2)
    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
