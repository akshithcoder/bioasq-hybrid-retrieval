import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics.json"
CEILINGS = ROOT / "results" / "recall_ceiling.json"
ADAPTER = ROOT / "weights" / "medcpt-cross-encoder-lora.pt"


def load_metrics() -> dict:
    return json.loads(METRICS.read_text())


def test_saved_result_progression() -> None:
    rows = {row["name"]: row for row in load_metrics()["systems"]}
    bm25 = rows["BM25"]["test_recall_at_20"]
    hybrid = rows["BM25/MedCPT union + LoRA reranker"]["test_recall_at_20"]
    assert bm25 == pytest.approx(0.6434538265344673)
    assert hybrid == pytest.approx(0.7190013065914587)
    assert hybrid - bm25 == pytest.approx(0.0755474800569914)


def test_full_coverage_counts_are_consistent() -> None:
    payload = load_metrics()
    rows = {row["name"]: row for row in payload["systems"]}
    hybrid = rows["BM25/MedCPT union + LoRA reranker"]
    assert payload["bm25"]["test_full_coverage_at_20"] * 340 == pytest.approx(101)
    assert hybrid["test_full_coverage_at_20"] * 340 == pytest.approx(136)


def test_committed_adapter_checksum() -> None:
    expected = load_metrics()["lora"]["adapter_sha256"]
    digest = hashlib.sha256(ADAPTER.read_bytes()).hexdigest()
    assert digest == expected


def test_committed_recall_ceilings_match_metrics() -> None:
    metrics = load_metrics()
    systems = {row["name"]: row for row in metrics["systems"]}
    ceilings = json.loads(CEILINGS.read_text())["splits"]
    for split in ("validation", "test"):
        result = ceilings[split]
        hybrid = systems["BM25/MedCPT union + LoRA reranker"][f"{split}_recall_at_20"]
        assert result["questions"] == 340
        assert result["systems"]["hybrid"]["recall_at_20"] == pytest.approx(hybrid)
        assert result["ceilings"]["top_20_capacity"] >= result["ceilings"]["hybrid_candidate_restricted"]
        assert result["ceilings"]["hybrid_candidate_restricted"] >= hybrid
