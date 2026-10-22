# Result provenance

`metrics.json` is a compact record of the aggregate outputs already present in the
original research workspace. The values were transcribed without alteration and were
later reproduced from the original corpus and saved rankings as described in
`VERIFICATION.md`. Absolute local paths and per-question text were left out of the
public artifact.

The source summaries were:

- `tfidf_recall20.json` and `tfidf_length_recall20.json`
- `bm25_recall20.json` and `bm25_test.json`
- `bm25_miss_analysis.json`
- `medcpt_rerank_recall20.json` and `medcpt_bi_recall.json`
- `lora_recall20.json` and `lora_bi_recall20.json`
- `union_lora_recall20.json`

`recall_ceiling.json` was computed from the complete saved validation and test
rankings. It records the per-question Recall@20 capacity, the hybrid
candidate-restricted oracle, full-coverage rates, and the remaining candidate and
ranking gaps. Run `scripts/analyze_recall_ceiling.py` to reproduce it when the ignored
ranking artifacts are available.

`retrieval_example.json` records one genuine edition-13 example selected from the
saved rankings and local searchable corpus. It contains the question, article titles,
PMIDs, relevance labels, and original BM25 ranks; it does not contain PubMed abstract
text.

The charts in `docs/retrieval-results.png` and `docs/recall-ceilings.png` are generated
from `metrics.json` and `recall_ceiling.json` by `scripts/plot_results.py` and
`scripts/plot_recall_ceiling.py`, respectively.
