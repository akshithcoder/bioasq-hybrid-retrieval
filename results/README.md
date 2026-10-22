# Result provenance

`metrics.json` is a compact record of the aggregate outputs already present in the
original research workspace. The values were not recomputed or altered while this
repository was prepared. Absolute local paths and per-question text were deliberately
left out of the public artifact.

The source summaries were:

- `tfidf_recall20.json` and `tfidf_length_recall20.json`
- `bm25_recall20.json` and `bm25_test.json`
- `bm25_miss_analysis.json`
- `medcpt_rerank_recall20.json` and `medcpt_bi_recall.json`
- `lora_recall20.json` and `lora_bi_recall20.json`
- `union_lora_recall20.json`

The chart in `docs/retrieval-results.png` is generated directly from `metrics.json` by
`scripts/plot_results.py`.
