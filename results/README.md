# Result provenance

`metrics.json` is a compact record of the retrieval evaluation. Recall@20 values were
transcribed from the original summaries and later reproduced. Recall@100 values were
computed from the original corpus for TF-IDF and from complete saved per-question
rankings for the other systems. `VERIFICATION.md` records the checks. Absolute local
paths and per-question text were left out of the public artifact.

The source summaries were:

- `tfidf_recall20.json` and `tfidf_length_recall20.json`
- `bm25_recall20.json` and `bm25_test.json`
- `bm25_miss_analysis.json`
- `medcpt_rerank_recall20.json` and `medcpt_bi_recall.json`
- `lora_recall20.json` and `lora_bi_recall20.json`
- `union_lora_recall20.json`

The neural source artifacts are the 200-candidate BM25, MedCPT, and LoRA ranking files
and the roughly 330-candidate hybrid union files under `data/processed/`. Each contains
all 340 validation or test questions and enough ranked candidates to evaluate both
cutoffs. The TF-IDF run did not save per-question rankings, so its Recall@100 value was
recomputed from the original question and corpus files.

`recall_ceiling.json` was computed from those complete saved rankings. It records the
per-question Recall@20 and Recall@100 capacities, hybrid candidate-restricted oracles,
full-coverage rates, every saved-ranking system, and the remaining candidate and
ranking gaps. Run `scripts/analyze_recall_ceiling.py` to reproduce it when the ignored
ranking artifacts are available.

`retrieval_example.json` records one genuine edition-13 example selected from the
saved rankings and local searchable corpus. It contains the question, article titles,
PMIDs, relevance labels, and original BM25 ranks; it does not contain PubMed abstract
text.

The charts in `docs/retrieval-results.png` and `docs/recall-ceilings.png` show both
cutoffs. They are generated from `metrics.json` and `recall_ceiling.json` by
`scripts/plot_results.py` and `scripts/plot_recall_ceiling.py`, respectively.
