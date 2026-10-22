# Verification report

Verified on 2026-10-22 on macOS with Python 3.12.14.

## Passed

- Created a clean virtual environment and installed the project from `pyproject.toml`.
- Installed the `test`, `neural`, `data`, and `plots` optional dependencies.
- `python -m pip check`: no broken requirements.
- `ruff check .`: passed.
- `pytest -q`: 10 tests passed.
- Loaded `ncbi/MedCPT-Cross-Encoder` and the committed LoRA adapter on Apple Metal.
  The adapter contained 48 tensors. A two-pair inference smoke test returned finite
  scores and ranked the medically relevant article above an unrelated article.
- Downloaded the pinned `JJBauer/bioasq-14b` revision. It contained 5,729 unique
  questions and reproduced the 5,049/340/340 temporal split. Its records matched the
  original experiment input exactly when compared by question ID and content.
- Regenerated the TF-IDF and BM25 metrics using the public scripts and the original
  local corpus. Values matched the saved summaries to floating-point precision.
- Recomputed every validation and test neural Recall@20 value in the README from the
  saved per-question candidate rankings and logits. Full-coverage values also matched.
- Regenerated `docs/retrieval-results.png` from `results/metrics.json`.
- Checked every external README link successfully.

## Reproduced metrics

| Check | Validation | Test |
|---|---:|---:|
| TF-IDF Recall@20 | 0.5549069568 | 0.5550812889 |
| BM25 Recall@20 | 0.6928069749 | 0.6434538265 |
| Frozen MedCPT reranker Recall@20 | 0.7305942596 | 0.6877363505 |
| BM25 + LoRA reranker Recall@20 | 0.7523244220 | 0.7105498908 |
| MedCPT candidates + matching LoRA Recall@20 | 0.7511184119 | 0.6998405698 |
| Hybrid union + LoRA Recall@20 | 0.7654934977 | 0.7190013066 |
| Hybrid full coverage@20 | 0.4294117647 | 0.4000000000 |

The separate BM25 test script also reproduced Recall@10 of 0.4994344599 and
Precision@10 of 0.6532352941.

## Not rerun

- The complete 58,048-record PubMed download was not repeated. The existing corpus
  was used for metric reproduction; the downloader was inspected and its NCBI request
  configuration was updated for a public repository.
- LoRA training was not repeated. The adapter checksum, checkpoint loading, and real
  inference were verified instead.
- Full MedCPT encoding and reranking were not repeated. Neural metrics were recomputed
  from the saved per-question candidate lists and logits.
- GitHub Actions cannot be observed until the repository is pushed. Its two commands
  were run successfully in the clean local environment.

No historical experiment artifact in the original research workspace was modified.
