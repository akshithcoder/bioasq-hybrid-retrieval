# Verification report

Verified on 2026-10-22 on macOS with Python 3.12.14.

## Passed

- Created a clean virtual environment and installed the project from `pyproject.toml`.
- Installed the `test`, `neural`, `data`, and `plots` optional dependencies.
- `python -m pip check`: no broken requirements.
- `ruff check .`: passed.
- `pytest -q`: 23 tests passed.
- Loaded `ncbi/MedCPT-Cross-Encoder` and the committed LoRA adapter on Apple Metal.
  The adapter contained 48 tensors. A two-pair inference smoke test returned finite
  scores and ranked the medically relevant article above an unrelated article.
- Downloaded the pinned `JJBauer/bioasq-14b` revision. It contained 5,729 unique
  questions and reproduced the 5,049/340/340 temporal split. Its records matched the
  original experiment input exactly when compared by question ID and content.
- Regenerated the TF-IDF and BM25 metrics using the public scripts and the original
  local corpus. Values matched the saved summaries to floating-point precision.
- Recomputed every validation and test neural Recall@20 and Recall@100 value in the
  README from the saved per-question candidate rankings and logits. Full-coverage
  values also matched.
- Recomputed TF-IDF Recall@100 from the original questions and searchable corpus.
- Computed the top-20 and top-100 capacities and hybrid candidate-restricted oracles
  from all 340 saved per-question rankings in each split.
- Validated all ten original BM25, dense, LoRA, dense-LoRA, and union ranking files for
  complete question coverage, unique IDs and candidates, and matching candidate/logit
  lengths. The BM25 and dense inputs also agree with the current question text and
  searchable-corpus gold PMIDs.
- Confirmed that all three pinned MedCPT revisions and the pinned BioASQ dataset
  revision remain available from Hugging Face.
- Regenerated both-cutoff versions of `docs/retrieval-results.png` and
  `docs/recall-ceilings.png` from the saved JSON results.
- Checked every external README link successfully.
- The CI run for the published Recall@100 commit `60eec9c` completed successfully:
  [GitHub Actions run 39894040877](https://github.com/akshithcoder/bioasq-hybrid-retrieval/actions/runs/39894040877).

## Reproduced metrics

| Check | Validation | Test |
|---|---:|---:|
| TF-IDF Recall@20 | 0.5549069568 | 0.5550812889 |
| BM25 Recall@20 | 0.6928069749 | 0.6434538265 |
| MedCPT bi-encoder Recall@20 | 0.6663962564 | 0.6131763504 |
| Frozen MedCPT reranker Recall@20 | 0.7305942596 | 0.6877363505 |
| MedCPT candidates + BM25-trained LoRA Recall@20 | 0.7499080318 | 0.6988125490 |
| BM25 + LoRA reranker Recall@20 | 0.7523244220 | 0.7105498908 |
| MedCPT candidates + matching LoRA Recall@20 | 0.7511184119 | 0.6998405698 |
| Hybrid union + LoRA Recall@20 | 0.7654934977 | 0.7190013066 |
| Hybrid full coverage@20 | 0.4294117647 | 0.4000000000 |

| Recall@100 check | Validation | Test |
|---|---:|---:|
| TF-IDF cosine | 0.8395160483 | 0.8287511203 |
| BM25 | 0.9241840622 | 0.9048314532 |
| MedCPT bi-encoder | 0.9334247670 | 0.9069377664 |
| Frozen MedCPT reranker | 0.9456882917 | 0.9333978055 |
| MedCPT candidates + BM25-trained LoRA | 0.9590324973 | 0.9484771237 |
| MedCPT candidates + matching LoRA | 0.9590522080 | 0.9493493927 |
| BM25 + LoRA reranker | 0.9484799836 | 0.9367577213 |
| Hybrid union + LoRA | 0.9808068731 | 0.9793512577 |

The historical Recall@100 reference of approximately 0.907 matches the standalone
MedCPT bi-encoder result of 0.9069377664. The historical hybrid and achievable-ceiling
references of approximately 0.979 and 0.990 match 0.9793512577 and the 0.9907725169
hybrid candidate oracle.

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
No historical experiment artifact in the original research workspace was modified.
