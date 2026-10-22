# Hybrid Biomedical Literature Retrieval with BM25 and MedCPT

This project retrieves PubMed articles for biomedical questions. I built a temporal
BioASQ benchmark, established TF-IDF and BM25 baselines, and then tested whether
MedCPT-based semantic retrieval and reranking could recover relevant articles that
lexical ranking placed too low.

The best system combines the top 200 BM25 results with the top 200 MedCPT bi-encoder
results, removes duplicates, and reranks the combined list with a LoRA-adapted MedCPT
cross-encoder. It reached **71.90% mean Recall@20** on the edition-13 test split,
compared with **64.35%** for BM25.

![Validation and test Recall@20](docs/retrieval-results.png)

## Results

| Retrieval system | Validation Recall@20 | Test Recall@20 |
|---|---:|---:|
| TF-IDF cosine | 55.49% | 55.51% |
| BM25 | 69.28% | 64.35% |
| MedCPT bi-encoder | 66.64% | 61.32% |
| BM25 + frozen MedCPT reranker | 73.06% | 68.77% |
| MedCPT candidates + matching LoRA reranker | 75.11% | 69.98% |
| BM25 + LoRA MedCPT reranker | 75.23% | 71.05% |
| **BM25/MedCPT union + LoRA reranker** | **76.55%** | **71.90%** |

The hybrid system improves on BM25 by 7.55 percentage points, or 11.74% relative,
on the test split. It retrieves every annotated article within the top 20 for 40.00%
of test questions; BM25 does so for 29.71%.

All displayed values come from the saved experiment summaries. Exact values and
provenance are in [`results/metrics.json`](results/metrics.json). The checks performed
while preparing the repository are recorded in [`VERIFICATION.md`](VERIFICATION.md).

### What Recall@20 means

For one question, Recall@20 is the fraction of its annotated relevant PubMed articles
found among the first 20 results. The reported value is the mean across questions.
The 71.90% result is therefore a retrieval recall—not answer accuracy, classification
accuracy, or precision.

### Recall@20 capacity and candidate oracle

Some questions have more than 20 annotated articles, so no top-20 ranking can retrieve
all of them. For a question with `g` annotated articles, the maximum possible
Recall@20 is `min(20, g) / g`. Averaging this value over questions gives the top-20
capacity below. The candidate oracle applies the same calculation after restricting
the gold articles to those present in the hybrid BM25/MedCPT candidate union; it asks
what a perfect reranker could achieve with the candidates the system actually found.

| Split | Questions with >20 gold | Full recall possible | Top-20 capacity | Hybrid candidate oracle | BM25 | Hybrid |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 146 (42.94%) | 194 (57.06%) | 83.99% | 83.43% | 69.28% | 76.55% |
| Test | 161 (47.35%) | 179 (52.65%) | 77.20% | 76.90% | 64.35% | 71.90% |

The validation gold counts have median 16, mean 19.73, and maximum 86; the test
counts have median 18, mean 25.96, and maximum 125.

![Recall@20 ceilings](docs/recall-ceilings.png)

On the test split, the hybrid system's 71.90% raw Recall@20 is 93.13% of the
77.20% top-20 capacity. It is 93.49% of its 76.90% candidate-restricted oracle.
These ratios describe how much of each ceiling was achieved; they are not alternative
recall scores. Of the 5.30-point gap to the top-20 capacity, 0.30 points come from
relevant articles missing from the candidate union, while 5.00 points separate the
observed result from a perfect reranking of the existing candidates. The corresponding
validation gaps are 0.57 and 6.88 points. Candidate exclusion therefore accounts for
5.64% of the test gap and 7.60% of the validation gap; the rest is attributable to
ranking within the available candidates.

Across validation and test respectively, BM25 achieved 82.48% and 83.35% of the
top-20 capacity; the hybrid achieved 91.14% and 93.13%. Against its own
candidate-restricted oracle, the hybrid achieved 91.76% and 93.49%.

Full coverage means that all annotated articles for a question were retrieved in the
top 20. It is impossible for the 146 validation and 161 test questions with more than
20 annotated articles.

| Split | System | Full coverage across all questions | Full coverage where possible |
|---|---|---:|---:|
| Validation | BM25 | 32.06% | 56.19% |
| Validation | Hybrid | 42.94% | 75.26% |
| Test | BM25 | 29.71% | 56.42% |
| Test | Hybrid | 40.00% | 75.98% |

The calculations use the actual per-question gold counts and saved candidate rankings,
not the median counts in the split summary. The aggregate output and definitions are
in [`results/recall_ceiling.json`](results/recall_ceiling.json). These ceilings apply
to this project's closed searchable corpus and annotated gold set, not to biomedical
retrieval over PubMed in general.

## Method

```text
BioASQ question
      │
      ├── BM25 top 200 ──────────────┐
      │                              ├── deduplicated union
      └── MedCPT bi-encoder top 200 ─┘          │
                                               ▼
                                 LoRA MedCPT cross-encoder
                                               │
                                               ▼
                                     ranked top 20 articles
```

### Data and temporal split

The questions come from the BioASQ 14b training collection. The searchable corpus was
built by collecting the unique PubMed IDs linked from its relevance annotations and
downloading their titles and abstracts through NCBI E-utilities.

| Split | BioASQ editions | Questions | Median annotated articles |
|---|---:|---:|---:|
| Train | 1–11 | 5,049 | 6 |
| Validation | 12 | 340 | 16 |
| Test | 13 | 340 | 18 |

The download requested 58,048 unique PubMed records. After empty records were
excluded, 58,012 title/abstract records were searchable. One training question had no
annotated article remaining in the corpus; every validation and test question had at
least one.

BM25's `k1` and `b` were selected on edition 12. The frozen cross-encoder candidate
depth was also selected on edition 12 from 50, 100, and 200. Editions 1–11 supplied
the LoRA training pairs. Edition 13 was used for the reported temporal test results.

### Retrieval stages

1. **Lexical baselines:** TF-IDF cosine similarity and BM25 over lowercased
   alphanumeric tokens.
2. **Frozen reranking:** `ncbi/MedCPT-Cross-Encoder` reranks BM25's top 200.
3. **Task adaptation:** pairwise LoRA training uses relevant and non-relevant articles
   sampled from each training question's BM25 candidate list.
4. **Dense candidates:** `ncbi/MedCPT-Query-Encoder` and
   `ncbi/MedCPT-Article-Encoder` provide a semantic top 200.
5. **Hybrid retrieval:** lexical and dense candidates are merged and reranked with the
   BM25-trained LoRA adapter.

The main LoRA run used 25,521 pairs from 5,029 training questions, rank 8, alpha 16,
dropout 0.1, learning rate `1e-4`, and one pass through the pairs. The committed
adapter's SHA-256 is recorded in `results/metrics.json`.

## Setup

Python 3.12 was used for the project.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
```

PyTorch automatically uses CUDA where available. The scripts also support CPU and
Apple Metal, with an MPS fallback enabled for unsupported operations.

The original neural experiments ran with Python 3.12.14, NumPy 2.5.3, PyTorch 2.14.1,
and Transformers 5.18.0. [`requirements-experiment.txt`](requirements-experiment.txt)
records that environment's complete package freeze; the ranges in `pyproject.toml`
remain the portable installation specification. Data preparation, plotting, and test
tools are provided by the corresponding project extras.

Model and dataset downloads are pinned to the revisions used by the experiment:

| Artifact | Revision |
|---|---|
| `JJBauer/bioasq-14b` | `451e95d26c5baadcb825dde145afb1f88c2d21b8` |
| `ncbi/MedCPT-Cross-Encoder` | `71caf65d4927987813984f54c284405a13fcca49` |
| `ncbi/MedCPT-Query-Encoder` | `d83a36cc6b8e3a5c5e9d9d6ba156808c1643dcbc` |
| `ncbi/MedCPT-Article-Encoder` | `d05a736da4bb84ee4057b7f7999485be6ed85465` |

## Prepare the data

The raw BioASQ and PubMed data are not committed. Download the pinned BioASQ
repackaging and then retrieve the linked PubMed records:

```bash
python scripts/prepare_bioasq.py
export NCBI_EMAIL="your-email@example.com"
python scripts/download_pubmed_abstracts.py
```

`NCBI_API_KEY` is optional. The PubMed downloader stores each batch before producing
`data/processed/abstracts.jsonl`, so an interrupted download can continue.

## Reproduce the retrieval runs

Run the stages in this order:

```bash
# Temporal split and lexical baselines
python scripts/tune_bm25.py
python scripts/run_tfidf_recall20.py
python scripts/run_recall20.py
python scripts/eval_bm25_test.py
python scripts/analyze_bm25_misses.py

# Frozen neural reranking and the saved LoRA adapter
python scripts/rerank_medcpt.py
python scripts/train_lora_reranker.py

# Dense retrieval and the final hybrid system
python scripts/rerank_medcpt_bi.py
python scripts/rerank_union_lora.py
```

`train_lora_reranker.py` detects the committed adapter and uses it for evaluation. To
repeat training from scratch, first move that file out of `weights/`; the script will
write a newly trained adapter at the same path. `train_lora_bi.py` reproduces the
separate ablation trained on dense candidate lists.

Neural evaluation is computationally expensive. Intermediate candidate lists,
embeddings, checkpoints, and logs are written under `data/processed/` and ignored by
Git.

Cached rankings are validated before they are reused. A split is complete only when
all expected question IDs are present; duplicate IDs, missing questions, malformed
candidate lists, and mismatched candidate/logit lengths stop the run with an error.
An interrupted run resumes only the missing or unscored questions.

Recompute the Recall@20 ceilings from the saved per-question rankings, then generate
both result figures with:

```bash
python scripts/analyze_recall_ceiling.py
python scripts/plot_results.py
python scripts/plot_recall_ceiling.py
```

## Error analysis

BM25 placed 3,735 annotated test articles in the first 20. Of the annotated articles
it missed, 5,064 shared at least one query term but ranked below position 20; only 28
had no query word. Many errors were therefore ranking errors rather than complete
lexical mismatches. This motivated neural reranking and the addition of dense
candidates.

The MedCPT bi-encoder alone did not beat BM25 at rank 20, but it recovered 593 of the
712 annotated test articles absent from BM25's top 200. Its value was complementary
candidate coverage rather than direct top-20 ranking.

One saved test example shows that complementarity. For “Are biologics used in
pancreatic cancer?” the question has 15 annotated articles. BM25 found one in its top
20, while the hybrid system found nine. Four of the hybrid's first five results had
appeared only at BM25 ranks 57, 93, 153, and 199:

| Hybrid rank | PMID | Annotated relevant | BM25 rank |
|---:|---|---:|---:|
| 1 | [33014285](https://pubmed.ncbi.nlm.nih.gov/33014285/) | Yes | 93 |
| 2 | [14599095](https://pubmed.ncbi.nlm.nih.gov/14599095/) | Yes | 199 |
| 3 | [18518788](https://pubmed.ncbi.nlm.nih.gov/18518788/) | Yes | 57 |
| 4 | [28736633](https://pubmed.ncbi.nlm.nih.gov/28736633/) | Yes | 1 |
| 5 | [26798437](https://pubmed.ncbi.nlm.nih.gov/26798437/) | Yes | 153 |

The question, titles, ranks, and provenance are saved in
[`results/retrieval_example.json`](results/retrieval_example.json).

## Limitations

- **Closed corpus:** the corpus is the union of PubMed articles cited by the complete
  BioASQ collection, including validation and test editions. Relevance annotations
  therefore define corpus membership across all splits. The benchmark measures
  ranking inside this known article pool, not retrieval over all of PubMed.
- **Repeated test evaluation:** several architecture variants were evaluated on
  edition 13 during development. It is a held-out temporal test split, but it should
  not be described as an untouched final test selected only once.
- **Incomplete relevance judgments:** an article not annotated by BioASQ may still be
  relevant. Training negatives and measured precision can contain unjudged positives.
- **Titles and abstracts only:** full text, MeSH terms, citation graphs, and publication
  dates were not used.
- **Distribution shift:** later editions contain substantially more annotated articles
  per question than the training editions.
- **Not an official BioASQ submission:** these experiments use a custom closed-corpus
  setup and should not be compared directly with official challenge leaderboards.

## Repository structure

```text
.
├── scripts/
│   ├── prepare_bioasq.py
│   ├── download_pubmed_abstracts.py
│   ├── tune_bm25.py
│   ├── run_tfidf_recall20.py
│   ├── run_recall20.py
│   ├── rerank_medcpt.py
│   ├── train_lora_reranker.py
│   ├── rerank_medcpt_bi.py
│   ├── rerank_union_lora.py
│   ├── ranking_cache.py
│   ├── analyze_recall_ceiling.py
│   ├── plot_results.py
│   └── plot_recall_ceiling.py
├── results/
│   ├── metrics.json
│   ├── recall_ceiling.json
│   └── retrieval_example.json
├── docs/
│   ├── retrieval-results.png
│   └── recall-ceilings.png
├── weights/medcpt-cross-encoder-lora.pt
├── requirements-experiment.txt
├── tests/
└── data/                         # local, ignored experiment data
```

## Tests

The lightweight test suite checks parsing, split assignment, retrieval metrics, cache
validation and resume behavior, ceiling calculations, committed result consistency,
and the adapter checksum without downloading the dataset or neural models.

```bash
ruff check .
pytest -q
```

The same checks run in GitHub Actions on Python 3.12.

## Data, models, and attribution

- BioASQ questions and annotations remain subject to the
  [BioASQ dataset terms](https://participants-area.bioasq.org/datasets/). The project
  used the [`JJBauer/bioasq-14b`](https://huggingface.co/datasets/JJBauer/bioasq-14b)
  repackaging and does not redistribute it.
- PubMed records are obtained from [NCBI E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/).
  Publisher-supplied abstracts may carry separate copyright restrictions, so the
  downloaded corpus is not committed.
- The MedCPT encoders and cross-encoder are provided by NCBI. See the
  [MedCPT project](https://github.com/ncbi/MedCPT) and individual Hugging Face model
  cards for their terms.

If you build on the project, cite the BioASQ corpus papers and MedCPT:

1. Krithara et al. “BioASQ-QA: A manually curated corpus for Biomedical Question
   Answering.” *Scientific Data*, 2023.
2. Tsatsaronis et al. “An overview of the BIOASQ large-scale biomedical semantic
   indexing and question answering competition.” *BMC Bioinformatics*, 2015.
3. Jin et al. “MedCPT: Contrastive Pre-trained Transformers with large-scale PubMed
   search logs for zero-shot biomedical information retrieval.” *Bioinformatics*, 2023.
