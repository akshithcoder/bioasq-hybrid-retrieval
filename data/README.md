# Local data layout

The repository does not redistribute BioASQ questions or PubMed titles and abstracts.
After running the preparation commands in the main README, the scripts expect:

```text
data/
├── raw/
│   ├── bioasq-14b/training14b.json
│   └── pubmed/
│       ├── parsed/
│       └── xml/
└── processed/
    ├── abstracts.jsonl
    ├── rerank/
    ├── lora/
    ├── medcpt_bi/
    └── union_lora/
```

`training14b.json` contains BioASQ material and remains subject to the BioASQ terms.
PubMed records can contain publisher-supplied abstracts; review the NLM terms before
redistributing them. Both trees are ignored by Git.
