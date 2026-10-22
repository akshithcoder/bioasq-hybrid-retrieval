from scripts import prepare_bioasq, tune_bm25


def test_pmid_from_pubmed_url() -> None:
    assert tune_bm25.pmid_from_url("http://www.ncbi.nlm.nih.gov/pubmed/12345") == "12345"
    assert tune_bm25.pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/12345/") == "12345"


def test_pmid_rejects_non_numeric_tail() -> None:
    assert tune_bm25.pmid_from_url("https://pubmed.ncbi.nlm.nih.gov/not-a-pmid") is None
    assert tune_bm25.pmid_from_url("") is None


def test_tokenizer_is_lowercase_and_alphanumeric() -> None:
    assert tune_bm25.tokenize("TNF-alpha, IL-6 & p53!") == ["tnf", "alpha", "il", "6", "p53"]


def test_temporal_split_boundaries() -> None:
    assert prepare_bioasq.temporal_split(1) == "train"
    assert prepare_bioasq.temporal_split(11) == "train"
    assert prepare_bioasq.temporal_split(12) == "validation"
    assert prepare_bioasq.temporal_split(13) == "test"
