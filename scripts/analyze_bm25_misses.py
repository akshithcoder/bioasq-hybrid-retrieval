"""Classify edition-13 gold articles that BM25 did not place in the top 20."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune_bm25 as bm

K1 = 0.9
B = 0.4
TOP_K = 20
OUT_PATH = bm.ROOT / "data" / "processed" / "bm25_miss_analysis.json"


def gold_pmids(question) -> list[str]:
    found = []
    for url in question["documents"]:
        pmid = bm.pmid_from_url(url)
        if pmid and pmid not in found:
            found.append(pmid)
    return found


def snippet_info(question) -> dict[str, list[dict]]:
    by_pmid = {}
    for snippet in question.get("snippets") or []:
        pmid = bm.pmid_from_url(snippet.get("document") or "")
        if not pmid:
            continue
        by_pmid.setdefault(pmid, []).append(
            {
                "section": snippet.get("beginSection") or "",
                "text": snippet.get("text") or "",
            }
        )
    return by_pmid


def main() -> None:
    questions = [
        question
        for question in bm.load_questions()
        if int(question["asq_challenge"]) == 13
    ]
    records = {}
    with bm.ABSTRACTS_PATH.open() as handle:
        for line in handle:
            record = json.loads(line)
            records[record["pmid"]] = record
    pmids, docs = bm.load_corpus()
    pmid_to_row = {pmid: row for row, pmid in enumerate(pmids)}
    idf, posting_docs, posting_tfs, lengths, avgdl = bm.build_index(docs)
    length_norm = (1 - B) + B * (lengths / avgdl)

    bins = Counter()
    section_bins = Counter()
    rank_bins = Counter()
    question_recalls = []
    examples = {"no_query_word": [], "matched_but_below_20": [], "outside_abstract": []}
    n_docs = len(pmids)

    for question in questions:
        query = [
            token
            for token in dict.fromkeys(bm.tokenize(question["question"]))
            if token in idf
        ]
        scores = np.zeros(n_docs, dtype=np.float64)
        for token in query:
            rows = posting_docs[token]
            tfs = posting_tfs[token]
            scores[rows] += idf[token] * (tfs * (K1 + 1.0)) / (tfs + K1 * length_norm[rows])
        top = np.argpartition(scores, -TOP_K)[-TOP_K:]
        found = {int(row) for row in top if scores[row] > 0}
        snippets = snippet_info(question)
        in_corpus = []
        for pmid in gold_pmids(question):
            row = pmid_to_row.get(pmid)
            record = records.get(pmid)
            sections = [item["section"] for item in snippets.get(pmid, [])]
            outside = bool(sections) and all(
                section not in {"abstract", "title"} for section in sections
            )
            if row is None:
                bins["no_text_in_pubmed"] += 1
                continue
            in_corpus.append(pmid)
            hit = row in found
            if hit:
                bins["in_top_20"] += 1
                continue
            title_only = bool(record and record.get("abstract_missing") and record.get("title"))
            if title_only:
                kind = "title_only"
            elif scores[row] <= 0:
                kind = "no_query_word"
            else:
                kind = "matched_but_below_20"
            bins[kind] += 1
            rank = int(1 + np.sum(scores > scores[row]))
            if kind == "matched_but_below_20":
                if rank <= 50:
                    rank_bins["21-50"] += 1
                elif rank <= 100:
                    rank_bins["51-100"] += 1
                elif rank <= 500:
                    rank_bins["101-500"] += 1
                else:
                    rank_bins["501+"] += 1
            if outside:
                section_bins["miss_snippet_outside_abstract"] += 1
            else:
                section_bins["miss_snippet_in_title_or_abstract"] += 1
            example = {
                "question": question["question"],
                "type": question["type"],
                "pmid": pmid,
                "rank": rank,
                "title": (record or {}).get("title") or "",
                "section": sections[0] if sections else "",
                "snippet": (snippets.get(pmid) or [{"text": ""}])[0]["text"][:280],
            }
            if kind in examples and len(examples[kind]) < 8:
                examples[kind].append(example)
            if outside and len(examples["outside_abstract"]) < 8:
                examples["outside_abstract"].append(example)
        if in_corpus:
            hits = sum(pmid_to_row[pmid] in found for pmid in in_corpus)
            question_recalls.append(hits / len(in_corpus))

    recalls = np.array(question_recalls)
    summary = {
        "k1": K1,
        "b": B,
        "top_k": TOP_K,
        "questions": len(question_recalls),
        "mean_recall_at_20": float(recalls.mean()),
        "questions_with_recall_0": int(np.sum(recalls == 0)),
        "questions_with_recall_1": int(np.sum(recalls == 1)),
        "gold_article_bins": dict(bins),
        "matched_miss_ranks": dict(rank_bins),
        "missed_article_snippet_location": dict(section_bins),
        "examples": examples,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in summary if key != "examples"}, indent=2))
    for name, items in examples.items():
        print(f"\n## {name}")
        for item in items[:5]:
            print(
                f"- {item['type']} rank={item['rank']} section={item['section']}\n"
                f"  Q: {item['question']}\n"
                f"  T: {item['title'][:180]}\n"
                f"  S: {item['snippet'][:180]}"
            )


if __name__ == "__main__":
    main()
