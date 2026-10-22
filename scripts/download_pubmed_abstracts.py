"""Download title + abstract for every unique BioASQ gold PMID.

The question file only stores PubMed links. This script turns those links
into text we can later search. Each batch is saved on its own, so a stopped
run continues from the first missing batch.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data" / "raw" / "bioasq-14b" / "training14b.json"
PUBMED_DIR = ROOT / "data" / "raw" / "pubmed"
PMID_PATH = PUBMED_DIR / "pmids.txt"
XML_DIR = PUBMED_DIR / "xml"
PARSED_DIR = PUBMED_DIR / "parsed"
OUT_PATH = ROOT / "data" / "processed" / "abstracts.jsonl"
FAIL_PATH = PUBMED_DIR / "failed_pmids.txt"

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BATCH_SIZE = 200
PAUSE_SECONDS = 0.4
MAX_RETRIES = 4
TOOL = "BioASQHybridRetrieval"
EMAIL = os.environ.get("NCBI_EMAIL")
API_KEY = os.environ.get("NCBI_API_KEY")


def pmid_from_url(url: str) -> str | None:
    tail = url.rstrip("/").split("/")[-1]
    return tail if tail.isdigit() else None


def load_pmids() -> list[str]:
    if PMID_PATH.exists():
        pmids = [line.strip() for line in PMID_PATH.read_text().splitlines() if line.strip()]
        return pmids
    data = json.loads(QUESTIONS.read_text())
    seen = set()
    ordered = []
    skipped = []
    for question in data["questions"]:
        for url in question["documents"]:
            pmid = pmid_from_url(url)
            if pmid is None:
                skipped.append(url)
                continue
            if pmid not in seen:
                seen.add(pmid)
                ordered.append(pmid)
    PUBMED_DIR.mkdir(parents=True, exist_ok=True)
    PMID_PATH.write_text("\n".join(ordered) + "\n")
    if skipped:
        (PUBMED_DIR / "non_pmid_urls.txt").write_text("\n".join(skipped) + "\n")
    print(f"saved {len(ordered)} unique PMIDs")
    return ordered


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def text_of(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def abstract_of(article: ET.Element) -> str:
    parts = []
    for node in article.iter():
        if local_name(node.tag) != "AbstractText":
            continue
        piece = "".join(node.itertext()).strip()
        if not piece:
            continue
        label = node.attrib.get("Label") or node.attrib.get("NlmCategory")
        parts.append(f"{label}: {piece}" if label else piece)
    return "\n".join(parts).strip()


def parse_batch(xml_bytes: bytes) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    records = []
    for node in root.iter():
        kind = local_name(node.tag)
        if kind not in {"PubmedArticle", "PubmedBookArticle"}:
            continue
        pmid = ""
        title = ""
        for child in node.iter():
            name = local_name(child.tag)
            if name == "PMID" and not pmid:
                pmid = (child.text or "").strip()
            elif name in {"ArticleTitle", "BookTitle"} and not title:
                title = text_of(child)
        abstract = abstract_of(node)
        if not pmid:
            continue
        records.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "abstract_missing": abstract == "",
            }
        )
    return records


def fetch_batch(pmids: list[str]) -> bytes:
    params = {
        "db": "pubmed",
        "retmode": "xml",
        "tool": TOOL,
        "id": ",".join(pmids),
    }
    if EMAIL:
        params["email"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY
    form = urllib.parse.urlencode(params).encode()
    request = urllib.request.Request(
        EFETCH,
        data=form,
        headers={"User-Agent": f"{TOOL}/0.1 (educational retrieval baseline)"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def download(pmids: list[str]) -> None:
    XML_DIR.mkdir(parents=True, exist_ok=True)
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    batches = [pmids[i : i + BATCH_SIZE] for i in range(0, len(pmids), BATCH_SIZE)]
    print(f"{len(pmids)} PMIDs in {len(batches)} batches")
    for index, batch in enumerate(batches):
        parsed_path = PARSED_DIR / f"batch_{index:05d}.jsonl"
        xml_path = XML_DIR / f"batch_{index:05d}.xml"
        if parsed_path.exists() and parsed_path.stat().st_size > 0:
            continue
        xml_bytes = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                xml_bytes = fetch_batch(batch)
                break
            except (urllib.error.URLError, TimeoutError) as error:
                wait = PAUSE_SECONDS * (2**attempt)
                print(f"batch {index} attempt {attempt} failed: {error}; waiting {wait:.1f}s")
                time.sleep(wait)
        if xml_bytes is None:
            print(f"batch {index} FAILED after retries")
            continue
        xml_path.write_bytes(xml_bytes)
        records = parse_batch(xml_bytes)
        returned = {record["pmid"] for record in records}
        missing = [pmid for pmid in batch if pmid not in returned]
        with parsed_path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            for pmid in missing:
                handle.write(
                    json.dumps(
                        {
                            "pmid": pmid,
                            "title": "",
                            "abstract": "",
                            "abstract_missing": True,
                            "not_returned": True,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(
            f"batch {index + 1}/{len(batches)}  returned {len(records)}/{len(batch)}"
            f"  missing_abstract {sum(r['abstract_missing'] for r in records)}"
        )
        time.sleep(PAUSE_SECONDS)


def records_in(path: Path):
    """Read one batch file without splitting inside an abstract.

    A few abstracts contain Unicode line separators. str.splitlines() cuts
    those records in half, so each file is decoded as a JSON stream instead.
    """
    text = path.read_text()
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index] in " \t\r\n":
            index += 1
        if index >= length:
            break
        record, index = decoder.raw_decode(text, index)
        yield record


def merge(pmids: list[str]) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    failed = []
    written = 0
    missing_abstract = 0
    with OUT_PATH.open("w") as out:
        for path in sorted(PARSED_DIR.glob("batch_*.jsonl")):
            for record in records_in(path):
                pmid = record["pmid"]
                if pmid in seen:
                    continue
                seen.add(pmid)
                if record.get("not_returned") or not record.get("title"):
                    failed.append(pmid)
                if record.get("abstract_missing"):
                    missing_abstract += 1
                line = json.dumps(record, ensure_ascii=False)
                line = line.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
                out.write(line + "\n")
                written += 1
    FAIL_PATH.write_text("\n".join(failed) + ("\n" if failed else ""))
    not_seen = [pmid for pmid in pmids if pmid not in seen]
    print(
        f"wrote {written} records to {OUT_PATH}\n"
        f"abstract_missing {missing_abstract}\n"
        f"failed_or_empty_title {len(failed)}\n"
        f"pmids_with_no_batch_file {len(not_seen)}"
    )


def main() -> None:
    pmids = load_pmids()
    download(pmids)
    merge(pmids)


if __name__ == "__main__":
    main()
