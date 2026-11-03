"""Parse the regulation PDFs and split them into article-level chunks with metadata.

Why article-level chunking: in a regulation the article is both the unit of meaning
and the unit that gets cited. Long articles are split by numbered paragraph, and every
piece keeps the article number and title so it still makes sense on its own.
"""
import json
import re
import urllib.request

import fitz  # PyMuPDF

from .config import CHUNKS_FILE, MAX_CHUNK_CHARS, PROCESSED_DIR, RAW_DIR
from .documents import DOCUMENTS

# Running header/footer of the Official Journal, repeated on every page
HEADER_RE = re.compile(r"^\s*(\d{1,2}\.\d{1,2}\.\d{4}|L \d+/\d+|Official Journal of the European Union|EN)\s*$")
ARTICLE_RE = re.compile(r"^Article (\d+)\s*$")
PARAGRAPH_RE = re.compile(r"^\d+\.\s*$|^\d+\.\s+\S")


def download(doc):
    path = RAW_DIR / doc["file"]
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {doc['short']} ...")
        urllib.request.urlretrieve(doc["url"], path)
    return path


def read_lines(pdf_path):
    """Return (page_number, line) pairs without the running headers."""
    lines = []
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            for line in page.get_text().splitlines():
                if line.strip() and not HEADER_RE.match(line):
                    lines.append((page_number, line.rstrip()))
    return lines


def join_table_cells(lines):
    """PDF tables come out one cell per line ("Uret:" then "0"). Join a short label
    ending in ':' with the value on the next line, so "Uret: 0" stays together."""
    joined = []
    for page, line in lines:
        prev = joined[-1][1].strip() if joined else ""
        if prev.endswith(":") and len(prev) <= 12:
            joined[-1] = (joined[-1][0], f"{prev} {line.strip()}")
        else:
            joined.append((page, line))
    return joined


def split_articles(lines):
    """Group lines into sections: the preamble, then one section per article."""
    sections = [{"article": None, "title": "Preamble and recitals", "lines": []}]
    i = 0
    while i < len(lines):
        page, line = lines[i]
        match = ARTICLE_RE.match(line.strip())
        if match and i + 1 < len(lines):
            title = lines[i + 1][1].strip()
            sections.append({"article": int(match.group(1)), "title": title, "lines": []})
            i += 2
            continue
        sections[-1]["lines"].append((page, line))
        i += 1
    return sections


def split_paragraphs(section_lines):
    """Split an article into its numbered paragraphs (1., 2., ...)."""
    paragraphs, current = [], []
    for page, line in section_lines:
        if PARAGRAPH_RE.match(line.strip()) and current:
            paragraphs.append(current)
            current = []
        current.append((page, line))
    if current:
        paragraphs.append(current)
    return paragraphs


def split_oversized(paragraph, max_chars):
    """Fallback for paragraphs longer than max_chars (e.g. the list of definitions):
    cut at line boundaries, preferring lines that start a new point such as (a) or (12)."""
    blocks, current, size = [], [], 0
    for page, line in paragraph:
        starts_point = re.match(r"^\s*\(\w{1,3}\)", line)
        if current and size > max_chars * 0.6 and (starts_point or size > max_chars):
            blocks.append(current)
            current, size = [], 0
        current.append((page, line))
        size += len(line) + 1
    if current:
        blocks.append(current)
    return blocks


def pack(paragraphs, max_chars):
    """Merge consecutive paragraphs into pieces of at most max_chars."""
    pieces, current, size = [], [], 0
    expanded = []
    for paragraph in paragraphs:
        if sum(len(line) + 1 for _, line in paragraph) > max_chars:
            expanded.extend(split_oversized(paragraph, max_chars))
        else:
            expanded.append(paragraph)
    for paragraph in expanded:
        length = sum(len(line) + 1 for _, line in paragraph)
        if current and size + length > max_chars:
            pieces.append(current)
            current, size = [], 0
        current.extend(paragraph)
        size += length
    if current:
        pieces.append(current)
    return pieces


def build_chunks(doc):
    lines = join_table_cells(read_lines(download(doc)))
    chunks = []
    for section in split_articles(lines):
        if not section["lines"]:
            continue
        pieces = pack(split_paragraphs(section["lines"]), MAX_CHUNK_CHARS)
        for part, piece in enumerate(pieces, start=1):
            text = "\n".join(line for _, line in piece)
            art = section["article"]
            label = f"Article {art} - {section['title']}" if art else section["title"]
            chunks.append({
                "chunk_id": f"{doc['doc_id']}:{art or 0}:{part}",
                "doc_id": doc["doc_id"],
                "document": doc["short"],
                "jurisdiction": doc["jurisdiction"],
                "language": doc["language"],
                "version": doc["version"],
                "valid_from": doc["valid_from"],
                "valid_to": doc["valid_to"],
                "article": art,
                "article_title": section["title"],
                "part": part,
                "parts": len(pieces),
                "page_start": piece[0][0],
                "page_end": piece[-1][0],
                # Context header prepended to the text: the chunk carries its own location
                "text": f"[{doc['short']} | {label}]\n{text}",
            })
    return chunks


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    all_chunks = []
    for doc in DOCUMENTS:
        chunks = build_chunks(doc)
        articles = {c["article"] for c in chunks if c["article"]}
        print(f"{doc['short']}: {len(articles)} articles -> {len(chunks)} chunks")
        all_chunks.extend(chunks)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"Saved {len(all_chunks)} chunks to {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
