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
TABLE_RE = re.compile(r"^Table \d+(?:\.\d+)?")
SUBSCRIPT_RE = re.compile(r"^(.{1,8}?)(\s*:|\s*-[\d,]+)?\n([a-z]+\d?)$")

# What each table applies to. The regulation says it in the text around the table (Articles 14, 16, 21, 25),
# so a row read alone cannot tell Table 3.2 from Table 7.2, or Table 6.1 from Table 6.2.
TABLE_SCOPE = {
    "Table 1": "capacity thresholds of type B, C and D per synchronous area",
    "Table 3.1": "synchronous modules of type B and C, and type D connected below 110 kV",
    "Table 3.2": "power park modules of type B and C, and type D connected below 110 kV",
    "Table 4": "frequency sensitive mode",
    "Table 5": "frequency sensitive mode, frequency step change",
    "Table 6.1": "type D, voltage base 110 kV to 300 kV",
    "Table 6.2": "type D, voltage base 300 kV to 400 kV",
    "Table 7.1": "synchronous modules of type D connected at or above 110 kV",
    "Table 7.2": "power park modules of type D connected at or above 110 kV",
    "Table 8": "reactive power capability at maximum capacity, synchronous modules of type C and D",
    "Table 9": "reactive power capability at maximum capacity, power park modules of type C and D",
    "Table 10": "AC-connected offshore power park modules",
    "Table 11": "reactive power capability at maximum capacity, AC-connected offshore power park modules",
}
# Table cells whose subscripts come out of the PDF in the wrong place
CELL_FIXES = {
    "Limit for maximum capacity threshold from which a power-generating module is of type ": "Maximum threshold for type ",
    "jΔP j Active power range related to maximum capacity 1 P max": "Active power range related to maximum capacity |ΔP1|/Pmax",
    "jΔP j Active power range related to maximum capacity (frequency response range) 1 P max":
        "Active power range related to maximum capacity (frequency response range) |ΔP1|/Pmax",
    "Δf i f n": "Δfi/fn",
    "Droop s 1": "Droop s1",
    "initial delay t unless 1 justified": "initial delay t1 unless justified",
    "initial delay t 1 unless": "initial delay t1 unless",
    "time t , unless longer activation times are 2 allowed": "time t2, unless longer activation times are allowed",
}


def download(doc):
    path = RAW_DIR / doc["file"]
    if not path.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {doc['short']} ...")
        urllib.request.urlretrieve(doc["url"], path)
    return path


def table_name(line):
    """"Table 6.1" if the line is a table caption or row, else None."""
    match = TABLE_RE.match(line)
    return match[0] if match else None


def clean_cell(cell):
    """Tidy a table cell: re-attach subscripts split by the PDF ("U :\\nret" -> "Uret:") and join lines."""
    cell = (cell or "").replace("\xad\n", "").replace("-\n", "-").replace("Q/P\nmax", "Q/Pmax")
    match = SUBSCRIPT_RE.match(cell)
    if match:
        cell = f"{match[1]}{match[3]}{(match[2] or '').strip()}"
    cell = " ".join(cell.split())
    for wrong, right in CELL_FIXES.items():
        cell = cell.replace(wrong, right)
    return cell


def table_label(name):
    """"Table 6.1 (type D, voltage base 110 kV to 300 kV)", or the bare name if it has no scope."""
    return f"{name} ({TABLE_SCOPE[name]})" if name in TABLE_SCOPE else name


def table_rows(table, name):
    """One self-contained line per row: "Table 6.1 (...) | Continental Europe | Voltage range: 0,85 pu-0,90 pu | ...".
    A merged first cell repeats its value, so no row loses its synchronous area."""
    rows = table.extract()
    header = [clean_cell(cell) for cell in rows[0]]
    label_columns = all(header)  # tables with merged header cells (FRT parameters) keep plain rows
    lines, first_cell = [], ""
    if label_columns and len(header) == 3:  # area, range, time (Tables 2, 6.1, 6.2, ...)
        areas = {}
        for row in rows[1:]:
            first_cell = clean_cell(row[0]) or first_cell
            areas.setdefault(first_cell, []).append(f"{clean_cell(row[1])} = {clean_cell(row[2])}")
        if any(len(values) > 1 for values in areas.values()):
            prefix = f"{table_label(name)} | " if name else ""
            return [f"{prefix}{area} | {header[1]} = {header[2]}: " + "; ".join(values) for area, values in areas.items()]
    for row in rows[1:] if label_columns else rows:
        first_cell = clean_cell(row[0]) or first_cell
        cells = [first_cell] + [clean_cell(cell) for cell in row[1:]]
        if label_columns:
            cells[1:] = [f"{head}: {cell}" for head, cell in zip(header[1:], cells[1:])]
        row_text = " | ".join(([table_label(name)] if name else []) + [cell for cell in cells if cell])
        lines.append(row_text.replace(": | ", ": "))  # "Uret: | 0" -> "Uret: 0"
    return lines


def read_lines(pdf_path):
    """Return (page_number, line) pairs without the running headers. Text inside a table is
    replaced by its rows, because plain extraction gives one cell per line and loses the structure."""
    lines = []
    last_table = None
    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            # A first cell starting with a digit means the header row was not detected: keep the plain text
            tables = [t for t in page.find_tables().tables if not re.match(r"\d", t.extract()[0][0] or "")]
            written = set()
            for block in page.get_text("dict")["blocks"]:
                for pdf_line in block.get("lines", []):
                    x0, y0, x1, y1 = pdf_line["bbox"]
                    centre = fitz.Point((x0 + x1) / 2, (y0 + y1) / 2)
                    index = next((i for i, t in enumerate(tables) if fitz.Rect(t.bbox).contains(centre)), None)
                    if index is not None:
                        if index not in written:
                            written.add(index)
                            # Caption in the last lines of this page; a table without one continues the previous table
                            captions = [TABLE_RE.match(text) for page_no, text in lines[-4:] if page_no == page_number]
                            last_table = next((m[0] for m in reversed(captions) if m), last_table)
                            lines.extend((page_number, row) for row in table_rows(tables[index], last_table))
                        continue
                    text = "".join(span["text"] for span in pdf_line["spans"]).rstrip()
                    if text.strip() and not HEADER_RE.match(text):
                        # A caption alone on its line also gets its scope (Table 3.1 stays as plain text)
                        lines.append((page_number, table_label(text.strip()) if text.strip() in TABLE_SCOPE else text))
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
    cut at line boundaries, preferring lines that start a new point such as (a) or (12).
    A table is never cut: all its rows stay in the same block."""
    blocks, current, size = [], [], 0
    for page, line in paragraph:
        starts_point = re.match(r"^\s*\(\w{1,3}\)", line)
        same_table = current and table_name(line) and table_name(line) == table_name(current[-1][1])
        if current and not same_table and size > max_chars * 0.6 and (starts_point or size > max_chars):
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
