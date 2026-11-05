"""Check that no table value is lost or altered: compare the numbers of each table read
through the table parser (ingest.table_rows) with the words on the same area of the page,
which is an independent extraction. Lone 1 and 2 are skipped (subscripts such as t2, s1).

Usage:  python -m eval.check_tables
"""
import re
import sys
from collections import Counter

import fitz

from src.gridrag.documents import DOCUMENTS
from src.gridrag.ingest import RAW_DIR, table_rows

sys.stdout.reconfigure(encoding="utf-8")
NUMBER_RE = re.compile(r"(?<![\d,.])\d+(?:,\d+)?")


def numbers(text):
    return Counter(n for n in NUMBER_RE.findall(text) if n not in ("1", "2"))


def main():
    tables = values = errors = 0
    for doc in DOCUMENTS:
        with fitz.open(RAW_DIR / doc["file"]) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                for table in page.find_tables().tables:
                    if re.match(r"\d", table.extract()[0][0] or ""):
                        print(f"p. {page_number}: header not detected, kept as plain text (check by hand)")
                        continue
                    parsed = numbers("\n".join(table_rows(table, None)))
                    raw = numbers(page.get_text(clip=fitz.Rect(table.bbox)))
                    tables += 1
                    values += sum(raw.values())
                    if parsed != raw:
                        errors += 1
                        print(f"p. {page_number}: DIFFERENT  missing {raw - parsed}  extra {parsed - raw}")
    print(f"{tables} tables, {values} numeric values compared, {errors} tables with differences")


if __name__ == "__main__":
    main()
