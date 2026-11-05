"""Deterministic tool: answer direct questions on table values by reading the table rows, without the LLM.

Same idea as rfg_types.py: a table lookup is an exact task, so the code does it. A small LLM can pick the
row of another area; this code cannot. Anything else (comparisons, yes/no, a specific value such as
"1.12 pu", a question in another language) returns None and goes to the LLM.
"""
import re

from .rfg_types import THRESHOLDS_MW, classify_module

INTENT_RE = re.compile(r"(^|[.:?!]\s+)(what|which|list|give|show|how much reactive)\b", re.I)  # also after the project data
LLM_RE = re.compile(r"\b(lowest|highest|differ|compare|why|who|when|higher|lower)\b|\d\s*(pu|Hz)\b", re.I)
LIMITED_MODE_RE = re.compile(r"LFSM|limited frequency|over-?frequency|under-?frequency", re.I)
NAMED_TABLE_RE = re.compile(r"\bTable (\d+(?:\.\d+)?)", re.I)
FRT_VALUES_RE = re.compile(r"\bUret\b|\btclear\b|retained voltage|fault.?ride.?through parameters", re.I)
SYNCHRONOUS_RE = re.compile(r"(?<!non-)synchronous (?!area)", re.I)
PPM_RE = re.compile(r"power park|\bPPM\b|wind|solar|\bPV\b|photovoltaic|inverter", re.I)
TABLE_3_1_RE = re.compile(r"^[Ut]\w+:", re.M)
RANGE_RE = re.compile(r"(\d+,\d+) (Hz|pu)-(\d+,\d+) (?:Hz|pu)(?: \(\*+\))? = ([^;]+)")  # "1,118 pu-1,15 pu = 20 minutes"


def _voltage(question):
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*kV\b", question)
    return float(match[1].replace(",", ".")) if match else None


def _project_type(question, voltage):
    """Module type named in the question, or computed from its MW and kV."""
    named = re.search(r"\btype ([A-D])\b", question)
    mw = re.search(r"(\d+(?:[.,]\d+)?)\s*MW\b", question)
    if named:
        return named[1]
    return classify_module(float(mw[1].replace(",", ".")), voltage)[0] if mw and voltage else None


def _technologies(question):
    if SYNCHRONOUS_RE.search(question):
        return ["synchronous"]
    return ["power park module"] if PPM_RE.search(question) else ["synchronous", "power park module"]


def select_tables(question):
    """The tables that answer the question, or [] if it is not a direct question on table values."""
    is_keywords = len(question.split()) <= 4 and "?" not in question  # "FSM droop deadband"
    if not (INTENT_RE.search(question) or is_keywords) or LLM_RE.search(question):
        return []
    named = NAMED_TABLE_RE.findall(question)
    if named:
        return [f"Table {n}" for n in named]
    if LIMITED_MODE_RE.search(question):  # the LFSM values are in the text of Articles 13 and 15, not in Tables 4 and 5
        return []

    voltage = _voltage(question)
    module_type = _project_type(question, voltage)
    technologies = _technologies(question)
    offshore = re.search(r"offshore", question, re.I)

    if re.search(r"capacity thresholds?|thresholds? for type|type [BCD] thresholds?|capacity limits?", question, re.I):
        return ["Table 1"]
    if re.search(r"frequency ranges?", question, re.I):
        return ["Table 2"]
    if re.search(r"droop|deadband|insensitivity", question, re.I):
        return ["Table 4"]
    if re.search(r"initial delay|activation time", question, re.I):
        return ["Table 5"]
    if re.search(r"voltage ranges?", question, re.I):
        if offshore:
            return ["Table 10"]
        if module_type in ("A", "B", "C") or (voltage is not None and voltage < 110):
            return []  # the voltage tables are for type D at 110 kV or above
        if voltage is None:
            return ["Table 6.1", "Table 6.2"]
        return ["Table 6.1" if voltage < 300 else "Table 6.2"]
    if FRT_VALUES_RE.search(question):
        if offshore or module_type == "A":
            return []
        low = voltage is not None and voltage < 110 or module_type in ("B", "C")
        high = voltage is not None and voltage >= 110
        groups = ["3"] if low else ["7"] if high else ["3", "7"]
        return [f"Table {g}.{1 if tech == 'synchronous' else 2}" for g in groups for tech in technologies]
    if re.search(r"reactive power capability|how much reactive|q/pmax", question, re.I):
        if module_type in ("A", "B"):
            return []  # Tables 8, 9 and 11 are for type C and D
        if offshore:
            return ["Table 11"]
        return ["Table 8" if tech == "synchronous" else "Table 9" for tech in technologies]
    return []


def _rows(chunk, name):
    """The rows of a table in a chunk, without its caption and header."""
    text = chunk["text"]
    if name == "Table 3.1" and name in text:  # it comes out of the PDF as plain text, one cell per line
        block = text.split(name, 1)[1].split("Table 3.2", 1)[0].replace("\xad\n", "")
        cells = [line for line in block.split("\n") if TABLE_3_1_RE.match(line)]
        return [" | ".join(cells[i:i + 2]) for i in range(0, len(cells), 2)]
    rows = []
    for line in text.split("\n"):
        if re.match(rf"{re.escape(name)}( \(.*?\))? \| ", line):
            row = line.split(" | ", 1)[1]
            if re.search(r"\d", row):  # not a header such as "Parameters | Ranges"
                rows.append(row)
    return rows


def duration_answer(question, chunks):
    """"How long at 1.12 pu / 48 Hz in <area>?": find the range that contains the value. Comparing numbers
    is an exact task, and an LLM can read the wrong range. Returns None if the question gives no such data."""
    value = re.search(r"(\d+(?:[.,]\d+)?)\s*(pu|Hz)\b", question)
    area = next((a for a in THRESHOLDS_MW if a.split()[0].lower() in question.lower()), None)
    if not (value and area and re.search(r"how long", question, re.I)) or re.search(r"offshore", question, re.I):
        return None
    voltage = _voltage(question)
    if value[2] == "Hz":
        name = "Table 2"
    elif voltage is not None and voltage >= 110:
        name = "Table 6.1" if voltage < 300 else "Table 6.2"
    else:
        return None
    number = float(value[1].replace(",", "."))
    for chunk in chunks:
        rows = [row for row in _rows(chunk, name) if row.startswith(area)]
        if rows:
            found = [f"- {low} {unit}-{high} {unit}: {time.strip()}" for low, unit, high, time in RANGE_RE.findall(rows[0])
                     if float(low.replace(",", ".")) <= number <= float(high.replace(",", "."))]
            caption = re.search(rf"^{re.escape(name)}( \(.*?\))?", chunk["text"], re.M)[0]
            body = "\n".join(found) if found else f"- No range of the table contains {value[1]} {value[2]}: {rows[0]}"
            return {"text": f"Value read from the tables of the regulation by code, not by the LLM.\n\n{caption} [Art. {chunk['article']}], {area}:\n{body}",
                    "chunks": [chunk]}
    return None


def table_answer(question, chunks):
    """Answer from the table rows, with their article. Returns None if the code should not answer."""
    duration = duration_answer(question, chunks)
    if duration:
        return duration
    tables = select_tables(question)
    if not tables:
        return None
    area = next((a for a in THRESHOLDS_MW if a.split()[0].lower() in question.lower()), None)
    assumed = ""
    if not area and re.search(r"\d\s*MW\b", question) and re.search(r"\d\s*kV\b", question):
        area, assumed = "Continental Europe", " (Continental Europe assumed)"
    blocks, used = [], []
    for name in tables:
        for chunk in chunks:
            rows = _rows(chunk, name)
            if not rows:
                continue
            has_areas = any(row.startswith(tuple(THRESHOLDS_MW)) for row in rows)
            if area and has_areas:
                rows = [row for row in rows if row.startswith(area)]
            caption = re.search(rf"^{re.escape(name)}( \(.*?\))?", chunk["text"], re.M)[0]
            notes = re.findall(r"^\(\*+\).*$", chunk["text"], re.M) if any("(*" in row for row in rows) else []
            blocks.append(f"{caption} [Art. {chunk['article']}]{assumed if has_areas else ''}:\n" + "\n".join([f"- {row}" for row in rows] + notes))
            used.append(chunk)
    if not blocks:
        return None
    return {"text": "Values read from the tables of the regulation by code, not by the LLM.\n\n" + "\n\n".join(blocks), "chunks": used}


def _report_item(chunks, category, name, requirement, area, conditions):
    """One requirement of the report, read from a table: all the values of its row, or of the whole table."""
    for chunk in chunks:
        rows = _rows(chunk, name)
        if rows:
            if area:
                rows = [row.split(" | ", 1)[1] for row in rows if row.startswith(area)]
            value = "; ".join(row.split(": ", 1)[1] if "=" in row else row.replace(" | ", "; ") for row in rows)
            notes = " ".join(re.findall(r"^\(\*+\).*$", chunk["text"], re.M)) if "(*" in value else ""
            return {"category": category, "requirement": requirement, "value": value,
                    "conditions": " ".join(part for part in (conditions, notes) if part),
                    "article": f"Art. {chunk['article']}, {name}"}
    return None


def table_requirements(module_type, technology, connection_kv, area, chunks, offshore=False):
    """Requirements of a project that are in tables, read by code: the LLM mixed rows of other cases."""
    synchronous = technology == "synchronous"
    items = [("Frequency ranges", "Table 2", "Frequency ranges and minimum time periods for operation", area, area)]
    if connection_kv >= 110 or offshore:
        name = "Table 10" if offshore else "Table 6.1" if connection_kv < 300 else "Table 6.2"
        items.append(("Voltage ranges", name, "Voltage ranges and minimum time periods for operation", area, area))
    if module_type in ("C", "D"):
        name = "Table 11" if offshore else "Table 8" if synchronous else "Table 9"
        items.append(("Reactive power capability", name, "Reactive power capability at maximum capacity", area, area))
    if module_type != "A":
        name = f"Table {7 if connection_kv >= 110 else 3}.{1 if synchronous else 2}"
        side = "at or above" if connection_kv >= 110 else "below"
        items.append(("Fault-ride-through", name, "Parameters of the voltage-against-time profile",
                      None, f"Connected {side} 110 kV. The TSO specifies the profile within these ranges."))
    found = [_report_item(chunks, *item) for item in items]
    return [item for item in found if item]
