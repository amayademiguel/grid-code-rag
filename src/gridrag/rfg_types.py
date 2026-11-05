"""Deterministic tool: classify a power-generating module (RfG Article 5, Table 1),
list the articles that apply to it and find the project data given in a question.

This is deliberately plain code, not the LLM: thresholds are exact rules, and a
classification error would send the whole search to the wrong articles.
"""

import re

# Article 5, Table 1: maximum limits (MW) of the type B, C and D thresholds per synchronous area.
# Each TSO may set lower national thresholds.
THRESHOLDS_MW = {
    "Continental Europe": (1.0, 50.0, 75.0),
    "Great Britain": (1.0, 50.0, 75.0),
    "Nordic": (1.5, 10.0, 30.0),
    "Ireland and Northern Ireland": (0.1, 5.0, 10.0),
    "Baltic": (0.5, 10.0, 15.0),
}


def classify_module(max_capacity_mw, connection_kv, synchronous_area="Continental Europe"):
    """Return the module type (A-D) using the Table 1 limits, with the reason."""
    if connection_kv >= 110:
        return "D", "connection point at 110 kV or above (Art. 5(2)(d))"
    b, c, d = THRESHOLDS_MW[synchronous_area]
    if max_capacity_mw >= d:
        return "D", f"below 110 kV and >= {d} MW in {synchronous_area} (Art. 5(2)(d), Table 1)"
    if max_capacity_mw >= c:
        return "C", f"below 110 kV and >= {c} MW in {synchronous_area} (Art. 5(2)(c), Table 1)"
    if max_capacity_mw >= b:
        return "B", f"below 110 kV and >= {b} MW in {synchronous_area} (Art. 5(2)(b), Table 1)"
    if max_capacity_mw * 1000 >= 0.8:
        return "A", "below 110 kV and >= 0.8 kW (Art. 5(2)(a))"
    return None, "below 0.8 kW: not a significant power-generating module"


def project_facts(question):
    """Project data in the question (MW, kV, area, technology) turned into facts by the rules in this file,
    because a small LLM does not compare numbers reliably ("132 kV is below 110 kV").
    Returns None if the question gives no project data."""
    mw = re.search(r"(\d+(?:[.,]\d+)?)\s*MW\b", question)
    kv = re.search(r"(\d+(?:[.,]\d+)?)\s*kV\b", question)
    named_type = re.search(r"\btype ([A-D])\b", question)
    asks_articles = re.search(r"\barticles\b", question, re.I)
    if not (mw and kv) and not (named_type and asks_articles):
        return None

    technology = None
    if re.search(r"synchronous", question, re.I) and not re.search(r"non-synchronous", question, re.I):
        technology = "synchronous"
    elif re.search(r"power park|\bPPM\b|wind|solar|\bPV\b|photovoltaic|inverter", question, re.I):
        technology = "power park module"

    if not (mw and kv):
        articles = applicable_articles(named_type[1], technology or "power park module")
        return {"text": f"Articles with technical requirements for a type {named_type[1]} "
                        f"{technology or 'power park module'}: {', '.join(map(str, articles))} (Art. 5 and the articles of each type).",
                "search": "", "articles": None, "final": True}

    area = next((a for a in THRESHOLDS_MW if a.split()[0].lower() in question.lower()), None)
    assumed = "" if area else " (assumed)"
    area = area or "Continental Europe"
    capacity, voltage = float(mw[1].replace(",", ".")), float(kv[1].replace(",", "."))
    module_type, reason = classify_module(capacity, voltage, area)
    if module_type is None:
        return None
    articles = applicable_articles(module_type, technology or "power park module")
    if technology is None:
        articles = sorted(set(articles + applicable_articles(module_type, "synchronous")))
    side = "at or above" if voltage >= 110 else "below"
    facts = [f"{capacity:g} MW connected at {voltage:g} kV is {side} 110 kV. Synchronous area: {area}{assumed}.",
             f"It is a type {module_type} module: {reason}.",
             f"Articles with technical requirements: {', '.join(map(str, articles))}."]
    tables = []
    if module_type != "A":
        high = voltage >= 110
        options = {"synchronous": "Table 7.1" if high else "Table 3.1", "power park module": "Table 7.2" if high else "Table 3.2"}
        frt = [options[technology]] if technology else list(options.values())
        tables += frt
        facts.append(f"Fault ride through: {' or '.join(frt)} applies.")
    if voltage >= 110:
        base = "Table 6.1" if voltage < 300 else "Table 6.2"
        tables.append(base)
        facts.append(f"Voltage ranges: {base} applies.")
    return {"text": " ".join(facts), "search": f"type {module_type} {' '.join(tables)}", "articles": articles + [5]}


def applicable_articles(module_type, technology="power park module", offshore=False):
    """Articles with technical requirements that apply. Requirements are cumulative:
    a type D module also meets the type A, B and C requirements (with exceptions)."""
    general = {"A": [13], "B": [13, 14], "C": [13, 14, 15], "D": [13, 14, 15, 16]}[module_type]
    if technology == "synchronous":
        specific = {"A": [], "B": [17], "C": [17, 18], "D": [17, 18, 19]}[module_type]
    else:
        specific = {"A": [], "B": [20], "C": [20, 21], "D": [20, 21, 22]}[module_type]
        if offshore:
            specific += [23, 24, 25, 26, 27, 28]
    return general + specific
