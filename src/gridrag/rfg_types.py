"""Deterministic tool: classify a power-generating module (RfG Article 5, Table 1)
and list the articles that apply to it.

This is deliberately plain code, not the LLM: thresholds are exact rules, and a
classification error would send the whole search to the wrong articles.
"""

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
