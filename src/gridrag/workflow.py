"""Requirements report for a project: an agentic workflow.

Given a project (capacity, connection voltage, technology, synchronous area), it:
  1. classifies the module with a deterministic tool (Article 5);
  2. derives the applicable articles;
  3. reads the requirements that are in tables with code (tables.py);
  4. for the requirements in the text, retrieves only from those articles, per category;
  5. extracts those requirements as structured JSON with the LLM;
  6. returns a JSON report with the source of every requirement.

The plan is fixed in code rather than left to the LLM. With a small local model a
code-controlled workflow is more reliable than free planning; a larger model could
take over the planning (choosing categories, following cross-references).
"""
from .answer import extract_requirements
from .retrieve import Retriever, cite
from .rfg_types import applicable_articles, classify_module
from .tables import table_requirements

# What the LLM reads from the text. The values in tables (frequency and voltage ranges, reactive power capability
# at maximum capacity, fault-ride-through parameters) are read by code.
CATEGORIES = {
    "Frequency ranges": "rate of change of frequency withstand capability",
    "Frequency response (LFSM-O / LFSM-U / FSM)": "limited frequency sensitive mode overfrequency LFSM-O, underfrequency LFSM-U, frequency sensitive mode, droop and threshold",
    "Reactive power capability": "reactive power capability below maximum capacity, voltage control",
    "Fault current and post-fault recovery": "fast fault current injection and post-fault active power recovery",
    "Protection, control and information exchange": "electrical protection schemes, control schemes, information exchange, instrumentation and simulation models",
}


def project_report(max_capacity_mw, connection_kv, technology="power park module",
                   synchronous_area="Continental Europe", offshore=False, retriever=None, k=4):
    retriever = retriever or Retriever()
    module_type, reason = classify_module(max_capacity_mw, connection_kv, synchronous_area)
    if module_type is None:
        return {"type": None, "reason": reason, "articles": [], "requirements": []}
    articles = applicable_articles(module_type, technology, offshore)

    rows = table_requirements(module_type, technology, connection_kv, synchronous_area, retriever.chunks, offshore)
    for category, query in CATEGORIES.items():
        chunks = retriever.search(query, k=k, articles=articles)
        print(f"  {category}: {', '.join(cite(c) for c in chunks)}")
        for item in extract_requirements(query, chunks):
            rows.append({"category": category, **item})
    return {"type": module_type, "reason": reason, "articles": articles, "requirements": rows}
