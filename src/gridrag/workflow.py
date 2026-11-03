"""Requirements report for a project: an agentic workflow.

Given a project (capacity, connection voltage, technology, synchronous area), it:
  1. classifies the module with a deterministic tool (Article 5);
  2. derives the applicable articles;
  3. for each requirement category, retrieves only from those articles;
  4. extracts the requirements as structured JSON with the LLM;
  5. returns a table with the source of every requirement.

The plan is fixed in code rather than left to the LLM. With a small local model a
code-controlled workflow is more reliable than free planning; a larger model could
take over the planning (choosing categories, following cross-references).
"""
from .answer import extract_requirements
from .retrieve import Retriever, cite
from .rfg_types import applicable_articles, classify_module

CATEGORIES = {
    "Frequency ranges": "frequency ranges and minimum time periods for operation without disconnecting; rate of change of frequency withstand",
    "Frequency response (LFSM-O / LFSM-U / FSM)": "limited frequency sensitive mode overfrequency LFSM-O, underfrequency LFSM-U, frequency sensitive mode, droop and threshold",
    "Voltage ranges": "voltage ranges and time periods for operation at the connection point",
    "Reactive power capability": "reactive power capability at maximum capacity and below maximum capacity, U-Q/Pmax profile, voltage control",
    "Fault-ride-through": "fault-ride-through capability, voltage-against-time profile, Uret, tclear",
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

    rows = []
    for category, query in CATEGORIES.items():
        chunks = retriever.search(query, k=k, articles=articles)
        print(f"  {category}: {', '.join(cite(c) for c in chunks)}")
        for item in extract_requirements(query, chunks):
            rows.append({"category": category, **item})
    return {"type": module_type, "reason": reason, "articles": articles, "requirements": rows}
