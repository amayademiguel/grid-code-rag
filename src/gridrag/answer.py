"""Generation: answer a question from the retrieved chunks only, with citations."""
import json
import re

import ollama

from .config import LLM_CONTEXT, LLM_MAX_ANSWER, LLM_MAX_REPORT, LLM_MODEL, LLM_REPEAT_PENALTY, TOP_K
from .retrieve import Retriever, cite
from .rfg_types import project_facts
from .tables import table_answer

# Questions on procedures can mention MW and kV, but they are not answered by the technical articles
PROCEDURE_RE = re.compile(r"notification|derogation|test|simulat|complian", re.I)
# Everyday wording and the term used in the regulation, for the search and for the LLM
GLOSSARY = {r"short circuit|voltage dip|stay connected": "fault-ride-through",
            r"over-?frequency": "LFSM-O", r"under-?frequency": "LFSM-U", r"batter(y|ies)": "storage devices"}

SYSTEM_PROMPT = """You are an assistant for grid connection engineers. You answer questions about grid code \
requirements using ONLY the regulation excerpts provided in the context, not outside knowledge.

Rules:
- Start with the answer itself, with no introduction and no steps. Always state the rule or the value, \
not only Yes or No. If the TSO or system operator can require something, say so instead of answering No.
- Give the numerical values, ranges and limits exactly as written in the excerpts. Saying only that \
the TSO specifies a value is not enough if the excerpts give its range.
- Cite the source of every requirement as [Art. N], with the article number in the header of the excerpt it comes from.
- Requirements differ by module type (A, B, C, D), technology (synchronous or power park module) and \
synchronous area. Use only the excerpts that match the question. If the question does not say which case, \
give the values of each case in the excerpts. Other details in the question do not matter.
- Table rows are written as "Table N (what it applies to) | area | columns: values". Take values only from the row \
of the area asked about, and from the table that applies to the case.
- If project facts are given, they are correct: follow them.
- If a requirement depends on conditions (synchronous area, decision of the TSO), state them.
- If the excerpts do not address the question, say that they do not address it and cite nothing."""


def build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"### Excerpt {i} ({cite(c)})\n{c['text']}")
    return "\n\n".join(blocks)


def generate(question, chunks, model=LLM_MODEL, facts=""):
    known = f"Project facts, computed by code from the question: {facts}\n\n" if facts else ""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{build_context(chunks)}\n\n{known}Question: {question}\n\n"
                                    "Answer from the excerpts above and cite [Art. N] for each requirement."},
    ]
    response = ollama.chat(
        model=model,
        messages=messages,
        # deterministic, enough room for the context; the penalty discourages repetition loops
        options={"temperature": 0, "num_ctx": LLM_CONTEXT, "num_predict": LLM_MAX_ANSWER, "repeat_penalty": LLM_REPEAT_PENALTY},
    )
    answer = response["message"]["content"].replace(" [Art. N]", "")  # the placeholder of the prompt, copied literally
    return answer + "\n[Answer cut: length limit reached]" if response["done_reason"] == "length" else answer


def check_citations(answer, chunks):
    """Flag cited articles that are not among the excerpts: the LLM can invent a citation."""
    retrieved = {c["article"] for c in chunks}
    invented = sorted({int(n) for n in re.findall(r"\bArt(?:icle)?\.?\s*(\d+)", answer)} - retrieved)
    if invented:
        answer += "\n[Check: not among the retrieved excerpts: " + ", ".join(f"Art. {n}" for n in invented) + "]"
    return answer


def ask(question, retriever=None, k=TOP_K, model=LLM_MODEL, **filters):
    retriever = retriever or Retriever()
    values = table_answer(question, retriever.chunks)
    if values:  # a direct question on table values: the code reads the rows, without the LLM
        return {"question": question, "answer": values["text"], "sources": [cite(c) for c in values["chunks"]], "llm": False}
    facts = project_facts(question)
    query = question
    known = []
    if facts:
        query = f"{question} {facts['search']}".strip()
        known.append(facts["text"])
        if facts["articles"] and not PROCEDURE_RE.search(question):
            filters.setdefault("articles", facts["articles"])
    terms = sorted({term for pattern, term in GLOSSARY.items() if re.search(pattern, question, re.I)})
    if terms:
        query += " " + " ".join(terms)
        known.append("Terms used in the regulation: " + ", ".join(terms) + ".")
    chunks = retriever.search(query, k=k, **filters)
    if facts and facts.get("final"):  # an exact rule: the code answers, without the LLM
        return {"question": question, "answer": facts["text"], "sources": [cite(c) for c in chunks], "llm": False}
    answer = generate(question, chunks, model, " ".join(known))
    return {"question": question, "answer": check_citations(answer, chunks), "sources": [cite(c) for c in chunks]}


REQUIREMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement": {"type": "string"},
                    "value": {"type": "string"},
                    "conditions": {"type": "string"},
                    "article": {"type": "string"},
                },
                "required": ["requirement", "value", "conditions", "article"],
            },
        }
    },
    "required": ["requirements"],
}


def extract_requirements(topic, chunks, model=LLM_MODEL):
    """Structured output: return the requirements on a topic as a JSON list."""
    prompt = (
        f"Context:\n\n{build_context(chunks)}\n\n"
        f"Extract every requirement about: {topic}. For each one give the requirement, its value "
        "(numbers exactly as written), the conditions under which it applies, and the article. "
        "Only use the context. Return an empty list if there is none."
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        format=REQUIREMENT_SCHEMA,  # the model is constrained to this JSON schema
        options={"temperature": 0, "num_ctx": LLM_CONTEXT, "num_predict": LLM_MAX_REPORT},
    )
    if response["done_reason"] == "length":  # a cut JSON cannot be read
        print(f"  [Cut: length limit reached, no requirements kept for: {topic}]")
        return []
    return json.loads(response["message"]["content"])["requirements"]
