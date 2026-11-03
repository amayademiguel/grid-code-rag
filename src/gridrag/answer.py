"""Generation: answer a question from the retrieved chunks only, with citations."""
import json

import ollama

from .config import LLM_CONTEXT, LLM_MODEL, TOP_K
from .retrieve import Retriever, cite

SYSTEM_PROMPT = """You are an assistant for grid connection engineers. You answer questions about grid code \
requirements using ONLY the regulation excerpts provided in the context.

Rules:
- Use only the information in the context. If the answer is not in the context, say \
"Not found in the provided documents." Do not use outside knowledge.
- Requirements differ by module type (A, B, C, D), technology (synchronous or power park module) and \
synchronous area. The excerpts may cover other types: use only the excerpts that match the question, \
and check the article title before taking a value from it.
- Cite the source of every requirement as [Art. N] using the article numbers shown in the excerpt headers.
- Quote numerical values (frequencies, voltages, times) exactly as written in the context.
- If a requirement depends on conditions (synchronous area, decision of the TSO), state them.
- Be concise and structured."""


def build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, start=1):
        blocks.append(f"### Excerpt {i} ({cite(c)})\n{c['text']}")
    return "\n\n".join(blocks)


def generate(question, chunks, model=LLM_MODEL):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n\n{build_context(chunks)}\n\nQuestion: {question}"},
    ]
    response = ollama.chat(
        model=model,
        messages=messages,
        options={"temperature": 0, "num_ctx": LLM_CONTEXT},  # deterministic, enough room for the context
    )
    return response["message"]["content"]


def ask(question, retriever=None, k=TOP_K, model=LLM_MODEL, **filters):
    retriever = retriever or Retriever()
    chunks = retriever.search(question, k=k, **filters)
    return {"question": question, "answer": generate(question, chunks, model), "sources": [cite(c) for c in chunks]}


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
        options={"temperature": 0, "num_ctx": LLM_CONTEXT},
    )
    return json.loads(response["message"]["content"])["requirements"]
