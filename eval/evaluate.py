"""Retrieval evaluation: does the correct article reach the LLM?

For each question the test set gives the article(s) that contain the answer.
We compare semantic, keyword (BM25) and hybrid search with:
  - recall@k: share of questions whose correct article is among the top k chunks
  - MRR: mean reciprocal rank of the first correct chunk (1 = always first)
In this use case recall matters most: a requirement that is never retrieved is
silently missing from the answer.

Usage:  python -m eval.evaluate
"""
import json
from pathlib import Path

from src.gridrag.retrieve import Retriever

# Questions outside the regulation (no gold article) only test the answers, not retrieval
QUESTIONS = [q for q in json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8")) if q["articles"]]
KS = (1, 4, 8)


def evaluate(retriever, mode, questions=QUESTIONS, max_k=max(KS)):
    hits = {k: 0 for k in KS}
    reciprocal_ranks = []
    misses = []
    for q in questions:
        results = retriever.search(q["question"], k=max_k, mode=mode)
        ranks = [i for i, c in enumerate(results, start=1) if c["article"] in q["articles"]]
        first = ranks[0] if ranks else None
        for k in KS:
            hits[k] += bool(first and first <= k)
        reciprocal_ranks.append(1 / first if first else 0.0)
        if not first or first > 4:
            misses.append((q["id"], [c["article"] for c in results[:4]]))
    n = len(questions)
    return {f"recall@{k}": hits[k] / n for k in KS} | {"MRR": sum(reciprocal_ranks) / n, "misses": misses}


def main():
    retriever = Retriever()
    groups = {
        "english": [q for q in QUESTIONS if q["lang"] == "en" and "kind" not in q],
        # the way real users ask: paraphrases, extra details, ambiguity, yes/no, keywords
        "varied wording": [q for q in QUESTIONS if q.get("kind") not in (None, "reference")],
        # questions naming a table: the exact identifier is what matters, so keyword search is needed
        "table references": [q for q in QUESTIONS if q.get("kind") == "reference"],
        # the document is in English: semantic search crosses languages, BM25 only matches words
        "other languages": [q for q in QUESTIONS if q["lang"] != "en"],
    }
    report = {}
    for name, questions in groups.items():
        print(f"\n{name}: {len(questions)} questions")
        print(f"{'mode':<10}" + "".join(f"{f'recall@{k}':>11}" for k in KS) + f"{'MRR':>8}")
        report[name] = {}
        for mode in ("semantic", "keyword", "hybrid"):
            r = evaluate(retriever, mode, questions)
            report[name][mode] = r
            print(f"{mode:<10}" + "".join(f"{r[f'recall@{k}']:>11.2f}" for k in KS) + f"{r['MRR']:>8.2f}")
    print("\nQuestions where the correct article is not in the top 4 (id, articles retrieved):")
    for name, modes in report.items():
        for mode, r in modes.items():
            print(f"  {name} / {mode}: {r['misses'] or 'none'}")
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
