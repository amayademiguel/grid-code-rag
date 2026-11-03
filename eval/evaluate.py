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

QUESTIONS = json.loads((Path(__file__).parent / "questions.json").read_text(encoding="utf-8"))
KS = (1, 4, 8)


def evaluate(retriever, mode, max_k=max(KS)):
    hits = {k: 0 for k in KS}
    reciprocal_ranks = []
    misses = []
    for q in QUESTIONS:
        results = retriever.search(q["question"], k=max_k, mode=mode)
        ranks = [i for i, c in enumerate(results, start=1) if c["article"] in q["articles"]]
        first = ranks[0] if ranks else None
        for k in KS:
            hits[k] += bool(first and first <= k)
        reciprocal_ranks.append(1 / first if first else 0.0)
        if not first or first > 4:
            misses.append((q["id"], [c["article"] for c in results[:4]]))
    n = len(QUESTIONS)
    return {f"recall@{k}": hits[k] / n for k in KS} | {"MRR": sum(reciprocal_ranks) / n, "misses": misses}


def main():
    retriever = Retriever()
    print(f"{len(QUESTIONS)} questions\n")
    print(f"{'mode':<10}" + "".join(f"{f'recall@{k}':>11}" for k in KS) + f"{'MRR':>8}")
    report = {}
    for mode in ("semantic", "keyword", "hybrid"):
        r = evaluate(retriever, mode)
        report[mode] = r
        print(f"{mode:<10}" + "".join(f"{r[f'recall@{k}']:>11.2f}" for k in KS) + f"{r['MRR']:>8.2f}")
    print("\nQuestions where the correct article is not in the top 4 (id, articles retrieved):")
    for mode, r in report.items():
        print(f"  {mode}: {r['misses'] or 'none'}")
    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
