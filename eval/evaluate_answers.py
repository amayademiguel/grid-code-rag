"""Answer evaluation: run every test question through the full pipeline and check the answer.

For each question the test set gives the key values a correct answer must contain
("expect", alternatives separated by |). An answer is checked for:
  - values: every expected value is present (numbers compared without spaces, "," = ".")
  - citation: it cites one of the correct articles ([Art. N])
  - not found: it opens by saying that the excerpts do not address the question
  - forbid: statements known to be wrong for this question (a value of another area, a wrong No)
Questions outside the regulation (no article) are correct only if the answer is "not found".
The check is automatic and strict on values, so the answers still need reading one by one.

Usage:  python -m eval.evaluate_answers [--model qwen2.5:7b] [--ids 8 21 44]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

from src.gridrag.answer import ask
from src.gridrag.config import LLM_MODEL
from src.gridrag.retrieve import Retriever

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent
QUESTIONS = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
REFUSAL_RE = re.compile(r"notfound|noanswer|notaddress|(do|does)not(contain|mention|provide|specify|state)")


def normalise(text):
    text = text.lower().replace("–", "-").replace("—", "-").replace(",", ".")
    return re.sub(r"\s+", "", text)


def check(q, answer):
    norm = normalise(answer)
    not_found = bool(REFUSAL_RE.search(norm[:150]))  # only at the start: a real answer may add a caveat later
    if not q["articles"]:
        return {"not_found": not_found, "values": None, "cited": None, "correct": not_found}
    values = all(any(alt in norm for alt in group.split("|")) for group in q["expect"])
    cited = any(re.search(rf"\bart(?:icle)?\.?\s*{a}\b", answer, re.I) for a in q["articles"])
    wrong = any(bad in norm for bad in q.get("forbid", []))
    correct = values and not wrong and (not not_found or q.get("not_found_ok", False))
    return {"not_found": not_found, "values": values, "cited": cited, "wrong": wrong, "correct": correct}


def save(results, name):
    out = HERE / f"answers_{name}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=LLM_MODEL)
    parser.add_argument("--ids", type=int, nargs="*", help="only these question ids")
    parser.add_argument("--questions", default="questions.json", help="question file in eval/ (questions_holdout.json)")
    args = parser.parse_args()

    source = json.loads((HERE / args.questions).read_text(encoding="utf-8"))
    questions = [q for q in source if not args.ids or q["id"] in args.ids]
    retriever = Retriever()
    name = args.model.replace(":", "_") + ("_subset" if args.ids else "") + ("_holdout" if "holdout" in args.questions else "")
    results = []
    for q in questions:
        start = time.time()
        r = ask(q["question"], retriever=retriever, model=args.model)
        result = {"id": q["id"], "kind": q.get("kind", "direct"), "question": q["question"], "answer": r["answer"], "sources": r["sources"],
                  "seconds": round(time.time() - start), **check(q, r["answer"])}
        results.append(result)
        save(results, name)  # after each answer, so a stopped run keeps what it has done
        flags = "OK " if result["correct"] else "BAD"
        print(f"[{flags}] {q['id']:>2} ({result['seconds']} s) values={result['values']} "
              f"cited={result['cited']} not_found={result['not_found']}", flush=True)

    in_scope = [r for r in results if r["values"] is not None]
    out_scope = [r for r in results if r["values"] is None]
    print(f"\nModel {args.model}, {len(results)} questions, "
          f"mean {sum(r['seconds'] for r in results) / len(results):.0f} s per answer")
    if in_scope:
        n = len(in_scope)
        print(f"  in the regulation ({n}): correct values {sum(r['correct'] for r in in_scope) / n:.2f}, "
              f"cites correct article {sum(r['cited'] for r in in_scope) / n:.2f}, "
              f"wrong 'not found' {sum(r['not_found'] for r in in_scope) / n:.2f}")
    if out_scope:
        print(f"  outside the regulation ({len(out_scope)}): correctly 'not found' "
              f"{sum(r['correct'] for r in out_scope)}/{len(out_scope)}")
    for kind in sorted({r["kind"] for r in in_scope}):
        group = [r for r in in_scope if r["kind"] == kind]
        print(f"    {kind}: {sum(r['correct'] for r in group)}/{len(group)}")
    print("  failed ids:", [r["id"] for r in results if not r["correct"]])

    print(f"\nSaved {save(results, name)}")


if __name__ == "__main__":
    main()
