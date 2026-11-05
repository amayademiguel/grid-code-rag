"""Results of a run after the review by hand: automatic check plus the corrections in manual_review.json.

Usage:  python -m eval.assessment
"""
import json
import sys
from pathlib import Path

from eval.evaluate_answers import QUESTIONS, check

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).parent


def group(q):
    if not q["articles"]:
        return "Out of scope"
    if q["lang"] != "en":
        return "Other languages"
    if q.get("kind") == "reference":
        return "Names a table"
    return "Direct" if "kind" not in q else "Varied wording"


def main():
    review = json.loads((HERE / "manual_review.json").read_text(encoding="utf-8"))
    answers = json.loads((HERE.parent / review["answers_file"]).read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in QUESTIONS}
    table = {}
    for r in answers:
        q = questions[r["id"]]
        result = check(q, r["answer"])
        correct = result["correct"]
        if str(r["id"]) in review["correct_but_rejected"]:
            correct = True
        if str(r["id"]) in review["wrong_but_accepted"]:
            correct = False
        row = table.setdefault(group(q), {"correct": 0, "refused": 0, "wrong": 0})
        # a right refusal (outside the regulation, false premise) counts as correct
        row["correct" if correct else "refused" if result["not_found"] else "wrong"] += 1
    for name, row in table.items():
        print(f"{name:<18} correct {row['correct']:>3}  refused {row['refused']:>2}  wrong or incomplete {row['wrong']:>3}")
    regulation = {k: v for k, v in table.items() if k != "Out of scope"}
    total = {k: sum(row[k] for row in regulation.values()) for k in ("correct", "refused", "wrong")}
    n = sum(total.values())
    print(f"In the regulation  correct {total['correct']}/{n} ({total['correct'] / n:.0%}), refused {total['refused']}, wrong or incomplete {total['wrong']}")
    print(f"Mean {sum(r['seconds'] for r in answers) / len(answers):.0f} s per answer")


if __name__ == "__main__":
    main()
