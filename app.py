"""Command-line interface of the grid code RAG assistant.

  python app.py build                          # download, chunk and index the regulations
  python app.py search "fault ride through"     # show the retrieved chunks only
  python app.py ask "What are the FRT requirements for type D power park modules?"
  python app.py project --mw 300 --kv 220       # requirements report for a project
"""
import argparse
import json
import sys
import time

from src.gridrag import index, ingest
from src.gridrag.answer import ask
from src.gridrag.config import LLM_MODEL
from src.gridrag.retrieve import Retriever, cite
from src.gridrag.workflow import project_report

sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Local RAG assistant for grid code requirements")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="download, chunk and index the documents")

    p_search = sub.add_parser("search", help="retrieve chunks without calling the LLM")
    p_search.add_argument("query")
    p_search.add_argument("-k", type=int, default=4)
    p_search.add_argument("--mode", choices=["semantic", "keyword", "hybrid"], default="hybrid")

    p_ask = sub.add_parser("ask", help="answer a question with citations")
    p_ask.add_argument("question")
    p_ask.add_argument("-k", type=int, default=4)
    p_ask.add_argument("--model", default=LLM_MODEL)

    p_proj = sub.add_parser("project", help="requirements report for a project")
    p_proj.add_argument("--mw", type=float, required=True, help="maximum capacity in MW")
    p_proj.add_argument("--kv", type=float, required=True, help="connection voltage in kV")
    p_proj.add_argument("--technology", choices=["power park module", "synchronous"], default="power park module")
    p_proj.add_argument("--area", default="Continental Europe")
    p_proj.add_argument("--out", help="save the report as JSON")

    args = parser.parse_args()

    if args.command == "build":
        ingest.main()
        index.main()

    elif args.command == "search":
        for c in Retriever().search(args.query, k=args.k, mode=args.mode):
            print(f"\n--- {cite(c)}  (score {c['score']})\n{c['text'][:600]}")

    elif args.command == "ask":
        start = time.time()
        result = ask(args.question, k=args.k, model=args.model)
        print(f"\n{result['answer']}\n\nSources retrieved:")
        for s in result["sources"]:
            print(f"  - {s}")
        print(f"\n({time.time() - start:.0f} s, model {args.model})")

    elif args.command == "project":
        print(f"Project: {args.mw} MW {args.technology}, {args.kv} kV, {args.area}")
        report = project_report(args.mw, args.kv, args.technology, args.area)
        print(f"\nType {report['type']}: {report['reason']}")
        print(f"Applicable articles: {report['articles']}\n")
        for r in report["requirements"]:
            print(f"[{r['category']}] {r['requirement']}: {r['value']}"
                  f"{' | ' + r['conditions'] if r['conditions'] else ''}  ({r['article']})")
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
