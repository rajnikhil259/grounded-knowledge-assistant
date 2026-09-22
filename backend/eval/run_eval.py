"""Evaluation harness. Run from the backend folder:

    python -m eval.run_eval --collection default
    python -m eval.run_eval --collection small --tag chunk300
    python -m eval.run_eval --collection default --no-validation --tag no_validation

Metrics
  retrieval hit@k     : for answerable questions, does ANY of the top-k chunks contain an expected keyword?
                        (measures the retriever alone - no LLM involved)
  answer pass rate    : answerable questions that got status 'answered' AND the answer contains a keyword
  correct refusal rate: unanswerable questions that were refused (not_found / unverified / blocked)
  hallucination rate  : unanswerable questions that got a confident 'answered' status (lower is better)
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean

from app import graph, retrieval
from app.config import TOP_K


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", default="default")
    ap.add_argument("--file", default=str(Path(__file__).with_name("eval_set.json")))
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--no-validation", action="store_true")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between questions (free-tier rate limits)")
    args = ap.parse_args()

    items = json.loads(Path(args.file).read_text(encoding="utf-8"))
    rows = []
    for n, item in enumerate(items, 1):
        q = item["question"]
        kws = [k.lower() for k in item.get("expected_keywords", [])]

        hit = None
        if item["answerable"]:
            chunks = retrieval.search(q, args.collection, args.top_k)
            hit = any(k in c["content"].lower() for c in chunks for k in kws)

        res = graph.run(q, args.collection, args.top_k, use_validation=not args.no_validation, log=False)

        if item["answerable"]:
            passed = res["status"] == "answered" and any(k in res["answer"].lower() for k in kws)
        else:
            passed = res["status"] != "answered"        # a refusal is the correct behaviour
        rows.append({**item, "hit": hit, "status": res["status"], "passed": passed, "answer": res["answer"],
                     "latency_ms": res["latency_ms"], "tokens": res["input_tokens"] + res["output_tokens"],
                     "cost": res["est_cost_usd"], "validation_passed": res["validation_passed"]})
        print(f"[{n:>2}/{len(items)}] {'PASS' if passed else 'FAIL'}  {res['status']:<10} {q[:70]}")
        time.sleep(args.delay)

    ans = [r for r in rows if r["answerable"]]
    unans = [r for r in rows if not r["answerable"]]
    summary = {
        "collection": args.collection, "tag": args.tag, "top_k": args.top_k,
        "validation": not args.no_validation, "questions": len(rows),
        "retrieval_hit_at_k": round(mean(r["hit"] for r in ans), 3) if ans else None,
        "answer_pass_rate": round(mean(r["passed"] for r in ans), 3) if ans else None,
        "correct_refusal_rate": round(mean(r["passed"] for r in unans), 3) if unans else None,
        "hallucination_rate": round(mean(r["status"] == "answered" for r in unans), 3) if unans else None,
        "avg_latency_ms": round(mean(r["latency_ms"] for r in rows)),
        "avg_tokens": round(mean(r["tokens"] for r in rows)),
        "total_est_cost_usd": round(sum(r["cost"] for r in rows), 5),
    }
    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:>22}: {v}")

    out = Path(__file__).parent / "results" / f"{args.collection}_{args.tag}.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved details to {out}")


if __name__ == "__main__":
    main()
