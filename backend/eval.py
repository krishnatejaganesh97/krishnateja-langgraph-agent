"""Eval script for the Teja LangGraph Agent.

What this teaches: agent evals — test the whole workflow, not one step.
For each of 5 sample tasks we check four things:
  1. Structured output is valid (AgentOutput Pydantic model, >=2 key points).
  2. The answer contains an expected keyword (on-task).
  3. Citations are returned AND used in the answer text (grounded).
  4. The run log shows the full graph: plan, research, draft, critique,
     finalize — and revisions never exceed 2 (the self-critique loop
     must terminate).

Run from the project root:
  python backend/eval.py
Exit code 0 = all 5 passed, 1 = at least one failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent import run_agent  # noqa: E402

# (task, expected keyword in the final answer, lowercased)
TESTS = [
    ("Summarize Teja's AI experience for an AI Engineer role", "langchain"),
    ("Compare RAG vs fine-tuning for a doc chatbot", "rag"),
    ("Where does Teja study for his PhD?", "cumberlands"),
    ("What did Teja build at Elite Tech Info?", "servicenow"),
    ("What programming languages does Teja use?", "python"),
]

REQUIRED_NODES = {"plan", "research", "draft", "critique", "finalize"}


def main() -> int:
    print(f"Mode: {run_agent.__module__} — running 5 tasks through the full graph\n")
    results = []
    passed = 0
    for i, (task, keyword) in enumerate(TESTS, 1):
        out = run_agent(task)
        final = out["final"]
        text = (final.get("summary", "") + " " + " ".join(final.get("key_points", []))).lower()
        has_keyword = keyword.lower() in text
        cites = final.get("citations", [])
        cites_used = bool(cites) and any(c in text or f"[{c}]" in text for c in cites)
        structured_ok = (
            len(final.get("key_points", [])) >= 2
            and len(final.get("summary", "")) >= 20
            and final.get("confidence") in {"low", "medium", "high"}
            and len(final.get("limitations", "")) >= 10
        )
        nodes_seen = {entry["node"] for entry in out["log"]}
        graph_ok = REQUIRED_NODES <= nodes_seen
        revisions_ok = out["revisions"] <= 2
        ok = has_keyword and cites_used and structured_ok and graph_ok and revisions_ok
        passed += ok
        results.append({
            "task": task, "expected_keyword": keyword,
            "has_keyword": has_keyword, "citations": cites,
            "cites_used_in_text": cites_used, "structured_ok": structured_ok,
            "nodes_seen": sorted(nodes_seen), "graph_ok": graph_ok,
            "revisions": out["revisions"], "revisions_ok": revisions_ok,
            "pass": ok, "final": final,
        })
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] Task {i}: {task}")
        print(f"       keyword '{keyword}': {'yes' if has_keyword else 'NO'} | "
              f"citations used: {'yes' if cites_used else 'NO'} {cites} | "
              f"structured: {'yes' if structured_ok else 'NO'} | "
              f"graph: {'yes' if graph_ok else 'NO'} {sorted(nodes_seen)} | "
              f"revisions: {out['revisions']}")
        print(f"       summary: {final.get('summary', '')[:220]}\n")

    print(f"Result: {passed}/{len(TESTS)} passed")
    out_path = Path(__file__).resolve().parent.parent / "eval_results.json"
    out_path.write_text(json.dumps(
        {"mode": results[0]["final"].get("mode", "mock") if results else "mock",
         "passed": passed, "total": len(TESTS), "results": results}, indent=2))
    print(f"Wrote {out_path}")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
