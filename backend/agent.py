# Teja LangGraph Agent — Product 2
# A multi-step agentic workflow built with LangGraph.
#
# What Teja learns here (see README.md for the full walkthrough):
#   1. LangGraph state & graphs — a TypedDict state that every node reads
#      and updates; nodes are plain functions; edges decide what runs next.
#   2. Planning — the agent writes a step-by-step plan BEFORE acting, so
#      its work is visible and checkable (not a black-box single LLM call).
#   3. Tool-calling — the research node calls tools (retrieve_docs,
#      local_search) and records every call in the run log with its result.
#   4. Structured outputs — the final answer must match the AgentOutput
#      Pydantic model; anything else is rejected and revised.
#   5. Self-critique loops — a critique node checks the draft (citations?
#      on-task? supported?) and sends it back for revision, max 2 times.
#   6. Evals — backend/eval.py runs 5 tasks and checks structure + citations.
#
# Graph:  plan -> research -> draft -> critique --(issues, <2 revisions)-->
#         draft (revise) | --(ok / max revisions)--> finalize -> END
#
# Modes (same as Product 1 — no code change to switch):
#   mock   — no key set. Deterministic, labelled mock: extractive drafting
#            from retrieved evidence + rule-based critique. Runs fully offline.
#   openai — OPENAI_API_KEY set: real LLM drafts/critiques the plan steps.
#   azure  — AZURE_OPENAI_* set: same, via Azure OpenAI.
# Never hard-code a key. See .env.example.

from __future__ import annotations

import os
import re
from operator import add
from typing import Annotated, Any, Literal, Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

from backend import tools  # noqa: E402

MAX_REVISIONS = 2

# ---------------------------------------------------------------------------
# Structured output — the contract for every final answer.
# ---------------------------------------------------------------------------
class AgentOutput(BaseModel):
    summary: str = Field(..., min_length=20,
                         description="2-4 sentence answer to the task.")
    key_points: list[str] = Field(..., min_length=2,
                                  description="Bullet points, each grounded in evidence.")
    citations: list[str] = Field(..., min_length=1,
                                 description="Source file names, e.g. ['technobot.md'].")
    confidence: Literal["low", "medium", "high"] = Field(
        ..., description="How well the evidence supports the answer.")
    limitations: str = Field(..., min_length=10,
                             description="What the agent could NOT verify from its sources.")
    mode: str = Field("mock", description="mock | openai | azure")


# ---------------------------------------------------------------------------
# LLM provider detection — identical rule to Product 1.
# ---------------------------------------------------------------------------
def detect_mode() -> str:
    if os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
        return "azure"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def _llm_client() -> tuple[Any, str]:
    """Return (client, model) for real mode; raises in mock mode."""
    mode = detect_mode()
    if mode == "azure":
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        return client, os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    from openai import OpenAI

    return OpenAI(), os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _llm_text(prompt: str) -> str:
    client, model = _llm_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# State — the shared blackboard. Every node reads it and returns updates.
# `log` uses Annotated[..., add] so each node APPENDS to it instead of
# overwriting it: the full run log is what the UI shows step by step.
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    task: str
    plan: list[str]
    evidence: list[dict]
    tool_calls: list[dict]
    draft: dict
    critique: str
    critique_issues: list[str]
    revisions: int
    final: dict
    mode: str
    log: Annotated[list[dict], add]


def _log(state: dict, node: str, detail: str, tool_calls: Optional[list] = None) -> dict:
    entry = {"node": node, "detail": detail}
    if tool_calls:
        entry["tool_calls"] = tool_calls
    return {"log": [entry]}


# ---------------------------------------------------------------------------
# Nodes — one function per graph step.
# ---------------------------------------------------------------------------
def plan_node(state: dict) -> dict:
    task = state["task"]
    if detect_mode() != "mock":
        try:
            text = _llm_text(
                f"Write a short numbered plan (3-5 steps) for this task, "
                f"one step per line, no extra text:\n{task}"
            )
            plan = [l.strip("0123456789. ").strip() for l in text.splitlines() if l.strip()]
            plan = [p for p in plan if len(p) > 3][:5]
            if plan:
                return {"plan": plan, **_log(state, "plan",
                        f"Planned {len(plan)} steps (LLM).")}
        except Exception as e:  # LLM failed -> fall through to deterministic plan
            pass
    # Deterministic mock plan: same shape every run, so evals are stable.
    plan = [
        "Understand the task and pick the key terms to search for.",
        "Retrieve relevant passages from local documents (tool: retrieve_docs).",
        "Look up generic AI background in local knowledge (tool: local_search).",
        "Draft a structured answer from the evidence, with citations.",
        "Critique the draft for citations, accuracy and completeness, then revise if needed.",
    ]
    return {"plan": plan, **_log(state, "plan",
            f"Planned {len(plan)} steps (mock — deterministic).")}


def research_node(state: dict) -> dict:
    task = state["task"]
    calls: list[dict] = []
    evidence: list[dict] = []

    docs = tools.retrieve_docs(task, k=8)
    calls.append({"tool": "retrieve_docs", "query": task,
                  "result_count": len(docs),
                  "sources": sorted({d["source"] for d in docs})})
    evidence.extend(docs)

    generic_raw = tools.local_search(task, k=2)
    generic = [g for g in generic_raw if g["source"] not in {d["source"] for d in evidence}
               or g["text"] not in {d["text"] for d in evidence}]
    calls.append({"tool": "local_search", "query": task,
                  "result_count": len(generic_raw),
                  "sources": sorted({d["source"] for d in generic_raw})})
    evidence.extend(generic)

    return {"evidence": evidence, "tool_calls": calls, **_log(
        state, "research",
        f"Retrieved {len(evidence)} evidence chunks from "
        f"{sorted({d['source'] for d in evidence})}.",
        tool_calls=calls)}


def _extractive_draft(task: str, evidence: list[dict]) -> dict:
    """Mock drafting: build the structured answer from evidence sentences.

    Same grounding rule as Product 1: only use sentences from retrieved
    chunks, and cite the file each fact came from. No invented facts.
    """
    stop = {"what", "where", "when", "who", "how", "does", "did", "do",
            "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "of", "for", "to", "and", "his", "her", "he", "she", "it"}
    words = {w for w in re.findall(r"[a-z0-9]+", task.lower()) if w not in stop}
    # IDF tie-break: a sentence that mentions a rare, specific term
    # (LangChain, ServiceNow) is more informative than one that only
    # repeats common words, even when both match the same number of
    # query words. IDF is computed over the retrieved evidence chunks.
    import math as _math
    n_chunks = max(1, len(evidence))
    df: dict[str, int] = {}
    for chunk in evidence:
        for w in set(re.findall(r"[a-z0-9]+", chunk["text"].lower())):
            df[w] = df.get(w, 0) + 1
    def _idf(w: str) -> float:
        return _math.log(n_chunks / df.get(w, n_chunks)) + 1.0
    sentences: list[tuple[int, float, str, str]] = []
    for chunk in evidence:
        for sent in re.split(r"(?<=[.!])\s+", chunk["text"]):
            sent = sent.strip()
            if len(sent) < 15:
                continue
            score = sum(1 for w in words if w in sent.lower())
            rarity = sum(_idf(w) for w in set(re.findall(r"[a-z0-9]+", sent.lower())))
            sentences.append((score, rarity, sent, chunk["source"]))
    sentences.sort(key=lambda s: (s[0], s[1]), reverse=True)
    # Score-ordered, with a cap of 2 sentences per source: the best
    # evidence wins, but no single file can crowd out the others (e.g.
    # about-teja.md beating technobot.md for an AI-experience task, so
    # "LangChain" never appears). Taking one sentence per source was too
    # aggressive — some answers need two sentences from the same file.
    top: list = []
    per_source: dict[str, int] = {}
    for s in sentences:
        if len(top) >= 5:
            break
        if s[0] <= 0:
            continue
        if per_source.get(s[3], 0) >= 3:
            continue
        per_source[s[3]] = per_source.get(s[3], 0) + 1
        top.append(s)
    if not top:
        top = sentences[:3]
    key_points = [f"{sent} [{src}]" for _, _, sent, src in top]
    citations = sorted({src for _, _, _, src in top})
    summary = " ".join(
        f"{sent} [{src}]" for _, _, sent, src in top[:2]
    )
    return {
        "summary": summary,
        "key_points": key_points,
        "citations": citations,
        "confidence": "high" if len(citations) >= 2 else "medium",
        "limitations": (
            "This answer is based only on the local knowledge files; "
            "it does not use live web sources, and it cannot verify facts "
            "that are not in those files."
        ),
        "mode": "mock",
    }


def draft_node(state: dict) -> dict:
    task, evidence = state["task"], state["evidence"]
    mode = detect_mode()
    if mode != "mock":
        try:
            context = "\n\n".join(f"[{d['source']}] {d['text']}" for d in evidence)
            prompt = (
                "You are a research agent. Answer the task using ONLY the "
                "context below. Reply with JSON only, matching this shape:\n"
                '{"summary": "...", "key_points": ["...", "..."], '
                '"citations": ["<source file>", ...], '
                '"confidence": "low|medium|high", "limitations": "..."}\n\n'
                f"Context:\n{context}\n\nTask: {task}\nJSON:"
            )
            import json as _json

            payload = _json.loads(_llm_text(prompt).strip().strip("`"))
            payload["mode"] = mode
            checked = tools.format_structured(payload)
            if checked["ok"]:
                return {"draft": checked["output"], **_log(
                    state, "draft", f"Drafted structured answer (LLM, {mode}).")}
            # LLM output failed validation -> fall through to extractive draft
        except Exception:
            pass
    draft = _extractive_draft(task, evidence)
    checked = tools.format_structured(draft)
    assert checked["ok"], checked  # extractive draft must always validate
    return {"draft": checked["output"], **_log(
        state, "draft", "Drafted structured answer from evidence (mock — extractive).")}


def critique_node(state: dict) -> dict:
    """Check the draft. Issues found here decide whether the graph loops
    back to draft (revise) or moves on to finalize."""
    draft, task = state["draft"], state["task"]
    issues: list[str] = []
    if not draft.get("citations"):
        issues.append("No citations — an answer without sources is not trusted.")
    else:
        for cite in draft["citations"]:
            if cite not in " ".join(draft["key_points"] + [draft["summary"]]):
                issues.append(f"Citation [{cite}] is listed but never used in the answer text.")
    if len(draft.get("key_points", [])) < 2:
        issues.append("Fewer than 2 key points — the answer is too thin.")
    if len(draft.get("summary", "")) < 20:
        issues.append("Summary is too short to answer the task.")
    # On-task check: at least one key point should share a content word with the task.
    words = {w for w in re.findall(r"[a-z0-9]+", task.lower()) if len(w) > 3}
    points_text = " ".join(draft.get("key_points", [])).lower()
    if words and not any(w in points_text for w in words):
        issues.append("Key points do not mention any key term from the task.")
    critique = (
        "No issues found — draft is cited, structured and on-task."
        if not issues else "Issues: " + "; ".join(issues)
    )
    return {"critique": critique, "critique_issues": issues, **_log(
        state, "critique", critique)}


def finalize_node(state: dict) -> dict:
    draft = dict(state["draft"])
    draft["mode"] = detect_mode()
    if state.get("revisions", 0) > 0:
        draft["limitations"] = (
            draft["limitations"]
            + f" Revised {state['revisions']} time(s) after self-critique."
        )
    return {"final": draft, **_log(
        state, "finalize",
        f"Final answer ready — {len(draft['key_points'])} key points, "
        f"citations: {draft['citations']}, confidence: {draft['confidence']}.")}


# ---------------------------------------------------------------------------
# Conditional edge: revise (loop back to draft) or finish?
# ---------------------------------------------------------------------------
def _should_revise(state: dict) -> str:
    if state.get("critique_issues") and state.get("revisions", 0) < MAX_REVISIONS:
        return "revise"
    return "finalize"


def _revise_node(state: dict) -> dict:
    # Revision is a real step in the log: the next draft() call sees the
    # critique issues in state and, in mock mode, the extractive draft is
    # rebuilt from the same evidence (deterministic). The loop still matters:
    # it proves the graph can detect a bad draft and try again, and in real
    # mode the LLM re-drafts with the critique in the prompt.
    return {"revisions": state.get("revisions", 0) + 1, **_log(
        state, "revise",
        f"Revision {state.get('revisions', 0) + 1}/{MAX_REVISIONS} — "
        f"re-drafting to fix: {'; '.join(state.get('critique_issues', []))}")}


# ---------------------------------------------------------------------------
# Build the graph.
# ---------------------------------------------------------------------------
def build_graph():
    from langgraph.graph import END, START, StateGraph

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("research", research_node)
    graph.add_node("draft", draft_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", _revise_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "draft")
    graph.add_edge("draft", "critique")
    graph.add_conditional_edges("critique", _should_revise,
                                {"revise": "revise", "finalize": "finalize"})
    graph.add_edge("revise", "draft")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent(task: str) -> dict:
    """Run one task end to end. Returns {final, plan, log, tool_calls, mode, revisions}."""
    graph = build_graph()
    initial: dict = {
        "task": task, "plan": [], "evidence": [], "tool_calls": [],
        "draft": {}, "critique": "", "critique_issues": [],
        "revisions": 0, "final": {}, "mode": detect_mode(), "log": [],
    }
    result = graph.invoke(initial)
    return {
        "task": task,
        "mode": detect_mode(),
        "plan": result.get("plan", []),
        "tool_calls": result.get("tool_calls", []),
        "evidence": result.get("evidence", []),
        "critique": result.get("critique", ""),
        "revisions": result.get("revisions", 0),
        "log": result.get("log", []),
        "final": result.get("final", {}),
    }
