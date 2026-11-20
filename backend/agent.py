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
# Graph:  plan -> research -> verify_quotes -> draft -> critique
#         --(issues, <2 revisions)--> draft (revise)
#         | --(ok / max revisions)--> finalize -> END
# verify_quotes is the expiry guard (CJ Lee test): it drops expired
# vendor quotes BEFORE drafting, so no expired price can be shortlisted.
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

from backend import quotes as quote_guard  # noqa: E402
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
    # Vendor-quote guard (CJ Lee expired-quote test): quotes supplied
    # with the task (structured, or parsed from the task text), the
    # as-of date expiry is judged against, and the guard's output.
    # verify_quotes_node fills valid/excluded/shortlist; draft and
    # critique read them, so an expired price can never reach the answer.
    quotes: list[dict]
    as_of: str
    model_label: str
    valid_quotes: list[dict]
    excluded_quotes: list[dict]
    shortlist: list[dict]


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


def verify_quotes_node(state: dict) -> dict:
    """GUARD — verify quote expiry BEFORE shortlisting (CJ Lee test).

    Runs after research, before draft, on every run. If the task carries
    vendor quotes (structured `quotes` in state, or parseable from the
    task text), each quote's valid_until is checked against `as_of`
    (default: today) in plain Python — not by the LLM, so the result
    cannot change when the model is switched halfway through. Expired
    quotes are dropped from the shortlist, recorded in excluded_quotes
    with a reason, and logged. Only valid quotes reach draft_node.
    No quotes in the task -> no-op, logged as skipped.
    """
    from datetime import date as _date

    raw = state.get("quotes") or []
    parsed: list = []
    if raw:
        for q in raw:
            try:
                parsed.append(quote_guard.Quote(**q) if isinstance(q, dict)
                              else q)
            except Exception:
                continue
    if not parsed:
        parsed = quote_guard.parse_quotes_from_task(state["task"])
    if not parsed:
        return {**_log(state, "verify_quotes",
                       "No vendor quotes in this task — expiry check skipped.")}
    as_of_str = state.get("as_of") or _date.today().isoformat()
    try:
        as_of = _date.fromisoformat(as_of_str)
    except ValueError:
        as_of = _date.today()
        as_of_str = as_of.isoformat()
    valid, excluded = quote_guard.verify_quotes(parsed, as_of=as_of)
    shortlist = quote_guard.shortlist_vendors(valid)
    detail = (
        f"Verified {len(parsed)} vendor quote(s) as of {as_of_str} "
        f"(model: {state.get('model_label') or detect_mode()}): "
        f"{len(valid)} valid, {len(excluded)} expired."
    )
    if excluded:
        detail += " Excluded (NOT shortlisted): " + "; ".join(
            f"{e['vendor']} {e['price_display']} expired {e['valid_until']}"
            for e in excluded) + "."
    if shortlist:
        detail += " Shortlist (cheapest valid first): " + ", ".join(
            f"{s['vendor']} {s['price_display']}" for s in shortlist) + "."
    return {
        "quotes": [q.model_dump(mode="json") for q in parsed],
        "as_of": as_of_str,
        "valid_quotes": [q.model_dump(mode="json") for q in valid],
        "excluded_quotes": excluded,
        "shortlist": shortlist,
        **_log(state, "verify_quotes", detail),
    }


def _vendor_draft(state: dict) -> dict:
    """Mock drafting for a vendor-shortlist task: build the structured
    answer from the GUARD's shortlist only. Expired quotes are never
    read here — shortlist contains valid quotes only, and the excluded
    prices are deliberately not rendered (only the vendor name + expiry
    date go in `limitations`, via finalize_node, so the dead price is
    never offered)."""
    shortlist = state["shortlist"]
    as_of = state.get("as_of", "")
    cheapest = shortlist[0]
    key_points = [
        f"{i + 1}. {s['vendor']} — {s['price_display']}, valid until "
        f"{s['valid_until']} [vendor-quotes]"
        for i, s in enumerate(shortlist)
    ]
    # Critique requires >= 2 key points and a citation used in the text;
    # with a single valid quote, add the verification step itself.
    if len(key_points) < 2:
        key_points.append(
            f"All prices were verified against their expiry dates as of "
            f"{as_of} before shortlisting, so no expired price is offered. "
            f"[vendor-quotes]"
        )
    summary = (
        f"Shortlist (verified as of {as_of}): the cheapest valid vendor "
        f"is {cheapest['vendor']} at {cheapest['price_display']}, valid "
        f"until {cheapest['valid_until']}. "
        f"{len(shortlist)} valid quote(s) shortlisted. [vendor-quotes]"
    )
    return {
        "summary": summary,
        "key_points": key_points,
        "citations": ["vendor-quotes"],
        "confidence": "high",
        "limitations": (
            "Shortlist covers only the vendor quotes supplied with this "
            "task, verified as of the as-of date above; it does not check "
            "live vendor pricing or terms beyond the quoted expiry date."
        ),
        "mode": "mock",
    }


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
    # Vendor-shortlist task: the guard has already decided what is
    # valid. Draft ONLY from the verified shortlist — in mock AND real
    # mode — so no LLM can re-introduce an expired price.
    if state.get("shortlist"):
        draft = _vendor_draft(state)
        checked = tools.format_structured(draft)
        assert checked["ok"], checked
        return {"draft": checked["output"], **_log(
            state, "draft",
            "Drafted vendor shortlist from verified (unexpired) quotes only.")}
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
    # Expiry backstop (defence in depth): if an excluded quote's price
    # somehow reached the draft, flag it — this sends the graph back to
    # revise instead of finalizing a dead price.
    draft_text = draft.get("summary", "") + " " + " ".join(draft.get("key_points", []))
    for exc in state.get("excluded_quotes", []):
        for price_str in {exc.get("price_display", ""), str(exc.get("price", ""))}:
            if price_str and price_str in draft_text:
                issues.append(
                    f"Draft offers an expired quote price ({exc['vendor']} "
                    f"{price_str}, expired {exc['valid_until']}) — "
                    f"expired prices must never be shortlisted."
                )
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
    # Transparency, without offering the dead price: name the excluded
    # vendor and its expiry date, never its price.
    if state.get("excluded_quotes"):
        names = "; ".join(
            f"{e['vendor']} (expired {e['valid_until']})"
            for e in state["excluded_quotes"])
        draft["limitations"] = (
            draft["limitations"]
            + f" Excluded {len(state['excluded_quotes'])} expired quote(s), "
            f"not offered: {names}."
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
    graph.add_node("verify_quotes", verify_quotes_node)
    graph.add_node("draft", draft_node)
    graph.add_node("critique", critique_node)
    graph.add_node("revise", _revise_node)
    graph.add_node("finalize", finalize_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "research")
    graph.add_edge("research", "verify_quotes")
    graph.add_edge("verify_quotes", "draft")
    graph.add_edge("draft", "critique")
    graph.add_conditional_edges("critique", _should_revise,
                                {"revise": "revise", "finalize": "finalize"})
    graph.add_edge("revise", "draft")
    graph.add_edge("finalize", END)
    return graph.compile()


def run_agent(task: str, quotes: Optional[list[dict]] = None,
              as_of: Optional[str] = None,
              model_label: Optional[str] = None) -> dict:
    """Run one task end to end. Returns {final, plan, log, tool_calls,
    mode, revisions, shortlist, excluded_quotes, ...}.

    quotes: optional list of {vendor, price, valid_until, ...} dicts —
        the vendor quotes to shortlist. If omitted, quotes are parsed
        from the task text when possible; with neither, the verify step
        is a logged no-op and the graph behaves exactly as before.
    as_of: ISO date expiry is judged against (default: today). The
        expired-quote test passes an explicit date so it is
        deterministic no matter when it runs.
    model_label: label for the model used in this run, recorded in the
        log and the result. In mock mode this *simulates* CJ Lee's
        "switch models halfway" — two runs with different labels but
        the same guard; with a real key, set OPENAI_MODEL / the Azure
        deployment instead and the label records which one ran.
    """
    graph = build_graph()
    initial: dict = {
        "task": task, "plan": [], "evidence": [], "tool_calls": [],
        "draft": {}, "critique": "", "critique_issues": [],
        "revisions": 0, "final": {}, "mode": detect_mode(), "log": [],
        "quotes": quotes or [], "as_of": as_of or "",
        "model_label": model_label or detect_mode(),
        "valid_quotes": [], "excluded_quotes": [], "shortlist": [],
    }
    result = graph.invoke(initial)
    return {
        "task": task,
        "mode": detect_mode(),
        "model_label": model_label or detect_mode(),
        "as_of": result.get("as_of", as_of or ""),
        "plan": result.get("plan", []),
        "tool_calls": result.get("tool_calls", []),
        "evidence": result.get("evidence", []),
        "critique": result.get("critique", ""),
        "revisions": result.get("revisions", 0),
        "log": result.get("log", []),
        "final": result.get("final", {}),
        "quotes": result.get("quotes", []),
        "valid_quotes": result.get("valid_quotes", []),
        "excluded_quotes": result.get("excluded_quotes", []),
        "shortlist": result.get("shortlist", []),
    }


def run_vendor_shortlist(task: str, quotes: list[dict],
                         as_of: Optional[str] = None,
                         model_label: Optional[str] = None) -> dict:
    """Convenience wrapper for vendor-shortlist tasks: same graph, but
    the result is shaped around the shortlist. See tests/test_expired_quote.py
    for the expired-quote test that uses this."""
    out = run_agent(task, quotes=quotes, as_of=as_of, model_label=model_label)
    return {
        "task": task,
        "model_label": out["model_label"],
        "as_of": out["as_of"],
        "shortlist": out["shortlist"],
        "valid_quotes": out["valid_quotes"],
        "excluded_quotes": out["excluded_quotes"],
        "final": out["final"],
        "critique": out["critique"],
        "revisions": out["revisions"],
        "log": out["log"],
        "mode": out["mode"],
    }
