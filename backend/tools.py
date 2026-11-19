# Teja LangGraph Agent — Product 2
# Tools the agent can call. All local and mockable — no paid API, no web
# scraping, no invented facts. Every tool returns evidence the agent must
# cite, so a tool call that finds nothing is visible in the run log.
#
# Tool 1: retrieve_docs — keyword retrieval over data/knowledge/*.md
#         (Teja's own public facts + one generic RAG-vs-fine-tuning note).
#         This is the same idea as Product 1's RAG retrieval, simplified:
#         chunk the docs, rank chunks by keyword overlap with the query.
# Tool 2: local_search — a small curated knowledge lookup for generic AI
#         topics (RAG, fine-tuning, LangGraph, FastAPI). Labelled as local
#         knowledge, NOT a live web search: it only knows what is written
#         in the knowledge files. If the topic is not in the files, it
#         says so instead of inventing an answer.
# Tool 3: format_structured — validates the agent's final answer against
#         the AgentOutput Pydantic model (see agent.py). A malformed
#         answer is rejected and sent back for revision.

from __future__ import annotations

import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"

STOPWORDS = {
    "what", "where", "when", "who", "how", "does", "did", "do", "the",
    "a", "an", "is", "are", "was", "were", "in", "on", "at", "of",
    "for", "to", "and", "his", "her", "he", "she", "it", "with",
    "summarize", "compare", "tell", "me", "about", "list", "give",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def _chunks() -> list[dict]:
    """Split every knowledge file into ~500-char chunks (same idea as Product 1)."""
    out: list[dict] = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            out.append({"source": path.name, "text": para})
    return out


def retrieve_docs(query: str, k: int = 3) -> list[dict]:
    """Tool 1 — retrieve the k most relevant chunks for a query.

    Returns [{source, text, score}]. Score = number of shared content words.
    In mock mode this keyword score is the retrieval signal (same approach
    as Product 1's demo mode); with a real embedding model you would rank
    by vector similarity instead — the tool's interface stays the same.
    """
    words = _content_words(query)
    scored = []
    for chunk in _chunks():
        text = chunk["text"].lower()
        score = sum(1 for w in words if w in text)
        # Small bonus: the whole query phrase appearing verbatim.
        if query.lower()[:30] in text:
            score += 2
        scored.append({**chunk, "score": score})
    scored.sort(key=lambda c: c["score"], reverse=True)
    # Diversity, round-robin: best chunk of each source, then second-best
    # of each, and so on, until k. One chunk per source was too aggressive
    # — the key fact often lives in a file's *second* paragraph (e.g. the
    # LangChain sentence is in technobot.md's second paragraph, the
    # ServiceNow sentence in elite-tech.md's second paragraph).
    by_source: dict[str, list] = {}
    for c in scored:
        by_source.setdefault(c["source"], []).append(c)
    ranked: list[dict] = []
    sources = sorted(by_source, key=lambda s: -by_source[s][0]["score"])
    round_no = 0
    while len(ranked) < k:
        added = False
        for src in sources:
            chunks = [c for c in by_source[src] if c["score"] > 0]
            if round_no < len(chunks):
                ranked.append(chunks[round_no])
                added = True
            if len(ranked) >= k:
                break
        if not added:
            break
        round_no += 1
    return ranked[:k] or scored[:1]


def local_search(query: str, k: int = 3) -> list[dict]:
    """Tool 2 — local knowledge lookup (NOT a live web search).

    Searches the same knowledge files, but only returns chunks that match
    generic AI-topic words (rag, fine-tuning, langgraph, agent, fastapi,
    vector, embedding). If nothing matches, it returns an empty list and
    the agent must say the topic is not in its local knowledge — it must
    not invent an answer. This is the honest version of a search tool:
    a real web-search tool would be swapped in here later, behind this
    same function signature.
    """
    topic_words = {"rag", "fine-tuning", "finetuning", "langgraph", "agent",
                   "fastapi", "vector", "embedding", "retrieval", "llm"}
    words = _content_words(query)
    if not (words & topic_words):
        return []
    return retrieve_docs(query, k=k)


def format_structured(payload: dict) -> dict:
    """Tool 3 — validate a structured answer (Pydantic, see agent.py).

    Returns {"ok": True, "output": ...} or {"ok": False, "errors": [...]}.
    The critique node uses this: a draft that fails validation goes back
    for revision instead of being shown to the user.
    """
    from backend.agent import AgentOutput  # local import: avoids a cycle

    try:
        out = AgentOutput(**payload)
        return {"ok": True, "output": out.model_dump()}
    except Exception as e:  # Pydantic ValidationError
        return {"ok": False, "errors": [str(e)]}
