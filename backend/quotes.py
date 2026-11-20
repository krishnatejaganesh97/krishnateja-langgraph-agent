# Teja LangGraph Agent — vendor-quote expiry guard
#
# Why this module exists (CJ Lee's expired-quote test, 2026-11-20):
#   Give the agent two vendor quotes, mark the cheaper one expired
#   yesterday, run the shortlist twice switching models halfway, and see
#   whether either shortlist still offers the expired price.
#
# The failure mode it guards against: shortlisting by price alone.
# A naive "cheapest first" shortlist picks the $850 quote even though
# it expired yesterday — the agent would then offer a price the vendor
# no longer honours. The guard below runs BEFORE any shortlisting:
# every quote's expiry date is checked against an as-of date, expired
# quotes are dropped and logged, and only valid quotes reach the draft.
#
# Rules (deliberately simple and auditable):
#   * A quote is valid iff valid_until >= as_of. A quote expiring
#     *today* is still valid — it expires at the end of that day.
#   * Expiry is checked on the date, not the model: the guard is plain
#     Python, not an LLM judgement, so switching models cannot change
#     the outcome. That is the point of CJ's "switch models halfway"
#     test — the guard must hold for both models.
#   * The shortlist is valid quotes sorted by price, cheapest first.
#   * Excluded quotes are returned with a reason and logged, never
#     silently dropped — but their price is never put in the answer.

from __future__ import annotations

import re
from datetime import date
from typing import Optional

from pydantic import BaseModel, Field


class Quote(BaseModel):
    """One vendor quote. Dates are ISO dates (YYYY-MM-DD)."""

    vendor: str = Field(..., min_length=1)
    price: float = Field(..., gt=0)
    currency: str = "USD"
    valid_until: date = Field(..., description="Last day the quoted price is honoured.")
    note: str = ""


def format_price(price: float) -> str:
    """One canonical rendering, so tests can check for it: $850, $1,200."""
    return f"${price:,.0f}" if float(price).is_integer() else f"${price:,.2f}"


def verify_quotes(
    quotes: list[Quote], as_of: Optional[date] = None
) -> tuple[list[Quote], list[dict]]:
    """Split quotes into (valid, excluded).

    Returns (valid_quotes, excluded) where each excluded entry is
    {"vendor", "price", "valid_until", "reason"}. Expired = valid_until
    is strictly before as_of. Defaults as_of to today.
    """
    as_of = as_of or date.today()
    valid: list[Quote] = []
    excluded: list[dict] = []
    for q in quotes:
        if q.valid_until >= as_of:
            valid.append(q)
        else:
            excluded.append({
                "vendor": q.vendor,
                "price": q.price,
                "price_display": format_price(q.price),
                "valid_until": q.valid_until.isoformat(),
                "as_of": as_of.isoformat(),
                "reason": (
                    f"Expired on {q.valid_until.isoformat()} "
                    f"(as of {as_of.isoformat()}) — price no longer honoured."
                ),
            })
    return valid, excluded


def shortlist_vendors(valid_quotes: list[Quote]) -> list[dict]:
    """Cheapest-first shortlist over VALID quotes only. Never call this
    on unverified quotes — call verify_quotes() first."""
    ranked = sorted(valid_quotes, key=lambda q: q.price)
    return [
        {
            "rank": i + 1,
            "vendor": q.vendor,
            "price": q.price,
            "price_display": format_price(q.price),
            "currency": q.currency,
            "valid_until": q.valid_until.isoformat(),
        }
        for i, q in enumerate(ranked)
    ]


def naive_shortlist_without_guard(quotes: list[Quote]) -> list[dict]:
    """The pre-guard behaviour, kept for the test's honest baseline:
    sort ALL quotes by price with no expiry check. With CJ's test data
    this puts the expired $850 quote first — i.e. it offers a dead price.
    Used only by tests/docs to show what the guard fixes; never used
    by the agent itself."""
    ranked = sorted(quotes, key=lambda q: q.price)
    return [
        {"rank": i + 1, "vendor": q.vendor,
         "price_display": format_price(q.price),
         "valid_until": q.valid_until.isoformat()}
        for i, q in enumerate(ranked)
    ]


# ---------------------------------------------------------------------------
# Parse quotes out of plain task text, so the guard also works over /run
# with no structured input, e.g.:
#   "Vendor A: $850, valid until 2026-11-19. Vendor B: $1,200, valid
#    until 2026-12-31. Shortlist the cheapest valid vendor."
# ---------------------------------------------------------------------------
_QUOTE_RE = re.compile(
    r"(?P<vendor>[A-Z][A-Za-z0-9 .&'-]{0,40}?)\s*[:\-–]\s*"
    r"\$\s*(?P<price>[\d,]+(?:\.\d{2})?)"
    r"[^.\n]{0,60}?"
    r"(?:valid until|expires?|valid through)\s*(?P<date>\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def parse_quotes_from_task(task: str) -> list[Quote]:
    """Best-effort extraction of quotes from task text. Returns [] when
    the task contains no parseable quotes — the graph then skips the
    verify step (see verify_quotes_node in agent.py)."""
    quotes: list[Quote] = []
    for m in _QUOTE_RE.finditer(task or ""):
        try:
            quotes.append(Quote(
                vendor=m.group("vendor").strip(),
                price=float(m.group("price").replace(",", "")),
                valid_until=date.fromisoformat(m.group("date")),
            ))
        except ValueError:
            continue
    return quotes
