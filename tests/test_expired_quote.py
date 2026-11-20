"""Expired-quote test (CJ Lee's test, suggested 2026-11-20).

The test, exactly as CJ put it: give the agent two vendor quotes,
tell it the cheaper one expired yesterday, run the shortlist twice
switching models halfway, and see whether either shortlist still
offers the expired price.

Setup (fixed dates, so the test is deterministic whenever it runs):
  * as-of date: 2026-11-20 (the day Teja ran the test)
  * Vendor A: $850,  valid until 2026-11-19 — expired YESTERDAY, cheaper
  * Vendor B: $1,200, valid until 2026-12-31 — still valid, dearer

Runs:
  * Run 1 — model label "mock-model-a (gpt-4o-mini)"
  * Run 2 — model label "mock-model-b (gpt-4o)"  (the model switch)
  Both run in mock mode (no API key). The "model switch" is simulated
  with two mock model configs/labels: in mock mode there is no real LLM
  call, so the label records which model each run stands for. The guard
  itself is plain Python, not an LLM judgement — that is precisely why
  it must hold across the switch. With a real key, run the same script
  with OPENAI_MODEL set differently per run instead.

What must hold in EACH run (a failure in either run fails the test):
  1. The expired price ($850 / 850) appears NOWHERE in the final
     summary or key points — the agent must not offer the dead price.
  2. Vendor A is NOT in the shortlist.
  3. The shortlist's cheapest (only) valid vendor is Vendor B, $1,200.
  4. The run log contains the verify_quotes guard step, and the
     excluded list names Vendor A with its expiry reason.

Baseline (honest pre-guard result — see docs/expired-quote-test-RESULTS.md):
  Before the guard, this agent had NO vendor-quote handling at all: a
  task carrying quotes got a generic extractive summary from the local
  knowledge files and no shortlist. And a naive cheapest-first
  shortlist — sort all quotes by price, no expiry check, which is what
  the obvious implementation (and an unguarded LLM draft) does — puts
  the expired Vendor A $850 first, i.e. it OFFERS the expired price.
  That baseline is reproduced here by naive_shortlist_without_guard()
  and must FAIL; if it ever passed, the test data would be wrong.

Run:
  python tests/test_expired_quote.py        # from the project root
  python -m pytest tests/test_expired_quote.py -v   # if pytest is installed
Exit code 0 = both guarded runs passed (and the baseline failed as expected).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agent import run_vendor_shortlist  # noqa: E402
from backend.quotes import Quote, naive_shortlist_without_guard  # noqa: E402

AS_OF = "2026-11-20"  # the day the test was run; Vendor A expired yesterday
TASK = "Shortlist vendors for an inventory tool. Pick the cheapest valid vendor."

QUOTES = [
    {"vendor": "Vendor A", "price": 850, "valid_until": "2026-11-19"},
    {"vendor": "Vendor B", "price": 1200, "valid_until": "2026-12-31"},
]

EXPIRED_PRICE_DISPLAY = "$850"
EXPIRED_VENDOR = "Vendor A"
VALID_VENDOR = "Vendor B"
VALID_PRICE_DISPLAY = "$1,200"


def check_run(out: dict, label: str) -> tuple[bool, list[str]]:
    """Check one guarded run. Returns (passed, findings)."""
    findings: list[str] = []
    final = out["final"]
    answer_text = final.get("summary", "") + " " + " ".join(final.get("key_points", []))
    shortlist = out["shortlist"]
    nodes = [e["node"] for e in out["log"]]

    # 1. Expired price must not be offered.
    if EXPIRED_PRICE_DISPLAY in answer_text or " 850" in answer_text or answer_text.strip().endswith("850"):
        findings.append(f"FAIL: expired price {EXPIRED_PRICE_DISPLAY} appears in the final answer.")
    else:
        findings.append(f"ok: expired price {EXPIRED_PRICE_DISPLAY} nowhere in the final answer.")
    # 2. Expired vendor must not be shortlisted.
    if any(s["vendor"] == EXPIRED_VENDOR for s in shortlist):
        findings.append(f"FAIL: {EXPIRED_VENDOR} is in the shortlist.")
    else:
        findings.append(f"ok: {EXPIRED_VENDOR} is not in the shortlist.")
    # 3. Cheapest valid vendor is Vendor B at $1,200.
    if shortlist and shortlist[0]["vendor"] == VALID_VENDOR and shortlist[0]["price_display"] == VALID_PRICE_DISPLAY:
        findings.append(f"ok: shortlist top is {VALID_VENDOR} {VALID_PRICE_DISPLAY}.")
    else:
        findings.append(f"FAIL: shortlist top is not {VALID_VENDOR} {VALID_PRICE_DISPLAY}: {shortlist}")
    # 4. Guard ran and recorded the exclusion.
    if "verify_quotes" in nodes:
        findings.append("ok: verify_quotes guard step ran (in run log).")
    else:
        findings.append(f"FAIL: verify_quotes step missing from log: {nodes}")
    if any(e["vendor"] == EXPIRED_VENDOR for e in out["excluded_quotes"]):
        findings.append(f"ok: {EXPIRED_VENDOR} recorded as excluded (expired 2026-11-19).")
    else:
        findings.append(f"FAIL: {EXPIRED_VENDOR} not in excluded_quotes: {out['excluded_quotes']}")
    # Also: the dead price must not leak via the limitations note either.
    if EXPIRED_PRICE_DISPLAY in final.get("limitations", ""):
        findings.append("FAIL: expired price leaked into limitations.")
    passed = all(f.startswith("ok") for f in findings)
    print(f"\n--- {label} (model: {out['model_label']}) ---")
    for f in findings:
        print(f"  {f}")
    print(f"  shortlist: {shortlist}")
    print(f"  summary: {final.get('summary', '')}")
    return passed, findings


def test_baseline_naive_shortlist_offers_expired_price():
    """Pre-guard baseline: cheapest-first with no expiry check offers $850."""
    naive = naive_shortlist_without_guard([Quote(**q) for q in QUOTES])
    assert naive[0]["vendor"] == EXPIRED_VENDOR
    assert naive[0]["price_display"] == EXPIRED_PRICE_DISPLAY


def test_expired_quote_run_1():
    out = run_vendor_shortlist(TASK, QUOTES, as_of=AS_OF,
                               model_label="mock-model-a (gpt-4o-mini)")
    passed, _ = check_run(out, "Run 1")
    assert passed


def test_expired_quote_run_2_model_switch():
    out = run_vendor_shortlist(TASK, QUOTES, as_of=AS_OF,
                               model_label="mock-model-b (gpt-4o)")
    passed, _ = check_run(out, "Run 2 (model switched)")
    assert passed


def test_quote_expiring_today_is_still_valid():
    """Guard boundary: valid_until == as_of is valid (expires end of day)."""
    from datetime import date

    from backend.quotes import verify_quotes

    valid, excluded = verify_quotes(
        [Quote(vendor="Vendor C", price=900, valid_until="2026-11-20")],
        as_of=date(2026, 11, 20),
    )
    assert len(valid) == 1 and not excluded


def main() -> int:
    print("Expired-quote test — CJ Lee's test (mock mode, no API key)")
    print(f"As of {AS_OF}: Vendor A $850 expired 2026-11-19 (yesterday), "
          f"Vendor B $1,200 valid until 2026-12-31\n")

    # Baseline first, honestly: the pre-guard approach fails.
    naive = naive_shortlist_without_guard([Quote(**q) for q in QUOTES])
    baseline_fails = (
        naive[0]["vendor"] == EXPIRED_VENDOR
        and naive[0]["price_display"] == EXPIRED_PRICE_DISPLAY
    )
    print(f"Baseline (naive cheapest-first, no expiry check): "
          f"top = {naive[0]['vendor']} {naive[0]['price_display']} — "
          f"{'OFFERS the expired price (FAIL, as expected)' if baseline_fails else 'unexpectedly passed?!'}")

    run1 = run_vendor_shortlist(TASK, QUOTES, as_of=AS_OF,
                                model_label="mock-model-a (gpt-4o-mini)")
    ok1, _ = check_run(run1, "Run 1")
    run2 = run_vendor_shortlist(TASK, QUOTES, as_of=AS_OF,
                                model_label="mock-model-b (gpt-4o)")
    ok2, _ = check_run(run2, "Run 2 (model switched halfway)")

    print(f"\nResult: baseline fails as expected: {'yes' if baseline_fails else 'NO — check test data'} | "
          f"Run 1: {'PASS' if ok1 else 'FAIL'} | Run 2: {'PASS' if ok2 else 'FAIL'}")
    return 0 if (baseline_fails and ok1 and ok2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
