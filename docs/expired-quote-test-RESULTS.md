# Expired-quote test — RESULTS

**Date run:** 2026-11-20 (mock mode, no API key)
**Test suggested by:** Choongjing (CJ) Lee, founder of Artificial Beings, in a
comment on Krishnateja's LinkedIn Post 1 (2026-11-20). Krishnateja publicly promised to run it.
**Code:** `backend/quotes.py` (guard), `backend/agent.py` (`verify_quotes` node),
test `tests/test_expired_quote.py`.

## The test, as CJ put it

> Give the agent two vendor quotes, tell it the cheaper one expired
> yesterday, run the shortlist twice switching models halfway, and see
> whether either shortlist still offers the expired price.

## Setup

| | |
|---|---|
| As-of date | **2026-11-20** (the day the test ran) |
| Vendor A | **$850**, valid until **2026-11-19** — expired *yesterday*, and cheaper |
| Vendor B | **$1,200**, valid until **2026-12-31** — still valid, dearer |
| Task | "Shortlist vendors for an inventory tool. Pick the cheapest valid vendor." |
| Run 1 model | `mock-model-a (gpt-4o-mini)` |
| Run 2 model | `mock-model-b (gpt-4o)` — the model switch, halfway through |

Both runs are in **mock mode** (no API key). The "model switch" is simulated
with two mock model configs/labels: in mock mode there is no real LLM call,
so the label records which model each run stands for. The guard itself is
plain Python, not an LLM judgement — that is precisely why it must hold
across the switch. With a real key, run the same script with `OPENAI_MODEL`
set differently per run instead.

## Baseline — what happened BEFORE the guard (honest)

Before this change the agent had **no vendor-quote handling at all**. Given
the same task, it returned a generic extractive summary from the local
knowledge files (Technobot vendor-selection sentences) and **no shortlist** —
verified by running `run_agent()` on the quote-carrying task before the
change: no `shortlist` field, no price mentioned. It could not do CJ's test.

And the obvious implementation — a naive cheapest-first shortlist with no
expiry check, which is also what an unguarded LLM draft does — **fails**:
it puts Vendor A $850 first, i.e. it offers a price that expired yesterday.
That baseline is reproduced in the test itself
(`naive_shortlist_without_guard()` in `backend/quotes.py`) and it fails
exactly as expected.

So the pre-guard result is: **FAIL — by omission (no shortlist, no expiry
check) and by naive price-sorting (offers the expired $850).**

## What the guard does

A new graph node, `verify_quotes`, runs after `research` and **before**
`draft` on every run:

```
plan → research → verify_quotes → draft → critique → (revise | finalize)
```

1. It takes the quotes (structured input, or parsed from the task text)
   and checks each `valid_until` against the as-of date in plain Python.
   Valid = `valid_until >= as_of` — a quote expiring *today* is still valid.
2. Expired quotes are dropped from the shortlist, recorded in
   `excluded_quotes` with a reason, and logged. Only valid quotes reach
   the draft — in mock **and** real-LLM mode, the vendor draft is built
   from the verified shortlist only, so no LLM can re-introduce a dead price.
3. Defence in depth: the `critique` node flags any excluded price that
   somehow reaches the draft, sending the graph back to revise instead
   of finalizing it. The final answer names the excluded vendor and its
   expiry date in `limitations`, but **never its price**.

## Results per run

| Check | Run 1 (mock-model-a) | Run 2 (mock-model-b, switched) |
|---|---|---|
| Expired price $850 in final answer | No — **PASS** | No — **PASS** |
| Vendor A in the shortlist | No — **PASS** | No — **PASS** |
| Shortlist top = Vendor B, $1,200 | Yes — **PASS** | Yes — **PASS** |
| `verify_quotes` guard step in run log | Yes — **PASS** | Yes — **PASS** |
| Vendor A recorded as excluded (expired 2026-11-19) | Yes — **PASS** | Yes — **PASS** |
| **Overall** | **PASS** | **PASS** |

Both runs returned the same shortlist:

```json
[{"rank": 1, "vendor": "Vendor B", "price_display": "$1,200", "valid_until": "2026-12-31"}]
```

Final summary (both runs, identical):
> "Shortlist (verified as of 2026-11-20): the cheapest valid vendor is
> Vendor B at $1,200, valid until 2026-12-31. 1 valid quote(s) shortlisted."

Limitations note (both runs): "Excluded 1 expired quote(s), not offered:
Vendor A (expired 2026-11-19)." — vendor named, dead price never shown.

Boundary check (in the test file): a quote with `valid_until` equal to the
as-of date is treated as **valid**.

## Regression checks (same session, 2026-11-20)

- Existing eval: `python backend/eval.py` → **5/5 passed** (unchanged).
- Server: started on port 8001, `/health` ok (`mode: mock`, graph shows
  `verify_quotes`), normal `/run` task works, `/run` with structured
  quotes works, `/run` with quotes in plain task text also works
  (quotes parsed from the text).
- `pytest` is not installed in the project venv, so the test was run as a
  script; the file also defines `test_*` functions for pytest when available.

## How to run it

```bash
python tests/test_expired_quote.py
# Exit 0 = both guarded runs passed (and the baseline failed as expected).
# Expected output ends: "Run 1: PASS | Run 2: PASS"
```

## Limitation worth stating

This test runs in mock mode, so it proves the **guard** is model-independent,
not that a real LLM would never hallucinate a price — the guard is designed
so it does not have to trust the LLM at all. The next step, with a real key,
is to repeat the two runs with `OPENAI_MODEL` switched between them.
