# Approvals & Pipeline — California RIF Copilot (Box 8 + orchestrator)

## Files

| File | Purpose |
|---|---|
| `approvals.py` | Box 8 — structured sign-off bound to a version fingerprint. |
| `rif_pipeline.py` | End-to-end orchestrator wiring boxes 1→8. |
| `test_approvals.py` | 42 tests. |

## Quick start

```python
from approvals import ApprovalLedger, ApprovalPackage

package = ApprovalPackage.from_pipeline(
    scenario="Scenario A", plan=plan, selection=selection,
    impact=impact, compliance=compliance, pay=pay,
)
ledger = ApprovalLedger()
ledger.submit(package, submitted_by="M. Chen", role="HR Business Partner")

ledger.approve("hr", "S. Patel", "HR Director", comment="Applied as documented.")
ledger.approve("legal", "R. Alvarez", "Employment Counsel",
               clears=["ADVERSE_IMPACT_INDICATED"],
               comment="Criteria confirmed job-related.")
ledger.approve("executive", "J. Park", "CFO", comment="Authorized.")

ledger.is_fully_approved(package)   # True
```

Whole pipeline in one call:

```bash
python rif_pipeline.py roster.csv --plan rif_plan.yaml \
    --separation-date 2026-10-30 --notice-date 2026-08-19 \
    --leave-policy separate --company-headcount 400 --outdir ./out
```

## The reason this module exists

An approval not bound to *what* was approved is decoration. The common failure
isn't people skipping sign-off — it's that the thing approved and the thing
executed drift apart and nobody notices. Legal signs off on a 17-person list,
someone raises the savings target, three more people appear, and the signature
is still sitting there looking valid.

So every submission is fingerprinted over the decision-relevant content: who's
on the cut list, what they scored, plan parameters, dates, impact verdicts, and
costs. Approvals record that fingerprint.

Demonstrated on the test roster: approving `a76e65b0542e6df1`, then raising the
target, produces `fdb1ebd0d510c8e2` — approvals stop applying, Box 7 refuses to
generate, and `verify()` names what moved:

> The plan has changed since it was submitted for approval. 3 employee(s) added
> to the cut list: E2032, E2065, E2087; Selection plan parameters changed. Prior
> approvals do not carry over; resubmit and re-approve.

The fingerprint is order-independent — sorting the cut list differently doesn't
invalidate anything — so it fires on substance, not formatting.

## What the chain enforces

- **Order.** HR → Legal → Executive. Executive authorization can't be the thing
  that surfaces a legal problem.
- **No self-approval.** The submitter can't sign. Configurable, off by default.
- **No one person signing two stages.** That isn't independent review.
- **Role authorization.** A CFO can't sign the HR stage.
- **Written reasons** on every rejection, revocation, and blocker clearance. An
  unexplained clearance is the record a plaintiff reads aloud.
- **Expiry.** Approvals older than `validity_days` (default 30) go stale, because
  a sign-off against a two-month-old picture isn't a sign-off against today's.
- **Append-only.** Revocation writes a record; it never deletes the approval.
  Superseded versions stay in the history.

## What a stage can clear

Box 5's blockers aren't all the same kind of problem, so each stage declares
what it's competent to clear:

| Stage | Can clear |
|---|---|
| HR | review queue, cut-boundary ties |
| Legal | adverse impact, protected leave, union members, visa holders, selection errors, WARN timing, decisional unit |
| Executive | **nothing** — business urgency is not a legal determination |

And `UNCLEARABLE_CODES` sits outside every stage's remit: missing pay data,
uncomputable severance, undeclared leave policy, missing establishment column.
Signing a form does not supply a pay rate. Attempting it raises.

## Retiring the Box 7 config-string override

Box 7 previously took `counsel_override_by` / `counsel_override_reason` as
config strings. That worked but couldn't bind to anything — the string survived
any change to the plan. Box 7 now accepts a ledger:

```python
DocumentGenerator(cfg).generate(
    compliance=comp, selection=sel, pay=pay, scores=sel.scores,
    approvals=ledger, package=package,
)
```

It verifies the fingerprint, derives the legal sign-off from the actual approval
record, and stamps it into all 23 documents. The config strings remain as a
fallback for use without Box 8, but the ledger is the real path.

## The orchestrator

`rif_pipeline.py` runs the boxes in dependency order and stops at the first
stage that can't honestly proceed, rather than continuing with a degraded result
that would look complete.

Order is 1 → 3 → 4 → 6 → 5 → 8 → 7, which isn't the diagram numbering:

- **4 before 5** — compliance folds the impact finding into its gate.
- **6 before 5** — compliance needs to see whether pay is computable.
- **7 last** — it depends on everything.

Current run on the test roster:

```
| Box 1 — Data Manager       | ok — 141 rows, 136 clean, 7 errors           |
| Box 3 — Selection Criteria | ok — 17 selected, $2,623,750 savings         |
| Box 4 — Adverse Impact     | ok — 1 indicated, 13 flagged                 |
| Box 6 — Severance & Pay    | ok — $666,837 total employer cost            |
| Box 5 — CA Compliance      | ok — WARN not triggered, gate BLOCKED        |
| Box 8 — Approvals          | ok — version 88e28ca7ef04f9ad               |
| Box 7 — Documents          | ok — blocked                                 |
```

`result.write(outdir)` produces every box's artifacts in subdirectories plus a
`PIPELINE_SUMMARY.md`.

## Remaining

Boxes 9 (Task Tracker) and 10 (Audit & Reporting). Both are now well-supplied:
Box 5 emits a dated obligation calendar for 9, and the approval ledger, document
manifest, and per-box reports are the audit trail for 10.
