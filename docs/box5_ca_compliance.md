# CA Compliance Engine — California RIF Copilot (Box 5)

Takes a proposed selection and works out what the law requires before it can be
carried out: whether WARN is triggered, when notice must go out, what the notice
must contain, what is owed on the final paycheck and when, and which employees
carry conditions needing individual legal handling.

It also produces the gate Box 7 depends on. `result.gate.may_generate_documents`
reports whether notices may be generated and, if not, exactly what is blocking.

## Files

| File | Purpose |
|---|---|
| `ca_compliance.py` | The engine. |
| `test_ca_compliance.py` | 61 tests. |

## Quick start

```python
from ca_compliance import ComplianceConfig, ComplianceEngine

cfg = ComplianceConfig(
    proposed_separation_date="2026-10-30",
    notice_date="2026-08-19",
    total_company_headcount=400,
    service_coordination="lwdb",          # SB 617 election
    lwdb_email="board@example.gov",
    lwdb_phone="(555) 555-0100",
    employer_contact_email="hr@acme.com",
    employer_contact_phone="(555) 555-0199",
)
result = ComplianceEngine(cfg).run(
    selection.scores, impact=analysis, selection=selection
)

result.gate.may_generate_documents   # the Box 7 gate
result.calendar                      # dated obligations
result.employee_flags                # per-employee legal review items
print(result.report.to_markdown())
```

```bash
python ca_compliance.py roster.csv --plan rif_plan.yaml \
    --separation-date 2026-10-30 --notice-date 2026-08-19 --outdir ./out
# exit 1 if the document gate is blocked
```

## Coverage

| Area | Authority |
|---|---|
| Cal-WARN triggers, notice timing | Lab. Code §§ 1400.5, 1401 |
| SB 617 notice content (eff. 2026-01-01) | Lab. Code § 1401(c)–(e) |
| Federal WARN, run in parallel | 29 U.S.C. § 2101 |
| Final pay timing, waiting-time penalty | Lab. Code §§ 201, 203 |
| Accrued vacation payout | Lab. Code § 227.3 |
| OWBPA consideration, revocation, disclosure | 29 U.S.C. § 626(f) |
| EDD change notice, DE 2320 | Unemp. Ins. Code § 1089 |
| COBRA / Cal-COBRA, HIPP | 29 U.S.C. § 1166; H&S Code §§ 1366.5, 1366.20 |
| Protected leave, CBA, work visa | CFRA, FMLA, NLRA |

**Verified against current sources during the build** — including EDD Workforce
Services Information Notice WSIN25-14 (2026-01-06) for SB 617, since those
requirements post-date my training data.

## Thresholds, and why California is the binding constraint

| | California | Federal |
|---|---|---|
| Employer/site coverage | 75+ at the establishment (12-month lookback, part-time counts) | 100+ employees (part-time excluded) |
| Mass layoff | 50+ in 30 days | 500+, or 50–499 if ≥ ⅓ of the site |
| Percentage test | **none** | one-third |
| Aggregation window | 30 days | 90 days |
| Employee counting | 6 of preceding 12 months | — |
| Exceptions | narrow — no unforeseeable-business-circumstances; faltering-company unavailable for mass layoffs | broader |

The absence of a percentage test is the practical difference: 50 people out of
a 500-person site clears Cal-WARN and misses federal entirely.

## Design decisions worth reviewing

**It computes whether a threshold is met; it does not help anyone stay under
one.** No output suggests splitting, staggering, or reclassifying to avoid
coverage, and a test asserts that language never appears. Where a reduction
lands within 5 employees of the trigger, `NEAR_WARN_THRESHOLD` says to treat it
as triggered for planning and explains why: Cal-WARN aggregates across any
30-day period, so *any* further separation at that site — including a
performance termination for unrelated reasons — can trigger notice
retroactively.

**Aggregation is checked against prior rounds.** Supply `prior_layoffs` and the
engine combines them. On the test case, 30 + 25 = 55: neither round reaches 50,
together they do, and notice was already required 60 days before the earlier
separations. That fires as an ERROR.

**SB 617 content deficiencies block the gate independently of timing.** A
notice delivered on time that omits any of the four required disclosures does
not satisfy § 1401, and each day of deficient notice is a separate violation.
The engine checks all four against config and blocks if any is unconfigured.

**Coverage is assessed per establishment, not company-wide.** Two sites cutting
30 each do not combine into a 60-person mass layoff — and a test enforces that,
because getting this wrong in either direction is expensive.

**The 6-of-12-months rule is applied to the threshold count only.** Short-tenure
employees don't count toward 50, but they still receive final pay, EDD notices,
and COBRA. The report says so explicitly rather than leaving the exclusion to be
misread as a general exemption.

**It refuses to invent a final-wages figure.** An earlier version multiplied a
daily rate by an arbitrary five days and printed it as "wages through
separation." That looked authoritative and was made up. Wages earned depend on
the pay period and days actually worked, which this module doesn't know — it now
reports *not computed* and says payroll must supply it. The vacation payout and
the § 203 penalty exposure are computed, because those genuinely can be.

**A blank vacation balance is not zero.** § 227.3 vests vacation; a missing
value gets a warning directing confirmation to payroll.

**The OWBPA decisional unit is not assumed to be the scoring comparison group.**
They are frequently different, defining it is a legal judgment, and the
disclosure of job titles and ages is precisely what tells a plaintiff's lawyer
whether the selection skewed by age.

## The gate

`may_generate_documents` is False if any of these hold:

- Any ERROR-severity compliance finding
- A missed statutory deadline
- Adverse impact **indicated** by Box 4
- Unresolved selection errors from Box 3
- Missing establishment column (coverage undetermined — it blocks rather than guessing)
- Employees on protected leave or in a bargaining unit on the cut list

Warnings that don't block but are surfaced: no Box 4 analysis supplied, a
non-empty selection review queue, flagged-but-not-indicated impact comparisons.

## Limitations

- **Waiting-time exposure is a ceiling, not a prediction.** It assumes the
  maximum 30 days for every employee.
- **Local ordinances aren't covered.** Several California cities have their own
  worker retention and severance rules.
- **CBA terms aren't read.** The engine flags union members; someone has to open
  the agreement, and its seniority or bumping provisions may override the
  selection criteria entirely.
- **Cal-WARN's 12-month lookback can't be verified from a current roster.** A
  site that has shrunk below 75 may still be covered; the report says so rather
  than implying the determination is complete.
- **Relocation distance is a config input**, not computed from addresses.

## Not legal advice

This is screening built from public statutory text. Statutes get amended —
Cal-WARN's notice content changed on 2026-01-01, mid-project — agencies issue
guidance, and courts interpret both. Every determination is an input to a
lawyer's analysis, not a substitute for one. Verify current requirements with
employment counsel before acting.

## Integration

- **Reads** `SelectionResult.scores`, and optionally the Box 3 and Box 4 result
  objects to fold their unresolved findings into the gate.
- **Gates** Box 7. Document generation should check
  `ComplianceResult.gate.may_generate_documents` before producing anything.
- **Feeds** Box 9 (Task Tracker) with the dated obligation calendar and Box 10
  (Audit) with the full determination record.
- **Box 6** (Severance & Pay) will supersede the provisional cost figures in
  Box 2 and should reuse this module's final-pay timing rules.
