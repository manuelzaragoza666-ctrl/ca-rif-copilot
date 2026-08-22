# California RIF Copilot

A modular Python system for planning a reduction in force under California law:
data ingestion, selection scoring, adverse impact testing, scenario comparison,
compliance determination, severance calculation, document drafting, structured
approvals, and execution tracking.

All ten boxes are built. **438 unit tests, 64 end-to-end pipeline checks.**

---

## Layout

```
ca-rif-copilot/
├── rif.py                  command-line entry point
├── run_tests.py            runs every suite + the pipeline harness
├── requirements.txt
├── rif_copilot/            the package — one module per box
├── tests/                  438 unit tests, one suite per box
├── tools/                  synthetic roster generator, pipeline harness
├── examples/               example plans, scenarios, rosters
└── docs/                   per-box design notes
```

## Install and run

```bash
pip install -r requirements.txt

# whole pipeline, one command
python rif.py pipeline examples/test_roster.csv \
    --plan examples/plan_test.yaml \
    --separation-date 2026-10-30 --notice-date 2026-08-19 \
    --leave-policy separate --company-headcount 400 --outdir ./out

# or a single box
python rif.py impact examples/test_roster.csv --plan examples/plan_test.yaml
python rif.py --help
```

As a library:

```python
from rif_copilot import PipelineConfig, run_pipeline

result = run_pipeline(PipelineConfig(
    roster_csv="roster.csv", plan_yaml="rif_plan.yaml",
    separation_date="2026-10-30", leave_policy="separate",
))
print(result.summary_markdown())
result.write("./out")
```

## Tests

```bash
python run_tests.py                # 438 unit tests + 64 pipeline checks
python run_tests.py compliance     # one suite
python tests/test_approvals.py     # standalone, no framework needed
pytest                             # also works if installed
```

Every suite runs without pytest via a built-in shim. Dependencies are pandas,
numpy, and PyYAML — Fisher's exact test is implemented directly with `lgamma`,
so there is no SciPy requirement and the system installs cleanly on ARM.

Generate a fresh synthetic roster with seeded edge cases:

```bash
python tools/make_test_roster.py --employees 140 --seed 7 \
    --out examples/test_roster.csv
python tools/verify_test_run.py examples/test_roster.csv examples/plan_test.yaml
```

---

## The boxes

| Box | Module | Tests | What it does |
|---|---|---|---|
| 1 | `workforce_data.py` | 28 | Ingest, validate, normalize; derive tenure, age, pay |
| 2 | `scenario_simulator.py` | 37 | Compare scenarios on cost, operations, compliance |
| 3 | `selection_criteria.py` | 42 | Score employees, produce a recommended cut list |
| 4 | `adverse_impact.py` | 36 | Four-fifths, Fisher's exact, SD analysis, per unit |
| 5 | `ca_compliance.py` | 61 | Cal-WARN, SB 617, final pay, OWBPA; **the gate** |
| 6 | `severance_pay.py` | 46 | Severance, vacation payout, withholding, cash flow |
| 7 | `document_generator.py` | 57 | Notices, letters, disclosures, scripts — all drafts |
| 8 | `approvals.py` | 42 | HR → Legal → Exec, bound to a version fingerprint |
| 9 | `task_tracker.py` | 48 | Dated tasks, acknowledgments, execution guardrails |
| 10 | `audit_reporting.py` | 41 | Hash-chained decision record, completeness, reports |
| — | `pipeline.py` | — | Orchestrator; runs 1→3→4→6→5→8→7→9→10 |

Per-box design notes are in `docs/`.

Run order isn't diagram order: box 4 runs before 5 because compliance folds the
impact finding into its gate; box 6 before 5 so compliance can see whether pay
is computable; box 7 near-last because it depends on everything.

Per-box detail is in `README_box1.md` through `README_box8.md`.

---

## What the system refuses to do

These are the decisions worth reviewing. Each is enforced by a test.

**It won't score on protected characteristics.** Configuring age, sex, race,
disability, or veteran status as a selection criterion raises at load time.
Impact is measured *after* selection, by box 4, on box 3's output.

**It won't tell you who to swap to fix a ratio.** If a group shows adverse
impact, removing someone because of their protected class is disparate treatment
and creates a claim for whoever is swapped in. Box 4 surfaces the disparity and
routes it to counsel; remedies run through the criteria.

**It won't help you stay under a WARN threshold.** Box 5 computes whether a
threshold is met. No output suggests splitting, staggering, or reclassifying to
avoid coverage — a test asserts that language never appears. Within five people
of the trigger, it says treat it as triggered and explains why.

**It won't rank scenarios.** Box 2 compares but produces no composite score,
because savings, operational risk, and adverse impact don't share a scale.

**It won't produce a signable release.** Box 7 emits a skeleton for counsel with
OWBPA elements enumerated and prohibited terms listed. An ADEA release missing an
OWBPA element stays enforceable as to everything *except* the age claim.

**It won't guess at ambiguous facts.** Undeclared leave policy → refuses to
compute. Missing decisional unit → no OWBPA disclosure. Unfillable values →
visible `[[PLACEHOLDER]]` tokens. Final wages → *not computed*, payroll supplies
it.

**It won't let a signature substitute for data.** Box 8's `UNCLEARABLE_CODES`
sits outside every approval stage's remit: signing a form does not supply a
missing pay rate. Box 9 applies the same rule to task completion.

**It won't pressure someone during a consideration period.** Box 9 schedules no
follow-up task inside an OWBPA window. Prompting for a signature during the
period the statute provides for deliberation undermines the voluntariness it
exists to protect.

**It won't produce a sanitized record.** Box 10's exports cannot omit adverse
findings — every error appears in every report, and there is no filter argument.
A curated audit package is the worst artifact to have created: discoverable,
demonstrably incomplete, and the omission does more damage than the finding it
hides. Entries are hash-chained, so editing or deleting one is detectable and
reported with its position.

**It won't claim a record is complete when it isn't.** Box 10 assesses its own
gaps and writes them into the log as errors. A record showing 87.5% completeness
with the approval gap named is more useful than one asserting completeness.

---

## Bugs the build surfaced

Each shaped a design:

- **Ranking inversion.** Box 3's `manual_review_below` threshold pulled low
  scorers out of the pool, which pushed selection *up* the ranking — a run cut a
  100-scoring senior engineer while the 0-scoring peer stayed. Now flagged for
  approval in place, with a rank-order invariant that errors if anyone is
  selected while a lower-scored peer is retained.
- **Four-fifths compression.** Age 40+ was selected at 16.9% vs 10.0% — a 1.69×
  disparity — and the four-fifths ratio read **0.92, passing**. At low selection
  rates the retention-basis ratio goes numb. Box 4 now runs a termination-rate
  screen alongside it.
- **Aggregate masking.** Company-wide, Age 40+ passes at 0.92; in Engineering it
  fails at 0.78. Impact is assessed where the decision was made.
- **A guardrail measuring the wrong thing, twice.** Box 2's tuning detector first
  compared raw selection rates (which barely move), then compared only the
  worst-off group per class — so two scenarios cutting *opposite* demographic
  groups both read 1.25 and it saw nothing. Now indexes per group and names the
  flip.
- **False all-clear.** Box 2 reported "0 managers lost" when manager IDs didn't
  resolve. Now `n/a` with a warning: unmeasured is not clean.
- **Invented precision.** Box 5 printed "wages through separation" from an
  arbitrary five-day multiplier. Removed.
- **Box 2 understating cost by 7.1%** — no employer FICA match on severance.
  Found by reconciling against box 6.

---

## Verified against current sources, not memory

- **SB 617** (eff. 2026-01-01) added four required Cal-WARN notice disclosures.
  Verified against EDD notice WSIN25-14. A timely notice omitting any of them is
  an independent violation, so the gate blocks on each separately.
- **CA supplemental withholding on severance is 6.6%**, not the 10.23% bonus
  rate. Sources genuinely conflict; DE 44 distinguishes them. Configurable, and
  the report names which rate it used.
- **CUIC § 1265**: true dismissal severance isn't subject to SDI and isn't wages
  for UI. The same money labeled "wages in lieu of notice" is both.

---

## Limits

This is screening and drafting support. It is not legal, tax, or payroll advice.
Every determination is an input to a lawyer's analysis, and statutes change —
Cal-WARN's notice content changed mid-project.

Adverse impact analyses are normally run at the direction of counsel so they're
privileged. A self-serve analysis in a shared drive is discoverable, and one
showing impact that wasn't acted on is the most damaging document in the case.
Reports carry privilege headers, but labeling and storage are counsel's call.

The data this system holds — salaries, birth dates, protected class fields, a
ranked cut list, privileged analyses — belongs on managed, encrypted,
access-logged infrastructure. Not a laptop, not an unmanaged device.

**Not covered:** local ordinances (several California cities have their own
worker retention rules), CBA terms (flagged, not read), intersectional impact
analysis, multi-state reductions beyond flagging them.

---

## Status

All ten boxes are built and wired. A single command runs the full chain and
produces every artifact:

```
| Box 1 — Data Manager       | 141 rows, 136 clean, 7 errors                |
| Box 3 — Selection Criteria | 17 selected, $2,623,750 savings              |
| Box 4 — Adverse Impact     | 1 indicated, 13 flagged                      |
| Box 6 — Severance & Pay    | $666,837 total employer cost                 |
| Box 5 — CA Compliance      | WARN not triggered, gate BLOCKED             |
| Box 8 — Approvals          | version 88e28ca7ef04f9ad                     |
| Box 7 — Documents          | blocked                                      |
| Box 9 — Task Tracker       | 140 tasks, 51 acknowledgment slots           |
| Box 10 — Audit & Reporting | 245 entries, chain intact                    |
```

Natural next steps, none of them blocking: HRIS connectors feeding
`load_workforce_dataframe()` so ingestion isn't CSV-only; a city-level minimum
wage table keyed on `work_city`; intersectional impact analysis where cell
counts support it; and a real security review before any of this touches
production employee data.
