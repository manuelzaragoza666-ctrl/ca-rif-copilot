# Adverse Impact Analyzer — California RIF Copilot (Box 4)

Reads the scored roster from `selection_criteria.py` and measures whether the
proposed selections fall disproportionately on a protected class.

## Files

| File | Purpose |
|---|---|
| `adverse_impact.py` | The analyzer. |
| `test_adverse_impact.py` | 36 tests, including validation of the statistics. |

## Quick start

```python
from adverse_impact import AdverseImpactAnalyzer

analysis = AdverseImpactAnalyzer().run(selection_result.scores, scenario="Scenario A")

analysis.results               # one row per group comparison
analysis.flagged               # only the comparisons that tripped a threshold
analysis.report.class_verdicts()  # {'Age 40+': 'Impact indicated', ...}
print(analysis.report.to_markdown())
analysis.write("./out")
```

CLI runs the whole pipeline:

```bash
python adverse_impact.py roster.csv --plan rif_plan.yaml --outdir ./out
# exit 1 if impact is indicated anywhere
```

## What it measures

For every protected group, against the **most-favored group** (highest
retention rate among groups large enough to be a stable benchmark):

| Metric | Meaning |
|---|---|
| **Four-fifths ratio** | Retention basis. Below 0.80 indicates impact under the Uniform Guidelines (29 CFR 1607.4(D)). |
| **Selection rate ratio** | Termination basis. At or above 1.25× mirrors the same concern from the other direction. |
| **Fisher's exact p** | Two-tailed, exact — appropriate for the small cell counts a RIF produces. |
| **Standard deviations** | Signed, hypergeometric. Courts have treated 2–3 SD as probative. |
| **Shortfall** | Selections above statistical expectation. |
| **Flip count** | How many individual outcomes would have to differ to change the four-fifths finding. |

Classes analyzed: Age 40+, Sex, Race/Ethnicity, Disability, Veteran Status —
each with its legal authority cited in the finding. Add your own via
`ProtectedClass`.

Each is run **company-wide and inside every department, worksite, and job
level**, because impact is generally assessed where the decision was made.

## Design decisions worth reviewing

**It refuses to recommend swaps.** If a group shows impact, this module will
not tell you who to remove from the cut list. Adjusting a specific person's
outcome because of their protected class is disparate treatment, and it creates
a claim for whoever gets swapped in. The report says so explicitly. Lawful
remedies run through the criteria — job-relatedness, consistency of
application, less discriminatory alternatives — and through counsel.

**Small groups are skipped, not flagged.** An early run produced 25 flags,
mostly from groups of 2–4 compared against a reference group of 1. That noise
buried the one real finding. Now a comparison requires a reference group
meeting `min_group_size` (default 10) and a group of at least 5; everything
else is counted and reported as **untested, not cleared**. A group below the
minimum still gets a statistical test and can be flagged on that basis, but its
four-fifths ratio alone won't trigger "Impact indicated."

**Two practical screens, because one has a blind spot.** When overall selection
rates are low, the retention-basis four-fifths ratio compresses toward 1.0. The
test roster hit this exactly: Age 40+ selected at 16.9% vs 10.0% — a 1.69×
disparity — yet the four-fifths ratio read 0.92 and passed. The termination-rate
screen catches it, and `FOUR_FIFTHS_UNDERSTATES_DISPARITY` explains the
divergence rather than leaving you to reconcile two numbers.

**Fragility is reported.** A flag that disappears if one person's outcome
differed is weak evidence — and so is the clean result sitting next to it. The
flip count makes that visible instead of implied.

**Aggregate masking is called out per class.** On the test roster, Age 40+
passes company-wide at 0.92 but fails in Engineering at 0.78.
`AGGREGATE_MASKS_UNIT_IMPACT` fires precisely on that pattern.

**Undisclosed is never a group.** Employees who declined to state are excluded
from the test and counted separately. Above 20% undisclosed, the report says
the "no flag" result carries little weight.

**Out-of-pool employees are excluded.** People who could never have been
selected would dilute every rate and hide real disparities.

**Fisher's exact is implemented directly** via log-gamma, so there's no hard
SciPy dependency. Validated against `scipy.stats.fisher_exact` across 400
random tables — max deviation 1e-13.

## Privilege

Adverse impact analyses are commonly run at the direction of counsel so results
are covered by attorney-client privilege. A self-serve analysis in a shared
drive is discoverable, and if it shows impact that wasn't acted on, it becomes
the most damaging document in the case. The markdown report carries a privilege
header by default. **Talk to your employment counsel about how this output
should be generated, labeled, and stored before running it on a real scenario.**

## Known limitations

- **No intersectional analysis.** Age×sex, race×age and similar combinations
  aren't tested; sample sizes usually collapse below interpretability in a
  single-company RIF. If you need it, the cell counts must be large enough to
  mean something.
- **Multiple comparisons.** Running dozens of tests means some cross p<.05 by
  chance. The report notes this; it does not apply a Bonferroni or FDR
  correction, because the appropriate correction is a judgment call that
  belongs with counsel and a statistician.
- **The four-fifths rule is a screening heuristic**, not a legal standard.
  Passing it is not a defense and failing it is not liability.
- **Rater consistency checking is coarse.** It flags managers whose selection
  rates diverge by 2+ SD, which needs a reasonable number of reports per
  manager to detect anything.

## What this cannot tell you

Whether discrimination occurred. This is statistical screening. It measures
outcomes, not intent, not job-relatedness, not business necessity — the things
that actually decide a disparate impact case. A clean report is not clearance
and a flagged report is not liability. Both are inputs to a legal analysis that
a lawyer has to do.

## Integration

- **Reads** `SelectionResult.scores` — it needs both selected and retained
  employees, plus the protected-class columns Module 3 deliberately ignored
  during scoring.
- **Feeds** Box 5 (CA Compliance) and Box 10 (Audit & Reporting): the
  comparison table is a compliance artifact and belongs in the audit trail.
- **Should gate** Box 7 (Document Generation). Notices shouldn't be generated
  from a scenario with unresolved `ADVERSE_IMPACT_INDICATED` findings.
