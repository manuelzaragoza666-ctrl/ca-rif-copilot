# Selection Criteria Engine — California RIF Copilot (Module 2)

Consumes the standardized roster from `workforce_data.py` and produces, per
department, a documented retention score for each employee plus a recommended
cut list sized to a cost savings target.

## Files

| File | Purpose |
|---|---|
| `selection_criteria.py` | The engine. |
| `rif_plan.yaml` | Example per-department plan config. |
| `test_selection_criteria.py` | 42 tests. Runs with or without pytest installed. |

## Quick start

```python
from workforce_data import load_workforce_csv
from selection_criteria import SelectionEngine, load_plan, format_cut_list

roster = load_workforce_csv("roster.csv", as_of="2026-10-30")
plan = load_plan("rif_plan.yaml")
result = SelectionEngine(plan).run(roster.data)

result.cut_list       # recommended selections, each with a rationale
result.scores         # every employee, scored, with the full breakdown
result.review_queue   # cases a human must resolve before the list is valid
print(result.report.to_markdown())
result.write("./out") # five artifacts
```

CLI:

```bash
python selection_criteria.py roster.csv --plan rif_plan.yaml \
    --as-of 2026-10-30 --outdir ./out
```

## The plan config

One entry per department, inheriting from `default` when absent:

```yaml
plan_name: "FY27 Restructuring — Scenario A"
cost_savings_target: 250000      # annualized, fully loaded dollars
burden_multiplier: 1.25          # base pay -> loaded cost
manual_review_below: 15          # scores under this need explicit sign-off
min_comparison_group_size: 2     # below this, ranking isn't a real comparison

departments:
  Engineering:
    mode: individual
    comparison_group: [department, job_level]
    criteria:
      performance:
        kind: performance
        source_column: performance_rating
        weight: 0.5
      critical_skills:
        kind: skills
        source_column: skills
        weight: 0.5
        critical_items: [Python, Kubernetes, Distributed Systems]

  Marketing:
    mode: position               # the role is going away
    eliminate_positions: [Marketing Manager]

  Operations:
    protected_positions: [Warehouse Associate]   # never selectable
    max_headcount: 2                             # cap this dept's reduction
    max_savings_share: 0.15                      # or cap its share of the target
```

## Two modes

**`individual`** — rank employees within a comparison group and select the
lowest scorers. Each criterion is min-max normalized to 0–100 *inside the
comparison group*, then weighted. Weights renormalize to 1 automatically.

**`position`** — eliminate whole job titles; every incumbent goes. Individual
performance is not scored, because it isn't what decided the outcome. The
report reminds you to confirm no retained employee is doing substantially the
same work, which would turn it back into individual selection.

## Scoring

`performance` maps rating labels to numbers via a built-in vocabulary
(`Exceeds` → 4, `Meets` → 3, and ~40 other phrasings) or a scale you pin in the
config. Bare numbers pass through.

`critical_skills` scores coverage of the department's critical items, reading
both `skills` and `certifications`, splitting on `| ; , /` and dropping
parentheticals like `(exp. 2025-04-01)`.

Also available: `numeric` and `ordinal` for any non-protected column.

## The guardrails, and why each exists

These are the parts worth reviewing, because each encodes a decision about
what the engine refuses to do.

- **Protected characteristics are firewalled.** Configuring `age_years`,
  `gender`, `race_ethnicity`, `disability_status`, `veteran_status`, or their
  derivatives as a criterion — or as a comparison-group column — raises
  `SelectionConfigError` at load time. The engine never reads them. Impact is
  measured *after* selection, by Module 3, on this module's output.
- **`leave_status`, `union_flag`, and `visa_status` are also blocked from
  scoring**, then re-attached to the output as `legal_review_flags`. Selecting
  on them directly invites an interference or retaliation claim; knowing about
  them before you send notice is necessary.
- **Missing data never becomes a low score.** An employee with no rating —
  including labels like `New`, `Not Rated`, `Pending` — goes to the review
  queue, not to the bottom of the list.
- **A comparison group of one is not a comparison.** Below
  `min_comparison_group_size`, the employee is routed to manual review with a
  note saying that if the role is truly going away it should be configured as a
  position elimination instead.
- **Ties at the cut boundary are an error, not a coin flip.** If the boundary
  splits a group of equal scores, the engine refuses to break it by row order
  and asks for a documented tie-breaker.
- **Rank-order invariant.** After selection, the engine verifies that nobody is
  on the list while a lower-scored peer in the same group is retained. A
  violation means something other than the stated criteria decided the outcome,
  so it's an ERROR. This check caught a real bug during development: the
  `manual_review_below` threshold originally removed low scorers from the pool,
  which pushed selection *up* the ranking and cut a top performer while the
  lowest scorer stayed. Low scores now flag for approval and keep their place.
- **Degraded comparison groups are reported.** If the plan asks to rank within
  `[department, job_level]` but `job_level` is missing, you get a warning
  rather than a silent cross-level ranking.
- **Pool exhaustion is flagged.** If the target consumes 50%+ of the rankable
  pool, the criteria have stopped selecting and the target is selecting.
- **An unmet target is reported, never forced.**

## Outputs

`result.write()` emits five files:

| File | Contents |
|---|---|
| `*_recommended_cut_list.csv` | Selections, ranked, with rationale, score breakdown, legal flags, and blank `human_decision` / `decision_maker` / `decision_date` / `override_reason` columns |
| `*_scores.csv` | Every employee with score, status, and breakdown |
| `*_review_queue.csv` | Cases blocking a final list |
| `*_report.json` | Machine-readable summary and findings |
| `*_report.md` | Human-readable report |

## Notes on the current configuration

You chose performance ratings plus skills/certifications. Worth knowing:

- **Performance ratings carry rater bias.** The engine emits a
  `PERFORMANCE_RATINGS_IN_USE` note on every department that scores on them.
  Before relying on the output, check whether rating distributions differ
  materially by manager — if they do, the score inherits that inconsistency and
  Module 3 will surface impact you can't explain from the criteria.
- **Critical skills lists are a judgment call made in advance.** Define them
  from documented business need before scoring, not after seeing who has what.
  A list assembled around known individuals is the pattern that makes an
  otherwise objective criterion look pretextual.
- **`SINGLE_CRITERION_SELECTION` fires** if a department ends up with only one
  criterion. Two is better; the config supports any number.

## Module 1 changes

`skills` and `certifications` were added to the canonical schema in
`workforce_data.py` (both `sparse_ok`, so they don't trigger completeness
warnings). Re-run ingestion to pick them up.

## Integration points

- **Module 3 (Adverse Impact Analyzer)** reads `result.scores` — it needs both
  selected and retained employees to compute selection rates, plus the
  protected-class columns this module deliberately ignored.
- **Module 5 (Severance Engine)** reads `result.cut_list`. Note that
  `annual_cost` is *gross* annualized savings; severance, PTO payout, and COBRA
  are Module 5's job and will reduce first-year net savings materially. Don't
  present `achieved_savings` as a net number.

## Scope

This module recommends. It does not decide. Every output carries that framing
deliberately, and the cut list ships with blank override columns because it
isn't finished until a human signs it.
