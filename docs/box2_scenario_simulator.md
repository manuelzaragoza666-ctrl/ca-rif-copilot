# Scenario Simulator — California RIF Copilot (Box 2)

Runs several restructuring scenarios against the same roster and compares their
financial, operational, and compliance consequences side by side. Each scenario
is a full RIF plan; the simulator runs Box 3 (Selection) and Box 4 (Adverse
Impact) on every one and adds its own analysis on top.

## Files

| File | Purpose |
|---|---|
| `scenario_simulator.py` | The simulator. |
| `scenarios.yaml` | Example scenario set — four variants over a shared base plan. |
| `test_scenario_simulator.py` | 37 tests. |

## Quick start

```python
from scenario_simulator import ScenarioSimulator, load_scenarios

scenarios = load_scenarios("scenarios.yaml")
sim = ScenarioSimulator().run(roster.data, scenarios)

sim.comparison            # one row per scenario
sim.by_name("A — Broad reduction")   # full outcome for one
print(sim.report.to_markdown())
sim.write("./out")        # comparison + complete artifacts per scenario
```

```bash
python scenario_simulator.py roster.csv --scenarios scenarios.yaml --outdir ./out
```

## Scenario config

`base` supplies the shared plan; each scenario deep-merges its overrides, so
variants stay readable and the difference between them is explicit.

```yaml
base:
  as_of_date: 2026-10-30
  departments: { ... }

scenarios:
  - name: "A — Broad reduction"
    rationale: >
      Board-approved $2.5M annualized reduction, spread across all functions.
      No function is being exited.
    plan:
      cost_savings_target: 2500000

  - name: "B — Exit brand function"
    rationale: >
      Brand work moves to an agency under a contract signed in Q3, removing the
      in-house role entirely.
    plan:
      cost_savings_target: 2500000
      departments:
        Marketing:
          mode: position
          eliminate_positions: [Marketing Associate]
```

## What it measures

**Financial** — annualized savings, provisional one-time separation cost
(severance, accrued vacation payout, COBRA, admin), first-year net, payback
period, cost per head.

**Operational** — headcount by department and worksite, managers lost and
orphaned reports, critical-skill coverage after the cut, single points of
failure, tenure lost, median tenure before and after.

**Compliance** — adverse impact verdict per protected class, review-queue size,
selections needing counsel sign-off, unresolved selection errors.

## The design question this box raises

Comparing scenarios is legitimate and, in the disparate impact context,
affirmatively useful: if one approach meets the same business need with less
impact, that is a **less discriminatory alternative**, and considering it is
exactly what a disparate impact analysis calls for.

The same mechanism makes something else easy — iterating criteria while
watching protected-class numbers move, and stopping when they look acceptable.
That is choosing criteria for their demographic output rather than their
business rationale, which is a different act legally, and it produces a
discoverable record of itself.

So the module is built to support the first and resist the second:

- **Every scenario requires a written business `rationale`.** Constructing a
  `Scenario` without one raises. Record it when you build the scenario, not
  after you see the result.
- **No composite score, no "recommended scenario."** Collapsing savings,
  operational risk, and adverse impact into one number implies they trade off
  on a common scale. They don't, and a court wouldn't treat them that way. A
  test asserts the comparison table contains no ranking column.
- **`WEIGHT_CHANGE_MOVES_DEMOGRAPHICS`** fires when scenarios differ *only* in
  criteria weights but produce materially different demographic outcomes. The
  warning explains the distinction rather than accusing — weights legitimately
  change who is selected — but flags that a weight chosen after seeing the
  shift is a weight chosen for its effect.
- **Every scenario is logged, including discarded ones.** The report says so on
  its face. An unexplained gap in the sequence is harder to defend than a
  documented decision to move on.

## Bugs the build surfaced

**"0 managers lost" was a false all-clear.** When `manager_id` values don't
resolve to employees in the roster, the org graph can't be walked — but the
count still read 0, which looks like good news. It now reports `n/a` and raises
`ORG_STRUCTURE_UNRESOLVABLE`, because unmeasured is not the same as clean.

**The tuning guardrail measured the wrong thing, twice.** First it compared raw
selection rates, which barely move under a weight change — on the test roster
the Sex disparity ratio swung 1.40× → 2.30× while the rate moved two points. It
now compares the disparity itself. Then a second failure: comparing only the
worst-off group per class meant that two scenarios cutting *opposite*
demographic groups both reported an index of 1.25 and the check saw no change.
It now indexes per group and detects when the disadvantaged group flips
identity — the loudest version of the signal, and the one it was blind to.

**The disparity ratio is undefined when the reference group has zero
selections**, which is common in small RIFs. `_disparity_index` falls back to
the inverse four-fifths ratio, which is defined wherever the reference group
retained anyone and sits on the same 1.0-at-parity scale.

## Cost model caveat

`CostAssumptions` is provisional. Box 6 (Severance & Pay Engine) owns the real
calculation; these figures exist so scenarios can be compared on a consistent
basis, and every output that uses them is labeled an estimate. Defaults: 2
weeks per year of service (4-week floor, 26-week cap), 3 months COBRA at
$1,400/month, $1,500 admin per employee, accrued vacation paid out per Labor
Code 227.3.

**Do not present `first_year_net` to a CFO as a final number.**

## Integration

- **Reads** the standardized roster from Box 1 and runs Boxes 3 and 4 per
  scenario.
- **Writes** per-scenario subdirectories containing the full selection and
  adverse impact artifacts, so a chosen scenario arrives at Box 7 already
  documented.
- **Feeds** Box 5 (CA Compliance) — headcount and worksite reductions per
  scenario are the inputs to WARN threshold analysis, which this module
  deliberately does not attempt.
