"""
scenario_simulator.py
=====================

Scenario Simulator for the California RIF Copilot (box 2).

Runs several restructuring scenarios against the same roster and compares their
financial, operational, and compliance consequences side by side.

Each scenario is a full RIF plan. The simulator runs the Selection Criteria
Engine and the Adverse Impact Analyzer on every one, then adds:

* **Financial impact** — annualized savings, estimated one-time separation
  cost, first-year net, and payback period.
* **Operational impact** — headcount by department, critical-skill coverage
  after the reduction, single points of failure, manager and span-of-control
  changes, and institutional tenure lost.
* **Compliance signals** — adverse impact verdicts, review-queue size, and
  employees whose selection needs counsel sign-off.

A word about what this tool makes easy
--------------------------------------
Comparing scenarios is legitimate and, in the disparate impact context,
affirmatively useful: if a scenario meets the same business need with less
impact, that is a less discriminatory alternative and the law expects an
employer to consider it.

The same mechanism makes something else easy, which this module is built to
resist. Iterating criteria while watching protected-class numbers move — and
stopping when they look acceptable — is choosing criteria for their demographic
output rather than their business rationale. That is a different act from
choosing criteria on the merits and then measuring impact, and courts treat it
differently. It also produces a discoverable record of exactly that process.

So this module:

* requires a written business ``rationale`` on every scenario, recorded before
  the results are seen rather than reconstructed afterward;
* refuses to compute a composite "best scenario" score, because collapsing
  savings and adverse impact into one number implies they trade off against
  each other on a common scale, and they do not;
* flags when scenarios differ only in criteria weights yet produce materially
  different demographic outcomes, since that is the pattern that looks like
  tuning whether or not it was;
* logs every scenario run, including the ones you discard.

Discard nothing quietly. The scenario you ran and abandoned is discoverable,
and an unexplained gap in the sequence is worse than a documented decision.

Usage
-----
    from .scenario_simulator import Scenario, ScenarioSimulator, load_scenarios

    scenarios = load_scenarios("scenarios.yaml")
    sim = ScenarioSimulator().run(roster_df, scenarios)

    sim.comparison        # one row per scenario
    print(sim.report.to_markdown())

CLI
---
    python scenario_simulator.py roster.csv --scenarios scenarios.yaml --outdir ./out
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .adverse_impact import AdverseImpactAnalyzer, AdverseImpactResult
from .selection_criteria import (
    RifPlan,
    SelectionConfigError,
    SelectionEngine,
    SelectionResult,
    load_plan,
    plan_from_dict,
    split_items,
)
from .workforce_data import Severity

__all__ = [
    "Scenario",
    "CostAssumptions",
    "FinancialImpact",
    "OperationalImpact",
    "ScenarioOutcome",
    "ScenarioSimulator",
    "SimulationResult",
    "load_scenarios",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------


@dataclass
class CostAssumptions:
    """Provisional separation-cost model.

    Box 6 (Severance & Pay Engine) owns the real calculation. These figures
    exist so scenarios can be compared on a consistent basis, and every output
    that uses them is labeled as an estimate. Do not put these numbers in front
    of a CFO as final.
    """

    #: Weeks of base pay per year of service.
    severance_weeks_per_year: float = 2.0
    #: Floor and cap on severance weeks, regardless of tenure.
    severance_min_weeks: float = 4.0
    severance_max_weeks: float = 26.0
    #: Employer-subsidized COBRA months assumed in the package.
    cobra_months: float = 3.0
    #: Monthly employer cost per covered employee.
    cobra_monthly_cost: float = 1_400.0
    #: Flat per-employee administrative and outplacement cost.
    admin_cost_per_employee: float = 1_500.0
    #: Accrued vacation must be paid out at separation (Labor Code 227.3).
    pay_out_accrued_vacation: bool = True
    #: Employer-side payroll tax on separation pay (FICA match). Omitting this
    #: understated total cost by ~7% against the Severance & Pay Engine.
    employer_payroll_tax_rate: float = 0.0765

    def to_dict(self) -> dict[str, Any]:
        return {
            "severance_weeks_per_year": self.severance_weeks_per_year,
            "severance_min_weeks": self.severance_min_weeks,
            "severance_max_weeks": self.severance_max_weeks,
            "cobra_months": self.cobra_months,
            "cobra_monthly_cost": self.cobra_monthly_cost,
            "admin_cost_per_employee": self.admin_cost_per_employee,
            "pay_out_accrued_vacation": self.pay_out_accrued_vacation,
            "employer_payroll_tax_rate": self.employer_payroll_tax_rate,
        }


@dataclass
class Scenario:
    """One restructuring scenario: a plan plus the reason it exists."""

    name: str
    plan: RifPlan
    #: The business reason this variant is being modeled. Required.
    rationale: str = ""
    #: Free-form notes carried into the report and the audit trail.
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.rationale.strip():
            raise SelectionConfigError(
                f"Scenario {self.name!r} has no rationale. Every scenario needs a "
                f"written business reason recorded before its results are seen — "
                f"a rationale reconstructed after the fact is worth very little, "
                f"and the sequence of scenarios you ran is discoverable."
            )


# ---------------------------------------------------------------------------
# Impact structures
# ---------------------------------------------------------------------------


@dataclass
class FinancialImpact:
    annualized_savings: float = 0.0
    severance_cost: float = 0.0
    vacation_payout: float = 0.0
    cobra_cost: float = 0.0
    admin_cost: float = 0.0
    employer_payroll_tax: float = 0.0
    headcount_reduction: int = 0

    @property
    def one_time_cost(self) -> float:
        return round(
            self.severance_cost + self.vacation_payout + self.cobra_cost
            + self.admin_cost + self.employer_payroll_tax,
            2,
        )

    @property
    def first_year_net(self) -> float:
        return round(self.annualized_savings - self.one_time_cost, 2)

    @property
    def payback_months(self) -> float | None:
        if self.annualized_savings <= 0:
            return None
        return round(self.one_time_cost / (self.annualized_savings / 12.0), 1)

    @property
    def cost_per_head(self) -> float | None:
        if not self.headcount_reduction:
            return None
        return round(self.one_time_cost / self.headcount_reduction, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "annualized_savings": round(self.annualized_savings, 2),
            "severance_cost": round(self.severance_cost, 2),
            "vacation_payout": round(self.vacation_payout, 2),
            "cobra_cost": round(self.cobra_cost, 2),
            "admin_cost": round(self.admin_cost, 2),
            "employer_payroll_tax": round(self.employer_payroll_tax, 2),
            "one_time_cost": self.one_time_cost,
            "first_year_net": self.first_year_net,
            "payback_months": self.payback_months,
            "cost_per_head": self.cost_per_head,
            "headcount_reduction": self.headcount_reduction,
        }


@dataclass
class OperationalImpact:
    headcount_before: int = 0
    headcount_after: int = 0
    department_reduction: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: None means the org graph could not be resolved — not that nothing was
    #: lost. The distinction matters; see ORG_STRUCTURE_UNRESOLVABLE.
    managers_lost: int | None = 0
    orphaned_reports: int | None = 0
    tenure_years_lost: float = 0.0
    median_tenure_before: float | None = None
    median_tenure_after: float | None = None
    skill_gaps: list[dict[str, Any]] = field(default_factory=list)
    single_points_of_failure: list[dict[str, Any]] = field(default_factory=list)
    worksite_reduction: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def reduction_pct(self) -> float:
        if not self.headcount_before:
            return 0.0
        return (self.headcount_before - self.headcount_after) / self.headcount_before

    def to_dict(self) -> dict[str, Any]:
        return {
            "headcount_before": self.headcount_before,
            "headcount_after": self.headcount_after,
            "reduction_pct": round(self.reduction_pct, 4),
            "managers_lost": self.managers_lost,
            "orphaned_reports": self.orphaned_reports,
            "tenure_years_lost": round(self.tenure_years_lost, 1),
            "median_tenure_before": self.median_tenure_before,
            "median_tenure_after": self.median_tenure_after,
            "skill_gaps": self.skill_gaps,
            "single_points_of_failure": self.single_points_of_failure,
            "department_reduction": self.department_reduction,
            "worksite_reduction": self.worksite_reduction,
        }


@dataclass
class ScenarioOutcome:
    """Everything known about one scenario after it has been run."""

    scenario: Scenario
    selection: SelectionResult
    impact: AdverseImpactResult
    financial: FinancialImpact
    operational: OperationalImpact
    findings: list[dict[str, Any]] = field(default_factory=list)

    @property
    def impact_indicated(self) -> int:
        return len(self.impact.report.indicated)

    @property
    def review_queue_size(self) -> int:
        return len(self.selection.review_queue)

    @property
    def legal_review_count(self) -> int:
        cut = self.selection.cut_list
        if cut.empty or "legal_review_flags" not in cut.columns:
            return 0
        return int((cut["legal_review_flags"].astype("string").fillna("") != "").sum())

    @property
    def selection_errors(self) -> int:
        return len([f for f in self.selection.report.findings
                    if f.severity == Severity.ERROR])

    def summary_row(self) -> dict[str, Any]:
        fin = self.financial.to_dict()
        return {
            "scenario": self.scenario.name,
            "rationale": self.scenario.rationale,
            "target": self.scenario.plan.cost_savings_target,
            "headcount_reduction": self.operational.headcount_before
            - self.operational.headcount_after,
            "reduction_pct": round(self.operational.reduction_pct, 4),
            "annualized_savings": fin["annualized_savings"],
            "target_met": self.selection.report.target_met,
            "one_time_cost": fin["one_time_cost"],
            "first_year_net": fin["first_year_net"],
            "payback_months": fin["payback_months"],
            "managers_lost": self.operational.managers_lost,
            "orphaned_reports": self.operational.orphaned_reports,
            "skill_gaps": len(self.operational.skill_gaps),
            "single_points_of_failure": len(self.operational.single_points_of_failure),
            "median_tenure_after": self.operational.median_tenure_after,
            "impact_indicated": self.impact_indicated,
            "impact_flagged": len(self.impact.report.flagged),
            "review_queue": self.review_queue_size,
            "legal_review_selections": self.legal_review_count,
            "selection_errors": self.selection_errors,
        }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SimulationReport:
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
    roster_size: int = 0
    as_of_date: str | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)
    outcomes: list[ScenarioOutcome] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, scenario: str | None = None) -> None:
        self.findings.append({
            "severity": severity, "code": code, "message": message,
            "scenario": scenario,
        })

    def comparison(self) -> pd.DataFrame:
        if not self.outcomes:
            return pd.DataFrame()
        return pd.DataFrame([o.summary_row() for o in self.outcomes])

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "roster_size": self.roster_size,
            "as_of_date": self.as_of_date,
            "assumptions": self.assumptions,
            "scenarios": [
                {
                    "name": o.scenario.name,
                    "rationale": o.scenario.rationale,
                    "notes": o.scenario.notes,
                    "summary": o.summary_row(),
                    "financial": o.financial.to_dict(),
                    "operational": o.operational.to_dict(),
                    "impact_verdicts": o.impact.report.class_verdicts(),
                }
                for o in self.outcomes
            ],
            "findings": self.findings,
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self) -> str:
        L: list[str] = []
        L.append("# Scenario Comparison")
        L.append("")
        L.append(
            "> **Prepared at the direction of counsel / privileged and "
            "confidential — confirm labeling before circulating.** This document "
            "records every scenario modeled, including those not pursued."
        )
        L.append("")
        L.append(f"**Generated:** {self.generated_at}  ")
        L.append(f"**Roster:** {self.roster_size} employees  ")
        L.append(f"**As of:** {self.as_of_date}  ")
        L.append("")

        if not self.outcomes:
            L.append("No scenarios were run.")
            return "\n".join(L)

        # -- financial ----------------------------------------------------
        L.append("## Financial impact")
        L.append("")
        L.append("| Scenario | Cut | Annualized savings | Target met | "
                 "One-time cost (est.) | First-year net | Payback |")
        L.append("|---|---|---|---|---|---|---|")
        for o in self.outcomes:
            f = o.financial
            hc = o.operational.headcount_before - o.operational.headcount_after
            payback = "—" if f.payback_months is None else f"{f.payback_months} mo"
            met = "yes" if o.selection.report.target_met else "**no**"
            L.append(
                f"| {o.scenario.name} | {hc} | ${f.annualized_savings:,.0f} | {met} | "
                f"${f.one_time_cost:,.0f} | ${f.first_year_net:,.0f} | {payback} |"
            )
        L.append("")
        L.append(
            "One-time cost is a **provisional estimate** from the assumptions "
            "below, not a severance calculation. Box 6 (Severance & Pay) owns "
            "the real figure and applies the leave-policy and withholding rules "
            "this estimate does not; these numbers exist only so scenarios can "
            "be compared on a consistent basis."
        )
        L.append("")

        # -- operational ---------------------------------------------------
        L.append("## Operational impact")
        L.append("")
        L.append("| Scenario | Headcount | Reduction | Managers lost | "
                 "Orphaned reports | Skill gaps | Single points of failure | "
                 "Median tenure after |")
        L.append("|---|---|---|---|---|---|---|---|")
        for o in self.outcomes:
            op = o.operational
            mt = "—" if op.median_tenure_after is None else f"{op.median_tenure_after:.1f}y"
            ml = "n/a" if op.managers_lost is None else str(op.managers_lost)
            orp = "n/a" if op.orphaned_reports is None else str(op.orphaned_reports)
            L.append(
                f"| {o.scenario.name} | {op.headcount_before} → {op.headcount_after} | "
                f"{op.reduction_pct:.1%} | {ml} | {orp} | "
                f"{len(op.skill_gaps)} | {len(op.single_points_of_failure)} | {mt} |"
            )
        L.append("")

        # -- compliance ----------------------------------------------------
        L.append("## Compliance signals")
        L.append("")
        L.append("| Scenario | Impact indicated | Impact flagged | "
                 "Selections needing counsel | Review queue | Selection errors |")
        L.append("|---|---|---|---|---|---|")
        for o in self.outcomes:
            ind = f"**{o.impact_indicated}**" if o.impact_indicated else "0"
            L.append(
                f"| {o.scenario.name} | {ind} | {len(o.impact.report.flagged)} | "
                f"{o.legal_review_count} | {o.review_queue_size} | "
                f"{o.selection_errors} |"
            )
        L.append("")

        # -- per-class verdicts --------------------------------------------
        classes: list[str] = []
        for o in self.outcomes:
            for c in o.impact.report.class_verdicts():
                if c not in classes:
                    classes.append(c)
        if classes:
            L.append("### Adverse impact verdict by class")
            L.append("")
            L.append("| Scenario | " + " | ".join(classes) + " |")
            L.append("|---" * (len(classes) + 1) + "|")
            for o in self.outcomes:
                v = o.impact.report.class_verdicts()
                L.append(
                    f"| {o.scenario.name} | "
                    + " | ".join(v.get(c, "—") for c in classes) + " |"
                )
            L.append("")

        # -- scenario detail ------------------------------------------------
        L.append("## Scenarios modeled")
        L.append("")
        for o in self.outcomes:
            L.append(f"### {o.scenario.name}")
            L.append("")
            L.append(f"**Business rationale:** {o.scenario.rationale}")
            if o.scenario.notes:
                L.append("")
                L.append(f"*{o.scenario.notes}*")
            L.append("")
            op = o.operational
            if op.department_reduction:
                L.append("| Department | Before | After | Reduction |")
                L.append("|---|---|---|---|")
                for d, v in sorted(op.department_reduction.items()):
                    L.append(
                        f"| {d} | {v['before']} | {v['after']} | {v['pct']:.0%} |"
                    )
                L.append("")
            if op.skill_gaps:
                L.append("**Critical skills lost entirely:**")
                for g in op.skill_gaps:
                    L.append(
                        f"- `{g['skill']}` in {g['department']} — "
                        f"{g['holders_before']} holder(s) before, none retained"
                    )
                L.append("")
            if op.single_points_of_failure:
                L.append("**Critical skills down to a single holder:**")
                for g in op.single_points_of_failure[:8]:
                    L.append(f"- `{g['skill']}` in {g['department']} — 1 holder remaining")
                L.append("")

        if self.findings:
            L.append("## Findings")
            L.append("")
            for f in self.findings:
                where = f" ({f['scenario']})" if f.get("scenario") else ""
                L.append(f"- **[{f['severity']}] {f['code']}**{where} — {f['message']}")
            L.append("")

        # -- assumptions ----------------------------------------------------
        L.append("## Cost assumptions")
        L.append("")
        for k, v in self.assumptions.items():
            L.append(f"- `{k}`: {v}")
        L.append("")

        L.append("## How to use this comparison")
        L.append("")
        L.append(
            "**There is no ranking here, and that is deliberate.** Savings, "
            "operational risk, and adverse impact do not share a scale, and a "
            "composite score would imply they trade off against one another in a "
            "way a court would not recognize. Choose on the business merits, then "
            "read the compliance column as a constraint on that choice rather "
            "than a term in it."
        )
        L.append("")
        L.append(
            "**Exploring alternatives is legitimate.** If a scenario meets the "
            "same business need with less impact, that is a less discriminatory "
            "alternative, and considering it is exactly what a disparate impact "
            "analysis calls for."
        )
        L.append("")
        L.append(
            "**Tuning criteria until the demographics look acceptable is not.** "
            "The distinction is whether a criteria change has a business reason "
            "that stands on its own. Record the rationale when you make the "
            "change, not after you see the result."
        )
        L.append("")
        L.append(
            "**This record is discoverable, including the scenarios you "
            "discarded.** That is a reason to document decisions carefully, not a "
            "reason to avoid writing them down. An unexplained gap in the sequence "
            "is harder to defend than a documented decision to move on."
        )
        L.append("")
        L.append("---")
        L.append(
            "_Estimates only. Severance and final-pay figures are provisional "
            "pending the Severance & Pay Engine; WARN determinations are made by "
            "the CA Compliance Engine; adverse impact results are statistical "
            "screening, not legal conclusions._"
        )
        return "\n".join(L)


@dataclass
class SimulationResult:
    outcomes: list[ScenarioOutcome]
    comparison: pd.DataFrame
    report: SimulationReport

    def by_name(self, name: str) -> ScenarioOutcome | None:
        for o in self.outcomes:
            if o.scenario.name == name:
                return o
        return None

    def write(self, outdir: str | Path, stem: str = "scenarios") -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "report_md": outdir / f"{stem}_comparison.md",
            "comparison": outdir / f"{stem}_comparison.csv",
            "report_json": outdir / f"{stem}_comparison.json",
        }
        self.comparison.to_csv(paths["comparison"], index=False)
        self.report.to_json(paths["report_json"])
        paths["report_md"].write_text(self.report.to_markdown(), encoding="utf-8")

        # Full artifacts per scenario, so a chosen scenario is fully documented.
        for o in self.outcomes:
            safe = "".join(
                ch if ch.isalnum() or ch in "-_" else "_" for ch in o.scenario.name
            ).strip("_") or "scenario"
            sub = outdir / safe
            o.selection.write(sub, stem="selection")
            o.impact.write(sub, stem="adverse_impact")
            paths[f"scenario:{o.scenario.name}"] = sub
        return paths


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


class ScenarioSimulator:
    """Runs and compares restructuring scenarios against one roster."""

    #: A department losing more than this fraction gets an operational warning.
    DEPT_REDUCTION_WARN = 0.30
    #: Swing in the disparity ratio between weight-only variants that warrants
    #: a note. Measured on the ratio, not the raw rate: a weight change can
    #: leave rates nearly flat while moving the ratio that decides the verdict.
    DISPARITY_DIVERGENCE = 0.30

    def __init__(
        self,
        assumptions: CostAssumptions | None = None,
        analyzer: AdverseImpactAnalyzer | None = None,
    ) -> None:
        self.assumptions = assumptions or CostAssumptions()
        self.analyzer = analyzer or AdverseImpactAnalyzer()

    # -- public ----------------------------------------------------------
    def run(
        self, roster: pd.DataFrame, scenarios: Sequence[Scenario]
    ) -> SimulationResult:
        report = SimulationReport(
            roster_size=len(roster), assumptions=self.assumptions.to_dict()
        )

        if not scenarios:
            report.add(Severity.ERROR, "NO_SCENARIOS", "No scenarios were supplied.")
            return SimulationResult([], pd.DataFrame(), report)

        as_of = {s.plan.as_of_date for s in scenarios if s.plan.as_of_date}
        if len(as_of) > 1:
            raise SelectionConfigError(
                f"Scenarios use different as_of_date values ({sorted(as_of)}). "
                f"Tenure, age, and pay all move with that date, so scenarios "
                f"dated differently are not comparable."
            )
        report.as_of_date = str(next(iter(as_of))) if as_of else None

        names = [s.name for s in scenarios]
        if len(set(names)) != len(names):
            raise SelectionConfigError("Scenario names must be unique.")

        outcomes: list[ScenarioOutcome] = []
        for scenario in scenarios:
            outcomes.append(self._run_one(roster, scenario, report))

        report.outcomes = outcomes
        self._cross_scenario_checks(outcomes, report)
        return SimulationResult(outcomes, report.comparison(), report)

    # -- one scenario -----------------------------------------------------
    def _run_one(
        self, roster: pd.DataFrame, scenario: Scenario, report: SimulationReport
    ) -> ScenarioOutcome:
        selection = SelectionEngine(scenario.plan).run(roster)
        impact = self.analyzer.run(selection.scores, scenario=scenario.name)
        financial = self._financials(selection, scenario)
        operational = self._operations(selection, scenario, report)

        if not selection.report.target_met:
            report.add(
                Severity.WARNING, "TARGET_NOT_MET",
                f"Reaches ${selection.report.achieved_savings:,.0f} of the "
                f"${scenario.plan.cost_savings_target:,.0f} target.",
                scenario=scenario.name,
            )
        if impact.report.indicated:
            classes = sorted({c.protected_class for c in impact.report.indicated})
            report.add(
                Severity.ERROR, "ADVERSE_IMPACT_INDICATED",
                f"Adverse impact indicated for {', '.join(classes)}. This scenario "
                f"cannot proceed to notice without counsel review, whatever its "
                f"financials look like.",
                scenario=scenario.name,
            )
        if selection.report.findings:
            errs = [f for f in selection.report.findings if f.severity == Severity.ERROR]
            if errs:
                report.add(
                    Severity.ERROR, "SELECTION_ERRORS",
                    f"{len(errs)} unresolved selection error(s), including: "
                    f"{errs[0].message}",
                    scenario=scenario.name,
                )

        return ScenarioOutcome(scenario, selection, impact, financial, operational)

    # -- financial --------------------------------------------------------
    def _financials(
        self, selection: SelectionResult, scenario: Scenario
    ) -> FinancialImpact:
        cut = selection.cut_list
        a = self.assumptions
        fin = FinancialImpact(
            annualized_savings=float(selection.report.achieved_savings),
            headcount_reduction=len(cut),
        )
        if cut.empty:
            return fin

        for _, row in cut.iterrows():
            annual = row.get("annualized_pay")
            annual = float(annual) if pd.notna(annual) else 0.0
            weekly = annual / 52.0

            tenure = row.get("tenure_years")
            tenure = float(tenure) if pd.notna(tenure) else 0.0
            weeks = min(
                max(tenure * a.severance_weeks_per_year, a.severance_min_weeks),
                a.severance_max_weeks,
            )
            fin.severance_cost += weekly * weeks

            if a.pay_out_accrued_vacation:
                hours = row.get("accrued_vacation_hours")
                rate = row.get("hourly_equivalent_rate")
                if pd.notna(hours) and pd.notna(rate):
                    fin.vacation_payout += float(hours) * float(rate)

            fin.cobra_cost += a.cobra_months * a.cobra_monthly_cost
            fin.admin_cost += a.admin_cost_per_employee

        fin.employer_payroll_tax = (
            (fin.severance_cost + fin.vacation_payout) * a.employer_payroll_tax_rate
        )
        return fin

    # -- operational ------------------------------------------------------
    def _operations(
        self, selection: SelectionResult, scenario: Scenario, report: SimulationReport
    ) -> OperationalImpact:
        scores = selection.scores
        op = OperationalImpact()
        if scores.empty:
            return op

        selected = scores["selected"].fillna(False).astype(bool)
        retained = scores.loc[~selected]
        cut = scores.loc[selected]

        op.headcount_before = len(scores)
        op.headcount_after = len(retained)

        # -- by department ------------------------------------------------
        for dept, group in scores.groupby("department", dropna=False):
            d = dept if isinstance(dept, str) else "(unassigned)"
            before = len(group)
            after = int((~group["selected"].fillna(False).astype(bool)).sum())
            pct = (before - after) / before if before else 0.0
            op.department_reduction[d] = {"before": before, "after": after,
                                          "pct": round(pct, 4)}
            if pct > self.DEPT_REDUCTION_WARN:
                report.add(
                    Severity.WARNING, "DEEP_DEPARTMENT_CUT",
                    f"{d} loses {pct:.0%} of its headcount ({before} → {after}). "
                    f"Check that the remaining team can carry the work, and note "
                    f"that concentrated reductions draw scrutiny.",
                    scenario=scenario.name,
                )

        if "worksite_name" in scores.columns:
            for site, group in scores.groupby("worksite_name", dropna=True):
                before = len(group)
                after = int((~group["selected"].fillna(False).astype(bool)).sum())
                op.worksite_reduction[str(site)] = {
                    "before": before, "after": after,
                    "pct": round((before - after) / before if before else 0.0, 4),
                }

        # -- managers and orphaned reports ---------------------------------
        if "employee_id" in scores.columns and "manager_id" in scores.columns:
            employee_ids = set(scores["employee_id"].dropna().astype(str))
            manager_ids = set(scores["manager_id"].dropna().astype(str))
            resolvable = manager_ids & employee_ids
            # If manager_id points at IDs that aren't employees in this roster,
            # the org graph can't be walked. Reporting "0 managers lost" would be
            # a false all-clear, so the counts stay None and the report says so.
            if manager_ids and not resolvable:
                op.managers_lost = None
                op.orphaned_reports = None
                already = any(
                    f["code"] == "ORG_STRUCTURE_UNRESOLVABLE" for f in report.findings
                )
                if already:
                    return op
                report.add(
                    Severity.WARNING, "ORG_STRUCTURE_UNRESOLVABLE",
                    f"None of the {len(manager_ids)} manager_id value(s) match an "
                    f"employee_id in this roster, so manager loss and orphaned "
                    f"reports could not be assessed. This is not a clean result — "
                    f"it is an unmeasured one. Map manager_id to employee_id in "
                    f"the source export to get it.",
                    scenario=scenario.name,
                )
            else:
                cut_ids = set(cut["employee_id"].dropna().astype(str))
                op.managers_lost = len(cut_ids & manager_ids)
                orphaned = retained["manager_id"].astype("string").isin(cut_ids)
                op.orphaned_reports = int(orphaned.sum())
                if op.orphaned_reports:
                    report.add(
                        Severity.INFO, "ORPHANED_REPORTS",
                        f"{op.orphaned_reports} retained employee(s) report to "
                        f"someone on the cut list and will need reassignment "
                        f"before notice day.",
                        scenario=scenario.name,
                    )

        # -- tenure --------------------------------------------------------
        if "tenure_years" in scores.columns:
            t_cut = cut["tenure_years"].astype("Float64").dropna()
            op.tenure_years_lost = float(t_cut.sum()) if len(t_cut) else 0.0
            before_t = scores["tenure_years"].astype("Float64").dropna()
            after_t = retained["tenure_years"].astype("Float64").dropna()
            op.median_tenure_before = (
                round(float(before_t.median()), 2) if len(before_t) else None
            )
            op.median_tenure_after = (
                round(float(after_t.median()), 2) if len(after_t) else None
            )

        # -- critical skill coverage ---------------------------------------
        self._skill_coverage(scores, retained, scenario, op, report)
        return op

    def _skill_coverage(
        self,
        scores: pd.DataFrame,
        retained: pd.DataFrame,
        scenario: Scenario,
        op: OperationalImpact,
        report: SimulationReport,
    ) -> None:
        """Check whether each department still holds the skills its plan calls
        critical. A scenario that saves money and deletes a capability has not
        actually succeeded."""
        if "skills" not in scores.columns:
            return

        def holders(frame: pd.DataFrame, dept: str, skill: str) -> int:
            sub = frame.loc[frame["department"] == dept]
            if sub.empty:
                return 0
            n = 0
            for _, row in sub.iterrows():
                held = set(split_items(row.get("skills")))
                if "certifications" in sub.columns:
                    held |= set(split_items(row.get("certifications")))
                if skill.lower() in held:
                    n += 1
            return n

        for dept in sorted({d for d in scores["department"].dropna().unique()}):
            dplan = scenario.plan.for_department(dept)
            if dplan is None:
                continue
            criticals: list[str] = []
            for crit in dplan.criteria:
                criticals.extend(crit.critical_items)
            for skill in sorted(set(criticals)):
                before = holders(scores, dept, skill)
                after = holders(retained, dept, skill)
                if before == 0:
                    continue
                if after == 0:
                    op.skill_gaps.append({
                        "department": dept, "skill": skill,
                        "holders_before": before, "holders_after": 0,
                    })
                    report.add(
                        Severity.ERROR, "CRITICAL_SKILL_ELIMINATED",
                        f"{dept} retains nobody with '{skill}', a skill its own "
                        f"plan calls critical ({before} holder(s) before). The "
                        f"scenario saves money by removing a capability the plan "
                        f"says the business needs.",
                        scenario=scenario.name,
                    )
                elif after == 1 and before > 1:
                    op.single_points_of_failure.append({
                        "department": dept, "skill": skill,
                        "holders_before": before, "holders_after": 1,
                    })

    # -- cross-scenario ----------------------------------------------------
    def _cross_scenario_checks(
        self, outcomes: list[ScenarioOutcome], report: SimulationReport
    ) -> None:
        if len(outcomes) < 2:
            return

        # Scenarios that reach the same target with materially different
        # operational or compliance consequences are the useful comparison.
        met = [o for o in outcomes if o.selection.report.target_met]
        if len(met) >= 2:
            clean = [o for o in met if not o.impact_indicated]
            flagged = [o for o in met if o.impact_indicated]
            if clean and flagged:
                report.add(
                    Severity.INFO, "ALTERNATIVE_MEETS_TARGET",
                    f"{len(clean)} scenario(s) reach the savings target without an "
                    f"adverse impact finding, while {len(flagged)} do not. Where "
                    f"two approaches serve the same business need, the one with "
                    f"less impact is the less discriminatory alternative, and a "
                    f"disparate impact analysis expects it to be considered. "
                    f"Record why whichever is chosen was chosen.",
                )

        # The pattern that looks like tuning: same targets and same departments,
        # differing only in criteria weights, with divergent demographic results.
        self._check_weight_tuning(outcomes, report)

    def _check_weight_tuning(
        self, outcomes: list[ScenarioOutcome], report: SimulationReport
    ) -> None:
        def shape(o: ScenarioOutcome) -> tuple:
            plan = o.scenario.plan
            return (
                plan.cost_savings_target,
                tuple(sorted(plan.departments)),
                tuple(sorted(
                    (d, dp.mode, tuple(sorted(c.name for c in dp.criteria)))
                    for d, dp in plan.departments.items()
                )),
            )

        groups: dict[tuple, list[ScenarioOutcome]] = {}
        for o in outcomes:
            groups.setdefault(shape(o), []).append(o)

        for _, group in groups.items():
            if len(group) < 2:
                continue
            # Same structure, differing only in weights or thresholds.
            weights = {
                tuple(sorted(
                    (d, c.name, round(c.weight, 4))
                    for d, dp in o.scenario.plan.departments.items()
                    for c in dp.normalized_criteria
                ))
                for o in group
            }
            if len(weights) < 2:
                continue

            for cls in {
                c for o in group for c in o.impact.report.class_verdicts()
            }:
                # Compare the *disparity*, not the raw selection rate. A weight
                # change can leave overall rates almost untouched while moving
                # the ratio that decides the verdict.
                #
                # Index per (group, scenario), not per class: two scenarios can
                # show an identical disparity magnitude while disadvantaging
                # opposite groups, and comparing only the worst group in each
                # would see no change at the moment the outcome flips completely.
                per_group: dict[str, dict[str, float]] = {}
                per_group_verdict: dict[str, dict[str, str]] = {}
                worst_group: dict[str, str] = {}

                for o in group:
                    comps = [
                        c for c in o.impact.report.comparisons
                        if c.protected_class == cls and c.unit_type == "overall"
                    ]
                    scored = [(c, _disparity_index(c)) for c in comps]
                    scored = [(c, i) for c, i in scored if i is not None]
                    if not scored:
                        continue
                    worst_group[o.scenario.name] = max(scored, key=lambda t: t[1])[0].group
                    for c, idx in scored:
                        per_group.setdefault(c.group, {})[o.scenario.name] = idx
                        per_group_verdict.setdefault(c.group, {})[o.scenario.name] = c.verdict

                if len(worst_group) < 2:
                    continue

                # The disadvantaged group changing identity is the loudest
                # possible version of this signal.
                flipped = len(set(worst_group.values())) > 1

                spreads = {
                    g: max(v.values()) - min(v.values())
                    for g, v in per_group.items() if len(v) >= 2
                }
                max_spread = max(spreads.values(), default=0.0)
                verdict_changed = any(
                    len(set(v.values())) > 1
                    for v in per_group_verdict.values() if len(v) >= 2
                )

                if not flipped and max_spread < self.DISPARITY_DIVERGENCE and not verdict_changed:
                    continue

                if flipped:
                    detail = ", ".join(f"{k}: {v}" for k, v in worst_group.items())
                    lead = (
                        f"Scenarios differing only in criteria weights shift which "
                        f"{cls} group bears the disparity ({detail})."
                    )
                else:
                    worst_g = max(spreads, key=spreads.get)
                    detail = ", ".join(
                        f"{k} {v:.2f}x" for k, v in per_group[worst_g].items()
                    )
                    lead = (
                        f"Scenarios differing only in criteria weights produce "
                        f"materially different {cls} disparity for group "
                        f"'{worst_g}': {detail}."
                    )

                extra = ""
                if verdict_changed:
                    changed = {
                        g: v for g, v in per_group_verdict.items()
                        if len(set(v.values())) > 1
                    }
                    g, v = next(iter(changed.items()))
                    extra = (
                        f" The adverse impact verdict for '{g}' also differs "
                        f"across these variants ({', '.join(f'{k}: {x}' for k, x in v.items())})."
                    )

                report.add(
                    Severity.WARNING, "WEIGHT_CHANGE_MOVES_DEMOGRAPHICS",
                    f"{lead}{extra} This is not wrong on its own — weights "
                    f"legitimately change who is selected. But a weight chosen "
                    f"after observing this shift is a weight chosen for its "
                    f"demographic effect, which is a different act from choosing "
                    f"it on the merits. Confirm the business rationale for the "
                    f"weighting was recorded independently of this comparison, "
                    f"and take it to counsel.",
                )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load a scenario set from YAML or JSON.

    Two forms are supported per scenario: ``plan`` as a path to a plan file, or
    ``plan`` as an inline mapping. A ``base`` plan may be declared once and
    overridden per scenario with ``overrides``.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:  # pragma: no cover
            raise SelectionConfigError("PyYAML is required to read YAML scenarios.")
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)

    if not isinstance(raw, dict) or "scenarios" not in raw:
        raise SelectionConfigError(
            f"{path.name} must be a mapping containing a 'scenarios' list."
        )

    base: dict[str, Any] | None = None
    if raw.get("base"):
        b = raw["base"]
        if isinstance(b, str):
            base_path = (path.parent / b) if not Path(b).is_absolute() else Path(b)
            btext = base_path.read_text(encoding="utf-8")
            base = (
                yaml.safe_load(btext)
                if base_path.suffix.lower() in (".yaml", ".yml") else json.loads(btext)
            )
        elif isinstance(b, dict):
            base = b
        else:
            raise SelectionConfigError("'base' must be a path or a mapping.")

    scenarios: list[Scenario] = []
    for entry in raw["scenarios"]:
        if not isinstance(entry, dict) or "name" not in entry:
            raise SelectionConfigError("Each scenario needs a 'name'.")
        name = str(entry["name"])

        if "plan" in entry and isinstance(entry["plan"], str):
            plan_path = Path(entry["plan"])
            if not plan_path.is_absolute():
                plan_path = path.parent / plan_path
            plan = load_plan(plan_path)
        else:
            merged = _deep_merge(base or {}, entry.get("plan") or entry.get("overrides") or {})
            if not merged:
                raise SelectionConfigError(
                    f"Scenario {name!r} defines no plan and no base was provided."
                )
            plan = plan_from_dict(merged, name=name)

        plan.plan_name = name
        scenarios.append(Scenario(
            name=name,
            plan=plan,
            rationale=str(entry.get("rationale", "")),
            notes=str(entry.get("notes", "")),
        ))
    return scenarios


def _disparity_index(comp) -> float | None:
    """A single comparable measure of how unevenly a group was selected.

    Prefers the termination-rate ratio, which is the most direct reading. That
    ratio is undefined whenever the reference group had zero selections — a
    common case in a small RIF — so it falls back to the inverse of the
    four-fifths retention ratio, which is defined wherever the reference group
    retained anyone. Both are 1.0 at parity and rise as the disparity widens,
    so they can be compared across scenarios on the same scale.
    """
    if comp.selection_rate_ratio is not None:
        return float(comp.selection_rate_ratio)
    if comp.impact_ratio is not None and comp.impact_ratio > 0:
        return 1.0 / float(comp.impact_ratio)
    return None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from .workforce_data import load_workforce_csv

    ap = argparse.ArgumentParser(
        description="Compare restructuring scenarios against one roster."
    )
    ap.add_argument("csv_path")
    ap.add_argument("--scenarios", required=True, help="Scenario set YAML/JSON.")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    scenarios = load_scenarios(args.scenarios)
    as_of = args.as_of
    if as_of is None:
        dates = {s.plan.as_of_date for s in scenarios if s.plan.as_of_date}
        as_of = str(next(iter(dates))) if len(dates) == 1 else None

    ingest = load_workforce_csv(args.csv_path, as_of=as_of)
    if ingest.report.is_blocking:
        print("Ingestion is blocking; fix the roster first.")
        return 2

    sim = ScenarioSimulator().run(ingest.data, scenarios)

    if not args.quiet:
        print(sim.report.to_markdown())

    if args.outdir:
        paths = sim.write(args.outdir)
        print("\nWrote:")
        for k, p in paths.items():
            print(f"  {k}: {p}")

    if any(o.impact_indicated for o in sim.outcomes):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
