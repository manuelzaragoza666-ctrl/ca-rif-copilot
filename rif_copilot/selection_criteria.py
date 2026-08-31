"""
selection_criteria.py
=====================

Selection Criteria Engine for the California RIF Copilot (module 2).

Consumes the standardized DataFrame produced by ``workforce_data.py`` and
produces, for each department, a documented retention score per employee and a
recommended cut list sized to a cost savings target.

What it does
------------
1. Loads a per-department plan config (YAML or JSON) that declares, for each
   department: the selection mode, the criteria weights, and any critical
   skills or protected positions.
2. Scores each eligible employee on the configured criteria, normalized within
   a comparison group so employees are only ranked against true peers.
3. Selects downward from the lowest retention score until the department's
   share of the cost savings target is met.
4. Emits a scored roster, a recommended cut list with per-employee rationale,
   and a SelectionReport describing every assumption, exclusion, and unresolved
   decision.

Two selection modes
-------------------
``individual``
    Rank employees within a comparison group and select the lowest scorers.
    Use when the work continues but fewer people are needed to do it.

``position``
    Score whole positions (job title groups) and eliminate them entirely;
    every incumbent in an eliminated position is on the list. Use when the
    role itself is going away. Individual performance does not drive position
    elimination — position-level criteria do.

Design commitments
------------------
* **Protected characteristics are firewalled out of scoring.** Attempting to
  configure a protected field as a criterion raises. The engine never reads
  gender, race, age, disability, or veteran status. Adverse impact is measured
  *after* selection, by module 3, on the output of this module.
* **Recommendations, never decisions.** Every selected employee carries a
  human-readable rationale and a score breakdown. Nothing is final until a
  human approves it, and the output schema has explicit override columns.
* **Missing data does not become a low score.** An employee with no
  performance rating is not silently ranked last; they are routed to manual
  review and excluded from automatic selection.
* **Ties are surfaced, not broken arbitrarily.** When the cut boundary falls
  inside a group of tied scores, the engine flags the tie for human
  resolution rather than picking by row order.

Usage
-----
    from .workforce_data import load_workforce_csv
    from .selection_criteria import SelectionEngine, load_plan

    roster = load_workforce_csv("roster.csv", as_of="2026-10-30")
    plan = load_plan("rif_plan.yaml")
    result = SelectionEngine(plan).run(roster.data)

    result.scores       # every eligible employee, scored
    result.cut_list     # recommended selections with rationale
    result.review_queue # cases a human must resolve before the list is valid
    print(result.report.to_markdown())

CLI
---
    python selection_criteria.py roster.csv --plan rif_plan.yaml \
        --as-of 2026-10-30 --outdir ./out
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:  # YAML is convenient for hand-edited plans but not required.
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

from .workforce_data import SCHEMA, SCHEMA_BY_NAME, Severity

__all__ = [
    "CriterionSpec",
    "DepartmentPlan",
    "RifPlan",
    "SelectionEngine",
    "SelectionResult",
    "SelectionReport",
    "load_plan",
    "plan_from_dict",
    "PROTECTED_FIELDS",
    "SelectionConfigError",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

#: Fields that must never influence a selection score. Sourced from the module 1
#: schema, plus the derived columns computed from them.
PROTECTED_FIELDS: frozenset[str] = frozenset(
    {f.name for f in SCHEMA if f.protected_class}
    | {f"{f.name}_raw" for f in SCHEMA if f.protected_class}
    | {
        "age_years", "age_band", "age_40_plus", "birth_date",
        "gender", "race_ethnicity", "disability_status", "veteran_status",
        # Not protected classes as such, but selecting on them directly invites
        # a retaliation or interference claim. Route through legal review instead.
        "leave_status", "union_flag", "visa_status", "work_email", "full_name",
        "first_name", "last_name",
    }
)

#: Post-selection conditions that require a human (usually counsel) to sign off.
#: These never influence the score; they annotate the resulting list.
LEGAL_REVIEW_RULES: tuple[tuple[str, str, str], ...] = (
    ("leave_status", "ON_PROTECTED_LEAVE",
     "Employee is on a leave of absence. Selecting an employee on protected "
     "leave (CFRA/FMLA/PDL) requires documented, leave-independent justification."),
    ("union_flag", "UNION_MEMBER",
     "Employee is in a bargaining unit. Check the CBA for seniority, bumping, "
     "and effects-bargaining obligations before finalizing."),
    ("visa_status", "WORK_VISA_HOLDER",
     "Employee holds a sponsored work authorization. Immigration counsel must "
     "advise on notice timing and status consequences."),
)


class SelectionConfigError(ValueError):
    """Raised when a plan config is invalid or unsafe."""


# ---------------------------------------------------------------------------
# Criterion definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionSpec:
    """One scoring dimension.

    Parameters
    ----------
    name : str
        Identifier used in the plan config and in the score breakdown.
    weight : float
        Relative weight. Weights are renormalized to sum to 1 per department.
    kind : str
        ``performance`` | ``skills`` | ``numeric`` | ``ordinal``
    source_column : str
        Column in the standardized roster this criterion reads.
    higher_is_better : bool
        Whether a larger raw value means a stronger retention case.
    scale : dict[str, float]
        For ordinal/performance criteria: raw label -> numeric value.
    critical_items : tuple[str, ...]
        For skills criteria: the skills/certifications the department needs to
        retain. Scored as coverage of this list.
    required : bool
        If True, an employee missing this value cannot be auto-selected and is
        routed to manual review instead of receiving a zero.
    """

    name: str
    weight: float
    kind: str
    source_column: str
    higher_is_better: bool = True
    scale: dict[str, float] = field(default_factory=dict)
    critical_items: tuple[str, ...] = ()
    required: bool = True

    def __post_init__(self) -> None:
        if self.source_column in PROTECTED_FIELDS:
            raise SelectionConfigError(
                f"Criterion {self.name!r} reads {self.source_column!r}, which is a "
                f"protected characteristic or a proxy routed to legal review. "
                f"Selection scores must not be computed from these fields. "
                f"Protected-class impact is measured after selection by the "
                f"Adverse Impact Analyzer."
            )
        if self.weight < 0:
            raise SelectionConfigError(f"Criterion {self.name!r} has a negative weight.")
        if self.kind not in ("performance", "skills", "numeric", "ordinal"):
            raise SelectionConfigError(
                f"Criterion {self.name!r} has unknown kind {self.kind!r}."
            )


#: Default rating vocabulary. Deliberately generous about phrasing because
#: rating scales differ everywhere; unmatched labels are reported, not guessed.
DEFAULT_PERFORMANCE_SCALE: dict[str, float] = {
    "far exceeds": 5.0, "significantly exceeds": 5.0, "outstanding": 5.0,
    "exceptional": 5.0, "top performer": 5.0, "5": 5.0, "a": 5.0,
    "exceeds": 4.0, "exceeds expectations": 4.0, "above expectations": 4.0,
    "strong": 4.0, "highly effective": 4.0, "4": 4.0, "b": 4.0,
    "meets": 3.0, "meets expectations": 3.0, "successful": 3.0,
    "effective": 3.0, "satisfactory": 3.0, "solid": 3.0, "3": 3.0, "c": 3.0,
    "partially meets": 2.0, "needs improvement": 2.0, "below": 2.0,
    "below expectations": 2.0, "inconsistent": 2.0, "developing": 2.0,
    "2": 2.0, "d": 2.0,
    "does not meet": 1.0, "unsatisfactory": 1.0, "poor": 1.0,
    "performance improvement plan": 1.0, "pip": 1.0, "1": 1.0, "f": 1.0,
}

#: Labels that mean "no rating exists", not "a bad rating".
NO_RATING_LABELS: frozenset[str] = frozenset(
    {"new", "new hire", "too new", "not rated", "n/a", "na", "no rating",
     "not applicable", "pending", "unrated", "no review", "not yet rated"}
)


# ---------------------------------------------------------------------------
# Plan configuration
# ---------------------------------------------------------------------------


@dataclass
class DepartmentPlan:
    """Selection configuration for one department."""

    department: str
    mode: str = "individual"  # individual | position
    criteria: tuple[CriterionSpec, ...] = ()
    #: Columns defining the peer group employees are ranked within.
    comparison_group: tuple[str, ...] = ("department", "job_level")
    #: Job titles that cannot be selected (business-critical, sole incumbent, etc.)
    protected_positions: tuple[str, ...] = ()
    #: Job titles targeted for elimination in ``position`` mode.
    eliminate_positions: tuple[str, ...] = ()
    #: Optional cap on this department's contribution to the savings target.
    max_savings_share: float | None = None
    #: Optional hard cap on headcount reduction in this department.
    max_headcount: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("individual", "position"):
            raise SelectionConfigError(
                f"Department {self.department!r}: mode must be 'individual' or "
                f"'position', got {self.mode!r}."
            )
        if self.mode == "individual" and not self.criteria:
            raise SelectionConfigError(
                f"Department {self.department!r} uses individual selection but "
                f"declares no criteria. Selection without documented criteria is "
                f"exactly the pattern that fails legal review."
            )
        for col in self.comparison_group:
            if col in PROTECTED_FIELDS:
                raise SelectionConfigError(
                    f"Department {self.department!r}: comparison_group includes "
                    f"protected field {col!r}. Peer groups must be defined by job "
                    f"structure, not by who is in them."
                )

    @property
    def normalized_criteria(self) -> tuple[CriterionSpec, ...]:
        total = sum(c.weight for c in self.criteria)
        if total <= 0:
            return self.criteria
        return tuple(
            CriterionSpec(
                name=c.name, weight=c.weight / total, kind=c.kind,
                source_column=c.source_column, higher_is_better=c.higher_is_better,
                scale=c.scale, critical_items=c.critical_items, required=c.required,
            )
            for c in self.criteria
        )


@dataclass
class RifPlan:
    """Whole-organization selection plan."""

    #: Total annualized payroll savings sought, in dollars.
    cost_savings_target: float
    departments: dict[str, DepartmentPlan] = field(default_factory=dict)
    default_plan: DepartmentPlan | None = None
    #: Multiplier applied to base pay to approximate fully loaded cost
    #: (benefits, payroll taxes, etc.). 1.0 = base pay only.
    burden_multiplier: float = 1.25
    #: Employees below this retention score are never auto-selected without
    #: review, regardless of savings math. None disables the floor.
    manual_review_below: float | None = None
    #: Minimum peers required before ranking is meaningful. Below this, an
    #: employee is routed to manual review instead of being auto-selected:
    #: "lowest ranked of one" is not a comparison and will not survive scrutiny.
    min_comparison_group_size: int = 2
    as_of_date: dt.date | None = None
    plan_name: str = "Untitled RIF scenario"

    def for_department(self, dept: str | None) -> DepartmentPlan | None:
        if dept and dept in self.departments:
            return self.departments[dept]
        if self.default_plan is not None:
            base = self.default_plan
            return DepartmentPlan(
                department=dept or "(unassigned)", mode=base.mode,
                criteria=base.criteria, comparison_group=base.comparison_group,
                protected_positions=base.protected_positions,
                eliminate_positions=(), max_savings_share=base.max_savings_share,
                max_headcount=base.max_headcount,
                notes=f"Inherited default plan. {base.notes}".strip(),
            )
        return None


def _criterion_from_dict(name: str, raw: dict[str, Any]) -> CriterionSpec:
    kind = str(raw.get("kind", "numeric"))
    scale = {str(k).lower(): float(v) for k, v in (raw.get("scale") or {}).items()}
    if kind == "performance" and not scale:
        scale = dict(DEFAULT_PERFORMANCE_SCALE)
    return CriterionSpec(
        name=name,
        weight=float(raw.get("weight", 1.0)),
        kind=kind,
        source_column=str(raw.get("source_column", raw.get("column", ""))),
        higher_is_better=bool(raw.get("higher_is_better", True)),
        scale=scale,
        critical_items=tuple(str(s) for s in (raw.get("critical_items") or ())),
        required=bool(raw.get("required", True)),
    )


def _dept_from_dict(name: str, raw: dict[str, Any]) -> DepartmentPlan:
    criteria = tuple(
        _criterion_from_dict(cname, craw)
        for cname, craw in (raw.get("criteria") or {}).items()
    )
    return DepartmentPlan(
        department=name,
        mode=str(raw.get("mode", "individual")),
        criteria=criteria,
        comparison_group=tuple(raw.get("comparison_group") or ("department", "job_level")),
        protected_positions=tuple(raw.get("protected_positions") or ()),
        eliminate_positions=tuple(raw.get("eliminate_positions") or ()),
        max_savings_share=(
            float(raw["max_savings_share"]) if raw.get("max_savings_share") is not None else None
        ),
        max_headcount=(
            int(raw["max_headcount"]) if raw.get("max_headcount") is not None else None
        ),
        notes=str(raw.get("notes", "")),
    )


def plan_from_dict(raw: dict[str, Any], name: str = "Untitled RIF scenario") -> RifPlan:
    """Build a RifPlan from an already-parsed mapping.

    Shared by ``load_plan`` and by the Scenario Simulator, which defines plan
    variants inline rather than in separate files.
    """
    if not isinstance(raw, dict):
        raise SelectionConfigError("Plan config did not parse to a mapping.")
    if "cost_savings_target" not in raw:
        raise SelectionConfigError(
            "Plan must declare 'cost_savings_target' (annualized dollars)."
        )

    departments = {
        dname: _dept_from_dict(dname, dept_raw)
        for dname, dept_raw in (raw.get("departments") or {}).items()
    }
    default_plan = (
        _dept_from_dict("(default)", raw["default"]) if raw.get("default") else None
    )
    as_of = raw.get("as_of_date")
    return RifPlan(
        cost_savings_target=float(raw["cost_savings_target"]),
        departments=departments,
        default_plan=default_plan,
        burden_multiplier=float(raw.get("burden_multiplier", 1.25)),
        manual_review_below=(
            float(raw["manual_review_below"])
            if raw.get("manual_review_below") is not None else None
        ),
        min_comparison_group_size=int(raw.get("min_comparison_group_size", 2)),
        as_of_date=pd.Timestamp(as_of).date() if as_of else None,
        plan_name=str(raw.get("plan_name", name)),
    )


def load_plan(path: str | Path) -> RifPlan:
    """Load a RIF plan from a YAML or JSON config file."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        if yaml is None:  # pragma: no cover
            raise SelectionConfigError("PyYAML is required to read YAML plans.")
        raw = yaml.safe_load(text)
    else:
        raw = json.loads(text)
    if not isinstance(raw, dict):
        raise SelectionConfigError(f"{path.name} did not parse to a mapping.")
    return plan_from_dict(raw, name=path.stem)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class SelectionFinding:
    severity: str
    code: str
    message: str
    department: str | None = None
    employee_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity, "code": self.code, "message": self.message,
            "department": self.department, "employee_id": self.employee_id,
        }


@dataclass
class SelectionReport:
    plan_name: str = ""
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
    cost_savings_target: float = 0.0
    achieved_savings: float = 0.0
    selected_count: int = 0
    eligible_count: int = 0
    findings: list[SelectionFinding] = field(default_factory=list)
    department_summary: list[dict[str, Any]] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, **kw: Any) -> None:
        self.findings.append(SelectionFinding(severity, code, message, **kw))

    @property
    def target_met(self) -> bool:
        return self.achieved_savings >= self.cost_savings_target

    def summary(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "generated_at": self.generated_at,
            "cost_savings_target": self.cost_savings_target,
            "achieved_savings": self.achieved_savings,
            "shortfall": max(0.0, self.cost_savings_target - self.achieved_savings),
            "target_met": self.target_met,
            "eligible_employees": self.eligible_count,
            "selected_employees": self.selected_count,
            "errors": len([f for f in self.findings if f.severity == Severity.ERROR]),
            "warnings": len([f for f in self.findings if f.severity == Severity.WARNING]),
        }

    def to_dataframe(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(
                columns=["severity", "code", "message", "department", "employee_id"]
            )
        order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        df = pd.DataFrame([f.to_dict() for f in self.findings])
        df["_s"] = df["severity"].map(order).fillna(9)
        return df.sort_values(["_s", "code"]).drop(columns="_s").reset_index(drop=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "departments": self.department_summary,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self, max_per_code: int = 6) -> str:
        s = self.summary()
        L: list[str] = []
        L.append("# Selection Criteria Report")
        L.append("")
        L.append(f"**Scenario:** {s['plan_name']}  ")
        L.append(f"**Generated:** {s['generated_at']}  ")
        L.append("")
        L.append("## Target")
        L.append("")
        L.append("| Metric | Value |")
        L.append("|---|---|")
        L.append(f"| Annualized savings target | ${s['cost_savings_target']:,.0f} |")
        L.append(f"| Projected savings from recommendation | ${s['achieved_savings']:,.0f} |")
        if not s["target_met"]:
            L.append(f"| **Shortfall** | **${s['shortfall']:,.0f}** |")
        L.append(f"| Employees evaluated | {s['eligible_employees']} |")
        L.append(f"| Employees recommended for selection | {s['selected_employees']} |")
        L.append("")

        if self.department_summary:
            L.append("## By department")
            L.append("")
            L.append("| Department | Mode | Evaluated | Selected | Savings |")
            L.append("|---|---|---|---|---|")
            for d in self.department_summary:
                L.append(
                    f"| {d['department']} | {d['mode']} | {d['evaluated']} | "
                    f"{d['selected']} | ${d['savings']:,.0f} |"
                )
            L.append("")

        if self.findings:
            L.append("## Findings")
            L.append("")
            df = self.to_dataframe()
            for code, group in df.groupby("code", sort=False):
                sev = group["severity"].iloc[0]
                L.append(f"**[{sev}] {code}** — {len(group)} occurrence(s)")
                for _, r in group.head(max_per_code).iterrows():
                    who = []
                    if isinstance(r.get("department"), str) and r["department"]:
                        who.append(r["department"])
                    if isinstance(r.get("employee_id"), str) and r["employee_id"]:
                        who.append(f"emp {r['employee_id']}")
                    prefix = f"({'; '.join(who)}) " if who else ""
                    L.append(f"- {prefix}{r['message']}")
                if len(group) > max_per_code:
                    L.append(f"- …and {len(group) - max_per_code} more")
                L.append("")

        L.append("---")
        L.append(
            "**This is a recommendation, not a decision.** Scores reflect only the "
            "criteria configured in the plan. No protected characteristic was used "
            "in scoring. Before this list is acted on it must be reviewed by a "
            "human decision-maker, run through adverse impact analysis, and "
            "cleared by employment counsel."
        )
        return "\n".join(L)


@dataclass
class SelectionResult:
    scores: pd.DataFrame
    cut_list: pd.DataFrame
    review_queue: pd.DataFrame
    report: SelectionReport
    plan: RifPlan

    def write(self, outdir: str | Path, stem: str = "selection") -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "cut_list": outdir / f"{stem}_recommended_cut_list.csv",
            "scores": outdir / f"{stem}_scores.csv",
            "review_queue": outdir / f"{stem}_review_queue.csv",
            "report_json": outdir / f"{stem}_report.json",
            "report_md": outdir / f"{stem}_report.md",
        }
        self.cut_list.to_csv(paths["cut_list"], index=False)
        self.scores.to_csv(paths["scores"], index=False)
        self.review_queue.to_csv(paths["review_queue"], index=False)
        self.report.to_json(paths["report_json"])
        paths["report_md"].write_text(self.report.to_markdown(), encoding="utf-8")
        return paths


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def split_items(value: Any) -> list[str]:
    """Split a delimited skills/certifications cell into normalized items."""
    if value is None or (isinstance(value, float) and math.isnan(value)) or pd.isna(value):
        return []
    parts = re.split(r"[|;,/]+", str(value))
    out = []
    for p in parts:
        p = re.sub(r"\s+", " ", p).strip()
        p = re.sub(r"\s*\(.*?\)\s*$", "", p).strip()  # drop "(exp. 2025-04-01)"
        if p:
            out.append(p.lower())
    return out


def score_performance(
    value: Any, scale: dict[str, float]
) -> tuple[float | None, str | None]:
    """Map a rating label to a numeric value. Returns (value, reason_if_missing)."""
    if value is None or pd.isna(value):
        return None, "no performance rating on record"
    label = re.sub(r"\s+", " ", str(value)).strip().lower().rstrip(".")
    if not label:
        return None, "no performance rating on record"
    if label in NO_RATING_LABELS:
        return None, f"rating {str(value)!r} means the employee has not been rated"
    if label in scale:
        return scale[label], None
    # A bare number on an unknown scale still carries ordinal information.
    try:
        return float(label), None
    except ValueError:
        pass
    return None, f"rating {str(value)!r} is not in the configured rating scale"


def score_skills(
    value_skills: Any, value_certs: Any, critical: Sequence[str]
) -> tuple[float | None, str | None, list[str]]:
    """Coverage of the department's critical skills, as a 0-1 fraction."""
    held = set(split_items(value_skills)) | set(split_items(value_certs))
    if not critical:
        return None, "no critical skills defined for this department", []
    if not held:
        return None, "no skills or certifications on record", []
    wanted = [c.lower() for c in critical]
    matched = [c for c in wanted if c in held]
    return len(matched) / len(wanted), None, matched


def _normalize_within(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """Min-max normalize to 0-100 within a comparison group."""
    vals = series.astype("Float64")
    valid = vals.dropna()
    if valid.empty:
        return pd.Series(pd.NA, index=series.index, dtype="Float64")
    lo, hi = float(valid.min()), float(valid.max())
    if math.isclose(lo, hi):
        # Everyone is identical on this criterion; it cannot differentiate.
        return pd.Series(
            [pd.NA if pd.isna(v) else 50.0 for v in vals],
            index=series.index, dtype="Float64",
        )
    scaled = (vals - lo) / (hi - lo) * 100.0
    if not higher_is_better:
        scaled = 100.0 - scaled
    return scaled.astype("Float64")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SelectionEngine:
    """Scores employees and builds a recommended cut list from a RifPlan."""

    def __init__(self, plan: RifPlan) -> None:
        self.plan = plan

    # -- public ----------------------------------------------------------
    def run(self, roster: pd.DataFrame) -> SelectionResult:
        report = SelectionReport(
            plan_name=self.plan.plan_name,
            cost_savings_target=self.plan.cost_savings_target,
        )
        df = self._prepare(roster, report)
        if df.empty:
            return SelectionResult(
                df, df.copy(), df.copy(), report, self.plan
            )

        scored_parts: list[pd.DataFrame] = []
        for dept, group in df.groupby("department", dropna=False):
            dept_name = dept if isinstance(dept, str) else "(unassigned)"
            dept_plan = self.plan.for_department(dept_name)
            if dept_plan is None:
                report.add(
                    Severity.ERROR, "NO_PLAN_FOR_DEPARTMENT",
                    f"No plan entry and no default for {dept_name!r}. Its "
                    f"{len(group)} employee(s) were excluded from selection.",
                    department=dept_name,
                )
                excluded = group.copy()
                excluded["selection_status"] = "excluded_no_plan"
                excluded["retention_score"] = pd.NA
                excluded["rationale"] = "Department has no configured selection plan."
                scored_parts.append(excluded)
                continue
            scored_parts.append(self._score_department(group, dept_plan, report))

        scores = pd.concat(scored_parts, ignore_index=True)
        # Departments excluded before scoring never got these columns; add them
        # so the downstream invariant checks and outputs have a stable shape.
        for col, default in (
            ("comparison_group", "(not scored)"), ("selection_mode", "none"),
            ("score_breakdown", ""), ("rationale", ""),
        ):
            if col not in scores.columns:
                scores[col] = default
            else:
                scores[col] = scores[col].fillna(default)
        scores = self._apply_legal_review_flags(scores)

        cut_list, scores = self._select(scores, report)
        review_queue = scores.loc[
            scores["selection_status"].isin(
                ["manual_review", "tie_at_boundary", "excluded_insufficient_data"]
            )
        ].copy()

        report.eligible_count = int((scores["selection_status"] != "excluded_no_plan").sum())
        report.selected_count = len(cut_list)
        self._finalize(scores, cut_list, report)
        return SelectionResult(scores, cut_list, review_queue, report, self.plan)

    # -- preparation -----------------------------------------------------
    def _prepare(self, roster: pd.DataFrame, report: SelectionReport) -> pd.DataFrame:
        df = roster.copy()

        if "has_blocking_error" in df.columns:
            bad = df["has_blocking_error"].fillna(False).astype(bool)
            if bad.any():
                report.add(
                    Severity.WARNING, "BLOCKING_DATA_ERRORS_EXCLUDED",
                    f"{int(bad.sum())} employee(s) carry unresolved data errors from "
                    f"ingestion and were excluded. Fix the source records and re-run; "
                    f"until then the savings math is incomplete.",
                )
                df = df.loc[~bad]

        if "is_active" in df.columns:
            inactive = ~df["is_active"].fillna(True).astype(bool)
            if inactive.any():
                report.add(
                    Severity.INFO, "INACTIVE_EXCLUDED",
                    f"{int(inactive.sum())} already-separated employee(s) excluded.",
                )
                df = df.loc[~inactive]

        if "annualized_pay" in df.columns:
            no_pay = df["annualized_pay"].isna()
            if no_pay.any():
                report.add(
                    Severity.WARNING, "NO_PAY_DATA",
                    f"{int(no_pay.sum())} employee(s) have no annualized pay and "
                    f"cannot contribute to a cost target; excluded from selection.",
                )
                df = df.loc[~no_pay]
            df = df.copy()
            df["annual_cost"] = (
                df["annualized_pay"].astype("Float64") * self.plan.burden_multiplier
            ).round(2)
        else:
            report.add(
                Severity.ERROR, "MISSING_PAY_COLUMN",
                "Roster has no annualized_pay column; a cost-based target cannot "
                "be computed. Run the Data Manager first.",
            )
            return df.iloc[0:0]

        return df.reset_index(drop=True)

    # -- scoring ---------------------------------------------------------
    def _score_department(
        self, group: pd.DataFrame, plan: DepartmentPlan, report: SelectionReport
    ) -> pd.DataFrame:
        g = group.copy()
        g["selection_mode"] = plan.mode
        g["selection_status"] = "eligible"
        g["rationale"] = ""
        g["score_breakdown"] = ""
        dept = plan.department

        if plan.mode == "position":
            return self._score_position_mode(g, plan, report)

        criteria = plan.normalized_criteria
        group_cols = [c for c in plan.comparison_group if c in g.columns]
        if not group_cols:
            group_cols = ["department"]
            report.add(
                Severity.WARNING, "COMPARISON_GROUP_UNAVAILABLE",
                f"None of the configured comparison_group columns exist; employees "
                f"were ranked against the whole department instead. Ranking across "
                f"dissimilar roles weakens the defensibility of the result.",
                department=dept,
            )
        dropped = [c for c in plan.comparison_group if c not in g.columns]
        if dropped and group_cols:
            report.add(
                Severity.WARNING, "COMPARISON_GROUP_DEGRADED",
                f"Comparison group was configured as "
                f"{list(plan.comparison_group)} but {dropped} are absent from the "
                f"roster, so employees were ranked using {group_cols} only. "
                f"Peers at different levels are being compared directly; populate "
                f"the missing column or narrow the group before relying on this.",
                department=dept,
            )
        g["comparison_group"] = (
            g[group_cols].astype("string").fillna("(blank)").agg(" | ".join, axis=1)
        )

        # -- raw criterion values -----------------------------------------
        missing_reasons: dict[int, list[str]] = {i: [] for i in g.index}
        raw_cols: dict[str, str] = {}

        for crit in criteria:
            raw_col = f"raw_{crit.name}"
            raw_cols[crit.name] = raw_col
            values: list[Any] = []

            for idx in g.index:
                if crit.kind == "performance":
                    val, why = score_performance(
                        g.at[idx, crit.source_column] if crit.source_column in g.columns else None,
                        crit.scale or DEFAULT_PERFORMANCE_SCALE,
                    )
                elif crit.kind == "skills":
                    certs = g.at[idx, "certifications"] if "certifications" in g.columns else None
                    val, why, matched = score_skills(
                        g.at[idx, crit.source_column] if crit.source_column in g.columns else None,
                        certs, crit.critical_items,
                    )
                    if val is not None:
                        g.at[idx, f"matched_{crit.name}"] = ", ".join(matched) or "(none)"
                else:
                    v = g.at[idx, crit.source_column] if crit.source_column in g.columns else None
                    val = None if pd.isna(v) else float(v)
                    why = None if val is not None else f"{crit.source_column} is blank"

                if val is None and crit.required:
                    missing_reasons[idx].append(f"{crit.name}: {why}")
                values.append(val)

            g[raw_col] = pd.Series(values, index=g.index, dtype="Float64")

            unscored = g[raw_col].isna().sum()
            if unscored:
                report.add(
                    Severity.WARNING, "CRITERION_DATA_GAPS",
                    f"Criterion {crit.name!r} could not be scored for "
                    f"{int(unscored)} of {len(g)} employee(s). Those employees are "
                    f"routed to manual review rather than scored as zero.",
                    department=dept,
                )

        # -- normalize within comparison group and combine ----------------
        for crit in criteria:
            norm_col = f"score_{crit.name}"
            g[norm_col] = pd.Series(pd.NA, index=g.index, dtype="Float64")
            for _, sub in g.groupby("comparison_group"):
                g.loc[sub.index, norm_col] = _normalize_within(
                    sub[raw_cols[crit.name]], crit.higher_is_better
                )

        total = pd.Series(0.0, index=g.index, dtype="float64")
        for crit in criteria:
            total = total + g[f"score_{crit.name}"].astype("Float64").fillna(0.0) * crit.weight
        g["retention_score"] = total.round(2).astype("Float64")

        # -- breakdown text for the audit trail ---------------------------
        for idx in g.index:
            parts = [
                f"{c.name} {float(g.at[idx, f'score_{c.name}']):.0f}/100 x{c.weight:.2f}"
                for c in criteria
                if pd.notna(g.at[idx, f"score_{c.name}"])
            ]
            g.at[idx, "score_breakdown"] = "; ".join(parts)

        # -- statuses ------------------------------------------------------
        for idx in g.index:
            if missing_reasons[idx]:
                g.at[idx, "selection_status"] = "excluded_insufficient_data"
                g.at[idx, "retention_score"] = pd.NA
                g.at[idx, "rationale"] = (
                    "Cannot be scored on all required criteria ("
                    + "; ".join(missing_reasons[idx])
                    + "). Routed to manual review; not auto-selected."
                )

        # -- degenerate comparison groups ---------------------------------
        # Min-max normalization inside a group of one produces a score with no
        # comparative content. Selecting on it is individual selection with the
        # appearance of a ranking, which is worse than no ranking at all.
        min_size = self.plan.min_comparison_group_size
        sizes = g["comparison_group"].value_counts()
        for grp_name, size in sizes.items():
            if size >= min_size:
                continue
            idxs = g.index[(g["comparison_group"] == grp_name)
                           & (g["selection_status"] == "eligible")]
            for idx in idxs:
                g.at[idx, "selection_status"] = "manual_review"
                g.at[idx, "rationale"] = (
                    f"Comparison group '{grp_name}' contains only {size} "
                    f"employee(s), so the retention score carries no comparative "
                    f"meaning. If this role is genuinely going away, configure it "
                    f"as a position elimination; if it is not, widen the "
                    f"comparison group. Not auto-selected."
                )
            if len(idxs):
                report.add(
                    Severity.WARNING, "DEGENERATE_COMPARISON_GROUP",
                    f"Comparison group '{grp_name}' has {size} member(s), below "
                    f"the minimum of {min_size}. Ranking within it is not a real "
                    f"comparison; {len(idxs)} employee(s) routed to manual review.",
                    department=dept,
                )

        if plan.protected_positions and "job_title" in g.columns:
            prot = {p.lower() for p in plan.protected_positions}
            mask = g["job_title"].astype("string").str.lower().isin(prot)
            g.loc[mask, "selection_status"] = "protected_position"
            g.loc[mask, "rationale"] = "Position is designated business-critical in the plan."

        # Single-criterion warning: worth saying once per department.
        if len(criteria) == 1:
            report.add(
                Severity.WARNING, "SINGLE_CRITERION_SELECTION",
                f"Selection rests on one criterion ({criteria[0].name}). A "
                f"single-factor score is harder to defend and concentrates any "
                f"bias in that factor. Consider adding an objective second factor.",
                department=dept,
            )
        if any(c.kind == "performance" for c in criteria):
            report.add(
                Severity.INFO, "PERFORMANCE_RATINGS_IN_USE",
                "Performance ratings drive part of this score. Confirm ratings "
                "were applied consistently across managers before relying on them; "
                "inconsistent rating practice is a common source of challengeable "
                "disparities.",
                department=dept,
            )
        return g

    def _score_position_mode(
        self, g: pd.DataFrame, plan: DepartmentPlan, report: SelectionReport
    ) -> pd.DataFrame:
        """Whole positions are eliminated; incumbents follow the position."""
        dept = plan.department
        g["comparison_group"] = g["job_title"].astype("string").fillna("(blank)")
        g["retention_score"] = pd.NA

        if not plan.eliminate_positions:
            report.add(
                Severity.ERROR, "NO_POSITIONS_DESIGNATED",
                f"{dept} uses position-elimination mode but lists no positions in "
                f"eliminate_positions. Nothing was selected here.",
                department=dept,
            )
            g["selection_status"] = "excluded_no_positions_designated"
            g["rationale"] = "Position elimination configured but no positions designated."
            return g

        targets = {p.lower() for p in plan.eliminate_positions}
        titles = g["job_title"].astype("string").str.lower()
        mask = titles.isin(targets)

        g["selection_status"] = "not_targeted"
        g["rationale"] = "Position is not designated for elimination."
        g.loc[mask, "selection_status"] = "position_eliminated"
        g.loc[mask, "rationale"] = (
            "Entire position is being eliminated; selection is at the position "
            "level, not based on individual performance."
        )

        found = set(titles[mask].dropna())
        for t in sorted(targets - found):
            report.add(
                Severity.WARNING, "DESIGNATED_POSITION_NOT_FOUND",
                f"Position {t!r} is designated for elimination but no incumbent "
                f"with that title exists in {dept}. Check the title spelling.",
                department=dept,
            )

        # A position held by only some of its incumbents is a red flag: if a
        # title survives elsewhere, elimination is really individual selection.
        report.add(
            Severity.INFO, "POSITION_MODE_IN_USE",
            f"{dept} eliminates {len(found)} position type(s) covering "
            f"{int(mask.sum())} incumbent(s). Confirm no retained employee is "
            f"performing substantially the same work, which would make this an "
            f"individual selection requiring documented criteria.",
            department=dept,
        )
        return g

    # -- legal review annotations ----------------------------------------
    @staticmethod
    def _apply_legal_review_flags(scores: pd.DataFrame) -> pd.DataFrame:
        flags: list[str] = []
        notes: list[str] = []
        for idx in scores.index:
            row_flags, row_notes = [], []
            for col, code, note in LEGAL_REVIEW_RULES:
                if col not in scores.columns:
                    continue
                v = scores.at[idx, col]
                hit = bool(v) if isinstance(v, (bool, np.bool_)) else (
                    pd.notna(v) and str(v).strip() != ""
                )
                if hit:
                    row_flags.append(code)
                    row_notes.append(note)
            flags.append("|".join(row_flags))
            notes.append(" ".join(row_notes))
        scores = scores.copy()
        scores["legal_review_flags"] = pd.Series(flags, index=scores.index, dtype="string")
        scores["legal_review_notes"] = pd.Series(notes, index=scores.index, dtype="string")
        return scores

    # -- selection --------------------------------------------------------
    def _select(
        self, scores: pd.DataFrame, report: SelectionReport
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        scores = scores.copy()
        scores["selected"] = False
        scores["selection_rank"] = pd.NA
        scores["cumulative_savings"] = pd.NA

        # Position-mode selections are not optional; they come first.
        forced = scores["selection_status"] == "position_eliminated"
        scores.loc[forced, "selected"] = True
        running = float(scores.loc[forced, "annual_cost"].astype("Float64").sum())

        target = self.plan.cost_savings_target
        if running >= target and forced.any():
            report.add(
                Severity.WARNING, "POSITION_ELIMINATIONS_EXCEED_TARGET",
                f"Position eliminations alone save ${running:,.0f} against a "
                f"${target:,.0f} target. No individual selections were needed; "
                f"consider whether the eliminations are larger than necessary.",
            )

        # Individual selection, cheapest-to-defend first: lowest score goes first.
        pool = scores.loc[
            (scores["selection_status"] == "eligible") & (~scores["selected"])
        ].copy()

        # A very low score flags the selection for mandatory sign-off. It must NOT
        # remove the employee from the pool: doing so pushes selection UP the
        # ranking and can cut a stronger performer while the lowest scorer stays.
        scores["requires_approval"] = False
        if self.plan.manual_review_below is not None:
            low = (
                pool["retention_score"].astype("Float64")
                < self.plan.manual_review_below
            ).fillna(False)
            for idx in pool.index[low]:
                scores.at[idx, "requires_approval"] = True
            if low.any():
                report.add(
                    Severity.WARNING, "SCORE_BELOW_REVIEW_THRESHOLD",
                    f"{int(low.sum())} employee(s) score below the plan's "
                    f"manual_review_below threshold of "
                    f"{self.plan.manual_review_below}. They remain in rank order, "
                    f"but each selection needs explicit sign-off — a score that "
                    f"low is often a data problem or a performance-management "
                    f"issue that should be handled on its own terms.",
                )

        pool = pool.sort_values(
            ["retention_score", "annual_cost"], ascending=[True, False], kind="mergesort"
        )

        dept_savings: dict[str, float] = {}
        dept_counts: dict[str, int] = {}
        rank = int(forced.sum())

        for idx in pool.index:
            if running >= target:
                break
            dept = scores.at[idx, "department"]
            dept = dept if isinstance(dept, str) else "(unassigned)"
            dplan = self.plan.for_department(dept)
            cost = float(scores.at[idx, "annual_cost"])

            if dplan is not None:
                if dplan.max_headcount is not None and dept_counts.get(dept, 0) >= dplan.max_headcount:
                    scores.at[idx, "rationale"] = (
                        f"Not selected: {dept} reached its plan cap of "
                        f"{dplan.max_headcount} reduction(s)."
                    )
                    continue
                if dplan.max_savings_share is not None:
                    cap = dplan.max_savings_share * target
                    if dept_savings.get(dept, 0.0) + cost > cap:
                        scores.at[idx, "rationale"] = (
                            f"Not selected: would exceed {dept}'s savings cap of "
                            f"${cap:,.0f}."
                        )
                        continue

            rank += 1
            scores.at[idx, "selected"] = True
            scores.at[idx, "selection_rank"] = rank
            running += cost
            scores.at[idx, "cumulative_savings"] = round(running, 2)
            dept_savings[dept] = dept_savings.get(dept, 0.0) + cost
            dept_counts[dept] = dept_counts.get(dept, 0) + 1
            score = scores.at[idx, "retention_score"]
            note = (
                " Score is below the plan's review threshold; requires explicit "
                "sign-off before notice."
                if bool(scores.at[idx, "requires_approval"]) else ""
            )
            scores.at[idx, "rationale"] = (
                f"Lowest retention score in comparison group "
                f"'{scores.at[idx, 'comparison_group']}' "
                f"({float(score):.1f}/100: {scores.at[idx, 'score_breakdown']})."
                + note
            )

        self._flag_boundary_ties(scores, pool, report)
        self._check_rank_order(scores, report)

        report.achieved_savings = round(running, 2)
        cut_list = scores.loc[scores["selected"]].copy()
        return cut_list, scores

    @staticmethod
    def _check_rank_order(scores: pd.DataFrame, report: SelectionReport) -> None:
        """Invariant: within a comparison group, no one is selected while a
        lower-scored peer is retained.

        A violation means something other than the stated criteria decided the
        outcome — a department cap, a cost tie-break, a bug. Whatever the cause,
        a list that cuts the higher scorer and keeps the lower one cannot be
        explained using the criteria it claims to apply, so it is an error.
        """
        eligible = scores.loc[
            scores["retention_score"].notna()
            & scores["selection_status"].isin(["eligible", "tie_at_boundary"])
        ]
        for grp, sub in eligible.groupby("comparison_group"):
            sel = sub.loc[sub["selected"]]
            unsel = sub.loc[~sub["selected"]]
            if sel.empty or unsel.empty:
                continue
            worst_retained = float(unsel["retention_score"].astype("Float64").min())
            inverted = sel.loc[
                sel["retention_score"].astype("Float64") > worst_retained
            ]
            for _, row in inverted.iterrows():
                report.add(
                    Severity.ERROR, "RANK_ORDER_VIOLATION",
                    f"In comparison group '{grp}', employee "
                    f"{row['employee_id']} scored "
                    f"{float(row['retention_score']):.1f} and is on the cut list "
                    f"while a peer scoring {worst_retained:.1f} is retained. The "
                    f"recommendation contradicts its own criteria and must not be "
                    f"issued until the cause is resolved.",
                    department=row.get("department"),
                    employee_id=str(row["employee_id"]),
                )

    def _flag_boundary_ties(
        self, scores: pd.DataFrame, pool: pd.DataFrame, report: SelectionReport
    ) -> None:
        """If the cut boundary splits a group of equal scores, say so."""
        sel = scores.loc[pool.index, "selected"]
        if sel.all() or not sel.any():
            return
        selected_idx = [i for i in pool.index if bool(scores.at[i, "selected"])]
        unselected_idx = [i for i in pool.index if not bool(scores.at[i, "selected"])]
        if not selected_idx or not unselected_idx:
            return
        boundary_score = scores.at[selected_idx[-1], "retention_score"]
        if pd.isna(boundary_score):
            return
        tied = [
            i for i in unselected_idx
            if pd.notna(scores.at[i, "retention_score"])
            and math.isclose(float(scores.at[i, "retention_score"]), float(boundary_score))
        ]
        if not tied:
            return
        tied_selected = [
            i for i in selected_idx
            if math.isclose(float(scores.at[i, "retention_score"]), float(boundary_score))
        ]
        for i in tied + tied_selected:
            scores.at[i, "selection_status"] = "tie_at_boundary"
        ids = [str(scores.at[i, "employee_id"]) for i in tied + tied_selected]
        report.add(
            Severity.ERROR, "TIE_AT_CUT_BOUNDARY",
            f"{len(ids)} employee(s) share the boundary score of "
            f"{float(boundary_score):.1f} but only some fit within the target: "
            f"{', '.join(ids)}. The engine will not break this tie by row order. "
            f"A human must either add a documented tie-breaking criterion or "
            f"decide explicitly and record why.",
        )

    # -- wrap up ----------------------------------------------------------
    def _finalize(
        self, scores: pd.DataFrame, cut_list: pd.DataFrame, report: SelectionReport
    ) -> None:
        for dept, group in scores.groupby("department", dropna=False):
            dname = dept if isinstance(dept, str) else "(unassigned)"
            sel = group.loc[group["selected"]]
            dplan = self.plan.for_department(dname)
            report.department_summary.append({
                "department": dname,
                "mode": dplan.mode if dplan else "none",
                "evaluated": int(len(group)),
                "selected": int(len(sel)),
                "savings": float(sel["annual_cost"].astype("Float64").sum() or 0.0),
            })

        rankable = scores.loc[scores["selection_status"].isin(["eligible", "tie_at_boundary"])]
        if len(rankable):
            share = float(rankable["selected"].sum()) / len(rankable)
            if share >= 0.5:
                report.add(
                    Severity.WARNING, "POOL_LARGELY_EXHAUSTED",
                    f"The recommendation selects {share:.0%} of the rankable pool. "
                    f"At that depth the score stops distinguishing anyone — the "
                    f"criteria are no longer doing the selecting, the target is. "
                    f"Revisit whether the savings target is achievable through "
                    f"selection, and expect WARN thresholds to be in play.",
                )

        if not report.target_met:
            report.add(
                Severity.WARNING, "TARGET_NOT_MET",
                f"The recommendation reaches ${report.achieved_savings:,.0f} of the "
                f"${report.cost_savings_target:,.0f} target, a shortfall of "
                f"${report.cost_savings_target - report.achieved_savings:,.0f}. "
                f"Closing it requires widening the eligible pool, raising a "
                f"department cap, resolving excluded records, or accepting less "
                f"savings — not lowering the evidentiary standard.",
            )

        flagged = cut_list.loc[cut_list["legal_review_flags"].astype("string").fillna("") != ""]
        if len(flagged):
            report.add(
                Severity.WARNING, "SELECTIONS_REQUIRE_LEGAL_REVIEW",
                f"{len(flagged)} selected employee(s) carry a condition requiring "
                f"counsel sign-off before notice (protected leave, union "
                f"membership, or sponsored work authorization). These flags did "
                f"not affect scoring.",
            )

        review = scores.loc[scores["selection_status"].isin(
            ["manual_review", "tie_at_boundary", "excluded_insufficient_data"]
        )]
        if len(review):
            report.add(
                Severity.WARNING, "REVIEW_QUEUE_NOT_EMPTY",
                f"{len(review)} employee(s) need a human decision before this list "
                f"is complete. The cut list is provisional until the queue is cleared.",
            )

        report.add(
            Severity.INFO, "PROTECTED_FIELDS_EXCLUDED",
            "Scoring read no protected characteristic. Age, sex, race, "
            "disability, and veteran status were withheld from the engine by "
            "design; run the Adverse Impact Analyzer on this output to measure "
            "whether the result nonetheless falls unevenly.",
        )


# ---------------------------------------------------------------------------
# Cut list presentation
# ---------------------------------------------------------------------------

CUT_LIST_COLUMNS = [
    "selection_rank", "employee_id", "job_title", "department", "job_level",
    "worksite_name", "comparison_group", "selection_mode", "retention_score",
    "score_breakdown", "annualized_pay", "annual_cost", "cumulative_savings",
    "tenure_years", "rationale", "legal_review_flags", "legal_review_notes",
]

#: Blank columns the reviewing human fills in. The list is not final until they do.
OVERRIDE_COLUMNS = [
    "human_decision", "decision_maker", "decision_date", "override_reason",
]


def format_cut_list(cut_list: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in CUT_LIST_COLUMNS if c in cut_list.columns]
    out = cut_list[cols].copy()
    out = out.sort_values(
        ["selection_rank"], na_position="first", kind="mergesort"
    ).reset_index(drop=True)
    for c in OVERRIDE_COLUMNS:
        out[c] = pd.Series([pd.NA] * len(out), dtype="string")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from .workforce_data import load_workforce_csv

    ap = argparse.ArgumentParser(
        description="Score employees and build a recommended RIF cut list."
    )
    ap.add_argument("csv_path", help="Workforce CSV (raw; it is ingested first).")
    ap.add_argument("--plan", required=True, help="Path to the RIF plan YAML/JSON.")
    ap.add_argument("--as-of", default=None, help="Date for tenure/pay math (YYYY-MM-DD).")
    ap.add_argument("--outdir", default=None, help="Directory for outputs.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    as_of = args.as_of or (str(plan.as_of_date) if plan.as_of_date else None)

    ingest = load_workforce_csv(args.csv_path, as_of=as_of)
    if ingest.report.is_blocking:
        print("Ingestion is blocking; fix the roster before running selection.")
        print(ingest.report.to_markdown())
        return 2

    result = SelectionEngine(plan).run(ingest.data)
    result.cut_list = format_cut_list(result.cut_list)

    if not args.quiet:
        print(result.report.to_markdown())

    if args.outdir:
        paths = result.write(args.outdir)
        print("\nWrote:")
        for k, p in paths.items():
            print(f"  {k}: {p}")

    if any(f.severity == Severity.ERROR for f in result.report.findings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
