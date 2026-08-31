"""
adverse_impact.py
=================

Adverse Impact Analyzer for the California RIF Copilot (box 4).

Consumes the scored roster produced by ``selection_criteria.py`` and measures
whether the proposed selections fall disproportionately on a protected class.

What it does
------------
1. Computes selection and retention rates for every protected group.
2. Applies the four-fifths (80%) rule against the most-favored group.
3. Runs a two-tailed Fisher's exact test and a standard-deviation analysis on
   each comparison, because the four-fifths rule alone flips on one person in
   a small unit and courts look at statistical significance too.
4. Repeats the analysis inside each decision-making unit — department,
   worksite, job level — because a clean company-wide number routinely hides a
   disparity in the unit where the decision was actually made.
5. Reports how fragile each result is: the number of individual decisions that
   would have to differ to change the finding.

What it deliberately does NOT do
--------------------------------
It does not tell you who to swap. If a group shows impact, the answer is never
"remove someone from the cut list because of their protected class" — that is
itself unlawful disparate treatment in most circumstances, and it is a
straight line from a well-meaning fix to a discrimination claim brought by the
person swapped in. This module surfaces the disparity and routes it to counsel.
The lawful responses run through the *criteria*: whether they are job-related
and consistent with business necessity, whether they were applied uniformly,
whether a less discriminatory alternative exists that serves the same business
need. Those are legal determinations, not arithmetic.

Privilege
---------
Adverse impact analyses are commonly run at the direction of counsel so the
results are covered by attorney-client privilege. A self-serve analysis sitting
in a shared drive is discoverable and, if it shows impact that was not acted
on, is the single most damaging document in the case. Talk to your employment
counsel about how this output should be generated, labeled, and stored before
you run it on a real scenario.

Usage
-----
    from .adverse_impact import AdverseImpactAnalyzer

    analyzer = AdverseImpactAnalyzer()
    analysis = analyzer.run(selection_result.scores)

    analysis.results        # one row per group comparison
    analysis.flagged        # comparisons that tripped a threshold
    print(analysis.report.to_markdown())

CLI
---
    python adverse_impact.py roster.csv --plan rif_plan.yaml --outdir ./out
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .workforce_data import Severity

__all__ = [
    "ProtectedClass",
    "PROTECTED_CLASSES",
    "ImpactComparison",
    "AdverseImpactAnalyzer",
    "AdverseImpactResult",
    "ImpactReport",
    "fisher_exact_two_tailed",
    "standard_deviations",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

#: The four-fifths rule from the Uniform Guidelines on Employee Selection
#: Procedures (29 CFR 1607.4(D)). A ratio below this is a rule-of-thumb
#: indicator, not a legal finding.
FOUR_FIFTHS = 0.80

#: Secondary practical-significance screen on the *termination* rate. When the
#: overall selection rate is low, the retention-basis four-fifths ratio is
#: mathematically compressed toward 1.0 and rarely trips even on a large
#: disparity: cutting 17% of one group and 10% of another is a 1.7x difference
#: but a 0.92 four-fifths ratio. This catches that case.
SELECTION_RATIO_THRESHOLD = 1.25

#: Conventional threshold for statistical significance in discrimination
#: analysis. Courts have treated disparities beyond 2-3 standard deviations as
#: probative; 1.96 SD corresponds to p = .05 two-tailed.
SD_THRESHOLD = 2.0
P_THRESHOLD = 0.05

#: Below this many people in a group, rate comparisons are too unstable to
#: interpret. The analysis still runs, but every result carries the caveat.
MIN_GROUP_SIZE = 10

#: Below this, the four-fifths ratio is reported but should not be relied on at
#: all; a single decision moves it by tens of percentage points.
TINY_GROUP_SIZE = 5

#: If more than this share of a class is undisclosed, the test has lost so much
#: power that a "no flag" result means very little.
MAX_UNDISCLOSED_SHARE = 0.20


# ---------------------------------------------------------------------------
# Protected class definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProtectedClass:
    """A protected characteristic and how to read it off the roster."""

    name: str
    column: str
    #: Values meaning "the employee did not disclose". Excluded from the test
    #: and counted separately, never folded into a group.
    undisclosed: tuple[str, ...] = ("not_disclosed",)
    #: Legal basis, surfaced in the report so the reader knows what is at stake.
    authority: str = ""
    #: If set, collapse the column to a two-group comparison using this label.
    binary_label: str | None = None


PROTECTED_CLASSES: tuple[ProtectedClass, ...] = (
    ProtectedClass(
        "Age 40+", "age_40_plus",
        undisclosed=(),
        authority="ADEA; FEHA (Gov. Code 12940). California's OWBPA disclosure "
                  "rules also require age data in group termination releases.",
        binary_label="Age 40+",
    ),
    ProtectedClass(
        "Sex", "gender",
        authority="Title VII; FEHA. California protects gender identity and "
                  "expression as well as sex.",
    ),
    ProtectedClass(
        "Race / Ethnicity", "race_ethnicity",
        authority="Title VII; 42 U.S.C. 1981; FEHA.",
    ),
    ProtectedClass(
        "Disability", "disability_status",
        undisclosed=("not_disclosed",),
        authority="ADA; FEHA, which uses a broader definition of disability "
                  "than federal law.",
        binary_label="yes",
    ),
    ProtectedClass(
        "Veteran Status", "veteran_status",
        undisclosed=("not_disclosed",),
        authority="USERRA; VEVRAA; Cal. Mil. & Vet. Code.",
        binary_label="yes",
    ),
)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _log_choose(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _hypergeom_logpmf(k: int, N: int, K: int, n: int) -> float:
    """log P(X = k) for X ~ Hypergeometric(N population, K successes, n draws)."""
    return _log_choose(K, k) + _log_choose(N - K, n - k) - _log_choose(N, n)


def fisher_exact_two_tailed(a: int, b: int, c: int, d: int) -> float:
    """Two-tailed Fisher's exact test p-value for the 2x2 table.

        [[a, b],
         [c, d]]

    Implemented directly so the module has no hard dependency on SciPy. Uses
    the standard "sum of tables no more probable than the observed" definition.
    """
    n1, n2 = a + b, c + d
    m1 = a + c
    N = n1 + n2
    if N == 0 or n1 == 0 or n2 == 0 or m1 == 0 or (b + d) == 0:
        return 1.0

    observed = _hypergeom_logpmf(a, N, m1, n1)
    # Floating point slack so tables of equal probability are included.
    tol = 1e-9
    total = 0.0
    lo = max(0, n1 - (N - m1))
    hi = min(n1, m1)
    for k in range(lo, hi + 1):
        lp = _hypergeom_logpmf(k, N, m1, n1)
        if lp <= observed + tol:
            total += math.exp(lp)
    return float(min(1.0, total))


def standard_deviations(
    group_selected: int, group_total: int, other_selected: int, other_total: int
) -> float | None:
    """Signed standard deviations of the group's selection count under the
    hypergeometric null (the "standard deviation analysis" courts reference).

    Positive means the group was selected *more* than expected.
    """
    N = group_total + other_total
    K = group_selected + other_selected
    n = group_total
    if N < 2 or n == 0 or K == 0 or K == N:
        return None
    expected = n * K / N
    variance = n * (K / N) * (1 - K / N) * ((N - n) / (N - 1))
    if variance <= 0:
        return None
    return float((group_selected - expected) / math.sqrt(variance))


def _flip_count(
    grp_sel: int, grp_tot: int, ref_sel: int, ref_tot: int, threshold: float = FOUR_FIFTHS
) -> int | None:
    """How many fewer selections in the group would clear the four-fifths rule.

    Answers "how fragile is this finding?". A flag that disappears if one
    person's outcome changes is a very different fact from one that needs six.
    """
    if grp_tot == 0 or ref_tot == 0:
        return None
    ref_ret = (ref_tot - ref_sel) / ref_tot
    if ref_ret <= 0:
        return None
    for moved in range(0, grp_sel + 1):
        grp_ret = (grp_tot - (grp_sel - moved)) / grp_tot
        if ref_ret > 0 and (grp_ret / ref_ret) >= threshold:
            return moved
    return None


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class ImpactComparison:
    """One protected group measured against the most-favored group."""

    protected_class: str
    unit_type: str          # overall | department | worksite | job_level
    unit: str
    group: str
    reference_group: str
    group_total: int
    group_selected: int
    reference_total: int
    reference_selected: int
    impact_ratio: float | None       # retention basis; <0.80 indicates impact
    selection_rate_ratio: float | None  # termination basis; >1.25 mirrors it
    fisher_p: float | None
    std_deviations: float | None
    shortfall: float | None          # selections above statistical expectation
    flip_count: int | None
    flags: tuple[str, ...] = ()
    notes: str = ""
    #: Whether the group is large enough for the four-fifths ratio to count as
    #: practical significance. Small groups still get a statistical test.
    interpretable: bool = True

    @property
    def group_selection_rate(self) -> float:
        return self.group_selected / self.group_total if self.group_total else 0.0

    @property
    def reference_selection_rate(self) -> float:
        return (
            self.reference_selected / self.reference_total
            if self.reference_total else 0.0
        )

    @property
    def severity(self) -> str:
        """How loudly this comparison should be surfaced."""
        practical = (
            self.interpretable
            and self.impact_ratio is not None
            and self.impact_ratio < FOUR_FIFTHS
        )
        # The termination-rate view, which stays sensitive when overall
        # selection rates are low and the four-fifths ratio goes numb.
        practical_alt = (
            self.interpretable
            and self.selection_rate_ratio is not None
            and self.selection_rate_ratio >= SELECTION_RATIO_THRESHOLD
        )
        statistical = (
            (self.std_deviations is not None and abs(self.std_deviations) >= SD_THRESHOLD)
            or (self.fisher_p is not None and self.fisher_p < P_THRESHOLD)
        )
        if (practical or practical_alt) and statistical:
            return Severity.ERROR
        if practical or practical_alt or statistical:
            return Severity.WARNING
        return Severity.INFO

    @property
    def diverges(self) -> bool:
        """True when the two practical screens disagree — the four-fifths rule
        passes but the termination-rate comparison does not."""
        return (
            self.interpretable
            and self.impact_ratio is not None
            and self.impact_ratio >= FOUR_FIFTHS
            and self.selection_rate_ratio is not None
            and self.selection_rate_ratio >= SELECTION_RATIO_THRESHOLD
        )

    @property
    def verdict(self) -> str:
        s = self.severity
        if s == Severity.ERROR:
            return "Impact indicated"
        if s == Severity.WARNING:
            return "Review"
        return "No flag"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protected_class": self.protected_class,
            "unit_type": self.unit_type,
            "unit": self.unit,
            "group": self.group,
            "reference_group": self.reference_group,
            "group_total": self.group_total,
            "group_selected": self.group_selected,
            "group_selection_rate": round(self.group_selection_rate, 4),
            "reference_total": self.reference_total,
            "reference_selected": self.reference_selected,
            "reference_selection_rate": round(self.reference_selection_rate, 4),
            "impact_ratio": None if self.impact_ratio is None else round(self.impact_ratio, 4),
            "selection_rate_ratio": (
                None if self.selection_rate_ratio is None
                else round(self.selection_rate_ratio, 4)
            ),
            "fisher_p": None if self.fisher_p is None else round(self.fisher_p, 5),
            "std_deviations": (
                None if self.std_deviations is None else round(self.std_deviations, 3)
            ),
            "shortfall": None if self.shortfall is None else round(self.shortfall, 2),
            "flip_count": self.flip_count,
            "verdict": self.verdict,
            "severity": self.severity,
            "interpretable": self.interpretable,
            "flags": "|".join(self.flags),
            "notes": self.notes,
        }


@dataclass
class ImpactFinding:
    severity: str
    code: str
    message: str
    protected_class: str | None = None
    unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity, "code": self.code, "message": self.message,
            "protected_class": self.protected_class, "unit": self.unit,
        }


@dataclass
class ImpactReport:
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
    scenario: str = ""
    population: int = 0
    selected: int = 0
    comparisons: list[ImpactComparison] = field(default_factory=list)
    findings: list[ImpactFinding] = field(default_factory=list)
    coverage: dict[str, dict[str, Any]] = field(default_factory=dict)
    skipped: list[dict[str, Any]] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, **kw: Any) -> None:
        self.findings.append(ImpactFinding(severity, code, message, **kw))

    # -- querying --------------------------------------------------------
    @property
    def flagged(self) -> list[ImpactComparison]:
        return [c for c in self.comparisons if c.severity in (Severity.ERROR, Severity.WARNING)]

    @property
    def indicated(self) -> list[ImpactComparison]:
        return [c for c in self.comparisons if c.severity == Severity.ERROR]

    def class_verdicts(self) -> dict[str, str]:
        """Worst verdict per protected class, matching the dashboard summary."""
        out: dict[str, str] = {}
        rank = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        for c in self.comparisons:
            cur = out.get(c.protected_class)
            if cur is None or rank[c.severity] < rank[_sev_of_verdict(cur)]:
                out[c.protected_class] = c.verdict
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "generated_at": self.generated_at,
            "population": self.population,
            "selected": self.selected,
            "overall_selection_rate": (
                round(self.selected / self.population, 4) if self.population else 0.0
            ),
            "comparisons": len(self.comparisons),
            "flagged": len(self.flagged),
            "impact_indicated": len(self.indicated),
            "class_verdicts": self.class_verdicts(),
        }

    # -- output ----------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self.comparisons:
            return pd.DataFrame()
        df = pd.DataFrame([c.to_dict() for c in self.comparisons])
        rank = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
        df["_s"] = df["severity"].map(rank).fillna(9)
        unit_rank = {"overall": 0, "department": 1, "worksite": 2, "job_level": 3}
        df["_u"] = df["unit_type"].map(unit_rank).fillna(9)
        return (
            df.sort_values(["_s", "_u", "protected_class", "unit"])
            .drop(columns=["_s", "_u"])
            .reset_index(drop=True)
        )

    def findings_dataframe(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(
                columns=["severity", "code", "message", "protected_class", "unit"]
            )
        return pd.DataFrame([f.to_dict() for f in self.findings])

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "coverage": self.coverage,
            "skipped": self.skipped,
            "comparisons": [c.to_dict() for c in self.comparisons],
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self, max_rows: int = 40) -> str:
        s = self.summary()
        L: list[str] = []
        L.append("# Adverse Impact Analysis")
        L.append("")
        L.append(
            "> **Prepared at the direction of counsel / privileged and "
            "confidential — confirm the correct label with your employment "
            "counsel before circulating.**"
        )
        L.append("")
        L.append(f"**Scenario:** {s['scenario'] or '(unnamed)'}  ")
        L.append(f"**Generated:** {s['generated_at']}  ")
        L.append(
            f"**Population:** {s['population']} employees, {s['selected']} selected "
            f"({s['overall_selection_rate']:.1%})  "
        )
        L.append("")

        L.append("## Summary by protected class")
        L.append("")
        L.append("| Protected class | Verdict |")
        L.append("|---|---|")
        for cls, verdict in s["class_verdicts"].items():
            L.append(f"| {cls} | {verdict} |")
        L.append("")

        # Coverage first: a "no flag" on 40% coverage is not a clean bill.
        if self.coverage:
            L.append("## Data coverage")
            L.append("")
            L.append("| Class | Analyzable | Undisclosed | Missing |")
            L.append("|---|---|---|---|")
            for cls, cov in self.coverage.items():
                L.append(
                    f"| {cls} | {cov['analyzable']} | {cov['undisclosed']} | "
                    f"{cov['missing']} |"
                )
            L.append("")

        flagged = [c for c in self.comparisons if c.severity != Severity.INFO]
        L.append("## Flagged comparisons")
        L.append("")
        if not flagged:
            L.append("No comparison tripped the four-fifths rule or reached "
                     "statistical significance.")
            L.append("")
        else:
            L.append(
                "| Class | Unit | Group | Sel. rate | Ref. rate | Rate ratio | "
                "4/5 ratio | SD | p | Flip | Verdict |"
            )
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for c in flagged[:max_rows]:
                ir = "—" if c.impact_ratio is None else f"{c.impact_ratio:.2f}"
                sd = "—" if c.std_deviations is None else f"{c.std_deviations:+.2f}"
                p = "—" if c.fisher_p is None else f"{c.fisher_p:.3f}"
                flip = "—" if c.flip_count is None else str(c.flip_count)
                unit = c.unit if c.unit_type != "overall" else "All"
                if not c.interpretable:
                    ir += "*"
                rr = (
                    "—" if c.selection_rate_ratio is None
                    else f"{c.selection_rate_ratio:.2f}x"
                )
                L.append(
                    f"| {c.protected_class} | {unit} | {c.group} | "
                    f"{c.group_selection_rate:.1%} ({c.group_selected}/{c.group_total}) | "
                    f"{c.reference_selection_rate:.1%} | {rr} | {ir} | {sd} | {p} | "
                    f"{flip} | {c.verdict} |"
                )
            if len(flagged) > max_rows:
                L.append(f"| … | | | | | | | | | | {len(flagged) - max_rows} more |")
            L.append("")
            L.append(
                "A ratio marked `*` comes from a group below the minimum size; it "
                "is shown for completeness but is not treated as practical "
                "significance on its own."
            )
            L.append("")
            L.append(
                "*Flip* is the number of individual selections that would have to "
                "differ for the four-fifths result to change. A flag with a flip "
                "count of 1 is a very different fact from one with a flip count of 6."
            )
            L.append("")

        if self.findings:
            L.append("## Notes")
            L.append("")
            for f in self.findings:
                where = f" ({f.unit})" if f.unit else ""
                L.append(f"- **[{f.severity}] {f.code}**{where} — {f.message}")
            L.append("")

        L.append("## How to read this")
        L.append("")
        L.append(
            "The four-fifths rule is a screening heuristic from the Uniform "
            "Guidelines, not a legal standard. A ratio under 0.80 does not "
            "establish discrimination, and a ratio over 0.80 does not establish "
            "its absence — particularly in small units, where one person's "
            "outcome swings the ratio dramatically. That is why every row also "
            "carries a significance test and a fragility count."
        )
        L.append("")
        L.append("**If a comparison is flagged, the lawful paths run through the "
                 "criteria, not the people:**")
        L.append("")
        L.append("1. Re-examine whether each criterion is job-related and "
                 "consistent with business necessity, and whether it was applied "
                 "consistently across managers and units.")
        L.append("2. Look for a less discriminatory alternative that serves the "
                 "same business need — a different weighting, a broader "
                 "comparison group, an objective criterion in place of a "
                 "subjective one.")
        L.append("3. Check whether the disparity traces to an upstream input. "
                 "Performance ratings are the usual culprit; if raters differed, "
                 "the score inherited it.")
        L.append("4. Document the business rationale contemporaneously, and take "
                 "the analysis to counsel before acting on it.")
        L.append("")
        L.append(
            "**Do not adjust individual selections to move the ratio.** Removing "
            "or adding a specific person because of their protected class is "
            "disparate treatment, and it creates a claim for whoever is swapped "
            "in. This tool will not recommend swaps, and neither should anyone "
            "reading it. Fix the criteria or accept and document the result — "
            "under advice of counsel."
        )
        L.append("")
        L.append("---")
        L.append(
            "_Statistical screening only. This is not a legal opinion and does "
            "not determine whether discrimination occurred. Small groups, "
            "undisclosed demographics, and multiple comparisons all affect how "
            "much weight any single row can bear._"
        )
        return "\n".join(L)


def _sev_of_verdict(verdict: str) -> str:
    return {
        "Impact indicated": Severity.ERROR,
        "Review": Severity.WARNING,
        "No flag": Severity.INFO,
    }.get(verdict, Severity.INFO)


@dataclass
class AdverseImpactResult:
    results: pd.DataFrame
    report: ImpactReport

    @property
    def flagged(self) -> pd.DataFrame:
        if self.results.empty:
            return self.results
        return self.results.loc[self.results["severity"] != Severity.INFO].copy()

    def write(self, outdir: str | Path, stem: str = "adverse_impact") -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "report_md": outdir / f"{stem}_report.md",
            "results": outdir / f"{stem}_comparisons.csv",
            "findings": outdir / f"{stem}_findings.csv",
            "report_json": outdir / f"{stem}_report.json",
        }
        self.results.to_csv(paths["results"], index=False)
        self.report.findings_dataframe().to_csv(paths["findings"], index=False)
        self.report.to_json(paths["report_json"])
        paths["report_md"].write_text(self.report.to_markdown(), encoding="utf-8")
        return paths


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class AdverseImpactAnalyzer:
    """Measures disparate impact in a proposed selection."""

    def __init__(
        self,
        classes: Sequence[ProtectedClass] = PROTECTED_CLASSES,
        unit_columns: Sequence[str] = ("department", "worksite_name", "job_level"),
        min_group_size: int = MIN_GROUP_SIZE,
        selected_column: str = "selected",
    ) -> None:
        self.classes = tuple(classes)
        self.unit_columns = tuple(unit_columns)
        self.min_group_size = min_group_size
        self.selected_column = selected_column

    # -- public ----------------------------------------------------------
    def run(self, scores: pd.DataFrame, scenario: str = "") -> AdverseImpactResult:
        report = ImpactReport(scenario=scenario)
        self._skipped: list[tuple[str, str, str, int]] = []

        df = self._prepare(scores, report)
        if df.empty:
            return AdverseImpactResult(pd.DataFrame(), report)

        report.population = len(df)
        report.selected = int(df["_selected"].sum())

        if report.selected == 0:
            report.add(
                Severity.WARNING, "NO_SELECTIONS",
                "No employees are marked as selected, so there is nothing to "
                "analyze. Run the Selection Criteria Engine first.",
            )
            return AdverseImpactResult(pd.DataFrame(), report)

        for pc in self.classes:
            self._analyze_class(df, pc, report)

        self._add_global_notes(df, report)
        return AdverseImpactResult(report.to_dataframe(), report)

    # -- preparation -----------------------------------------------------
    def _prepare(self, scores: pd.DataFrame, report: ImpactReport) -> pd.DataFrame:
        if scores is None or scores.empty:
            report.add(Severity.ERROR, "EMPTY_INPUT", "No scored roster provided.")
            return pd.DataFrame()

        df = scores.copy()
        if self.selected_column not in df.columns:
            report.add(
                Severity.ERROR, "MISSING_SELECTED_COLUMN",
                f"Input has no {self.selected_column!r} column. This module reads "
                f"the output of the Selection Criteria Engine, which must include "
                f"both selected and retained employees.",
            )
            return pd.DataFrame()

        df["_selected"] = df[self.selected_column].fillna(False).astype(bool)

        # Employees who were never in the decision pool must not dilute the
        # analysis: including people who could not have been selected understates
        # every rate and can hide a real disparity.
        if "selection_status" in df.columns:
            out_of_pool = {"excluded_no_plan", "not_targeted"}
            mask = df["selection_status"].isin(out_of_pool) & ~df["_selected"]
            if mask.any():
                report.add(
                    Severity.INFO, "OUT_OF_POOL_EXCLUDED",
                    f"{int(mask.sum())} employee(s) were never in the decision "
                    f"pool (no plan, or a position not targeted) and are excluded "
                    f"from rate calculations.",
                )
                df = df.loc[~mask]

        return df.reset_index(drop=True)

    # -- per-class analysis ----------------------------------------------
    def _class_series(
        self, df: pd.DataFrame, pc: ProtectedClass, report: ImpactReport
    ) -> pd.Series | None:
        if pc.column not in df.columns:
            report.add(
                Severity.WARNING, "CLASS_COLUMN_ABSENT",
                f"{pc.name} cannot be analyzed: column {pc.column!r} is not in the "
                f"roster. This class is untested, which is not the same as clear.",
                protected_class=pc.name,
            )
            return None

        raw = df[pc.column]

        # Booleans (age_40_plus, and any yes/no coded as bool) become labels.
        if raw.dtype == "boolean" or raw.dtype == bool:
            label = pc.binary_label or pc.name
            s = raw.map(
                lambda v: pd.NA if pd.isna(v) else (label if bool(v) else f"Not {label}")
            )
            return s.astype("string")

        s = raw.astype("string")
        if pc.binary_label:
            lab = pc.binary_label
            s = s.map(
                lambda v: pd.NA if pd.isna(v) else (
                    v if v in (lab,) + tuple(pc.undisclosed) else f"not_{lab}"
                )
            ).astype("string")
        return s

    def _analyze_class(
        self, df: pd.DataFrame, pc: ProtectedClass, report: ImpactReport
    ) -> None:
        series = self._class_series(df, pc, report)
        if series is None:
            report.coverage[pc.name] = {
                "analyzable": 0, "undisclosed": 0, "missing": len(df),
            }
            return

        missing = int(series.isna().sum())
        undisclosed = int(series.isin(list(pc.undisclosed)).sum()) if pc.undisclosed else 0
        analyzable_mask = series.notna() & ~series.isin(list(pc.undisclosed))
        analyzable = int(analyzable_mask.sum())

        report.coverage[pc.name] = {
            "analyzable": analyzable, "undisclosed": undisclosed, "missing": missing,
        }

        if analyzable == 0:
            report.add(
                Severity.WARNING, "CLASS_NOT_ANALYZABLE",
                f"{pc.name}: no employee has a usable value, so no test could be "
                f"run. Untested, not clear.",
                protected_class=pc.name,
            )
            return

        excluded = missing + undisclosed
        if excluded and excluded / len(df) > MAX_UNDISCLOSED_SHARE:
            report.add(
                Severity.WARNING, "LOW_CLASS_COVERAGE",
                f"{pc.name}: {excluded} of {len(df)} employees "
                f"({excluded / len(df):.0%}) are undisclosed or missing and were "
                f"excluded from the test. With that much of the population "
                f"unmeasured, a 'no flag' result carries little weight.",
                protected_class=pc.name,
            )

        work = df.loc[analyzable_mask].copy()
        work["_group"] = series.loc[analyzable_mask]

        # Overall, then inside each decision-making unit.
        self._compare_within(work, pc, "overall", "All", report)
        for col in self.unit_columns:
            if col not in work.columns:
                continue
            for unit, sub in work.groupby(work[col].astype("string"), dropna=True):
                if pd.isna(unit) or len(sub) < TINY_GROUP_SIZE:
                    continue
                self._compare_within(sub, pc, _unit_type(col), str(unit), report)

    def _compare_within(
        self,
        frame: pd.DataFrame,
        pc: ProtectedClass,
        unit_type: str,
        unit: str,
        report: ImpactReport,
    ) -> None:
        counts = frame.groupby("_group")["_selected"].agg(["size", "sum"])
        counts = counts.rename(columns={"size": "total", "sum": "selected"})
        if len(counts) < 2:
            return  # nothing to compare against

        counts["retention_rate"] = (counts["total"] - counts["selected"]) / counts["total"]

        # The reference is the most-favored group: highest retention rate among
        # groups large enough to be a stable benchmark. If no group qualifies,
        # the unit is not analyzable at all — picking a 2-person group as the
        # benchmark manufactures disparities that mean nothing and buries the
        # findings that do.
        eligible_ref = counts.loc[counts["total"] >= self.min_group_size]
        if eligible_ref.empty:
            self._skipped.append((pc.name, unit_type, unit, int(counts["total"].sum())))
            return
        ref_group = eligible_ref["retention_rate"].idxmax()
        ref_total = int(counts.at[ref_group, "total"])
        ref_selected = int(counts.at[ref_group, "selected"])
        ref_retention = float(counts.at[ref_group, "retention_rate"])

        for group, row in counts.iterrows():
            if group == ref_group:
                continue
            g_total, g_selected = int(row["total"]), int(row["selected"])

            # Groups this small cannot support a rate comparison. Reporting them
            # as "Review" would be false precision, so they are counted as
            # untested and named in the report instead.
            if g_total < TINY_GROUP_SIZE:
                self._skipped.append((pc.name, unit_type, f"{unit} / {group}", g_total))
                continue

            g_retention = float(row["retention_rate"])

            impact_ratio = g_retention / ref_retention if ref_retention > 0 else None
            g_rate = g_selected / g_total if g_total else 0.0
            r_rate = ref_selected / ref_total if ref_total else 0.0
            sel_ratio = (g_rate / r_rate) if r_rate > 0 else None

            p = fisher_exact_two_tailed(
                g_selected, g_total - g_selected,
                ref_selected, ref_total - ref_selected,
            )
            sd = standard_deviations(g_selected, g_total, ref_selected, ref_total)
            expected = (
                g_total * (g_selected + ref_selected) / (g_total + ref_total)
                if (g_total + ref_total) else None
            )
            shortfall = (g_selected - expected) if expected is not None else None
            flip = _flip_count(g_selected, g_total, ref_selected, ref_total)

            flags: list[str] = []
            notes: list[str] = []
            if g_total < self.min_group_size:
                flags.append("SMALL_GROUP")
                notes.append(
                    f"{g_total} employee(s) in this group, below the minimum of "
                    f"{self.min_group_size}. The four-fifths ratio is reported but "
                    f"is not treated as practical significance on its own."
                )
            if flip is not None and flip <= 1 and impact_ratio is not None and impact_ratio < FOUR_FIFTHS:
                flags.append("FRAGILE")
                notes.append("A single different outcome would clear the rule.")
            if impact_ratio is not None and impact_ratio < FOUR_FIFTHS:
                flags.append("FOUR_FIFTHS")
            if sd is not None and abs(sd) >= SD_THRESHOLD:
                flags.append("STATISTICALLY_SIGNIFICANT")
            if p is not None and p < P_THRESHOLD:
                flags.append("P_LT_05")

            comp = ImpactComparison(
                protected_class=pc.name, unit_type=unit_type, unit=unit,
                group=str(group), reference_group=str(ref_group),
                group_total=g_total, group_selected=g_selected,
                reference_total=ref_total, reference_selected=ref_selected,
                impact_ratio=impact_ratio, selection_rate_ratio=sel_ratio,
                fisher_p=p, std_deviations=sd, shortfall=shortfall,
                flip_count=flip, flags=tuple(flags), notes=" ".join(notes),
                interpretable=g_total >= self.min_group_size,
            )
            report.comparisons.append(comp)

            if comp.severity == Severity.ERROR:
                where = "company-wide" if unit_type == "overall" else f"in {unit}"
                report.add(
                    Severity.ERROR, "ADVERSE_IMPACT_INDICATED",
                    f"{pc.name}: group '{group}' {where} was selected at "
                    f"{comp.group_selection_rate:.1%} versus "
                    f"{comp.reference_selection_rate:.1%} for '{ref_group}' "
                    f"(four-fifths ratio {impact_ratio:.2f}, "
                    f"{sd:+.2f} SD, p={p:.3f}). This clears both the practical and "
                    f"the statistical threshold and needs counsel review before "
                    f"the list is issued. {pc.authority}",
                    protected_class=pc.name, unit=unit,
                )

    # -- global notes -----------------------------------------------------
    def _add_global_notes(self, df: pd.DataFrame, report: ImpactReport) -> None:
        if self._skipped:
            by_class: dict[str, int] = {}
            for cls, _ut, _u, n in self._skipped:
                by_class[cls] = by_class.get(cls, 0) + 1
            detail = ", ".join(f"{k}: {v}" for k, v in sorted(by_class.items()))
            report.add(
                Severity.WARNING, "COMPARISONS_TOO_SMALL_TO_TEST",
                f"{len(self._skipped)} group comparison(s) were skipped because "
                f"the group or the available reference was below "
                f"{self.min_group_size} employees ({detail}). These are untested, "
                f"not cleared — small units are where impact most often hides, and "
                f"they are also where the arithmetic is least able to detect it.",
            )
            report.skipped = [
                {"protected_class": c, "unit_type": ut, "unit": u, "size": n}
                for c, ut, u, n in self._skipped
            ]

        for cls in {c.protected_class for c in report.comparisons}:
            cls_comps = [c for c in report.comparisons if c.protected_class == cls]
            overall_flagged = [
                c for c in cls_comps
                if c.unit_type == "overall" and c.severity != Severity.INFO
            ]
            unit_flagged = [
                c for c in cls_comps
                if c.unit_type != "overall" and c.severity != Severity.INFO
            ]
            if unit_flagged and not overall_flagged:
                units = sorted({f"{c.unit} ({c.group})" for c in unit_flagged})[:4]
                worst = min(
                    (c for c in unit_flagged if c.impact_ratio is not None),
                    key=lambda c: c.impact_ratio, default=None,
                )
                ratio_txt = (
                    f" The worst is {worst.impact_ratio:.2f} in {worst.unit}."
                    if worst else ""
                )
                report.add(
                    Severity.WARNING, "AGGREGATE_MASKS_UNIT_IMPACT",
                    f"{cls}: the company-wide numbers show no flag, but "
                    f"{len(unit_flagged)} comparison(s) inside individual units are "
                    f"flagged ({', '.join(units)}).{ratio_txt} Impact is generally "
                    f"assessed where the decision was actually made, so a clean "
                    f"aggregate does not resolve a disparity inside a department.",
                    protected_class=cls,
                )

        n_comparisons = len(report.comparisons)
        if n_comparisons > 20:
            report.add(
                Severity.INFO, "MULTIPLE_COMPARISONS",
                f"{n_comparisons} comparisons were run. Across that many tests, "
                f"some will cross p<.05 by chance alone. Weigh the pattern and "
                f"the size of each disparity, not the count of flags.",
            )

        diverging = [c for c in report.comparisons if c.diverges]
        if diverging:
            worst = max(diverging, key=lambda c: c.selection_rate_ratio)
            report.add(
                Severity.WARNING, "FOUR_FIFTHS_UNDERSTATES_DISPARITY",
                f"{len(diverging)} comparison(s) pass the four-fifths rule but "
                f"show a materially higher termination rate. The worst is "
                f"{worst.protected_class} '{worst.group}' in "
                f"{worst.unit if worst.unit_type != 'overall' else 'the company overall'}: "
                f"selected at {worst.group_selection_rate:.1%} versus "
                f"{worst.reference_selection_rate:.1%}, which is "
                f"{worst.selection_rate_ratio:.2f}x the rate, yet the four-fifths "
                f"ratio reads {worst.impact_ratio:.2f}. When few people are "
                f"selected overall, the retention-basis ratio compresses toward "
                f"1.0 and will pass disparities that matter. Do not treat the "
                f"four-fifths number alone as clearance.",
            )

        fragile = [c for c in report.comparisons if "FRAGILE" in c.flags]
        if fragile:
            report.add(
                Severity.INFO, "FRAGILE_FINDINGS",
                f"{len(fragile)} flagged comparison(s) would clear the four-fifths "
                f"rule if a single selection differed. Fragility cuts both ways: "
                f"it means the flag is weak evidence, and also that the clean "
                f"result next to it is weak evidence.",
            )

        report.add(
            Severity.INFO, "NO_SWAP_RECOMMENDATIONS",
            "This analysis intentionally makes no recommendation about which "
            "individuals to select or retain. Adjusting a specific person's "
            "outcome because of their protected class is disparate treatment. "
            "Remedies run through the criteria and through counsel.",
        )

        if "performance_rating" in df.columns and "manager_id" in df.columns:
            self._check_rater_consistency(df, report)

    def _check_rater_consistency(self, df: pd.DataFrame, report: ImpactReport) -> None:
        """Flag managers whose rating distribution departs sharply from the norm.

        Inconsistent rating practice is the most common upstream source of a
        disparity that the selection criteria then faithfully reproduce.
        """
        work = df.loc[df["performance_rating"].notna() & df["manager_id"].notna()]
        if len(work) < 20:
            return
        counts = work.groupby("manager_id").size()
        managers = counts[counts >= 4].index
        if len(managers) < 2:
            return

        sel_rate = float(work["_selected"].mean())
        outliers: list[str] = []
        for m in managers:
            sub = work.loc[work["manager_id"] == m]
            rate = float(sub["_selected"].mean())
            sd = standard_deviations(
                int(sub["_selected"].sum()), len(sub),
                int(work["_selected"].sum()) - int(sub["_selected"].sum()),
                len(work) - len(sub),
            )
            if sd is not None and abs(sd) >= SD_THRESHOLD:
                outliers.append(f"{m} ({rate:.0%} vs {sel_rate:.0%} overall)")

        if outliers:
            report.add(
                Severity.WARNING, "RATER_INCONSISTENCY",
                f"Selection rates differ sharply by manager: "
                f"{', '.join(outliers[:5])}. If the underlying ratings were not "
                f"applied on a common standard, the score inherited that "
                f"inconsistency — check rating calibration before treating the "
                f"criteria as objective.",
            )


def _unit_type(column: str) -> str:
    return {
        "department": "department",
        "worksite_name": "worksite",
        "job_level": "job_level",
    }.get(column, column)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from .selection_criteria import SelectionEngine, load_plan
    from .workforce_data import load_workforce_csv

    ap = argparse.ArgumentParser(
        description="Measure adverse impact in a proposed RIF selection."
    )
    ap.add_argument("csv_path", help="Workforce CSV.")
    ap.add_argument("--plan", required=True, help="RIF plan YAML/JSON.")
    ap.add_argument("--as-of", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--min-group-size", type=int, default=MIN_GROUP_SIZE)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    as_of = args.as_of or (str(plan.as_of_date) if plan.as_of_date else None)

    ingest = load_workforce_csv(args.csv_path, as_of=as_of)
    if ingest.report.is_blocking:
        print("Ingestion is blocking; fix the roster first.")
        return 2

    selection = SelectionEngine(plan).run(ingest.data)
    analysis = AdverseImpactAnalyzer(min_group_size=args.min_group_size).run(
        selection.scores, scenario=plan.plan_name
    )

    if not args.quiet:
        print(analysis.report.to_markdown())

    if args.outdir:
        paths = analysis.write(args.outdir)
        print("\nWrote:")
        for k, p in paths.items():
            print(f"  {k}: {p}")

    return 1 if analysis.report.indicated else 0


if __name__ == "__main__":
    raise SystemExit(main())
