"""
ca_compliance.py
================

California Compliance Engine for the California RIF Copilot (box 5).

Takes a proposed selection (from box 3, or one scenario from box 2) and works
out what the law requires before it can be carried out: whether WARN notice is
triggered, when notice must go out, what each notice must contain, what is owed
on the final paycheck and when, and which employees carry conditions that need
individual legal handling.

It also produces the gate that box 7 depends on. Notices should not be
generated from a scenario with unresolved compliance findings, so
``ComplianceResult.gate`` reports whether document generation may proceed and,
if not, exactly what is blocking it.

Coverage
--------
* **Cal-WARN** (Lab. Code §§ 1400–1408) — covered establishment, mass layoff,
  relocation and termination triggers, the 60-day notice date, and the four
  disclosures added by SB 617 effective 2026-01-01.
* **Federal WARN** (29 U.S.C. § 2101) — run in parallel, because the two have
  different thresholds and either can bite.
* **Final pay** (Lab. Code §§ 201, 203) — immediate payment on involuntary
  termination and the waiting-time penalty exposure if it is late.
* **Accrued vacation** (Lab. Code § 227.3) — vested, payable at the final rate.
* **OWBPA** (29 U.S.C. § 626(f)) — the 45-day consideration period, 7-day
  revocation period, and decisional-unit disclosure required for an
  enforceable age release in a group termination program.
* **Benefits and agency notices** — COBRA/Cal-COBRA, EDD change of
  relationship, and the DE 2320 pamphlet.
* **Individual conditions** — protected leave, union representation, and
  sponsored work authorization.

What this module will not do
----------------------------
It computes whether a threshold is met. It does not help anyone stay under one.
Restructuring a reduction for the purpose of dropping below a WARN trigger is
not a compliance strategy: Cal-WARN aggregates layoffs across any 30-day
period, courts look at the substance of a sequence rather than its packaging,
and a plan visibly engineered around the threshold tends to establish the very
intent it was meant to obscure. Where a reduction lands near a threshold, this
module says so plainly and routes it to counsel. It offers no advice on
splitting, staggering, or reclassifying to avoid coverage, and it flags
sequences that look like it may already have happened.

Not legal advice
----------------
This is a screening tool built from public statutory text. Statutes are
amended, agencies issue guidance, and courts interpret both — Cal-WARN's notice
content changed as recently as January 2026. Every determination here is an
input to a lawyer's analysis, not a substitute for it. Verify current
requirements with employment counsel before acting.

Usage
-----
    from .ca_compliance import ComplianceConfig, ComplianceEngine

    cfg = ComplianceConfig(
        proposed_separation_date="2026-10-30",
        employer_name="Acme Inc.",
        employer_contact_email="hr@acme.com",
        employer_contact_phone="(415) 555-0100",
    )
    result = ComplianceEngine(cfg).run(selection.scores, impact=analysis)

    result.gate.may_generate_documents
    print(result.report.to_markdown())
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .workforce_data import Severity

__all__ = [
    "ComplianceConfig",
    "ComplianceEngine",
    "ComplianceResult",
    "ComplianceGate",
    "Obligation",
    "WarnAnalysis",
    "CAL_WARN_ESTABLISHMENT_THRESHOLD",
    "CAL_WARN_MASS_LAYOFF_THRESHOLD",
    "SB617_REQUIRED_DISCLOSURES",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Statutory constants
#
# Verified against Labor Code §§ 1400.5, 1401 and EDD Workforce Services
# Information Notice WSIN25-14 (issued 2026-01-06). Re-verify before relying on
# these: the SB 617 content requirements below did not exist before 2026.
# ---------------------------------------------------------------------------

#: Cal-WARN covered establishment: employs, or has employed in the preceding
#: 12 months, 75 or more persons. Part-time employees count (unlike federal).
CAL_WARN_ESTABLISHMENT_THRESHOLD = 75

#: Cal-WARN mass layoff: 50 or more employees at a covered establishment in any
#: 30-day period. There is no percentage-of-workforce test, unlike federal.
CAL_WARN_MASS_LAYOFF_THRESHOLD = 50
CAL_WARN_LOOKBACK_DAYS = 30
CAL_WARN_NOTICE_DAYS = 60

#: Cal-WARN counts only employees employed at least 6 of the 12 months
#: preceding the date notice is required.
CAL_WARN_MIN_SERVICE_MONTHS = 6

#: A relocation of all or substantially all operations 100+ miles away.
CAL_WARN_RELOCATION_MILES = 100

#: Federal WARN: 100+ employees. Plant closing = 50+ at a single site. Mass
#: layoff = 500+, or 50-499 if that is at least 33% of the site's workforce.
FED_WARN_EMPLOYER_THRESHOLD = 100
FED_WARN_PLANT_CLOSING_THRESHOLD = 50
FED_WARN_MASS_LAYOFF_ABSOLUTE = 500
FED_WARN_MASS_LAYOFF_FLOOR = 50
FED_WARN_MASS_LAYOFF_PCT = 1 / 3
FED_WARN_NOTICE_DAYS = 60
FED_WARN_AGGREGATION_DAYS = 90

#: SB 617 (eff. 2026-01-01) content requirements. A notice that is timely but
#: omits any of these does not satisfy Labor Code § 1401.
SB617_EFFECTIVE_DATE = dt.date(2026, 1, 1)
SB617_REQUIRED_DISCLOSURES: tuple[tuple[str, str], ...] = (
    ("service_coordination_statement",
     "A statement of whether the employer plans to coordinate services (such as "
     "a rapid response orientation) through the Local Workforce Development "
     "Board, through a different entity, or not at all."),
    ("lwdb_contact_and_description",
     "A functioning email address and telephone number for the Local Workforce "
     "Development Board, plus the statutory description of rapid response "
     "activities."),
    ("calfresh_information",
     "A description of the CalFresh food assistance program, the CalFresh "
     "benefits helpline, and a link to the program website."),
    ("employer_contact",
     "A functioning employer contact email address and telephone number."),
)

#: If the employer elects to coordinate services, coordination must occur
#: within 30 days of the date the notice is issued.
SB617_COORDINATION_DAYS = 30

#: OWBPA: for a group termination program, an age release requires 45 days to
#: consider and 7 days to revoke, plus decisional-unit disclosures.
OWBPA_GROUP_CONSIDERATION_DAYS = 45
OWBPA_INDIVIDUAL_CONSIDERATION_DAYS = 21
OWBPA_REVOCATION_DAYS = 7

#: Waiting time penalty: daily wage for each day late, capped at 30 days.
WAITING_TIME_PENALTY_MAX_DAYS = 30

#: COBRA election notice deadline after the qualifying event.
COBRA_NOTICE_DAYS = 44

#: Proximity to a threshold that warrants an explicit note.
THRESHOLD_PROXIMITY = 5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ComplianceConfig:
    """Facts about the planned action that the roster cannot supply."""

    proposed_separation_date: dt.date
    #: When notice will actually be issued. Defaults to today.
    notice_date: dt.date | None = None

    employer_name: str = ""
    employer_contact_email: str = ""
    employer_contact_phone: str = ""

    #: Column defining the Cal-WARN "covered establishment". A single facility,
    #: not the whole company.
    establishment_column: str = "worksite_name"

    #: Total employees company-wide, for the federal 100-employee test. If None,
    #: the roster size is used, which understates it when the roster is partial.
    total_company_headcount: int | None = None

    #: Whether the action closes or substantially ceases operations at a site.
    is_termination_of_operations: bool = False
    #: Whether operations are moving 100+ miles.
    is_relocation: bool = False
    relocation_distance_miles: float | None = None

    #: SB 617 election: "lwdb" | "other" | "none".
    service_coordination: str = ""
    lwdb_name: str = ""
    lwdb_email: str = ""
    lwdb_phone: str = ""

    #: Whether a release of claims will be offered (triggers OWBPA analysis).
    offering_severance_agreement: bool = True
    #: Whether this is a group termination program under OWBPA.
    is_group_termination_program: bool = True

    #: Prior layoff dates and counts at each establishment, for aggregation.
    #: {establishment: [(date, count), ...]}
    prior_layoffs: dict[str, list[tuple[dt.date, int]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.proposed_separation_date = _as_date(self.proposed_separation_date)
        if self.notice_date is not None:
            self.notice_date = _as_date(self.notice_date)
        else:
            self.notice_date = dt.date.today()


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    return pd.Timestamp(value).date()


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------


@dataclass
class Obligation:
    """One thing the employer must do, and by when."""

    code: str
    title: str
    authority: str
    description: str
    due_date: dt.date | None = None
    applies_to: str = "all affected employees"
    #: True if the deadline has already passed given the configured dates.
    missed: bool = False
    severity: str = Severity.WARNING

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "title": self.title, "authority": self.authority,
            "description": self.description,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "applies_to": self.applies_to, "missed": self.missed,
            "severity": self.severity,
        }


@dataclass
class WarnAnalysis:
    """Result of the WARN threshold analysis for one jurisdiction."""

    jurisdiction: str                 # "California" | "Federal"
    triggered: bool
    reason: str
    establishment: str | None = None
    covered_establishment: bool = False
    counted_employees: int = 0
    affected_employees: int = 0
    site_workforce: int = 0
    threshold: int = 0
    earliest_notice_date: dt.date | None = None
    latest_separation_date: dt.date | None = None
    near_threshold: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "jurisdiction": self.jurisdiction,
            "triggered": self.triggered,
            "reason": self.reason,
            "establishment": self.establishment,
            "covered_establishment": self.covered_establishment,
            "counted_employees": self.counted_employees,
            "affected_employees": self.affected_employees,
            "site_workforce": self.site_workforce,
            "threshold": self.threshold,
            "earliest_notice_date": (
                self.earliest_notice_date.isoformat()
                if self.earliest_notice_date else None
            ),
            "latest_separation_date": (
                self.latest_separation_date.isoformat()
                if self.latest_separation_date else None
            ),
            "near_threshold": self.near_threshold,
            "notes": self.notes,
        }


@dataclass
class ComplianceGate:
    """Whether box 7 may generate notices from this scenario."""

    may_generate_documents: bool = True
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def block(self, reason: str) -> None:
        self.may_generate_documents = False
        self.blockers.append(reason)

    def warn(self, reason: str) -> None:
        self.warnings.append(reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "may_generate_documents": self.may_generate_documents,
            "blockers": self.blockers,
            "warnings": self.warnings,
        }


@dataclass
class ComplianceFinding:
    severity: str
    code: str
    message: str
    authority: str = ""
    employee_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity, "code": self.code, "message": self.message,
            "authority": self.authority, "employee_id": self.employee_id,
        }


@dataclass
class ComplianceReport:
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
    scenario: str = ""
    separation_date: str = ""
    notice_date: str = ""
    affected_count: int = 0
    warn: list[WarnAnalysis] = field(default_factory=list)
    obligations: list[Obligation] = field(default_factory=list)
    findings: list[ComplianceFinding] = field(default_factory=list)
    final_pay: dict[str, Any] = field(default_factory=dict)
    gate: ComplianceGate = field(default_factory=ComplianceGate)

    def add(self, severity: str, code: str, message: str, **kw: Any) -> None:
        self.findings.append(ComplianceFinding(severity, code, message, **kw))

    @property
    def warn_triggered(self) -> bool:
        return any(w.triggered for w in self.warn)

    @property
    def missed_deadlines(self) -> list[Obligation]:
        return [o for o in self.obligations if o.missed]

    def calendar(self) -> pd.DataFrame:
        rows = [
            {
                "due_date": o.due_date, "code": o.code, "title": o.title,
                "authority": o.authority, "applies_to": o.applies_to,
                "missed": o.missed, "severity": o.severity,
            }
            for o in self.obligations if o.due_date is not None
        ]
        if not rows:
            return pd.DataFrame(
                columns=["due_date", "code", "title", "authority", "applies_to",
                         "missed", "severity"]
            )
        return pd.DataFrame(rows).sort_values("due_date").reset_index(drop=True)

    def summary(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "generated_at": self.generated_at,
            "separation_date": self.separation_date,
            "notice_date": self.notice_date,
            "affected_count": self.affected_count,
            "warn_triggered": self.warn_triggered,
            "obligations": len(self.obligations),
            "missed_deadlines": len(self.missed_deadlines),
            "errors": len([f for f in self.findings if f.severity == Severity.ERROR]),
            "warnings": len([f for f in self.findings if f.severity == Severity.WARNING]),
            "may_generate_documents": self.gate.may_generate_documents,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "warn": [w.to_dict() for w in self.warn],
            "obligations": [o.to_dict() for o in self.obligations],
            "final_pay": self.final_pay,
            "findings": [f.to_dict() for f in self.findings],
            "gate": self.gate.to_dict(),
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self) -> str:
        s = self.summary()
        L: list[str] = []
        L.append("# California Compliance Analysis")
        L.append("")
        L.append(
            "> **Prepared at the direction of counsel / privileged and "
            "confidential — confirm labeling before circulating.**"
        )
        L.append("")
        L.append(f"**Scenario:** {s['scenario'] or '(unnamed)'}  ")
        L.append(f"**Generated:** {s['generated_at']}  ")
        L.append(f"**Notice date:** {s['notice_date']}  ")
        L.append(f"**Proposed separation date:** {s['separation_date']}  ")
        L.append(f"**Affected employees:** {s['affected_count']}  ")
        L.append("")

        # -- gate ----------------------------------------------------------
        L.append("## Document generation gate")
        L.append("")
        if self.gate.may_generate_documents:
            L.append("**Status: CLEAR** — no blocking compliance findings.")
            if self.gate.warnings:
                L.append("")
                L.append("Outstanding items that should be resolved first:")
                for w in self.gate.warnings:
                    L.append(f"- {w}")
        else:
            L.append("**Status: BLOCKED** — notices must not be generated until "
                     "the following are resolved:")
            L.append("")
            for b in self.gate.blockers:
                L.append(f"- {b}")
            if self.gate.warnings:
                L.append("")
                L.append("Also outstanding:")
                for w in self.gate.warnings:
                    L.append(f"- {w}")
        L.append("")

        # -- WARN ----------------------------------------------------------
        L.append("## WARN analysis")
        L.append("")
        if not self.warn:
            L.append("No WARN analysis could be performed.")
        else:
            L.append("| Jurisdiction | Establishment | Affected | Threshold | "
                     "Triggered | Earliest notice date |")
            L.append("|---|---|---|---|---|---|")
            for w in self.warn:
                trig = "**YES**" if w.triggered else "no"
                nd = w.earliest_notice_date.isoformat() if w.earliest_notice_date else "—"
                L.append(
                    f"| {w.jurisdiction} | {w.establishment or '—'} | "
                    f"{w.affected_employees} | {w.threshold} | {trig} | {nd} |"
                )
            L.append("")
            for w in self.warn:
                if w.reason:
                    L.append(f"- **{w.jurisdiction}"
                             + (f" / {w.establishment}" if w.establishment else "")
                             + f":** {w.reason}")
                for n in w.notes:
                    L.append(f"  - {n}")
            L.append("")

        # -- obligations ----------------------------------------------------
        cal = self.calendar()
        L.append("## Compliance calendar")
        L.append("")
        if cal.empty:
            L.append("No dated obligations were derived.")
        else:
            L.append("| Due | Obligation | Authority | Applies to | Status |")
            L.append("|---|---|---|---|---|")
            for _, r in cal.iterrows():
                status = "**MISSED**" if r["missed"] else "pending"
                L.append(
                    f"| {r['due_date']} | {r['title']} | {r['authority']} | "
                    f"{r['applies_to']} | {status} |"
                )
        L.append("")

        undated = [o for o in self.obligations if o.due_date is None]
        if undated:
            L.append("### Obligations without a computed date")
            L.append("")
            for o in undated:
                L.append(f"- **{o.title}** ({o.authority}) — {o.description}")
            L.append("")

        # -- final pay ------------------------------------------------------
        if self.final_pay:
            fp = self.final_pay
            L.append("## Final pay")
            L.append("")
            L.append("| Item | Amount |")
            L.append("|---|---|")
            L.append(f"| Accrued vacation payout (§ 227.3) | ${fp.get('vacation_payout', 0):,.2f} |")
            L.append("| Wages earned through separation | *not computed* |")
            L.append(f"| Waiting-time penalty exposure if late (§ 203) | ${fp.get('waiting_time_exposure', 0):,.2f} |")
            L.append("")
            L.append(
                "Wages earned through the separation date depend on the pay "
                "period and days actually worked, which this module does not "
                "know — payroll must supply that figure. What is shown is the "
                "vacation payout, which is computable from the roster, and the "
                "penalty exposure if payment is late."
            )
            L.append("")
            L.append(
                "Final wages, including all vested vacation, are due **at the "
                "time of termination** for an involuntary separation. Late "
                "payment accrues a penalty of one day's wages per day, up to 30 "
                "days, per employee — which is why the exposure figure is large "
                "relative to the vacation amount."
            )
            L.append("")

        # -- SB 617 ---------------------------------------------------------
        if self.warn_triggered:
            L.append("## Cal-WARN notice content (SB 617)")
            L.append("")
            L.append(
                "Effective 2026-01-01, Labor Code § 1401 requires the following "
                "in every Cal-WARN notice. **A notice that is timely but omits "
                "any of these does not satisfy the statute**, and each day of "
                "deficient notice is treated as a separate violation."
            )
            L.append("")
            for code, desc in SB617_REQUIRED_DISCLOSURES:
                L.append(f"- `{code}` — {desc}")
            L.append("")

        # -- findings -------------------------------------------------------
        if self.findings:
            L.append("## Findings")
            L.append("")
            order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
            for f in sorted(self.findings, key=lambda x: order.get(x.severity, 9)):
                auth = f" *({f.authority})*" if f.authority else ""
                L.append(f"- **[{f.severity}] {f.code}**{auth} — {f.message}")
            L.append("")

        L.append("---")
        L.append(
            "_Compliance screening only, built from public statutory text. "
            "Cal-WARN's notice content changed on 2026-01-01 and statutes are "
            "amended regularly; verify current requirements with employment "
            "counsel before acting. This module computes whether a threshold is "
            "met and does not advise on structuring a reduction to avoid one._"
        )
        return "\n".join(L)


@dataclass
class ComplianceResult:
    report: ComplianceReport
    obligations: pd.DataFrame
    calendar: pd.DataFrame
    employee_flags: pd.DataFrame

    @property
    def gate(self) -> ComplianceGate:
        return self.report.gate

    def write(self, outdir: str | Path, stem: str = "compliance") -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "report_md": outdir / f"{stem}_report.md",
            "calendar": outdir / f"{stem}_calendar.csv",
            "obligations": outdir / f"{stem}_obligations.csv",
            "employee_flags": outdir / f"{stem}_employee_flags.csv",
            "report_json": outdir / f"{stem}_report.json",
        }
        self.calendar.to_csv(paths["calendar"], index=False)
        self.obligations.to_csv(paths["obligations"], index=False)
        self.employee_flags.to_csv(paths["employee_flags"], index=False)
        self.report.to_json(paths["report_json"])
        paths["report_md"].write_text(self.report.to_markdown(), encoding="utf-8")
        return paths


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class ComplianceEngine:
    """Determines statutory obligations arising from a proposed selection."""

    def __init__(self, config: ComplianceConfig) -> None:
        self.cfg = config

    # -- public ----------------------------------------------------------
    def run(
        self,
        scores: pd.DataFrame,
        impact: Any = None,
        selection: Any = None,
        scenario: str = "",
    ) -> ComplianceResult:
        cfg = self.cfg
        report = ComplianceReport(
            scenario=scenario,
            separation_date=cfg.proposed_separation_date.isoformat(),
            notice_date=cfg.notice_date.isoformat(),
        )

        if scores is None or scores.empty or "selected" not in scores.columns:
            report.add(
                Severity.ERROR, "NO_SELECTION",
                "No scored roster with a 'selected' column was provided; nothing "
                "to analyze.",
            )
            report.gate.block("No selection to analyze.")
            return self._package(report)

        cut = scores.loc[scores["selected"].fillna(False).astype(bool)].copy()
        report.affected_count = len(cut)

        if cut.empty:
            report.add(
                Severity.WARNING, "NO_AFFECTED_EMPLOYEES",
                "The scenario selects nobody, so no separation obligations arise.",
            )
            return self._package(report)

        self._check_dates(report)
        self._analyze_warn(scores, cut, report)
        self._final_pay(cut, report)
        self._owbpa(cut, report)
        self._benefit_and_agency_notices(report)
        flags = self._individual_conditions(cut, report)
        self._apply_gate(report, impact, selection)
        return self._package(report, flags)

    # -- dates -----------------------------------------------------------
    def _check_dates(self, report: ComplianceReport) -> None:
        cfg = self.cfg
        lead = (cfg.proposed_separation_date - cfg.notice_date).days
        if lead < 0:
            report.add(
                Severity.ERROR, "SEPARATION_BEFORE_NOTICE",
                f"The proposed separation date ({cfg.proposed_separation_date}) is "
                f"before the notice date ({cfg.notice_date}).",
            )
        elif lead < CAL_WARN_NOTICE_DAYS:
            report.add(
                Severity.WARNING, "SHORT_NOTICE_WINDOW",
                f"Only {lead} days separate notice from the proposed separation "
                f"date. If WARN is triggered, 60 days are required and this "
                f"schedule does not meet it.",
                authority="Lab. Code § 1401(a)",
            )

    # -- WARN ------------------------------------------------------------
    def _counted_employees(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Cal-WARN counts employees with at least 6 of the preceding 12 months
        of service. Employees below that are excluded from the threshold count
        (they are still owed final pay and everything else)."""
        if "tenure_years" not in frame.columns:
            return frame
        months = frame["tenure_years"].astype("Float64") * 12.0
        return frame.loc[months.fillna(0) >= CAL_WARN_MIN_SERVICE_MONTHS]

    def _analyze_warn(
        self, scores: pd.DataFrame, cut: pd.DataFrame, report: ComplianceReport
    ) -> None:
        cfg = self.cfg
        col = cfg.establishment_column

        if col not in scores.columns:
            report.add(
                Severity.ERROR, "NO_ESTABLISHMENT_COLUMN",
                f"Column {col!r} is not in the roster, so Cal-WARN coverage "
                f"cannot be determined. Cal-WARN applies per covered "
                f"establishment, not company-wide, so this cannot be inferred "
                f"from headcount alone.",
                authority="Lab. Code § 1400.5(a)",
            )
            report.gate.block(
                "WARN coverage is undetermined — no establishment column in the roster."
            )
            return

        establishments = sorted(
            {str(v) for v in scores[col].dropna().unique()}
        )
        if not establishments:
            report.add(
                Severity.ERROR, "NO_ESTABLISHMENTS",
                f"No values in {col!r}; WARN coverage cannot be determined.",
            )
            report.gate.block("WARN coverage is undetermined.")
            return

        any_ca_triggered = False
        for est in establishments:
            site_all = scores.loc[scores[col].astype("string") == est]
            site_cut = cut.loc[cut[col].astype("string") == est]
            if site_cut.empty:
                continue
            ca = self._cal_warn(est, site_all, site_cut, report)
            report.warn.append(ca)
            any_ca_triggered = any_ca_triggered or ca.triggered

            fed = self._federal_warn(est, site_all, site_cut, scores)
            report.warn.append(fed)

        if any_ca_triggered:
            self._warn_obligations(report)
        else:
            report.add(
                Severity.INFO, "CAL_WARN_NOT_TRIGGERED",
                "No establishment meets the Cal-WARN mass layoff threshold on "
                "this scenario. Coverage is assessed per establishment and per "
                "30-day period, so this determination changes if the reduction "
                "grows, if a second round follows within 30 days, or if "
                "operations at a site cease.",
                authority="Lab. Code §§ 1400.5(d), 1401",
            )

    def _cal_warn(
        self, est: str, site_all: pd.DataFrame, site_cut: pd.DataFrame,
        report: ComplianceReport,
    ) -> WarnAnalysis:
        cfg = self.cfg
        counted_all = self._counted_employees(site_all)
        counted_cut = self._counted_employees(site_cut)

        covered = len(counted_all) >= CAL_WARN_ESTABLISHMENT_THRESHOLD
        affected = len(counted_cut)

        a = WarnAnalysis(
            jurisdiction="California",
            establishment=est,
            triggered=False,
            reason="",
            covered_establishment=covered,
            counted_employees=len(counted_all),
            affected_employees=affected,
            site_workforce=len(site_all),
            threshold=CAL_WARN_MASS_LAYOFF_THRESHOLD,
        )

        if len(counted_all) < len(site_all):
            a.notes.append(
                f"{len(site_all) - len(counted_all)} employee(s) at this site have "
                f"under 6 months of service and are excluded from the threshold "
                f"count. They are still owed final pay and all other separation "
                f"obligations."
            )

        if not covered:
            a.reason = (
                f"{est} has {len(counted_all)} counted employee(s), below the "
                f"{CAL_WARN_ESTABLISHMENT_THRESHOLD}-person covered establishment "
                f"threshold. Note the threshold looks back 12 months, so a site "
                f"that has shrunk can still be covered — verify against headcount "
                f"history, not just today's roster."
            )
            a.notes.append(
                "Coverage is measured on employment within the preceding 12 "
                "months, which this roster may not capture."
            )
            return a

        triggers: list[str] = []
        if affected >= CAL_WARN_MASS_LAYOFF_THRESHOLD:
            triggers.append(
                f"mass layoff — {affected} employees at a covered establishment "
                f"within a 30-day period (no percentage test applies)"
            )
        if cfg.is_termination_of_operations:
            triggers.append(
                "termination — cessation or substantial cessation of operations "
                "at the establishment (no minimum employee count)"
            )
        if cfg.is_relocation:
            miles = cfg.relocation_distance_miles
            if miles is None or miles >= CAL_WARN_RELOCATION_MILES:
                triggers.append(
                    f"relocation — operations moving "
                    f"{'100+' if miles is None else f'{miles:.0f}'} miles"
                )

        if triggers:
            a.triggered = True
            a.reason = "Cal-WARN triggered: " + "; ".join(triggers) + "."
            a.earliest_notice_date = (
                cfg.proposed_separation_date - dt.timedelta(days=CAL_WARN_NOTICE_DAYS)
            )
            a.latest_separation_date = (
                cfg.notice_date + dt.timedelta(days=CAL_WARN_NOTICE_DAYS)
            )
        else:
            a.reason = (
                f"{est} is a covered establishment ({len(counted_all)} counted "
                f"employees) but {affected} affected is below the "
                f"{CAL_WARN_MASS_LAYOFF_THRESHOLD}-employee mass layoff threshold."
            )
            gap = CAL_WARN_MASS_LAYOFF_THRESHOLD - affected
            if gap <= THRESHOLD_PROXIMITY:
                a.near_threshold = True
                report.add(
                    Severity.WARNING, "NEAR_WARN_THRESHOLD",
                    f"{est} is {gap} employee(s) below the Cal-WARN mass layoff "
                    f"threshold. Cal-WARN aggregates layoffs across any 30-day "
                    f"period, so any further separation at this site within that "
                    f"window can trigger notice retroactively — including "
                    f"performance terminations and roles cut for unrelated "
                    f"reasons. Treat this as triggered for planning purposes and "
                    f"confirm with counsel.",
                    authority="Lab. Code § 1400.5(d)",
                )

        self._check_aggregation(est, affected, a, report)
        return a

    def _check_aggregation(
        self, est: str, affected: int, a: WarnAnalysis, report: ComplianceReport
    ) -> None:
        """Look at prior rounds at the same establishment.

        This exists to catch a sequence that already adds up to a triggering
        event, not to help anyone design one.
        """
        prior = self.cfg.prior_layoffs.get(est) or []
        if not prior:
            return
        window_start = self.cfg.proposed_separation_date - dt.timedelta(
            days=CAL_WARN_LOOKBACK_DAYS
        )
        in_window = [(d, n) for d, n in ((_as_date(d), n) for d, n in prior)
                     if window_start <= d <= self.cfg.proposed_separation_date]
        recent_total = sum(n for _, n in in_window)
        if not recent_total:
            return

        combined = affected + recent_total
        a.notes.append(
            f"{recent_total} prior separation(s) at this establishment fall within "
            f"the 30-day window; combined total is {combined}."
        )
        if combined >= CAL_WARN_MASS_LAYOFF_THRESHOLD and not a.triggered:
            a.triggered = True
            a.reason = (
                f"Cal-WARN triggered by aggregation: {affected} in this action "
                f"plus {recent_total} within the preceding 30 days at {est} "
                f"totals {combined}, at or above the "
                f"{CAL_WARN_MASS_LAYOFF_THRESHOLD}-employee threshold."
            )
            a.earliest_notice_date = (
                self.cfg.proposed_separation_date
                - dt.timedelta(days=CAL_WARN_NOTICE_DAYS)
            )
            report.add(
                Severity.ERROR, "WARN_TRIGGERED_BY_AGGREGATION",
                f"Separately, neither round reaches 50 at {est}; together they do "
                f"({combined}). Cal-WARN counts any 30-day period, so the "
                f"threshold is met and notice was required 60 days before the "
                f"earlier separations. Escalate to counsel immediately.",
                authority="Lab. Code § 1400.5(d)",
            )

    def _federal_warn(
        self, est: str, site_all: pd.DataFrame, site_cut: pd.DataFrame,
        scores: pd.DataFrame,
    ) -> WarnAnalysis:
        cfg = self.cfg
        company = cfg.total_company_headcount or len(scores)
        affected = len(site_cut)
        site_size = len(site_all)

        a = WarnAnalysis(
            jurisdiction="Federal",
            establishment=est,
            triggered=False,
            reason="",
            covered_establishment=company >= FED_WARN_EMPLOYER_THRESHOLD,
            counted_employees=company,
            affected_employees=affected,
            site_workforce=site_size,
            threshold=FED_WARN_MASS_LAYOFF_FLOOR,
        )

        if cfg.total_company_headcount is None:
            a.notes.append(
                "Company-wide headcount was not supplied; the roster size was "
                "used instead, which understates coverage if the roster is "
                "partial. Set total_company_headcount for a reliable federal "
                "determination."
            )

        if company < FED_WARN_EMPLOYER_THRESHOLD:
            a.reason = (
                f"Employer headcount of {company} is below the federal "
                f"{FED_WARN_EMPLOYER_THRESHOLD}-employee threshold. Federal WARN "
                f"excludes part-time employees from that count, so the figure "
                f"used here may differ from the statutory one."
            )
            return a

        pct = (affected / site_size) if site_size else 0.0
        triggers: list[str] = []
        if cfg.is_termination_of_operations and affected >= FED_WARN_PLANT_CLOSING_THRESHOLD:
            triggers.append(f"plant closing — {affected} employees at a single site")
        if affected >= FED_WARN_MASS_LAYOFF_ABSOLUTE:
            triggers.append(f"mass layoff — {affected} employees")
        elif affected >= FED_WARN_MASS_LAYOFF_FLOOR and pct >= FED_WARN_MASS_LAYOFF_PCT:
            triggers.append(
                f"mass layoff — {affected} employees, {pct:.0%} of the site "
                f"workforce (at or above the one-third test)"
            )

        if triggers:
            a.triggered = True
            a.reason = "Federal WARN triggered: " + "; ".join(triggers) + "."
            a.earliest_notice_date = (
                cfg.proposed_separation_date - dt.timedelta(days=FED_WARN_NOTICE_DAYS)
            )
        else:
            a.reason = (
                f"{affected} affected ({pct:.0%} of the site) does not meet the "
                f"federal thresholds (500+, or 50+ and at least one-third of the "
                f"site). Cal-WARN has no percentage test and may still apply."
            )
        a.notes.append(
            f"Federal WARN aggregates employment losses over "
            f"{FED_WARN_AGGREGATION_DAYS} days, a longer window than Cal-WARN's 30."
        )
        return a

    def _warn_obligations(self, report: ComplianceReport) -> None:
        cfg = self.cfg
        required_notice_date = (
            cfg.proposed_separation_date - dt.timedelta(days=CAL_WARN_NOTICE_DAYS)
        )
        missed = cfg.notice_date > required_notice_date

        report.obligations.append(Obligation(
            code="CAL_WARN_NOTICE",
            title="Issue Cal-WARN 60-day notice",
            authority="Lab. Code § 1401(a)",
            description=(
                "Written notice to affected employees, the EDD, the Local "
                "Workforce Development Board, and the chief elected official of "
                "each city and county where the separation occurs."
            ),
            due_date=required_notice_date,
            missed=missed,
            severity=Severity.ERROR if missed else Severity.WARNING,
        ))

        if missed:
            days_late = (cfg.notice_date - required_notice_date).days
            report.add(
                Severity.ERROR, "WARN_NOTICE_DATE_PASSED",
                f"Notice was required by {required_notice_date} and the notice "
                f"date is {cfg.notice_date}, {days_late} day(s) late. The "
                f"separation date must move to at least "
                f"{cfg.notice_date + dt.timedelta(days=CAL_WARN_NOTICE_DAYS)}, or "
                f"the employer is exposed to back pay and benefits for each "
                f"affected employee plus civil penalties. California's "
                f"exceptions are narrower than the federal ones — there is no "
                f"unforeseeable-business-circumstances exception, and the "
                f"faltering-company exception does not apply to a mass layoff.",
                authority="Lab. Code §§ 1401, 1402, 1402.5",
            )

        for code, desc in SB617_REQUIRED_DISCLOSURES:
            report.obligations.append(Obligation(
                code=f"SB617_{code.upper()}",
                title=f"Include in notice: {code.replace('_', ' ')}",
                authority="Lab. Code § 1401(c)-(e) (SB 617, eff. 2026-01-01)",
                description=desc,
                due_date=required_notice_date,
                applies_to="every Cal-WARN notice",
            ))

        # Configuration-driven completeness checks on the SB 617 content.
        if not cfg.service_coordination:
            report.add(
                Severity.ERROR, "SB617_COORDINATION_UNDECLARED",
                "SB 617 requires the notice to state whether services will be "
                "coordinated through the Local Workforce Development Board, "
                "another entity, or not at all. No election has been configured, "
                "so a compliant notice cannot be produced.",
                authority="Lab. Code § 1401(c)",
            )
        elif cfg.service_coordination in ("lwdb", "other"):
            report.obligations.append(Obligation(
                code="SB617_COORDINATE_SERVICES",
                title="Arrange coordinated transition services",
                authority="Lab. Code § 1401(c) (SB 617)",
                description=(
                    "Having elected to coordinate services, arrangements must be "
                    "made within 30 days of the notice date."
                ),
                due_date=cfg.notice_date + dt.timedelta(days=SB617_COORDINATION_DAYS),
            ))

        if not (cfg.lwdb_email and cfg.lwdb_phone):
            report.add(
                Severity.ERROR, "SB617_LWDB_CONTACT_MISSING",
                "SB 617 requires a functioning email address and telephone number "
                "for the Local Workforce Development Board in the notice. These "
                "are not configured.",
                authority="Lab. Code § 1401(d)",
            )
        if not (cfg.employer_contact_email and cfg.employer_contact_phone):
            report.add(
                Severity.ERROR, "SB617_EMPLOYER_CONTACT_MISSING",
                "SB 617 requires a functioning employer contact email address and "
                "telephone number in the notice. These are not configured.",
                authority="Lab. Code § 1401(d)",
            )

        report.add(
            Severity.WARNING, "SB617_CONTENT_IS_INDEPENDENT",
            "A Cal-WARN notice that is delivered on time but omits any SB 617 "
            "disclosure does not satisfy Labor Code § 1401, and each day of "
            "deficient notice is treated as a separate violation. Timing "
            "compliance does not cure content deficiency.",
            authority="Lab. Code § 1401 (as amended by SB 617)",
        )

    # -- final pay --------------------------------------------------------
    def _final_pay(self, cut: pd.DataFrame, report: ComplianceReport) -> None:
        cfg = self.cfg
        wages = 0.0
        vacation = 0.0
        daily_total = 0.0
        missing_rate = 0

        # Wages earned through the separation date depend on the pay period and
        # actual days worked, neither of which this module knows. Rather than
        # invent a figure that would be relied on, only what can be computed is
        # reported: the vacation payout and the penalty exposure.
        missing_vacation_data = 0
        for _, row in cut.iterrows():
            annual = row.get("annualized_pay")
            if pd.isna(annual):
                missing_rate += 1
            else:
                daily_total += float(annual) / 260.0

            hours = row.get("accrued_vacation_hours")
            rate = row.get("hourly_equivalent_rate")
            if pd.notna(hours) and pd.notna(rate):
                vacation += float(hours) * float(rate)
            else:
                missing_vacation_data += 1

        exposure = daily_total * WAITING_TIME_PENALTY_MAX_DAYS

        report.final_pay = {
            "vacation_payout": round(vacation, 2),
            "waiting_time_exposure": round(exposure, 2),
            "employees_missing_pay_data": missing_rate,
            "employees_missing_vacation_data": missing_vacation_data,
            "wages_through_separation": None,
        }

        report.obligations.append(Obligation(
            code="FINAL_PAY",
            title="Pay all final wages, including vested vacation",
            authority="Lab. Code §§ 201, 227.3",
            description=(
                "On an involuntary termination, all earned and unpaid wages are "
                "due at the time of termination. Vested vacation is wages and "
                "cannot be forfeited; it is paid at the final rate."
            ),
            due_date=cfg.proposed_separation_date,
            severity=Severity.ERROR,
        ))

        report.add(
            Severity.WARNING, "FINAL_PAY_TIMING",
            f"Final pay is due at the moment of separation, not on the next "
            f"regular payday. Late payment accrues a penalty of one day's wages "
            f"per employee per day, capped at 30 days — roughly "
            f"${exposure:,.0f} of exposure across this group if payment slips. "
            f"Payroll needs the final figures before notice day, not after.",
            authority="Lab. Code §§ 201, 203",
        )

        if missing_rate:
            report.add(
                Severity.ERROR, "FINAL_PAY_UNCOMPUTABLE",
                f"{missing_rate} affected employee(s) have no pay data, so their "
                f"final pay cannot be computed. Separating an employee without a "
                f"correct final check is what generates § 203 penalties.",
                authority="Lab. Code § 203",
            )

        if "accrued_vacation_hours" in cut.columns:
            missing_vac = int(cut["accrued_vacation_hours"].isna().sum())
            if missing_vac:
                report.add(
                    Severity.WARNING, "VACATION_BALANCE_MISSING",
                    f"{missing_vac} affected employee(s) have no accrued vacation "
                    f"balance on record. Under § 227.3 any vested balance must be "
                    f"paid out; a blank is not the same as zero and needs "
                    f"confirmation from payroll.",
                    authority="Lab. Code § 227.3",
                )

    # -- OWBPA ------------------------------------------------------------
    def _owbpa(self, cut: pd.DataFrame, report: ComplianceReport) -> None:
        cfg = self.cfg
        if not cfg.offering_severance_agreement:
            report.add(
                Severity.INFO, "NO_RELEASE_OFFERED",
                "No release of claims is being offered, so OWBPA's consideration "
                "and disclosure requirements do not apply.",
            )
            return

        over_40 = 0
        if "age_40_plus" in cut.columns:
            over_40 = int(cut["age_40_plus"].fillna(False).astype(bool).sum())
        unknown_age = (
            int(cut["age_40_plus"].isna().sum()) if "age_40_plus" in cut.columns
            else len(cut)
        )

        if unknown_age:
            report.add(
                Severity.WARNING, "AGE_UNKNOWN_FOR_OWBPA",
                f"{unknown_age} affected employee(s) have no age on record. If any "
                f"is 40 or older, an ADEA release without the OWBPA disclosures is "
                f"unenforceable as to their age claim — while remaining "
                f"enforceable as to everything else, so the employer pays for a "
                f"release it does not get.",
                authority="29 U.S.C. § 626(f)",
            )

        if not over_40 and not unknown_age:
            report.add(
                Severity.INFO, "NO_ADEA_RELEASES",
                "No affected employee is 40 or older, so OWBPA's group "
                "termination requirements do not apply to this action.",
            )
            return

        days = (
            OWBPA_GROUP_CONSIDERATION_DAYS if cfg.is_group_termination_program
            else OWBPA_INDIVIDUAL_CONSIDERATION_DAYS
        )
        report.obligations.append(Obligation(
            code="OWBPA_DELIVER_AGREEMENT",
            title="Deliver release agreement and OWBPA disclosures",
            authority="29 U.S.C. § 626(f)(1)",
            description=(
                f"The consideration period runs from delivery of the final "
                f"agreement, so it must be delivered at least {days} days before "
                f"any signature is accepted. Material changes restart the clock."
            ),
            due_date=cfg.proposed_separation_date - dt.timedelta(days=days),
            applies_to=f"{over_40} employee(s) age 40+",
            severity=Severity.ERROR,
        ))
        report.obligations.append(Obligation(
            code="OWBPA_CONSIDERATION",
            title=f"{days}-day consideration period ends",
            authority="29 U.S.C. § 626(f)(1)(F)",
            description=(
                f"An employee 40 or older must have {days} days to consider a "
                f"release of ADEA claims"
                + (" in a group termination program" if cfg.is_group_termination_program
                   else "")
                + ". The period runs from delivery of the final agreement; "
                "material changes restart it."
            ),
            due_date=cfg.proposed_separation_date,
            applies_to=f"{over_40} employee(s) age 40+",
            severity=Severity.ERROR,
        ))
        report.obligations.append(Obligation(
            code="OWBPA_REVOCATION",
            title="7-day revocation period ends; release becomes effective",
            authority="29 U.S.C. § 626(f)(1)(G)",
            description=(
                "After signing, the employee has 7 days to revoke. The agreement "
                "is not effective until that period expires, and it cannot be "
                "waived or shortened."
            ),
            due_date=(
                cfg.proposed_separation_date
                + dt.timedelta(days=OWBPA_REVOCATION_DAYS)
            ),
            applies_to=f"{over_40} employee(s) age 40+",
            severity=Severity.ERROR,
        ))

        if cfg.is_group_termination_program:
            report.obligations.append(Obligation(
                code="OWBPA_DISCLOSURE",
                title="Provide decisional unit disclosure (job titles and ages)",
                authority="29 U.S.C. § 626(f)(1)(H)",
                description=(
                    "In a group termination program, each employee 40 or older "
                    "must receive, with the agreement, the class or unit covered, "
                    "the eligibility factors, the time limits, and the job titles "
                    "and ages of all individuals selected and all individuals in "
                    "the same unit not selected. Defining the decisional unit is "
                    "a legal judgment, not a reporting choice — and the "
                    "disclosure is what tells a plaintiff's lawyer whether the "
                    "selection skewed by age."
                ),
                due_date=cfg.proposed_separation_date,
                applies_to=f"{over_40} employee(s) age 40+",
                severity=Severity.ERROR,
            ))
            report.add(
                Severity.WARNING, "OWBPA_DECISIONAL_UNIT_REQUIRED",
                f"{over_40} affected employee(s) are 40 or older and a group "
                f"termination program is indicated, so the OWBPA disclosure of "
                f"job titles and ages is required. The decisional unit must be "
                f"defined by counsel before the disclosure is prepared; the "
                f"comparison groups used for selection scoring are not "
                f"automatically the correct decisional unit.",
                authority="29 U.S.C. § 626(f)(1)(H)",
            )

    # -- benefits and agency notices --------------------------------------
    def _benefit_and_agency_notices(self, report: ComplianceReport) -> None:
        sep = self.cfg.proposed_separation_date
        report.obligations.append(Obligation(
            code="EDD_CHANGE_NOTICE",
            title="Provide EDD Notice of Change in Relationship",
            authority="Unemp. Ins. Code § 1089; 22 CCR § 1089-1",
            description=(
                "Written notice of the change in employment relationship must be "
                "given to the employee no later than the effective date."
            ),
            due_date=sep,
        ))
        report.obligations.append(Obligation(
            code="DE2320_PAMPHLET",
            title="Provide EDD 'For Your Benefit' pamphlet (DE 2320)",
            authority="Unemp. Ins. Code § 1089",
            description="Must be given to each separating employee at separation.",
            due_date=sep,
        ))
        report.obligations.append(Obligation(
            code="COBRA_NOTICE",
            title="Send COBRA / Cal-COBRA election notice",
            authority="29 U.S.C. § 1166; Health & Safety Code § 1366.20 et seq.",
            description=(
                "The plan administrator must furnish the election notice within "
                "44 days of the qualifying event when the employer is also the "
                "administrator."
            ),
            due_date=sep + dt.timedelta(days=COBRA_NOTICE_DAYS),
        ))
        report.obligations.append(Obligation(
            code="HIPP_NOTICE",
            title="Provide Health Insurance Premium Payment (HIPP) notice",
            authority="Cal. Health & Safety Code § 1366.5",
            description="California-specific notice accompanying continuation coverage.",
            due_date=sep,
        ))

    # -- individual conditions --------------------------------------------
    def _individual_conditions(
        self, cut: pd.DataFrame, report: ComplianceReport
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []

        for _, r in cut.iterrows():
            flags: list[str] = []
            notes: list[str] = []

            leave = r.get("leave_status")
            if pd.notna(leave) and str(leave).strip():
                flags.append("PROTECTED_LEAVE")
                notes.append(
                    f"On leave ({leave}). CFRA/FMLA/PDL do not immunize an "
                    f"employee from a reduction, but the employer must show the "
                    f"selection would have occurred regardless of the leave, and "
                    f"reinstatement rights may apply. Individual legal review "
                    f"required."
                )
            if bool(r.get("union_flag")):
                flags.append("UNION")
                notes.append(
                    "Bargaining unit member. Check the CBA for seniority, "
                    "bumping, and notice provisions, and assess the duty to "
                    "bargain over the decision and its effects."
                )
            visa = r.get("visa_status")
            if pd.notna(visa) and str(visa).strip():
                flags.append("WORK_VISA")
                notes.append(
                    f"Sponsored work authorization ({visa}). Termination may "
                    f"trigger return-transportation obligations and a limited "
                    f"grace period; immigration counsel must advise before notice."
                )
            if r.get("age_40_plus") is True or bool(r.get("age_40_plus")):
                flags.append("AGE_40_PLUS")

            if flags:
                rows.append({
                    "employee_id": r.get("employee_id"),
                    "department": r.get("department"),
                    "job_title": r.get("job_title"),
                    "flags": "|".join(flags),
                    "notes": " ".join(notes),
                })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["employee_id", "department", "job_title", "flags", "notes"]
        )

        leave_n = int(df["flags"].str.contains("PROTECTED_LEAVE").sum()) if len(df) else 0
        union_n = int(df["flags"].str.contains("UNION").sum()) if len(df) else 0
        visa_n = int(df["flags"].str.contains("WORK_VISA").sum()) if len(df) else 0

        if leave_n:
            report.add(
                Severity.ERROR, "SELECTED_ON_PROTECTED_LEAVE",
                f"{leave_n} affected employee(s) are on protected leave. Each "
                f"requires documented, leave-independent justification reviewed "
                f"by counsel before notice is issued.",
                authority="Gov. Code § 12945.2 (CFRA); 29 U.S.C. § 2615 (FMLA)",
            )
        if union_n:
            report.add(
                Severity.ERROR, "SELECTED_UNION_MEMBERS",
                f"{union_n} affected employee(s) are in a bargaining unit. The "
                f"CBA may impose seniority or bumping rights that override the "
                f"selection criteria entirely, and effects bargaining may be "
                f"required before implementation.",
                authority="29 U.S.C. § 158(d)",
            )
        if visa_n:
            report.add(
                Severity.WARNING, "SELECTED_VISA_HOLDERS",
                f"{visa_n} affected employee(s) hold sponsored work "
                f"authorization. Immigration counsel must advise on notice "
                f"timing and status consequences.",
            )
        return df

    # -- gate --------------------------------------------------------------
    def _apply_gate(
        self, report: ComplianceReport, impact: Any, selection: Any
    ) -> None:
        gate = report.gate

        for f in report.findings:
            if f.severity == Severity.ERROR:
                gate.block(f"{f.code}: {f.message.split('.')[0]}.")

        if report.missed_deadlines:
            for o in report.missed_deadlines:
                gate.block(
                    f"{o.code}: deadline of {o.due_date} has already passed."
                )

        # Adverse impact from box 4 gates document generation.
        if impact is not None:
            rep = getattr(impact, "report", impact)
            indicated = getattr(rep, "indicated", None)
            if indicated:
                classes = sorted({c.protected_class for c in indicated})
                gate.block(
                    f"Adverse impact indicated for {', '.join(classes)}; notices "
                    f"must not be generated until counsel has reviewed the "
                    f"finding."
                )
            flagged = getattr(rep, "flagged", None)
            if flagged and not indicated:
                gate.warn(
                    f"{len(flagged)} adverse impact comparison(s) are flagged for "
                    f"review but none reached the indicated threshold."
                )
        else:
            gate.warn(
                "No adverse impact analysis was supplied. Notices should not be "
                "generated before box 4 has run on this scenario."
            )

        # Unresolved selection issues from box 3.
        if selection is not None:
            rep = getattr(selection, "report", selection)
            errs = [
                f for f in getattr(rep, "findings", [])
                if getattr(f, "severity", None) == Severity.ERROR
            ]
            if errs:
                gate.block(
                    f"{len(errs)} unresolved selection error(s) from the "
                    f"Selection Criteria Engine, including: {errs[0].message}"
                )
            queue = getattr(selection, "review_queue", None)
            if queue is not None and len(queue):
                gate.warn(
                    f"{len(queue)} employee(s) remain in the selection review "
                    f"queue; the cut list is provisional until it is cleared."
                )

    # -- packaging ---------------------------------------------------------
    def _package(
        self, report: ComplianceReport, flags: pd.DataFrame | None = None
    ) -> ComplianceResult:
        obligations = (
            pd.DataFrame([o.to_dict() for o in report.obligations])
            if report.obligations else pd.DataFrame(
                columns=["code", "title", "authority", "description", "due_date",
                         "applies_to", "missed", "severity"]
            )
        )
        if flags is None:
            flags = pd.DataFrame(
                columns=["employee_id", "department", "job_title", "flags", "notes"]
            )
        return ComplianceResult(report, obligations, report.calendar(), flags)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from .adverse_impact import AdverseImpactAnalyzer
    from .selection_criteria import SelectionEngine, load_plan
    from .workforce_data import load_workforce_csv

    ap = argparse.ArgumentParser(
        description="Determine California compliance obligations for a RIF scenario."
    )
    ap.add_argument("csv_path")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--separation-date", required=True)
    ap.add_argument("--notice-date", default=None)
    ap.add_argument("--company-headcount", type=int, default=None)
    ap.add_argument("--coordination", default="lwdb", choices=["lwdb", "other", "none"])
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    ingest = load_workforce_csv(
        args.csv_path,
        as_of=str(plan.as_of_date) if plan.as_of_date else args.separation_date,
    )
    if ingest.report.is_blocking:
        print("Ingestion is blocking; fix the roster first.")
        return 2

    selection = SelectionEngine(plan).run(ingest.data)
    impact = AdverseImpactAnalyzer().run(selection.scores, scenario=plan.plan_name)

    cfg = ComplianceConfig(
        proposed_separation_date=args.separation_date,
        notice_date=args.notice_date,
        total_company_headcount=args.company_headcount,
        service_coordination=args.coordination,
    )
    result = ComplianceEngine(cfg).run(
        selection.scores, impact=impact, selection=selection,
        scenario=plan.plan_name,
    )

    if not args.quiet:
        print(result.report.to_markdown())

    if args.outdir:
        paths = result.write(args.outdir)
        print("\nWrote:")
        for k, p in paths.items():
            print(f"  {k}: {p}")

    return 0 if result.gate.may_generate_documents else 1


if __name__ == "__main__":
    raise SystemExit(main())
