"""
severance_pay.py
================

Severance & Pay Engine for the California RIF Copilot (box 6).

Turns a cut list into per-employee money: severance under a documented formula,
final wages, accrued vacation payout, COBRA subsidy, estimated withholding, and
the aggregate payroll impact with its cash-flow timing.

This supersedes the provisional ``CostAssumptions`` in box 2. Where box 2 needed
a consistent basis for comparing scenarios, this module produces the figures a
payroll team would actually work from — and is correspondingly fussier about the
distinctions that make those figures wrong.

The distinctions that matter
----------------------------
**Vacation is paid at the final rate, not the rate it was earned at.** Vested
vacation is wages under Labor Code § 227.3 and cannot be forfeited. An employee
who accrued time three raises ago is paid out at today's rate.

**Paid sick leave is generally not payable on separation; vacation is.** If the
employer runs separate sick and vacation banks, only vacation is cashed out. If
it runs a combined PTO bank, the whole balance is generally treated as vacation
and the whole balance is payable. Getting this backwards either shorts employees
on wages or hands out money that was never owed, so the module refuses to guess:
``leave_policy`` must be declared.

**Final wages are due at separation regardless of what the severance agreement
says.** Severance can be conditioned on a release. Wages the employee has
already earned cannot be. If the only way an employee receives their final
paycheck is by signing, that is not a release — it is a § 203 violation with a
release stapled to it.

**Labeling changes the tax and UI treatment.** True dismissal severance under
CUIC § 1265 is not wages for unemployment purposes and is not subject to SDI.
The same money labeled "wages in lieu of notice" is both. The label is a legal
choice with consequences, not a drafting preference.

Withholding figures are estimates
---------------------------------
Every tax figure here is an estimate for budgeting. Actual withholding depends
on year-to-date wages, W-4 and DE 4 elections, benefit deductions, garnishments,
and the method payroll elects. Payroll runs payroll; this module sizes the
liability.

Usage
-----
    from .severance_pay import PayConfig, SeveranceFormula, SeverancePayEngine

    cfg = PayConfig(
        separation_date="2026-10-30",
        leave_policy="separate",
        formula=SeveranceFormula(weeks_per_year=2.0, min_weeks=4, max_weeks=26),
    )
    result = SeverancePayEngine(cfg).run(selection.cut_list)

    result.register        # per-employee payment detail
    result.report.totals   # aggregate payroll impact
    print(result.report.to_markdown())
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .workforce_data import Severity

__all__ = [
    "SeveranceFormula",
    "TaxAssumptions",
    "PayConfig",
    "SeverancePayEngine",
    "SeverancePayResult",
    "PayReport",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Tax constants (2026)
#
# Verified against public 2026 payroll guidance during the build. These change
# annually and some are genuinely contested in secondary sources — see the note
# on CA_SUPPLEMENTAL_SEVERANCE_RATE. Confirm every figure with payroll or a CPA
# before relying on the output.
# ---------------------------------------------------------------------------

#: Federal flat supplemental withholding, first $1M of supplemental wages.
FEDERAL_SUPPLEMENTAL_RATE = 0.22
#: Mandatory rate on cumulative supplemental wages above $1M in a calendar year.
FEDERAL_SUPPLEMENTAL_RATE_HIGH = 0.37
FEDERAL_SUPPLEMENTAL_HIGH_THRESHOLD = 1_000_000.0

#: California flat supplemental PIT withholding on severance.
#:
#: EDD Publication DE 44 sets 10.23% for *bonuses and stock options* and 6.6%
#: for *other* supplemental wages. Severance is other supplemental wages, so
#: 6.6% is the applicable rate — but secondary sources routinely lump severance
#: in with bonuses and quote 10.23%, and a wrongly applied bonus rate
#: over-withholds by 3.63 points. Confirm with payroll; the engine reports which
#: rate it used.
CA_SUPPLEMENTAL_SEVERANCE_RATE = 0.066
CA_SUPPLEMENTAL_BONUS_RATE = 0.1023

#: FICA.
SOCIAL_SECURITY_RATE = 0.062
SOCIAL_SECURITY_WAGE_BASE = 184_500.0   # 2026; verify — sources disagree
MEDICARE_RATE = 0.0145
ADDITIONAL_MEDICARE_RATE = 0.009
ADDITIONAL_MEDICARE_THRESHOLD = 200_000.0

#: California SDI: 1.3% of all wages, no wage cap (SB 951 removed it).
CA_SDI_RATE = 0.013

#: Employer-side taxes, on the first $7,000 of wages per employee per year.
CA_UI_WAGE_BASE = 7_000.0
CA_ETT_RATE = 0.001
FUTA_WAGE_BASE = 7_000.0

#: Working days per year, for converting annual pay to a daily rate.
WORKING_DAYS_PER_YEAR = 260.0
WEEKS_PER_YEAR = 52.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SeveranceFormula:
    """A documented, uniformly applied severance formula.

    Applying a formula uniformly is what makes severance defensible. Ad hoc
    per-person amounts invite a comparison the employer will have to explain,
    so the engine computes from the formula and records any override
    separately rather than silently absorbing it.
    """

    #: Weeks of base pay per year of service.
    weeks_per_year: float = 2.0
    min_weeks: float = 4.0
    max_weeks: float = 26.0
    #: Optional per-level overrides: {job_level: weeks_per_year}.
    weeks_per_year_by_level: dict[str, float] = field(default_factory=dict)
    #: Optional flat weeks added regardless of tenure.
    base_weeks: float = 0.0
    #: Round service years down to whole years, or credit partial years.
    credit_partial_years: bool = True
    #: Round the resulting weeks to this increment (0 disables).
    round_weeks_to: float = 0.0
    #: Include target bonus in the severance base.
    include_target_bonus: bool = False

    def weeks_for(
        self, tenure_years: float, job_level: str | None = None
    ) -> tuple[float, bool, bool]:
        """Return (weeks, hit_floor, hit_cap)."""
        rate = self.weeks_per_year
        if job_level and job_level in self.weeks_per_year_by_level:
            rate = self.weeks_per_year_by_level[job_level]

        years = tenure_years if self.credit_partial_years else math.floor(tenure_years)
        weeks = self.base_weeks + years * rate

        hit_floor = weeks < self.min_weeks
        hit_cap = weeks > self.max_weeks
        weeks = min(max(weeks, self.min_weeks), self.max_weeks)

        if self.round_weeks_to:
            weeks = round(weeks / self.round_weeks_to) * self.round_weeks_to
        return round(weeks, 2), hit_floor, hit_cap

    def to_dict(self) -> dict[str, Any]:
        return {
            "weeks_per_year": self.weeks_per_year,
            "min_weeks": self.min_weeks,
            "max_weeks": self.max_weeks,
            "weeks_per_year_by_level": self.weeks_per_year_by_level,
            "base_weeks": self.base_weeks,
            "credit_partial_years": self.credit_partial_years,
            "round_weeks_to": self.round_weeks_to,
            "include_target_bonus": self.include_target_bonus,
        }


@dataclass
class TaxAssumptions:
    federal_supplemental_rate: float = FEDERAL_SUPPLEMENTAL_RATE
    ca_supplemental_rate: float = CA_SUPPLEMENTAL_SEVERANCE_RATE
    social_security_rate: float = SOCIAL_SECURITY_RATE
    social_security_wage_base: float = SOCIAL_SECURITY_WAGE_BASE
    medicare_rate: float = MEDICARE_RATE
    ca_sdi_rate: float = CA_SDI_RATE
    #: True dismissal severance under CUIC § 1265 is not subject to SDI. Money
    #: labeled "wages in lieu of notice" is. Set True only if the payment is
    #: genuinely structured as wages in lieu of notice.
    is_wages_in_lieu_of_notice: bool = False
    #: Employer-side burden on severance, as a fraction (FICA match etc.).
    employer_burden_rate: float = 0.0765

    def to_dict(self) -> dict[str, Any]:
        return {
            "federal_supplemental_rate": self.federal_supplemental_rate,
            "ca_supplemental_rate": self.ca_supplemental_rate,
            "social_security_rate": self.social_security_rate,
            "social_security_wage_base": self.social_security_wage_base,
            "medicare_rate": self.medicare_rate,
            "ca_sdi_rate": self.ca_sdi_rate,
            "is_wages_in_lieu_of_notice": self.is_wages_in_lieu_of_notice,
            "employer_burden_rate": self.employer_burden_rate,
        }


@dataclass
class PayConfig:
    separation_date: dt.date
    formula: SeveranceFormula = field(default_factory=SeveranceFormula)
    taxes: TaxAssumptions = field(default_factory=TaxAssumptions)

    #: "separate" — distinct vacation and sick banks; only vacation pays out.
    #: "combined" — one PTO bank; the whole balance is treated as vacation.
    #: Must be declared; the payout differs materially between them.
    leave_policy: str = ""

    #: Employer-subsidized COBRA months and monthly employer cost.
    cobra_months: float = 3.0
    cobra_monthly_cost: float = 1_400.0

    #: Flat per-employee outplacement and administrative cost.
    outplacement_cost: float = 1_500.0
    admin_cost_per_employee: float = 250.0

    #: When severance is actually disbursed. Final wages are due at separation
    #: regardless; this only affects cash-flow timing of the severance itself.
    severance_payment_date: dt.date | None = None
    #: Pay severance as a lump sum or over a salary-continuation schedule.
    severance_schedule: str = "lump_sum"   # lump_sum | continuation

    #: Per-employee overrides, {employee_id: weeks}. Recorded and reported, not
    #: silently folded into the formula result.
    week_overrides: dict[str, float] = field(default_factory=dict)

    #: If WARN notice is short, the back-pay obligation may be offset by
    #: severance only if the agreement is structured for it.
    warn_shortfall_days: int = 0

    def __post_init__(self) -> None:
        self.separation_date = _as_date(self.separation_date)
        if self.severance_payment_date is not None:
            self.severance_payment_date = _as_date(self.severance_payment_date)


def _as_date(value: Any) -> dt.date:
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    return pd.Timestamp(value).date()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class PayFinding:
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
class PayReport:
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
    scenario: str = ""
    separation_date: str = ""
    employee_count: int = 0
    totals: dict[str, Any] = field(default_factory=dict)
    cash_flow: list[dict[str, Any]] = field(default_factory=list)
    assumptions: dict[str, Any] = field(default_factory=dict)
    findings: list[PayFinding] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, **kw: Any) -> None:
        self.findings.append(PayFinding(severity, code, message, **kw))

    @property
    def has_errors(self) -> bool:
        return any(f.severity == Severity.ERROR for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "scenario": self.scenario,
            "separation_date": self.separation_date,
            "employee_count": self.employee_count,
            "totals": self.totals,
            "cash_flow": self.cash_flow,
            "assumptions": self.assumptions,
            "findings": [f.to_dict() for f in self.findings],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self) -> str:
        t = self.totals
        L: list[str] = []
        L.append("# Severance & Pay Analysis")
        L.append("")
        L.append(
            "> **Prepared at the direction of counsel / privileged and "
            "confidential — confirm labeling before circulating.**"
        )
        L.append("")
        L.append(f"**Scenario:** {self.scenario or '(unnamed)'}  ")
        L.append(f"**Generated:** {self.generated_at}  ")
        L.append(f"**Separation date:** {self.separation_date}  ")
        L.append(f"**Employees:** {self.employee_count}  ")
        L.append("")

        L.append("## Cost summary")
        L.append("")
        L.append("| Component | Amount |")
        L.append("|---|---|")
        L.append(f"| Severance (gross) | ${t.get('severance_gross', 0):,.2f} |")
        L.append(f"| Accrued vacation payout (§ 227.3) | ${t.get('vacation_payout', 0):,.2f} |")
        wages = t.get("final_wages")
        L.append(
            f"| Final wages through separation | "
            + (f"${wages:,.2f} |" if wages is not None else "*supplied by payroll* |")
        )
        L.append(f"| COBRA subsidy | ${t.get('cobra_cost', 0):,.2f} |")
        L.append(f"| Outplacement and administration | ${t.get('admin_cost', 0):,.2f} |")
        L.append(f"| Employer payroll tax on severance | ${t.get('employer_tax', 0):,.2f} |")
        L.append(f"| **Total employer cost** | **${t.get('total_employer_cost', 0):,.2f}** |")
        L.append("")
        L.append(f"| Average severance per employee | ${t.get('avg_severance', 0):,.2f} |")
        L.append(f"| Median weeks of severance | {t.get('median_weeks', 0):.1f} |")
        L.append(f"| Estimated employee withholding | ${t.get('employee_withholding', 0):,.2f} |")
        L.append(f"| Estimated net to employees | ${t.get('net_to_employees', 0):,.2f} |")
        L.append("")

        if self.cash_flow:
            L.append("## Cash flow")
            L.append("")
            L.append("| Date | Component | Amount |")
            L.append("|---|---|---|")
            for c in self.cash_flow:
                L.append(f"| {c['date']} | {c['component']} | ${c['amount']:,.2f} |")
            L.append("")

        L.append("## Formula")
        L.append("")
        f = self.assumptions.get("formula", {})
        for k, v in f.items():
            if v not in (None, {}, 0, 0.0, False) or k in ("weeks_per_year",):
                L.append(f"- `{k}`: {v}")
        L.append("")
        L.append(
            "A formula applied uniformly is what makes severance defensible. "
            "Per-person deviations are reported separately in the register "
            "rather than folded into the formula result, because a deviation "
            "nobody can explain later is the one that gets compared."
        )
        L.append("")

        L.append("## Tax assumptions")
        L.append("")
        for k, v in self.assumptions.get("taxes", {}).items():
            L.append(f"- `{k}`: {v}")
        L.append("")
        L.append(
            "**Withholding figures are budgeting estimates.** Actual withholding "
            "depends on year-to-date wages, W-4 and DE 4 elections, benefit "
            "deductions, garnishments, and the method payroll elects. California "
            "sets 10.23% for bonuses and stock options and 6.6% for other "
            "supplemental wages; severance falls in the latter, but secondary "
            "sources frequently conflate the two. Confirm the rate with payroll."
        )
        L.append("")

        if self.findings:
            L.append("## Findings")
            L.append("")
            order = {Severity.ERROR: 0, Severity.WARNING: 1, Severity.INFO: 2}
            for fi in sorted(self.findings, key=lambda x: order.get(x.severity, 9)):
                auth = f" *({fi.authority})*" if fi.authority else ""
                who = f" [{fi.employee_id}]" if fi.employee_id else ""
                L.append(f"- **[{fi.severity}] {fi.code}**{auth}{who} — {fi.message}")
            L.append("")

        L.append("## Non-negotiables")
        L.append("")
        L.append(
            "**Final wages are due at separation regardless of the severance "
            "agreement.** Severance may be conditioned on a release. Wages "
            "already earned — including vested vacation — may not. If an "
            "employee can only obtain their final paycheck by signing, that is a "
            "Labor Code § 203 problem with a release attached, and the release "
            "itself may be unenforceable for want of consideration."
        )
        L.append("")
        L.append(
            "**Vested vacation is paid at the final rate of pay**, not the rate "
            "at which it accrued, and it cannot be forfeited or capped "
            "retroactively (Lab. Code § 227.3)."
        )
        L.append("")
        L.append(
            "**How the payment is labeled changes its treatment.** Dismissal "
            "severance under CUIC § 1265 is not wages for unemployment purposes "
            "and is not subject to SDI. The identical amount labeled \"wages in "
            "lieu of notice\" is both, and will delay the employee's "
            "unemployment benefits. Choose the label deliberately with counsel."
        )
        L.append("")
        L.append("---")
        L.append(
            "_Estimates for planning. Not tax, payroll, or legal advice. Rates "
            "change annually; verify against EDD Publication DE 44 and IRS "
            "Publication 15 for the current year before processing._"
        )
        return "\n".join(L)


@dataclass
class SeverancePayResult:
    register: pd.DataFrame
    report: PayReport

    def write(self, outdir: str | Path, stem: str = "severance") -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "report_md": outdir / f"{stem}_report.md",
            "register": outdir / f"{stem}_payroll_register.csv",
            "report_json": outdir / f"{stem}_report.json",
        }
        self.register.to_csv(paths["register"], index=False)
        self.report.to_json(paths["report_json"])
        paths["report_md"].write_text(self.report.to_markdown(), encoding="utf-8")
        return paths


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class SeverancePayEngine:
    """Computes severance, final pay, and payroll impact for a cut list."""

    def __init__(self, config: PayConfig) -> None:
        self.cfg = config

    # -- public ----------------------------------------------------------
    def run(self, cut_list: pd.DataFrame, scenario: str = "") -> SeverancePayResult:
        cfg = self.cfg
        report = PayReport(
            scenario=scenario,
            separation_date=cfg.separation_date.isoformat(),
            assumptions={
                "formula": cfg.formula.to_dict(),
                "taxes": cfg.taxes.to_dict(),
                "leave_policy": cfg.leave_policy,
                "cobra_months": cfg.cobra_months,
                "cobra_monthly_cost": cfg.cobra_monthly_cost,
            },
        )

        if cut_list is None or cut_list.empty:
            report.add(
                Severity.WARNING, "NO_EMPLOYEES",
                "The cut list is empty; there is nothing to compute.",
            )
            return SeverancePayResult(_empty_register(), report)

        if not self._check_leave_policy(report):
            return SeverancePayResult(_empty_register(), report)

        rows = [self._compute_employee(r, report) for _, r in cut_list.iterrows()]
        register = pd.DataFrame(rows)
        report.employee_count = len(register)

        self._aggregate(register, report)
        self._cash_flow(register, report)
        self._global_checks(register, report)
        return SeverancePayResult(register, report)

    # -- policy ------------------------------------------------------------
    def _check_leave_policy(self, report: PayReport) -> bool:
        policy = (self.cfg.leave_policy or "").strip().lower()
        if policy not in ("separate", "combined"):
            report.add(
                Severity.ERROR, "LEAVE_POLICY_UNDECLARED",
                "leave_policy must be 'separate' or 'combined'. California paid "
                "sick leave is generally not payable on separation, while vested "
                "vacation is. Under a combined PTO bank the whole balance is "
                "generally treated as vacation and is payable. Guessing either "
                "way is wrong: one shorts employees on earned wages, the other "
                "pays out money never owed.",
                authority="Lab. Code §§ 227.3, 246",
            )
            return False
        if policy == "combined":
            report.add(
                Severity.WARNING, "COMBINED_PTO_BANK",
                "A combined PTO bank is declared, so the entire balance is "
                "treated as vested vacation and paid out. Confirm the plan "
                "document actually combines the banks — if sick time is tracked "
                "separately anywhere in the policy, that portion is generally "
                "not payable.",
                authority="Lab. Code § 227.3",
            )
        return True

    # -- per employee -------------------------------------------------------
    def _compute_employee(self, row: pd.Series, report: PayReport) -> dict[str, Any]:
        cfg = self.cfg
        emp_id = str(row.get("employee_id")) if pd.notna(row.get("employee_id")) else None

        annual = row.get("annualized_pay")
        annual = float(annual) if pd.notna(annual) else None
        tenure = row.get("tenure_years")
        tenure = float(tenure) if pd.notna(tenure) else None
        level = row.get("job_level") if pd.notna(row.get("job_level")) else None

        out: dict[str, Any] = {
            "employee_id": emp_id,
            "job_title": row.get("job_title"),
            "department": row.get("department"),
            "job_level": level,
            "tenure_years": tenure,
            "annualized_pay": annual,
        }

        if annual is None:
            report.add(
                Severity.ERROR, "NO_PAY_DATA",
                "No annualized pay on record, so severance and final pay cannot "
                "be computed. This employee cannot be paid correctly on "
                "separation day, which is precisely how § 203 penalties start.",
                authority="Lab. Code § 203", employee_id=emp_id,
            )
            out.update(_blank_amounts())
            out["status"] = "uncomputable"
            return out

        if tenure is None:
            report.add(
                Severity.ERROR, "NO_TENURE_DATA",
                "No tenure on record, so severance weeks cannot be determined "
                "under the formula.",
                employee_id=emp_id,
            )
            out.update(_blank_amounts())
            out["status"] = "uncomputable"
            return out

        # -- severance base --------------------------------------------------
        base = annual
        if cfg.formula.include_target_bonus:
            bonus_pct = row.get("target_bonus_pct")
            if pd.notna(bonus_pct):
                base = annual * (1.0 + float(bonus_pct))
        weekly = base / WEEKS_PER_YEAR

        weeks, hit_floor, hit_cap = cfg.formula.weeks_for(tenure, level)
        formula_weeks = weeks

        override = cfg.week_overrides.get(emp_id or "")
        if override is not None:
            weeks = float(override)
            report.add(
                Severity.WARNING, "SEVERANCE_OVERRIDE",
                f"Severance overridden from {formula_weeks:g} to {weeks:g} weeks. "
                f"Uniform application of a documented formula is what makes a "
                f"severance program defensible; each deviation needs a recorded "
                f"business reason and should be reviewed for pattern across "
                f"protected groups before the offers go out.",
                employee_id=emp_id,
            )

        severance = round(weekly * weeks, 2)

        if hit_floor:
            report.add(
                Severity.INFO, "SEVERANCE_FLOOR_APPLIED",
                f"Formula produced less than the {cfg.formula.min_weeks:g}-week "
                f"minimum; the floor was applied.",
                employee_id=emp_id,
            )
        if hit_cap:
            report.add(
                Severity.INFO, "SEVERANCE_CAP_APPLIED",
                f"Formula produced more than the {cfg.formula.max_weeks:g}-week "
                f"cap ({tenure:.1f} years of service); the cap was applied.",
                employee_id=emp_id,
            )

        # -- vacation payout, at the FINAL rate -------------------------------
        vac_hours = row.get("accrued_vacation_hours")
        hourly = row.get("hourly_equivalent_rate")
        if pd.isna(hourly) and annual:
            hourly = annual / (WORKING_DAYS_PER_YEAR * 8)
        vacation = None
        if pd.notna(vac_hours) and pd.notna(hourly):
            vacation = round(float(vac_hours) * float(hourly), 2)
        else:
            report.add(
                Severity.WARNING, "VACATION_BALANCE_MISSING",
                "No accrued vacation balance on record. A blank is not zero — "
                "vested vacation is wages and must be paid out. Confirm the "
                "balance with payroll before separation day.",
                authority="Lab. Code § 227.3", employee_id=emp_id,
            )

        # Sick leave is tracked but not paid out under a separate-bank policy.
        sick_hours = row.get("accrued_sick_hours")
        sick_payout = 0.0
        if pd.notna(sick_hours) and float(sick_hours) > 0:
            if cfg.leave_policy == "combined":
                sick_payout = round(float(sick_hours) * float(hourly), 2)
            else:
                report.add(
                    Severity.INFO, "SICK_LEAVE_NOT_PAID_OUT",
                    f"{float(sick_hours):.0f} hour(s) of accrued sick leave are "
                    f"not paid out under a separate-bank policy. If the employee "
                    f"is rehired within 12 months, unused sick leave generally "
                    f"must be reinstated.",
                    authority="Lab. Code § 246(f)", employee_id=emp_id,
                )

        # -- withholding estimate ---------------------------------------------
        taxable = severance + (vacation or 0.0) + sick_payout
        tax = self._withholding(taxable)

        cobra = round(cfg.cobra_months * cfg.cobra_monthly_cost, 2)
        admin = round(cfg.outplacement_cost + cfg.admin_cost_per_employee, 2)
        employer_tax = round(taxable * cfg.taxes.employer_burden_rate, 2)

        out.update({
            "severance_weeks": weeks,
            "formula_weeks": formula_weeks,
            "overridden": override is not None,
            "weekly_rate": round(weekly, 2),
            "severance_gross": severance,
            "vacation_hours": (
                float(vac_hours) if pd.notna(vac_hours) else None
            ),
            "vacation_payout": vacation,
            "sick_hours_not_paid": (
                float(sick_hours) if pd.notna(sick_hours) and cfg.leave_policy == "separate"
                else 0.0
            ),
            "sick_payout": sick_payout,
            "taxable_separation_pay": round(taxable, 2),
            "est_federal_withholding": tax["federal"],
            "est_ca_withholding": tax["ca"],
            "est_fica": tax["fica"],
            "est_sdi": tax["sdi"],
            "est_total_withholding": tax["total"],
            "est_net_to_employee": round(taxable - tax["total"], 2),
            "cobra_cost": cobra,
            "admin_cost": admin,
            "employer_payroll_tax": employer_tax,
            "total_employer_cost": round(
                severance + (vacation or 0.0) + sick_payout + cobra + admin + employer_tax, 2
            ),
            "status": "computed",
        })
        return out

    def _withholding(self, amount: float) -> dict[str, float]:
        t = self.cfg.taxes
        if amount <= 0:
            return {"federal": 0.0, "ca": 0.0, "fica": 0.0, "sdi": 0.0, "total": 0.0}

        federal = amount * t.federal_supplemental_rate
        ca = amount * t.ca_supplemental_rate

        # Social Security is capped; without year-to-date wages the engine
        # assumes the cap has not been reached, which overstates withholding for
        # high earners. Flagged in the report rather than silently assumed.
        ss = min(amount, t.social_security_wage_base) * t.social_security_rate
        medicare = amount * t.medicare_rate
        fica = ss + medicare

        # SDI applies to wages in lieu of notice but not to true dismissal
        # severance under CUIC § 1265.
        sdi = amount * t.ca_sdi_rate if t.is_wages_in_lieu_of_notice else 0.0

        total = federal + ca + fica + sdi
        return {
            "federal": round(federal, 2), "ca": round(ca, 2),
            "fica": round(fica, 2), "sdi": round(sdi, 2), "total": round(total, 2),
        }

    # -- aggregates ---------------------------------------------------------
    def _aggregate(self, reg: pd.DataFrame, report: PayReport) -> None:
        computed = reg.loc[reg["status"] == "computed"]

        def total(col: str) -> float:
            if col not in computed.columns or computed.empty:
                return 0.0
            return float(computed[col].fillna(0).sum())

        severance = total("severance_gross")
        report.totals = {
            "severance_gross": round(severance, 2),
            "vacation_payout": round(total("vacation_payout"), 2),
            "sick_payout": round(total("sick_payout"), 2),
            "final_wages": None,  # payroll supplies this; see FINAL_WAGES_NOT_COMPUTED
            "cobra_cost": round(total("cobra_cost"), 2),
            "admin_cost": round(total("admin_cost"), 2),
            "employer_tax": round(total("employer_payroll_tax"), 2),
            "total_employer_cost": round(total("total_employer_cost"), 2),
            "employee_withholding": round(total("est_total_withholding"), 2),
            "net_to_employees": round(total("est_net_to_employee"), 2),
            "avg_severance": (
                round(severance / len(computed), 2) if len(computed) else 0.0
            ),
            "median_weeks": (
                float(computed["severance_weeks"].median()) if len(computed) else 0.0
            ),
            "uncomputable_employees": int((reg["status"] != "computed").sum()),
        }

    def _cash_flow(self, reg: pd.DataFrame, report: PayReport) -> None:
        cfg = self.cfg
        t = report.totals
        sep = cfg.separation_date
        pay_date = cfg.severance_payment_date or sep

        flows = [
            {
                "date": sep.isoformat(),
                "component": "Final wages and vested vacation (due at separation)",
                "amount": t.get("vacation_payout", 0.0) + t.get("sick_payout", 0.0),
            },
        ]
        if cfg.severance_schedule == "continuation":
            weeks = float(reg["severance_weeks"].median()) if len(reg) else 0.0
            flows.append({
                "date": f"{pay_date.isoformat()} onward",
                "component": f"Severance via salary continuation (~{weeks:.0f} weeks)",
                "amount": t.get("severance_gross", 0.0),
            })
        else:
            flows.append({
                "date": pay_date.isoformat(),
                "component": "Severance lump sum",
                "amount": t.get("severance_gross", 0.0),
            })
        flows.append({
            "date": f"{sep.isoformat()} + {cfg.cobra_months:.0f} months",
            "component": "COBRA subsidy",
            "amount": t.get("cobra_cost", 0.0),
        })
        flows.append({
            "date": sep.isoformat(),
            "component": "Outplacement and administration",
            "amount": t.get("admin_cost", 0.0),
        })
        report.cash_flow = flows

    # -- global checks -------------------------------------------------------
    def _global_checks(self, reg: pd.DataFrame, report: PayReport) -> None:
        cfg = self.cfg

        report.add(
            Severity.WARNING, "FINAL_WAGES_NOT_COMPUTED",
            "Wages earned through the separation date are not computed here: "
            "they depend on the pay period, days actually worked, and any "
            "outstanding expense or commission items. Payroll must supply that "
            "figure, and it is due at the moment of separation.",
            authority="Lab. Code § 201",
        )

        report.add(
            Severity.WARNING, "RELEASE_CANNOT_COVER_EARNED_WAGES",
            "Severance may be conditioned on a release; earned wages and vested "
            "vacation may not. Final pay must be delivered at separation whether "
            "or not the employee signs. If the agreement bundles them, the § 203 "
            "clock runs and the release may fail for want of consideration.",
            authority="Lab. Code §§ 201, 203, 206.5",
        )

        if cfg.severance_payment_date and cfg.severance_payment_date < cfg.separation_date:
            report.add(
                Severity.WARNING, "SEVERANCE_BEFORE_SEPARATION",
                "Severance is scheduled to pay before the separation date. If a "
                "release is involved, paying before the OWBPA consideration and "
                "revocation periods have run may undermine enforceability.",
                authority="29 U.S.C. § 626(f)",
            )

        if cfg.taxes.is_wages_in_lieu_of_notice:
            report.add(
                Severity.WARNING, "WAGES_IN_LIEU_OF_NOTICE_ELECTED",
                "The payment is configured as wages in lieu of notice rather than "
                "dismissal severance. That makes it subject to SDI and treats it "
                "as wages for unemployment purposes, which will delay the "
                "employee's UI benefits. If the intent was ordinary severance "
                "under CUIC § 1265, change the label and the structure.",
                authority="Unemp. Ins. Code § 1265",
            )
        else:
            report.add(
                Severity.INFO, "DISMISSAL_SEVERANCE_TREATMENT",
                "Treated as dismissal severance under CUIC § 1265: no SDI "
                "withholding, and not wages for unemployment purposes. This "
                "depends on how the agreement is actually drafted, not on the "
                "setting in this config.",
                authority="Unemp. Ins. Code § 1265",
            )

        if cfg.warn_shortfall_days > 0:
            report.add(
                Severity.ERROR, "WARN_SHORTFALL_OFFSET",
                f"A WARN notice shortfall of {cfg.warn_shortfall_days} day(s) is "
                f"configured. Back pay for the shortfall is a separate statutory "
                f"obligation from severance. Severance offsets WARN liability "
                f"only where the agreement is specifically structured to do so — "
                f"an ordinary severance payment does not automatically reduce it, "
                f"and treating it as though it does understates the liability.",
                authority="Lab. Code § 1402",
            )

        # Severance dispersion across protected groups is not measured here, but
        # overrides are the place it would show up first.
        overridden = reg.loc[reg.get("overridden", False) == True]  # noqa: E712
        if len(overridden):
            report.add(
                Severity.WARNING, "OVERRIDES_NEED_IMPACT_REVIEW",
                f"{len(overridden)} employee(s) received a severance amount "
                f"outside the formula. Run the same adverse impact review over "
                f"severance amounts that was run over selection — discretionary "
                f"deviations are where disparities appear even when the selection "
                f"itself was clean.",
            )

        if reg["status"].ne("computed").any():
            n = int(reg["status"].ne("computed").sum())
            report.add(
                Severity.ERROR, "INCOMPLETE_REGISTER",
                f"{n} employee(s) could not be computed and are excluded from the "
                f"totals. The register is not ready for payroll until every "
                f"employee has a figure.",
            )

        # High earners: the SS cap assumption overstates withholding.
        if "annualized_pay" in reg.columns:
            high = reg.loc[
                reg["annualized_pay"].fillna(0) > self.cfg.taxes.social_security_wage_base
            ]
            if len(high):
                report.add(
                    Severity.INFO, "SS_CAP_ASSUMPTION",
                    f"{len(high)} employee(s) earn above the Social Security wage "
                    f"base. Without year-to-date payroll data the engine assumes "
                    f"the cap has not been reached, which overstates their "
                    f"withholding. Payroll's figures will be lower.",
                )


def _blank_amounts() -> dict[str, Any]:
    return {
        "severance_weeks": None, "formula_weeks": None, "overridden": False,
        "weekly_rate": None, "severance_gross": None, "vacation_hours": None,
        "vacation_payout": None, "sick_hours_not_paid": None, "sick_payout": None,
        "taxable_separation_pay": None, "est_federal_withholding": None,
        "est_ca_withholding": None, "est_fica": None, "est_sdi": None,
        "est_total_withholding": None, "est_net_to_employee": None,
        "cobra_cost": None, "admin_cost": None, "employer_payroll_tax": None,
        "total_employer_cost": None,
    }


def _empty_register() -> pd.DataFrame:
    cols = ["employee_id", "job_title", "department", "job_level", "tenure_years",
            "annualized_pay", "status"] + list(_blank_amounts())
    return pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from .selection_criteria import SelectionEngine, load_plan
    from .workforce_data import load_workforce_csv

    ap = argparse.ArgumentParser(
        description="Compute severance, final pay, and payroll impact."
    )
    ap.add_argument("csv_path")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--separation-date", required=True)
    ap.add_argument("--leave-policy", required=True, choices=["separate", "combined"])
    ap.add_argument("--weeks-per-year", type=float, default=2.0)
    ap.add_argument("--min-weeks", type=float, default=4.0)
    ap.add_argument("--max-weeks", type=float, default=26.0)
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
    cfg = PayConfig(
        separation_date=args.separation_date,
        leave_policy=args.leave_policy,
        formula=SeveranceFormula(
            weeks_per_year=args.weeks_per_year,
            min_weeks=args.min_weeks,
            max_weeks=args.max_weeks,
        ),
    )
    result = SeverancePayEngine(cfg).run(selection.cut_list, scenario=plan.plan_name)

    if not args.quiet:
        print(result.report.to_markdown())

    if args.outdir:
        paths = result.write(args.outdir)
        print("\nWrote:")
        for k, p in paths.items():
            print(f"  {k}: {p}")

    return 1 if result.report.has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
