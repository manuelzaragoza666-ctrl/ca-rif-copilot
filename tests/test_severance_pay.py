"""
Tests for severance_pay.py

Run with pytest, or standalone:  python test_severance_pay.py
"""

from __future__ import annotations

# Make the package importable whether this file is run directly, via pytest, or
# from another working directory.
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import datetime as dt
import json
import tempfile

import pandas as pd

try:
    import pytest
except ImportError:  # minimal stand-in so the suite runs without pytest
    import contextlib
    import types

    class _Approx:
        def __init__(self, expected, rel=1e-6):
            self.expected, self.rel = expected, rel

        def __eq__(self, other):
            return abs(float(other) - self.expected) <= max(
                self.rel * abs(self.expected), 1e-9
            )

    @contextlib.contextmanager
    def _raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"expected {exc.__name__}")

    pytest = types.SimpleNamespace(approx=_Approx, raises=_raises)

from rif_copilot.severance_pay import (
    CA_SUPPLEMENTAL_BONUS_RATE,
    CA_SUPPLEMENTAL_SEVERANCE_RATE,
    FEDERAL_SUPPLEMENTAL_RATE,
    PayConfig,
    SeveranceFormula,
    SeverancePayEngine,
    TaxAssumptions,
)
from rif_copilot.workforce_data import Severity

SEP = dt.date(2026, 10, 30)


# --- fixtures --------------------------------------------------------------


def employee(**kw) -> dict:
    base = {
        "employee_id": "E1", "job_title": "Engineer", "department": "Engineering",
        "job_level": "L3", "tenure_years": 10.0, "annualized_pay": 156000.0,
        "hourly_equivalent_rate": 75.0, "accrued_vacation_hours": 100.0,
    }
    base.update(kw)
    return base


def cut(*emps) -> pd.DataFrame:
    return pd.DataFrame(list(emps) or [employee()])


def cfg(**kw) -> PayConfig:
    defaults = dict(separation_date=SEP, leave_policy="separate")
    defaults.update(kw)
    return PayConfig(**defaults)


def run(df=None, config=None, **kw):
    return SeverancePayEngine(config or cfg()).run(df if df is not None else cut(), **kw)


def codes(result) -> set[str]:
    return {f.code for f in result.report.findings}


# --- the formula -----------------------------------------------------------


def test_weeks_are_service_times_rate():
    f = SeveranceFormula(weeks_per_year=2.0, min_weeks=0, max_weeks=100)
    weeks, floor, cap = f.weeks_for(7.0)
    assert weeks == 14.0
    assert not floor and not cap


def test_minimum_weeks_floor_is_applied():
    f = SeveranceFormula(weeks_per_year=2.0, min_weeks=4.0, max_weeks=26.0)
    weeks, floor, cap = f.weeks_for(0.5)
    assert weeks == 4.0
    assert floor and not cap


def test_maximum_weeks_cap_is_applied():
    f = SeveranceFormula(weeks_per_year=2.0, min_weeks=4.0, max_weeks=26.0)
    weeks, floor, cap = f.weeks_for(30.0)
    assert weeks == 26.0
    assert cap and not floor


def test_per_level_rate_overrides_the_default():
    f = SeveranceFormula(weeks_per_year=2.0, max_weeks=100,
                         weeks_per_year_by_level={"M3": 4.0})
    assert f.weeks_for(5.0, "L3")[0] == 10.0
    assert f.weeks_for(5.0, "M3")[0] == 20.0


def test_partial_years_can_be_excluded():
    partial = SeveranceFormula(weeks_per_year=2.0, min_weeks=0, max_weeks=100,
                               credit_partial_years=True)
    whole = SeveranceFormula(weeks_per_year=2.0, min_weeks=0, max_weeks=100,
                             credit_partial_years=False)
    assert partial.weeks_for(7.9)[0] == pytest.approx(15.8, rel=1e-6)
    assert whole.weeks_for(7.9)[0] == 14.0


def test_base_weeks_are_added_regardless_of_tenure():
    f = SeveranceFormula(weeks_per_year=2.0, base_weeks=4.0, min_weeks=0, max_weeks=100)
    assert f.weeks_for(3.0)[0] == 10.0


def test_weeks_can_be_rounded_to_an_increment():
    f = SeveranceFormula(weeks_per_year=2.0, min_weeks=0, max_weeks=100,
                         round_weeks_to=1.0)
    assert f.weeks_for(3.6)[0] == 7.0


def test_severance_amount_is_weekly_rate_times_weeks():
    c = cfg(formula=SeveranceFormula(weeks_per_year=2.0, min_weeks=0, max_weeks=100))
    r = run(cut(employee(annual_placeholder=None, annualized_pay=104000.0,
                         tenure_years=5.0)), c)
    row = r.register.iloc[0]
    assert row["weekly_rate"] == pytest.approx(2000.0, rel=1e-6)
    assert row["severance_weeks"] == 10.0
    assert row["severance_gross"] == pytest.approx(20000.0, rel=1e-6)


def test_target_bonus_can_be_included_in_the_severance_base():
    c = cfg(formula=SeveranceFormula(weeks_per_year=1.0, min_weeks=0, max_weeks=100,
                                     include_target_bonus=True))
    r = run(cut(employee(annualized_pay=104000.0, tenure_years=1.0,
                         target_bonus_pct=0.10)), c)
    # Base becomes 114,400 -> weekly 2,200 -> one week.
    assert r.register.iloc[0]["severance_gross"] == pytest.approx(2200.0, rel=1e-6)


# --- leave policy ----------------------------------------------------------


def test_undeclared_leave_policy_refuses_to_compute():
    """Guessing costs money in one direction and wages in the other."""
    r = run(config=cfg(leave_policy=""))
    assert "LEAVE_POLICY_UNDECLARED" in codes(r)
    assert r.register.empty
    assert r.report.has_errors


def test_sick_leave_is_not_paid_out_under_a_separate_bank():
    r = run(cut(employee(accrued_sick_hours=48.0)), cfg(leave_policy="separate"))
    row = r.register.iloc[0]
    assert row["sick_payout"] == 0.0
    assert row["sick_hours_not_paid"] == 48.0
    assert "SICK_LEAVE_NOT_PAID_OUT" in codes(r)


def test_sick_leave_is_paid_out_under_a_combined_pto_bank():
    r = run(cut(employee(accrued_sick_hours=48.0)), cfg(leave_policy="combined"))
    row = r.register.iloc[0]
    assert row["sick_payout"] == pytest.approx(48.0 * 75.0, rel=1e-6)
    assert "COMBINED_PTO_BANK" in codes(r)


def test_combined_bank_warning_says_to_verify_the_plan_document():
    r = run(cut(employee(accrued_sick_hours=8.0)), cfg(leave_policy="combined"))
    msg = next(f.message for f in r.report.findings if f.code == "COMBINED_PTO_BANK")
    assert "plan document" in msg


def test_sick_leave_reinstatement_on_rehire_is_mentioned():
    r = run(cut(employee(accrued_sick_hours=24.0)), cfg(leave_policy="separate"))
    msg = next(f.message for f in r.report.findings
               if f.code == "SICK_LEAVE_NOT_PAID_OUT")
    assert "rehired" in msg


# --- vacation --------------------------------------------------------------


def test_vacation_is_paid_at_the_hourly_equivalent_rate():
    r = run(cut(employee(accrued_vacation_hours=100.0, hourly_equivalent_rate=50.0)))
    assert r.register.iloc[0]["vacation_payout"] == pytest.approx(5000.0, rel=1e-6)


def test_missing_vacation_balance_warns_that_blank_is_not_zero():
    r = run(cut(employee(accrued_vacation_hours=None)))
    assert "VACATION_BALANCE_MISSING" in codes(r)
    msg = next(f.message for f in r.report.findings
               if f.code == "VACATION_BALANCE_MISSING")
    assert "blank is not zero" in msg


def test_hourly_rate_is_derived_when_absent():
    r = run(cut(employee(hourly_equivalent_rate=None, annualized_pay=208000.0,
                         accrued_vacation_hours=10.0)))
    # 208,000 / (260 * 8) = 100/hr
    assert r.register.iloc[0]["vacation_payout"] == pytest.approx(1000.0, rel=1e-6)


# --- withholding -----------------------------------------------------------


def test_federal_supplemental_rate_is_twenty_two_percent():
    r = run(cut(employee(annualized_pay=52000.0, tenure_years=1.0,
                         accrued_vacation_hours=0.0)),
            cfg(formula=SeveranceFormula(weeks_per_year=1.0, min_weeks=1, max_weeks=100)))
    row = r.register.iloc[0]
    assert row["est_federal_withholding"] == pytest.approx(
        row["taxable_separation_pay"] * FEDERAL_SUPPLEMENTAL_RATE, rel=1e-4
    )


def test_california_severance_rate_defaults_to_six_point_six_not_the_bonus_rate():
    """EDD DE 44 sets 10.23% for bonuses and stock options; severance is other
    supplemental wages at 6.6%. Secondary sources routinely conflate them."""
    assert CA_SUPPLEMENTAL_SEVERANCE_RATE == 0.066
    assert CA_SUPPLEMENTAL_BONUS_RATE == 0.1023
    r = run(cut(employee(accrued_vacation_hours=0.0)))
    row = r.register.iloc[0]
    assert row["est_ca_withholding"] == pytest.approx(
        row["taxable_separation_pay"] * 0.066, rel=1e-4
    )


def test_supplemental_rate_is_configurable_for_payroll_disagreement():
    c = cfg(taxes=TaxAssumptions(ca_supplemental_rate=CA_SUPPLEMENTAL_BONUS_RATE))
    r = run(cut(employee(accrued_vacation_hours=0.0)), c)
    row = r.register.iloc[0]
    assert row["est_ca_withholding"] == pytest.approx(
        row["taxable_separation_pay"] * 0.1023, rel=1e-4
    )


def test_sdi_is_not_withheld_from_dismissal_severance():
    r = run()
    assert r.register.iloc[0]["est_sdi"] == 0.0
    assert "DISMISSAL_SEVERANCE_TREATMENT" in codes(r)


def test_sdi_applies_to_wages_in_lieu_of_notice():
    c = cfg(taxes=TaxAssumptions(is_wages_in_lieu_of_notice=True))
    r = run(config=c)
    row = r.register.iloc[0]
    assert row["est_sdi"] > 0
    assert "WAGES_IN_LIEU_OF_NOTICE_ELECTED" in codes(r)


def test_wages_in_lieu_warning_mentions_the_unemployment_consequence():
    c = cfg(taxes=TaxAssumptions(is_wages_in_lieu_of_notice=True))
    r = run(config=c)
    msg = next(f.message for f in r.report.findings
               if f.code == "WAGES_IN_LIEU_OF_NOTICE_ELECTED")
    assert "unemployment" in msg.lower()
    assert "1265" in msg


def test_social_security_cap_assumption_is_disclosed_for_high_earners():
    r = run(cut(employee(annualized_pay=400000.0)))
    assert "SS_CAP_ASSUMPTION" in codes(r)


def test_net_to_employee_is_taxable_less_withholding():
    r = run()
    row = r.register.iloc[0]
    assert row["est_net_to_employee"] == pytest.approx(
        row["taxable_separation_pay"] - row["est_total_withholding"], rel=1e-6
    )


# --- non-negotiables -------------------------------------------------------


def test_release_cannot_be_conditioned_on_earned_wages():
    r = run()
    assert "RELEASE_CANNOT_COVER_EARNED_WAGES" in codes(r)
    msg = next(f.message for f in r.report.findings
               if f.code == "RELEASE_CANNOT_COVER_EARNED_WAGES")
    assert "may not" in msg
    assert "206.5" in next(
        f.authority for f in r.report.findings
        if f.code == "RELEASE_CANNOT_COVER_EARNED_WAGES"
    )


def test_final_wages_are_not_fabricated():
    r = run()
    assert "FINAL_WAGES_NOT_COMPUTED" in codes(r)
    assert r.report.totals["final_wages"] is None
    assert "supplied by payroll" in r.report.to_markdown()


def test_markdown_states_vacation_is_paid_at_the_final_rate():
    md = run().report.to_markdown()
    assert "final rate of pay" in md
    assert "227.3" in md


def test_severance_paid_before_separation_is_flagged():
    c = cfg(severance_payment_date=SEP - dt.timedelta(days=5))
    r = run(config=c)
    assert "SEVERANCE_BEFORE_SEPARATION" in codes(r)


def test_warn_shortfall_offset_is_an_error_not_an_assumption():
    r = run(config=cfg(warn_shortfall_days=15))
    assert "WARN_SHORTFALL_OFFSET" in codes(r)
    msg = next(f.message for f in r.report.findings if f.code == "WARN_SHORTFALL_OFFSET")
    assert "does not automatically reduce" in msg


# --- overrides -------------------------------------------------------------


def test_override_replaces_the_formula_and_is_recorded_separately():
    c = cfg(week_overrides={"E1": 20.0},
            formula=SeveranceFormula(weeks_per_year=1.0, min_weeks=0, max_weeks=100))
    r = run(cut(employee(tenure_years=5.0)), c)
    row = r.register.iloc[0]
    assert row["formula_weeks"] == 5.0
    assert row["severance_weeks"] == 20.0
    assert bool(row["overridden"])
    assert "SEVERANCE_OVERRIDE" in codes(r)


def test_overrides_trigger_an_impact_review_reminder():
    c = cfg(week_overrides={"E1": 20.0})
    r = run(config=c)
    assert "OVERRIDES_NEED_IMPACT_REVIEW" in codes(r)
    msg = next(f.message for f in r.report.findings
               if f.code == "OVERRIDES_NEED_IMPACT_REVIEW")
    assert "adverse impact" in msg


def test_no_override_finding_when_the_formula_is_applied_uniformly():
    r = run()
    assert "SEVERANCE_OVERRIDE" not in codes(r)
    assert "OVERRIDES_NEED_IMPACT_REVIEW" not in codes(r)


# --- missing data ----------------------------------------------------------


def test_missing_pay_makes_the_employee_uncomputable_and_errors():
    r = run(cut(employee(annualized_pay=None)))
    assert "NO_PAY_DATA" in codes(r)
    assert r.register.iloc[0]["status"] == "uncomputable"
    assert "INCOMPLETE_REGISTER" in codes(r)
    assert r.report.has_errors


def test_missing_tenure_makes_the_employee_uncomputable():
    r = run(cut(employee(tenure_years=None)))
    assert "NO_TENURE_DATA" in codes(r)
    assert r.register.iloc[0]["status"] == "uncomputable"


def test_uncomputable_employees_are_excluded_from_totals_but_counted():
    r = run(cut(employee(employee_id="OK"),
                employee(employee_id="BAD", annualized_pay=None)))
    assert r.report.totals["uncomputable_employees"] == 1
    assert r.report.totals["severance_gross"] > 0


def test_empty_cut_list_is_handled():
    r = run(pd.DataFrame())
    assert "NO_EMPLOYEES" in codes(r)
    assert r.register.empty


# --- aggregates and outputs -------------------------------------------------


def test_totals_sum_the_register():
    r = run(cut(employee(employee_id="A"), employee(employee_id="B")))
    assert r.report.totals["severance_gross"] == pytest.approx(
        float(r.register["severance_gross"].sum()), rel=1e-6
    )
    assert r.report.employee_count == 2


def test_total_employer_cost_includes_every_component():
    r = run()
    row = r.register.iloc[0]
    expected = (
        row["severance_gross"] + row["vacation_payout"] + row["sick_payout"]
        + row["cobra_cost"] + row["admin_cost"] + row["employer_payroll_tax"]
    )
    assert row["total_employer_cost"] == pytest.approx(expected, rel=1e-6)


def test_cash_flow_places_vacation_on_the_separation_date():
    r = run()
    first = r.report.cash_flow[0]
    assert first["date"] == SEP.isoformat()
    assert "due at separation" in first["component"]


def test_salary_continuation_is_reflected_in_the_cash_flow():
    r = run(config=cfg(severance_schedule="continuation"))
    assert any("continuation" in c["component"] for c in r.report.cash_flow)


def test_median_weeks_is_reported():
    r = run(cut(employee(employee_id="A", tenure_years=5.0),
                employee(employee_id="B", tenure_years=10.0)))
    assert r.report.totals["median_weeks"] > 0


def test_write_produces_all_artifacts():
    r = run()
    with tempfile.TemporaryDirectory() as tmp:
        paths = r.write(tmp)
        assert set(paths) == {"report_md", "register", "report_json"}
        assert all(p.exists() for p in paths.values())


def test_report_json_round_trips():
    payload = json.loads(run().report.to_json())
    assert "totals" in payload and "assumptions" in payload
    assert payload["assumptions"]["formula"]["weeks_per_year"] == 2.0


def test_report_carries_privilege_header_and_estimate_disclaimer():
    md = run().report.to_markdown()
    assert "privileged" in md.lower()
    assert "budgeting estimates" in md.lower()
    assert "de 44" in md.lower()


def test_assumptions_are_recorded_for_the_audit_trail():
    c = cfg(formula=SeveranceFormula(weeks_per_year=3.0))
    r = run(config=c)
    assert r.report.assumptions["formula"]["weeks_per_year"] == 3.0
    assert r.report.assumptions["leave_policy"] == "separate"


if __name__ == "__main__":
    import sys
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
