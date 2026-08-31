"""
Tests for ca_compliance.py

Run with pytest, or standalone:  python test_ca_compliance.py
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

from rif_copilot.ca_compliance import (
    CAL_WARN_ESTABLISHMENT_THRESHOLD,
    CAL_WARN_MASS_LAYOFF_THRESHOLD,
    CAL_WARN_NOTICE_DAYS,
    SB617_REQUIRED_DISCLOSURES,
    ComplianceConfig,
    ComplianceEngine,
)
from rif_copilot.workforce_data import Severity

SEP = dt.date(2026, 10, 30)


# --- fixtures --------------------------------------------------------------


def site(n: int, selected: int, name: str = "SF HQ", prefix: str = "E", **attrs):
    base = {
        "worksite_name": name, "department": "Engineering", "job_title": "Engineer",
        "tenure_years": 3.0, "annualized_pay": 120000.0,
        "hourly_equivalent_rate": 57.69, "accrued_vacation_hours": 80.0,
        "age_40_plus": False, "leave_status": None, "union_flag": False,
        "visa_status": None,
    }
    return [
        {**base, "employee_id": f"{prefix}{i}", "selected": i < selected, **attrs}
        for i in range(n)
    ]


def frame(*groups) -> pd.DataFrame:
    rows = []
    for g in groups:
        rows.extend(g)
    return pd.DataFrame(rows)


def cfg(**kw) -> ComplianceConfig:
    defaults = dict(
        proposed_separation_date=SEP,
        notice_date=SEP - dt.timedelta(days=90),
        total_company_headcount=400,
        service_coordination="lwdb",
        lwdb_email="board@example.gov",
        lwdb_phone="(555) 555-0100",
        employer_contact_email="hr@acme.com",
        employer_contact_phone="(555) 555-0199",
    )
    defaults.update(kw)
    return ComplianceConfig(**defaults)


def run(df, config=None, **kw):
    return ComplianceEngine(config or cfg()).run(df, **kw)


def codes(result) -> set[str]:
    return {f.code for f in result.report.findings}


def warn_for(result, jurisdiction: str, establishment: str | None = None):
    for w in result.report.warn:
        if w.jurisdiction == jurisdiction and (
            establishment is None or w.establishment == establishment
        ):
            return w
    return None


# --- Cal-WARN thresholds ---------------------------------------------------


def test_cal_warn_triggers_at_fifty_with_no_percentage_test():
    """California has no one-third test: 50 at a covered establishment is enough."""
    df = frame(site(200, 50))
    r = run(df)
    ca = warn_for(r, "California", "SF HQ")
    assert ca.triggered
    assert ca.threshold == CAL_WARN_MASS_LAYOFF_THRESHOLD
    # 50 of 200 is 25%, which would miss the federal one-third test.
    fed = warn_for(r, "Federal", "SF HQ")
    assert not fed.triggered


def test_cal_warn_does_not_trigger_below_fifty():
    r = run(frame(site(200, 49)))
    assert not warn_for(r, "California", "SF HQ").triggered


def test_establishment_below_seventy_five_is_not_covered():
    r = run(frame(site(74, 60)))
    ca = warn_for(r, "California", "SF HQ")
    assert not ca.covered_establishment
    assert not ca.triggered
    assert "75-person covered establishment" in ca.reason


def test_covered_establishment_threshold_is_seventy_five():
    r = run(frame(site(CAL_WARN_ESTABLISHMENT_THRESHOLD, 60)))
    assert warn_for(r, "California", "SF HQ").covered_establishment


def test_employees_under_six_months_are_excluded_from_the_threshold_count():
    """Cal-WARN counts only employees with 6 of the preceding 12 months."""
    df = frame(
        site(100, 45, prefix="LONG", tenure_years=3.0),
        site(20, 20, prefix="NEW", tenure_years=0.1),
    )
    r = run(df)
    ca = warn_for(r, "California", "SF HQ")
    # 45 long-tenured selected; the 20 new hires do not count toward 50.
    assert ca.affected_employees == 45
    assert not ca.triggered
    assert any("under 6 months" in n for n in ca.notes)


def test_short_service_employees_still_receive_other_obligations():
    df = frame(site(100, 60, tenure_years=0.1))
    r = run(df)
    ca = warn_for(r, "California", "SF HQ")
    assert ca.affected_employees == 0
    # Final pay and agency notices do not depend on the WARN count.
    obligation_codes = {o.code for o in r.report.obligations}
    assert "FINAL_PAY" in obligation_codes
    assert "EDD_CHANGE_NOTICE" in obligation_codes


def test_analysis_is_per_establishment_not_company_wide():
    """Two sites of 30 each do not combine into a 60-person mass layoff."""
    df = frame(site(100, 30, name="SF HQ", prefix="A"),
               site(100, 30, name="Sacramento", prefix="B"))
    r = run(df)
    assert not warn_for(r, "California", "SF HQ").triggered
    assert not warn_for(r, "California", "Sacramento").triggered


def test_termination_of_operations_triggers_with_no_minimum_count():
    r = run(frame(site(100, 5)), cfg(is_termination_of_operations=True))
    ca = warn_for(r, "California", "SF HQ")
    assert ca.triggered
    assert "termination" in ca.reason


def test_relocation_over_one_hundred_miles_triggers():
    r = run(frame(site(100, 5)),
            cfg(is_relocation=True, relocation_distance_miles=150))
    assert warn_for(r, "California", "SF HQ").triggered


def test_short_relocation_does_not_trigger():
    r = run(frame(site(100, 5)),
            cfg(is_relocation=True, relocation_distance_miles=20))
    assert not warn_for(r, "California", "SF HQ").triggered


# --- federal WARN ----------------------------------------------------------


def test_federal_mass_layoff_needs_one_third_below_five_hundred():
    r = run(frame(site(150, 60)))  # 40% of the site
    assert warn_for(r, "Federal", "SF HQ").triggered
    r2 = run(frame(site(400, 60)))  # 15% of the site
    assert not warn_for(r2, "Federal", "SF HQ").triggered


def test_federal_not_triggered_below_one_hundred_employee_employer():
    r = run(frame(site(90, 60)), cfg(total_company_headcount=90))
    fed = warn_for(r, "Federal", "SF HQ")
    assert not fed.triggered
    assert "100-employee threshold" in fed.reason


def test_missing_company_headcount_is_disclosed_as_a_limitation():
    r = run(frame(site(100, 60)), cfg(total_company_headcount=None))
    fed = warn_for(r, "Federal", "SF HQ")
    assert any("was not supplied" in n for n in fed.notes)


# --- aggregation -----------------------------------------------------------


def test_layoffs_aggregate_across_a_thirty_day_window():
    """Two rounds that each miss 50 can combine into a triggering event."""
    df = frame(site(200, 30))
    config = cfg(prior_layoffs={"SF HQ": [(dt.date(2026, 10, 15), 25)]})
    r = run(df, config)
    ca = warn_for(r, "California", "SF HQ")
    assert ca.triggered
    assert "aggregation" in ca.reason.lower()
    assert "WARN_TRIGGERED_BY_AGGREGATION" in codes(r)


def test_prior_layoffs_outside_the_window_do_not_aggregate():
    df = frame(site(200, 30))
    config = cfg(prior_layoffs={"SF HQ": [(dt.date(2026, 1, 1), 40)]})
    r = run(df, config)
    assert not warn_for(r, "California", "SF HQ").triggered


def test_near_threshold_is_flagged_without_advising_how_to_stay_under():
    r = run(frame(site(200, 47)))
    assert "NEAR_WARN_THRESHOLD" in codes(r)
    msg = next(f.message for f in r.report.findings if f.code == "NEAR_WARN_THRESHOLD")
    assert "aggregates" in msg
    assert "Treat this as triggered for planning purposes" in msg
    # The module must not coach anyone toward avoidance.
    for bad in ("stay under", "avoid the threshold", "keep it below",
                "reduce the count to"):
        assert bad not in msg.lower()


def test_no_output_field_advises_threshold_avoidance():
    r = run(frame(site(200, 47)))
    blob = (r.report.to_markdown() + r.report.to_json()).lower()
    for bad in ("stay under the threshold", "avoid triggering warn",
                "keep the layoff below"):
        assert bad not in blob


# --- notice dates ----------------------------------------------------------


def test_earliest_notice_date_is_sixty_days_before_separation():
    r = run(frame(site(200, 60)))
    ca = warn_for(r, "California", "SF HQ")
    assert ca.earliest_notice_date == SEP - dt.timedelta(days=CAL_WARN_NOTICE_DAYS)


def test_late_notice_is_an_error_and_names_the_earliest_lawful_separation_date():
    late = SEP - dt.timedelta(days=30)
    r = run(frame(site(200, 60)), cfg(notice_date=late))
    assert "WARN_NOTICE_DATE_PASSED" in codes(r)
    msg = next(f.message for f in r.report.findings
               if f.code == "WARN_NOTICE_DATE_PASSED")
    assert (late + dt.timedelta(days=CAL_WARN_NOTICE_DAYS)).isoformat() in msg
    assert not r.gate.may_generate_documents


def test_late_notice_message_states_california_exceptions_are_narrow():
    r = run(frame(site(200, 60)), cfg(notice_date=SEP - dt.timedelta(days=10)))
    msg = next(f.message for f in r.report.findings
               if f.code == "WARN_NOTICE_DATE_PASSED")
    assert "unforeseeable-business-circumstances" in msg
    assert "faltering-company" in msg


def test_separation_before_notice_is_an_error():
    r = run(frame(site(200, 60)), cfg(notice_date=SEP + dt.timedelta(days=5)))
    assert "SEPARATION_BEFORE_NOTICE" in codes(r)


def test_short_window_is_warned_even_when_warn_is_not_triggered():
    r = run(frame(site(200, 10)), cfg(notice_date=SEP - dt.timedelta(days=20)))
    assert "SHORT_NOTICE_WINDOW" in codes(r)


# --- SB 617 ----------------------------------------------------------------


def test_sb617_disclosures_become_obligations_when_warn_triggers():
    r = run(frame(site(200, 60)))
    obligation_codes = {o.code for o in r.report.obligations}
    for code, _ in SB617_REQUIRED_DISCLOSURES:
        assert f"SB617_{code.upper()}" in obligation_codes


def test_missing_coordination_election_blocks_document_generation():
    r = run(frame(site(200, 60)), cfg(service_coordination=""))
    assert "SB617_COORDINATION_UNDECLARED" in codes(r)
    assert not r.gate.may_generate_documents


def test_missing_lwdb_contact_blocks_document_generation():
    r = run(frame(site(200, 60)), cfg(lwdb_email="", lwdb_phone=""))
    assert "SB617_LWDB_CONTACT_MISSING" in codes(r)
    assert not r.gate.may_generate_documents


def test_missing_employer_contact_blocks_document_generation():
    r = run(frame(site(200, 60)),
            cfg(employer_contact_email="", employer_contact_phone=""))
    assert "SB617_EMPLOYER_CONTACT_MISSING" in codes(r)
    assert not r.gate.may_generate_documents


def test_electing_coordination_creates_a_thirty_day_arrangement_deadline():
    notice = SEP - dt.timedelta(days=90)
    r = run(frame(site(200, 60)), cfg(notice_date=notice, service_coordination="lwdb"))
    o = next(o for o in r.report.obligations if o.code == "SB617_COORDINATE_SERVICES")
    assert o.due_date == notice + dt.timedelta(days=30)


def test_declining_coordination_creates_no_arrangement_deadline():
    r = run(frame(site(200, 60)), cfg(service_coordination="none"))
    assert not any(o.code == "SB617_COORDINATE_SERVICES" for o in r.report.obligations)


def test_report_states_that_timely_notice_does_not_cure_deficient_content():
    r = run(frame(site(200, 60)))
    assert "SB617_CONTENT_IS_INDEPENDENT" in codes(r)
    md = r.report.to_markdown()
    assert "timely but omits" in md.lower()


def test_sb617_section_absent_when_warn_not_triggered():
    r = run(frame(site(200, 10)))
    assert "SB 617" not in r.report.to_markdown()


# --- final pay -------------------------------------------------------------


def test_vacation_payout_is_computed_at_the_hourly_equivalent_rate():
    df = frame(site(100, 1, accrued_vacation_hours=100.0, hourly_equivalent_rate=50.0))
    r = run(df)
    assert r.report.final_pay["vacation_payout"] == pytest.approx(5000.0, rel=1e-6)


def test_wages_through_separation_are_not_invented():
    """Regression: an arbitrary five-day multiplier produced a figure that
    looked authoritative and was not."""
    r = run(frame(site(100, 10)))
    assert r.report.final_pay["wages_through_separation"] is None
    assert "not computed" in r.report.to_markdown()


def test_waiting_time_exposure_is_thirty_days_of_daily_wages():
    df = frame(site(100, 1, annualized_pay=260000.0))  # $1,000/working day
    r = run(df)
    assert r.report.final_pay["waiting_time_exposure"] == pytest.approx(30000.0, rel=1e-3)


def test_final_pay_is_due_on_the_separation_date():
    r = run(frame(site(100, 10)))
    o = next(o for o in r.report.obligations if o.code == "FINAL_PAY")
    assert o.due_date == SEP
    assert o.severity == Severity.ERROR


def test_missing_pay_data_blocks_document_generation():
    df = frame(site(100, 10, annualized_pay=None))
    r = run(df)
    assert "FINAL_PAY_UNCOMPUTABLE" in codes(r)
    assert not r.gate.may_generate_documents


def test_missing_vacation_balance_is_warned_not_assumed_zero():
    df = frame(site(100, 10, accrued_vacation_hours=None))
    r = run(df)
    assert "VACATION_BALANCE_MISSING" in codes(r)
    msg = next(f.message for f in r.report.findings
               if f.code == "VACATION_BALANCE_MISSING")
    assert "blank is not the same as zero" in msg


# --- OWBPA -----------------------------------------------------------------


def test_group_termination_program_requires_forty_five_days():
    df = frame(site(100, 10, age_40_plus=True))
    r = run(df, cfg(is_group_termination_program=True))
    o = next(o for o in r.report.obligations if o.code == "OWBPA_DELIVER_AGREEMENT")
    assert o.due_date == SEP - dt.timedelta(days=45)


def test_individual_agreement_requires_twenty_one_days():
    df = frame(site(100, 10, age_40_plus=True))
    r = run(df, cfg(is_group_termination_program=False))
    o = next(o for o in r.report.obligations if o.code == "OWBPA_DELIVER_AGREEMENT")
    assert o.due_date == SEP - dt.timedelta(days=21)


def test_decisional_unit_disclosure_required_for_group_programs():
    df = frame(site(100, 10, age_40_plus=True))
    r = run(df)
    assert any(o.code == "OWBPA_DISCLOSURE" for o in r.report.obligations)
    assert "OWBPA_DECISIONAL_UNIT_REQUIRED" in codes(r)


def test_decisional_unit_is_not_assumed_to_be_the_scoring_comparison_group():
    df = frame(site(100, 10, age_40_plus=True))
    r = run(df)
    msg = next(f.message for f in r.report.findings
               if f.code == "OWBPA_DECISIONAL_UNIT_REQUIRED")
    assert "not automatically the correct decisional unit" in msg


def test_revocation_period_is_seven_days_and_cannot_be_waived():
    df = frame(site(100, 10, age_40_plus=True))
    r = run(df)
    o = next(o for o in r.report.obligations if o.code == "OWBPA_REVOCATION")
    assert o.due_date == SEP + dt.timedelta(days=7)
    assert "cannot be waived" in o.description


def test_unknown_age_is_flagged_as_owbpa_risk():
    df = frame(site(100, 10, age_40_plus=None))
    r = run(df)
    assert "AGE_UNKNOWN_FOR_OWBPA" in codes(r)


def test_no_owbpa_analysis_when_no_release_is_offered():
    df = frame(site(100, 10, age_40_plus=True))
    r = run(df, cfg(offering_severance_agreement=False))
    assert "NO_RELEASE_OFFERED" in codes(r)
    assert not any(o.code.startswith("OWBPA") for o in r.report.obligations)


# --- agency and benefit notices --------------------------------------------


def test_standard_separation_notices_are_always_generated():
    r = run(frame(site(100, 10)))
    obligation_codes = {o.code for o in r.report.obligations}
    for code in ("EDD_CHANGE_NOTICE", "DE2320_PAMPHLET", "COBRA_NOTICE", "HIPP_NOTICE"):
        assert code in obligation_codes


def test_cobra_notice_is_due_forty_four_days_after_separation():
    r = run(frame(site(100, 10)))
    o = next(o for o in r.report.obligations if o.code == "COBRA_NOTICE")
    assert o.due_date == SEP + dt.timedelta(days=44)


# --- individual conditions --------------------------------------------------


def test_protected_leave_blocks_and_is_listed_per_employee():
    df = frame(site(100, 5, leave_status="CFRA"), site(50, 0, prefix="X"))
    r = run(df)
    assert "SELECTED_ON_PROTECTED_LEAVE" in codes(r)
    assert not r.gate.may_generate_documents
    assert (r.employee_flags["flags"].str.contains("PROTECTED_LEAVE")).any()


def test_union_membership_blocks_pending_cba_review():
    df = frame(site(100, 5, union_flag=True), site(50, 0, prefix="X"))
    r = run(df)
    assert "SELECTED_UNION_MEMBERS" in codes(r)
    msg = next(f.message for f in r.report.findings if f.code == "SELECTED_UNION_MEMBERS")
    assert "override the selection criteria" in msg


def test_visa_holders_are_flagged_for_immigration_counsel():
    df = frame(site(100, 5, visa_status="H-1B"), site(50, 0, prefix="X"))
    r = run(df)
    assert "SELECTED_VISA_HOLDERS" in codes(r)


def test_clean_roster_produces_no_individual_flags():
    r = run(frame(site(100, 10)))
    assert r.employee_flags.empty or not len(r.employee_flags)


# --- the gate --------------------------------------------------------------


def test_gate_is_clear_on_a_compliant_scenario():
    class _Rep:
        indicated: list = []
        flagged: list = []

    class _Impact:
        report = _Rep()

    r = run(frame(site(200, 10)), impact=_Impact())
    assert r.gate.may_generate_documents
    assert not r.gate.blockers


def test_gate_blocks_on_indicated_adverse_impact():
    class _Comp:
        protected_class = "Age 40+"

    class _Rep:
        indicated = [_Comp()]
        flagged = [_Comp()]

    class _Impact:
        report = _Rep()

    r = run(frame(site(200, 10)), impact=_Impact())
    assert not r.gate.may_generate_documents
    assert any("Age 40+" in b for b in r.gate.blockers)


def test_missing_impact_analysis_is_a_gate_warning():
    r = run(frame(site(200, 10)))
    assert any("box 4" in w for w in r.gate.warnings)


def test_gate_blocks_on_unresolved_selection_errors():
    class _F:
        severity = Severity.ERROR
        message = "Tie at the cut boundary."

    class _Rep:
        findings = [_F()]

    class _Sel:
        report = _Rep()
        review_queue = pd.DataFrame()

    r = run(frame(site(200, 10)), selection=_Sel())
    assert not r.gate.may_generate_documents


def test_missing_establishment_column_blocks_rather_than_guesses():
    df = frame(site(200, 60)).drop(columns=["worksite_name"])
    r = run(df)
    assert "NO_ESTABLISHMENT_COLUMN" in codes(r)
    assert not r.gate.may_generate_documents


def test_empty_selection_is_handled():
    r = run(pd.DataFrame())
    assert "NO_SELECTION" in codes(r)
    assert not r.gate.may_generate_documents


def test_nobody_selected_produces_no_obligations():
    r = run(frame(site(100, 0)))
    assert "NO_AFFECTED_EMPLOYEES" in codes(r)


# --- outputs ---------------------------------------------------------------


def test_calendar_is_sorted_by_due_date():
    r = run(frame(site(200, 60, age_40_plus=True)))
    cal = r.calendar
    assert not cal.empty
    assert list(cal["due_date"]) == sorted(cal["due_date"])


def test_report_carries_privilege_header_and_not_legal_advice_disclaimer():
    r = run(frame(site(200, 60)))
    md = r.report.to_markdown()
    assert "privileged" in md.lower()
    assert "verify current requirements with employment" in md.lower()
    assert "does not advise on structuring" in md.lower()


def test_write_produces_all_artifacts():
    r = run(frame(site(200, 60, age_40_plus=True)))
    with tempfile.TemporaryDirectory() as tmp:
        paths = r.write(tmp)
        assert set(paths) == {"report_md", "calendar", "obligations",
                              "employee_flags", "report_json"}
        assert all(p.exists() for p in paths.values())


def test_report_json_round_trips():
    r = run(frame(site(200, 60)))
    payload = json.loads(r.report.to_json())
    assert "warn" in payload and "gate" in payload
    assert payload["summary"]["warn_triggered"] is True


def test_every_obligation_cites_an_authority():
    r = run(frame(site(200, 60, age_40_plus=True)))
    for o in r.report.obligations:
        assert o.authority.strip(), f"{o.code} has no cited authority"


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
