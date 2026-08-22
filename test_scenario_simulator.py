"""
Tests for scenario_simulator.py

Run with pytest, or standalone:  python test_scenario_simulator.py
"""

from __future__ import annotations

# Make the package importable whether this file is run directly, via pytest, or
# from another working directory.
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import json
import tempfile
from pathlib import Path

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

from rif_copilot.scenario_simulator import (
    CostAssumptions,
    Scenario,
    ScenarioSimulator,
    load_scenarios,
)
from rif_copilot.selection_criteria import SelectionConfigError, plan_from_dict
from rif_copilot.workforce_data import Severity


# --- fixtures --------------------------------------------------------------


def roster(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "employee_id": "E0", "department": "Engineering", "job_title": "Engineer",
        "job_level": "L3", "worksite_name": "HQ", "manager_id": "M1",
        "performance_rating": "Meets", "skills": "Python", "certifications": "",
        "annualized_pay": 100000.0, "hourly_equivalent_rate": 48.08,
        "accrued_vacation_hours": 40.0, "tenure_years": 3.0, "is_active": True,
        "has_blocking_error": False, "leave_status": None, "union_flag": False,
        "visa_status": None, "gender": "female", "race_ethnicity": "white",
        "age_40_plus": False, "disability_status": "no", "veteran_status": "no",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def cohort(n: int, prefix: str, **attrs) -> list[dict]:
    return [{"employee_id": f"{prefix}{i}", **attrs} for i in range(n)]


BASE_PLAN = {
    "cost_savings_target": 250000,
    "burden_multiplier": 1.0,
    "default": {
        "mode": "individual",
        "comparison_group": ["department"],
        "criteria": {
            "performance": {
                "kind": "performance",
                "source_column": "performance_rating",
                "weight": 1.0,
            }
        },
    },
}


def scenario(name: str, target: float = 250000, rationale: str = "Test scenario.",
             **overrides) -> Scenario:
    raw = dict(BASE_PLAN, cost_savings_target=target)
    raw.update(overrides)
    return Scenario(name=name, plan=plan_from_dict(raw, name), rationale=rationale)


def mixed_roster(n: int = 30) -> pd.DataFrame:
    ratings = ["Exceeds", "Meets", "Below", "Does Not Meet"]
    return roster([
        {"employee_id": f"E{i}", "performance_rating": ratings[i % 4],
         "annualized_pay": 100000.0 + (i % 5) * 10000}
        for i in range(n)
    ])


def codes(sim) -> set[str]:
    return {f["code"] for f in sim.report.findings}


# --- rationale is mandatory ------------------------------------------------


def test_scenario_without_rationale_is_rejected():
    with pytest.raises(SelectionConfigError):
        Scenario(name="A", plan=plan_from_dict(BASE_PLAN, "A"), rationale="")


def test_whitespace_only_rationale_is_rejected():
    with pytest.raises(SelectionConfigError):
        Scenario(name="A", plan=plan_from_dict(BASE_PLAN, "A"), rationale="   \n ")


def test_rationale_is_carried_into_the_comparison_and_report():
    sim = ScenarioSimulator().run(
        mixed_roster(), [scenario("A", rationale="Board-approved reduction.")]
    )
    assert sim.comparison.loc[0, "rationale"] == "Board-approved reduction."
    assert "Board-approved reduction." in sim.report.to_markdown()


# --- comparability guards --------------------------------------------------


def test_scenarios_with_different_as_of_dates_are_rejected():
    a = scenario("A")
    b = scenario("B")
    a.plan.as_of_date = pd.Timestamp("2026-10-30").date()
    b.plan.as_of_date = pd.Timestamp("2027-01-15").date()
    with pytest.raises(SelectionConfigError):
        ScenarioSimulator().run(mixed_roster(), [a, b])


def test_duplicate_scenario_names_are_rejected():
    with pytest.raises(SelectionConfigError):
        ScenarioSimulator().run(mixed_roster(), [scenario("A"), scenario("A")])


def test_no_scenarios_is_reported_not_crashed():
    sim = ScenarioSimulator().run(mixed_roster(), [])
    assert "NO_SCENARIOS" in codes(sim)
    assert sim.comparison.empty


# --- no composite ranking --------------------------------------------------


def test_no_composite_score_or_recommended_scenario_is_produced():
    sim = ScenarioSimulator().run(
        mixed_roster(), [scenario("A", 200000), scenario("B", 400000)]
    )
    cols = set(sim.comparison.columns)
    for forbidden in ("score", "rank", "recommended", "best", "overall_score"):
        assert not any(forbidden in c.lower() for c in cols), (
            f"comparison must not imply a ranking; found {forbidden!r}"
        )
    md = sim.report.to_markdown().lower()
    assert "there is no ranking here" in md


def test_report_explains_the_tuning_distinction():
    sim = ScenarioSimulator().run(mixed_roster(), [scenario("A")])
    md = sim.report.to_markdown().lower()
    assert "less discriminatory alternative" in md
    assert "discoverable" in md


# --- financial model -------------------------------------------------------


def test_severance_estimate_respects_floor_and_cap():
    a = CostAssumptions(
        severance_weeks_per_year=2.0, severance_min_weeks=4.0,
        severance_max_weeks=26.0, cobra_months=0, cobra_monthly_cost=0,
        admin_cost_per_employee=0, pay_out_accrued_vacation=False,
    )
    # 30 years of service would be 60 weeks; the cap holds it to 26.
    df = roster([
        {"employee_id": "OLD", "performance_rating": "Below", "tenure_years": 30.0,
         "annualized_pay": 104000.0},
        {"employee_id": "TOP", "performance_rating": "Exceeds", "tenure_years": 30.0},
    ])
    sim = ScenarioSimulator(assumptions=a).run(df, [scenario("A", 1.0)])
    fin = sim.outcomes[0].financial
    assert fin.headcount_reduction == 1
    assert fin.severance_cost == pytest.approx(2000.0 * 26, rel=1e-6)


def test_short_tenure_gets_the_minimum_severance_weeks():
    a = CostAssumptions(
        severance_weeks_per_year=2.0, severance_min_weeks=4.0,
        cobra_months=0, cobra_monthly_cost=0, admin_cost_per_employee=0,
        pay_out_accrued_vacation=False,
    )
    df = roster([
        {"employee_id": "NEW", "performance_rating": "Below", "tenure_years": 0.5,
         "annualized_pay": 52000.0},
        {"employee_id": "TOP", "performance_rating": "Exceeds"},
    ])
    sim = ScenarioSimulator(assumptions=a).run(df, [scenario("A", 1.0)])
    # 0.5 years * 2 = 1 week, floored to 4.
    assert sim.outcomes[0].financial.severance_cost == pytest.approx(1000.0 * 4, rel=1e-6)


def test_accrued_vacation_is_paid_out():
    a = CostAssumptions(
        severance_weeks_per_year=0, severance_min_weeks=0, cobra_months=0,
        cobra_monthly_cost=0, admin_cost_per_employee=0,
        pay_out_accrued_vacation=True,
    )
    df = roster([
        {"employee_id": "A", "performance_rating": "Below",
         "accrued_vacation_hours": 100.0, "hourly_equivalent_rate": 50.0},
        {"employee_id": "B", "performance_rating": "Exceeds"},
    ])
    sim = ScenarioSimulator(assumptions=a).run(df, [scenario("A", 1.0)])
    assert sim.outcomes[0].financial.vacation_payout == pytest.approx(5000.0, rel=1e-6)


def test_first_year_net_and_payback_are_derived_consistently():
    sim = ScenarioSimulator().run(mixed_roster(), [scenario("A", 200000)])
    fin = sim.outcomes[0].financial
    assert fin.first_year_net == pytest.approx(
        fin.annualized_savings - fin.one_time_cost, rel=1e-6
    )
    if fin.annualized_savings > 0:
        assert fin.payback_months == pytest.approx(
            round(fin.one_time_cost / (fin.annualized_savings / 12), 1), rel=1e-3
        )


def test_one_time_cost_is_labeled_provisional():
    sim = ScenarioSimulator().run(mixed_roster(), [scenario("A")])
    md = sim.report.to_markdown()
    assert "provisional estimate" in md.lower()
    assert "box 6" in md.lower()


def test_payback_is_none_when_nothing_is_saved():
    df = roster(cohort(4, "E", performance_rating="Meets"))
    sim = ScenarioSimulator().run(df, [scenario("A", 0.0)])
    assert sim.outcomes[0].financial.payback_months is None


# --- operational model -----------------------------------------------------


def test_department_reduction_is_reported_per_department():
    df = roster(
        cohort(10, "E", department="Engineering", performance_rating="Below")
        + cohort(10, "S", department="Sales", performance_rating="Exceeds")
    )
    sim = ScenarioSimulator().run(df, [scenario("A", 300000)])
    dept = sim.outcomes[0].operational.department_reduction
    assert dept["Engineering"]["before"] == 10
    assert dept["Engineering"]["after"] < 10


def test_deep_department_cut_is_warned_about():
    df = roster(
        cohort(10, "E", department="Engineering", performance_rating="Below")
        + cohort(30, "S", department="Sales", performance_rating="Exceeds")
    )
    sim = ScenarioSimulator().run(df, [scenario("A", 500000)])
    assert "DEEP_DEPARTMENT_CUT" in codes(sim)


def test_unresolvable_org_structure_reports_na_not_zero():
    """Regression: reporting '0 managers lost' when manager_id can't be
    resolved is a false all-clear."""
    df = roster(cohort(20, "E", manager_id="M999", performance_rating="Below"))
    sim = ScenarioSimulator().run(df, [scenario("A", 200000)])
    op = sim.outcomes[0].operational
    assert op.managers_lost is None
    assert op.orphaned_reports is None
    assert "ORG_STRUCTURE_UNRESOLVABLE" in codes(sim)
    assert "n/a" in sim.report.to_markdown()


def test_resolvable_org_structure_counts_managers_and_orphans():
    rows = (
        [{"employee_id": "MGR", "manager_id": "CEO", "performance_rating": "Below",
          "annualized_pay": 300000.0}]
        + cohort(6, "R", manager_id="MGR", performance_rating="Exceeds")
        + [{"employee_id": "CEO", "manager_id": "CEO", "performance_rating": "Exceeds"}]
    )
    sim = ScenarioSimulator().run(roster(rows), [scenario("A", 200000)])
    op = sim.outcomes[0].operational
    assert op.managers_lost == 1
    assert op.orphaned_reports == 6
    assert "ORPHANED_REPORTS" in codes(sim)


def test_eliminating_a_critical_skill_entirely_is_an_error():
    plan = {
        "cost_savings_target": 200000,
        "burden_multiplier": 1.0,
        "departments": {
            "Engineering": {
                "mode": "individual",
                "comparison_group": ["department"],
                "criteria": {
                    "performance": {
                        "kind": "performance",
                        "source_column": "performance_rating",
                        "weight": 1.0,
                    },
                    "critical_skills": {
                        "kind": "skills", "source_column": "skills", "weight": 0.0,
                        "required": False, "critical_items": ["Cobol"],
                    },
                },
            }
        },
    }
    df = roster(
        [{"employee_id": "COB", "skills": "Cobol", "performance_rating": "Does Not Meet"}]
        + cohort(9, "E", skills="Python", performance_rating="Exceeds")
    )
    sc = Scenario("A", plan_from_dict(plan, "A"), rationale="Test.")
    sim = ScenarioSimulator().run(df, [sc])
    gaps = sim.outcomes[0].operational.skill_gaps
    assert any(g["skill"] == "Cobol" for g in gaps)
    assert "CRITICAL_SKILL_ELIMINATED" in codes(sim)


def test_tenure_loss_is_measured():
    df = roster(
        [{"employee_id": "VET", "tenure_years": 20.0, "performance_rating": "Below"}]
        + cohort(9, "E", tenure_years=2.0, performance_rating="Exceeds")
    )
    sim = ScenarioSimulator().run(df, [scenario("A", 100000)])
    op = sim.outcomes[0].operational
    assert op.tenure_years_lost >= 20.0
    assert op.median_tenure_before is not None


def test_reduction_pct_is_computed():
    df = roster(cohort(20, "E", performance_rating="Below"))
    sim = ScenarioSimulator().run(df, [scenario("A", 300000)])
    op = sim.outcomes[0].operational
    expected = (op.headcount_before - op.headcount_after) / op.headcount_before
    assert op.reduction_pct == pytest.approx(expected, rel=1e-9)


# --- compliance integration ------------------------------------------------


def test_adverse_impact_is_run_per_scenario_and_surfaced():
    df = roster(
        cohort(20, "OLD", age_40_plus=True, performance_rating="Below")
        + cohort(20, "YNG", age_40_plus=False, performance_rating="Exceeds")
    )
    sim = ScenarioSimulator().run(df, [scenario("A", 500000)])
    o = sim.outcomes[0]
    assert o.impact.report.comparisons
    assert "Age 40+" in o.impact.report.class_verdicts()
    if o.impact_indicated:
        assert "ADVERSE_IMPACT_INDICATED" in codes(sim)


def test_impact_finding_says_financials_do_not_override_it():
    df = roster(
        cohort(20, "OLD", age_40_plus=True, performance_rating="Does Not Meet")
        + cohort(20, "YNG", age_40_plus=False, performance_rating="Exceeds")
    )
    sim = ScenarioSimulator().run(df, [scenario("A", 500000)])
    msgs = [f["message"] for f in sim.report.findings
            if f["code"] == "ADVERSE_IMPACT_INDICATED"]
    if msgs:
        assert "whatever its financials look like" in msgs[0]


def test_review_queue_and_legal_review_counts_appear_in_the_comparison():
    sim = ScenarioSimulator().run(mixed_roster(), [scenario("A")])
    for col in ("review_queue", "legal_review_selections", "impact_indicated"):
        assert col in sim.comparison.columns


def test_alternative_meeting_the_target_without_impact_is_pointed_out():
    df = roster(
        cohort(20, "OLD", age_40_plus=True, performance_rating="Does Not Meet",
               annualized_pay=100000.0)
        + cohort(20, "YNG", age_40_plus=False, performance_rating="Exceeds",
                 annualized_pay=100000.0)
    )
    sim = ScenarioSimulator().run(
        df, [scenario("Deep", 1500000), scenario("Shallow", 100000)]
    )
    if any(o.impact_indicated for o in sim.outcomes) and not all(
        o.impact_indicated for o in sim.outcomes
    ):
        assert "ALTERNATIVE_MEETS_TARGET" in codes(sim)


# --- the tuning guardrail --------------------------------------------------


def weighted_scenario(name: str, perf_w: float, skill_w: float) -> Scenario:
    plan = {
        "cost_savings_target": 400000,
        "burden_multiplier": 1.0,
        "departments": {
            "Engineering": {
                "mode": "individual",
                "comparison_group": ["department"],
                "criteria": {
                    "performance": {
                        "kind": "performance",
                        "source_column": "performance_rating",
                        "weight": perf_w,
                    },
                    "critical_skills": {
                        "kind": "skills", "source_column": "skills",
                        "weight": skill_w, "required": False,
                        "critical_items": ["Python", "Kubernetes"],
                    },
                },
            }
        },
    }
    return Scenario(name, plan_from_dict(plan, name), rationale="Weighting variant.")


def divergent_roster() -> pd.DataFrame:
    """A roster where performance and skills point at different people.

    Older employees rate poorly but hold the critical skills; younger employees
    rate well but do not. Weighting performance therefore cuts one group and
    weighting skills cuts the other, which is exactly the situation the tuning
    guardrail exists to notice.
    """
    rows = []
    for i in range(20):
        rows.append({
            "employee_id": f"O{i}", "age_40_plus": True,
            "performance_rating": "Does Not Meet" if i < 16 else "Meets",
            "skills": "Python|Kubernetes",
            "annualized_pay": 100000.0,
        })
    for i in range(20):
        rows.append({
            "employee_id": f"Y{i}", "age_40_plus": False,
            "performance_rating": "Exceeds",
            "skills": "Cobol" if i < 16 else "Python|Kubernetes",
            "annualized_pay": 100000.0,
        })
    return roster(rows)


def test_weight_only_variants_that_move_the_disparity_are_flagged():
    """Regression: measuring raw selection rates missed this — the rates barely
    moved while the disparity ratio nearly doubled."""
    sim = ScenarioSimulator().run(
        divergent_roster(),
        [weighted_scenario("Perf-heavy", 0.9, 0.1),
         weighted_scenario("Skills-heavy", 0.1, 0.9)],
    )
    # The two weightings must actually select different people, or there is
    # nothing for the guardrail to catch.
    cut_a = set(sim.by_name("Perf-heavy").selection.cut_list["employee_id"])
    cut_b = set(sim.by_name("Skills-heavy").selection.cut_list["employee_id"])
    assert cut_a != cut_b
    assert "WEIGHT_CHANGE_MOVES_DEMOGRAPHICS" in codes(sim)


def test_tuning_warning_explains_the_distinction_rather_than_accusing():
    sim = ScenarioSimulator().run(
        divergent_roster(),
        [weighted_scenario("Perf-heavy", 0.9, 0.1),
         weighted_scenario("Skills-heavy", 0.1, 0.9)],
    )
    msg = next((f["message"] for f in sim.report.findings
                if f["code"] == "WEIGHT_CHANGE_MOVES_DEMOGRAPHICS"), "")
    assert "not wrong on its own" in msg
    assert "recorded independently" in msg


def test_scenarios_with_different_structures_are_not_treated_as_tuning():
    """Different targets are a different question, not a weighting variant."""
    sim = ScenarioSimulator().run(
        mixed_roster(40), [scenario("A", 200000), scenario("B", 600000)]
    )
    assert "WEIGHT_CHANGE_MOVES_DEMOGRAPHICS" not in codes(sim)


# --- config loading --------------------------------------------------------


def test_load_scenarios_merges_a_base_plan_with_overrides():
    cfg = {
        "base": BASE_PLAN,
        "scenarios": [
            {"name": "A", "rationale": "Baseline.", "plan": {"cost_savings_target": 100000}},
            {"name": "B", "rationale": "Deeper.", "plan": {"cost_savings_target": 500000}},
        ],
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.json"
        p.write_text(json.dumps(cfg))
        scenarios = load_scenarios(p)
    assert [s.name for s in scenarios] == ["A", "B"]
    assert scenarios[0].plan.cost_savings_target == 100000
    assert scenarios[1].plan.cost_savings_target == 500000
    # The base criteria survived the merge.
    assert scenarios[0].plan.default_plan is not None


def test_load_scenarios_requires_a_rationale_per_scenario():
    cfg = {"base": BASE_PLAN, "scenarios": [{"name": "A", "plan": {}}]}
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.json"
        p.write_text(json.dumps(cfg))
        with pytest.raises(SelectionConfigError):
            load_scenarios(p)


def test_load_scenarios_rejects_a_file_without_a_scenarios_list():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.json"
        p.write_text(json.dumps({"base": BASE_PLAN}))
        with pytest.raises(SelectionConfigError):
            load_scenarios(p)


def test_sample_scenarios_yaml_loads():
    path = Path(__file__).resolve().parent.parent / "examples" / "scenarios.yaml"
    if not path.exists():
        return
    scenarios = load_scenarios(path)
    assert len(scenarios) >= 2
    assert all(s.rationale.strip() for s in scenarios)


# --- outputs ---------------------------------------------------------------


def test_write_produces_comparison_and_per_scenario_artifacts():
    sim = ScenarioSimulator().run(
        mixed_roster(), [scenario("A", 200000), scenario("B", 400000)]
    )
    with tempfile.TemporaryDirectory() as tmp:
        paths = sim.write(tmp)
        assert paths["report_md"].exists()
        assert paths["comparison"].exists()
        assert paths["report_json"].exists()
        for name in ("A", "B"):
            sub = paths[f"scenario:{name}"]
            assert (sub / "selection_recommended_cut_list.csv").exists()
            assert (sub / "adverse_impact_report.md").exists()


def test_report_json_round_trips():
    sim = ScenarioSimulator().run(mixed_roster(), [scenario("A")])
    payload = json.loads(sim.report.to_json())
    assert payload["scenarios"][0]["rationale"]
    assert "assumptions" in payload


def test_by_name_retrieves_a_scenario_outcome():
    sim = ScenarioSimulator().run(
        mixed_roster(), [scenario("A", 200000), scenario("B", 400000)]
    )
    assert sim.by_name("B") is not None
    assert sim.by_name("nope") is None


def test_report_carries_the_privilege_header():
    sim = ScenarioSimulator().run(mixed_roster(), [scenario("A")])
    md = sim.report.to_markdown()
    assert "privileged" in md.lower()
    assert "including those not pursued" in md.lower()


def test_assumptions_are_recorded_in_the_output():
    a = CostAssumptions(severance_weeks_per_year=3.0)
    sim = ScenarioSimulator(assumptions=a).run(mixed_roster(), [scenario("A")])
    assert sim.report.assumptions["severance_weeks_per_year"] == 3.0
    assert "severance_weeks_per_year" in sim.report.to_markdown()


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
