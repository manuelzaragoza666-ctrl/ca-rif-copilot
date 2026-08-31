"""
Tests for selection_criteria.py

Run with pytest, or standalone:  python test_selection_criteria.py
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
        def __init__(self, expected: float, rel: float = 1e-6) -> None:
            self.expected, self.rel = expected, rel

        def __eq__(self, other: object) -> bool:
            return abs(float(other) - self.expected) <= max(
                self.rel * abs(self.expected), 1e-9
            )

        def __repr__(self) -> str:  # pragma: no cover
            return f"approx({self.expected})"

    @contextlib.contextmanager
    def _raises(exc):
        try:
            yield
        except exc:
            return
        raise AssertionError(f"expected {exc.__name__} to be raised")

    pytest = types.SimpleNamespace(approx=_Approx, raises=_raises)

from rif_copilot.selection_criteria import (
    CriterionSpec,
    DepartmentPlan,
    RifPlan,
    SelectionConfigError,
    SelectionEngine,
    format_cut_list,
    load_plan,
    score_performance,
    score_skills,
    split_items,
    DEFAULT_PERFORMANCE_SCALE,
)
from rif_copilot.workforce_data import Severity


# --- fixtures --------------------------------------------------------------


def make_roster(rows: list[dict]) -> pd.DataFrame:
    """Build a frame shaped like the Data Manager's standardized output."""
    defaults = {
        "employee_id": "E0", "department": "Engineering", "job_title": "Engineer",
        "job_level": "L3", "performance_rating": "Meets", "skills": "Python",
        "certifications": "", "annualized_pay": 100000.0, "tenure_years": 3.0,
        "is_active": True, "has_blocking_error": False, "leave_status": None,
        "union_flag": False, "visa_status": None, "gender": "female",
        "race_ethnicity": "asian", "age_years": 35.0, "age_40_plus": False,
        "worksite_name": "HQ",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def perf_criterion(weight: float = 1.0, **kw) -> CriterionSpec:
    return CriterionSpec(
        name="performance", weight=weight, kind="performance",
        source_column="performance_rating", scale=dict(DEFAULT_PERFORMANCE_SCALE), **kw
    )


def skills_criterion(items: tuple[str, ...], weight: float = 1.0, **kw) -> CriterionSpec:
    return CriterionSpec(
        name="critical_skills", weight=weight, kind="skills",
        source_column="skills", critical_items=items, **kw
    )


def simple_plan(target: float = 100000.0, **kw) -> RifPlan:
    dept = DepartmentPlan(
        department="Engineering", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department",),
    )
    return RifPlan(
        cost_savings_target=target, departments={"Engineering": dept},
        burden_multiplier=1.0, **kw
    )


def codes(result) -> set[str]:
    return {f.code for f in result.report.findings}


# --- guardrails ------------------------------------------------------------


def test_protected_field_as_criterion_is_rejected():
    for col in ("age_years", "gender", "race_ethnicity", "disability_status",
                "veteran_status", "birth_date", "age_40_plus"):
        with pytest.raises(SelectionConfigError):
            CriterionSpec(name="x", weight=1.0, kind="numeric", source_column=col)


def test_protected_field_in_comparison_group_is_rejected():
    with pytest.raises(SelectionConfigError):
        DepartmentPlan(
            department="Engineering", criteria=(perf_criterion(),),
            comparison_group=("department", "gender"),
        )


def test_individual_mode_requires_criteria():
    with pytest.raises(SelectionConfigError):
        DepartmentPlan(department="Engineering", mode="individual", criteria=())


def test_unknown_mode_is_rejected():
    with pytest.raises(SelectionConfigError):
        DepartmentPlan(department="X", mode="vibes", criteria=(perf_criterion(),))


def test_engine_reports_that_protected_fields_were_excluded():
    roster = make_roster([
        {"employee_id": f"E{i}", "performance_rating": r, "annualized_pay": 100000.0}
        for i, r in enumerate(["Exceeds", "Meets", "Below"])
    ])
    result = SelectionEngine(simple_plan()).run(roster)
    assert "PROTECTED_FIELDS_EXCLUDED" in codes(result)
    # No protected column leaks into the cut list.
    cut = format_cut_list(result.cut_list)
    assert not ({"gender", "race_ethnicity", "age_years", "age_40_plus"} & set(cut.columns))


# --- scoring primitives ----------------------------------------------------


def test_score_performance_maps_labels_and_flags_unknowns():
    assert score_performance("Exceeds", DEFAULT_PERFORMANCE_SCALE)[0] == 4.0
    assert score_performance("meets expectations", DEFAULT_PERFORMANCE_SCALE)[0] == 3.0
    assert score_performance("4", DEFAULT_PERFORMANCE_SCALE)[0] == 4.0
    val, why = score_performance("Sparkling", DEFAULT_PERFORMANCE_SCALE)
    assert val is None and "not in the configured rating scale" in why


def test_unrated_employee_is_missing_not_worst():
    for label in ("New", "Not Rated", "N/A", "pending"):
        val, why = score_performance(label, DEFAULT_PERFORMANCE_SCALE)
        assert val is None, f"{label!r} must not resolve to a numeric score"
        assert "not been rated" in why or "no performance rating" in why


def test_split_items_handles_delimiters_and_parentheticals():
    assert split_items("Python|SQL, Go; Rust") == ["python", "sql", "go", "rust"]
    assert split_items("Forklift Certification (exp. 2025-04-01)") == [
        "forklift certification"
    ]
    assert split_items(None) == []


def test_score_skills_is_coverage_of_critical_list():
    val, why, matched = score_skills("Python|Kubernetes", "", ("Python", "Kubernetes", "Go"))
    assert val == pytest.approx(2 / 3)
    assert set(matched) == {"python", "kubernetes"}
    val, why, _ = score_skills(None, None, ("Python",))
    assert val is None and why is not None


def test_certifications_count_toward_skills_coverage():
    val, _, matched = score_skills("Excel", "PMP", ("Excel", "PMP"))
    assert val == 1.0
    assert set(matched) == {"excel", "pmp"}


# --- ranking correctness ---------------------------------------------------


def test_lowest_scorer_is_selected_first():
    roster = make_roster([
        {"employee_id": "HIGH", "performance_rating": "Exceeds"},
        {"employee_id": "MID", "performance_rating": "Meets"},
        {"employee_id": "LOW", "performance_rating": "Below"},
    ])
    result = SelectionEngine(simple_plan(target=100000.0)).run(roster)
    assert result.cut_list["employee_id"].tolist() == ["LOW"]


def test_low_score_review_threshold_does_not_invert_the_ranking():
    """Regression: pulling low scorers out of the pool once caused the highest
    performer to be selected while the lowest was retained."""
    roster = make_roster([
        {"employee_id": "HIGH", "performance_rating": "Exceeds"},
        {"employee_id": "LOW", "performance_rating": "Does Not Meet"},
    ])
    plan = simple_plan(target=100000.0, manual_review_below=15)
    result = SelectionEngine(plan).run(roster)

    selected = set(result.cut_list["employee_id"])
    assert selected == {"LOW"}
    low_row = result.scores.set_index("employee_id").loc["LOW"]
    assert bool(low_row["requires_approval"]) is True
    assert "SCORE_BELOW_REVIEW_THRESHOLD" in codes(result)
    assert "RANK_ORDER_VIOLATION" not in codes(result)


def test_rank_order_violation_is_reported_when_a_cap_inverts_selection():
    """A savings cap that skips an expensive low scorer in favour of a cheaper
    high scorer contradicts the criteria and must be flagged as an error."""
    dept = DepartmentPlan(
        department="Engineering", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department",),
        max_savings_share=0.5,
    )
    plan = RifPlan(
        cost_savings_target=200000.0, departments={"Engineering": dept},
        burden_multiplier=1.0,
    )
    roster = make_roster([
        {"employee_id": "LOW_EXPENSIVE", "performance_rating": "Below",
         "annualized_pay": 180000.0},
        {"employee_id": "HIGH_CHEAP", "performance_rating": "Exceeds",
         "annualized_pay": 50000.0},
    ])
    result = SelectionEngine(plan).run(roster)
    if "HIGH_CHEAP" in set(result.cut_list["employee_id"]):
        assert "RANK_ORDER_VIOLATION" in codes(result)


def test_scores_are_normalized_within_comparison_group_not_across():
    """A weak performer in a strong group should not be dragged below a strong
    performer in a weak group when the groups are separate."""
    dept = DepartmentPlan(
        department="Engineering", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department", "job_level"),
    )
    plan = RifPlan(
        cost_savings_target=1.0, departments={"Engineering": dept},
        burden_multiplier=1.0,
    )
    roster = make_roster([
        {"employee_id": "L3_LOW", "job_level": "L3", "performance_rating": "Meets"},
        {"employee_id": "L3_HIGH", "job_level": "L3", "performance_rating": "Exceeds"},
        {"employee_id": "L5_LOW", "job_level": "L5", "performance_rating": "Below"},
        {"employee_id": "L5_HIGH", "job_level": "L5", "performance_rating": "Meets"},
    ])
    result = SelectionEngine(plan).run(roster)
    s = result.scores.set_index("employee_id")
    assert s.loc["L3_LOW", "comparison_group"] != s.loc["L5_LOW", "comparison_group"]
    # Bottom of each group normalizes to 0 regardless of absolute rating.
    assert float(s.loc["L3_LOW", "retention_score"]) == 0.0
    assert float(s.loc["L5_LOW", "retention_score"]) == 0.0


def test_weights_are_renormalized_and_both_criteria_contribute():
    dept = DepartmentPlan(
        department="Engineering", mode="individual",
        criteria=(perf_criterion(weight=3.0), skills_criterion(("Python", "Go"), weight=1.0)),
        comparison_group=("department",),
    )
    assert sum(c.weight for c in dept.normalized_criteria) == pytest.approx(1.0)
    plan = RifPlan(cost_savings_target=1.0, departments={"Engineering": dept},
                   burden_multiplier=1.0)
    roster = make_roster([
        {"employee_id": "A", "performance_rating": "Exceeds", "skills": "Python|Go"},
        {"employee_id": "B", "performance_rating": "Below", "skills": "Cobol"},
    ])
    result = SelectionEngine(plan).run(roster)
    s = result.scores.set_index("employee_id")
    assert float(s.loc["A", "retention_score"]) == 100.0
    assert float(s.loc["B", "retention_score"]) == 0.0
    assert "performance" in s.loc["A", "score_breakdown"]
    assert "critical_skills" in s.loc["A", "score_breakdown"]


# --- data gaps -------------------------------------------------------------


def test_missing_required_criterion_routes_to_review_not_a_zero_score():
    roster = make_roster([
        {"employee_id": "RATED", "performance_rating": "Meets"},
        {"employee_id": "UNRATED", "performance_rating": "New"},
        {"employee_id": "OTHER", "performance_rating": "Exceeds"},
    ])
    result = SelectionEngine(simple_plan(target=500000.0)).run(roster)
    s = result.scores.set_index("employee_id")
    assert s.loc["UNRATED", "selection_status"] == "excluded_insufficient_data"
    assert pd.isna(s.loc["UNRATED", "retention_score"])
    assert "UNRATED" not in set(result.cut_list["employee_id"])
    assert "UNRATED" in set(result.review_queue["employee_id"])
    assert "CRITERION_DATA_GAPS" in codes(result)


def test_employees_with_no_pay_cannot_be_selected_against_a_cost_target():
    roster = make_roster([
        {"employee_id": "A", "annualized_pay": None, "performance_rating": "Below"},
        {"employee_id": "B", "performance_rating": "Exceeds"},
    ])
    result = SelectionEngine(simple_plan()).run(roster)
    assert "NO_PAY_DATA" in codes(result)
    assert "A" not in set(result.scores["employee_id"])


def test_rows_with_ingestion_errors_are_excluded_and_reported():
    roster = make_roster([
        {"employee_id": "BAD", "has_blocking_error": True, "performance_rating": "Below"},
        {"employee_id": "OK", "performance_rating": "Meets"},
    ])
    result = SelectionEngine(simple_plan()).run(roster)
    assert "BLOCKING_DATA_ERRORS_EXCLUDED" in codes(result)
    assert "BAD" not in set(result.scores["employee_id"])


def test_already_separated_employees_are_excluded():
    roster = make_roster([
        {"employee_id": "GONE", "is_active": False, "performance_rating": "Below"},
        {"employee_id": "HERE", "performance_rating": "Meets"},
    ])
    result = SelectionEngine(simple_plan()).run(roster)
    assert "GONE" not in set(result.scores["employee_id"])


# --- degenerate comparisons ------------------------------------------------


def test_comparison_group_of_one_is_not_auto_selected():
    dept = DepartmentPlan(
        department="Engineering", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department", "job_level"),
    )
    plan = RifPlan(cost_savings_target=500000.0, departments={"Engineering": dept},
                   burden_multiplier=1.0)
    roster = make_roster([
        {"employee_id": "SOLO", "job_level": "L7", "performance_rating": "Below"},
        {"employee_id": "A", "job_level": "L3", "performance_rating": "Meets"},
        {"employee_id": "B", "job_level": "L3", "performance_rating": "Exceeds"},
    ])
    result = SelectionEngine(plan).run(roster)
    s = result.scores.set_index("employee_id")
    assert s.loc["SOLO", "selection_status"] == "manual_review"
    assert "SOLO" not in set(result.cut_list["employee_id"])
    assert "DEGENERATE_COMPARISON_GROUP" in codes(result)


def test_missing_comparison_column_is_reported_as_degraded():
    dept = DepartmentPlan(
        department="Engineering", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department", "job_level"),
    )
    plan = RifPlan(cost_savings_target=1.0, departments={"Engineering": dept},
                   burden_multiplier=1.0)
    roster = make_roster([
        {"employee_id": "A", "performance_rating": "Meets"},
        {"employee_id": "B", "performance_rating": "Below"},
    ]).drop(columns=["job_level"])
    result = SelectionEngine(plan).run(roster)
    assert "COMPARISON_GROUP_DEGRADED" in codes(result)


def test_single_criterion_selection_is_warned_about():
    roster = make_roster([
        {"employee_id": "A", "performance_rating": "Meets"},
        {"employee_id": "B", "performance_rating": "Below"},
    ])
    result = SelectionEngine(simple_plan()).run(roster)
    assert "SINGLE_CRITERION_SELECTION" in codes(result)


def test_tie_at_the_cut_boundary_is_an_error_not_an_arbitrary_pick():
    roster = make_roster([
        {"employee_id": "T1", "performance_rating": "Below", "annualized_pay": 100000.0},
        {"employee_id": "T2", "performance_rating": "Below", "annualized_pay": 100000.0},
        {"employee_id": "TOP", "performance_rating": "Exceeds", "annualized_pay": 100000.0},
    ])
    result = SelectionEngine(simple_plan(target=100000.0)).run(roster)
    assert "TIE_AT_CUT_BOUNDARY" in codes(result)


# --- position mode ---------------------------------------------------------


def test_position_mode_selects_all_incumbents_of_the_named_title():
    dept = DepartmentPlan(
        department="Marketing", mode="position",
        eliminate_positions=("Marketing Manager",),
    )
    plan = RifPlan(cost_savings_target=1.0, departments={"Marketing": dept},
                   burden_multiplier=1.0)
    roster = make_roster([
        {"employee_id": "M1", "department": "Marketing", "job_title": "Marketing Manager"},
        {"employee_id": "M2", "department": "Marketing", "job_title": "Marketing Manager"},
        {"employee_id": "M3", "department": "Marketing", "job_title": "Designer"},
    ])
    result = SelectionEngine(plan).run(roster)
    assert set(result.cut_list["employee_id"]) == {"M1", "M2"}
    s = result.scores.set_index("employee_id")
    assert "individual performance" in s.loc["M1", "rationale"]


def test_position_mode_does_not_score_on_performance():
    dept = DepartmentPlan(
        department="Marketing", mode="position",
        eliminate_positions=("Marketing Manager",),
    )
    plan = RifPlan(cost_savings_target=1.0, departments={"Marketing": dept},
                   burden_multiplier=1.0)
    roster = make_roster([
        {"employee_id": "STAR", "department": "Marketing",
         "job_title": "Marketing Manager", "performance_rating": "Exceeds"},
    ])
    result = SelectionEngine(plan).run(roster)
    assert pd.isna(result.scores.iloc[0]["retention_score"])
    assert "STAR" in set(result.cut_list["employee_id"])


def test_position_mode_without_designated_positions_is_an_error():
    dept = DepartmentPlan(department="Marketing", mode="position")
    plan = RifPlan(cost_savings_target=1.0, departments={"Marketing": dept})
    roster = make_roster([{"employee_id": "M1", "department": "Marketing"}])
    result = SelectionEngine(plan).run(roster)
    assert "NO_POSITIONS_DESIGNATED" in codes(result)
    assert result.cut_list.empty


def test_misspelled_designated_position_is_reported():
    dept = DepartmentPlan(
        department="Marketing", mode="position",
        eliminate_positions=("Markting Manger",),
    )
    plan = RifPlan(cost_savings_target=1.0, departments={"Marketing": dept})
    roster = make_roster([
        {"employee_id": "M1", "department": "Marketing", "job_title": "Marketing Manager"},
    ])
    result = SelectionEngine(plan).run(roster)
    assert "DESIGNATED_POSITION_NOT_FOUND" in codes(result)


# --- caps and protections --------------------------------------------------


def test_protected_positions_are_never_selected():
    dept = DepartmentPlan(
        department="Operations", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department",),
        protected_positions=("Warehouse Associate",),
    )
    plan = RifPlan(cost_savings_target=500000.0, departments={"Operations": dept},
                   burden_multiplier=1.0)
    roster = make_roster([
        {"employee_id": "WA", "department": "Operations",
         "job_title": "Warehouse Associate", "performance_rating": "Does Not Meet"},
        {"employee_id": "OPS", "department": "Operations",
         "job_title": "Operations Lead", "performance_rating": "Meets"},
    ])
    result = SelectionEngine(plan).run(roster)
    assert "WA" not in set(result.cut_list["employee_id"])
    s = result.scores.set_index("employee_id")
    assert s.loc["WA", "selection_status"] == "protected_position"


def test_department_headcount_cap_is_respected():
    dept = DepartmentPlan(
        department="Engineering", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department",),
        max_headcount=1,
    )
    plan = RifPlan(cost_savings_target=10_000_000.0,
                   departments={"Engineering": dept}, burden_multiplier=1.0)
    roster = make_roster([
        {"employee_id": f"E{i}", "performance_rating": r}
        for i, r in enumerate(["Below", "Meets", "Exceeds", "Does Not Meet"])
    ])
    result = SelectionEngine(plan).run(roster)
    assert len(result.cut_list) == 1


def test_department_without_a_plan_entry_inherits_the_default():
    default = DepartmentPlan(
        department="(default)", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department",),
    )
    plan = RifPlan(cost_savings_target=100000.0, departments={},
                   default_plan=default, burden_multiplier=1.0)
    roster = make_roster([
        {"employee_id": "A", "department": "Legal", "performance_rating": "Below"},
        {"employee_id": "B", "department": "Legal", "performance_rating": "Exceeds"},
    ])
    result = SelectionEngine(plan).run(roster)
    assert set(result.cut_list["employee_id"]) == {"A"}


def test_department_with_no_plan_and_no_default_is_excluded_with_an_error():
    plan = RifPlan(cost_savings_target=100000.0, departments={}, default_plan=None)
    roster = make_roster([{"employee_id": "A", "department": "Legal"}])
    result = SelectionEngine(plan).run(roster)
    assert "NO_PLAN_FOR_DEPARTMENT" in codes(result)
    assert result.cut_list.empty


# --- cost target -----------------------------------------------------------


def test_selection_stops_once_the_savings_target_is_met():
    plan = simple_plan(target=100000.0)
    roster = make_roster([
        {"employee_id": f"E{i}", "performance_rating": r, "annualized_pay": 100000.0}
        for i, r in enumerate(["Does Not Meet", "Below", "Meets", "Exceeds"])
    ])
    result = SelectionEngine(plan).run(roster)
    assert len(result.cut_list) == 1
    assert result.report.target_met


def test_burden_multiplier_inflates_the_cost_of_each_selection():
    dept = DepartmentPlan(
        department="Engineering", mode="individual",
        criteria=(perf_criterion(),), comparison_group=("department",),
    )
    plan = RifPlan(cost_savings_target=1.0, departments={"Engineering": dept},
                   burden_multiplier=1.5)
    roster = make_roster([
        {"employee_id": "A", "performance_rating": "Below", "annualized_pay": 100000.0},
        {"employee_id": "B", "performance_rating": "Exceeds", "annualized_pay": 100000.0},
    ])
    result = SelectionEngine(plan).run(roster)
    assert float(result.cut_list.iloc[0]["annual_cost"]) == 150000.0


def test_unmet_target_is_reported_rather_than_forced():
    plan = simple_plan(target=10_000_000.0)
    roster = make_roster([
        {"employee_id": "A", "performance_rating": "Below"},
        {"employee_id": "B", "performance_rating": "Meets"},
    ])
    result = SelectionEngine(plan).run(roster)
    assert "TARGET_NOT_MET" in codes(result)
    assert not result.report.target_met


def test_exhausting_most_of_the_pool_is_flagged():
    plan = simple_plan(target=10_000_000.0)
    roster = make_roster([
        {"employee_id": f"E{i}", "performance_rating": r}
        for i, r in enumerate(["Below", "Meets", "Exceeds", "Does Not Meet"])
    ])
    result = SelectionEngine(plan).run(roster)
    assert "POOL_LARGELY_EXHAUSTED" in codes(result)


# --- legal review annotations ---------------------------------------------


def test_leave_union_and_visa_are_flagged_without_affecting_the_score():
    roster = make_roster([
        {"employee_id": "LEAVE", "performance_rating": "Below", "leave_status": "CFRA"},
        {"employee_id": "UNION", "performance_rating": "Below", "union_flag": True},
        {"employee_id": "VISA", "performance_rating": "Below", "visa_status": "H-1B"},
        {"employee_id": "PLAIN", "performance_rating": "Exceeds"},
    ])
    result = SelectionEngine(simple_plan(target=10_000_000.0)).run(roster)
    s = result.scores.set_index("employee_id")
    # All three "Below" employees score identically: the flags did not move them.
    assert float(s.loc["LEAVE", "retention_score"]) == float(s.loc["PLAIN", "retention_score"]) - 100.0
    assert "ON_PROTECTED_LEAVE" in s.loc["LEAVE", "legal_review_flags"]
    assert "UNION_MEMBER" in s.loc["UNION", "legal_review_flags"]
    assert "WORK_VISA_HOLDER" in s.loc["VISA", "legal_review_flags"]
    assert "SELECTIONS_REQUIRE_LEGAL_REVIEW" in codes(result)


# --- outputs ---------------------------------------------------------------


def test_cut_list_carries_rationale_and_blank_override_columns():
    roster = make_roster([
        {"employee_id": "A", "performance_rating": "Below"},
        {"employee_id": "B", "performance_rating": "Exceeds"},
    ])
    result = SelectionEngine(simple_plan()).run(roster)
    cut = format_cut_list(result.cut_list)
    assert cut.loc[0, "rationale"]
    for col in ("human_decision", "decision_maker", "decision_date", "override_reason"):
        assert col in cut.columns
        assert cut[col].isna().all()


def test_result_writes_all_five_artifacts():
    roster = make_roster([
        {"employee_id": "A", "performance_rating": "Below"},
        {"employee_id": "B", "performance_rating": "Exceeds"},
    ])
    result = SelectionEngine(simple_plan()).run(roster)
    with tempfile.TemporaryDirectory() as tmp:
        paths = result.write(tmp)
        assert set(paths) == {"cut_list", "scores", "review_queue",
                              "report_json", "report_md"}
        assert all(p.exists() for p in paths.values())
        md = paths["report_md"].read_text()
        assert "recommendation, not a decision" in md


def test_report_json_round_trips():
    roster = make_roster([{"employee_id": "A", "performance_rating": "Below"},
                          {"employee_id": "B", "performance_rating": "Meets"}])
    result = SelectionEngine(simple_plan()).run(roster)
    payload = json.loads(result.report.to_json())
    assert "summary" in payload and "findings" in payload


# --- plan loading ----------------------------------------------------------


def test_load_plan_from_json_and_reject_protected_criteria():
    good = {
        "cost_savings_target": 100000,
        "departments": {
            "Engineering": {
                "mode": "individual",
                "criteria": {"performance": {"kind": "performance",
                                             "source_column": "performance_rating",
                                             "weight": 1.0}},
            }
        },
    }
    bad = {
        "cost_savings_target": 100000,
        "departments": {
            "Engineering": {
                "mode": "individual",
                "criteria": {"youth": {"kind": "numeric",
                                       "source_column": "age_years",
                                       "weight": 1.0,
                                       "higher_is_better": False}},
            }
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        gp = Path(tmp) / "good.json"
        gp.write_text(json.dumps(good))
        plan = load_plan(gp)
        assert plan.cost_savings_target == 100000
        assert "Engineering" in plan.departments

        bp = Path(tmp) / "bad.json"
        bp.write_text(json.dumps(bad))
        with pytest.raises(SelectionConfigError):
            load_plan(bp)


def test_plan_without_a_target_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "p.json"
        p.write_text(json.dumps({"departments": {}}))
        with pytest.raises(SelectionConfigError):
            load_plan(p)


def test_sample_yaml_plan_loads():
    path = Path(__file__).resolve().parent.parent / "examples" / "rif_plan.yaml"
    if not path.exists():
        return
    plan = load_plan(path)
    assert plan.cost_savings_target > 0
    assert plan.departments
    assert plan.departments["Marketing"].mode == "position"


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
