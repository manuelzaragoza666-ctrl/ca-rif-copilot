"""
Tests for adverse_impact.py

Run with pytest, or standalone:  python test_adverse_impact.py
"""

from __future__ import annotations

# Make the package importable whether this file is run directly, via pytest, or
# from another working directory.
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import math
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

from rif_copilot.adverse_impact import (
    FOUR_FIFTHS,
    SELECTION_RATIO_THRESHOLD,
    AdverseImpactAnalyzer,
    ProtectedClass,
    fisher_exact_two_tailed,
    standard_deviations,
)
from rif_copilot.workforce_data import Severity


# --- helpers ---------------------------------------------------------------


def roster(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "employee_id": "E0", "department": "Engineering", "job_level": "L3",
        "worksite_name": "HQ", "selected": False, "selection_status": "eligible",
        "gender": "female", "race_ethnicity": "white", "age_40_plus": False,
        "disability_status": "no", "veteran_status": "no",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def cohort(n: int, selected: int, prefix: str, **attrs) -> list[dict]:
    """n employees sharing attributes, `selected` of whom are on the cut list."""
    return [
        {"employee_id": f"{prefix}{i}", "selected": i < selected, **attrs}
        for i in range(n)
    ]


def codes(analysis) -> set[str]:
    return {f.code for f in analysis.report.findings}


def find(analysis, cls: str, unit: str = "All"):
    for c in analysis.report.comparisons:
        if c.protected_class == cls and c.unit == unit:
            return c
    return None


# --- statistics ------------------------------------------------------------


def test_fisher_matches_known_values():
    # Classic tea-tasting table.
    assert fisher_exact_two_tailed(3, 1, 1, 3) == pytest.approx(0.4857142857, rel=1e-6)
    # Independent table -> p near 1.
    assert fisher_exact_two_tailed(10, 10, 10, 10) == pytest.approx(1.0, rel=1e-9)
    # Complete separation -> very small p.
    assert fisher_exact_two_tailed(20, 0, 0, 20) < 1e-9


def test_fisher_is_symmetric_and_bounded():
    for a, b, c, d in [(5, 3, 2, 9), (12, 4, 7, 15), (1, 30, 20, 2)]:
        p = fisher_exact_two_tailed(a, b, c, d)
        assert 0.0 <= p <= 1.0
        assert p == pytest.approx(fisher_exact_two_tailed(c, d, a, b), rel=1e-9)


def test_fisher_handles_degenerate_tables():
    assert fisher_exact_two_tailed(0, 0, 0, 0) == 1.0
    assert fisher_exact_two_tailed(0, 10, 0, 10) == 1.0


def test_standard_deviations_sign_and_magnitude():
    # Group selected far more than expected -> large positive.
    sd = standard_deviations(18, 20, 2, 20)
    assert sd is not None and sd > 3
    # Group selected far less -> large negative.
    sd = standard_deviations(2, 20, 18, 20)
    assert sd is not None and sd < -3
    # Identical rates -> approximately zero.
    sd = standard_deviations(5, 20, 5, 20)
    assert sd is not None and abs(sd) < 0.5


def test_standard_deviations_returns_none_when_undefined():
    assert standard_deviations(0, 0, 0, 0) is None
    assert standard_deviations(0, 10, 0, 10) is None  # nobody selected


# --- core detection --------------------------------------------------------


def test_clear_disparity_is_flagged_as_impact_indicated():
    df = roster(
        cohort(40, 16, "O", age_40_plus=True)      # 40% selected
        + cohort(40, 2, "Y", age_40_plus=False)    # 5% selected
    )
    analysis = AdverseImpactAnalyzer().run(df)
    c = find(analysis, "Age 40+")
    assert c is not None
    assert c.impact_ratio < FOUR_FIFTHS
    assert c.verdict == "Impact indicated"
    assert "ADVERSE_IMPACT_INDICATED" in codes(analysis)


def test_even_selection_produces_no_flag():
    df = roster(
        cohort(40, 4, "O", age_40_plus=True)
        + cohort(40, 4, "Y", age_40_plus=False)
    )
    analysis = AdverseImpactAnalyzer().run(df)
    c = find(analysis, "Age 40+")
    assert c.verdict == "No flag"
    assert "ADVERSE_IMPACT_INDICATED" not in codes(analysis)


def test_reference_group_is_the_most_favored_group():
    df = roster(
        cohort(20, 8, "A", race_ethnicity="asian")     # 40%
        + cohort(20, 4, "B", race_ethnicity="white")   # 20%
        + cohort(20, 1, "C", race_ethnicity="hispanic_latino")  # 5%
    )
    analysis = AdverseImpactAnalyzer().run(df)
    comps = [c for c in analysis.report.comparisons
             if c.protected_class == "Race / Ethnicity" and c.unit_type == "overall"]
    assert comps
    assert all(c.reference_group == "hispanic_latino" for c in comps)


def test_impact_ratio_uses_retention_basis():
    df = roster(
        cohort(50, 25, "A", age_40_plus=True)     # 50% selected, 50% retained
        + cohort(50, 0, "B", age_40_plus=False)   # 0% selected, 100% retained
    )
    c = find(AdverseImpactAnalyzer().run(df), "Age 40+")
    assert c.impact_ratio == pytest.approx(0.5, rel=1e-6)


def test_selection_rate_ratio_is_reported_on_the_termination_basis():
    df = roster(
        cohort(50, 10, "A", age_40_plus=True)     # 20%
        + cohort(50, 5, "B", age_40_plus=False)   # 10%
    )
    c = find(AdverseImpactAnalyzer().run(df), "Age 40+")
    assert c.selection_rate_ratio == pytest.approx(2.0, rel=1e-6)


# --- the four-fifths blind spot -------------------------------------------


def test_four_fifths_passes_but_high_termination_ratio_is_still_flagged():
    """Regression: with low overall selection rates the retention-basis ratio
    compresses toward 1.0 and passes disparities that matter."""
    df = roster(
        cohort(100, 17, "A", age_40_plus=True)    # 17% selected
        + cohort(100, 8, "B", age_40_plus=False)  # 8% selected
    )
    c = find(AdverseImpactAnalyzer().run(df), "Age 40+")
    assert c.impact_ratio > FOUR_FIFTHS, "four-fifths rule should pass here"
    assert c.selection_rate_ratio >= SELECTION_RATIO_THRESHOLD
    assert c.diverges
    assert c.verdict != "No flag"


def test_divergence_between_the_two_screens_is_explained_in_the_report():
    df = roster(
        cohort(100, 17, "A", age_40_plus=True)
        + cohort(100, 8, "B", age_40_plus=False)
    )
    analysis = AdverseImpactAnalyzer().run(df)
    assert "FOUR_FIFTHS_UNDERSTATES_DISPARITY" in codes(analysis)


# --- Simpson's paradox -----------------------------------------------------

def test_unit_level_impact_is_caught_when_the_aggregate_looks_clean():
    """The company-wide ratio passes while one department fails."""
    rows = (
        # Engineering: heavy disparity.
        cohort(20, 8, "EO", department="Engineering", age_40_plus=True)
        + cohort(20, 0, "EY", department="Engineering", age_40_plus=False)
        # Sales: reversed, which pulls the aggregate back to even.
        + cohort(40, 2, "SO", department="Sales", age_40_plus=True)
        + cohort(40, 10, "SY", department="Sales", age_40_plus=False)
    )
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    overall = find(analysis, "Age 40+", "All")
    eng = find(analysis, "Age 40+", "Engineering")
    assert overall.verdict == "No flag"
    assert eng.verdict == "Impact indicated"
    assert "AGGREGATE_MASKS_UNIT_IMPACT" in codes(analysis)


def test_analysis_runs_across_department_worksite_and_level():
    rows = (
        cohort(20, 8, "A", department="Engineering", worksite_name="SF",
               job_level="L3", age_40_plus=True)
        + cohort(20, 1, "B", department="Engineering", worksite_name="SF",
                 job_level="L3", age_40_plus=False)
    )
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    unit_types = {c.unit_type for c in analysis.report.comparisons}
    assert {"overall", "department", "worksite", "job_level"} <= unit_types


# --- small samples ---------------------------------------------------------


def test_tiny_groups_are_skipped_not_flagged():
    rows = (
        cohort(30, 3, "BIG", gender="female")
        + cohort(3, 2, "TINY", gender="non_binary")
    )
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    tiny = [c for c in analysis.report.comparisons if c.group == "non_binary"]
    assert not tiny, "a 3-person group must not produce a rate comparison"
    assert "COMPARISONS_TOO_SMALL_TO_TEST" in codes(analysis)


def test_skipped_comparisons_are_reported_as_untested_not_cleared():
    rows = cohort(30, 3, "BIG", gender="female") + cohort(3, 2, "T", gender="male")
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    msg = next(f.message for f in analysis.report.findings
               if f.code == "COMPARISONS_TOO_SMALL_TO_TEST")
    assert "not cleared" in msg
    assert analysis.report.skipped


def test_small_group_ratio_does_not_count_as_practical_significance_alone():
    """A group of 6 with a bad ratio but no statistical signal is Review at
    most, never 'Impact indicated'."""
    rows = (
        cohort(60, 6, "BIG", gender="female")
        + cohort(6, 2, "SM", gender="male")
    )
    analysis = AdverseImpactAnalyzer(min_group_size=10).run(roster(rows))
    small = [c for c in analysis.report.comparisons if c.group == "male"]
    for c in small:
        assert not c.interpretable
        assert c.verdict != "Impact indicated"


def test_no_reference_group_of_adequate_size_means_the_unit_is_skipped():
    rows = cohort(6, 1, "A", gender="female") + cohort(6, 3, "B", gender="male")
    analysis = AdverseImpactAnalyzer(min_group_size=10).run(roster(rows))
    assert not [c for c in analysis.report.comparisons
                if c.protected_class == "Sex"]


# --- fragility -------------------------------------------------------------


def test_flip_count_measures_how_many_decisions_would_change_the_finding():
    df = roster(
        cohort(20, 10, "A", age_40_plus=True)
        + cohort(20, 0, "B", age_40_plus=False)
    )
    c = find(AdverseImpactAnalyzer().run(df), "Age 40+")
    # Retention 10/20 = 0.50 vs 1.00; need group retention >= 0.80, i.e. at
    # most 4 selected, so 6 selections would have to change.
    assert c.flip_count == 6


def test_flip_count_is_zero_when_the_rule_already_passes():
    df = roster(
        cohort(20, 2, "A", age_40_plus=True)
        + cohort(20, 2, "B", age_40_plus=False)
    )
    c = find(AdverseImpactAnalyzer().run(df), "Age 40+")
    assert c.flip_count == 0


def test_fragile_findings_are_called_out():
    df = roster(
        cohort(10, 3, "A", age_40_plus=True)
        + cohort(30, 0, "B", age_40_plus=False)
    )
    analysis = AdverseImpactAnalyzer().run(df)
    c = find(analysis, "Age 40+")
    if c and c.impact_ratio < FOUR_FIFTHS and c.flip_count is not None and c.flip_count <= 1:
        assert "FRAGILE" in c.flags


# --- data coverage ---------------------------------------------------------


def test_undisclosed_values_are_excluded_from_the_test_not_grouped():
    rows = (
        cohort(30, 3, "A", gender="female")
        + cohort(30, 3, "B", gender="male")
        + cohort(20, 10, "U", gender="not_disclosed")
    )
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    groups = {c.group for c in analysis.report.comparisons
              if c.protected_class == "Sex"}
    assert "not_disclosed" not in groups
    assert analysis.report.coverage["Sex"]["undisclosed"] == 20


def test_heavy_undisclosure_weakens_the_conclusion_and_says_so():
    rows = (
        cohort(20, 2, "A", gender="female")
        + cohort(20, 2, "B", gender="male")
        + cohort(40, 4, "U", gender="not_disclosed")
    )
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    assert "LOW_CLASS_COVERAGE" in codes(analysis)


def test_absent_class_column_is_reported_as_untested():
    df = roster(cohort(30, 3, "A") + cohort(30, 3, "B")).drop(columns=["veteran_status"])
    analysis = AdverseImpactAnalyzer().run(df)
    assert "CLASS_COLUMN_ABSENT" in codes(analysis)
    msg = next(f.message for f in analysis.report.findings
               if f.code == "CLASS_COLUMN_ABSENT")
    assert "not the same as clear" in msg


def test_coverage_is_reported_for_every_class():
    analysis = AdverseImpactAnalyzer().run(roster(cohort(30, 3, "A") + cohort(30, 3, "B")))
    for cls in ("Age 40+", "Sex", "Race / Ethnicity", "Disability", "Veteran Status"):
        assert cls in analysis.report.coverage


# --- input handling --------------------------------------------------------


def test_missing_selected_column_is_an_error():
    df = roster(cohort(10, 0, "A")).drop(columns=["selected"])
    analysis = AdverseImpactAnalyzer().run(df)
    assert "MISSING_SELECTED_COLUMN" in codes(analysis)
    assert analysis.results.empty


def test_empty_input_is_handled():
    analysis = AdverseImpactAnalyzer().run(pd.DataFrame())
    assert "EMPTY_INPUT" in codes(analysis)


def test_no_selections_is_reported_rather_than_divided_by_zero():
    analysis = AdverseImpactAnalyzer().run(roster(cohort(20, 0, "A")))
    assert "NO_SELECTIONS" in codes(analysis)


def test_out_of_pool_employees_do_not_dilute_the_rates():
    """Including people who could never have been selected understates every
    rate and can hide a real disparity."""
    rows = (
        cohort(20, 8, "A", age_40_plus=True)
        + cohort(20, 1, "B", age_40_plus=False)
        + [{"employee_id": f"X{i}", "selected": False,
            "selection_status": "not_targeted", "age_40_plus": True}
           for i in range(60)]
    )
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    assert "OUT_OF_POOL_EXCLUDED" in codes(analysis)
    assert analysis.report.population == 40
    c = find(analysis, "Age 40+")
    assert c.group_total == 20


# --- reporting -------------------------------------------------------------


def test_report_never_recommends_swapping_individuals():
    analysis = AdverseImpactAnalyzer().run(
        roster(cohort(30, 15, "A", age_40_plus=True) + cohort(30, 1, "B", age_40_plus=False))
    )
    assert "NO_SWAP_RECOMMENDATIONS" in codes(analysis)
    md = analysis.report.to_markdown().lower()
    assert "do not adjust individual selections" in md
    # No output field should name a person to add or remove.
    assert "swap" not in analysis.results.to_csv().lower()


def test_markdown_carries_the_privilege_warning():
    analysis = AdverseImpactAnalyzer().run(roster(cohort(20, 2, "A") + cohort(20, 2, "B")))
    md = analysis.report.to_markdown()
    assert "privileged" in md.lower()
    assert "direction of counsel" in md.lower()


def test_class_verdicts_take_the_worst_result_per_class():
    rows = (
        cohort(20, 10, "EO", department="Engineering", age_40_plus=True)
        + cohort(20, 0, "EY", department="Engineering", age_40_plus=False)
        + cohort(40, 2, "SO", department="Sales", age_40_plus=True)
        + cohort(40, 8, "SY", department="Sales", age_40_plus=False)
    )
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    assert analysis.report.class_verdicts()["Age 40+"] == "Impact indicated"


def test_authority_is_cited_when_impact_is_indicated():
    analysis = AdverseImpactAnalyzer().run(
        roster(cohort(30, 15, "A", age_40_plus=True) + cohort(30, 1, "B", age_40_plus=False))
    )
    msg = next(f.message for f in analysis.report.findings
               if f.code == "ADVERSE_IMPACT_INDICATED")
    assert "ADEA" in msg or "FEHA" in msg


def test_multiple_comparisons_caveat_appears_on_large_analyses():
    rows = []
    for d in ("Engineering", "Sales", "Ops", "Finance"):
        rows += cohort(20, 4, f"{d}A", department=d, age_40_plus=True,
                       gender="female", race_ethnicity="white")
        rows += cohort(20, 2, f"{d}B", department=d, age_40_plus=False,
                       gender="male", race_ethnicity="asian")
    analysis = AdverseImpactAnalyzer().run(roster(rows))
    if len(analysis.report.comparisons) > 20:
        assert "MULTIPLE_COMPARISONS" in codes(analysis)


def test_results_dataframe_and_write_produce_all_artifacts():
    analysis = AdverseImpactAnalyzer().run(
        roster(cohort(30, 12, "A", age_40_plus=True) + cohort(30, 1, "B", age_40_plus=False))
    )
    assert not analysis.results.empty
    assert "impact_ratio" in analysis.results.columns
    assert not analysis.flagged.empty
    with tempfile.TemporaryDirectory() as tmp:
        paths = analysis.write(tmp)
        assert set(paths) == {"report_md", "results", "findings", "report_json"}
        assert all(p.exists() for p in paths.values())


def test_custom_protected_class_can_be_supplied():
    pc = ProtectedClass("Language", "primary_language", undisclosed=())
    rows = (
        cohort(20, 8, "A", primary_language="spanish")
        + cohort(20, 1, "B", primary_language="english")
    )
    analysis = AdverseImpactAnalyzer(classes=(pc,)).run(roster(rows))
    assert any(c.protected_class == "Language" for c in analysis.report.comparisons)


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
