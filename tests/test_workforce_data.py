"""
Tests for workforce_data.py

Run with pytest, or standalone:  python test_workforce_data.py
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
import io
import tempfile
from pathlib import Path

import pandas as pd

from rif_copilot.workforce_data import (
    IngestConfig,
    Severity,
    clean_text,
    load_workforce_csv,
    load_workforce_dataframe,
    map_columns,
    normalize_category,
    normalize_state,
    parse_bool,
    parse_date,
    parse_number,
    tenure_band,
)

AS_OF = dt.date(2026, 10, 30)


def _codes(result) -> set[str]:
    return {i.code for i in result.report.issues}


def _codes_for_row(result, row_number: int) -> set[str]:
    return {i.code for i in result.report.issues if i.row_number == row_number}


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows, dtype=str)


BASE_ROW = {
    "Employee ID": "E1",
    "First Name": "Ada",
    "Last Name": "Lovelace",
    "Job Title": "Engineer",
    "Department": "Engineering",
    "Hire Date": "2020-01-15",
    "Pay Type": "Salary",
    "Base Salary": "150000",
}


# --- scalar parsers --------------------------------------------------------


def test_clean_text_handles_null_tokens_and_whitespace():
    assert clean_text("  Ada   Lovelace ") == "Ada Lovelace"
    assert clean_text("N/A") is None
    assert clean_text("#N/A") is None
    assert clean_text("\u00a0") is None
    assert clean_text("") is None


def test_parse_number_handles_currency_percent_and_shorthand():
    assert parse_number("$168,500.00")[0] == 168500.0
    assert parse_number("(1,200)")[0] == -1200.0
    assert parse_number("15%")[0] == 0.15
    assert parse_number("85k")[0] == 85000.0
    assert parse_number("42.50 /hr")[0] == 42.5
    assert parse_number("not a number")[0] is None
    assert parse_number("not a number")[1] is not None


def test_parse_date_handles_common_formats_and_excel_serials():
    assert parse_date("03/15/2018")[0] == pd.Timestamp("2018-03-15")
    assert parse_date("2018-03-15")[0] == pd.Timestamp("2018-03-15")
    assert parse_date("15-Mar-2018")[0] == pd.Timestamp("2018-03-15")
    assert parse_date("March 15, 2018")[0] == pd.Timestamp("2018-03-15")
    assert parse_date("45000")[0] == pd.Timestamp("2023-03-15")  # Excel serial
    assert parse_date("nope")[0] is None
    assert parse_date(None)[0] is None


def test_parse_bool_and_state_and_category():
    assert parse_bool("Yes")[0] is True
    assert parse_bool("0")[0] is False
    assert parse_bool("maybe")[0] is None
    assert normalize_state("California")[0] == "CA"
    assert normalize_state("ca")[0] == "CA"
    assert normalize_state("Ontario")[0] is None
    assert normalize_category("employment_type", "F/T")[0] == "full_time"
    assert normalize_category("flsa_status", "Non-Exempt")[0] == "non_exempt"
    assert normalize_category("gender", "Prefer not to say")[0] == "not_disclosed"
    assert normalize_category("race_ethnicity", "Black or African American")[0] == (
        "black_african_american"
    )
    assert normalize_category("employment_type", "Zebra")[0] is None


def test_tenure_bands_are_ordered_and_exclusive():
    assert tenure_band(0.4) == "<1 year"
    assert tenure_band(1.0) == "1 to <3 years"
    assert tenure_band(4.8) == "3 to <5 years"
    assert tenure_band(9.99) == "5 to <10 years"
    assert tenure_band(20.0) == "20+ years"
    assert tenure_band(None) is None


# --- header mapping --------------------------------------------------------


def test_map_columns_matches_aliases_and_messy_headers():
    mapping, unmapped, _ = map_columns(
        ["Emp ID", "First  Name", "Last Name", "DEPT", "Date of Hire", "Widget Count"]
    )
    assert mapping["Emp ID"] == "employee_id"
    assert mapping["First  Name"] == "first_name"
    assert mapping["DEPT"] == "department"
    assert mapping["Date of Hire"] == "hire_date"
    assert "Widget Count" in unmapped


def test_map_columns_never_assigns_one_canonical_field_twice():
    mapping, _, _ = map_columns(["Employee ID", "Emp ID", "employee_number"])
    assert list(mapping.values()).count("employee_id") == 1


# --- structural validation -------------------------------------------------


def test_missing_required_column_is_blocking():
    df = _frame([{k: v for k, v in BASE_ROW.items() if k != "Hire Date"}])
    result = load_workforce_dataframe(df, as_of=AS_OF)
    assert result.report.is_blocking
    assert "hire_date" in result.report.missing_required_columns
    # Schema shape is preserved even when the column is absent.
    assert "hire_date" in result.data.columns


def test_empty_file_reports_cleanly():
    result = load_workforce_dataframe(pd.DataFrame(columns=["Employee ID"]), as_of=AS_OF)
    assert "EMPTY_FILE" in _codes(result)
    assert result.data.empty


def test_unmapped_columns_are_preserved_with_prefix():
    row = dict(BASE_ROW, **{"Legacy Code": "LG-77"})
    result = load_workforce_dataframe(_frame([row]), as_of=AS_OF)
    assert "x_legacy_code" in result.data.columns
    assert result.data.loc[0, "x_legacy_code"] == "LG-77"


# --- row validation --------------------------------------------------------


def test_missing_and_duplicate_employee_ids_are_errors():
    rows = [
        dict(BASE_ROW, **{"Employee ID": ""}),
        dict(BASE_ROW, **{"Employee ID": "E9"}),
        dict(BASE_ROW, **{"Employee ID": "E9"}),
    ]
    result = load_workforce_dataframe(_frame(rows), as_of=AS_OF)
    codes = _codes(result)
    assert "MISSING_EMPLOYEE_ID" in codes
    assert "DUPLICATE_EMPLOYEE_ID" in codes
    # Both duplicate rows are quarantined, not just the first.
    assert result.data["has_blocking_error"].tolist() == [True, True, True]
    assert len(result.clean) == 0


def test_date_ordering_violations_are_caught():
    rows = [
        dict(BASE_ROW, **{"Employee ID": "E1", "Term Date": "2019-01-01"}),
        dict(BASE_ROW, **{"Employee ID": "E2", "Adjusted Hire Date": "2018-01-01"}),
        dict(BASE_ROW, **{"Employee ID": "E3", "Hire Date": "2030-01-01"}),
    ]
    result = load_workforce_dataframe(_frame(rows), as_of=AS_OF)
    assert "TERM_BEFORE_HIRE" in _codes_for_row(result, 2)
    assert "REHIRE_BEFORE_HIRE" in _codes_for_row(result, 3)
    assert "FUTURE_HIRE_DATE" in _codes_for_row(result, 4)


def test_invalid_date_is_coerced_and_flagged_not_dropped():
    result = load_workforce_dataframe(
        _frame([dict(BASE_ROW, **{"Hire Date": "sometime in 2020"})]), as_of=AS_OF
    )
    assert "INVALID_DATE" in _codes(result)
    assert len(result.data) == 1  # row retained
    assert pd.isna(result.data.loc[0, "hire_date"])
    assert bool(result.data.loc[0, "has_blocking_error"]) is True


def test_pay_validations():
    rows = [
        dict(BASE_ROW, **{"Employee ID": "E1", "Pay Type": "Hourly", "Base Salary": "12.00"}),
        dict(BASE_ROW, **{"Employee ID": "E2", "Base Salary": "0"}),
        dict(BASE_ROW, **{"Employee ID": "E3", "Exempt Status": "Exempt", "Base Salary": "48000"}),
        dict(BASE_ROW, **{"Employee ID": "E4", "PTO Balance": "-10"}),
    ]
    result = load_workforce_dataframe(_frame(rows), as_of=AS_OF)
    assert "BELOW_MINIMUM_WAGE" in _codes_for_row(result, 2)
    assert "NONPOSITIVE_PAY" in _codes_for_row(result, 3)
    assert "EXEMPT_BELOW_SALARY_FLOOR" in _codes_for_row(result, 4)
    assert "NEGATIVE_VACATION_BALANCE" in _codes_for_row(result, 5)


def test_mislabeled_hourly_salary_does_not_produce_absurd_annual_pay():
    result = load_workforce_dataframe(
        _frame([dict(BASE_ROW, **{"Pay Type": "Hourly", "Base Salary": "215000"})]),
        as_of=AS_OF,
    )
    assert "PAY_TYPE_MISMATCH" in _codes(result)
    assert "PAY_BASIS_OVERRIDDEN" in _codes(result)
    assert float(result.data.loc[0, "annualized_pay"]) == 215000.0


# --- derived fields --------------------------------------------------------


def test_tenure_uses_adjusted_service_date_when_configured():
    row = dict(BASE_ROW, **{"Hire Date": "2010-01-01", "Adjusted Hire Date": "2020-01-01"})

    adjusted = load_workforce_dataframe(_frame([row]), as_of=AS_OF)
    assert adjusted.data.loc[0, "service_start_date"] == pd.Timestamp("2020-01-01")
    assert 6.5 < float(adjusted.data.loc[0, "tenure_years"]) < 7.0

    cfg = IngestConfig(as_of_date=AS_OF, use_adjusted_service_date=False)
    continuous = load_workforce_dataframe(_frame([row]), config=cfg)
    assert continuous.data.loc[0, "service_start_date"] == pd.Timestamp("2010-01-01")
    assert float(continuous.data.loc[0, "tenure_years"]) > 16


def test_age_40_plus_is_derived_for_adea_analysis():
    rows = [
        dict(BASE_ROW, **{"Employee ID": "E1", "DOB": "1979-06-02"}),
        dict(BASE_ROW, **{"Employee ID": "E2", "DOB": "1994-11-20"}),
        dict(BASE_ROW, **{"Employee ID": "E3"}),
    ]
    result = load_workforce_dataframe(_frame(rows), as_of=AS_OF)
    assert result.data.loc[0, "age_40_plus"] is True or bool(result.data.loc[0, "age_40_plus"])
    assert not bool(result.data.loc[1, "age_40_plus"])
    assert pd.isna(result.data.loc[2, "age_40_plus"])
    assert result.data.loc[0, "age_band"] == "40-49"


def test_hourly_pay_is_annualized_using_scheduled_hours():
    row = dict(BASE_ROW, **{"Pay Type": "Hourly", "Base Salary": "30", "Hours Per Week": "20"})
    result = load_workforce_dataframe(_frame([row]), as_of=AS_OF)
    assert float(result.data.loc[0, "annualized_pay"]) == 30 * 20 * 52
    assert float(result.data.loc[0, "hourly_equivalent_rate"]) == 30.0


def test_fte_percentage_is_converted_to_fraction():
    result = load_workforce_dataframe(
        _frame([dict(BASE_ROW, **{"FTE": "80"})]), as_of=AS_OF
    )
    assert float(result.data.loc[0, "fte"]) == 0.8


def test_terminated_employee_is_marked_inactive():
    result = load_workforce_dataframe(
        _frame([dict(BASE_ROW, **{"Term Date": "2026-06-30"})]), as_of=AS_OF
    )
    assert not bool(result.data.loc[0, "is_active"])
    assert "ALREADY_TERMINATED" in _codes(result)


# --- protected class handling ---------------------------------------------


def test_unmapped_category_preserves_raw_value_and_warns():
    result = load_workforce_dataframe(
        _frame([dict(BASE_ROW, **{"Gender": "Zorb"})]), as_of=AS_OF
    )
    assert "UNMAPPED_CATEGORY" in _codes(result)
    assert result.data.loc[0, "gender_raw"] == "Zorb"
    assert pd.isna(result.data.loc[0, "gender"])


def test_low_protected_class_coverage_is_flagged():
    rows = [dict(BASE_ROW, **{"Employee ID": f"E{i}", "Gender": "F" if i < 2 else ""})
            for i in range(10)]
    result = load_workforce_dataframe(_frame(rows), as_of=AS_OF)
    assert "LOW_PROTECTED_CLASS_COVERAGE" in _codes(result)


def test_high_not_disclosed_rate_is_flagged():
    rows = [dict(BASE_ROW, **{"Employee ID": f"E{i}",
                              "Race/Ethnicity": "Declined to State" if i < 5 else "White"})
            for i in range(10)]
    result = load_workforce_dataframe(_frame(rows), as_of=AS_OF)
    assert "HIGH_NOT_DISCLOSED_RATE" in _codes(result)


# --- compliance context checks --------------------------------------------


def test_out_of_state_and_union_and_leave_are_surfaced():
    rows = [
        dict(BASE_ROW, **{"Employee ID": "E1", "State": "TX"}),
        dict(BASE_ROW, **{"Employee ID": "E2", "State": "CA", "Union": "Yes"}),
        dict(BASE_ROW, **{"Employee ID": "E3", "State": "CA", "LOA": "FMLA"}),
    ]
    result = load_workforce_dataframe(_frame(rows), as_of=AS_OF)
    codes = _codes(result)
    assert {"OUT_OF_STATE_EMPLOYEES", "UNION_EMPLOYEES_PRESENT", "EMPLOYEES_ON_LEAVE"} <= codes


# --- config and IO ---------------------------------------------------------


def test_drop_error_rows_removes_bad_rows_but_keeps_the_findings():
    rows = [dict(BASE_ROW, **{"Employee ID": "E1"}),
            dict(BASE_ROW, **{"Employee ID": ""})]
    cfg = IngestConfig(as_of_date=AS_OF, drop_error_rows=True)
    result = load_workforce_dataframe(_frame(rows), config=cfg)
    assert len(result.data) == 1
    assert "MISSING_EMPLOYEE_ID" in _codes(result)


def test_csv_roundtrip_and_report_outputs():
    csv = io.StringIO()
    _frame([BASE_ROW]).to_csv(csv, index=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "roster.csv"
        path.write_text(csv.getvalue(), encoding="utf-8")
        result = load_workforce_csv(path, as_of=AS_OF)
        assert len(result.data) == 1
        paths = result.write(Path(tmp) / "out")
        assert all(p.exists() for p in paths.values())
        assert "Workforce Data Validation Report" in paths["report_md"].read_text()
        assert isinstance(result.report.to_json(), str)
        assert isinstance(result.report.to_dataframe(), pd.DataFrame)


def test_missing_file_returns_parse_failure_not_exception():
    result = load_workforce_csv("/nonexistent/roster.csv", as_of=AS_OF)
    assert "PARSE_FAILURE" in _codes(result)
    assert result.data.empty


def test_row_numbers_match_source_file_lines():
    rows = [dict(BASE_ROW, **{"Employee ID": f"E{i}"}) for i in range(3)]
    result = load_workforce_dataframe(_frame(rows), as_of=AS_OF)
    assert result.data["row_number"].tolist() == [2, 3, 4]  # header is line 1


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
