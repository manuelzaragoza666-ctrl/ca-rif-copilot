"""
workforce_data.py
=================

Data Manager for the California RIF Copilot (system architecture module 1).

Responsibilities
----------------
1. Import employee workforce data from CSV (or Excel-exported CSV).
2. Map messy real-world headers onto a canonical schema.
3. Validate that required columns and values are present and well formed.
4. Normalize records (whitespace, casing, dates, currency, categoricals).
5. Flag missing / invalid / suspicious values with severity and row context.
6. Derive tenure, age, pay, and protected-class fields needed downstream by
   the Adverse Impact Analyzer, CA Compliance Engine, and Severance Engine.
7. Emit a standardized DataFrame plus a structured ValidationReport.

Design notes
------------
* Nothing is silently dropped. Bad values are coerced to NA *and* recorded as
  an issue, so the audit trail can always explain what changed and why.
* Protected-class columns (age, sex, race/ethnicity, disability, veteran) are
  never normalized into new categories that the source did not contain; unknown
  values are preserved in a `*_raw` column and flagged for human review.
* Severity model:
      ERROR   -> blocks downstream analysis for that row (or the whole file)
      WARNING -> analysis can proceed, but a human should review
      INFO    -> a normalization was applied, recorded for the audit trail

Usage
-----
    from .workforce_data import load_workforce_csv

    result = load_workforce_csv("roster.csv", as_of="2026-10-30")
    result.data                  # standardized pandas DataFrame
    result.report.summary()      # dict of counts
    print(result.report.to_markdown())
    result.report.to_dataframe() # one row per issue

CLI
---
    python workforce_data.py roster.csv --as-of 2026-10-30 --outdir ./out
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "FieldSpec",
    "SCHEMA",
    "Issue",
    "Severity",
    "ValidationReport",
    "IngestResult",
    "IngestConfig",
    "load_workforce_csv",
    "load_workforce_dataframe",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Configuration constants (California-specific, 2026 defaults)
# ---------------------------------------------------------------------------

# CA statewide minimum wage. Update annually; local ordinances may be higher.
CA_MIN_HOURLY_WAGE = 16.90

# CA exempt salary floor = 2x state minimum wage for full-time employment.
CA_EXEMPT_ANNUAL_FLOOR = CA_MIN_HOURLY_WAGE * 2 * 40 * 52

# Protected-class coverage below this fraction makes adverse impact testing
# statistically unreliable, so we surface it loudly.
MIN_PROTECTED_CLASS_COVERAGE = 0.90

DEFAULT_HOURS_PER_WEEK = 40.0
WEEKS_PER_YEAR = 52.0
DAYS_PER_YEAR = 365.25


class Severity:
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"

    ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    """Canonical description of one column in the standardized roster."""

    name: str
    dtype: str  # string | date | float | int | bool | category
    required: bool = False
    aliases: tuple[str, ...] = ()
    allowed: tuple[str, ...] = ()
    description: str = ""
    # Fields that feed statistical adverse-impact testing.
    protected_class: bool = False
    # Fields required for CA WARN / final pay math.
    compliance_critical: bool = False
    # Legitimately blank for most employees (leave, visa); do not flag as incomplete.
    sparse_ok: bool = False


SCHEMA: tuple[FieldSpec, ...] = (
    # ---- identity -------------------------------------------------------
    FieldSpec(
        "employee_id", "string", required=True,
        aliases=("emp id", "employee number", "emp no", "worker id", "person id",
                 "associate id", "eeid", "employee code", "id"),
        description="Unique, stable identifier for the employee.",
        compliance_critical=True,
    ),
    FieldSpec(
        "first_name", "string",
        aliases=("given name", "fname", "legal first name", "preferred first name"),
    ),
    FieldSpec(
        "last_name", "string", required=True,
        aliases=("surname", "family name", "lname", "legal last name"),
    ),
    FieldSpec(
        "work_email", "string",
        aliases=("email", "email address", "work email address", "company email"),
    ),
    # ---- job ------------------------------------------------------------
    FieldSpec(
        "job_title", "string", required=True,
        aliases=("title", "position", "position title", "job", "role"),
    ),
    FieldSpec(
        "job_level", "string",
        aliases=("level", "grade", "job grade", "band", "pay grade"),
    ),
    FieldSpec(
        "department", "string", required=True,
        aliases=("dept", "department name", "org", "organization", "business unit",
                 "cost center name", "function"),
    ),
    FieldSpec(
        "cost_center", "string",
        aliases=("cc", "cost centre", "cost center code", "gl code"),
    ),
    FieldSpec(
        "manager_id", "string",
        aliases=("supervisor id", "manager employee id", "reports to id",
                 "manager", "supervisor"),
    ),
    FieldSpec(
        "worksite_name", "string",
        aliases=("site", "location", "office", "work location", "facility",
                 "location name", "worksite"),
        description="Single site of employment; drives CA WARN thresholds.",
        compliance_critical=True,
    ),
    FieldSpec(
        "work_city", "string",
        aliases=("city", "work city", "location city"),
        compliance_critical=True,
    ),
    FieldSpec(
        "work_state", "string",
        aliases=("state", "work state", "location state", "province", "region"),
        description="Two-letter state code; determines which WARN act applies.",
        compliance_critical=True,
    ),
    FieldSpec(
        "remote_flag", "bool",
        aliases=("is remote", "remote", "work arrangement", "telecommuter"),
        description="Remote employees are assigned to a reporting worksite for WARN.",
        sparse_ok=True,
    ),
    # ---- employment terms -----------------------------------------------
    FieldSpec(
        "employment_type", "category",
        aliases=("emp type", "worker type", "employee type", "employment status",
                 "worker category"),
        allowed=("full_time", "part_time", "temporary", "seasonal", "intern",
                 "contractor", "on_call"),
        description="Part-time/temp status affects WARN headcount inclusion.",
        compliance_critical=True,
    ),
    FieldSpec(
        "flsa_status", "category",
        aliases=("exempt status", "exempt", "flsa", "overtime eligible", "exemption"),
        allowed=("exempt", "non_exempt"),
    ),
    FieldSpec(
        "fte", "float",
        aliases=("fte percent", "full time equivalent", "fte %", "scheduled fte"),
    ),
    FieldSpec(
        "standard_hours_per_week", "float",
        aliases=("hours per week", "weekly hours", "scheduled hours", "std hours"),
    ),
    FieldSpec(
        "union_flag", "bool",
        aliases=("union", "is union", "cba", "union member", "bargaining unit"),
        description="Union employees may trigger separate bargaining obligations.",
        compliance_critical=True,
        sparse_ok=True,
    ),
    FieldSpec(
        "visa_status", "string",
        aliases=("work authorization", "visa", "immigration status", "sponsorship"),
        description="Sponsored employees need separate notification handling.",
        sparse_ok=True,
    ),
    FieldSpec(
        "leave_status", "string",
        aliases=("loa", "leave", "on leave", "leave type", "absence status"),
        description="Employees on protected leave require extra legal review.",
        compliance_critical=True,
        sparse_ok=True,
    ),
    # ---- dates ----------------------------------------------------------
    FieldSpec(
        "hire_date", "date", required=True,
        aliases=("start date", "date of hire", "original hire date", "doh",
                 "hired", "employment start date"),
        compliance_critical=True,
    ),
    FieldSpec(
        "rehire_date", "date",
        aliases=("adjusted hire date", "most recent hire date", "seniority date",
                 "continuous service date", "last hire date"),
        sparse_ok=True,
    ),
    FieldSpec(
        "termination_date", "date",
        aliases=("term date", "separation date", "end date", "last day worked",
                 "date of termination"),
        sparse_ok=True,
    ),
    FieldSpec(
        "birth_date", "date",
        aliases=("dob", "date of birth", "birthdate", "birth dt"),
        description="Used only to derive the age-40+ protected class.",
        protected_class=True,
    ),
    # ---- pay ------------------------------------------------------------
    FieldSpec(
        "pay_type", "category",
        aliases=("pay basis", "salary type", "compensation type", "rate type",
                 "hourly or salary"),
        allowed=("salary", "hourly"),
        compliance_critical=True,
    ),
    FieldSpec(
        "pay_rate", "float", required=True,
        aliases=("rate", "base pay", "base salary", "annual salary", "salary",
                 "hourly rate", "compensation", "base rate", "pay"),
        compliance_critical=True,
    ),
    FieldSpec(
        "pay_frequency", "category",
        aliases=("pay period", "payroll frequency", "frequency", "pay cycle"),
        allowed=("hourly", "weekly", "biweekly", "semimonthly", "monthly", "annual"),
    ),
    FieldSpec(
        "target_bonus_pct", "float",
        aliases=("bonus target", "bonus %", "target bonus", "incentive target"),
    ),
    FieldSpec(
        "accrued_vacation_hours", "float",
        aliases=("pto balance", "vacation balance", "accrued pto", "vacation hours",
                 "pto hours", "unused vacation"),
        description="CA Labor Code 227.3 requires payout of all accrued vacation.",
        compliance_critical=True,
    ),
    # ---- performance ----------------------------------------------------
    FieldSpec(
        "performance_rating", "string",
        aliases=("rating", "last rating", "performance score", "review rating",
                 "annual rating", "perf rating"),
    ),
    FieldSpec(
        "last_review_date", "date",
        aliases=("review date", "last performance review", "last review"),
    ),
    # ---- capability -----------------------------------------------------
    FieldSpec(
        "skills", "string",
        aliases=("skill set", "skills list", "competencies", "capabilities",
                 "technical skills", "skill tags"),
        description="Delimited list of skills; used by the Selection Criteria Engine.",
        sparse_ok=True,
    ),
    FieldSpec(
        "certifications", "string",
        aliases=("certs", "certificates", "licenses", "credentials",
                 "professional certifications", "license"),
        description="Delimited list of certifications or licenses held.",
        sparse_ok=True,
    ),
    # ---- protected classes ---------------------------------------------
    FieldSpec(
        "gender", "category",
        aliases=("sex", "gender identity", "gender code"),
        allowed=("female", "male", "non_binary", "not_disclosed"),
        protected_class=True,
    ),
    FieldSpec(
        "race_ethnicity", "category",
        aliases=("race", "ethnicity", "race/ethnicity", "eeo race", "ethnic group",
                 "race ethnicity"),
        allowed=("american_indian_alaska_native", "asian", "black_african_american",
                 "hispanic_latino", "native_hawaiian_pacific_islander",
                 "two_or_more_races", "white", "not_disclosed"),
        protected_class=True,
    ),
    FieldSpec(
        "disability_status", "category",
        aliases=("disability", "has disability", "disabled", "ada status",
                 "self identified disability"),
        allowed=("yes", "no", "not_disclosed"),
        protected_class=True,
    ),
    FieldSpec(
        "veteran_status", "category",
        aliases=("veteran", "is veteran", "protected veteran", "military status"),
        allowed=("yes", "no", "not_disclosed"),
        protected_class=True,
    ),
)

SCHEMA_BY_NAME: dict[str, FieldSpec] = {f.name: f for f in SCHEMA}

# Derived columns produced by this module (not expected in source files).
DERIVED_COLUMNS = (
    "full_name",
    "service_start_date",
    "tenure_days",
    "tenure_years",
    "tenure_months",
    "tenure_band",
    "age_years",
    "age_band",
    "age_40_plus",
    "annualized_pay",
    "hourly_equivalent_rate",
    "is_active",
    "row_number",
    "has_blocking_error",
)


# ---------------------------------------------------------------------------
# Value vocabularies for categorical normalization
# ---------------------------------------------------------------------------

CATEGORY_MAPS: dict[str, dict[str, str]] = {
    "employment_type": {
        "ft": "full_time", "f/t": "full_time", "full time": "full_time",
        "fulltime": "full_time", "full-time": "full_time", "regular": "full_time",
        "regular full time": "full_time", "r": "full_time",
        "pt": "part_time", "p/t": "part_time", "part time": "part_time",
        "parttime": "part_time", "part-time": "part_time",
        "temp": "temporary", "temporary": "temporary", "contingent": "temporary",
        "fixed term": "temporary", "fixed-term": "temporary",
        "seasonal": "seasonal",
        "intern": "intern", "internship": "intern", "co-op": "intern",
        "contractor": "contractor", "contract": "contractor",
        "consultant": "contractor", "1099": "contractor", "agency": "contractor",
        "on call": "on_call", "on-call": "on_call", "per diem": "on_call",
        "casual": "on_call",
    },
    "flsa_status": {
        "e": "exempt", "exempt": "exempt", "salaried exempt": "exempt",
        "exempt salaried": "exempt", "y": "exempt", "yes": "exempt",
        "true": "exempt", "1": "exempt",
        "ne": "non_exempt", "n": "non_exempt", "no": "non_exempt",
        "false": "non_exempt", "0": "non_exempt",
        "non exempt": "non_exempt", "nonexempt": "non_exempt",
        "non-exempt": "non_exempt", "hourly non exempt": "non_exempt",
        "overtime eligible": "non_exempt",
    },
    "pay_type": {
        "s": "salary", "sal": "salary", "salary": "salary", "salaried": "salary",
        "annual": "salary", "annually": "salary", "exempt": "salary",
        "h": "hourly", "hr": "hourly", "hourly": "hourly", "wage": "hourly",
        "per hour": "hourly", "non exempt": "hourly",
    },
    "pay_frequency": {
        "hourly": "hourly", "per hour": "hourly",
        "weekly": "weekly", "wk": "weekly", "52": "weekly",
        "biweekly": "biweekly", "bi-weekly": "biweekly", "bi weekly": "biweekly",
        "every two weeks": "biweekly", "26": "biweekly",
        "semimonthly": "semimonthly", "semi-monthly": "semimonthly",
        "semi monthly": "semimonthly", "twice monthly": "semimonthly", "24": "semimonthly",
        "monthly": "monthly", "12": "monthly",
        "annual": "annual", "annually": "annual", "yearly": "annual", "1": "annual",
    },
    "gender": {
        "f": "female", "female": "female", "woman": "female", "w": "female",
        "m": "male", "male": "male", "man": "male",
        "nb": "non_binary", "non binary": "non_binary", "non-binary": "non_binary",
        "nonbinary": "non_binary", "x": "non_binary", "genderqueer": "non_binary",
        "u": "not_disclosed", "unknown": "not_disclosed", "n/a": "not_disclosed",
        "decline": "not_disclosed", "declined": "not_disclosed",
        "declined to state": "not_disclosed", "prefer not to say": "not_disclosed",
        "not specified": "not_disclosed", "not disclosed": "not_disclosed",
    },
    "race_ethnicity": {
        "white": "white", "caucasian": "white", "white (not hispanic or latino)": "white",
        "black": "black_african_american",
        "african american": "black_african_american",
        "black or african american": "black_african_american",
        "black/african american": "black_african_american",
        "asian": "asian", "asian american": "asian",
        "hispanic": "hispanic_latino", "latino": "hispanic_latino",
        "latinx": "hispanic_latino", "hispanic or latino": "hispanic_latino",
        "hispanic/latino": "hispanic_latino",
        "native american": "american_indian_alaska_native",
        "american indian": "american_indian_alaska_native",
        "american indian or alaska native": "american_indian_alaska_native",
        "alaska native": "american_indian_alaska_native",
        "pacific islander": "native_hawaiian_pacific_islander",
        "native hawaiian": "native_hawaiian_pacific_islander",
        "native hawaiian or other pacific islander": "native_hawaiian_pacific_islander",
        "two or more": "two_or_more_races",
        "two or more races": "two_or_more_races",
        "multiracial": "two_or_more_races", "mixed": "two_or_more_races",
        "other": "not_disclosed", "unknown": "not_disclosed",
        "decline": "not_disclosed", "declined to state": "not_disclosed",
        "prefer not to say": "not_disclosed", "not disclosed": "not_disclosed",
        "n/a": "not_disclosed",
    },
    "disability_status": {
        "y": "yes", "yes": "yes", "true": "yes", "1": "yes", "disabled": "yes",
        "n": "no", "no": "no", "false": "no", "0": "no", "not disabled": "no",
        "unknown": "not_disclosed", "decline": "not_disclosed",
        "declined to state": "not_disclosed", "prefer not to say": "not_disclosed",
        "not disclosed": "not_disclosed", "n/a": "not_disclosed",
    },
    "veteran_status": {
        "y": "yes", "yes": "yes", "true": "yes", "1": "yes", "veteran": "yes",
        "protected veteran": "yes",
        "n": "no", "no": "no", "false": "no", "0": "no", "not a veteran": "no",
        "non veteran": "no", "nonveteran": "no",
        "unknown": "not_disclosed", "decline": "not_disclosed",
        "declined to state": "not_disclosed", "prefer not to say": "not_disclosed",
        "not disclosed": "not_disclosed", "n/a": "not_disclosed",
    },
}

TRUE_TOKENS = {"y", "yes", "true", "t", "1", "1.0", "x", "on", "remote"}
FALSE_TOKENS = {"n", "no", "false", "f", "0", "0.0", "off", "onsite", "on site",
                "in office", "in-office", "hybrid"}

NULL_TOKENS = {
    "", "na", "n/a", "n.a.", "none", "null", "nil", "nan", "-", "--", "---",
    "#n/a", "#null!", "#value!", "#ref!", "unknown", "tbd", ".", "?",
    "not applicable", "no data",
}

US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",
    "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",
    "PR", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
}

STATE_NAMES = {
    "california": "CA", "new york": "NY", "texas": "TX", "washington": "WA",
    "oregon": "OR", "nevada": "NV", "arizona": "AZ", "colorado": "CO",
    "illinois": "IL", "florida": "FL", "massachusetts": "MA", "georgia": "GA",
    "north carolina": "NC", "new jersey": "NJ", "pennsylvania": "PA",
    "virginia": "VA", "michigan": "MI", "ohio": "OH", "utah": "UT",
    "minnesota": "MN", "district of columbia": "DC",
}

DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y", "%Y/%m/%d",
    "%m-%d-%Y", "%d-%b-%Y", "%d-%b-%y", "%b %d, %Y", "%B %d, %Y",
    "%d %B %Y", "%Y%m%d", "%m.%d.%Y", "%Y-%m-%d %H:%M:%S",
)

PAY_FREQUENCY_MULTIPLIER = {
    "hourly": None,  # handled via hours/week
    "weekly": 52.0,
    "biweekly": 26.0,
    "semimonthly": 24.0,
    "monthly": 12.0,
    "annual": 1.0,
}


# ---------------------------------------------------------------------------
# Issues and reporting
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    """A single validation finding, scoped to the file, a column, or a row."""

    severity: str
    code: str
    message: str
    column: str | None = None
    row_number: int | None = None  # 1-based source row (excluding header)
    employee_id: str | None = None
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["value"] is not None and not isinstance(d["value"], (str, int, float, bool)):
            d["value"] = str(d["value"])
        return d


@dataclass
class ValidationReport:
    """Structured, auditable record of everything found during ingestion."""

    source: str = ""
    generated_at: str = field(default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))
    as_of_date: str | None = None
    row_count: int = 0
    column_count: int = 0
    issues: list[Issue] = field(default_factory=list)
    column_map: dict[str, str] = field(default_factory=dict)
    unmapped_columns: list[str] = field(default_factory=list)
    missing_required_columns: list[str] = field(default_factory=list)
    completeness: dict[str, float] = field(default_factory=dict)

    # -- construction ----------------------------------------------------
    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        column: str | None = None,
        row_number: int | None = None,
        employee_id: str | None = None,
        value: Any = None,
    ) -> None:
        self.issues.append(
            Issue(severity, code, message, column, row_number, employee_id, value)
        )

    # -- querying --------------------------------------------------------
    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def is_blocking(self) -> bool:
        """True if the file cannot be used downstream at all."""
        blocking_codes = {"MISSING_REQUIRED_COLUMN", "EMPTY_FILE", "PARSE_FAILURE"}
        return any(i.code in blocking_codes for i in self.errors)

    def error_row_numbers(self) -> set[int]:
        return {i.row_number for i in self.errors if i.row_number is not None}

    def summary(self) -> dict[str, Any]:
        by_code: dict[str, int] = {}
        for i in self.issues:
            by_code[i.code] = by_code.get(i.code, 0) + 1
        return {
            "source": self.source,
            "generated_at": self.generated_at,
            "as_of_date": self.as_of_date,
            "rows": self.row_count,
            "columns_mapped": len(self.column_map),
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "info": len([i for i in self.issues if i.severity == Severity.INFO]),
            "rows_with_errors": len(self.error_row_numbers()),
            "blocking": self.is_blocking,
            "missing_required_columns": self.missing_required_columns,
            "unmapped_columns": self.unmapped_columns,
            "issues_by_code": dict(sorted(by_code.items(), key=lambda kv: -kv[1])),
        }

    # -- output ----------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self.issues:
            return pd.DataFrame(
                columns=["severity", "code", "message", "column", "row_number",
                         "employee_id", "value"]
            )
        df = pd.DataFrame([i.to_dict() for i in self.issues])
        df["_sev"] = df["severity"].map(Severity.ORDER).fillna(9)
        df = df.sort_values(
            ["_sev", "code", "row_number"], na_position="first"
        ).drop(columns="_sev").reset_index(drop=True)
        return df

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "completeness": self.completeness,
            "column_map": self.column_map,
            "issues": [i.to_dict() for i in self.issues],
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path is not None:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self, max_rows_per_code: int = 5) -> str:
        s = self.summary()
        lines: list[str] = []
        lines.append("# Workforce Data Validation Report")
        lines.append("")
        lines.append(f"**Source:** `{s['source'] or 'in-memory dataframe'}`  ")
        lines.append(f"**Generated:** {s['generated_at']}  ")
        lines.append(f"**Tenure as of:** {s['as_of_date']}  ")
        lines.append("")
        status = "BLOCKED" if s["blocking"] else ("REVIEW REQUIRED" if s["errors"] or s["warnings"] else "CLEAN")
        lines.append(f"## Status: {status}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        lines.append(f"| Rows imported | {s['rows']} |")
        lines.append(f"| Rows with errors | {s['rows_with_errors']} |")
        lines.append(f"| Errors | {s['errors']} |")
        lines.append(f"| Warnings | {s['warnings']} |")
        lines.append(f"| Columns mapped | {s['columns_mapped']} |")
        lines.append("")

        if self.missing_required_columns:
            lines.append("### Missing required columns")
            for c in self.missing_required_columns:
                lines.append(f"- `{c}` — {SCHEMA_BY_NAME[c].description or 'required by schema'}")
            lines.append("")

        if self.unmapped_columns:
            lines.append("### Unmapped source columns (carried through as `x_*`)")
            lines.append(", ".join(f"`{c}`" for c in self.unmapped_columns))
            lines.append("")

        # Completeness table for compliance-critical + protected fields.
        watch = [f.name for f in SCHEMA if f.protected_class or f.compliance_critical]
        rows = [(c, self.completeness[c]) for c in watch if c in self.completeness]
        if rows:
            lines.append("### Field completeness (compliance & protected-class fields)")
            lines.append("| Field | Populated | Flag |")
            lines.append("|---|---|---|")
            for c, pct in rows:
                if SCHEMA_BY_NAME[c].sparse_ok:
                    flag = "n/a (sparse by design)"
                else:
                    flag = "OK" if pct >= MIN_PROTECTED_CLASS_COVERAGE else "REVIEW"
                lines.append(f"| {c} | {pct:.1%} | {flag} |")
            lines.append("")

        if self.issues:
            lines.append("### Findings")
            df = self.to_dataframe()
            for code, group in df.groupby("code", sort=False):
                sev = group["severity"].iloc[0]
                lines.append(f"**[{sev}] {code}** — {len(group)} occurrence(s)")
                for _, r in group.head(max_rows_per_code).iterrows():
                    where = []
                    if pd.notna(r.get("row_number")):
                        where.append(f"row {int(r['row_number'])}")
                    if isinstance(r.get("employee_id"), str) and r["employee_id"]:
                        where.append(f"emp {r['employee_id']}")
                    if isinstance(r.get("column"), str) and r["column"]:
                        where.append(f"col `{r['column']}`")
                    prefix = f"({'; '.join(where)}) " if where else ""
                    lines.append(f"- {prefix}{r['message']}")
                if len(group) > max_rows_per_code:
                    lines.append(f"- …and {len(group) - max_rows_per_code} more")
                lines.append("")
        else:
            lines.append("No issues found.")
            lines.append("")

        lines.append("---")
        lines.append(
            "_Automated data validation only. Selection decisions, legal review, "
            "and final compliance determinations require human judgment._"
        )
        return "\n".join(lines)


@dataclass
class IngestConfig:
    """Tunable ingestion behavior."""

    as_of_date: dt.date = field(default_factory=dt.date.today)
    # Use rehire_date (adjusted service date) instead of hire_date for tenure.
    use_adjusted_service_date: bool = True
    min_hourly_wage: float = CA_MIN_HOURLY_WAGE
    exempt_annual_floor: float = CA_EXEMPT_ANNUAL_FLOOR
    default_hours_per_week: float = DEFAULT_HOURS_PER_WEEK
    # Drop rows that hit a blocking row-level error (missing id, bad hire date).
    drop_error_rows: bool = False
    # Keep unmapped source columns, prefixed with x_.
    keep_extra_columns: bool = True
    # Fuzzy header matching threshold (0-1); set to 1.0 to require exact aliases.
    header_match_cutoff: float = 0.86
    expected_states: tuple[str, ...] = ("CA",)


@dataclass
class IngestResult:
    """Standardized dataframe plus its validation report."""

    data: pd.DataFrame
    report: ValidationReport
    config: IngestConfig

    @property
    def clean(self) -> pd.DataFrame:
        """Rows with no blocking row-level error — safe for downstream modules."""
        if "has_blocking_error" not in self.data.columns:
            return self.data
        return self.data.loc[~self.data["has_blocking_error"]].copy()

    def write(self, outdir: str | Path, stem: str = "workforce") -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "data": outdir / f"{stem}_standardized.csv",
            "issues": outdir / f"{stem}_validation_issues.csv",
            "report_json": outdir / f"{stem}_validation_report.json",
            "report_md": outdir / f"{stem}_validation_report.md",
        }
        self.data.to_csv(paths["data"], index=False)
        self.report.to_dataframe().to_csv(paths["issues"], index=False)
        self.report.to_json(paths["report_json"])
        paths["report_md"].write_text(self.report.to_markdown(), encoding="utf-8")
        return paths


# ---------------------------------------------------------------------------
# Scalar normalizers
# ---------------------------------------------------------------------------


def _norm_key(text: str) -> str:
    """Normalize a header for matching: lowercase, alphanumeric only."""
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", " ", text)


def clean_text(value: Any) -> str | None:
    """Trim, collapse whitespace, strip zero-width chars; NULL tokens -> None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (pd.Timestamp, dt.date, dt.datetime)):
        return str(value)
    s = str(value)
    s = s.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    if s.lower() in NULL_TOKENS:
        return None
    return s or None


def parse_date(value: Any) -> tuple[pd.Timestamp | None, str | None]:
    """Parse a date. Returns (timestamp, error_reason)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    if isinstance(value, (pd.Timestamp, dt.datetime)):
        return pd.Timestamp(value).normalize(), None
    if isinstance(value, dt.date):
        return pd.Timestamp(value), None

    # Excel serial dates (e.g. 45000 -> 2023-03-15)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        serial = float(value)
        if 20000 <= serial <= 60000:
            base = pd.Timestamp("1899-12-30")
            return (base + pd.Timedelta(days=serial)).normalize(), None
        return None, f"numeric value {value!r} is not a recognizable date"

    s = clean_text(value)
    if s is None:
        return None, None
    if re.fullmatch(r"\d{5}", s):  # Excel serial as text
        return (pd.Timestamp("1899-12-30") + pd.Timedelta(days=int(s))).normalize(), None
    for fmt in DATE_FORMATS:
        try:
            return pd.Timestamp(dt.datetime.strptime(s, fmt)).normalize(), None
        except ValueError:
            continue
    try:
        ts = pd.to_datetime(s, errors="raise")
        return pd.Timestamp(ts).normalize(), None
    except Exception:
        return None, f"unrecognized date format: {s!r}"


def parse_number(value: Any) -> tuple[float | None, str | None]:
    """Parse currency/percent/number strings. Returns (float, error_reason)."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    if isinstance(value, bool):
        return float(value), None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), None

    s = clean_text(value)
    if s is None:
        return None, None

    negative = s.startswith("(") and s.endswith(")")
    s = s.strip("()")
    is_pct = s.endswith("%")
    s = s.rstrip("%")
    # Strip trailing unit suffixes: "42.50 /hr", "85,000 per year", "60000 USD".
    s = re.sub(
        r"(?i)\s*(/\s*(hr|hour|yr|year)|per\s+(hour|hr|year|yr|annum)|"
        r"\b(usd|annually|hourly|hrs?|yrs?)\b)\s*$",
        "", s,
    ).strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")

    # 85k / 1.2m shorthand
    mult = 1.0
    if re.fullmatch(r"-?\d+(\.\d+)?[kK]", s):
        mult, s = 1_000.0, s[:-1]
    elif re.fullmatch(r"-?\d+(\.\d+)?[mM]", s):
        mult, s = 1_000_000.0, s[:-1]

    try:
        num = float(s) * mult
    except ValueError:
        return None, f"could not parse numeric value: {value!r}"
    if negative:
        num = -num
    if is_pct:
        num = num / 100.0
    return num, None


def parse_bool(value: Any) -> tuple[bool | None, str | None]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None
    if isinstance(value, (bool, np.bool_)):
        return bool(value), None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value), None
    s = clean_text(value)
    if s is None:
        return None, None
    low = s.lower()
    if low in TRUE_TOKENS:
        return True, None
    if low in FALSE_TOKENS:
        return False, None
    return None, f"unrecognized boolean value: {value!r}"


def normalize_category(field_name: str, value: Any) -> tuple[str | None, str | None]:
    """Map a raw categorical value onto the canonical vocabulary."""
    s = clean_text(value)
    if s is None:
        return None, None
    key = s.lower().strip().rstrip(".")
    mapping = CATEGORY_MAPS.get(field_name, {})
    if key in mapping:
        return mapping[key], None
    snake = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    allowed = SCHEMA_BY_NAME[field_name].allowed
    if snake in allowed:
        return snake, None
    close = difflib.get_close_matches(key, list(mapping.keys()), n=1, cutoff=0.9)
    if close:
        return mapping[close[0]], None
    return None, f"value {s!r} is not a recognized {field_name} category"


def normalize_state(value: Any) -> tuple[str | None, str | None]:
    s = clean_text(value)
    if s is None:
        return None, None
    up = s.upper().replace(".", "")
    if up in US_STATES:
        return up, None
    low = s.lower()
    if low in STATE_NAMES:
        return STATE_NAMES[low], None
    return None, f"unrecognized state value: {s!r}"


def normalize_name(value: Any) -> str | None:
    """Title-case a person's name while preserving common particles."""
    s = clean_text(value)
    if s is None:
        return None
    if s.isupper() or s.islower():
        parts = []
        for token in s.split(" "):
            sub = re.split(r"([-'])", token)
            parts.append("".join(p if p in "-'" else p.capitalize() for p in sub))
        s = " ".join(parts)
    return s


EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


# ---------------------------------------------------------------------------
# Header mapping
# ---------------------------------------------------------------------------


def map_columns(
    columns: Sequence[str], cutoff: float = 0.86
) -> tuple[dict[str, str], list[str], list[tuple[str, str, str]]]:
    """Map source headers to canonical field names.

    Returns (source_col -> canonical, unmapped_source_cols, fuzzy_matches).
    """
    # Build lookup: normalized alias -> canonical name.
    lookup: dict[str, str] = {}
    for spec in SCHEMA:
        lookup[_norm_key(spec.name)] = spec.name
        lookup[_norm_key(spec.name.replace("_", " "))] = spec.name
        for alias in spec.aliases:
            lookup.setdefault(_norm_key(alias), spec.name)

    mapping: dict[str, str] = {}
    unmapped: list[str] = []
    fuzzy: list[tuple[str, str, str]] = []
    taken: set[str] = set()

    # Pass 1: exact matches on normalized keys.
    normalized = {col: _norm_key(col) for col in columns}
    for col, key in normalized.items():
        canon = lookup.get(key)
        if canon and canon not in taken:
            mapping[col] = canon
            taken.add(canon)

    # Pass 2: fuzzy matching for the leftovers.
    remaining_keys = [k for k, v in lookup.items() if v not in taken]
    for col, key in normalized.items():
        if col in mapping:
            continue
        match = difflib.get_close_matches(key, remaining_keys, n=1, cutoff=cutoff)
        if match:
            canon = lookup[match[0]]
            if canon not in taken:
                mapping[col] = canon
                taken.add(canon)
                fuzzy.append((col, canon, match[0]))
                remaining_keys = [k for k in remaining_keys if lookup[k] != canon]
                continue
        unmapped.append(col)

    return mapping, unmapped, fuzzy


# ---------------------------------------------------------------------------
# Core normalization pipeline
# ---------------------------------------------------------------------------


def _row_label(df: pd.DataFrame, idx: int) -> tuple[int, str | None]:
    row_no = int(df.at[idx, "row_number"]) if "row_number" in df.columns else int(idx) + 2
    emp = df.at[idx, "employee_id"] if "employee_id" in df.columns else None
    return row_no, (emp if isinstance(emp, str) else None)


def _normalize_column(
    df: pd.DataFrame, spec: FieldSpec, report: ValidationReport, cfg: IngestConfig
) -> None:
    """Normalize one canonical column in place and record any coercion issues."""
    col = spec.name
    raw = df[col]

    if spec.dtype == "date":
        parsed: list[Any] = []
        for idx, val in raw.items():
            ts, err = parse_date(val)
            if ts is not None and isinstance(val, str) and re.fullmatch(r"\d{5}", val.strip()):
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.INFO, "EXCEL_SERIAL_DATE",
                    f"{col}: value {val!r} was read as an Excel serial date and "
                    f"converted to {ts.date()}. Verify against the source system.",
                    column=col, row_number=row_no, employee_id=emp, value=val,
                )
            if err:
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.ERROR if spec.required else Severity.WARNING,
                    "INVALID_DATE", f"{col}: {err}. Value set to blank.",
                    column=col, row_number=row_no, employee_id=emp, value=val,
                )
            parsed.append(ts)
        df[col] = pd.Series(parsed, index=df.index, dtype="datetime64[ns]")

    elif spec.dtype in ("float", "int"):
        parsed = []
        for idx, val in raw.items():
            num, err = parse_number(val)
            if err:
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.ERROR if spec.required else Severity.WARNING,
                    "INVALID_NUMBER", f"{col}: {err}. Value set to blank.",
                    column=col, row_number=row_no, employee_id=emp, value=val,
                )
            parsed.append(num)
        df[col] = pd.Series(parsed, index=df.index, dtype="Float64")

    elif spec.dtype == "bool":
        parsed = []
        for idx, val in raw.items():
            b, err = parse_bool(val)
            if err:
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.WARNING, "INVALID_BOOLEAN",
                    f"{col}: {err}. Value set to blank.",
                    column=col, row_number=row_no, employee_id=emp, value=val,
                )
            parsed.append(b)
        df[col] = pd.Series(parsed, index=df.index, dtype="boolean")

    elif spec.dtype == "category":
        # Preserve the source value verbatim for the audit trail.
        df[f"{col}_raw"] = raw.map(clean_text)
        parsed = []
        for idx, val in raw.items():
            norm, err = normalize_category(col, val)
            if err:
                row_no, emp = _row_label(df, idx)
                sev = Severity.WARNING if not spec.protected_class else Severity.WARNING
                report.add(
                    sev, "UNMAPPED_CATEGORY",
                    f"{col}: {err}. Original preserved in {col}_raw; "
                    f"normalized value left blank.",
                    column=col, row_number=row_no, employee_id=emp, value=val,
                )
            parsed.append(norm)
        df[col] = pd.Series(parsed, index=df.index, dtype="string")

    else:  # string
        if col in ("first_name", "last_name"):
            df[col] = raw.map(normalize_name).astype("string")
        elif col == "work_state":
            parsed = []
            for idx, val in raw.items():
                st, err = normalize_state(val)
                if err:
                    row_no, emp = _row_label(df, idx)
                    report.add(
                        Severity.WARNING, "INVALID_STATE",
                        f"work_state: {err}. WARN jurisdiction cannot be "
                        f"determined for this employee.",
                        column=col, row_number=row_no, employee_id=emp, value=val,
                    )
                parsed.append(st)
            df[col] = pd.Series(parsed, index=df.index, dtype="string")
        elif col == "work_email":
            df[col] = raw.map(lambda v: (clean_text(v) or "").lower() or None).astype("string")
        else:
            df[col] = raw.map(clean_text).astype("string")


def _validate_identity(df: pd.DataFrame, report: ValidationReport) -> None:
    """Employee ID presence and uniqueness — the join key for every module."""
    if "employee_id" not in df.columns:
        return
    ids = df["employee_id"]

    for idx, val in ids.items():
        if pd.isna(val) or not str(val).strip():
            row_no, _ = _row_label(df, idx)
            report.add(
                Severity.ERROR, "MISSING_EMPLOYEE_ID",
                "employee_id is blank; the row cannot be tracked through "
                "selection, notice, or payroll.",
                column="employee_id", row_number=row_no,
            )

    dupes = ids[ids.notna() & ids.duplicated(keep=False)]
    for emp_id, group in dupes.groupby(dupes):
        rows = [int(df.at[i, "row_number"]) for i in group.index]
        # One issue per affected row so every duplicate is quarantined, not just
        # the first occurrence.
        for row_no in rows:
            report.add(
                Severity.ERROR, "DUPLICATE_EMPLOYEE_ID",
                f"employee_id {emp_id!r} appears {len(rows)} times (rows {rows}). "
                f"Headcount and adverse impact math will be wrong until resolved.",
                column="employee_id", row_number=row_no, employee_id=str(emp_id),
            )

    # Same person, different ID: name + birth_date collision.
    if {"first_name", "last_name", "birth_date"} <= set(df.columns):
        key = (
            df["first_name"].fillna("").str.lower() + "|"
            + df["last_name"].fillna("").str.lower() + "|"
            + df["birth_date"].astype("string").fillna("")
        )
        mask = key.duplicated(keep=False) & df["last_name"].notna() & df["birth_date"].notna()
        for k, group in df[mask].groupby(key[mask]):
            rows = [int(r) for r in group["row_number"]]
            report.add(
                Severity.WARNING, "POSSIBLE_DUPLICATE_PERSON",
                f"Rows {rows} share the same name and birth date under different "
                f"employee IDs. Confirm these are not duplicate records.",
                row_number=rows[0],
            )


def _validate_required_values(df: pd.DataFrame, report: ValidationReport) -> None:
    for spec in SCHEMA:
        if not spec.required or spec.name not in df.columns:
            continue
        if spec.name == "employee_id":
            continue  # handled in _validate_identity
        missing = df[spec.name].isna()
        for idx in df.index[missing]:
            row_no, emp = _row_label(df, idx)
            report.add(
                Severity.ERROR, "MISSING_REQUIRED_VALUE",
                f"{spec.name} is required but blank.",
                column=spec.name, row_number=row_no, employee_id=emp,
            )


def _validate_dates(df: pd.DataFrame, report: ValidationReport, cfg: IngestConfig) -> None:
    as_of = pd.Timestamp(cfg.as_of_date)
    floor = pd.Timestamp("1940-01-01")

    def each(col: str) -> Iterable[tuple[Any, pd.Timestamp, int, str | None]]:
        if col not in df.columns:
            return []
        out = []
        for idx, val in df[col].items():
            if pd.notna(val):
                row_no, emp = _row_label(df, idx)
                out.append((idx, val, row_no, emp))
        return out

    for idx, val, row_no, emp in each("hire_date"):
        if val > as_of:
            report.add(
                Severity.WARNING, "FUTURE_HIRE_DATE",
                f"hire_date {val.date()} is after the as-of date {as_of.date()}; "
                f"tenure will be zero. Confirm the employee has actually started.",
                column="hire_date", row_number=row_no, employee_id=emp, value=str(val.date()),
            )
        elif val < floor:
            report.add(
                Severity.ERROR, "IMPLAUSIBLE_HIRE_DATE",
                f"hire_date {val.date()} is implausibly early; likely a parsing "
                f"or data entry error.",
                column="hire_date", row_number=row_no, employee_id=emp, value=str(val.date()),
            )

    for idx, val, row_no, emp in each("rehire_date"):
        hire = df.at[idx, "hire_date"] if "hire_date" in df.columns else pd.NaT
        if pd.notna(hire) and val < hire:
            report.add(
                Severity.ERROR, "REHIRE_BEFORE_HIRE",
                f"rehire_date {val.date()} precedes hire_date {hire.date()}.",
                column="rehire_date", row_number=row_no, employee_id=emp,
            )

    for idx, val, row_no, emp in each("termination_date"):
        hire = df.at[idx, "hire_date"] if "hire_date" in df.columns else pd.NaT
        if pd.notna(hire) and val < hire:
            report.add(
                Severity.ERROR, "TERM_BEFORE_HIRE",
                f"termination_date {val.date()} precedes hire_date {hire.date()}.",
                column="termination_date", row_number=row_no, employee_id=emp,
            )
        if val <= as_of:
            report.add(
                Severity.WARNING, "ALREADY_TERMINATED",
                f"termination_date {val.date()} is on or before the as-of date; "
                f"employee excluded from active headcount.",
                column="termination_date", row_number=row_no, employee_id=emp,
            )

    for idx, val, row_no, emp in each("birth_date"):
        age = (as_of - val).days / DAYS_PER_YEAR
        if age < 14 or age > 100:
            report.add(
                Severity.WARNING, "IMPLAUSIBLE_BIRTH_DATE",
                f"birth_date {val.date()} implies an age of {age:.0f}. "
                f"Age-40+ analysis for this employee is unreliable.",
                column="birth_date", row_number=row_no, employee_id=emp,
            )
        hire = df.at[idx, "hire_date"] if "hire_date" in df.columns else pd.NaT
        if pd.notna(hire) and (hire - val).days / DAYS_PER_YEAR < 14:
            report.add(
                Severity.WARNING, "HIRED_BEFORE_AGE_14",
                "hire_date implies the employee was under 14 at hire; check "
                "whether birth_date or hire_date is wrong.",
                column="birth_date", row_number=row_no, employee_id=emp,
            )


def _validate_pay(df: pd.DataFrame, report: ValidationReport, cfg: IngestConfig) -> None:
    if "pay_rate" not in df.columns:
        return
    for idx, val in df["pay_rate"].items():
        if pd.isna(val):
            continue
        row_no, emp = _row_label(df, idx)
        pay_type = df.at[idx, "pay_type"] if "pay_type" in df.columns else None
        if val <= 0:
            report.add(
                Severity.ERROR, "NONPOSITIVE_PAY",
                f"pay_rate {val} is not a positive number; severance and final "
                f"pay cannot be calculated.",
                column="pay_rate", row_number=row_no, employee_id=emp, value=float(val),
            )
            continue
        # Detect pay_type mislabeling: an "hourly" rate of 95,000 is a salary.
        if pay_type == "hourly" and val > 500:
            report.add(
                Severity.WARNING, "PAY_TYPE_MISMATCH",
                f"pay_type is hourly but pay_rate is {val:,.2f}; the value looks "
                f"like an annual salary. Confirm before computing severance.",
                column="pay_rate", row_number=row_no, employee_id=emp, value=float(val),
            )
        elif pay_type == "salary" and val < 1000:
            report.add(
                Severity.WARNING, "PAY_TYPE_MISMATCH",
                f"pay_type is salary but pay_rate is {val:,.2f}; the value looks "
                f"like an hourly rate.",
                column="pay_rate", row_number=row_no, employee_id=emp, value=float(val),
            )
        elif pay_type == "hourly" and val < cfg.min_hourly_wage:
            report.add(
                Severity.WARNING, "BELOW_MINIMUM_WAGE",
                f"Hourly rate {val:,.2f} is below the {cfg.min_hourly_wage:,.2f} "
                f"California minimum wage. Verify the rate and check for wage "
                f"exposure before separation.",
                column="pay_rate", row_number=row_no, employee_id=emp, value=float(val),
            )

    # Exempt classification sanity check against the CA salary floor.
    if {"flsa_status", "annualized_pay"} <= set(df.columns):
        mask = (df["flsa_status"] == "exempt") & df["annualized_pay"].notna()
        for idx in df.index[mask]:
            ann = float(df.at[idx, "annualized_pay"])
            emp_type = df.at[idx, "employment_type"] if "employment_type" in df.columns else None
            if emp_type == "part_time":
                continue  # part-time exempt math differs; skip rather than mislead
            if ann < cfg.exempt_annual_floor:
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.WARNING, "EXEMPT_BELOW_SALARY_FLOOR",
                    f"Classified exempt at {ann:,.0f}/yr, below the "
                    f"{cfg.exempt_annual_floor:,.0f} California exempt salary "
                    f"threshold. Possible misclassification — route to counsel.",
                    column="flsa_status", row_number=row_no, employee_id=emp,
                )

    if "accrued_vacation_hours" in df.columns:
        neg = df["accrued_vacation_hours"].notna() & (df["accrued_vacation_hours"] < 0)
        for idx in df.index[neg]:
            row_no, emp = _row_label(df, idx)
            report.add(
                Severity.WARNING, "NEGATIVE_VACATION_BALANCE",
                "accrued_vacation_hours is negative. CA Labor Code 227.3 requires "
                "payout of accrued vacation; a negative balance needs review.",
                column="accrued_vacation_hours", row_number=row_no, employee_id=emp,
            )
        big = df["accrued_vacation_hours"].notna() & (df["accrued_vacation_hours"] > 1000)
        for idx in df.index[big]:
            row_no, emp = _row_label(df, idx)
            report.add(
                Severity.WARNING, "IMPLAUSIBLE_VACATION_BALANCE",
                f"accrued_vacation_hours of {df.at[idx, 'accrued_vacation_hours']:.0f} "
                f"exceeds 1,000; confirm units are hours, not dollars.",
                column="accrued_vacation_hours", row_number=row_no, employee_id=emp,
            )

    if "fte" in df.columns:
        for idx, val in df["fte"].items():
            if pd.isna(val):
                continue
            if val > 1.5 or val <= 0:
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.WARNING, "IMPLAUSIBLE_FTE",
                    f"fte value {val} is outside the expected 0–1.5 range "
                    f"(percentages like 100 are converted to 1.0 automatically).",
                    column="fte", row_number=row_no, employee_id=emp, value=float(val),
                )


def _validate_misc(df: pd.DataFrame, report: ValidationReport, cfg: IngestConfig) -> None:
    if "work_email" in df.columns:
        for idx, val in df["work_email"].items():
            if isinstance(val, str) and val and not EMAIL_RE.match(val):
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.WARNING, "INVALID_EMAIL",
                    f"work_email {val!r} is not a valid address; notice delivery "
                    f"may fail.",
                    column="work_email", row_number=row_no, employee_id=emp, value=val,
                )

    if "manager_id" in df.columns and "employee_id" in df.columns:
        known = set(df["employee_id"].dropna().astype(str))
        for idx, val in df["manager_id"].items():
            if not isinstance(val, str) or not val:
                continue
            if val not in known:
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.INFO, "MANAGER_NOT_IN_ROSTER",
                    f"manager_id {val!r} does not appear as an employee_id in this "
                    f"file; org rollups may be incomplete.",
                    column="manager_id", row_number=row_no, employee_id=emp, value=val,
                )
            own_id = df.at[idx, "employee_id"]
            if isinstance(own_id, str) and val == own_id:
                row_no, emp = _row_label(df, idx)
                report.add(
                    Severity.WARNING, "SELF_REPORTING_MANAGER",
                    "manager_id equals employee_id.",
                    column="manager_id", row_number=row_no, employee_id=emp,
                )

    if "work_state" in df.columns and cfg.expected_states:
        outside = df["work_state"].notna() & ~df["work_state"].isin(cfg.expected_states)
        if outside.any():
            states = sorted(set(df.loc[outside, "work_state"].dropna()))
            report.add(
                Severity.WARNING, "OUT_OF_STATE_EMPLOYEES",
                f"{int(outside.sum())} employee(s) work outside "
                f"{'/'.join(cfg.expected_states)} ({', '.join(states)}). Other "
                f"state WARN acts and final-pay rules may apply.",
                column="work_state",
            )

    if "leave_status" in df.columns:
        on_leave = df["leave_status"].notna()
        if on_leave.any():
            report.add(
                Severity.WARNING, "EMPLOYEES_ON_LEAVE",
                f"{int(on_leave.sum())} employee(s) have a leave status recorded. "
                f"Separating an employee on protected leave requires legal review.",
                column="leave_status",
            )

    if "union_flag" in df.columns:
        union = df["union_flag"].fillna(False).astype(bool)
        if union.any():
            report.add(
                Severity.WARNING, "UNION_EMPLOYEES_PRESENT",
                f"{int(union.sum())} employee(s) are in a bargaining unit. Check "
                f"the CBA for notice, bumping, and effects-bargaining obligations.",
                column="union_flag",
            )

    if "visa_status" in df.columns:
        visa = df["visa_status"].notna()
        if visa.any():
            report.add(
                Severity.INFO, "VISA_HOLDERS_PRESENT",
                f"{int(visa.sum())} employee(s) have a work-authorization status "
                f"recorded; sponsored employees need immigration counsel input.",
                column="visa_status",
            )


def _validate_coverage(df: pd.DataFrame, report: ValidationReport) -> None:
    """Column-level completeness, with hard flags on protected-class fields."""
    n = len(df)
    if n == 0:
        return
    for spec in SCHEMA:
        if spec.name not in df.columns:
            report.completeness[spec.name] = 0.0
            continue
        col = df[spec.name]
        populated = float(col.notna().sum()) / n
        report.completeness[spec.name] = populated

        if spec.protected_class and populated < MIN_PROTECTED_CLASS_COVERAGE:
            report.add(
                Severity.WARNING, "LOW_PROTECTED_CLASS_COVERAGE",
                f"{spec.name} is populated for only {populated:.1%} of employees. "
                f"Adverse impact testing on this class will be unreliable; "
                f"consider sourcing the field from the HRIS before proceeding.",
                column=spec.name,
            )
        elif spec.compliance_critical and not spec.sparse_ok and populated < 1.0:
            missing = n - int(col.notna().sum())
            if missing:
                report.add(
                    Severity.WARNING, "INCOMPLETE_COMPLIANCE_FIELD",
                    f"{spec.name} is blank for {missing} employee(s); "
                    f"{spec.description or 'this field feeds compliance calculations'}",
                    column=spec.name,
                )

    # not_disclosed is technically populated but useless for impact testing.
    for spec in SCHEMA:
        if not spec.protected_class or spec.name not in df.columns:
            continue
        if spec.dtype != "category":
            continue
        nd = float((df[spec.name] == "not_disclosed").sum()) / n
        if nd > 0.20:
            report.add(
                Severity.WARNING, "HIGH_NOT_DISCLOSED_RATE",
                f"{nd:.1%} of {spec.name} values are 'not_disclosed'. Statistical "
                f"significance testing will have reduced power for this class.",
                column=spec.name,
            )


# ---------------------------------------------------------------------------
# Derived fields
# ---------------------------------------------------------------------------


def _years_between(start: pd.Series, end: pd.Timestamp) -> pd.Series:
    days = (end - start).dt.days
    return (days / DAYS_PER_YEAR).astype("Float64")


def tenure_band(years: float | None) -> str | None:
    if years is None or pd.isna(years):
        return None
    if years < 1:
        return "<1 year"
    if years < 3:
        return "1 to <3 years"
    if years < 5:
        return "3 to <5 years"
    if years < 10:
        return "5 to <10 years"
    if years < 20:
        return "10 to <20 years"
    return "20+ years"


def age_band(years: float | None) -> str | None:
    if years is None or pd.isna(years):
        return None
    if years < 30:
        return "Under 30"
    if years < 40:
        return "30-39"
    if years < 50:
        return "40-49"
    if years < 60:
        return "50-59"
    return "60+"


def _add_derived(df: pd.DataFrame, report: ValidationReport, cfg: IngestConfig) -> None:
    as_of = pd.Timestamp(cfg.as_of_date)

    # -- names ------------------------------------------------------------
    first = df["first_name"] if "first_name" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    last = df["last_name"] if "last_name" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")
    df["full_name"] = (
        first.fillna("").astype(str).str.strip() + " " + last.fillna("").astype(str).str.strip()
    ).str.strip().replace("", pd.NA).astype("string")

    # -- service start / tenure -------------------------------------------
    hire = df["hire_date"] if "hire_date" in df.columns else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    rehire = df["rehire_date"] if "rehire_date" in df.columns else pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    if cfg.use_adjusted_service_date:
        service = rehire.where(rehire.notna() & (rehire >= hire), hire)
    else:
        service = hire
    df["service_start_date"] = service

    days = (as_of - service).dt.days
    days = days.where(days.notna() & (days >= 0))
    df["tenure_days"] = days.astype("Int64")
    df["tenure_years"] = (days / DAYS_PER_YEAR).round(2).astype("Float64")
    df["tenure_months"] = (days / (DAYS_PER_YEAR / 12)).round(1).astype("Float64")
    df["tenure_band"] = df["tenure_years"].map(tenure_band).astype("string")

    # -- age / age 40+ (ADEA & FEHA protected class) -----------------------
    if "birth_date" in df.columns:
        age_days = (as_of - df["birth_date"]).dt.days
        df["age_years"] = (age_days / DAYS_PER_YEAR).round(1).astype("Float64")
    else:
        df["age_years"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
    df["age_band"] = df["age_years"].map(age_band).astype("string")
    df["age_40_plus"] = pd.Series(
        [pd.NA if pd.isna(a) else bool(a >= 40) for a in df["age_years"]],
        index=df.index, dtype="boolean",
    )

    # -- pay annualization -------------------------------------------------
    hours = (
        df["standard_hours_per_week"]
        if "standard_hours_per_week" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="Float64")
    )
    fte = df["fte"] if "fte" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Float64")
    # FTE expressed as a percentage (100, 80) -> fraction.
    fte = fte.map(lambda v: v / 100.0 if (pd.notna(v) and v > 1.5) else v).astype("Float64")
    if "fte" in df.columns:
        df["fte"] = fte

    annual: list[Any] = []
    hourly_eq: list[Any] = []
    for idx in df.index:
        rate = df.at[idx, "pay_rate"] if "pay_rate" in df.columns else pd.NA
        ptype = df.at[idx, "pay_type"] if "pay_type" in df.columns else pd.NA
        pfreq = df.at[idx, "pay_frequency"] if "pay_frequency" in df.columns else pd.NA
        hrs = hours.at[idx]
        if pd.isna(hrs):
            f = fte.at[idx]
            hrs = cfg.default_hours_per_week * (float(f) if pd.notna(f) else 1.0)
        hrs = float(hrs) if pd.notna(hrs) and hrs > 0 else cfg.default_hours_per_week

        if pd.isna(rate):
            annual.append(pd.NA)
            hourly_eq.append(pd.NA)
            continue
        rate = float(rate)

        # Infer basis: explicit pay_type wins, then pay_frequency, then magnitude.
        basis = None
        if isinstance(ptype, str) and ptype in ("hourly", "salary"):
            basis = ptype
        elif isinstance(pfreq, str) and pfreq == "hourly":
            basis = "hourly"
        elif isinstance(pfreq, str) and pfreq in PAY_FREQUENCY_MULTIPLIER:
            mult = PAY_FREQUENCY_MULTIPLIER[pfreq]
            if mult:
                ann = rate * mult
                annual.append(round(ann, 2))
                hourly_eq.append(round(ann / (hrs * WEEKS_PER_YEAR), 2))
                continue
        if basis is None:
            basis = "hourly" if rate < 500 else "salary"
            row_no, emp = _row_label(df, idx)
            report.add(
                Severity.INFO, "PAY_BASIS_INFERRED",
                f"pay_type was blank; inferred {basis!r} from the rate magnitude "
                f"({rate:,.2f}). Confirm before running severance math.",
                column="pay_type", row_number=row_no, employee_id=emp,
            )
        elif basis == "hourly" and rate > 500:
            # A declared "hourly" rate in the thousands is a mislabeled salary.
            # Annualizing it literally would produce an absurd figure and poison
            # every downstream cost model, so override and log the correction.
            basis = "salary"
            row_no, emp = _row_label(df, idx)
            report.add(
                Severity.INFO, "PAY_BASIS_OVERRIDDEN",
                f"pay_type was 'hourly' but the rate is {rate:,.2f}; annualized "
                f"pay was computed on a salary basis instead. See the matching "
                f"PAY_TYPE_MISMATCH warning and correct the source record.",
                column="pay_type", row_number=row_no, employee_id=emp,
            )
        elif basis == "salary" and 0 < rate < 500:
            basis = "hourly"
            row_no, emp = _row_label(df, idx)
            report.add(
                Severity.INFO, "PAY_BASIS_OVERRIDDEN",
                f"pay_type was 'salary' but the rate is {rate:,.2f}; annualized "
                f"pay was computed on an hourly basis instead. See the matching "
                f"PAY_TYPE_MISMATCH warning and correct the source record.",
                column="pay_type", row_number=row_no, employee_id=emp,
            )

        if basis == "hourly":
            ann = rate * hrs * WEEKS_PER_YEAR
            annual.append(round(ann, 2))
            hourly_eq.append(round(rate, 2))
        else:
            annual.append(round(rate, 2))
            hourly_eq.append(round(rate / (hrs * WEEKS_PER_YEAR), 2))

    df["annualized_pay"] = pd.Series(annual, index=df.index, dtype="Float64")
    df["hourly_equivalent_rate"] = pd.Series(hourly_eq, index=df.index, dtype="Float64")

    # -- active flag -------------------------------------------------------
    if "termination_date" in df.columns:
        term = df["termination_date"]
        df["is_active"] = pd.Series(
            [(pd.isna(t) or pd.Timestamp(t) > as_of) for t in term],
            index=df.index, dtype="boolean",
        )
    else:
        df["is_active"] = pd.Series(True, index=df.index, dtype="boolean")


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def _read_csv(path: str | Path, report: ValidationReport, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV defensively: encoding fallbacks, everything as string."""
    path = Path(path)
    last_err: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                dtype=str,
                keep_default_na=True,
                na_values=sorted(NULL_TOKENS - {""}),
                skipinitialspace=True,
                encoding=encoding,
                **kwargs,
            )
            if encoding not in ("utf-8-sig", "utf-8"):
                report.add(
                    Severity.INFO, "ENCODING_FALLBACK",
                    f"File decoded using {encoding} after UTF-8 failed. Check for "
                    f"garbled characters in names.",
                )
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:  # malformed CSV structure
            raise ValueError(f"Could not parse {path.name}: {e}") from e
    raise ValueError(f"Could not decode {path.name}: {last_err}")


def load_workforce_dataframe(
    raw: pd.DataFrame,
    *,
    as_of: str | dt.date | None = None,
    config: IngestConfig | None = None,
    source: str = "",
) -> IngestResult:
    """Normalize and validate an already-loaded roster DataFrame."""
    cfg = config or IngestConfig()
    if as_of is not None:
        cfg.as_of_date = as_of if isinstance(as_of, dt.date) else pd.Timestamp(as_of).date()

    report = ValidationReport(source=source, as_of_date=str(cfg.as_of_date))

    raw = raw.copy()
    raw.columns = [str(c) for c in raw.columns]

    # Drop fully empty rows and columns (common in Excel exports).
    before = len(raw)
    raw = raw.dropna(how="all")
    if len(raw) < before:
        report.add(
            Severity.INFO, "BLANK_ROWS_DROPPED",
            f"Removed {before - len(raw)} completely empty row(s).",
        )
    raw = raw.loc[:, [c for c in raw.columns if not (c.startswith("Unnamed:") and raw[c].isna().all())]]

    if raw.empty:
        report.add(Severity.ERROR, "EMPTY_FILE", "The file contains no data rows.")
        report.row_count = 0
        return IngestResult(pd.DataFrame(), report, cfg)

    # Preserve original row numbers (1-based, header = row 1).
    raw = raw.reset_index(drop=True)
    row_numbers = pd.Series(raw.index + 2, index=raw.index)

    # -- header mapping ---------------------------------------------------
    mapping, unmapped, fuzzy = map_columns(list(raw.columns), cfg.header_match_cutoff)
    report.column_map = mapping
    report.unmapped_columns = unmapped
    report.column_count = len(raw.columns)

    for src, canon, matched_alias in fuzzy:
        report.add(
            Severity.INFO, "FUZZY_COLUMN_MATCH",
            f"Source column {src!r} was matched to {canon!r} by similarity "
            f"(closest known alias: {matched_alias!r}). Verify the mapping.",
            column=canon,
        )

    df = pd.DataFrame(index=raw.index)
    df["row_number"] = row_numbers
    for src, canon in mapping.items():
        df[canon] = raw[src]

    missing_required = [f.name for f in SCHEMA if f.required and f.name not in df.columns]
    report.missing_required_columns = missing_required
    for name in missing_required:
        spec = SCHEMA_BY_NAME[name]
        report.add(
            Severity.ERROR, "MISSING_REQUIRED_COLUMN",
            f"Required column {name!r} was not found. Expected one of: "
            f"{', '.join((name,) + spec.aliases[:6])}.",
            column=name,
        )
        df[name] = pd.NA  # keep the schema shape so downstream code is stable

    for name in unmapped:
        report.add(
            Severity.INFO, "UNMAPPED_COLUMN",
            f"Source column {name!r} did not match the schema and was "
            f"{'carried through as x_' + _norm_key(name).replace(' ', '_') if cfg.keep_extra_columns else 'dropped'}.",
            column=name,
        )

    # -- normalization ----------------------------------------------------
    for spec in SCHEMA:
        if spec.name in df.columns:
            _normalize_column(df, spec, report, cfg)

    # -- validation & derivation -----------------------------------------
    _validate_identity(df, report)
    _validate_required_values(df, report)
    _validate_dates(df, report, cfg)
    _add_derived(df, report, cfg)
    _validate_pay(df, report, cfg)
    _validate_misc(df, report, cfg)
    _validate_coverage(df, report)

    # -- row-level blocking flag ------------------------------------------
    blocking_codes = {
        "MISSING_EMPLOYEE_ID", "DUPLICATE_EMPLOYEE_ID", "MISSING_REQUIRED_VALUE",
        "INVALID_DATE", "NONPOSITIVE_PAY", "TERM_BEFORE_HIRE",
        "REHIRE_BEFORE_HIRE", "IMPLAUSIBLE_HIRE_DATE",
    }
    bad_rows = {
        i.row_number for i in report.issues
        if i.severity == Severity.ERROR and i.code in blocking_codes and i.row_number is not None
    }
    df["has_blocking_error"] = df["row_number"].isin(bad_rows)

    # -- passthrough columns ----------------------------------------------
    if cfg.keep_extra_columns:
        for name in unmapped:
            df[f"x_{_norm_key(name).replace(' ', '_')}"] = raw[name].map(clean_text).astype("string")

    # -- column ordering ---------------------------------------------------
    ordered = ["row_number", "employee_id", "full_name"]
    ordered += [f.name for f in SCHEMA if f.name not in ordered and f.name in df.columns]
    ordered += [c for c in DERIVED_COLUMNS if c not in ordered and c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    df = df[ordered]

    report.row_count = len(df)

    if cfg.drop_error_rows and df["has_blocking_error"].any():
        n = int(df["has_blocking_error"].sum())
        df = df.loc[~df["has_blocking_error"]].copy()
        report.add(
            Severity.INFO, "ERROR_ROWS_DROPPED",
            f"{n} row(s) with blocking errors were excluded from the standardized "
            f"dataset per configuration. They remain listed in this report.",
        )

    return IngestResult(df.reset_index(drop=True), report, cfg)


def load_workforce_csv(
    path: str | Path,
    *,
    as_of: str | dt.date | None = None,
    config: IngestConfig | None = None,
    **read_csv_kwargs: Any,
) -> IngestResult:
    """Load, normalize, and validate a workforce CSV file.

    Parameters
    ----------
    path : str | Path
        Path to the CSV export from the HRIS, payroll, or manual template.
    as_of : str | date, optional
        Date used for tenure and age calculations. Use the planned RIF
        notification or separation date so tenure-based severance is accurate.
        Defaults to today.
    config : IngestConfig, optional
        Overrides for wage floors, service-date handling, and row dropping.

    Returns
    -------
    IngestResult with `.data` (standardized DataFrame) and `.report`.
    """
    cfg = config or IngestConfig()
    if as_of is not None:
        cfg.as_of_date = as_of if isinstance(as_of, dt.date) else pd.Timestamp(as_of).date()

    report_stub = ValidationReport(source=str(path), as_of_date=str(cfg.as_of_date))
    try:
        raw = _read_csv(path, report_stub, **read_csv_kwargs)
    except (ValueError, FileNotFoundError) as e:
        report_stub.add(Severity.ERROR, "PARSE_FAILURE", str(e))
        return IngestResult(pd.DataFrame(), report_stub, cfg)

    result = load_workforce_dataframe(raw, config=cfg, source=str(path))
    # Preserve any read-time info issues (encoding fallback, etc.).
    result.report.issues = report_stub.issues + result.report.issues
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import and validate a workforce CSV for RIF analysis."
    )
    parser.add_argument("csv_path", help="Path to the workforce CSV file.")
    parser.add_argument("--as-of", default=None,
                        help="Date for tenure/age math (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--outdir", default=None,
                        help="Directory to write standardized data and report files.")
    parser.add_argument("--drop-error-rows", action="store_true",
                        help="Exclude rows with blocking errors from the output data.")
    parser.add_argument("--strict-headers", action="store_true",
                        help="Disable fuzzy header matching.")
    parser.add_argument("--quiet", action="store_true", help="Suppress the report printout.")
    args = parser.parse_args(argv)

    cfg = IngestConfig(
        drop_error_rows=args.drop_error_rows,
        header_match_cutoff=1.0 if args.strict_headers else IngestConfig.header_match_cutoff,
    )
    result = load_workforce_csv(args.csv_path, as_of=args.as_of, config=cfg)

    if not args.quiet:
        print(result.report.to_markdown())

    if args.outdir:
        paths = result.write(args.outdir)
        print("\nWrote:")
        for label, p in paths.items():
            print(f"  {label}: {p}")

    if result.report.is_blocking:
        return 2
    return 1 if result.report.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
