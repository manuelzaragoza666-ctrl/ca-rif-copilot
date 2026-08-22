# Data Manager — California RIF Copilot (Module 1)

Ingests workforce data from CSV, maps messy headers onto a canonical schema,
normalizes and validates every record, derives tenure/age/pay fields, and emits
a standardized DataFrame plus a structured validation report.

Everything downstream — the Selection Criteria Engine, Adverse Impact Analyzer,
CA Compliance Engine, and Severance & Pay Engine — reads the output of this
module, so the validation is deliberately strict about the fields those modules
depend on.

## Files

| File | Purpose |
|---|---|
| `workforce_data.py` | The module. Zero dependencies beyond pandas/numpy. |
| `test_workforce_data.py` | 28 tests covering parsers, mapping, validation, derived fields. |
| `sample_roster.csv` | Deliberately messy 12-row roster for exercising the error paths. |

## Quick start

```python
from workforce_data import load_workforce_csv

# Use the planned separation date so tenure-based severance is accurate.
result = load_workforce_csv("roster.csv", as_of="2026-10-30")

result.data                  # standardized DataFrame (all rows)
result.clean                 # rows with no blocking error — safe for analysis
result.report.summary()      # dict of counts
result.report.to_dataframe() # one row per finding
print(result.report.to_markdown())

result.write("./out")        # CSV + issues CSV + JSON + Markdown report
```

CLI:

```bash
python workforce_data.py roster.csv --as-of 2026-10-30 --outdir ./out
# exit 0 = clean, 1 = errors present, 2 = blocking (unusable file)
```

## What it does

**Header mapping.** Each canonical field carries a list of real-world aliases
(`Emp ID`, `Worker Number`, `EEID` → `employee_id`). Exact normalized matches
run first; a fuzzy pass catches the rest and logs every fuzzy match as an INFO
finding so a human can verify it. One canonical field is never claimed twice.
Unmatched source columns are carried through as `x_*` rather than dropped.

**Normalization.** Whitespace and zero-width characters, name casing, emails,
dates (13 formats plus Excel serial numbers), currency (`$168,500`, `85k`,
`(1,200)`, `42.50 /hr`), percentages, booleans, and categorical vocabularies
(`F/T` → `full_time`, `Non-Exempt` → `non_exempt`, `Declined to State` →
`not_disclosed`).

**Nothing is silently dropped.** Bad values are coerced to NA *and* recorded as
an issue with the source row number, so the audit trail can always explain what
changed. For protected-class fields the original value is preserved in a
`*_raw` column and unrecognized categories are never force-fit into an existing
bucket.

**Derived fields.** `service_start_date` (honors adjusted/rehire dates,
configurable), `tenure_days` / `tenure_years` / `tenure_months` / `tenure_band`,
`age_years` / `age_band` / `age_40_plus` (ADEA and FEHA), `annualized_pay` and
`hourly_equivalent_rate` (from pay type, frequency, and scheduled hours),
`is_active`, `full_name`.

## Validation rules

**Errors** (block the row, or the whole file):
missing required column, missing or duplicate `employee_id`, missing required
value, unparseable required date, `termination_date` before `hire_date`,
`rehire_date` before `hire_date`, implausible hire date, non-positive pay.

**Warnings** (analysis proceeds, human reviews):
future hire date, already-terminated employees, implausible birth date,
possible duplicate person (same name + DOB, different ID), hourly rate below
the CA minimum wage, exempt classification below the CA salary floor, pay-type
mismatch, negative or implausible vacation balance, invalid email, unrecognized
state, out-of-state employees, employees on leave, union members, low
protected-class coverage, high `not_disclosed` rate, out-of-range FTE.

**Info** (audit trail):
encoding fallback, fuzzy column match, Excel serial date conversion, pay basis
inferred or overridden, manager not present in roster, unmapped columns.

## Design decisions worth knowing

- **`row_number` is the source CSV line number** (header = line 1), so every
  finding points at a row the HR analyst can actually open and fix.
- **Blocking errors quarantine rows rather than deleting them.** `result.data`
  keeps everything with a `has_blocking_error` flag; `result.clean` is the
  filtered view. Set `drop_error_rows=True` to exclude them from the output
  entirely — the findings still appear in the report.
- **Both sides of a duplicate ID are quarantined**, not just the second one,
  since there's no way to know which record is authoritative.
- **A pay basis that contradicts the value is overridden, not obeyed.** An
  "hourly" rate of 215,000 is annualized as a salary rather than emitting
  $447M and poisoning every downstream cost model. The mismatch is still
  flagged for correction at the source.
- **Low protected-class coverage is a warning, not a silent gap.** Adverse
  impact testing on a field that's 60% populated will produce confident-looking
  and meaningless results, so the report says so before anyone runs it.
- **Sparse-by-design fields** (`leave_status`, `visa_status`, `union_flag`,
  `termination_date`, `rehire_date`) are exempt from completeness warnings.

## Configuration

```python
from workforce_data import IngestConfig, load_workforce_csv
import datetime as dt

cfg = IngestConfig(
    as_of_date=dt.date(2026, 10, 30),
    use_adjusted_service_date=True,   # False = continuous service from original hire
    min_hourly_wage=16.90,            # update annually; local ordinances may be higher
    exempt_annual_floor=70_304.0,     # 2x CA minimum wage, full time
    default_hours_per_week=40.0,
    drop_error_rows=False,
    keep_extra_columns=True,
    header_match_cutoff=0.86,         # 1.0 disables fuzzy header matching
    expected_states=("CA",),
)
result = load_workforce_csv("roster.csv", config=cfg)
```

## Things to update before production

- `CA_MIN_HOURLY_WAGE` and `CA_EXEMPT_ANNUAL_FLOOR` are 2026 statewide defaults
  and need an annual review; many CA cities set higher local minimums, so a
  city-level wage table keyed on `work_city` would be more accurate than the
  single statewide constant.
- The computed exempt-salary floor is a screening heuristic. Exempt status also
  turns on duties tests this module can't evaluate, so
  `EXEMPT_BELOW_SALARY_FLOOR` should route to counsel rather than auto-classify.
- Consider adding EEO-1 job category as a schema field if the adverse impact
  analysis needs to group by it.
- The CSV reader is the only ingest path today. Direct HRIS connectors
  (Workday, BambooHR, ADP) should produce a DataFrame and call
  `load_workforce_dataframe()` so they inherit the same validation.

## Scope

This module validates data quality only. It makes no selection decisions and
renders no legal conclusions — its job is to make sure the humans and modules
downstream are working from data they can trust, and to say plainly where they
can't.
