"""
make_test_roster.py
===================

Generates a synthetic California workforce roster for exercising modules 1 and 2.

The data is random but the *edge cases are deliberate*. Each one is seeded on
purpose so you can check that the pipeline reacts the way it should, and each
is listed in the manifest printed at the end.

    python make_test_roster.py --employees 140 --seed 7 --out test_roster.csv

Every value is synthetic. No real person's data is used or implied.
"""

from __future__ import annotations

# Make the package importable whether this file is run directly, via pytest, or
# from another working directory.
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import argparse
import csv
import datetime as dt
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

DEPARTMENTS = {
    "Engineering": {
        "titles": ["Engineer I", "Engineer II", "Senior Engineer", "Staff Engineer",
                   "Engineering Manager"],
        "levels": ["L2", "L3", "L4", "L5", "M3"],
        "skills": ["Python", "Kubernetes", "Distributed Systems", "React", "Go",
                   "SQL", "Terraform", "Mentoring", "Java", "Cobol"],
        "pay": (110_000, 240_000),
        "site": "SF HQ",
    },
    "Customer Success": {
        "titles": ["Support Specialist", "Senior Support Specialist", "CS Manager"],
        "levels": ["L2", "L3", "M2"],
        "skills": ["Zendesk", "SQL", "Spanish", "Onboarding", "Escalation Management"],
        "pay": (58_000, 105_000),
        "site": "Sacramento",
    },
    "Operations": {
        "titles": ["Warehouse Associate", "Operations Coordinator", "Operations Lead",
                   "Operations Manager"],
        "levels": ["L1", "L2", "L3", "M2"],
        "skills": ["Forklift Operation", "Inventory Systems", "Vendor Management",
                   "Lean Six Sigma", "Excel"],
        "pay": (48_000, 130_000),
        "site": "Stockton",
    },
    "Finance": {
        "titles": ["Analyst", "Senior Analyst", "Finance Manager"],
        "levels": ["L2", "L3", "M2"],
        "skills": ["Financial Modeling", "SQL", "Excel", "NetSuite", "Forecasting"],
        "pay": (85_000, 175_000),
        "site": "SF HQ",
    },
    "Sales": {
        "titles": ["Sales Rep", "Senior Sales Rep", "Account Executive", "Sales Director"],
        "levels": ["L2", "L3", "L4", "M3"],
        "skills": ["Salesforce", "Negotiation", "Forecasting", "Team Leadership"],
        "pay": (70_000, 210_000),
        "site": "SF HQ",
    },
    "Marketing": {
        "titles": ["Marketing Associate", "Marketing Manager", "Content Strategist"],
        "levels": ["L2", "L3", "L3"],
        "skills": ["SEO", "Content Strategy", "Adobe Suite", "Marketo", "Analytics"],
        "pay": (72_000, 145_000),
        "site": "SF HQ",
    },
}

RATINGS = ["Exceeds", "Meets", "Meets", "Meets", "Below", "Exceeds", "Far Exceeds"]
CERTS = ["PMP", "CPA", "AWS Solutions Architect", "SHRM-CP", "Six Sigma Green Belt",
         "Forklift Certification", "Google Analytics", "CFA Level II"]
GENDERS = ["F", "M", "M", "F", "Non-binary", "Prefer not to say"]
RACES = ["White", "Asian", "Hispanic or Latino", "Black or African American",
         "Two or More Races", "Declined to State", "White", "Asian"]

FIRST = ["Maria", "James", "Wei", "Andre", "Priya", "Sofia", "David", "Grace",
         "Tom", "Ana", "Ravi", "Chen", "Fatima", "Luis", "Nina", "Omar", "Elena",
         "Kwame", "Yuki", "Diego", "Hannah", "Ibrahim", "Rosa", "Peter", "Aisha",
         "Marco", "Lin", "Jordan", "Amara", "Sean", "Mei", "Carlos", "Ruth",
         "Tariq", "Ingrid", "Hugo", "Leila", "Noah", "Zara", "Felix"]
LAST = ["Garcia", "O'Brien", "Chen", "Johnson", "Patel", "Rossi", "Nguyen",
        "Okafor", "Baker", "Ruiz", "Kim", "Silva", "Haddad", "Moreno", "Novak",
        "Farouk", "Petrova", "Mensah", "Tanaka", "Alvarez", "Weiss", "Diallo",
        "Castillo", "Larsen", "Rahman", "Bianchi", "Zhao", "Reed", "Eze",
        "Murphy", "Wang", "Ortega", "Cohen", "Aziz", "Lindqvist"]


def money(lo: int, hi: int, rng: random.Random) -> int:
    return int(rng.uniform(lo, hi) // 500 * 500)


def build(n: int, seed: int) -> tuple[list[dict], list[str]]:
    rng = random.Random(seed)
    rows: list[dict] = []
    manifest: list[str] = []
    today = dt.date(2026, 10, 30)
    dept_names = list(DEPARTMENTS)

    for i in range(n):
        dept = rng.choices(dept_names, weights=[34, 16, 22, 10, 22, 12])[0]
        cfg = DEPARTMENTS[dept]
        ti = rng.randrange(len(cfg["titles"]))
        title = cfg["titles"][ti]
        level = cfg["levels"][ti]

        age = int(rng.triangular(23, 64, 36))
        birth = today.replace(year=today.year - age) - dt.timedelta(days=rng.randrange(365))
        tenure_days = int(rng.triangular(30, 6500, 900))
        hire = today - dt.timedelta(days=tenure_days)

        lo, hi = cfg["pay"]
        base = money(lo + ti * 8000, hi, rng)
        hourly = title in ("Warehouse Associate", "Support Specialist")

        skills = rng.sample(cfg["skills"], k=rng.randint(2, 4))
        certs = rng.sample(CERTS, k=rng.randint(0, 2))

        rows.append({
            "Emp ID": f"E{2000 + i}",
            "First Name": rng.choice(FIRST),
            "Last Name": rng.choice(LAST),
            "Email Address": f"user{2000+i}@acme.com",
            "Job Title": title,
            "Level": level,
            "Dept": dept,
            "Supervisor ID": f"M{rng.randrange(10, 20)}",
            "Work Location": cfg["site"],
            "City": {"SF HQ": "San Francisco", "Sacramento": "Sacramento",
                     "Stockton": "Stockton"}[cfg["site"]],
            "State": "CA",
            "Worker Type": rng.choices(["FT", "PT", "Temp"], weights=[88, 9, 3])[0],
            "Exempt Status": "Non-Exempt" if hourly else "Exempt",
            "Hire Date": hire.isoformat(),
            "Adjusted Hire Date": "",
            "Term Date": "",
            "DOB": birth.isoformat(),
            "Pay Basis": "Hourly" if hourly else "Salary",
            "Base Salary": f"{round(base/2080, 2)}" if hourly else f"{base}",
            "Pay Period": "Biweekly" if hourly else "Annual",
            "PTO Balance": str(rng.randrange(0, 180)),
            "Hours Per Week": "40",
            "FTE": "100",
            "Gender": rng.choice(GENDERS),
            "Race/Ethnicity": rng.choice(RACES),
            "Veteran": rng.choices(["N", "Y"], weights=[92, 8])[0],
            "Disability": rng.choices(["No", "Yes", "Prefer not to say"],
                                      weights=[85, 8, 7])[0],
            "Last Rating": rng.choice(RATINGS),
            "Union": "Y" if (dept == "Operations" and rng.random() < 0.6) else "N",
            "LOA": "",
            "Skills": "|".join(skills),
            "Certifications": "|".join(certs),
        })

    def find(pred, k=1):
        hits = [r for r in rows if pred(r)]
        rng.shuffle(hits)
        return hits[:k]

    # ---- deliberate edge cases -------------------------------------------
    # 1. Data-quality problems that Module 1 must catch.
    for r in find(lambda r: True, 2):
        r["Hire Date"] = "not a date"
    manifest.append("2 unparseable hire dates -> INVALID_DATE, rows quarantined")

    dup = find(lambda r: True, 1)[0]
    clone = dict(dup)
    clone["Email Address"] = "dupe@acme.com"
    rows.append(clone)
    manifest.append(f"1 duplicated employee_id ({dup['Emp ID']}) -> both rows quarantined")

    for r in find(lambda r: True, 1):
        r["Emp ID"] = ""
    manifest.append("1 blank employee_id -> MISSING_EMPLOYEE_ID")

    for r in find(lambda r: r["Pay Basis"] == "Hourly", 1):
        r["Base Salary"] = "14.00"
    manifest.append("1 hourly rate below CA minimum -> BELOW_MINIMUM_WAGE")

    for r in find(lambda r: r["Exempt Status"] == "Exempt", 1):
        r["Base Salary"] = "62000"
    manifest.append("1 exempt employee under the salary floor -> EXEMPT_BELOW_SALARY_FLOOR")

    for r in find(lambda r: True, 1):
        r["PTO Balance"] = "1400"
    manifest.append("1 implausible PTO balance -> IMPLAUSIBLE_VACATION_BALANCE")

    for r in find(lambda r: True, 1):
        r["Hire Date"] = "45210"
    manifest.append("1 Excel serial hire date -> EXCEL_SERIAL_DATE, converted")

    for r in find(lambda r: True, 1):
        r["State"] = "TX"
        r["City"] = "Austin"
    manifest.append("1 out-of-state employee -> OUT_OF_STATE_EMPLOYEES")

    # 2. Cases Module 2 must handle rather than score around.
    for r in find(lambda r: True, 4):
        r["Last Rating"] = rng.choice(["New", "Not Rated", ""])
    manifest.append("4 unrated employees -> review queue, NOT scored as zero")

    for r in find(lambda r: True, 3):
        r["LOA"] = rng.choice(["CFRA", "FMLA", "Pregnancy Disability Leave"])
    manifest.append("3 employees on protected leave -> legal_review_flags if selected")

    for r in find(lambda r: True, 2):
        r["Visa"] = "H-1B"
    manifest.append("2 visa holders (column added) -> WORK_VISA_HOLDER flag")

    # A comparison group of exactly one, to trip the degenerate-group guard.
    solo = find(lambda r: r["Dept"] == "Finance", 1)[0]
    solo["Job Title"] = "Treasury Specialist"
    solo["Level"] = "L9"
    manifest.append(f"1 sole-incumbent level in Finance ({solo['Emp ID']}, L9) "
                    "-> DEGENERATE_COMPARISON_GROUP")

    # Exact ties at the same level, to trip the boundary-tie guard.
    ties = find(lambda r: r["Dept"] == "Sales" and r["Level"] == "L3", 3)
    for r in ties:
        r["Last Rating"] = "Below"
        r["Base Salary"] = "95000"
        r["Skills"] = "Salesforce"
    if ties:
        manifest.append(f"{len(ties)} identical Sales L3 profiles -> TIE_AT_CUT_BOUNDARY")

    # 3. A rater-bias pattern, so Module 3 has something real to find later.
    lenient = [r for r in rows if r["Supervisor ID"] == "M11"]
    for r in lenient:
        r["Last Rating"] = "Exceeds"
    if lenient:
        manifest.append(
            f"Manager M11 rated all {len(lenient)} reports 'Exceeds' -> rating "
            "inconsistency across managers; their reports are structurally "
            "protected from selection"
        )

    # An age-correlated rating skew. Not detectable by module 2 by design --
    # it is exactly what module 3 exists to surface.
    older = [r for r in rows if r["DOB"] and r["DOB"] < "1976-01-01"]
    skewed = older[: max(1, len(older) // 2)]
    for r in skewed:
        r["Last Rating"] = "Below"
    manifest.append(
        f"{len(skewed)} employees aged 50+ given 'Below' ratings -> a planted "
        "age skew. Module 2 will NOT see it (age is firewalled from scoring); "
        "Module 3 should surface it. If it doesn't, that's a bug."
    )

    for r in rows:
        r.setdefault("Visa", "")
    return rows, manifest


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a synthetic test roster.")
    ap.add_argument("--employees", type=int, default=140)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="test_roster.csv")
    args = ap.parse_args()

    rows, manifest = build(args.employees, args.seed)
    cols = list(rows[0].keys())
    path = Path(args.out)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {path}\n")
    print("Seeded edge cases — each should show up somewhere in the pipeline:")
    for i, m in enumerate(manifest, 1):
        print(f"  {i:2}. {m}")
    print("\nAll data is synthetic.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
