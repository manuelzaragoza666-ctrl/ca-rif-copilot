#!/usr/bin/env python3
"""
Run every test suite and report the totals.

    python run_tests.py            # all suites
    python run_tests.py selection  # only suites matching a substring

Each suite also runs standalone (``python tests/test_workforce_data.py``) and
under pytest if it is installed. No test framework is required.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"


def main(argv: list[str]) -> int:
    pattern = argv[1] if len(argv) > 1 else ""
    files = sorted(p for p in TESTS.glob("test_*.py") if pattern in p.name)
    if not files:
        print(f"No test files match {pattern!r}.")
        return 1

    total = passed = failed_files = 0
    width = max(len(p.name) for p in files) + 2

    for p in files:
        proc = subprocess.run(
            [sys.executable, str(p)], capture_output=True, text=True, cwd=ROOT
        )
        line = next(
            (ln for ln in proc.stdout.splitlines() if ln.strip().endswith("passed")),
            "",
        ).strip()
        if not line:
            print(f"{p.name:<{width}} ERROR")
            print(proc.stdout[-2000:] or proc.stderr[-2000:])
            failed_files += 1
            continue
        n, rest = line.split("/", 1)
        d = rest.split()[0]
        total += int(d)
        passed += int(n)
        status = "" if n == d else "   <-- FAILURES"
        print(f"{p.name:<{width}}{line}{status}")
        if n != d:
            failed_files += 1
            print(proc.stdout[-3000:])

    print()
    print(f"{passed}/{total} unit tests passed across {len(files)} suite(s).")
    if failed_files:
        print(f"{failed_files} suite(s) reported failures.")
        return 1

    print()
    print("Pipeline check:")
    harness = subprocess.run(
        [
            sys.executable, str(ROOT / "tools" / "verify_test_run.py"),
            str(ROOT / "examples" / "test_roster.csv"),
            str(ROOT / "examples" / "plan_test.yaml"),
        ],
        capture_output=True, text=True, cwd=ROOT,
    )
    tail = [ln for ln in harness.stdout.splitlines() if "pipeline checks" in ln]
    print("  " + (tail[-1] if tail else "harness did not report"))
    return 0 if tail and tail[-1].split("/")[0].strip() == tail[-1].split("/")[1].split()[0] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
