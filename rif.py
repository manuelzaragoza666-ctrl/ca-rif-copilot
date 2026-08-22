#!/usr/bin/env python3
"""
Command-line entry point for the California RIF Copilot.

    python rif.py pipeline  roster.csv --plan plan.yaml --separation-date 2026-10-30 \
                            --leave-policy separate --outdir ./out
    python rif.py ingest    roster.csv --as-of 2026-10-30
    python rif.py select    roster.csv --plan plan.yaml
    python rif.py impact    roster.csv --plan plan.yaml
    python rif.py scenarios roster.csv --scenarios scenarios.yaml
    python rif.py compliance roster.csv --plan plan.yaml --separation-date 2026-10-30
    python rif.py severance roster.csv --plan plan.yaml --separation-date 2026-10-30 \
                            --leave-policy separate
    python rif.py documents roster.csv --plan plan.yaml --separation-date 2026-10-30
    python rif.py audit     roster.csv --plan plan.yaml --separation-date 2026-10-30 \
                            --leave-policy separate

Each subcommand forwards its remaining arguments to the matching module, so
``python rif.py select --help`` shows that module's own options. The modules
can equally be invoked directly with ``python -m rif_copilot.selection_criteria``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMANDS = {
    "ingest": ("rif_copilot.workforce_data", "box 1 — validate and normalize a roster"),
    "scenarios": ("rif_copilot.scenario_simulator", "box 2 — compare scenarios"),
    "select": ("rif_copilot.selection_criteria", "box 3 — score and build a cut list"),
    "impact": ("rif_copilot.adverse_impact", "box 4 — adverse impact analysis"),
    "compliance": ("rif_copilot.ca_compliance", "box 5 — Cal-WARN, final pay, OWBPA"),
    "severance": ("rif_copilot.severance_pay", "box 6 — severance and payroll impact"),
    "documents": ("rif_copilot.document_generator", "box 7 — draft notices and letters"),
    "audit": ("rif_copilot.audit_reporting", "box 10 — assemble the decision record"),
    "pipeline": ("rif_copilot.pipeline", "run every box end to end"),
}


def usage() -> int:
    print("California RIF Copilot\n")
    print("Usage: python rif.py <command> [options]\n")
    width = max(len(c) for c in COMMANDS) + 2
    for name, (_, desc) in COMMANDS.items():
        print(f"  {name:<{width}}{desc}")
    print("\nRun `python rif.py <command> --help` for a command's options.")
    print("Run `python run_tests.py` to execute the test suites.")
    return 1


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[1] in ("-h", "--help", "help"):
        return usage()
    command = argv[1]
    if command not in COMMANDS:
        print(f"Unknown command {command!r}.\n")
        return usage()

    module_name = COMMANDS[command][0]
    import importlib

    module = importlib.import_module(module_name)
    return int(module.main(argv[2:]) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
