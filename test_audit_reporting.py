"""
Tests for audit_reporting.py

Run with pytest, or standalone:  python test_audit_reporting.py
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

from rif_copilot.audit_reporting import (
    PRIVILEGE_CLASSES,
    AuditEntry,
    AuditLog,
    AuditPackage,
    RetentionPolicy,
)
from rif_copilot.workforce_data import Severity


# --- fakes -----------------------------------------------------------------


class _Finding:
    def __init__(self, code, message, severity=Severity.ERROR):
        self.code, self.message, self.severity = code, message, severity


class _IngestReport:
    def __init__(self, errors=1):
        self.issues = [_Finding("INVALID_DATE", "Bad date.")] * errors

    def summary(self):
        return {"rows": 100, "errors": len(self.issues), "warnings": 2,
                "rows_with_errors": 1, "blocking": False}


class _Ingest:
    def __init__(self, errors=1):
        self.report = _IngestReport(errors)


class _SelReport:
    def __init__(self):
        self.findings = [_Finding("TIE_AT_CUT_BOUNDARY", "Tie at boundary.")]
        self.department_summary = [
            {"department": "Engineering", "mode": "individual",
             "evaluated": 40, "selected": 5, "savings": 500000.0}
        ]

    def summary(self):
        return {"selected_employees": 5, "eligible_employees": 40,
                "achieved_savings": 500000.0, "cost_savings_target": 500000.0}


class _Selection:
    def __init__(self, queue=0):
        self.report = _SelReport()
        self.review_queue = pd.DataFrame([{"employee_id": f"E{i}"}
                                          for i in range(queue)])


class _Comparison:
    protected_class = "Age 40+"
    group = "Age 40+"
    unit = "Engineering"
    unit_type = "department"
    group_selection_rate = 0.217
    reference_selection_rate = 0.0
    impact_ratio = 0.78
    std_deviations = 2.29
    fisher_p = 0.049

    def to_dict(self):
        return {"protected_class": self.protected_class, "unit": self.unit}


class _ImpactReport:
    def __init__(self, indicated=True):
        self.indicated = [_Comparison()] if indicated else []
        self.comparisons = [_Comparison()]
        self.findings = [_Finding("ADVERSE_IMPACT_INDICATED", "Impact indicated.")]

    def class_verdicts(self):
        return {"Age 40+": "Impact indicated" if self.indicated else "No flag",
                "Sex": "No flag"}


class _Impact:
    def __init__(self, indicated=True):
        self.report = _ImpactReport(indicated)


class _PayReport:
    def __init__(self):
        self.totals = {"severance_gross": 400000.0, "vacation_payout": 90000.0,
                       "total_employer_cost": 600000.0}
        self.assumptions = {"formula": {"weeks_per_year": 2.0}}
        self.findings = []
        self.employee_count = 5


class _Pay:
    def __init__(self):
        self.report = _PayReport()


class _Warn:
    jurisdiction = "California"
    establishment = "SF HQ"
    triggered = False
    reason = "Below threshold."

    def to_dict(self):
        return {"jurisdiction": self.jurisdiction}


class _Obligation:
    def __init__(self, missed=False):
        self.title = "Pay final wages"
        self.due_date = dt.date(2026, 10, 30)
        self.authority = "Lab. Code § 201"
        self.missed = missed


class _CompGate:
    def __init__(self, blockers=()):
        self.blockers = list(blockers)
        self.may_generate_documents = not blockers


class _CompReport:
    def __init__(self, missed=False):
        self.warn = [_Warn()]
        self.obligations = [_Obligation(missed)]
        self.findings = [_Finding("SELECTED_UNION_MEMBERS", "Union members.")]
        self.separation_date = "2026-10-30"

    @property
    def missed_deadlines(self):
        return [o for o in self.obligations if o.missed]


class _Compliance:
    def __init__(self, missed=False, blockers=("blocked",)):
        self.report = _CompReport(missed)
        self.gate = _CompGate(blockers)


class _LedgerRecord:
    def __init__(self, action="approved", stage="legal", actor="R. Alvarez",
                 clears=(), comment="Cleared."):
        self.action, self.stage, self.actor = action, stage, actor
        self.role = "Employment Counsel"
        self.timestamp = dt.datetime.now().isoformat(timespec="seconds")
        self.fingerprint = "abc123"
        self.clears, self.comment = tuple(clears), comment


class _LedgerStatus:
    def __init__(self, complete=True, cleared=()):
        self.complete = complete
        self.blocked_reason = "" if complete else "Awaiting Legal review."
        self.fingerprint = "abc123"
        self.cleared_codes = tuple(cleared)


class _Ledger:
    def __init__(self, complete=True, cleared=()):
        self.records = [
            _LedgerRecord("submitted", None, "M. Chen"),
            _LedgerRecord("approved", "legal", "R. Alvarez", cleared),
        ]
        self._status = _LedgerStatus(complete, cleared)

    def status(self, package=None):
        return self._status


class _Doc:
    def __init__(self, doc_type="separation_letter"):
        self.doc_type = doc_type


class _Documents:
    def __init__(self, blocked=False, override=None, n=3):
        self.blocked = blocked
        self.blockers = ["Impact indicated"] if blocked else []
        self.documents = [_Doc() for _ in range(n)]
        self.override_record = override
        self.findings = []


class _Board:
    def __init__(self, overdue_statutory=0):
        self.log = [{"timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                     "action": "completed", "target": "PREP-PAYROLL",
                     "actor": "Payroll", "detail": "Register #4471"}]
        self.findings = []
        self._overdue = overdue_statutory

    def summary(self):
        return {"total_tasks": 140, "overdue": self._overdue,
                "overdue_statutory": self._overdue, "percent_complete": 10.0}


class _Package:
    fingerprint = "abc123"
    scenario = "Scenario A"


def full(**over):
    kw = dict(
        scenario="Scenario A", ingest=_Ingest(), selection=_Selection(),
        impact=_Impact(), compliance=_Compliance(), pay=_Pay(),
        documents=_Documents(), ledger=_Ledger(), board=_Board(),
        package=_Package(),
    )
    kw.update(over)
    return AuditPackage.assemble(**kw)


# --- hash chain -------------------------------------------------------------


def test_fresh_log_verifies_intact():
    log = AuditLog()
    for i in range(5):
        log.append("test", f"event{i}", "detail")
    r = log.verify()
    assert r.intact
    assert r.entries_checked == 5


def test_each_entry_links_to_the_previous_one():
    log = AuditLog()
    a = log.append("test", "one", "d")
    b = log.append("test", "two", "d")
    assert b.prev_hash == a.entry_hash


def test_editing_an_entry_breaks_the_chain():
    """The failure this module exists to make detectable."""
    log = AuditLog()
    log.append("impact", "adverse_impact_indicated", "Impact indicated.",
               severity=Severity.ERROR)
    log.append("approval", "approved", "Approved.")
    e = log._entries[0]
    log._entries[0] = AuditEntry(
        **{**e.__dict__, "detail": "No impact found.", "severity": Severity.INFO}
    )
    r = log.verify()
    assert not r.intact
    assert r.broken_at == 0
    assert "modified since it was written" in r.problem


def test_deleting_an_entry_breaks_the_chain():
    log = AuditLog()
    for i in range(4):
        log.append("test", f"e{i}", "d")
    del log._entries[1]
    r = log.verify()
    assert not r.intact
    assert "removed, or inserted" in r.problem


def test_inserting_an_entry_breaks_the_chain():
    log = AuditLog()
    for i in range(3):
        log.append("test", f"e{i}", "d")
    forged = AuditEntry(sequence=1, timestamp="2026-01-01T00:00:00",
                        category="test", event="forged", detail="d")
    log._entries.insert(1, AuditEntry(
        **{**forged.__dict__, "entry_hash": forged.compute_hash()}
    ))
    assert not log.verify().intact


def test_appending_after_verification_keeps_the_chain_intact():
    log = AuditLog()
    log.append("test", "one", "d")
    assert log.verify().intact
    log.append("test", "two", "d")
    assert log.verify().intact


# --- no filtering -----------------------------------------------------------


def test_every_finding_is_recorded_at_every_severity():
    audit = full()
    events = {e.event for e in audit.log.entries}
    assert "INVALID_DATE" in events          # box 1 error
    assert "TIE_AT_CUT_BOUNDARY" in events   # box 3 error
    assert "ADVERSE_IMPACT_INDICATED" in events
    assert "SELECTED_UNION_MEMBERS" in events


def test_executive_summary_lists_every_error_and_says_it_cannot_be_filtered():
    audit = full()
    md = audit.executive_summary()
    errors = audit.log.by_severity(Severity.ERROR)
    assert f"Every error in the record ({len(errors)})" in md
    assert "not filtered and cannot be" in md


def test_adverse_impact_appears_in_the_executive_summary():
    md = full().executive_summary()
    assert "Age 40+" in md
    assert "Impact indicated" in md


def test_adverse_impact_detail_includes_the_statistics():
    audit = full()
    entry = next(e for e in audit.log.entries
                 if e.event == "adverse_impact_indicated")
    assert "0.78" in entry.detail
    assert "+2.29 SD" in entry.detail


def test_counsel_gate_override_is_recorded_as_a_warning():
    audit = full(documents=_Documents(override={
        "by": "R. Alvarez", "date": "2026-08-19", "reason": "Reviewed.",
    }))
    entry = next(e for e in audit.log.entries
                 if e.event == "gate_cleared_by_counsel")
    assert entry.severity == Severity.WARNING
    assert "R. Alvarez" in entry.detail


def test_discarded_scenarios_are_retained_in_the_record():
    class _Scen:
        name = "B — not pursued"
        rationale = "Downside case."

    class _Outcome:
        scenario = _Scen()

        def summary_row(self):
            return {"scenario": "B — not pursued"}

    class _SimReport:
        findings = []

    class _Sim:
        outcomes = [_Outcome()]
        report = _SimReport()

    audit = full(simulation=_Sim())
    events = [e for e in audit.log.entries if e.event == "scenario_modeled"]
    assert events
    assert any(e.event == "scenarios_retained" for e in audit.log.entries)


# --- completeness -----------------------------------------------------------


def test_a_complete_record_scores_full_marks():
    c = full().completeness()
    assert c.complete
    assert c.score == 100.0


def test_missing_approval_record_is_a_gap():
    c = full(ledger=None).completeness()
    assert not c.complete
    assert any("Approval record" in item for item, _ in c.missing)


def test_missing_impact_analysis_gap_explains_what_its_absence_means():
    c = full(impact=None).completeness()
    why = next(w for item, w in c.missing if "impact" in item.lower())
    assert "never measured" in why


def test_gaps_are_written_into_the_log_as_errors():
    audit = full(ledger=None)
    gaps = [e for e in audit.log.entries if e.event == "record_gap"]
    assert gaps
    assert gaps[0].severity == Severity.ERROR


def test_indicated_impact_without_clearance_is_a_weakness():
    audit = full(ledger=_Ledger(complete=True, cleared=()))
    weak = " ".join(w for _, w in audit.completeness().weak)
    assert "most damaging document" in weak


def test_cleared_impact_is_not_flagged_as_a_weakness():
    audit = full(ledger=_Ledger(complete=True,
                                cleared=("ADVERSE_IMPACT_INDICATED",)))
    items = [i for i, _ in audit.completeness().weak]
    assert "Adverse impact finding" not in items


def test_incomplete_approval_chain_is_a_weakness():
    audit = full(ledger=_Ledger(complete=False))
    items = [i for i, _ in audit.completeness().weak]
    assert "Approval chain" in items


def test_open_review_queue_is_a_weakness():
    audit = full(selection=_Selection(queue=14))
    weak = " ".join(w for _, w in audit.completeness().weak)
    assert "provisional" in weak


def test_missed_deadlines_are_a_weakness():
    audit = full(compliance=_Compliance(missed=True))
    items = [i for i, _ in audit.completeness().weak]
    assert "Statutory deadlines" in items


def test_absent_documents_and_board_are_weak_not_missing():
    c = full(documents=None, board=None).completeness()
    assert c.complete
    items = [i for i, _ in c.weak]
    assert "Document set" in items and "Task board" in items


# --- privilege and retention ------------------------------------------------


def test_artifacts_are_classified_for_privilege():
    p = full().privilege_summary()
    assert "privileged" in p
    assert "business_record" in p


def test_adverse_impact_is_classified_privileged():
    assert PRIVILEGE_CLASSES["adverse_impact"] == "privileged"
    assert PRIVILEGE_CLASSES["payroll_register"] == "business_record"


def test_compliance_report_flags_the_mixed_package_without_deciding():
    md = full().compliance_report()
    assert "weaken a privilege claim" in md
    assert "counsel decides" in md


def test_retention_guidance_cites_a_basis_for_every_row():
    for item, years, basis in RetentionPolicy().guidance():
        assert years > 0
        assert basis.strip()


def test_retention_section_states_a_litigation_hold_overrides():
    md = full().compliance_report()
    assert "litigation hold overrides" in md
    assert "discarded scenarios" in md


# --- reports ----------------------------------------------------------------


def test_decision_history_groups_entries_by_category():
    md = full().decision_history()
    for heading in ("Data", "Selection", "Impact", "Compliance", "Approval"):
        assert heading in md


def test_decision_history_explains_the_hash_chain():
    md = full().decision_history()
    assert "hash-chained" in md
    assert "no filter" in md


def test_reports_carry_the_privilege_header():
    audit = full()
    for md in (audit.executive_summary(), audit.decision_history(),
               audit.compliance_report()):
        assert "Privileged and confidential" in md


def test_executive_summary_reports_integrity_status():
    audit = full()
    assert "intact" in audit.executive_summary()
    e = audit.log._entries[0]
    audit.log._entries[0] = AuditEntry(**{**e.__dict__, "detail": "tampered"})
    assert "**BROKEN**" in audit.executive_summary()


def test_summary_is_shorter_than_the_record_not_quieter():
    md = full().executive_summary()
    assert "shorter than the record, not quieter" in md


def test_counts_reflect_the_log():
    audit = full()
    c = audit.counts()
    assert c["entries"] == len(audit.log.entries)
    assert c["errors"] == len(audit.log.by_severity(Severity.ERROR))


def test_selection_records_that_protected_fields_were_excluded():
    audit = full()
    assert any(e.event == "protected_fields_excluded" for e in audit.log.entries)


def test_approval_records_are_ingested_with_their_original_timestamps():
    ledger = _Ledger()
    audit = full(ledger=ledger)
    entries = [e for e in audit.log.entries if e.category == "approval"
               and e.event.startswith("approval_")]
    assert entries
    assert any(e.actor == "R. Alvarez" for e in entries)


def test_task_activity_is_carried_into_the_record():
    audit = full(board=_Board())
    assert any(e.event == "task_completed" for e in audit.log.entries)


def test_overdue_statutory_task_is_an_error_in_the_record():
    audit = full(board=_Board(overdue_statutory=2))
    entry = next(e for e in audit.log.entries if e.event == "task_board_built")
    assert entry.severity == Severity.ERROR


# --- outputs ----------------------------------------------------------------


def test_write_produces_all_artifacts():
    with tempfile.TemporaryDirectory() as tmp:
        paths = full().write(tmp)
        assert set(paths) == {"executive_summary", "decision_history",
                              "compliance_report", "log", "artifacts", "json"}
        assert all(p.exists() for p in paths.values())


def test_json_round_trips_and_includes_integrity():
    payload = json.loads(full().to_json())
    assert payload["integrity"]["intact"]
    assert payload["completeness"]["complete"]
    assert payload["log"]["entries"]


def test_log_dataframe_includes_the_entry_hash():
    df = full().log.to_dataframe()
    assert "entry_hash" in df.columns
    assert len(df) > 0


def test_empty_assembly_reports_every_gap():
    audit = AuditPackage.assemble(scenario="Empty")
    c = audit.completeness()
    assert not c.complete
    assert len(c.missing) >= 6


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
