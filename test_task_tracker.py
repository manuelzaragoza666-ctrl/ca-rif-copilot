"""
Tests for task_tracker.py

Run with pytest, or standalone:  python test_task_tracker.py
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

from rif_copilot.task_tracker import (
    Task,
    TaskBoard,
    TaskError,
    TrackerConfig,
)
from rif_copilot.workforce_data import Severity

SEP = dt.date(2026, 10, 30)


# --- fakes -----------------------------------------------------------------


class _Obligation:
    def __init__(self, code, title, due, authority="Lab. Code § 201",
                 description="", missed=False):
        self.code, self.title, self.due_date = code, title, due
        self.authority, self.description, self.missed = authority, description, missed


class _CompReport:
    def __init__(self, obligations=None, sep=SEP):
        self.obligations = obligations if obligations is not None else [
            _Obligation("FINAL_PAY", "Pay all final wages", sep),
            _Obligation("COBRA_NOTICE", "Send COBRA election notice",
                        sep + dt.timedelta(days=44), "29 U.S.C. § 1166"),
        ]
        self.separation_date = sep.isoformat()
        self.scenario = "Scenario A"


class _Compliance:
    def __init__(self, obligations=None, sep=SEP):
        self.report = _CompReport(obligations, sep)


class _Pay:
    def __init__(self, ids=("E1", "E2")):
        self.register = pd.DataFrame([
            {"employee_id": i, "status": "computed", "severance_gross": 10000.0}
            for i in ids
        ])


class _Package:
    def __init__(self, fingerprint="abc123", codes=()):
        self.fingerprint = fingerprint
        self.scenario = "Scenario A"
        self.blocker_codes = tuple(codes)


class _Documents:
    def __init__(self, blocked=False, blockers=(), incomplete=()):
        self.blocked = blocked
        self.blockers = list(blockers)
        self.incomplete = list(incomplete)


def board(**kw) -> TaskBoard:
    defaults = dict(
        compliance=_Compliance(), pay=_Pay(), package=_Package(),
        config=TrackerConfig(),
    )
    defaults.update(kw)
    return TaskBoard.build(**defaults)


def codes(b) -> set[str]:
    return {f["code"] for f in b.findings}


# --- generation ------------------------------------------------------------


def test_statutory_obligations_become_immovable_tasks():
    b = board()
    t = b.get("STAT-FINAL_PAY")
    assert t.immovable
    assert t.due_date == SEP
    assert t.authority


def test_logistics_tasks_are_generated():
    b = board()
    for tid in ("PREP-PAYROLL", "PREP-MANAGERS", "DAY-NOTIFY", "DAY-DELIVER-PAY"):
        assert tid in {t.id for t in b.tasks}


def test_payroll_prep_precedes_notice_day():
    b = board()
    assert b.get("PREP-PAYROLL").due_date < b.get("DAY-NOTIFY").due_date


def test_notification_depends_on_preparation():
    b = board()
    assert "PREP-PAYROLL" in b.get("DAY-NOTIFY").depends_on


def test_per_employee_tasks_are_generated_for_each_person():
    b = board(pay=_Pay(ids=("E1", "E2", "E3")))
    for emp in ("E1", "E2", "E3"):
        assert f"EMP-{emp}-NOTIFY" in {t.id for t in b.tasks}
        assert f"EMP-{emp}-FINALPAY" in {t.id for t in b.tasks}


def test_per_employee_tasks_can_be_disabled():
    b = board(config=TrackerConfig(per_employee_tasks=False))
    assert not [t for t in b.tasks if t.category == "per_employee"]


def test_open_blockers_become_approval_tasks():
    b = board(package=_Package(codes=("ADVERSE_IMPACT_INDICATED",)))
    assert "APPR-ADVERSE_IMPACT_INDICATED" in {t.id for t in b.tasks}


def test_blocked_document_generation_creates_an_unblock_task():
    b = board(documents=_Documents(blocked=True, blockers=["Impact indicated"]))
    assert "DOC-UNBLOCK" in {t.id for t in b.tasks}


def test_documents_with_placeholders_create_a_fill_task():
    b = board(documents=_Documents(incomplete=[object(), object()]))
    assert "DOC-PLACEHOLDERS" in {t.id for t in b.tasks}
    assert "DOC-PLACEHOLDERS" in b.get("DOC-COUNSEL-REVIEW").depends_on


def test_counsel_review_task_always_exists_when_documents_generated():
    b = board(documents=_Documents())
    assert "DOC-COUNSEL-REVIEW" in {t.id for t in b.tasks}


def test_missing_compliance_input_is_an_error_not_an_empty_board():
    b = TaskBoard.build(compliance=None)
    assert "NO_COMPLIANCE_INPUT" in codes(b)


def test_no_statutory_obligations_is_flagged_as_unusual():
    b = board(compliance=_Compliance(obligations=[]))
    assert "NO_STATUTORY_TASKS" in codes(b)


# --- the OWBPA consideration window ----------------------------------------


def test_no_followup_task_falls_inside_the_consideration_period():
    """A reminder to chase a signature during the statutory deliberation
    window converts a protection into a pressure campaign."""
    b = board()
    consider = b.get("EMP-E1-CONSIDER")
    followup = b.get("EMP-E1-FOLLOWUP")
    assert followup.not_before == consider.due_date
    assert followup.due_date >= consider.due_date


def test_followup_cannot_be_completed_during_the_consideration_period():
    b = board()
    early = SEP + dt.timedelta(days=5)
    with pytest.raises(TaskError):
        b.complete("EMP-E1-FOLLOWUP", by="HR", today=early)


def test_consideration_task_explains_why_no_reminder_exists():
    b = board()
    desc = b.get("EMP-E1-CONSIDER").description
    assert "No follow-up task is scheduled inside this window" in desc
    assert "voluntariness" in desc


def test_board_markdown_states_the_no_reminder_policy():
    md = board().to_markdown()
    assert "consideration period" in md.lower()
    assert "voluntariness" in md


def test_consideration_period_length_is_configurable():
    b = board(config=TrackerConfig(consideration_days=21))
    assert b.get("EMP-E1-CONSIDER").due_date == SEP + dt.timedelta(days=21)


# --- completion guardrails --------------------------------------------------


def test_evidence_is_required_where_declared():
    b = board()
    with pytest.raises(TaskError):
        b.complete("STAT-FINAL_PAY", by="Payroll")
    b.complete("STAT-FINAL_PAY", by="Payroll", evidence="Register #4471")
    assert b.get("STAT-FINAL_PAY").status == "complete"


def test_evidence_error_explains_why_it_matters():
    b = board()
    try:
        b.complete("STAT-FINAL_PAY", by="Payroll")
        raise AssertionError("should have raised")
    except TaskError as e:
        assert "stops anyone looking" in str(e)


def test_completion_requires_naming_who_completed_it():
    b = board()
    with pytest.raises(TaskError):
        b.complete("PREP-MANAGERS", by="  ")


def test_revocation_task_cannot_be_closed_early():
    b = board()
    t = b.get("EMP-E1-REVOKE")
    with pytest.raises(TaskError):
        b.complete("EMP-E1-REVOKE", by="HR", evidence="signed",
                   today=t.not_before - dt.timedelta(days=1))
    b.complete("EMP-E1-REVOKE", by="HR", evidence="signed", today=t.not_before)
    assert b.get("EMP-E1-REVOKE").status == "complete"


def test_early_completion_error_explains_the_waiting_is_the_point():
    b = board()
    t = b.get("EMP-E1-REVOKE")
    try:
        b.complete("EMP-E1-REVOKE", by="HR", evidence="x",
                   today=t.not_before - dt.timedelta(days=1))
        raise AssertionError("should have raised")
    except TaskError as e:
        assert "substance of this task" in str(e)


def test_dependencies_must_be_complete_first():
    b = board()
    with pytest.raises(TaskError):
        b.complete("DAY-DELIVER-PAY", by="HR", evidence="run #1")
    b.complete("PREP-PAYROLL", by="Payroll", evidence="Register #4471")
    b.complete("DAY-DELIVER-PAY", by="HR", evidence="run #1")
    assert b.get("DAY-DELIVER-PAY").status == "complete"


def test_not_applicable_satisfies_a_dependency():
    b = board()
    b.set_status("PREP-PAYROLL", "not_applicable", by="HR", notes="No pay due.")
    b.complete("DAY-DELIVER-PAY", by="HR", evidence="n/a")
    assert b.get("DAY-DELIVER-PAY").status == "complete"


def test_set_status_cannot_be_used_to_bypass_completion_checks():
    b = board()
    with pytest.raises(TaskError):
        b.set_status("STAT-FINAL_PAY", "complete", by="HR")


def test_unknown_status_is_rejected():
    b = board()
    with pytest.raises(TaskError):
        b.set_status("PREP-ROOMS", "nearly", by="HR")


def test_unknown_task_id_is_rejected():
    b = board()
    with pytest.raises(TaskError):
        b.get("NOPE")


# --- rescheduling ----------------------------------------------------------


def test_statutory_deadlines_cannot_be_rescheduled():
    b = board()
    with pytest.raises(TaskError):
        b.reschedule("STAT-FINAL_PAY", SEP + dt.timedelta(days=30),
                     by="HR", reason="Running late.")


def test_reschedule_error_explains_the_right_way_to_change_it():
    b = board()
    try:
        b.reschedule("STAT-COBRA_NOTICE", SEP, by="HR", reason="x")
        raise AssertionError("should have raised")
    except TaskError as e:
        assert "regenerate the board" in str(e)
        assert "looks authoritative" in str(e)


def test_operational_tasks_can_be_rescheduled_with_a_reason():
    b = board()
    new = SEP - dt.timedelta(days=1)
    b.reschedule("PREP-ROOMS", new, by="HR Ops", reason="Rooms unavailable.")
    assert b.get("PREP-ROOMS").due_date == new


def test_rescheduling_requires_a_reason():
    b = board()
    with pytest.raises(TaskError):
        b.reschedule("PREP-ROOMS", SEP, by="HR Ops", reason="")


def test_reschedule_is_logged():
    b = board()
    b.reschedule("PREP-ROOMS", SEP, by="HR Ops", reason="Rooms unavailable.")
    assert any(e["action"] == "rescheduled" for e in b.log)


# --- overdue and status ----------------------------------------------------


def test_passed_statutory_deadline_is_an_error_at_build():
    past = dt.date.today() - dt.timedelta(days=10)
    b = board(compliance=_Compliance(
        obligations=[_Obligation("WARN", "Issue WARN notice", past)], sep=past,
    ))
    assert "STATUTORY_DEADLINE_PASSED" in codes(b)


def test_passed_deadline_message_says_it_cannot_be_worked_faster():
    past = dt.date.today() - dt.timedelta(days=10)
    b = board(compliance=_Compliance(
        obligations=[_Obligation("WARN", "Issue WARN notice", past)], sep=past,
    ))
    msg = next(f["message"] for f in b.findings
               if f["code"] == "STATUTORY_DEADLINE_PASSED")
    assert "cannot be met by working faster" in msg


def test_overdue_excludes_completed_tasks():
    past = dt.date.today() - dt.timedelta(days=5)
    b = board()
    b.get("PREP-ROOMS").due_date = past
    assert any(t.id == "PREP-ROOMS" for t in b.overdue())
    b.complete("PREP-ROOMS", by="HR Ops")
    assert not any(t.id == "PREP-ROOMS" for t in b.overdue())


def test_due_soon_uses_the_configured_window():
    b = board(config=TrackerConfig(due_soon_days=3))
    soon = dt.date.today() + dt.timedelta(days=2)
    b.get("PREP-ROOMS").due_date = soon
    assert any(t.id == "PREP-ROOMS" for t in b.due_soon())


def test_blocked_tasks_are_identified():
    b = board()
    assert any(t.id == "DAY-DELIVER-PAY" for t in b.blocked_tasks())


def test_summary_counts_progress():
    b = board()
    before = b.summary()["percent_complete"]
    b.complete("PREP-MANAGERS", by="HR")
    assert b.summary()["percent_complete"] > before


# --- acknowledgments --------------------------------------------------------


def test_acknowledgment_slots_are_created_per_employee():
    b = board(pay=_Pay(ids=("E1", "E2")))
    df = b.acknowledgments_dataframe()
    assert set(df["employee_id"]) == {"E1", "E2"}
    assert set(df["item"]) == {"notice", "packet", "final_pay"}


def test_recording_an_acknowledgment_marks_it_received():
    b = board()
    b.record_acknowledgment("E1", "packet", by="HR", reference="Signed receipt")
    df = b.acknowledgments_dataframe()
    row = df.loc[(df["employee_id"] == "E1") & (df["item"] == "packet")].iloc[0]
    assert bool(row["received"])
    assert row["reference"] == "Signed receipt"


def test_acknowledgment_counts_appear_in_the_summary():
    b = board()
    b.record_acknowledgment("E1", "notice", by="HR")
    s = b.summary()
    assert s["acknowledgments_received"] == 1
    assert s["acknowledgments_total"] > 1


# --- version binding --------------------------------------------------------


def test_board_detects_a_changed_plan_version():
    b = board(package=_Package(fingerprint="v1"))
    ok, msg = b.check_version(_Package(fingerprint="v2"))
    assert not ok
    assert "Rebuild the board" in msg


def test_matching_version_passes():
    b = board(package=_Package(fingerprint="v1"))
    ok, msg = b.check_version(_Package(fingerprint="v1"))
    assert ok and not msg


# --- outputs ---------------------------------------------------------------


def test_activity_log_records_completions():
    b = board()
    b.complete("PREP-MANAGERS", by="S. Patel")
    assert any(e["action"] == "completed" and e["actor"] == "S. Patel"
               for e in b.log)


def test_dataframe_is_ordered_by_category_then_date():
    df = board().to_dataframe()
    cats = [c for c in df["category"].tolist()]
    assert cats.index("statutory") < cats.index("logistics")


def test_markdown_marks_statutory_deadlines():
    md = board().to_markdown()
    assert "🔒" in md
    assert "cannot be rescheduled here" in md


def test_write_produces_all_artifacts():
    b = board()
    with tempfile.TemporaryDirectory() as tmp:
        paths = b.write(tmp)
        assert set(paths) == {"board_md", "tasks", "acknowledgments", "log", "json"}
        assert all(p.exists() for p in paths.values())


def test_json_round_trips():
    payload = json.loads(board().to_json())
    assert "tasks" in payload and "summary" in payload
    assert payload["summary"]["total_tasks"] > 0


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
