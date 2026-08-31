"""
Tests for approvals.py

Run with pytest, or standalone:  python test_approvals.py
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

from rif_copilot.approvals import (
    UNCLEARABLE_CODES,
    ApprovalError,
    ApprovalLedger,
    ApprovalPackage,
    ApprovalPolicy,
    ApprovalRecord,
    ApprovalStage,
    default_policy,
)
from rif_copilot.workforce_data import Severity


# --- fakes -----------------------------------------------------------------


class _Selection:
    def __init__(self, ids, scores=None):
        self.cut_list = pd.DataFrame([
            {"employee_id": i, "retention_score": (scores or {}).get(i, 10.0),
             "job_title": "Engineer", "department": "Engineering"}
            for i in ids
        ])


class _Finding:
    def __init__(self, code, severity=Severity.ERROR):
        self.code, self.severity = code, severity


class _Gate:
    def __init__(self, blockers):
        self.blockers = blockers
        self.may_generate_documents = not blockers


class _CompReport:
    def __init__(self, codes, sep="2026-10-30"):
        self.findings = [_Finding(c) for c in codes]
        self.separation_date = sep
        self.notice_date = "2026-07-01"
        self.warn_triggered = False


class _Compliance:
    def __init__(self, codes=(), sep="2026-10-30"):
        self.report = _CompReport(list(codes), sep)
        self.gate = _Gate([f"{c}: detail" for c in codes])


def package(ids=("E1", "E2", "E3"), codes=(), sep="2026-10-30", scenario="A"):
    return ApprovalPackage.from_pipeline(
        scenario=scenario,
        selection=_Selection(ids),
        compliance=_Compliance(codes, sep),
    )


def approved_ledger(pkg=None, policy=None):
    pkg = pkg or package()
    led = ApprovalLedger(policy=policy)
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director", comment="Reviewed.")
    led.approve("legal", "R. Alvarez", "Employment Counsel", comment="Cleared.")
    led.approve("executive", "J. Park", "CFO", comment="Authorized.")
    return led, pkg


# --- fingerprinting --------------------------------------------------------


def test_identical_content_produces_the_same_fingerprint():
    assert package().fingerprint == package().fingerprint


def test_row_order_does_not_change_the_fingerprint():
    a = package(ids=("E1", "E2", "E3"))
    b = package(ids=("E3", "E1", "E2"))
    assert a.fingerprint == b.fingerprint


def test_changing_the_cut_list_changes_the_fingerprint():
    assert package(ids=("E1", "E2")).fingerprint != package(ids=("E1", "E2", "E3")).fingerprint


def test_changing_the_separation_date_changes_the_fingerprint():
    assert package(sep="2026-10-30").fingerprint != package(sep="2026-11-15").fingerprint


def test_diff_names_what_changed():
    a = package(ids=("E1", "E2"))
    b = package(ids=("E1", "E2", "E3"))
    changes = " ".join(b.diff(a))
    assert "added to the cut list" in changes
    assert "E3" in changes


def test_diff_reports_removals():
    a = package(ids=("E1", "E2", "E3"))
    b = package(ids=("E1",))
    changes = " ".join(b.diff(a))
    assert "removed from the cut list" in changes


def test_diff_reports_a_date_change():
    a = package(sep="2026-10-30")
    b = package(sep="2026-12-01")
    assert any("Separation date changed" in c for c in b.diff(a))


# --- approvals bind to a version -------------------------------------------


def test_approval_does_not_survive_a_change_to_the_plan():
    """The failure mode this whole module exists to prevent."""
    led, pkg = approved_ledger()
    assert led.is_fully_approved(pkg)

    changed = package(ids=("E1", "E2", "E3", "E4"))
    assert not led.is_fully_approved(changed)
    ok, problems = led.verify(changed)
    assert not ok
    assert "do not carry over" in problems[0]


def test_verify_explains_what_changed():
    led, _ = approved_ledger()
    ok, problems = led.verify(package(ids=("E1", "E2")))
    assert "removed from the cut list" in problems[0]


def test_resubmitting_supersedes_prior_approvals_without_deleting_them():
    led, pkg = approved_ledger()
    new = package(ids=("E1", "E2", "E3", "E4"))
    led.submit(new, submitted_by="M. Chen", role="HR Business Partner")

    assert not led.status(new).complete
    superseded = [r for r in led.records if r.action == "superseded"]
    assert superseded
    assert superseded[0].fingerprint == pkg.fingerprint
    # The original approvals are still in the history.
    assert any(r.action == "approved" and r.fingerprint == pkg.fingerprint
               for r in led.records)


def test_ledger_is_append_only_on_revocation():
    led, pkg = approved_ledger()
    before = len(led.records)
    led.revoke("legal", "R. Alvarez", "Employment Counsel",
               comment="New information about the CBA.")
    assert len(led.records) == before + 1
    assert any(r.action == "approved" and r.stage == "legal" for r in led.records)
    assert not led.status(pkg).complete


def test_revocation_requires_a_reason():
    led, _ = approved_ledger()
    with pytest.raises(ApprovalError):
        led.revoke("legal", "R. Alvarez", "Employment Counsel")


# --- chain order and independence ------------------------------------------


def test_stages_must_be_approved_in_order():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    with pytest.raises(ApprovalError):
        led.approve("executive", "J. Park", "CFO")
    with pytest.raises(ApprovalError):
        led.approve("legal", "R. Alvarez", "Employment Counsel")
    led.approve("hr", "S. Patel", "HR Director")
    led.approve("legal", "R. Alvarez", "Employment Counsel")


def test_order_error_explains_why_the_order_exists():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    try:
        led.approve("executive", "J. Park", "CFO")
        raise AssertionError("should have raised")
    except ApprovalError as e:
        assert "surfaces a legal problem" in str(e)


def test_submitter_cannot_approve_their_own_package():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    with pytest.raises(ApprovalError):
        led.approve("hr", "M. Chen", "HR Business Partner")


def test_self_approval_can_be_permitted_by_policy():
    policy = default_policy()
    policy.allow_self_approval = True
    pkg = package()
    led = ApprovalLedger(policy=policy)
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "M. Chen", "HR Business Partner")
    assert led.status(pkg).stages["hr"]["approved"]


def test_one_person_cannot_sign_two_stages():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director")
    with pytest.raises(ApprovalError):
        led.approve("legal", "S. Patel", "Employment Counsel")


def test_role_must_be_authorized_for_the_stage():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    with pytest.raises(ApprovalError):
        led.approve("hr", "J. Park", "CFO")


def test_approval_requires_a_named_approver():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    with pytest.raises(ApprovalError):
        led.approve("hr", "   ", "HR Director")


def test_submission_requires_a_named_submitter():
    led = ApprovalLedger()
    with pytest.raises(ApprovalError):
        led.submit(package(), submitted_by="")


def test_approving_before_submission_is_refused():
    led = ApprovalLedger()
    with pytest.raises(ApprovalError):
        led.approve("hr", "S. Patel", "HR Director")


# --- clearing blockers ------------------------------------------------------


def test_legal_can_clear_an_adverse_impact_finding():
    pkg = package(codes=("ADVERSE_IMPACT_INDICATED",))
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director")
    led.approve("legal", "R. Alvarez", "Employment Counsel",
                clears=["ADVERSE_IMPACT_INDICATED"],
                comment="Criteria confirmed job-related; documented.")
    led.approve("executive", "J. Park", "CFO")
    st = led.status(pkg)
    assert "ADVERSE_IMPACT_INDICATED" in st.cleared_codes
    assert st.complete


def test_data_blockers_cannot_be_cleared_by_anyone():
    pkg = package(codes=("FINAL_PAY_UNCOMPUTABLE",))
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director")
    with pytest.raises(ApprovalError):
        led.approve("legal", "R. Alvarez", "Employment Counsel",
                    clears=["FINAL_PAY_UNCOMPUTABLE"], comment="Proceed.")


def test_unclearable_message_explains_why():
    pkg = package(codes=("NO_PAY_DATA",))
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director")
    try:
        led.approve("legal", "R. Alvarez", "Employment Counsel",
                    clears=["NO_PAY_DATA"], comment="Fine.")
        raise AssertionError("should have raised")
    except ApprovalError as e:
        assert "does not supply a pay rate" in str(e)


def test_a_stage_cannot_clear_outside_its_competence():
    pkg = package(codes=("ADVERSE_IMPACT_INDICATED",))
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    with pytest.raises(ApprovalError):
        led.approve("hr", "S. Patel", "HR Director",
                    clears=["ADVERSE_IMPACT_INDICATED"], comment="Fine by me.")


def test_executive_stage_cannot_clear_anything():
    assert default_policy().stage("executive").can_clear == ()


def test_clearing_requires_a_written_basis():
    pkg = package(codes=("ADVERSE_IMPACT_INDICATED",))
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director")
    with pytest.raises(ApprovalError):
        led.approve("legal", "R. Alvarez", "Employment Counsel",
                    clears=["ADVERSE_IMPACT_INDICATED"])


def test_uncleared_blockers_prevent_completion():
    pkg = package(codes=("ADVERSE_IMPACT_INDICATED",))
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director")
    led.approve("legal", "R. Alvarez", "Employment Counsel", comment="Noted.")
    led.approve("executive", "J. Park", "CFO")
    st = led.status(pkg)
    assert not st.complete
    assert "ADVERSE_IMPACT_INDICATED" in st.uncleared_codes


def test_unclearable_set_covers_the_data_blockers():
    for code in ("FINAL_PAY_UNCOMPUTABLE", "NO_PAY_DATA", "LEAVE_POLICY_UNDECLARED",
                 "NO_ESTABLISHMENT_COLUMN", "INCOMPLETE_REGISTER"):
        assert code in UNCLEARABLE_CODES


# --- rejection and staleness ------------------------------------------------


def test_rejection_blocks_the_chain_and_records_the_reason():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director")
    led.reject("legal", "R. Alvarez", "Employment Counsel",
               comment="Decisional unit is wrong.")
    st = led.status(pkg)
    assert not st.complete
    assert "Decisional unit is wrong" in st.blocked_reason


def test_rejection_requires_a_reason():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    with pytest.raises(ApprovalError):
        led.reject("hr", "S. Patel", "HR Director")


def test_stale_approvals_expire():
    pkg = package()
    policy = default_policy()
    policy.validity_days = 30
    led = ApprovalLedger(policy=policy)
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    old = (dt.datetime.now() - dt.timedelta(days=60)).isoformat(timespec="seconds")
    led._records.append(ApprovalRecord(
        action="approved", stage="hr", actor="S. Patel", role="HR Director",
        timestamp=old, fingerprint=pkg.fingerprint, comment="Reviewed.",
    ))
    st = led.status(pkg)
    assert st.stale
    assert not st.stages["hr"]["approved"]
    assert "older than" in st.blocked_reason


# --- status and clearance ---------------------------------------------------


def test_next_stage_is_reported():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    assert led.status(pkg).next_stage == "hr"
    led.approve("hr", "S. Patel", "HR Director")
    assert led.status(pkg).next_stage == "legal"


def test_clearance_reports_the_legal_approver_for_downstream_boxes():
    pkg = package(codes=("ADVERSE_IMPACT_INDICATED",))
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director")
    led.approve("legal", "R. Alvarez", "Employment Counsel",
                clears=["ADVERSE_IMPACT_INDICATED"], comment="Job-related.")
    led.approve("executive", "J. Park", "CFO")
    c = led.clearance(pkg)
    assert c["approved"]
    assert c["legal_approver"] == "R. Alvarez"
    assert c["legal_basis"] == "Job-related."
    assert c["fingerprint"] == pkg.fingerprint


def test_clearance_reports_the_blocking_reason_when_incomplete():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    c = led.clearance(pkg)
    assert not c["approved"]
    assert "HR review" in c["blocked_reason"]


def test_multiple_approvers_can_be_required():
    policy = ApprovalPolicy(stages=(
        ApprovalStage(key="hr", name="HR review", min_approvers=2),
    ))
    pkg = package()
    led = ApprovalLedger(policy=policy)
    led.submit(pkg, submitted_by="M. Chen", role="HR")
    led.approve("hr", "S. Patel", "HR Director")
    assert not led.status(pkg).complete
    led.approve("hr", "L. Wong", "CHRO")
    assert led.status(pkg).complete


def test_conditional_approval_records_its_conditions():
    pkg = package()
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    led.approve("hr", "S. Patel", "HR Director",
                conditions="Subject to payroll confirming vacation balances.")
    assert "payroll" in led.status(pkg).stages["hr"]["conditions"][0]


# --- persistence and output -------------------------------------------------


def test_ledger_round_trips_through_json():
    led, pkg = approved_ledger()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "ledger.json"
        led.to_json(p)
        restored = ApprovalLedger.from_json(p)
    assert len(restored.records) == len(led.records)
    assert restored.is_fully_approved(pkg)


def test_history_dataframe_lists_every_action():
    led, _ = approved_ledger()
    df = led.to_dataframe()
    assert len(df) == len(led.records)
    assert set(df["action"]) >= {"submitted", "approved"}


def test_markdown_shows_the_chain_and_the_history():
    led, _ = approved_ledger()
    md = led.to_markdown()
    assert "Status: APPROVED" in md
    assert "HR review" in md
    assert "History" in md
    assert "append-only" in md


def test_markdown_explains_uncleared_data_blockers():
    pkg = package(codes=("FINAL_PAY_UNCOMPUTABLE",))
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")
    md = led.to_markdown()
    assert "cannot be cleared by approval" in md


def test_package_summary_captures_the_decision_facts():
    pkg = package(ids=("E1", "E2"))
    assert pkg.summary["affected"] == 2
    assert pkg.summary["separation_date"] == "2026-10-30"


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
