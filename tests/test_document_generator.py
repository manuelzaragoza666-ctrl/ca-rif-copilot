"""
Tests for document_generator.py

Run with pytest, or standalone:  python test_document_generator.py
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

from rif_copilot.ca_compliance import ComplianceConfig, ComplianceEngine
from rif_copilot.document_generator import (
    NON_OVERRIDABLE_CODES,
    DocumentConfig,
    DocumentGenerator,
)
from rif_copilot.severance_pay import PayConfig, SeveranceFormula, SeverancePayEngine
from rif_copilot.workforce_data import Severity

SEP = dt.date(2026, 10, 30)
NOTICE = dt.date(2026, 7, 1)


# --- fixtures --------------------------------------------------------------


def population(n=100, selected=5, **over) -> pd.DataFrame:
    base = {
        "worksite_name": "SF HQ", "department": "Engineering",
        "job_title": "Engineer", "job_level": "L3", "full_name": "Pat Doe",
        "tenure_years": 3.0, "annualized_pay": 120000.0,
        "hourly_equivalent_rate": 57.69, "accrued_vacation_hours": 80.0,
        "age_40_plus": False, "age_years": 35.0, "leave_status": None,
        "union_flag": False, "visa_status": None,
    }
    base.update(over)
    return pd.DataFrame([
        {**base, "employee_id": f"E{i}", "selected": i < selected} for i in range(n)
    ])


def compliance(scores: pd.DataFrame, **over):
    kw = dict(
        proposed_separation_date=SEP, notice_date=NOTICE,
        total_company_headcount=400, service_coordination="lwdb",
        lwdb_email="board@example.gov", lwdb_phone="(555) 555-0100",
        employer_contact_email="hr@acme.com",
        employer_contact_phone="(555) 555-0199",
    )
    kw.update(over)
    return ComplianceEngine(ComplianceConfig(**kw)).run(scores)


def pay_for(scores: pd.DataFrame):
    cut = scores.loc[scores["selected"]]
    return SeverancePayEngine(PayConfig(
        separation_date=SEP, leave_policy="separate",
        formula=SeveranceFormula(weeks_per_year=2.0, min_weeks=4, max_weeks=26),
    )).run(cut)


def cfg(**kw) -> DocumentConfig:
    defaults = dict(
        employer_name="Acme Inc.",
        employer_address="1 Market St, San Francisco, CA 94105",
        employer_contact_name="Dana Reyes",
        employer_contact_email="hr@acme.com",
        employer_contact_phone="(555) 555-0199",
        signatory_name="Dana Reyes", signatory_title="VP, People",
        lwdb_name="SF Workforce Board", lwdb_email="board@example.gov",
        lwdb_phone="(555) 555-0100", service_coordination="lwdb",
    )
    defaults.update(kw)
    return DocumentConfig(**defaults)


def generate(scores=None, config=None, comp=None, **kw):
    scores = population() if scores is None else scores
    comp = comp or compliance(scores)
    return DocumentGenerator(config or cfg()).generate(
        compliance=comp, scores=scores, pay=pay_for(scores), **kw
    )


def codes(docs) -> set[str]:
    return {f["code"] for f in docs.findings}


def body_of(docs, doc_type: str) -> str:
    d = docs.by_type(doc_type)
    return d[0].body if d else ""


# --- the gate --------------------------------------------------------------


def test_generation_requires_a_compliance_analysis():
    docs = DocumentGenerator(cfg()).generate(compliance=None)
    assert docs.blocked
    assert "NO_COMPLIANCE_INPUT" in codes(docs)
    assert not docs.documents


def test_blocked_gate_produces_nothing():
    scores = population(selected=5, leave_status="CFRA")
    docs = generate(scores)
    assert docs.blocked
    assert not docs.documents
    assert "GATE_BLOCKED" in codes(docs)


def test_blocked_summary_explains_why_nothing_was_generated():
    docs = generate(population(selected=5, union_flag=True))
    md = docs.summary_markdown()
    assert "BLOCKED — nothing was generated" in md
    assert "dated record of the defect" in md


def test_legal_judgment_blockers_clear_with_recorded_counsel_signoff():
    scores = population(selected=5, union_flag=True)
    docs = generate(scores, cfg(
        counsel_override_by="R. Alvarez, Counsel",
        counsel_override_reason="CBA reviewed; no bumping rights apply.",
    ))
    assert not docs.blocked
    assert docs.documents
    assert "GATE_OVERRIDDEN" in codes(docs)
    assert docs.override_record["by"] == "R. Alvarez, Counsel"


def test_override_requires_both_a_name_and_a_reason():
    scores = population(selected=5, union_flag=True)
    assert generate(scores, cfg(counsel_override_by="Counsel")).blocked
    assert generate(scores, cfg(counsel_override_reason="Because.")).blocked


def test_data_completeness_blockers_cannot_be_overridden():
    """No sign-off fills in a blank pay rate; the document would be wrong."""
    scores = population(selected=5)
    scores.loc[0, "annualized_pay"] = None
    docs = generate(scores, cfg(
        counsel_override_by="Counsel",
        counsel_override_reason="We need these today.",
    ))
    assert docs.blocked
    assert not docs.documents
    assert "DATA_COMPLETENESS_BLOCKED" in codes(docs)


def test_missing_establishment_column_is_non_overridable():
    scores = population(selected=60).drop(columns=["worksite_name"])
    docs = generate(scores, cfg(
        counsel_override_by="Counsel", counsel_override_reason="Proceed.",
    ))
    assert docs.blocked
    assert "DATA_COMPLETENESS_BLOCKED" in codes(docs)


def test_non_overridable_set_covers_the_data_blockers():
    for code in ("FINAL_PAY_UNCOMPUTABLE", "NO_ESTABLISHMENT_COLUMN",
                 "SB617_LWDB_CONTACT_MISSING", "LEAVE_POLICY_UNDECLARED"):
        assert code in NON_OVERRIDABLE_CODES


def test_override_is_stamped_into_every_generated_document():
    scores = population(selected=5, union_flag=True)
    docs = generate(scores, cfg(
        counsel_override_by="R. Alvarez",
        counsel_override_reason="Reviewed and cleared.",
    ))
    assert docs.documents
    for d in docs.documents:
        assert "Compliance gate cleared by R. Alvarez" in d.body


def test_override_appears_in_the_summary_manifest():
    scores = population(selected=5, union_flag=True)
    docs = generate(scores, cfg(
        counsel_override_by="R. Alvarez", counsel_override_reason="Cleared.",
    ))
    md = docs.summary_markdown()
    assert "Counsel sign-off recorded" in md
    assert "R. Alvarez" in md


# --- draft discipline ------------------------------------------------------


def test_every_document_carries_the_draft_banner():
    docs = generate()
    assert docs.documents
    for d in docs.documents:
        assert "DRAFT — NOT FOR DISTRIBUTION" in d.body


def test_all_documents_are_drafts_finding_is_always_raised():
    docs = generate()
    assert "ALL_DOCUMENTS_ARE_DRAFTS" in codes(docs)


def test_severance_agreement_is_a_skeleton_not_a_signable_release():
    docs = generate()
    body = body_of(docs, "severance_agreement_draft")
    assert "SKELETON FOR COUNSEL" in body
    assert "This is not a contract" in body
    assert "must be drafted by a lawyer" in body


def test_severance_skeleton_enumerates_owbpa_elements():
    docs = generate()
    body = body_of(docs, "severance_agreement_draft")
    for element in ("days to consider", "days to revoke", "consult an attorney",
                    "ADEA"):
        assert element in body


def test_severance_skeleton_lists_prohibited_terms():
    docs = generate()
    body = body_of(docs, "severance_agreement_draft")
    assert "What must not be in it" in body
    assert "waiver of the right to file a charge" in body
    assert "206.5" in body


def test_placeholders_are_detected_and_reported():
    docs = generate(config=cfg(signatory_name="", employer_contact_name=""))
    assert docs.incomplete
    assert "PLACEHOLDERS_REMAIN" in codes(docs)
    ph = {p for d in docs.documents for p in d.placeholders}
    assert "SIGNATORY_NAME" in ph


def test_placeholders_are_left_visible_rather_than_guessed():
    docs = generate(config=cfg(employer_address=""))
    md = docs.summary_markdown()
    assert "left visible rather than guessed" in md


# --- Cal-WARN notice -------------------------------------------------------


def test_warn_notice_is_generated_when_triggered():
    scores = population(n=200, selected=60)
    docs = generate(scores)
    assert docs.by_type("warn_notice_employee")


def test_no_warn_notice_when_not_triggered():
    docs = generate(population(n=200, selected=5))
    assert not docs.by_type("warn_notice_employee")
    assert "NO_WARN_NOTICE" in codes(docs)


def test_no_warn_notice_message_warns_the_determination_can_change():
    docs = generate(population(n=200, selected=5))
    msg = next(f["message"] for f in docs.findings if f["code"] == "NO_WARN_NOTICE")
    assert "30 days" in msg


def test_warn_notice_contains_all_four_sb617_disclosures():
    docs = generate(population(n=200, selected=60))
    body = body_of(docs, "warn_notice_employee")
    # 1. coordination election
    assert "coordinate transition services" in body
    # 2. LWDB contact + rapid response description
    assert "board@example.gov" in body
    assert "rapid response" in body.lower()
    # 3. CalFresh
    assert "CalFresh" in body
    assert "1-877-847-3663" in body
    assert "getcalfresh.org" in body
    # 4. employer contact
    assert "hr@acme.com" in body


def test_warn_notice_states_bumping_rights_either_way():
    no_bump = body_of(generate(population(n=200, selected=60)), "warn_notice_employee")
    assert "do not have bumping rights" in no_bump
    yes_bump = body_of(
        generate(population(n=200, selected=60), cfg(bumping_rights=True)),
        "warn_notice_employee",
    )
    assert "collective bargaining agreement" in yes_bump


def test_undeclared_coordination_leaves_a_visible_placeholder():
    docs = generate(population(n=200, selected=60), cfg(service_coordination=""))
    body = body_of(docs, "warn_notice_employee")
    assert "[[SERVICE_COORDINATION_STATEMENT]]" in body


def test_agency_notices_go_to_all_three_recipients():
    docs = generate(population(n=200, selected=60))
    agency = docs.by_type("warn_notice_agency")
    assert len(agency) == 3
    audiences = " ".join(d.audience for d in agency).lower()
    assert "employment development" in audiences
    assert "workforce" in audiences
    assert "elected official" in audiences


def test_warn_notice_includes_a_counsel_review_checklist():
    docs = generate(population(n=200, selected=60))
    body = body_of(docs, "warn_notice_employee")
    assert "Counsel review checklist" in body
    assert "12-month employment history" in body


# --- separation letters ----------------------------------------------------


def test_one_separation_letter_per_affected_employee():
    docs = generate(population(selected=7))
    assert len(docs.by_type("separation_letter")) == 7


def test_separation_letter_states_final_pay_is_unconditional():
    docs = generate()
    body = docs.by_type("separation_letter")[0].body
    assert "regardless of whether you accept" in body
    assert "your earned wages are not" in body.lower()


def test_separation_letter_carries_the_computed_severance_figure():
    docs = generate(population(selected=1))
    body = docs.by_type("separation_letter")[0].body
    assert "separation pay of" in body.lower()
    assert "$" in body


def test_separation_letter_does_not_state_a_performance_reason():
    docs = generate()
    body = docs.by_type("separation_letter")[0].body.lower()
    assert "is not a statement about your conduct" in body
    assert "poor performance" not in body


def test_final_wages_are_left_as_a_placeholder_not_invented():
    docs = generate()
    body = docs.by_type("separation_letter")[0].body
    assert "[[FINAL_WAGES]]" in body


# --- OWBPA disclosure ------------------------------------------------------


def test_owbpa_disclosure_requires_an_explicit_decisional_unit():
    docs = generate()
    assert not docs.by_type("owbpa_disclosure")
    assert "DECISIONAL_UNIT_NOT_DEFINED" in codes(docs)


def test_decisional_unit_is_never_inferred_from_comparison_groups():
    docs = generate()
    msg = next(f["message"] for f in docs.findings
               if f["code"] == "DECISIONAL_UNIT_NOT_DEFINED")
    assert "not the same as the comparison groups" in msg
    assert "will not infer" in msg


def test_owbpa_disclosure_lists_selected_and_non_selected():
    docs = generate(config=cfg(decisional_unit="All Engineering at SF HQ"))
    body = body_of(docs, "owbpa_disclosure")
    assert "Employees selected for the program" in body
    assert "not selected" in body


def test_owbpa_disclosure_contains_titles_and_ages_but_no_names():
    scores = population(selected=5)
    docs = generate(scores, cfg(decisional_unit="All Engineering at SF HQ"))
    body = body_of(docs, "owbpa_disclosure")
    assert "| Job title | Age |" in body
    assert "Pat Doe" not in body
    assert "E0" not in body


def test_owbpa_disclosure_needs_the_full_population_not_just_the_cut_list():
    scores = population(selected=5)
    comp = compliance(scores)
    docs = DocumentGenerator(cfg(decisional_unit="Unit")).generate(
        compliance=comp, scores=None, pay=pay_for(scores),
        selection=type("S", (), {"cut_list": scores.loc[scores["selected"]]})(),
    )
    assert "OWBPA_NEEDS_FULL_POPULATION" in codes(docs)


def test_owbpa_disclosure_can_be_scoped_by_column():
    scores = population(n=60, selected=5)
    scores.loc[30:, "department"] = "Sales"
    docs = generate(scores, cfg(
        decisional_unit="Engineering only",
        decisional_unit_column="department",
        decisional_unit_value="Engineering",
    ))
    body = body_of(docs, "owbpa_disclosure")
    assert "Engineering only" in body


def test_empty_decisional_unit_scope_is_an_error():
    docs = generate(config=cfg(
        decisional_unit="Nobody",
        decisional_unit_column="department",
        decisional_unit_value="Nonexistent",
    ))
    assert "DECISIONAL_UNIT_EMPTY" in codes(docs)


# --- manager script --------------------------------------------------------


def test_manager_script_forbids_age_and_retirement_references():
    body = body_of(generate(), "manager_script")
    assert "Do not say" in body
    assert "age, retirement" in body
    assert "next chapter" in body


def test_manager_script_forbids_leave_and_health_references():
    body = body_of(generate(), "manager_script")
    assert "health, leave, disability, or pregnancy" in body


def test_manager_script_forbids_improvising_a_reason():
    body = body_of(generate(), "manager_script")
    assert "do not improvise a reason" in body.lower()
    assert "contradicts the record" in body


def test_manager_script_forbids_promises_and_negotiation():
    body = body_of(generate(), "manager_script")
    assert "Promises." in body
    assert "Negotiation." in body


def test_manager_script_handles_threatened_legal_action():
    body = body_of(generate(), "manager_script")
    assert "threatens legal action" in body
    assert "end the meeting" in body.lower()


def test_manager_script_covers_employee_distress():
    body = body_of(generate(), "manager_script")
    assert "becomes upset" in body
    assert "escalation procedure" in body


# --- employee FAQ ----------------------------------------------------------


def test_faq_states_final_pay_does_not_require_signing():
    body = body_of(generate(), "employee_faq")
    assert "do not have to sign anything to receive it" in body


def test_faq_explains_sick_leave_is_generally_not_paid_out():
    body = body_of(generate(), "employee_faq")
    assert "unused sick leave" in body.lower()
    assert "Generally no" in body


def test_faq_gives_real_unemployment_and_benefit_resources():
    body = body_of(generate(), "employee_faq")
    assert "edd.ca.gov" in body
    assert "coveredca.com" in body
    assert "getcalfresh.org" in body


def test_faq_says_signing_is_optional_and_affects_nothing_else():
    body = body_of(generate(), "employee_faq")
    assert "It is optional" in body
    assert "not signing does not affect" in body.lower()


def test_faq_advises_the_right_to_consult_an_attorney():
    body = body_of(generate(), "employee_faq")
    assert "consult an attorney" in body


# --- checklist and summary -------------------------------------------------


def test_checklist_lists_dated_obligations_from_compliance():
    body = body_of(generate(), "notice_day_checklist")
    assert "Before notice day" in body
    assert "Lab. Code" in body


def test_checklist_puts_final_pay_before_the_first_meeting():
    body = body_of(generate(), "notice_day_checklist")
    assert "Final paychecks physically available before the first meeting" in body


def test_checklist_warns_against_processing_a_release_early():
    body = body_of(generate(), "notice_day_checklist")
    assert "revocation period expires" in body


def test_hr_summary_reminds_that_the_record_is_discoverable():
    body = body_of(generate(), "hr_summary")
    assert "discoverable" in body
    assert "not pursued" in body


# --- outputs ---------------------------------------------------------------


def test_include_filter_limits_document_types():
    docs = generate(config=cfg(include=("employee_faq",)))
    assert {d.doc_type for d in docs.documents} == {"employee_faq"}


def test_manifest_lists_every_document():
    docs = generate()
    m = docs.manifest()
    assert len(m) == len(docs.documents)
    assert "placeholders" in m.columns


def test_write_creates_files_grouped_by_type():
    docs = generate(population(selected=2))
    with tempfile.TemporaryDirectory() as tmp:
        paths = docs.write(tmp)
        assert paths["summary"].exists()
        assert paths["manifest"].exists()
        assert (Path(tmp) / "separation_letter").is_dir()
        assert len(list((Path(tmp) / "separation_letter").glob("*.md"))) == 2


def test_blocked_run_still_writes_a_summary_explaining_why():
    docs = generate(population(selected=5, leave_status="CFRA"))
    with tempfile.TemporaryDirectory() as tmp:
        paths = docs.write(tmp)
        text = paths["summary"].read_text()
        assert "BLOCKED" in text
        assert len(list(Path(tmp).glob("*/*.md"))) == 0


def test_summary_states_the_severance_agreement_is_not_signable():
    docs = generate()
    md = docs.summary_markdown()
    assert "not a signable instrument" in md


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
