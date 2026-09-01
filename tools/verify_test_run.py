"""
verify_test_run.py
==================

Runs the full pipeline on a generated test roster and checks that the
deliberately seeded edge cases actually surfaced. This is a smoke test for the
*pipeline*, distinct from the unit tests: it answers "did the guardrails fire
on realistic data" rather than "does each function work".

    python make_test_roster.py --out test_roster.csv
    python verify_test_run.py test_roster.csv plan_test.yaml
"""

from __future__ import annotations

# Make the package importable whether this file is run directly, via pytest, or
# from another working directory.
import sys as _sys
from pathlib import Path as _Path
_root = _Path(__file__).resolve().parent.parent
if str(_root) not in _sys.path:
    _sys.path.insert(0, str(_root))

import sys
from pathlib import Path

import pandas as pd

from rif_copilot.selection_criteria import SelectionEngine, load_plan
from rif_copilot.workforce_data import Severity, load_workforce_csv


def check(label: str, ok: bool, detail: str = "") -> bool:
    mark = "PASS" if ok else "MISS"
    print(f"  [{mark}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main(roster_path: str, plan_path: str) -> int:
    ingest = load_workforce_csv(roster_path, as_of="2026-10-30")
    codes1 = {i.code for i in ingest.report.issues}

    print("\n=== Module 1: Data Manager ===")
    print(f"  {ingest.report.row_count} rows, {len(ingest.report.errors)} errors, "
          f"{len(ingest.report.warnings)} warnings, "
          f"{len(ingest.clean)} clean rows\n")

    results = [
        check("Unparseable dates caught", "INVALID_DATE" in codes1),
        check("Duplicate employee ID caught", "DUPLICATE_EMPLOYEE_ID" in codes1),
        check("Blank employee ID caught", "MISSING_EMPLOYEE_ID" in codes1),
        check("Sub-minimum wage caught", "BELOW_MINIMUM_WAGE" in codes1),
        check("Exempt below salary floor caught", "EXEMPT_BELOW_SALARY_FLOOR" in codes1),
        check("Implausible PTO caught", "IMPLAUSIBLE_VACATION_BALANCE" in codes1),
        check("Excel serial date converted", "EXCEL_SERIAL_DATE" in codes1),
        check("Out-of-state employee caught", "OUT_OF_STATE_EMPLOYEES" in codes1),
        check("Employees on leave surfaced", "EMPLOYEES_ON_LEAVE" in codes1),
    ]

    plan = load_plan(plan_path)
    result = SelectionEngine(plan).run(ingest.data)
    codes2 = {f.code for f in result.report.findings}
    s = result.report.summary()

    print("\n=== Module 2: Selection Criteria Engine ===")
    print(f"  {s['eligible_employees']} evaluated, {s['selected_employees']} selected, "
          f"${s['achieved_savings']:,.0f} of ${s['cost_savings_target']:,.0f}\n")

    results += [
        check("Upstream data errors excluded", "BLOCKING_DATA_ERRORS_EXCLUDED" in codes2),
        check("Unrated employees routed to review", "CRITERION_DATA_GAPS" in codes2),
        check("Sole-incumbent group blocked", "DEGENERATE_COMPARISON_GROUP" in codes2),
        check("Boundary tie refused", "TIE_AT_CUT_BOUNDARY" in codes2),
        check("Legal review flags applied", "SELECTIONS_REQUIRE_LEGAL_REVIEW" in codes2),
        check("Protected-field exclusion asserted", "PROTECTED_FIELDS_EXCLUDED" in codes2),
    ]

    # Invariants that must hold on any run.
    print("\n=== Invariants ===")
    scores = result.scores
    cut = result.cut_list

    protected_cols = {"gender", "race_ethnicity", "age_years", "age_40_plus",
                      "disability_status", "veteran_status"}
    breakdown = " ".join(scores["score_breakdown"].astype(str).tolist())
    results.append(check(
        "No protected field appears in any score breakdown",
        not any(c in breakdown for c in protected_cols),
    ))
    results.append(check(
        "No rank-order violation", "RANK_ORDER_VIOLATION" not in codes2,
    ))

    protected_pos = scores.loc[scores["selection_status"] == "protected_position"]
    results.append(check(
        "No protected position selected",
        not protected_pos["selected"].any(),
        f"{len(protected_pos)} protected",
    ))

    unrated = scores.loc[scores["selection_status"] == "excluded_insufficient_data"]
    results.append(check(
        "No unrated employee auto-selected",
        not unrated["selected"].any(),
        f"{len(unrated)} unrated",
    ))

    # Age skew: module 2 must NOT have used age, but the planted skew should
    # still show up in the outcome. That gap is the whole reason module 3 exists.
    # --- Module 4 -------------------------------------------------------
    from rif_copilot.adverse_impact import AdverseImpactAnalyzer
    analysis = AdverseImpactAnalyzer().run(result.scores, scenario=plan.plan_name)
    codes4 = {f.code for f in analysis.report.findings}
    verdicts = analysis.report.class_verdicts()

    print("\n=== Module 4: Adverse Impact Analyzer ===")
    for cls, v in verdicts.items():
        print(f"  {cls:18} {v}")
    print()

    results.append(check(
        "Planted age skew detected",
        verdicts.get("Age 40+") in ("Impact indicated", "Review"),
        f"verdict: {verdicts.get('Age 40+')}",
    ))
    results.append(check(
        "Small groups skipped rather than flagged",
        "COMPARISONS_TOO_SMALL_TO_TEST" in codes4,
    ))
    results.append(check(
        "No-swap guidance present", "NO_SWAP_RECOMMENDATIONS" in codes4,
    ))
    results.append(check(
        "Every flagged comparison came from an adequate group",
        all(c.interpretable or c.severity != "ERROR"
            for c in analysis.report.comparisons),
    ))
    results.append(check(
        "Privilege warning on the report",
        "privileged" in analysis.report.to_markdown().lower(),
    ))

    # --- Box 2: Scenario Simulator --------------------------------------
    from rif_copilot.scenario_simulator import ScenarioSimulator, load_scenarios
    from pathlib import Path as _P

    scen_path = _P(plan_path).with_name("scenarios.yaml")
    if scen_path.exists():
        scenarios = load_scenarios(scen_path)
        sim = ScenarioSimulator().run(ingest.data, scenarios)
        codes2b = {f["code"] for f in sim.report.findings}

        print("\n=== Box 2: Scenario Simulator ===")
        for _, row in sim.comparison.iterrows():
            print(f"  {row['scenario'][:26]:28} cut {row['headcount_reduction']:>3}  "
                  f"${row['annualized_savings']:>10,.0f}  "
                  f"impact_indicated {row['impact_indicated']}")
        print()

        results.append(check(
            "Every scenario carries a business rationale",
            all(s.rationale.strip() for s in scenarios),
        ))
        results.append(check(
            "No composite ranking column emitted",
            not any(k in c.lower() for c in sim.comparison.columns
                    for k in ("score", "rank", "recommended", "best")),
        ))
        results.append(check(
            "Unresolvable org structure reported, not zeroed",
            "ORG_STRUCTURE_UNRESOLVABLE" in codes2b,
        ))
        results.append(check(
            "Weight-only variants flagged for demographic movement",
            "WEIGHT_CHANGE_MOVES_DEMOGRAPHICS" in codes2b,
        ))
        results.append(check(
            "Scenario report records discarded scenarios",
            "including those not pursued" in sim.report.to_markdown().lower(),
        ))

    # --- Box 5: CA Compliance Engine ------------------------------------
    from rif_copilot.ca_compliance import ComplianceConfig, ComplianceEngine

    comp_cfg = ComplianceConfig(
        proposed_separation_date="2026-10-30",
        notice_date="2026-08-19",
        total_company_headcount=400,
        service_coordination="lwdb",
        lwdb_email="board@example.gov",
        lwdb_phone="(555) 555-0100",
        employer_contact_email="hr@acme.com",
        employer_contact_phone="(555) 555-0199",
    )
    comp = ComplianceEngine(comp_cfg).run(
        result.scores, impact=analysis, selection=result, scenario=plan.plan_name
    )
    codes5 = {f.code for f in comp.report.findings}

    print("\n=== Box 5: CA Compliance Engine ===")
    print(f"  WARN triggered: {comp.report.warn_triggered}")
    print(f"  Obligations: {len(comp.report.obligations)}   "
          f"Missed deadlines: {len(comp.report.missed_deadlines)}")
    print(f"  Document gate: "
          f"{'CLEAR' if comp.gate.may_generate_documents else 'BLOCKED'}")
    for b in comp.gate.blockers[:4]:
        print(f"    - {b[:96]}")
    print()

    results.append(check(
        "Gate blocks on indicated adverse impact",
        (not comp.gate.may_generate_documents)
        if analysis.report.indicated else comp.gate.may_generate_documents,
    ))
    results.append(check(
        "Final pay obligation dated to separation",
        any(o.code == "FINAL_PAY" and o.due_date is not None
            for o in comp.report.obligations),
    ))
    results.append(check(
        "Every obligation cites an authority",
        all(o.authority.strip() for o in comp.report.obligations),
    ))
    results.append(check(
        "Wages figure not fabricated",
        comp.report.final_pay.get("wages_through_separation") is None,
    ))
    results.append(check(
        "No threshold-avoidance guidance in output",
        not any(x in (comp.report.to_markdown() + comp.report.to_json()).lower()
                for x in ("stay under the threshold", "avoid triggering warn",
                          "keep the layoff below")),
    ))

    # --- Box 6: Severance & Pay Engine ----------------------------------
    from rif_copilot.severance_pay import PayConfig, SeveranceFormula, SeverancePayEngine

    pay_cfg = PayConfig(
        separation_date="2026-10-30",
        leave_policy="separate",
        formula=SeveranceFormula(weeks_per_year=2.0, min_weeks=4, max_weeks=26),
    )
    pay = SeverancePayEngine(pay_cfg).run(result.cut_list, scenario=plan.plan_name)
    codes6 = {f.code for f in pay.report.findings}
    t6 = pay.report.totals

    print("\n=== Box 6: Severance & Pay Engine ===")
    print(f"  Severance          ${t6['severance_gross']:>12,.0f}")
    print(f"  Vacation payout    ${t6['vacation_payout']:>12,.0f}")
    print(f"  Employer tax       ${t6['employer_tax']:>12,.0f}")
    print(f"  Total employer     ${t6['total_employer_cost']:>12,.0f}")
    print(f"  Median weeks       {t6['median_weeks']:>13.1f}")
    print()

    results.append(check(
        "Leave policy required rather than guessed",
        "LEAVE_POLICY_UNDECLARED" in {
            f.code for f in SeverancePayEngine(
                PayConfig(separation_date="2026-10-30", leave_policy="")
            ).run(result.cut_list).report.findings
        },
    ))
    results.append(check(
        "Final wages not fabricated",
        t6["final_wages"] is None and "FINAL_WAGES_NOT_COMPUTED" in codes6,
    ))
    results.append(check(
        "Release cannot cover earned wages",
        "RELEASE_CANNOT_COVER_EARNED_WAGES" in codes6,
    ))
    results.append(check(
        "Severance formula applied uniformly (no silent overrides)",
        not pay.register["overridden"].any(),
    ))
    results.append(check(
        "Every computed employee has a severance figure",
        bool((pay.register["status"] == "computed").all()),
        f"{int((pay.register['status'] != 'computed').sum())} uncomputable",
    ))

    # --- Box 7: Document Generator --------------------------------------
    from rif_copilot.document_generator import DocumentConfig, DocumentGenerator

    doc_cfg = DocumentConfig(
        employer_name="Acme Inc.", employer_address="1 Market St, SF, CA",
        employer_contact_name="Dana Reyes", employer_contact_email="hr@acme.com",
        employer_contact_phone="(555) 555-0199",
        signatory_name="Dana Reyes", signatory_title="VP, People",
        lwdb_name="SF Workforce Board", lwdb_email="board@example.gov",
        lwdb_phone="(555) 555-0100", service_coordination="lwdb",
    )
    docs = DocumentGenerator(doc_cfg).generate(
        compliance=comp, selection=result, pay=pay, scores=result.scores
    )

    print("\n=== Box 7: Document Generator ===")
    print(f"  Gate: {'BLOCKED' if docs.blocked else 'CLEAR'}   "
          f"Documents generated: {len(docs.documents)}")
    for b in docs.blockers[:3]:
        print(f"    - {b[:92]}")
    print()

    results.append(check(
        "Blocked compliance gate produces zero documents",
        docs.blocked and not docs.documents,
    ))

    # Counsel sign-off clears legal-judgment blockers only.
    cleared_cfg = DocumentConfig(
        **{**doc_cfg.__dict__,
           "counsel_override_by": "R. Alvarez, Employment Counsel",
           "counsel_override_reason": "Reviewed 2026-08-18; cleared to proceed.",
           "decisional_unit": "All Engineering employees at SF HQ"}
    )
    cleared = DocumentGenerator(cleared_cfg).generate(
        compliance=comp, selection=result, pay=pay, scores=result.scores
    )
    results.append(check(
        "Counsel sign-off clears legal-judgment blockers",
        (not cleared.blocked) and bool(cleared.documents),
        f"{len(cleared.documents)} documents",
    ))
    results.append(check(
        "Override stamped into every document",
        all("Compliance gate cleared by" in d.body for d in cleared.documents),
    ))
    results.append(check(
        "Every document marked DRAFT",
        all("DRAFT — NOT FOR DISTRIBUTION" in d.body for d in cleared.documents),
    ))
    results.append(check(
        "No signable release produced",
        all("SKELETON FOR COUNSEL" in d.body
            for d in cleared.by_type("severance_agreement_draft")),
    ))
    results.append(check(
        "OWBPA disclosure carries no employee names",
        all("Pat Doe" not in d.body and "full_name" not in d.body
            for d in cleared.by_type("owbpa_disclosure")),
    ))

    # --- Box 8: Approvals ------------------------------------------------
    from rif_copilot.approvals import ApprovalError, ApprovalLedger, ApprovalPackage

    pkg = ApprovalPackage.from_pipeline(
        scenario=plan.plan_name, plan=plan, selection=result,
        impact=analysis, compliance=comp, pay=pay,
    )
    led = ApprovalLedger()
    led.submit(pkg, submitted_by="M. Chen", role="HR Business Partner")

    print("\n=== Box 8: Approvals ===")
    print(f"  Version fingerprint: {pkg.fingerprint}")
    print(f"  Open blocker codes: {', '.join(pkg.blocker_codes) or 'none'}")

    def _refused(fn):
        try:
            fn()
            return False
        except ApprovalError:
            return True

    results.append(check(
        "Self-approval refused",
        _refused(lambda: led.approve("hr", "M. Chen", "HR Business Partner")),
    ))
    results.append(check(
        "Out-of-order approval refused",
        _refused(lambda: led.approve("executive", "J. Park", "CFO")),
    ))
    results.append(check(
        "Data blocker cannot be cleared by signature",
        _refused(lambda: led.approve(
            "legal", "R. Alvarez", "Employment Counsel",
            clears=["FINAL_PAY_UNCOMPUTABLE"], comment="Proceed.")),
    ))

    led.approve("hr", "S. Patel", "HR Director",
                comment="Selection applied as documented.",
                clears=["TIE_AT_CUT_BOUNDARY"])
    led.approve("legal", "R. Alvarez", "Employment Counsel",
                clears=[c for c in pkg.blocker_codes
                        if c in led.policy.stage("legal").can_clear],
                comment="Impact reviewed; criteria job-related. CBA checked.")
    led.approve("executive", "J. Park", "CFO", comment="Authorized.")
    st = led.status(pkg)
    print(f"  Chain complete: {st.complete}")
    print()

    results.append(check("Full chain completes", st.complete))

    # The invariant the module exists for.
    import copy as _copy
    changed_plan = _copy.deepcopy(plan)
    changed_plan.cost_savings_target = plan.cost_savings_target * 1.4
    changed_sel = SelectionEngine(changed_plan).run(ingest.data)
    pkg2 = ApprovalPackage.from_pipeline(
        scenario=plan.plan_name, plan=changed_plan, selection=changed_sel,
        impact=analysis, compliance=comp, pay=pay,
    )
    results.append(check(
        "Approval does not survive a change to the plan",
        not led.is_fully_approved(pkg2),
        f"{pkg.fingerprint} -> {pkg2.fingerprint}",
    ))

    docs_ok = DocumentGenerator(cleared_cfg).generate(
        compliance=comp, selection=result, pay=pay, scores=result.scores,
        approvals=led, package=pkg,
    )
    results.append(check(
        "Approved chain unlocks document generation",
        (not docs_ok.blocked) and bool(docs_ok.documents),
        f"{len(docs_ok.documents)} documents",
    ))
    docs_stale = DocumentGenerator(cleared_cfg).generate(
        compliance=comp, selection=changed_sel, pay=pay, scores=changed_sel.scores,
        approvals=led, package=pkg2,
    )
    results.append(check(
        "Stale approval blocks document generation",
        docs_stale.blocked and not docs_stale.documents,
    ))

    # --- Box 9: Task Tracker ---------------------------------------------
    from rif_copilot.task_tracker import TaskBoard, TaskError, TrackerConfig
    import datetime as _dt

    tb = TaskBoard.build(compliance=comp, pay=pay, documents=docs, package=pkg,
                         selection=result, config=TrackerConfig())
    bs = tb.summary()
    print("\n=== Box 9: Task Tracker ===")
    print(f"  Tasks {bs['total_tasks']}   overdue {bs['overdue']} "
          f"({bs['overdue_statutory']} statutory)   blocked {bs['blocked']}")
    print(f"  Acknowledgment slots: {bs['acknowledgments_total']}")
    print()

    def _refused_task(fn):
        try:
            fn()
            return False
        except TaskError:
            return True

    results.append(check(
        "Statutory deadline cannot be rescheduled",
        _refused_task(lambda: tb.reschedule(
            "STAT-FINAL_PAY", _dt.date(2027, 1, 1), by="HR", reason="late")),
    ))
    results.append(check(
        "Completion without evidence refused",
        _refused_task(lambda: tb.complete("STAT-FINAL_PAY", by="Payroll")),
    ))
    rev = [t for t in tb.tasks if t.id.endswith("-REVOKE")]
    results.append(check(
        "Revocation period cannot be closed early",
        bool(rev) and _refused_task(lambda: tb.complete(
            rev[0].id, by="HR", evidence="signed",
            today=rev[0].not_before - _dt.timedelta(days=1))),
    ))
    results.append(check(
        "Dependency order enforced",
        _refused_task(lambda: tb.complete(
            "DAY-DELIVER-PAY", by="HR", evidence="run #1")),
    ))
    followups = [t for t in tb.tasks if t.id.endswith("-FOLLOWUP")]
    considers = {t.employee_id: t for t in tb.tasks if t.id.endswith("-CONSIDER")}
    results.append(check(
        "No follow-up scheduled inside a consideration period",
        all(f.not_before is not None
            and considers.get(f.employee_id) is not None
            and f.not_before >= considers[f.employee_id].due_date
            for f in followups),
        f"{len(followups)} follow-up task(s)",
    ))

    # --- Box 10: Audit & Reporting ---------------------------------------
    from rif_copilot.audit_reporting import AuditEntry, AuditPackage

    audit = AuditPackage.assemble(
        scenario=plan.plan_name, ingest=ingest, selection=result,
        impact=analysis, compliance=comp, pay=pay, documents=docs,
        ledger=led, board=tb, package=pkg,
    )
    ac = audit.counts()
    comp_res = audit.completeness()
    print("\n=== Box 10: Audit & Reporting ===")
    print(f"  Entries {ac['entries']}   errors {ac['errors']}   "
          f"warnings {ac['warnings']}")
    print(f"  Completeness {comp_res.score}%   chain "
          f"{'intact' if audit.verify().intact else 'BROKEN'}")
    print()

    results.append(check("Audit chain verifies intact", audit.verify().intact))

    # Tamper detection: edit an inconvenient entry in place.
    import copy as _c
    tampered = _c.deepcopy(audit)
    idx = next((i for i, e in enumerate(tampered.log._entries)
                if e.severity == Severity.ERROR), 0)
    e0 = tampered.log._entries[idx]
    tampered.log._entries[idx] = AuditEntry(
        **{**e0.__dict__, "detail": "No issue found.", "severity": Severity.INFO}
    )
    results.append(check(
        "Editing an entry is detected",
        not tampered.verify().intact,
        f"broken at {tampered.verify().broken_at}",
    ))

    deleted = _c.deepcopy(audit)
    del deleted.log._entries[idx]
    results.append(check("Deleting an entry is detected", not deleted.verify().intact))

    summary_md = audit.executive_summary()
    results.append(check(
        "Every error appears in the executive summary",
        f"Every error in the record ({ac['errors']})" in summary_md,
    ))
    results.append(check(
        "Adverse impact is not omitted from the summary",
        ("Impact indicated" in summary_md) if analysis.report.indicated else True,
    ))
    results.append(check(
        "Record self-reports its own gaps",
        isinstance(comp_res.score, float)
        and (comp_res.complete or bool(comp_res.missing)),
        f"{len(comp_res.missing)} gap(s), {len(comp_res.weak)} weakness(es)",
    ))
    results.append(check(
        "Privilege classification present",
        bool(audit.privilege_summary()),
    ))

    passed = sum(bool(r) for r in results)
    print(f"\n{passed}/{len(results)} pipeline checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    roster = sys.argv[1] if len(sys.argv) > 1 else "test_roster.csv"
    plan = sys.argv[2] if len(sys.argv) > 2 else "plan_test.yaml"
    raise SystemExit(main(roster, plan))
