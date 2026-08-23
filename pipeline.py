"""
rif_pipeline.py
===============

Orchestrator for the California RIF Copilot.

Runs the boxes in dependency order, passes each one's output to the boxes that
need it, and stops at the first stage that cannot honestly proceed. With seven
modules the wiring itself has become a place to make mistakes — forgetting to
pass ``impact`` into the compliance engine silently weakens the gate — so the
wiring lives here rather than in each caller.

Stage order and what gates what:

    1. Data Manager          -> standardized roster
    2. Scenario Simulator    -> optional; compares scenarios
    3. Selection Criteria    -> scores and cut list
    4. Adverse Impact        -> measured on the selection
    6. Severance & Pay       -> money per employee
    5. CA Compliance         -> obligations; produces the document gate
    8. Approvals             -> HR -> Legal -> Executive, bound to a fingerprint
    7. Documents             -> refuses unless 5 and 8 permit

Box 4 runs before box 5 because compliance folds the impact finding into its
gate. Box 6 runs before box 5 so compliance can see whether pay is computable.
Box 7 runs last because it depends on everything.

Usage
-----
    from .pipeline import PipelineConfig, run_pipeline

    result = run_pipeline(PipelineConfig(
        roster_csv="roster.csv",
        plan_yaml="rif_plan.yaml",
        separation_date="2026-10-30",
        notice_date="2026-08-19",
        leave_policy="separate",
    ))
    print(result.summary_markdown())

CLI
---
    python rif_pipeline.py roster.csv --plan rif_plan.yaml \
        --separation-date 2026-10-30 --leave-policy separate --outdir ./out
"""

from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .adverse_impact import AdverseImpactAnalyzer
from .approvals import ApprovalLedger, ApprovalPackage
from .ca_compliance import ComplianceConfig, ComplianceEngine
from .document_generator import DocumentConfig, DocumentGenerator
from .selection_criteria import SelectionEngine, load_plan
from .audit_reporting import AuditPackage
from .task_tracker import TaskBoard, TrackerConfig
from .severance_pay import PayConfig, SeveranceFormula, SeverancePayEngine
from .workforce_data import Severity, load_workforce_csv

__all__ = ["PipelineConfig", "PipelineResult", "run_pipeline"]

__version__ = "1.0.0"


@dataclass
class PipelineConfig:
    roster_csv: str
    plan_yaml: str
    separation_date: str
    notice_date: str | None = None
    leave_policy: str = ""
    as_of: str | None = None

    total_company_headcount: int | None = None
    service_coordination: str = ""
    lwdb_name: str = ""
    lwdb_email: str = ""
    lwdb_phone: str = ""
    employer_name: str = ""
    employer_address: str = ""
    employer_contact_name: str = ""
    employer_contact_email: str = ""
    employer_contact_phone: str = ""
    signatory_name: str = ""
    signatory_title: str = ""
    decisional_unit: str = ""

    severance_weeks_per_year: float = 2.0
    severance_min_weeks: float = 4.0
    severance_max_weeks: float = 26.0

    #: An existing ledger. Without one the pipeline runs through box 7 and
    #: reports that documents are blocked pending approval, which is correct.
    ledger: ApprovalLedger | None = None
    submitted_by: str = ""
    submitter_role: str = ""


@dataclass
class PipelineResult:
    config: PipelineConfig
    ingest: Any = None
    selection: Any = None
    impact: Any = None
    pay: Any = None
    compliance: Any = None
    package: ApprovalPackage | None = None
    documents: Any = None
    board: Any = None
    audit: Any = None
    stopped_at: str | None = None
    stop_reason: str = ""
    stages: list[dict[str, Any]] = field(default_factory=list)

    def note(self, stage: str, ok: bool, detail: str) -> None:
        self.stages.append({"stage": stage, "ok": ok, "detail": detail})

    @property
    def completed(self) -> bool:
        return self.stopped_at is None

    def write(self, outdir: str | Path) -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        if self.ingest is not None:
            paths.update({f"box1_{k}": v for k, v in
                          self.ingest.write(outdir / "box1_data").items()})
        if self.selection is not None:
            paths.update({f"box3_{k}": v for k, v in
                          self.selection.write(outdir / "box3_selection").items()})
        if self.impact is not None:
            paths.update({f"box4_{k}": v for k, v in
                          self.impact.write(outdir / "box4_impact").items()})
        if self.pay is not None:
            paths.update({f"box6_{k}": v for k, v in
                          self.pay.write(outdir / "box6_pay").items()})
        if self.compliance is not None:
            paths.update({f"box5_{k}": v for k, v in
                          self.compliance.write(outdir / "box5_compliance").items()})
        if self.config.ledger is not None:
            sub = outdir / "box8_approvals"
            sub.mkdir(parents=True, exist_ok=True)
            self.config.ledger.to_json(sub / "approval_ledger.json")
            (sub / "approval_record.md").write_text(
                self.config.ledger.to_markdown(), encoding="utf-8"
            )
            self.config.ledger.to_dataframe().to_csv(
                sub / "approval_history.csv", index=False
            )
            paths["box8_ledger"] = sub / "approval_ledger.json"
        if self.documents is not None:
            paths.update({f"box7_{k}": v for k, v in
                          self.documents.write(outdir / "box7_documents").items()})
        if self.board is not None:
            paths.update({f"box9_{k}": v for k, v in
                          self.board.write(outdir / "box9_tasks").items()})
        if self.audit is not None:
            paths.update({f"box10_{k}": v for k, v in
                          self.audit.write(outdir / "box10_audit").items()})
        p = outdir / "PIPELINE_SUMMARY.md"
        p.write_text(self.summary_markdown(), encoding="utf-8")
        paths["summary"] = p
        return paths

    def summary_markdown(self) -> str:
        L: list[str] = []
        L.append("# RIF Pipeline Summary")
        L.append("")
        L.append(
            "> **Privileged and confidential — prepared at the direction of "
            "counsel.**"
        )
        L.append("")
        L.append(f"**Roster:** `{self.config.roster_csv}`  ")
        L.append(f"**Plan:** `{self.config.plan_yaml}`  ")
        L.append(f"**Separation date:** {self.config.separation_date}  ")
        L.append("")

        L.append("## Stages")
        L.append("")
        L.append("| Stage | Result |")
        L.append("|---|---|")
        for s in self.stages:
            mark = "ok" if s["ok"] else "**stopped**"
            L.append(f"| {s['stage']} | {mark} — {s['detail']} |")
        L.append("")

        if self.stopped_at:
            L.append(f"## Stopped at: {self.stopped_at}")
            L.append("")
            L.append(self.stop_reason)
            L.append("")
            L.append(
                "The pipeline stops rather than continuing with a degraded "
                "result, because every downstream box would inherit the problem "
                "and the output would look complete."
            )
            L.append("")

        if self.impact is not None:
            L.append("## Adverse impact")
            L.append("")
            for cls, verdict in self.impact.report.class_verdicts().items():
                L.append(f"- {cls}: **{verdict}**")
            L.append("")

        if self.compliance is not None:
            r = self.compliance.report
            L.append("## Compliance")
            L.append("")
            L.append(f"- Cal-WARN triggered: **{r.warn_triggered}**")
            L.append(f"- Obligations: {len(r.obligations)}")
            L.append(f"- Missed deadlines: {len(r.missed_deadlines)}")
            L.append(
                f"- Document gate: "
                f"**{'CLEAR' if self.compliance.gate.may_generate_documents else 'BLOCKED'}**"
            )
            for b in self.compliance.gate.blockers:
                L.append(f"  - {b}")
            L.append("")

        if self.pay is not None:
            t = self.pay.report.totals
            L.append("## Cost")
            L.append("")
            L.append(f"- Severance: ${t.get('severance_gross', 0):,.0f}")
            L.append(f"- Vacation payout: ${t.get('vacation_payout', 0):,.0f}")
            L.append(f"- Total employer cost: "
                     f"${t.get('total_employer_cost', 0):,.0f}")
            L.append("")

        if self.package is not None:
            L.append("## Approval")
            L.append("")
            L.append(f"- Version fingerprint: `{self.package.fingerprint}`")
            if self.config.ledger is not None:
                st = self.config.ledger.status(self.package)
                L.append(f"- Chain complete: **{st.complete}**")
                if st.blocked_reason:
                    L.append(f"- {st.blocked_reason}")
            else:
                L.append("- No approval ledger supplied; documents remain blocked.")
            L.append("")

        if self.board is not None:
            bs = self.board.summary()
            L.append("## Execution")
            L.append("")
            L.append(f"- Tasks: {bs['total_tasks']} ({bs['percent_complete']}% complete)")
            L.append(f"- Overdue: {bs['overdue']} "
                     f"({bs['overdue_statutory']} statutory)")
            L.append(f"- Blocked by dependencies: {bs['blocked']}")
            L.append("")

        if self.documents is not None:
            L.append("## Documents")
            L.append("")
            if self.documents.blocked:
                L.append("**Blocked.** No documents were generated.")
                for b in self.documents.blockers:
                    L.append(f"- {b}")
            else:
                L.append(f"{len(self.documents.documents)} draft document(s) "
                         f"generated.")
                if self.documents.incomplete:
                    L.append(f"{len(self.documents.incomplete)} contain unfilled "
                             f"placeholders.")
            L.append("")

        if self.audit is not None:
            c = self.audit.counts()
            comp_res = self.audit.completeness()
            L.append("## Decision record")
            L.append("")
            L.append(f"- Log entries: {c['entries']} "
                     f"({c['errors']} errors, {c['warnings']} warnings)")
            L.append(f"- Integrity: "
                     f"{'chain intact' if self.audit.verify().intact else 'BROKEN'}")
            L.append(f"- Completeness: {comp_res.score}%")
            for item, why in comp_res.missing:
                L.append(f"  - gap: {item} — {why}")
            L.append("")

        L.append("---")
        L.append(
            "_Screening and drafting support only. Not legal, tax, or payroll "
            "advice. Every determination is an input to a lawyer's analysis._"
        )
        return "\n".join(L)


def run_pipeline(cfg: PipelineConfig) -> PipelineResult:
    result = PipelineResult(config=cfg)

    # -- box 1 ---------------------------------------------------------------
    plan = load_plan(cfg.plan_yaml)
    as_of = cfg.as_of or (str(plan.as_of_date) if plan.as_of_date else cfg.separation_date)
    ingest = load_workforce_csv(cfg.roster_csv, as_of=as_of)
    result.ingest = ingest
    if ingest.report.is_blocking:
        result.stopped_at = "Box 1 — Data Manager"
        result.stop_reason = (
            "The roster has blocking errors: "
            + "; ".join(i.message for i in ingest.report.errors[:3])
        )
        result.note("Box 1 — Data Manager", False, result.stop_reason[:80])
        return result
    result.note(
        "Box 1 — Data Manager", True,
        f"{ingest.report.row_count} rows, {len(ingest.clean)} clean, "
        f"{len(ingest.report.errors)} errors",
    )

    # -- box 3 ---------------------------------------------------------------
    selection = SelectionEngine(plan).run(ingest.data)
    result.selection = selection
    if selection.cut_list.empty:
        result.stopped_at = "Box 3 — Selection Criteria"
        result.stop_reason = "No employees were selected; there is nothing to process."
        result.note("Box 3 — Selection Criteria", False, result.stop_reason)
        return result
    result.note(
        "Box 3 — Selection Criteria", True,
        f"{len(selection.cut_list)} selected, "
        f"${selection.report.achieved_savings:,.0f} savings, "
        f"{len(selection.review_queue)} in review queue",
    )

    # -- box 4 ---------------------------------------------------------------
    impact = AdverseImpactAnalyzer().run(selection.scores, scenario=plan.plan_name)
    result.impact = impact
    result.note(
        "Box 4 — Adverse Impact", True,
        f"{len(impact.report.indicated)} indicated, "
        f"{len(impact.report.flagged)} flagged",
    )

    # -- box 6 ---------------------------------------------------------------
    pay = SeverancePayEngine(PayConfig(
        separation_date=cfg.separation_date,
        leave_policy=cfg.leave_policy,
        formula=SeveranceFormula(
            weeks_per_year=cfg.severance_weeks_per_year,
            min_weeks=cfg.severance_min_weeks,
            max_weeks=cfg.severance_max_weeks,
        ),
    )).run(selection.cut_list, scenario=plan.plan_name)
    result.pay = pay
    if pay.register.empty and pay.report.has_errors:
        result.stopped_at = "Box 6 — Severance & Pay"
        result.stop_reason = (
            "Pay could not be computed: "
            + "; ".join(f.message for f in pay.report.findings
                        if f.severity == Severity.ERROR)[:300]
        )
        result.note("Box 6 — Severance & Pay", False, result.stop_reason[:80])
        return result
    result.note(
        "Box 6 — Severance & Pay", True,
        f"${pay.report.totals.get('total_employer_cost', 0):,.0f} total employer cost",
    )

    # -- box 5 ---------------------------------------------------------------
    compliance = ComplianceEngine(ComplianceConfig(
        proposed_separation_date=cfg.separation_date,
        notice_date=cfg.notice_date,
        total_company_headcount=cfg.total_company_headcount,
        service_coordination=cfg.service_coordination,
        lwdb_name=cfg.lwdb_name, lwdb_email=cfg.lwdb_email,
        lwdb_phone=cfg.lwdb_phone,
        employer_name=cfg.employer_name,
        employer_contact_email=cfg.employer_contact_email,
        employer_contact_phone=cfg.employer_contact_phone,
    )).run(selection.scores, impact=impact, selection=selection,
           scenario=plan.plan_name)
    result.compliance = compliance
    result.note(
        "Box 5 — CA Compliance", True,
        f"WARN {'triggered' if compliance.report.warn_triggered else 'not triggered'}, "
        f"gate {'CLEAR' if compliance.gate.may_generate_documents else 'BLOCKED'}",
    )

    # -- box 8 ---------------------------------------------------------------
    package = ApprovalPackage.from_pipeline(
        scenario=plan.plan_name, plan=plan, selection=selection,
        impact=impact, compliance=compliance, pay=pay,
    )
    result.package = package
    if cfg.ledger is not None and cfg.submitted_by:
        cfg.ledger.submit(package, submitted_by=cfg.submitted_by,
                          role=cfg.submitter_role)
    if cfg.ledger is not None:
        st = cfg.ledger.status(package)
        result.note(
            "Box 8 — Approvals", True,
            f"version {package.fingerprint}, "
            f"{'complete' if st.complete else st.blocked_reason[:60]}",
        )
    else:
        result.note(
            "Box 8 — Approvals", True,
            f"version {package.fingerprint}, no ledger supplied",
        )

    # -- box 7 ---------------------------------------------------------------
    docs = DocumentGenerator(DocumentConfig(
        employer_name=cfg.employer_name,
        employer_address=cfg.employer_address,
        employer_contact_name=cfg.employer_contact_name,
        employer_contact_email=cfg.employer_contact_email,
        employer_contact_phone=cfg.employer_contact_phone,
        signatory_name=cfg.signatory_name, signatory_title=cfg.signatory_title,
        lwdb_name=cfg.lwdb_name, lwdb_email=cfg.lwdb_email,
        lwdb_phone=cfg.lwdb_phone,
        service_coordination=cfg.service_coordination,
        decisional_unit=cfg.decisional_unit,
    )).generate(
        compliance=compliance, selection=selection, pay=pay,
        scores=selection.scores,
        approvals=cfg.ledger, package=package if cfg.ledger else None,
    )
    result.documents = docs
    result.note(
        "Box 7 — Documents", True,
        "blocked" if docs.blocked else f"{len(docs.documents)} drafts generated",
    )

    # -- box 9 ---------------------------------------------------------------
    board = TaskBoard.build(
        compliance=compliance, pay=pay, documents=docs, package=package,
        selection=selection, config=TrackerConfig(),
    )
    result.board = board
    bs = board.summary()
    result.note(
        "Box 9 — Task Tracker", True,
        f"{bs['total_tasks']} tasks, {bs['overdue']} overdue "
        f"({bs['overdue_statutory']} statutory)",
    )

    # -- box 10 --------------------------------------------------------------
    audit = AuditPackage.assemble(
        scenario=plan.plan_name, ingest=ingest, selection=selection,
        impact=impact, compliance=compliance, pay=pay, documents=docs,
        ledger=cfg.ledger, board=board, package=package,
    )
    result.audit = audit
    comp_res = audit.completeness()
    integ = audit.verify()
    result.note(
        "Box 10 — Audit & Reporting", True,
        f"{audit.counts()['entries']} entries, record "
        f"{comp_res.score}% complete, chain "
        f"{'intact' if integ.intact else 'BROKEN'}",
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the full RIF pipeline.")
    ap.add_argument("csv_path")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--separation-date", required=True)
    ap.add_argument("--notice-date", default=None)
    ap.add_argument("--leave-policy", required=True, choices=["separate", "combined"])
    ap.add_argument("--company-headcount", type=int, default=None)
    ap.add_argument("--employer", default="")
    ap.add_argument("--decisional-unit", default="")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    result = run_pipeline(PipelineConfig(
        roster_csv=args.csv_path,
        plan_yaml=args.plan,
        separation_date=args.separation_date,
        notice_date=args.notice_date,
        leave_policy=args.leave_policy,
        total_company_headcount=args.company_headcount,
        employer_name=args.employer,
        decisional_unit=args.decisional_unit,
    ))
    print(result.summary_markdown())

    if args.outdir:
        paths = result.write(args.outdir)
        print(f"\nWrote {len(paths)} file(s) to {args.outdir}")

    return 0 if result.completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
