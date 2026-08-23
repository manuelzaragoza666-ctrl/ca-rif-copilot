"""
audit_reporting.py
==================

Audit & Reporting for the California RIF Copilot (box 10).

Assembles the complete decision record — every finding, every approval, every
document, every task — into a tamper-evident log, checks it for the gaps that
would make it indefensible, and produces the reports a reader needs.

A record that can be filtered is not a record
---------------------------------------------
The single most consequential design choice in this module is that exports
cannot omit adverse findings. Someone will eventually want a clean summary with
the adverse impact result left out, or a compliance report that does not mention
the blocker Legal cleared. This module will not produce one. Summaries are
shorter than the full record, never selectively quieter: every error and warning
appears in every export, and ``verify()`` fails if the log has been altered.

The reasoning is practical rather than moralistic. A sanitized audit package is
the worst possible artifact to have created — it is discoverable, it is
demonstrably incomplete, and the omission is more damaging than whatever was
omitted. An honest record showing a problem that was identified and addressed is
a defense. A curated one showing no problems is an exhibit.

Tamper evidence
---------------
Entries are hash-chained: each carries the hash of the one before it, so any
alteration or deletion breaks the chain from that point forward and ``verify()``
reports exactly where. This does not make the log immutable — anything on a
filesystem can be rewritten — but it makes silent editing detectable, which is
the property that matters.

Privilege
---------
The record mixes material prepared at the direction of counsel (adverse impact
analysis, legal review) with ordinary business records (payroll register, task
list). Bundling them can weaken a privilege claim over the former. The module
classifies each artifact and flags mixed packages rather than deciding the
question, because it is counsel's to decide.

Usage
-----
    from .audit_reporting import AuditPackage, RetentionPolicy

    audit = AuditPackage.assemble(
        scenario="Scenario A", ingest=ingest, selection=selection,
        impact=impact, compliance=compliance, pay=pay,
        documents=docs, ledger=ledger, board=board, package=package,
    )
    audit.completeness()          # what is missing from the record
    audit.verify()                # has the log been altered
    print(audit.executive_summary())
    audit.write("./audit")
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .workforce_data import Severity

__all__ = [
    "AuditEntry",
    "AuditLog",
    "AuditPackage",
    "IntegrityResult",
    "CompletenessResult",
    "RetentionPolicy",
    "PRIVILEGE_CLASSES",
]

__version__ = "1.0.0"

GENESIS = "0" * 64


#: How each artifact should be treated for privilege purposes. The module
#: classifies; counsel decides.
PRIVILEGE_CLASSES: dict[str, str] = {
    "adverse_impact": "privileged",
    "legal_review": "privileged",
    "scenario_comparison": "privileged",
    "compliance_analysis": "privileged",
    "selection_scores": "mixed",
    "approval_ledger": "mixed",
    "cut_list": "mixed",
    "data_validation": "business_record",
    "payroll_register": "business_record",
    "task_board": "business_record",
    "documents": "business_record",
}


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


@dataclass
class RetentionPolicy:
    """Minimum retention periods. Verify against current requirements —
    these are general defaults, and a litigation hold overrides all of them."""

    payroll_records_years: int = 4       # Lab. Code § 1174(d): at least 3; 4 is safer
    selection_records_years: int = 4     # FEHA (Gov. Code § 12946) requires 4
    eeo_records_years: int = 2           # 29 CFR 1602 baseline
    warn_records_years: int = 3
    benefit_records_years: int = 6       # ERISA § 107

    def guidance(self) -> list[tuple[str, int, str]]:
        return [
            ("Selection criteria, scores, and cut list",
             self.selection_records_years,
             "Gov. Code § 12946 — personnel records relating to a termination "
             "must be kept 4 years from the action"),
            ("Adverse impact analysis and supporting data",
             self.selection_records_years,
             "Retain with the selection record; a litigation hold extends this"),
            ("Payroll register, final pay, vacation payout",
             self.payroll_records_years,
             "Lab. Code § 1174(d)"),
            ("WARN notices and delivery proof",
             self.warn_records_years,
             "Proof of delivery is what establishes the 60-day period ran"),
            ("Signed releases, acknowledgments, revocation tracking",
             self.selection_records_years,
             "The enforceability of a release turns on this record"),
            ("Benefit and COBRA records",
             self.benefit_records_years, "ERISA § 107"),
            ("Approval ledger and decision history",
             self.selection_records_years,
             "Establishes who decided what, when, and on which version"),
        ]


# ---------------------------------------------------------------------------
# Hash-chained log
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AuditEntry:
    sequence: int
    timestamp: str
    category: str
    event: str
    detail: str
    severity: str = Severity.INFO
    actor: str = ""
    source_box: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    prev_hash: str = GENESIS
    entry_hash: str = ""

    def compute_hash(self) -> str:
        body = json.dumps(
            {
                "sequence": self.sequence, "timestamp": self.timestamp,
                "category": self.category, "event": self.event,
                "detail": self.detail, "severity": self.severity,
                "actor": self.actor, "source_box": self.source_box,
                "payload": self.payload, "prev_hash": self.prev_hash,
            },
            sort_keys=True, separators=(",", ":"), default=str,
        )
        return hashlib.sha256(body.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence, "timestamp": self.timestamp,
            "category": self.category, "event": self.event, "detail": self.detail,
            "severity": self.severity, "actor": self.actor,
            "source_box": self.source_box, "payload": self.payload,
            "prev_hash": self.prev_hash, "entry_hash": self.entry_hash,
        }


@dataclass
class IntegrityResult:
    intact: bool
    entries_checked: int
    broken_at: int | None = None
    problem: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "intact": self.intact, "entries_checked": self.entries_checked,
            "broken_at": self.broken_at, "problem": self.problem,
        }


class AuditLog:
    """Append-only, hash-chained record of everything the pipeline decided."""

    def __init__(self) -> None:
        self._entries: list[AuditEntry] = []

    def append(
        self, category: str, event: str, detail: str,
        severity: str = Severity.INFO, actor: str = "", source_box: str = "",
        payload: dict[str, Any] | None = None,
        timestamp: str | None = None,
    ) -> AuditEntry:
        prev = self._entries[-1].entry_hash if self._entries else GENESIS
        draft = AuditEntry(
            sequence=len(self._entries),
            timestamp=timestamp or dt.datetime.now().isoformat(timespec="seconds"),
            category=category, event=event, detail=detail, severity=severity,
            actor=actor, source_box=source_box, payload=payload or {},
            prev_hash=prev,
        )
        entry = AuditEntry(**{**draft.__dict__, "entry_hash": draft.compute_hash()})
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[AuditEntry, ...]:
        return tuple(self._entries)

    def verify(self) -> IntegrityResult:
        prev = GENESIS
        for e in self._entries:
            recomputed = AuditEntry(**{**e.__dict__, "entry_hash": ""}).compute_hash()
            if e.prev_hash != prev:
                return IntegrityResult(
                    False, len(self._entries), e.sequence,
                    f"Entry {e.sequence} does not link to the previous entry. An "
                    f"entry was altered, removed, or inserted at or before this "
                    f"point.",
                )
            if recomputed != e.entry_hash:
                return IntegrityResult(
                    False, len(self._entries), e.sequence,
                    f"Entry {e.sequence} ({e.event}) has been modified since it "
                    f"was written.",
                )
            prev = e.entry_hash
        return IntegrityResult(True, len(self._entries))

    def by_severity(self, severity: str) -> list[AuditEntry]:
        return [e for e in self._entries if e.severity == severity]

    def to_dataframe(self) -> pd.DataFrame:
        if not self._entries:
            return pd.DataFrame(
                columns=["sequence", "timestamp", "category", "event", "severity",
                         "source_box", "actor", "detail", "entry_hash"]
            )
        df = pd.DataFrame([e.to_dict() for e in self._entries])
        return df[["sequence", "timestamp", "category", "event", "severity",
                   "source_box", "actor", "detail", "entry_hash"]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "entries": [e.to_dict() for e in self._entries],
            "integrity": self.verify().to_dict(),
        }


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------


@dataclass
class CompletenessResult:
    present: list[str] = field(default_factory=list)
    missing: list[tuple[str, str]] = field(default_factory=list)
    weak: list[tuple[str, str]] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing

    @property
    def score(self) -> float:
        total = len(self.present) + len(self.missing)
        return round(100 * len(self.present) / total, 1) if total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "complete": self.complete, "score": self.score,
            "present": self.present,
            "missing": [{"item": i, "why": w} for i, w in self.missing],
            "weak": [{"item": i, "why": w} for i, w in self.weak],
        }


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------


@dataclass
class AuditPackage:
    scenario: str = ""
    fingerprint: str = ""
    assembled_at: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
    log: AuditLog = field(default_factory=AuditLog)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    retention: RetentionPolicy = field(default_factory=RetentionPolicy)
    _completeness: CompletenessResult | None = None

    # -- assembly ----------------------------------------------------------
    @classmethod
    def assemble(
        cls,
        scenario: str = "",
        ingest: Any = None,
        simulation: Any = None,
        selection: Any = None,
        impact: Any = None,
        compliance: Any = None,
        pay: Any = None,
        documents: Any = None,
        ledger: Any = None,
        board: Any = None,
        package: Any = None,
        retention: RetentionPolicy | None = None,
    ) -> "AuditPackage":
        audit = cls(
            scenario=scenario,
            fingerprint=getattr(package, "fingerprint", ""),
            retention=retention or RetentionPolicy(),
        )
        log = audit.log
        log.append(
            "lifecycle", "audit_assembled",
            f"Audit record assembled for scenario {scenario or '(unnamed)'}.",
            payload={"fingerprint": audit.fingerprint},
        )

        audit._record_ingest(ingest)
        audit._record_simulation(simulation)
        audit._record_selection(selection)
        audit._record_impact(impact)
        audit._record_pay(pay)
        audit._record_compliance(compliance)
        audit._record_approvals(ledger)
        audit._record_documents(documents)
        audit._record_tasks(board)

        audit._completeness = audit._assess_completeness(
            ingest, selection, impact, compliance, pay, documents, ledger, board
        )
        for item, why in audit._completeness.missing:
            log.append(
                "completeness", "record_gap", f"{item}: {why}",
                severity=Severity.ERROR, source_box="box 10",
            )
        for item, why in audit._completeness.weak:
            log.append(
                "completeness", "record_weakness", f"{item}: {why}",
                severity=Severity.WARNING, source_box="box 10",
            )
        return audit

    # -- per-box recorders --------------------------------------------------
    def _artifact(self, name: str, kind: str, summary: str, rows: int = 0) -> None:
        self.artifacts.append({
            "name": name, "kind": kind, "summary": summary, "rows": rows,
            "privilege": PRIVILEGE_CLASSES.get(kind, "unclassified"),
        })

    def _record_findings(self, findings: Iterable[Any], box: str) -> None:
        """Record every finding, at every severity. There is no filter here by
        design: an audit log that drops the inconvenient entries is the artifact
        that does the damage."""
        for f in findings or []:
            sev = getattr(f, "severity", None) or (
                f.get("severity") if isinstance(f, dict) else Severity.INFO
            )
            code = getattr(f, "code", None) or (
                f.get("code") if isinstance(f, dict) else "FINDING"
            )
            msg = getattr(f, "message", None) or (
                f.get("message") if isinstance(f, dict) else str(f)
            )
            self.log.append(
                "finding", code, msg, severity=sev, source_box=box,
            )

    def _record_ingest(self, ingest: Any) -> None:
        if ingest is None:
            return
        rep = ingest.report
        s = rep.summary()
        self.log.append(
            "data", "roster_ingested",
            f"{s['rows']} row(s) ingested; {s['errors']} error(s), "
            f"{s['warnings']} warning(s); {s['rows_with_errors']} row(s) "
            f"quarantined.",
            severity=Severity.ERROR if s["blocking"] else Severity.INFO,
            source_box="box 1", payload=s,
        )
        self._record_findings(rep.issues, "box 1")
        self._artifact("Data validation report", "data_validation",
                       f"{s['rows']} rows, {s['errors']} errors", s["rows"])

    def _record_simulation(self, simulation: Any) -> None:
        if simulation is None:
            return
        for o in getattr(simulation, "outcomes", []):
            self.log.append(
                "scenario", "scenario_modeled",
                f"Scenario '{o.scenario.name}' modeled. Rationale: "
                f"{o.scenario.rationale.strip()[:300]}",
                source_box="box 2",
                payload=o.summary_row(),
            )
        self._record_findings(getattr(simulation.report, "findings", []), "box 2")
        self.log.append(
            "scenario", "scenarios_retained",
            "All scenarios modeled are recorded, including any not pursued. "
            "The sequence is discoverable; an unexplained gap is harder to "
            "defend than a documented decision to move on.",
            source_box="box 2",
        )
        self._artifact("Scenario comparison", "scenario_comparison",
                       f"{len(getattr(simulation, 'outcomes', []))} scenarios")

    def _record_selection(self, selection: Any) -> None:
        if selection is None:
            return
        rep = selection.report
        s = rep.summary()
        self.log.append(
            "selection", "selection_run",
            f"{s['selected_employees']} of {s['eligible_employees']} employees "
            f"selected; ${s['achieved_savings']:,.0f} of "
            f"${s['cost_savings_target']:,.0f} target.",
            source_box="box 3", payload=s,
        )
        for d in rep.department_summary:
            self.log.append(
                "selection", "department_result",
                f"{d['department']}: {d['selected']} of {d['evaluated']} selected "
                f"({d['mode']} mode), ${d['savings']:,.0f}.",
                source_box="box 3", payload=d,
            )
        self.log.append(
            "selection", "protected_fields_excluded",
            "Scoring read no protected characteristic. Age, sex, race, "
            "disability, and veteran status were withheld from the engine by "
            "design.",
            source_box="box 3",
        )
        self._record_findings(rep.findings, "box 3")
        self._artifact("Selection scores", "selection_scores",
                       f"{s['eligible_employees']} scored",
                       s["eligible_employees"])
        self._artifact("Recommended cut list", "cut_list",
                       f"{s['selected_employees']} selected",
                       s["selected_employees"])

    def _record_impact(self, impact: Any) -> None:
        if impact is None:
            return
        rep = impact.report
        for cls, verdict in rep.class_verdicts().items():
            sev = {
                "Impact indicated": Severity.ERROR,
                "Review": Severity.WARNING,
            }.get(verdict, Severity.INFO)
            self.log.append(
                "impact", "class_verdict", f"{cls}: {verdict}",
                severity=sev, source_box="box 4",
            )
        for c in rep.indicated:
            self.log.append(
                "impact", "adverse_impact_indicated",
                f"{c.protected_class} group '{c.group}' in "
                f"{c.unit if c.unit_type != 'overall' else 'the company overall'}: "
                f"selected at {c.group_selection_rate:.1%} versus "
                f"{c.reference_selection_rate:.1%}; four-fifths ratio "
                f"{c.impact_ratio:.2f}, {c.std_deviations:+.2f} SD, "
                f"p={c.fisher_p:.4f}.",
                severity=Severity.ERROR, source_box="box 4", payload=c.to_dict(),
            )
        self._record_findings(rep.findings, "box 4")
        self._artifact("Adverse impact analysis", "adverse_impact",
                       f"{len(rep.comparisons)} comparisons, "
                       f"{len(rep.indicated)} indicated", len(rep.comparisons))

    def _record_pay(self, pay: Any) -> None:
        if pay is None:
            return
        rep = pay.report
        t = rep.totals
        self.log.append(
            "pay", "severance_computed",
            f"Severance ${t.get('severance_gross', 0):,.2f}, vacation payout "
            f"${t.get('vacation_payout', 0):,.2f}, total employer cost "
            f"${t.get('total_employer_cost', 0):,.2f}.",
            source_box="box 6", payload=t,
        )
        self.log.append(
            "pay", "formula_recorded",
            f"Severance formula applied: {json.dumps(rep.assumptions.get('formula', {}), default=str)}",
            source_box="box 6",
        )
        self._record_findings(rep.findings, "box 6")
        self._artifact("Payroll register", "payroll_register",
                       f"{rep.employee_count} employees", rep.employee_count)

    def _record_compliance(self, compliance: Any) -> None:
        if compliance is None:
            return
        rep = compliance.report
        for w in rep.warn:
            self.log.append(
                "compliance", "warn_determination",
                f"{w.jurisdiction}"
                + (f" / {w.establishment}" if w.establishment else "")
                + f": {'TRIGGERED' if w.triggered else 'not triggered'}. {w.reason}",
                severity=Severity.WARNING if w.triggered else Severity.INFO,
                source_box="box 5", payload=w.to_dict(),
            )
        for o in rep.obligations:
            self.log.append(
                "compliance", "obligation",
                f"{o.title} due {o.due_date} ({o.authority})"
                + (" — MISSED" if o.missed else ""),
                severity=Severity.ERROR if o.missed else Severity.INFO,
                source_box="box 5",
            )
        self.log.append(
            "compliance", "document_gate",
            f"Document generation gate: "
            f"{'CLEAR' if compliance.gate.may_generate_documents else 'BLOCKED'}"
            + ("" if compliance.gate.may_generate_documents
               else " — " + "; ".join(compliance.gate.blockers)),
            severity=Severity.INFO if compliance.gate.may_generate_documents
            else Severity.WARNING,
            source_box="box 5",
        )
        self._record_findings(rep.findings, "box 5")
        self._artifact("Compliance analysis", "compliance_analysis",
                       f"{len(rep.obligations)} obligations",
                       len(rep.obligations))

    def _record_approvals(self, ledger: Any) -> None:
        if ledger is None:
            return
        for r in getattr(ledger, "records", []):
            sev = Severity.WARNING if r.action in ("rejected", "revoked", "superseded") \
                else Severity.INFO
            detail = f"{r.action}"
            if r.stage:
                detail += f" at {r.stage}"
            detail += f" by {r.actor} ({r.role or 'role not stated'})"
            if r.clears:
                detail += f"; cleared {', '.join(r.clears)}"
            if r.comment:
                detail += f". {r.comment}"
            self.log.append(
                "approval", f"approval_{r.action}", detail,
                severity=sev, actor=r.actor, source_box="box 8",
                payload={"fingerprint": r.fingerprint, "clears": list(r.clears)},
                timestamp=r.timestamp,
            )
        status = ledger.status()
        self.log.append(
            "approval", "approval_status",
            f"Chain {'complete' if status.complete else 'incomplete'}"
            + (f": {status.blocked_reason}" if status.blocked_reason else "")
            + f" (version {status.fingerprint}).",
            severity=Severity.INFO if status.complete else Severity.WARNING,
            source_box="box 8",
        )
        self._artifact("Approval ledger", "approval_ledger",
                       f"{len(getattr(ledger, 'records', []))} records")

    def _record_documents(self, documents: Any) -> None:
        if documents is None:
            return
        if getattr(documents, "blocked", False):
            self.log.append(
                "documents", "generation_blocked",
                "Document generation was blocked: "
                + "; ".join(getattr(documents, "blockers", [])),
                severity=Severity.WARNING, source_box="box 7",
            )
            return
        if getattr(documents, "override_record", None):
            o = documents.override_record
            self.log.append(
                "documents", "gate_cleared_by_counsel",
                f"Compliance gate cleared by {o['by']} on {o['date']}: "
                f"{o['reason']}",
                severity=Severity.WARNING, actor=o["by"], source_box="box 7",
            )
        counts: dict[str, int] = {}
        for d in getattr(documents, "documents", []):
            counts[d.doc_type] = counts.get(d.doc_type, 0) + 1
        self.log.append(
            "documents", "documents_generated",
            f"{len(getattr(documents, 'documents', []))} draft document(s): "
            + ", ".join(f"{k} ({v})" for k, v in sorted(counts.items())),
            source_box="box 7", payload=counts,
        )
        self._record_findings(getattr(documents, "findings", []), "box 7")
        self._artifact("Document set", "documents",
                       f"{len(getattr(documents, 'documents', []))} drafts")

    def _record_tasks(self, board: Any) -> None:
        if board is None:
            return
        s = board.summary()
        self.log.append(
            "execution", "task_board_built",
            f"{s['total_tasks']} task(s); {s['overdue']} overdue "
            f"({s['overdue_statutory']} statutory); "
            f"{s['percent_complete']}% complete.",
            severity=Severity.ERROR if s["overdue_statutory"] else Severity.INFO,
            source_box="box 9", payload=s,
        )
        for e in getattr(board, "log", []):
            self.log.append(
                "execution", f"task_{e['action']}",
                f"{e['target']} by {e['actor']}"
                + (f": {e['detail']}" if e.get("detail") else ""),
                actor=e["actor"], source_box="box 9", timestamp=e["timestamp"],
            )
        self._record_findings(getattr(board, "findings", []), "box 9")
        self._artifact("Task board", "task_board", f"{s['total_tasks']} tasks",
                       s["total_tasks"])

    # -- completeness --------------------------------------------------------
    def _assess_completeness(
        self, ingest, selection, impact, compliance, pay, documents, ledger, board
    ) -> CompletenessResult:
        r = CompletenessResult()

        checks = [
            (ingest, "Data validation record",
             "Without it there is no evidence the roster the decision rested on "
             "was sound."),
            (selection, "Selection criteria and scores",
             "The criteria and their application are the first thing a "
             "challenge examines."),
            (impact, "Adverse impact analysis",
             "Its absence is itself a fact: it means the disparity was never "
             "measured."),
            (compliance, "Compliance determination",
             "WARN, final pay, and OWBPA obligations are undetermined without it."),
            (pay, "Payroll register",
             "No record of what each employee was actually owed and paid."),
            (ledger, "Approval record",
             "No evidence of who authorized the action or on what basis."),
        ]
        for obj, name, why in checks:
            if obj is None:
                r.missing.append((name, why))
            else:
                r.present.append(name)

        for obj, name in ((documents, "Document set"), (board, "Task board")):
            if obj is None:
                r.weak.append((
                    name,
                    "Not present. Acceptable if the action has not reached that "
                    "stage; a gap if it has.",
                ))
            else:
                r.present.append(name)

        # Substantive weaknesses within what is present.
        if impact is not None and getattr(impact.report, "indicated", None):
            cleared = set()
            if ledger is not None:
                cleared = set(ledger.status().cleared_codes)
            if "ADVERSE_IMPACT_INDICATED" not in cleared:
                r.weak.append((
                    "Adverse impact finding",
                    "Impact is indicated and no approval record clears it. A "
                    "finding identified and not addressed is the most damaging "
                    "document in the record.",
                ))
        if ledger is not None and not ledger.status().complete:
            r.weak.append((
                "Approval chain",
                f"Incomplete: {ledger.status().blocked_reason}",
            ))
        if selection is not None and len(getattr(selection, "review_queue", [])):
            r.weak.append((
                "Selection review queue",
                f"{len(selection.review_queue)} employee(s) still require a human "
                f"decision; the cut list is provisional.",
            ))
        if compliance is not None and compliance.report.missed_deadlines:
            r.weak.append((
                "Statutory deadlines",
                f"{len(compliance.report.missed_deadlines)} deadline(s) recorded "
                f"as missed.",
            ))
        return r

    # -- public --------------------------------------------------------------
    def completeness(self) -> CompletenessResult:
        return self._completeness or CompletenessResult()

    def verify(self) -> IntegrityResult:
        return self.log.verify()

    def privilege_summary(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for a in self.artifacts:
            out.setdefault(a["privilege"], []).append(a["name"])
        return out

    def counts(self) -> dict[str, int]:
        return {
            "entries": len(self.log.entries),
            "errors": len(self.log.by_severity(Severity.ERROR)),
            "warnings": len(self.log.by_severity(Severity.WARNING)),
            "artifacts": len(self.artifacts),
        }

    # -- reports -------------------------------------------------------------
    def executive_summary(self) -> str:
        c = self.counts()
        comp = self.completeness()
        integ = self.verify()
        L: list[str] = []
        L.append("# RIF Decision Record — Executive Summary")
        L.append("")
        L.append(
            "> **Privileged and confidential — prepared at the direction of "
            "counsel.** Confirm labeling before circulating."
        )
        L.append("")
        L.append(f"**Scenario:** {self.scenario or '(unnamed)'}  ")
        L.append(f"**Plan version:** `{self.fingerprint or '(unbound)'}`  ")
        L.append(f"**Assembled:** {self.assembled_at}  ")
        L.append("")
        L.append("| | |")
        L.append("|---|---|")
        L.append(f"| Record integrity | {'intact' if integ.intact else '**BROKEN**'} |")
        L.append(f"| Record completeness | {comp.score}% "
                 f"({'complete' if comp.complete else f'{len(comp.missing)} gap(s)'}) |")
        L.append(f"| Log entries | {c['entries']} |")
        L.append(f"| Errors recorded | {c['errors']} |")
        L.append(f"| Warnings recorded | {c['warnings']} |")
        L.append("")

        errors = self.log.by_severity(Severity.ERROR)
        L.append(f"## Every error in the record ({len(errors)})")
        L.append("")
        L.append(
            "This section is not filtered and cannot be. A summary that omits an "
            "adverse finding is worse than no summary: it is discoverable, "
            "demonstrably incomplete, and the omission does more damage than the "
            "finding it hides."
        )
        L.append("")
        if errors:
            for e in errors:
                L.append(f"- **{e.event}** ({e.source_box or 'pipeline'}) — "
                         f"{e.detail[:280]}")
        else:
            L.append("No errors were recorded.")
        L.append("")

        if comp.missing:
            L.append("## Gaps in the record")
            L.append("")
            for item, why in comp.missing:
                L.append(f"- **{item}** — {why}")
            L.append("")
        if comp.weak:
            L.append("## Weaknesses")
            L.append("")
            for item, why in comp.weak:
                L.append(f"- **{item}** — {why}")
            L.append("")

        L.append("---")
        L.append(
            "_See the full decision history for the complete chronology. This "
            "summary is shorter than the record, not quieter._"
        )
        return "\n".join(L)

    def decision_history(self) -> str:
        L: list[str] = []
        L.append("# Decision History")
        L.append("")
        L.append(
            "> **Privileged and confidential — prepared at the direction of "
            "counsel.**"
        )
        L.append("")
        L.append(f"**Scenario:** {self.scenario or '(unnamed)'}  ")
        L.append(f"**Plan version:** `{self.fingerprint or '(unbound)'}`  ")
        integ = self.verify()
        L.append(f"**Integrity:** {'chain intact' if integ.intact else integ.problem}  ")
        L.append(f"**Entries:** {len(self.log.entries)}  ")
        L.append("")

        order = ["lifecycle", "data", "scenario", "selection", "impact", "pay",
                 "compliance", "approval", "documents", "execution",
                 "completeness", "finding"]
        by_cat: dict[str, list[AuditEntry]] = {}
        for e in self.log.entries:
            by_cat.setdefault(e.category, []).append(e)

        for cat in order + [c for c in by_cat if c not in order]:
            entries = by_cat.get(cat)
            if not entries:
                continue
            L.append(f"## {cat.replace('_', ' ').title()} ({len(entries)})")
            L.append("")
            L.append("| Seq | When | Severity | Event | Detail |")
            L.append("|---|---|---|---|---|")
            for e in entries:
                detail = e.detail.replace("|", "\\|")[:220]
                L.append(
                    f"| {e.sequence} | {e.timestamp} | {e.severity} | {e.event} | "
                    f"{detail} |"
                )
            L.append("")

        L.append("---")
        L.append(
            "_Entries are hash-chained: each carries the hash of the one before "
            "it, so altering or removing any entry breaks the chain from that "
            "point and is reported by verification. Every finding at every "
            "severity is included; the log has no filter._"
        )
        return "\n".join(L)

    def compliance_report(self) -> str:
        L: list[str] = []
        L.append("# Compliance Report")
        L.append("")
        L.append(
            "> **Privileged and confidential — prepared at the direction of "
            "counsel.**"
        )
        L.append("")
        for cat, title in (
            ("compliance", "Statutory determinations and obligations"),
            ("impact", "Adverse impact"),
            ("approval", "Authorization"),
        ):
            entries = [e for e in self.log.entries if e.category == cat]
            if not entries:
                continue
            L.append(f"## {title}")
            L.append("")
            for e in entries:
                mark = "**" if e.severity == Severity.ERROR else ""
                L.append(f"- {mark}{e.event}{mark}: {e.detail[:300]}")
            L.append("")

        L.append("## Privilege classification")
        L.append("")
        L.append(
            "This package mixes material prepared at the direction of counsel "
            "with ordinary business records. Bundling them can weaken a "
            "privilege claim over the former. The classification below is a "
            "starting point, not a determination — counsel decides."
        )
        L.append("")
        for cls, names in sorted(self.privilege_summary().items()):
            L.append(f"**{cls.replace('_', ' ').title()}**")
            for n in names:
                L.append(f"- {n}")
            L.append("")

        L.append("## Retention")
        L.append("")
        L.append("| Record | Minimum years | Basis |")
        L.append("|---|---|---|")
        for item, years, basis in self.retention.guidance():
            L.append(f"| {item} | {years} | {basis} |")
        L.append("")
        L.append(
            "**A litigation hold overrides every period above.** Once a claim is "
            "reasonably anticipated, nothing in this record may be deleted on a "
            "routine schedule — including drafts, discarded scenarios, and "
            "superseded approvals."
        )
        L.append("")
        L.append("---")
        L.append(
            "_Retention periods are general defaults; verify against current "
            "requirements. Not legal advice._"
        )
        return "\n".join(L)

    # -- output --------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "fingerprint": self.fingerprint,
            "assembled_at": self.assembled_at,
            "counts": self.counts(),
            "integrity": self.verify().to_dict(),
            "completeness": self.completeness().to_dict(),
            "artifacts": self.artifacts,
            "privilege": self.privilege_summary(),
            "log": self.log.to_dict(),
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def write(self, outdir: str | Path, stem: str = "audit") -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "executive_summary": outdir / f"{stem}_executive_summary.md",
            "decision_history": outdir / f"{stem}_decision_history.md",
            "compliance_report": outdir / f"{stem}_compliance_report.md",
            "log": outdir / f"{stem}_log.csv",
            "artifacts": outdir / f"{stem}_artifacts.csv",
            "json": outdir / f"{stem}_record.json",
        }
        paths["executive_summary"].write_text(self.executive_summary(), encoding="utf-8")
        paths["decision_history"].write_text(self.decision_history(), encoding="utf-8")
        paths["compliance_report"].write_text(self.compliance_report(), encoding="utf-8")
        self.log.to_dataframe().to_csv(paths["log"], index=False)
        pd.DataFrame(self.artifacts or [], columns=["name", "kind", "summary",
                                                    "rows", "privilege"]).to_csv(
            paths["artifacts"], index=False
        )
        self.to_json(paths["json"])
        return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from .pipeline import PipelineConfig, run_pipeline

    ap = argparse.ArgumentParser(
        description="Assemble the audit record for a RIF scenario."
    )
    ap.add_argument("csv_path")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--separation-date", required=True)
    ap.add_argument("--notice-date", default=None)
    ap.add_argument("--leave-policy", required=True, choices=["separate", "combined"])
    ap.add_argument("--company-headcount", type=int, default=None)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    result = run_pipeline(PipelineConfig(
        roster_csv=args.csv_path, plan_yaml=args.plan,
        separation_date=args.separation_date, notice_date=args.notice_date,
        leave_policy=args.leave_policy,
        total_company_headcount=args.company_headcount,
    ))
    audit = AuditPackage.assemble(
        scenario=result.package.scenario if result.package else "",
        ingest=result.ingest, selection=result.selection, impact=result.impact,
        compliance=result.compliance, pay=result.pay, documents=result.documents,
        ledger=result.config.ledger, board=result.board, package=result.package,
    )
    print(audit.executive_summary())

    if args.outdir:
        paths = audit.write(args.outdir)
        print("\nWrote:")
        for k, p in paths.items():
            print(f"  {k}: {p}")

    return 0 if audit.completeness().complete and audit.verify().intact else 1


if __name__ == "__main__":
    raise SystemExit(main())
