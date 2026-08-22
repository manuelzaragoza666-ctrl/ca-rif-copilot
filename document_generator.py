"""
document_generator.py
=====================

Document & Communication Generator for the California RIF Copilot (box 7).

Produces the paperwork a reduction actually requires — Cal-WARN notices,
separation letters, the OWBPA decisional-unit disclosure, manager talking
points, an employee FAQ, and a notice-day checklist — from the outputs of
boxes 3 through 6.

The gate
--------
This module will not generate anything while the CA Compliance Engine reports
unresolved blockers. That is the point of building box 5 first: a notice
generated from a non-compliant scenario is worse than no notice, because it
creates a dated artifact memorializing the defect.

Two kinds of blocker are distinguished, because they are not the same problem:

*Legal-judgment blockers* — an adverse impact finding, employees on protected
leave, union members whose CBA has not been checked. These turn on an
assessment a lawyer makes. They can be cleared by recorded counsel sign-off:
a named person, a written reason, and a date, stamped into every document
produced and into the manifest.

*Data-completeness blockers* — a missing pay rate, an uncomputable severance
figure, an undetermined WARN establishment. No amount of legal judgment fills
in a blank field. These cannot be overridden at all, because the document
would simply be wrong.

What this module does not produce
---------------------------------
It does not produce a signable release of claims. A release is a contract, and
an ADEA release carries strict OWBPA requirements whose failure is invisible
until it is litigated. What it produces is a **draft skeleton for counsel**:
the computed figures, the required OWBPA elements enumerated, and every
judgment call left as a visible placeholder. It is marked DRAFT on every page
and the generator refuses to remove that marking.

Every document is a draft. Nothing here has been reviewed by a lawyer, and the
statutes change — Cal-WARN's required notice content changed in January 2026.

Usage
-----
    from .document_generator import DocumentConfig, DocumentGenerator

    cfg = DocumentConfig(
        employer_name="Acme Inc.",
        employer_address="1 Market St, San Francisco, CA 94105",
        signatory_name="Dana Reyes",
        signatory_title="VP, People",
        decisional_unit="All Engineering employees at the SF HQ establishment",
    )
    docs = DocumentGenerator(cfg).generate(
        compliance=compliance_result,
        selection=selection_result,
        pay=pay_result,
        scores=selection_result.scores,
    )

    docs.blocked          # True if the gate refused
    docs.documents        # list of GeneratedDocument
    docs.write("./out")
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .workforce_data import Severity

__all__ = [
    "DocumentConfig",
    "DocumentGenerator",
    "DocumentSet",
    "GeneratedDocument",
    "DOCUMENT_TYPES",
    "NON_OVERRIDABLE_CODES",
]

__version__ = "1.0.0"


# ---------------------------------------------------------------------------
# Gate policy
# ---------------------------------------------------------------------------

#: Blockers that no sign-off can clear, because they mean a field would be
#: blank or wrong rather than that a judgment is outstanding.
NON_OVERRIDABLE_CODES: frozenset[str] = frozenset({
    "NO_ESTABLISHMENT_COLUMN",
    "NO_ESTABLISHMENTS",
    "FINAL_PAY_UNCOMPUTABLE",
    "NO_PAY_DATA",
    "NO_TENURE_DATA",
    "INCOMPLETE_REGISTER",
    "LEAVE_POLICY_UNDECLARED",
    "SB617_COORDINATION_UNDECLARED",
    "SB617_LWDB_CONTACT_MISSING",
    "SB617_EMPLOYER_CONTACT_MISSING",
    "NO_SELECTION",
    "SEPARATION_BEFORE_NOTICE",
})

DOCUMENT_TYPES: tuple[str, ...] = (
    "warn_notice_employee",
    "warn_notice_agency",
    "separation_letter",
    "owbpa_disclosure",
    "severance_agreement_draft",
    "manager_script",
    "employee_faq",
    "notice_day_checklist",
    "hr_summary",
)

PLACEHOLDER_RE = re.compile(r"\[\[([A-Z0-9_ ]+)\]\]")

DRAFT_BANNER = (
    "> **DRAFT — NOT FOR DISTRIBUTION.** Generated automatically from scenario "
    "data. Every document in this set requires review by employment counsel "
    "before it is sent to anyone. Statutory requirements change; Cal-WARN's "
    "required notice content changed on 2026-01-01."
)

PRIVILEGE_BANNER = (
    "> **Privileged and confidential — prepared at the direction of counsel.** "
    "Confirm the correct label with your employment counsel before circulating."
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DocumentConfig:
    employer_name: str = ""
    employer_address: str = ""
    employer_contact_name: str = ""
    employer_contact_email: str = ""
    employer_contact_phone: str = ""

    signatory_name: str = ""
    signatory_title: str = ""

    #: Cal-WARN / SB 617 details.
    lwdb_name: str = ""
    lwdb_email: str = ""
    lwdb_phone: str = ""
    service_coordination: str = ""      # lwdb | other | none
    bumping_rights: bool = False        # Cal-WARN notice must state this

    #: OWBPA decisional unit. Defining it is a legal judgment; the generator
    #: will not infer it from the scoring comparison groups.
    decisional_unit: str = ""
    decisional_unit_column: str = ""    # column identifying membership
    decisional_unit_value: str = ""

    #: Severance agreement parameters left to counsel are placeholders; these
    #: are the few facts the generator can legitimately fill.
    consideration_period_days: int = 45
    revocation_period_days: int = 7

    #: Recorded counsel sign-off clearing legal-judgment blockers.
    #:
    #: Prefer passing an ApprovalLedger from box 8 to ``generate()``: it binds
    #: the sign-off to a fingerprint of exactly what was approved, so the
    #: clearance cannot silently survive a change to the plan. These fields
    #: remain for use without box 8 and are populated automatically from the
    #: ledger when one is supplied.
    counsel_override_by: str = ""
    counsel_override_reason: str = ""
    counsel_override_date: dt.date | None = None

    #: Which document types to produce. Empty means all.
    include: tuple[str, ...] = ()

    def has_override(self) -> bool:
        return bool(
            self.counsel_override_by.strip() and self.counsel_override_reason.strip()
        )


# ---------------------------------------------------------------------------
# Output structures
# ---------------------------------------------------------------------------


@dataclass
class GeneratedDocument:
    doc_type: str
    title: str
    body: str
    filename: str
    audience: str = ""
    employee_id: str | None = None
    placeholders: tuple[str, ...] = ()
    review_required: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.placeholders

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type, "title": self.title,
            "filename": self.filename, "audience": self.audience,
            "employee_id": self.employee_id,
            "placeholders": list(self.placeholders),
            "review_required": list(self.review_required),
            "complete": self.is_complete,
        }


@dataclass
class DocumentSet:
    documents: list[GeneratedDocument] = field(default_factory=list)
    blocked: bool = False
    blockers: list[str] = field(default_factory=list)
    overridden_blockers: list[str] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    generated_at: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds")
    )
    override_record: dict[str, Any] | None = None

    def add(self, severity: str, code: str, message: str) -> None:
        self.findings.append({"severity": severity, "code": code, "message": message})

    def by_type(self, doc_type: str) -> list[GeneratedDocument]:
        return [d for d in self.documents if d.doc_type == doc_type]

    @property
    def incomplete(self) -> list[GeneratedDocument]:
        return [d for d in self.documents if not d.is_complete]

    def manifest(self) -> pd.DataFrame:
        if not self.documents:
            return pd.DataFrame(
                columns=["doc_type", "title", "filename", "audience",
                         "employee_id", "placeholders", "complete"]
            )
        rows = []
        for d in self.documents:
            r = d.to_dict()
            r["placeholders"] = "|".join(r["placeholders"])
            r["review_required"] = "|".join(r["review_required"])
            rows.append(r)
        return pd.DataFrame(rows)

    def summary_markdown(self) -> str:
        L: list[str] = []
        L.append("# Document Generation Summary")
        L.append("")
        L.append(PRIVILEGE_BANNER)
        L.append("")
        L.append(f"**Generated:** {self.generated_at}  ")
        L.append("")

        if self.blocked:
            L.append("## Status: BLOCKED — nothing was generated")
            L.append("")
            L.append(
                "The CA Compliance Engine reports unresolved blockers. Notices "
                "were not produced, because a notice generated from a "
                "non-compliant scenario creates a dated record of the defect."
            )
            L.append("")
            for b in self.blockers:
                L.append(f"- {b}")
            L.append("")
            L.append("### Resolving this")
            L.append("")
            L.append(
                "Data-completeness blockers must be fixed at the source — no "
                "sign-off fills in a blank pay rate. Legal-judgment blockers can "
                "be cleared by recorded counsel sign-off "
                "(`counsel_override_by`, `counsel_override_reason`), which is "
                "stamped into every document produced and into this manifest."
            )
            L.append("")
            return "\n".join(L)

        L.append(f"## Status: {len(self.documents)} document(s) generated as DRAFT")
        L.append("")

        if self.override_record:
            o = self.override_record
            L.append("### Counsel sign-off recorded")
            L.append("")
            L.append(f"- **Cleared by:** {o['by']}")
            L.append(f"- **Date:** {o['date']}")
            L.append(f"- **Reason:** {o['reason']}")
            L.append(f"- **Blockers cleared:** {len(self.overridden_blockers)}")
            for b in self.overridden_blockers:
                L.append(f"  - {b}")
            L.append("")

        L.append("| Document | Audience | Placeholders | Status |")
        L.append("|---|---|---|---|")
        counts: dict[str, list[GeneratedDocument]] = {}
        for d in self.documents:
            counts.setdefault(d.doc_type, []).append(d)
        for doc_type, docs in counts.items():
            n_ph = sum(len(d.placeholders) for d in docs)
            status = "needs completion" if n_ph else "ready for counsel review"
            label = f"{doc_type} ({len(docs)})" if len(docs) > 1 else doc_type
            L.append(
                f"| {label} | {docs[0].audience} | {n_ph} | {status} |"
            )
        L.append("")

        if self.incomplete:
            L.append("### Unfilled placeholders")
            L.append("")
            L.append(
                "Each of these is a fact the generator could not supply. They are "
                "left visible rather than guessed at."
            )
            L.append("")
            seen: dict[str, int] = {}
            for d in self.incomplete:
                for p in d.placeholders:
                    seen[p] = seen.get(p, 0) + 1
            for p, n in sorted(seen.items(), key=lambda kv: -kv[1]):
                L.append(f"- `[[{p}]]` — {n} occurrence(s)")
            L.append("")

        if self.findings:
            L.append("### Findings")
            L.append("")
            for f in self.findings:
                L.append(f"- **[{f['severity']}] {f['code']}** — {f['message']}")
            L.append("")

        L.append("---")
        L.append(
            "_Every document is a draft. Nothing here has been reviewed by a "
            "lawyer. The severance agreement is a skeleton for counsel, not a "
            "signable instrument._"
        )
        return "\n".join(L)

    def write(self, outdir: str | Path) -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {
            "summary": outdir / "00_SUMMARY.md",
            "manifest": outdir / "00_manifest.csv",
        }
        paths["summary"].write_text(self.summary_markdown(), encoding="utf-8")
        self.manifest().to_csv(paths["manifest"], index=False)

        if self.documents:
            for d in self.documents:
                sub = outdir / d.doc_type
                sub.mkdir(parents=True, exist_ok=True)
                p = sub / d.filename
                p.write_text(d.body, encoding="utf-8")
                paths[d.filename] = p
        return paths


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class DocumentGenerator:
    """Produces draft RIF documents, subject to the compliance gate."""

    def __init__(self, config: DocumentConfig) -> None:
        self.cfg = config

    # -- public ------------------------------------------------------------
    def generate(
        self,
        compliance: Any,
        selection: Any = None,
        pay: Any = None,
        scores: pd.DataFrame | None = None,
        approvals: Any = None,
        package: Any = None,
    ) -> DocumentSet:
        out = DocumentSet()
        cfg = self.cfg

        if approvals is not None:
            if not self._apply_approvals(approvals, package, out):
                return out

        gate = getattr(compliance, "gate", None)
        report = getattr(compliance, "report", None)
        if gate is None or report is None:
            out.blocked = True
            out.blockers = ["No compliance analysis supplied; box 5 must run first."]
            out.add(
                Severity.ERROR, "NO_COMPLIANCE_INPUT",
                "Document generation requires a ComplianceResult. Running box 7 "
                "without box 5 would defeat the gate entirely.",
            )
            return out

        if not self._check_gate(gate, report, out):
            return out

        cut = self._cut_list(selection, scores, out)
        if cut is None:
            return out

        register = getattr(pay, "register", None)
        wanted = set(cfg.include) if cfg.include else set(DOCUMENT_TYPES)

        if report.warn_triggered and "warn_notice_employee" in wanted:
            out.documents.append(self._warn_notice_employee(report, cut))
        if report.warn_triggered and "warn_notice_agency" in wanted:
            out.documents.extend(self._warn_notice_agency(report, cut))
        elif not report.warn_triggered:
            out.add(
                Severity.INFO, "NO_WARN_NOTICE",
                "Cal-WARN was not triggered on this scenario, so no WARN notice "
                "was generated. If the reduction grows or a second round follows "
                "within 30 days, re-run box 5 — the determination can change.",
            )

        if "separation_letter" in wanted:
            out.documents.extend(self._separation_letters(report, cut, register))
        if "owbpa_disclosure" in wanted:
            doc = self._owbpa_disclosure(cut, scores, out)
            if doc:
                out.documents.append(doc)
        if "severance_agreement_draft" in wanted:
            out.documents.append(self._severance_skeleton(report, register))
        if "manager_script" in wanted:
            out.documents.append(self._manager_script(report))
        if "employee_faq" in wanted:
            out.documents.append(self._employee_faq(report, pay))
        if "notice_day_checklist" in wanted:
            out.documents.append(self._checklist(report))
        if "hr_summary" in wanted:
            out.documents.append(self._hr_summary(report, cut, pay))

        for d in out.documents:
            d.placeholders = tuple(sorted(set(PLACEHOLDER_RE.findall(d.body))))

        if out.incomplete:
            out.add(
                Severity.WARNING, "PLACEHOLDERS_REMAIN",
                f"{len(out.incomplete)} document(s) contain unfilled placeholders. "
                f"These are facts the generator could not supply and would not "
                f"guess. Fill them before counsel review, not after.",
            )
        out.add(
            Severity.WARNING, "ALL_DOCUMENTS_ARE_DRAFTS",
            "Every document is marked DRAFT and must be reviewed by employment "
            "counsel before distribution. The severance agreement in particular "
            "is a skeleton enumerating required elements, not a signable release.",
        )
        return out

    # -- approvals (box 8) ---------------------------------------------------
    def _apply_approvals(self, ledger: Any, package: Any, out: DocumentSet) -> bool:
        """Derive clearance from the approval ledger rather than a config string.

        The ledger binds sign-off to a fingerprint of what was approved, so a
        clearance cannot survive a change to the cut list, the plan, or the
        dates. A config string cannot do that, which is why it is the fallback
        rather than the primary path.
        """
        if package is not None:
            ok, problems = ledger.verify(package)
            if not ok:
                out.blocked = True
                out.blockers = problems
                out.add(
                    Severity.ERROR, "APPROVAL_INVALID",
                    "The approval chain does not cover the current plan. "
                    + " ".join(problems),
                )
                return False

        clearance = ledger.clearance(package)
        if not clearance.get("approved"):
            out.blocked = True
            out.blockers = [clearance.get("blocked_reason") or "Not fully approved."]
            out.add(
                Severity.ERROR, "NOT_APPROVED",
                f"Documents cannot be generated before the approval chain is "
                f"complete. {clearance.get('blocked_reason', '')}",
            )
            return False

        if clearance.get("legal_approver"):
            self.cfg.counsel_override_by = clearance["legal_approver"]
            self.cfg.counsel_override_reason = (
                clearance.get("legal_basis")
                or f"Legal approval recorded against version "
                   f"{clearance.get('fingerprint')}"
            )
            if clearance.get("legal_date"):
                self.cfg.counsel_override_date = dt.date.fromisoformat(
                    clearance["legal_date"]
                )
        out.add(
            Severity.INFO, "APPROVAL_VERIFIED",
            f"Approval chain complete against version "
            f"{clearance.get('fingerprint')}; legal sign-off by "
            f"{clearance.get('legal_approver') or '(none recorded)'}.",
        )
        return True

    # -- gate ---------------------------------------------------------------
    def _check_gate(self, gate: Any, report: Any, out: DocumentSet) -> bool:
        cfg = self.cfg
        blockers = list(getattr(gate, "blockers", []))
        if getattr(gate, "may_generate_documents", True) and not blockers:
            return True

        # Split by whether sign-off could legitimately clear it.
        non_overridable = [
            b for b in blockers
            if any(code in b for code in NON_OVERRIDABLE_CODES)
        ]
        overridable = [b for b in blockers if b not in non_overridable]

        if non_overridable:
            out.blocked = True
            out.blockers = blockers
            out.add(
                Severity.ERROR, "DATA_COMPLETENESS_BLOCKED",
                f"{len(non_overridable)} blocker(s) concern missing or "
                f"uncomputable data, which no sign-off can cure — the document "
                f"would simply be wrong. Fix these at the source and re-run.",
            )
            return False

        if not cfg.has_override():
            out.blocked = True
            out.blockers = blockers
            out.add(
                Severity.ERROR, "GATE_BLOCKED",
                f"{len(overridable)} unresolved compliance blocker(s). These turn "
                f"on legal judgment and can be cleared by recorded counsel "
                f"sign-off, which will be stamped into every document produced.",
            )
            return False

        out.overridden_blockers = overridable
        out.override_record = {
            "by": cfg.counsel_override_by,
            "reason": cfg.counsel_override_reason,
            "date": (cfg.counsel_override_date or dt.date.today()).isoformat(),
        }
        out.add(
            Severity.WARNING, "GATE_OVERRIDDEN",
            f"{len(overridable)} compliance blocker(s) cleared by "
            f"{cfg.counsel_override_by} on "
            f"{out.override_record['date']}: {cfg.counsel_override_reason}. This "
            f"record appears in every generated document and cannot be removed.",
        )
        return True

    # -- inputs -------------------------------------------------------------
    def _cut_list(
        self, selection: Any, scores: pd.DataFrame | None, out: DocumentSet
    ) -> pd.DataFrame | None:
        cut = getattr(selection, "cut_list", None)
        if cut is None and scores is not None and "selected" in scores.columns:
            cut = scores.loc[scores["selected"].fillna(False).astype(bool)]
        if cut is None or len(cut) == 0:
            out.blocked = True
            out.blockers = ["No cut list available to generate documents from."]
            out.add(Severity.ERROR, "NO_CUT_LIST", "No selected employees.")
            return None
        return cut.copy()

    # -- headers ------------------------------------------------------------
    def _header(self, title: str, extra: Sequence[str] = ()) -> list[str]:
        L = [f"# {title}", "", DRAFT_BANNER, ""]
        if self.cfg.has_override():
            L.append(
                f"> **Compliance gate cleared by {self.cfg.counsel_override_by} on "
                f"{(self.cfg.counsel_override_date or dt.date.today()).isoformat()}"
                f"** — {self.cfg.counsel_override_reason}"
            )
            L.append("")
        for e in extra:
            L.append(e)
        if extra:
            L.append("")
        return L

    def _employer(self) -> str:
        return self.cfg.employer_name or "[[EMPLOYER_NAME]]"

    # -- Cal-WARN notices ----------------------------------------------------
    def _warn_notice_employee(self, report: Any, cut: pd.DataFrame) -> GeneratedDocument:
        cfg = self.cfg
        sep = report.separation_date
        warns = [w for w in report.warn if w.jurisdiction == "California" and w.triggered]
        est = warns[0].establishment if warns else "[[ESTABLISHMENT]]"
        notice_date = (
            warns[0].earliest_notice_date.isoformat()
            if warns and warns[0].earliest_notice_date else "[[NOTICE_DATE]]"
        )

        L = self._header(
            "Notice of Mass Layoff — California WARN Act",
            [PRIVILEGE_BANNER],
        )
        L += [
            f"**{self._employer()}**  ",
            f"{cfg.employer_address or '[[EMPLOYER_ADDRESS]]'}  ",
            f"**Date of notice:** {notice_date}",
            "",
            "---",
            "",
            "To our affected employees:",
            "",
            f"This notice is given under the California Worker Adjustment and "
            f"Retraining Notification Act (Labor Code sections 1400 through "
            f"1408). {self._employer()} will carry out a mass layoff at its "
            f"establishment located at {est}.",
            "",
            "## Required information",
            "",
            f"- **Action:** Mass layoff at {est}.",
            f"- **Expected separation date:** {sep}. Separations are expected to "
            f"be permanent unless stated otherwise in your individual notice.",
            f"- **Number of employees affected:** {len(cut)}.",
            f"- **Job titles affected:** see the schedule at the end of this notice.",
            f"- **Bumping rights:** "
            + ("Employees have bumping rights as described in the applicable "
               "collective bargaining agreement. [[BUMPING_RIGHTS_DETAIL]]"
               if cfg.bumping_rights
               else "Employees do not have bumping rights."),
            f"- **Employer contact:** "
            f"{cfg.employer_contact_name or '[[CONTACT_NAME]]'}, "
            f"{cfg.employer_contact_email or '[[CONTACT_EMAIL]]'}, "
            f"{cfg.employer_contact_phone or '[[CONTACT_PHONE]]'}.",
            "",
            "## Transition services",
            "",
        ]

        coord = (cfg.service_coordination or "").lower()
        if coord == "lwdb":
            L.append(
                f"{self._employer()} plans to coordinate transition services, "
                f"including a rapid response orientation, through the Local "
                f"Workforce Development Board."
            )
        elif coord == "other":
            L.append(
                f"{self._employer()} plans to coordinate transition services "
                f"through [[COORDINATING_ENTITY]] rather than the Local Workforce "
                f"Development Board."
            )
        elif coord == "none":
            L.append(
                f"{self._employer()} does not plan to coordinate transition "
                f"services through the Local Workforce Development Board or any "
                f"other entity."
            )
        else:
            L.append("[[SERVICE_COORDINATION_STATEMENT]]")

        L += [
            "",
            f"**Local Workforce Development Board:** "
            f"{cfg.lwdb_name or '[[LWDB_NAME]]'}  ",
            f"Email: {cfg.lwdb_email or '[[LWDB_EMAIL]]'}  ",
            f"Phone: {cfg.lwdb_phone or '[[LWDB_PHONE]]'}",
            "",
            "Rapid response activities are services designed to help workers "
            "facing job loss move quickly to new employment. They may include "
            "career counseling, job search and placement assistance, information "
            "about retraining and education programs, and help applying for "
            "unemployment insurance and other support.",
            "",
            "## CalFresh food assistance",
            "",
            "CalFresh is California's food assistance program. It provides "
            "monthly benefits to help households with low income buy food. "
            "Eligibility is based on household size, income, and certain "
            "expenses, and a recent loss of employment is taken into account.",
            "",
            "- **CalFresh helpline:** 1-877-847-3663 (1-877-847-FOOD)",
            "- **Website:** https://www.getcalfresh.org",
            "- **County information:** https://www.cdss.ca.gov/food-nutrition/calfresh",
            "",
            "## Schedule of affected job titles",
            "",
            "| Job title | Number affected |",
            "|---|---|",
        ]
        if "job_title" in cut.columns:
            counts = cut["job_title"].astype("string").fillna("(unspecified)").value_counts()
            for title, n in counts.items():
                L.append(f"| {title} | {n} |")
        else:
            L.append("| [[JOB_TITLE_SCHEDULE]] | |")

        L += [
            "",
            "---",
            "",
            f"{cfg.signatory_name or '[[SIGNATORY_NAME]]'}  ",
            f"{cfg.signatory_title or '[[SIGNATORY_TITLE]]'}  ",
            f"{self._employer()}",
            "",
            "---",
            "",
            "**Counsel review checklist for this notice**",
            "",
            "- [ ] Confirm the establishment is correctly identified and that "
            "coverage was assessed on 12-month employment history, not current "
            "headcount alone",
            "- [ ] Confirm all four SB 617 disclosures are present and the "
            "contact details are functioning",
            "- [ ] Confirm the bumping rights statement matches any applicable CBA",
            "- [ ] Confirm delivery to the EDD, the Local Workforce Development "
            "Board, and the chief elected official of each city and county",
            "- [ ] Confirm the 60-day period runs from actual delivery",
        ]
        return GeneratedDocument(
            doc_type="warn_notice_employee",
            title="Cal-WARN notice to affected employees",
            body="\n".join(L),
            filename="warn_notice_employees.md",
            audience="Affected employees",
            review_required=("employment counsel", "SB 617 content check"),
        )

    def _warn_notice_agency(
        self, report: Any, cut: pd.DataFrame
    ) -> list[GeneratedDocument]:
        recipients = [
            ("edd", "California Employment Development Department"),
            ("lwdb", self.cfg.lwdb_name or "Local Workforce Development Board"),
            ("local_official", "Chief elected official of each affected city and county"),
        ]
        docs: list[GeneratedDocument] = []
        sep = report.separation_date
        warns = [w for w in report.warn if w.jurisdiction == "California" and w.triggered]
        est = warns[0].establishment if warns else "[[ESTABLISHMENT]]"

        for key, name in recipients:
            L = self._header(f"Cal-WARN Notice — {name}", [PRIVILEGE_BANNER])
            L += [
                f"**To:** {name}",
                f"**From:** {self._employer()}, "
                f"{self.cfg.employer_address or '[[EMPLOYER_ADDRESS]]'}",
                f"**Date:** [[NOTICE_DELIVERY_DATE]]",
                "",
                "Notice is given under Labor Code section 1401 of a mass layoff:",
                "",
                f"- **Establishment address:** {est} — [[ESTABLISHMENT_ADDRESS]]",
                f"- **Expected separation date:** {sep}",
                f"- **Employees affected:** {len(cut)}",
                f"- **Permanent or temporary:** [[PERMANENT_OR_TEMPORARY]]",
                f"- **Bumping rights:** "
                + ("yes — see CBA" if self.cfg.bumping_rights else "no"),
                f"- **Employer contact:** "
                f"{self.cfg.employer_contact_name or '[[CONTACT_NAME]]'}, "
                f"{self.cfg.employer_contact_email or '[[CONTACT_EMAIL]]'}, "
                f"{self.cfg.employer_contact_phone or '[[CONTACT_PHONE]]'}",
                "",
                "A schedule of affected job titles and the number of employees in "
                "each is attached.",
                "",
                f"{self.cfg.signatory_name or '[[SIGNATORY_NAME]]'}  ",
                f"{self.cfg.signatory_title or '[[SIGNATORY_TITLE]]'}",
            ]
            docs.append(GeneratedDocument(
                doc_type="warn_notice_agency",
                title=f"Cal-WARN notice — {name}",
                body="\n".join(L),
                filename=f"warn_notice_{key}.md",
                audience=name,
                review_required=("employment counsel",),
            ))
        return docs

    # -- separation letters --------------------------------------------------
    def _separation_letters(
        self, report: Any, cut: pd.DataFrame, register: pd.DataFrame | None
    ) -> list[GeneratedDocument]:
        docs: list[GeneratedDocument] = []
        pay_by_id: dict[str, pd.Series] = {}
        if register is not None and not register.empty:
            for _, r in register.iterrows():
                if pd.notna(r.get("employee_id")):
                    pay_by_id[str(r["employee_id"])] = r

        for _, row in cut.iterrows():
            emp_id = str(row.get("employee_id")) if pd.notna(row.get("employee_id")) else None
            name = row.get("full_name")
            name = name if isinstance(name, str) and name.strip() else "[[EMPLOYEE_NAME]]"
            title = row.get("job_title") or "[[JOB_TITLE]]"
            sep = report.separation_date
            p = pay_by_id.get(emp_id or "")

            L = self._header("Notice of Separation of Employment")
            L += [
                f"**To:** {name} ({emp_id or '[[EMPLOYEE_ID]]'})  ",
                f"**Position:** {title}  ",
                f"**Date:** [[LETTER_DATE]]",
                "",
                "---",
                "",
                f"Dear {name},",
                "",
                f"This letter confirms that your employment with "
                f"{self._employer()} will end on **{sep}** as a result of a "
                f"reduction in force. This decision reflects the elimination of "
                f"positions and is not a statement about your conduct.",
                "",
                "## Your final pay",
                "",
                "California law requires that all final wages, including all "
                "vested vacation, be paid to you at the time your employment "
                "ends. You will receive:",
                "",
            ]
            if p is not None and pd.notna(p.get("vacation_payout")):
                L.append(
                    f"- Accrued, unused vacation: **${float(p['vacation_payout']):,.2f}** "
                    f"({float(p['vacation_hours']):.1f} hours)"
                    if pd.notna(p.get("vacation_hours"))
                    else f"- Accrued, unused vacation: "
                         f"**${float(p['vacation_payout']):,.2f}**"
                )
            else:
                L.append("- Accrued, unused vacation: [[VACATION_PAYOUT]]")
            L.append("- Wages earned through your last day: [[FINAL_WAGES]]")
            L += [
                "",
                "## Separation pay",
                "",
            ]
            if p is not None and pd.notna(p.get("severance_gross")):
                L += [
                    f"You are being offered separation pay of "
                    f"**${float(p['severance_gross']):,.2f}** "
                    f"({float(p['severance_weeks']):.0f} weeks), calculated under "
                    f"the company's severance guideline based on your length of "
                    f"service.",
                    "",
                    "This offer is separate from your final wages. **Your final "
                    "wages and vacation payout are yours regardless of whether "
                    "you accept the separation agreement.** Separation pay is "
                    "conditioned on signing the enclosed agreement; your earned "
                    "wages are not.",
                ]
            else:
                L.append("[[SEVERANCE_TERMS]]")
            L += [
                "",
                "## Benefits",
                "",
                "Your group health coverage will end on [[BENEFITS_END_DATE]]. "
                "You will receive information about continuing coverage under "
                "COBRA and Cal-COBRA separately, within the timeframe required by "
                "law.",
                "",
                "## Included with this letter",
                "",
                "- Notice of Change in Relationship (EDD)",
                "- *For Your Benefit: California's Programs for the Unemployed* "
                "(DE 2320)",
                "- Health Insurance Premium Payment (HIPP) notice",
                "- Separation agreement and general release (if applicable)",
                "",
                "## Questions",
                "",
                f"Please contact "
                f"{self.cfg.employer_contact_name or '[[CONTACT_NAME]]'} at "
                f"{self.cfg.employer_contact_email or '[[CONTACT_EMAIL]]'} or "
                f"{self.cfg.employer_contact_phone or '[[CONTACT_PHONE]]'}.",
                "",
                f"{self.cfg.signatory_name or '[[SIGNATORY_NAME]]'}  ",
                f"{self.cfg.signatory_title or '[[SIGNATORY_TITLE]]'}  ",
                f"{self._employer()}",
            ]
            docs.append(GeneratedDocument(
                doc_type="separation_letter",
                title=f"Separation letter — {emp_id or name}",
                body="\n".join(L),
                filename=f"separation_letter_{emp_id or 'unknown'}.md",
                audience="Individual employee",
                employee_id=emp_id,
                review_required=("employment counsel",),
            ))
        return docs

    # -- OWBPA disclosure ----------------------------------------------------
    def _owbpa_disclosure(
        self, cut: pd.DataFrame, scores: pd.DataFrame | None, out: DocumentSet
    ) -> GeneratedDocument | None:
        cfg = self.cfg
        if not cfg.decisional_unit.strip():
            out.add(
                Severity.ERROR, "DECISIONAL_UNIT_NOT_DEFINED",
                "The OWBPA disclosure was not generated because no decisional "
                "unit is defined. Defining it is a legal judgment about the scope "
                "of the decision — it is frequently not the same as the "
                "comparison groups used for scoring, and the generator will not "
                "infer one. Set decisional_unit and, to scope the listing, "
                "decisional_unit_column and decisional_unit_value.",
            )
            return None

        if scores is None or scores.empty:
            out.add(
                Severity.ERROR, "OWBPA_NEEDS_FULL_POPULATION",
                "The OWBPA disclosure must list both selected and non-selected "
                "employees in the decisional unit, so the full scored population "
                "is required — a cut list alone is not enough.",
            )
            return None

        pop = scores
        if cfg.decisional_unit_column and cfg.decisional_unit_column in scores.columns:
            pop = scores.loc[
                scores[cfg.decisional_unit_column].astype("string")
                == cfg.decisional_unit_value
            ]
        if pop.empty:
            out.add(
                Severity.ERROR, "DECISIONAL_UNIT_EMPTY",
                f"No employees match the configured decisional unit "
                f"({cfg.decisional_unit_column}={cfg.decisional_unit_value!r}).",
            )
            return None

        if "age_years" not in pop.columns:
            out.add(
                Severity.ERROR, "OWBPA_NEEDS_AGES",
                "The disclosure requires the ages of everyone in the decisional "
                "unit and no age data is present.",
            )
            return None

        selected = pop.loc[pop["selected"].fillna(False).astype(bool)]
        retained = pop.loc[~pop["selected"].fillna(False).astype(bool)]

        L = self._header(
            "OWBPA Disclosure — Decisional Unit Information",
            [PRIVILEGE_BANNER],
        )
        L += [
            "*Provided under the Older Workers Benefit Protection Act, "
            "29 U.S.C. section 626(f)(1)(H), to each employee age 40 or older "
            "who is asked to release age discrimination claims as part of a group "
            "termination program.*",
            "",
            "## Decisional unit",
            "",
            cfg.decisional_unit,
            "",
            "## Eligibility factors",
            "",
            "[[ELIGIBILITY_FACTORS]]",
            "",
            "*State the factors used to determine who was and was not selected, "
            "in the terms the decision was actually made in.*",
            "",
            "## Time limits",
            "",
            f"Employees have {cfg.consideration_period_days} days from receipt of "
            f"the agreement to consider it, and {cfg.revocation_period_days} days "
            f"after signing to revoke.",
            "",
            f"## Employees selected for the program ({len(selected)})",
            "",
            "| Job title | Age |",
            "|---|---|",
        ]
        for _, r in selected.sort_values("age_years", na_position="last").iterrows():
            age = r.get("age_years")
            L.append(
                f"| {r.get('job_title') or '(unspecified)'} | "
                f"{int(age) if pd.notna(age) else '[[AGE]]'} |"
            )

        L += [
            "",
            f"## Employees in the same unit not selected ({len(retained)})",
            "",
            "| Job title | Age |",
            "|---|---|",
        ]
        for _, r in retained.sort_values("age_years", na_position="last").iterrows():
            age = r.get("age_years")
            L.append(
                f"| {r.get('job_title') or '(unspecified)'} | "
                f"{int(age) if pd.notna(age) else '[[AGE]]'} |"
            )

        L += [
            "",
            "---",
            "",
            "**Before this is distributed**",
            "",
            "- [ ] Counsel has confirmed the decisional unit is correctly scoped — "
            "too narrow understates the comparison and can void the release",
            "- [ ] Eligibility factors are stated in the terms the decision was "
            "actually made in",
            "- [ ] No names are included; the disclosure lists job titles and ages "
            "only",
            "- [ ] The employer has reviewed the age distribution this reveals, "
            "because a reader will",
        ]
        return GeneratedDocument(
            doc_type="owbpa_disclosure",
            title="OWBPA decisional unit disclosure",
            body="\n".join(L),
            filename="owbpa_disclosure.md",
            audience="Employees age 40+ receiving a release",
            review_required=("employment counsel", "decisional unit scoping"),
        )

    # -- severance skeleton --------------------------------------------------
    def _severance_skeleton(
        self, report: Any, register: pd.DataFrame | None
    ) -> GeneratedDocument:
        cfg = self.cfg
        L = self._header(
            "Separation Agreement — SKELETON FOR COUNSEL",
            [PRIVILEGE_BANNER],
        )
        L += [
            "> **This is not a contract and must not be sent to anyone.** It "
            "enumerates the elements a California separation agreement with a "
            "release typically needs, and records the computed figures. A release "
            "of claims must be drafted by a lawyer. An ADEA release that misses "
            "an OWBPA element is unenforceable as to age claims while remaining "
            "enforceable as to everything else — the employer pays for a release "
            "it does not receive, and the defect is invisible until it is "
            "litigated.",
            "",
            "## Computed figures available for drafting",
            "",
        ]
        if register is not None and not register.empty:
            L += [
                "| Employee | Weeks | Severance | Vacation payout |",
                "|---|---|---|---|",
            ]
            for _, r in register.iterrows():
                if r.get("status") != "computed":
                    continue
                L.append(
                    f"| {r.get('employee_id')} | "
                    f"{float(r.get('severance_weeks') or 0):.0f} | "
                    f"${float(r.get('severance_gross') or 0):,.2f} | "
                    f"${float(r.get('vacation_payout') or 0):,.2f} |"
                )
        else:
            L.append("[[SEVERANCE_FIGURES]] — run box 6 to populate.")

        L += [
            "",
            "## Elements for counsel to draft",
            "",
            "1. **Consideration.** The separation payment must be something the "
            "employee is not already entitled to. Earned wages and vested "
            "vacation cannot serve as consideration for a release "
            "(Lab. Code § 206.5).",
            "2. **Scope of release.** [[RELEASE_SCOPE]]",
            "3. **Civil Code § 1542 waiver.** [[SECTION_1542_LANGUAGE]]",
            "4. **Claims that cannot be released** — including workers' "
            "compensation, unemployment insurance, and the right to file a charge "
            "with or participate in an investigation by the EEOC, DFEH/CRD, or "
            "NLRB. [[NON_RELEASABLE_CLAIMS]]",
            "5. **OWBPA elements** for any employee age 40 or older:",
            f"   - Written in plain language the employee can understand",
            f"   - Specific reference to ADEA rights",
            f"   - No waiver of rights arising after the signature date",
            f"   - Advice in writing to consult an attorney",
            f"   - {cfg.consideration_period_days} days to consider",
            f"   - {cfg.revocation_period_days} days to revoke after signing",
            f"   - Decisional unit disclosure attached",
            "6. **Confidentiality and non-disparagement.** California limits both "
            "in agreements involving claims of harassment or discrimination "
            "(Gov. Code § 12964.5; Code Civ. Proc. § 1001). [[CONFIDENTIALITY]]",
            "7. **Return of property, cooperation, references.** [[MISC_TERMS]]",
            "",
            "## What must not be in it",
            "",
            "- Any condition on the payment of earned wages or vested vacation",
            "- Any waiver of the right to file a charge with a government agency",
            "- Any provision preventing disclosure of unlawful acts in the "
            "workplace",
            "- A consideration or revocation period shorter than the statutory "
            "minimum, or any purported waiver of the revocation period",
        ]
        return GeneratedDocument(
            doc_type="severance_agreement_draft",
            title="Separation agreement skeleton (for counsel)",
            body="\n".join(L),
            filename="severance_agreement_SKELETON.md",
            audience="Employment counsel only",
            review_required=("employment counsel — drafting required",),
        )

    # -- manager script ------------------------------------------------------
    def _manager_script(self, report: Any) -> GeneratedDocument:
        sep = report.separation_date
        L = self._header("Manager Talking Points — Notification Meeting")
        L += [
            "These meetings are short. The employee will remember how it felt far "
            "longer than what was said, and most of the legal risk in a RIF comes "
            "from what a manager improvises in the room.",
            "",
            "## Before the meeting",
            "",
            "- Read the separation letter and know the employee's final pay and "
            "severance figures",
            "- Have HR present; do not conduct the meeting alone",
            "- Book a private room and allow 15 minutes",
            "- Have the packet physically in hand before you start",
            "",
            "## Opening (say this early and plainly)",
            "",
            f"> \"I have difficult news. Your position is being eliminated as part "
            f"of a reduction in force, effective {sep}. This decision is final.\"",
            "",
            "Do not lead with context, apologies, or business rationale. Say what "
            "is happening in the first fifteen seconds, then stop and let them "
            "react.",
            "",
            "## The reason",
            "",
            "> \"The company is reducing roles across several teams. Positions "
            "were selected using criteria applied consistently across the group. "
            "This is not about your conduct.\"",
            "",
            "If asked why them specifically: **do not improvise a reason.**",
            "",
            "> \"I'm not able to go into the individual comparison. HR can walk "
            "you through the process that was used.\"",
            "",
            "## What to cover",
            "",
            "1. Last day of work and last day of pay",
            "2. Final pay: wages and vacation are paid at separation regardless "
            "of whether they sign anything",
            "3. Severance offer and the deadline to consider it",
            "4. Benefits end date and that COBRA information follows",
            "5. Logistics: equipment, access, personal belongings",
            "6. Who to contact with questions",
            "",
            "## Do not say",
            "",
            "Each of these has generated litigation:",
            "",
            "- **Anything about age, retirement, or \"next chapter.\"** Not "
            "\"you've earned a rest,\" not \"you're close to retirement anyway.\" "
            "This is the single most common source of an age claim.",
            "- **Anything about health, leave, disability, or pregnancy** — "
            "including sympathetic references to a recent absence.",
            "- **A performance reason, unless performance was in fact the "
            "documented criterion** and you are certain of it. Saying \"it was "
            "performance\" when the criteria were mixed contradicts the record.",
            "- **Other employees' status.** Not who else is affected, not who was "
            "retained, not why.",
            "- **Promises.** No assurances about rehire, references, extensions, "
            "or additional money.",
            "- **Negotiation.** You have no authority to change terms in the room. "
            "\"I'll pass that to HR\" is the complete answer.",
            "- **Speculation about the company's finances or future.**",
            "- **\"I fought for you\" or \"this wasn't my decision.\"** It "
            "undermines the process and invites the employee to look for the real "
            "reason.",
            "",
            "## If the employee becomes upset",
            "",
            "- Silence is fine. Let it be uncomfortable.",
            "- \"That's a fair reaction\" is better than \"I understand.\"",
            "- Do not defend the decision or argue the criteria.",
            "- If they become distressed in a way that concerns you, stay with "
            "them, involve HR, and follow your escalation procedure.",
            "",
            "## If the employee threatens legal action",
            "",
            "> \"I understand. I'm not the right person for that conversation — "
            "please contact [[HR_CONTACT]] and they'll take it from there.\"",
            "",
            "Then end the meeting. Do not respond to the substance, do not "
            "apologize for the decision, and report it to HR the same day.",
            "",
            "## After",
            "",
            "- Notify HR that the meeting occurred and note anything unusual",
            "- Do not discuss the conversation with the remaining team",
            "- Expect questions from the team and route them to the prepared "
            "messaging",
        ]
        return GeneratedDocument(
            doc_type="manager_script",
            title="Manager talking points",
            body="\n".join(L),
            filename="manager_talking_points.md",
            audience="Managers conducting notifications",
            review_required=("HR review",),
        )

    # -- employee FAQ --------------------------------------------------------
    def _employee_faq(self, report: Any, pay: Any) -> GeneratedDocument:
        sep = report.separation_date
        L = self._header("Employee FAQ — Reduction in Force")
        L += [
            "## When does my employment end?",
            "",
            f"Your last day is {sep} unless your individual notice says otherwise.",
            "",
            "## When do I get my final paycheck?",
            "",
            "California law requires your employer to pay all final wages, "
            "including every hour of vested vacation, at the time your employment "
            "ends. You do not have to sign anything to receive it. If it is late, "
            "you may be entitled to a penalty of a day's wages for each day of "
            "delay, up to 30 days.",
            "",
            "## Is my unused sick leave paid out?",
            "",
            "Generally no, unless your employer combines sick time and vacation "
            "into a single PTO bank — in which case the whole balance is usually "
            "paid. Your separation letter states what you are receiving. If you "
            "are rehired within a year, unused sick leave is generally restored.",
            "",
            "## Do I have to sign the separation agreement?",
            "",
            "No. It is optional. Signing is how you receive the severance "
            "payment; not signing does not affect your final wages, your vacation "
            "payout, your unemployment eligibility, or your right to continue "
            "health coverage.",
            "",
            "## How long do I have to decide?",
            "",
            f"If you are 40 or older, you have "
            f"{self.cfg.consideration_period_days} days to consider the agreement "
            f"and {self.cfg.revocation_period_days} days after signing to change "
            f"your mind. You are advised in writing to consult an attorney, and "
            f"you should feel free to do so.",
            "",
            "## Can I collect unemployment?",
            "",
            "A layoff is generally a qualifying separation. File with the "
            "Employment Development Department as soon as your employment ends — "
            "benefits are not retroactive to before you file. Severance paid as "
            "dismissal pay generally does not reduce or delay benefits in "
            "California, though the EDD makes that determination.",
            "",
            "- **File online:** https://edd.ca.gov/en/unemployment",
            "- **Phone:** 1-800-300-5616",
            "",
            "## What happens to my health insurance?",
            "",
            "Coverage ends on the date in your letter. You will receive a COBRA "
            "election notice explaining how to continue the same coverage at your "
            "own expense. Covered California may be less expensive — compare "
            "before electing COBRA, and note that losing job-based coverage opens "
            "a special enrollment period.",
            "",
            "- **Covered California:** https://www.coveredca.com — 1-800-300-1506",
            "",
            "## What about my 401(k)?",
            "",
            "Your balance is yours. You can generally leave it in the plan, roll "
            "it into an IRA or a new employer's plan, or withdraw it — though "
            "withdrawal before 59½ usually carries taxes and a penalty. The plan "
            "administrator will send instructions. [[401K_ADMINISTRATOR]]",
            "",
            "## Can I get help with food or bills?",
            "",
            "CalFresh provides monthly food benefits and a recent job loss is "
            "taken into account in determining eligibility.",
            "",
            "- **CalFresh:** https://www.getcalfresh.org — 1-877-847-3663",
            "",
            "## Will I get a reference?",
            "",
            "[[REFERENCE_POLICY]]",
            "",
            "## Who do I contact?",
            "",
            f"{self.cfg.employer_contact_name or '[[CONTACT_NAME]]'} — "
            f"{self.cfg.employer_contact_email or '[[CONTACT_EMAIL]]'} — "
            f"{self.cfg.employer_contact_phone or '[[CONTACT_PHONE]]'}",
        ]
        return GeneratedDocument(
            doc_type="employee_faq",
            title="Employee FAQ",
            body="\n".join(L),
            filename="employee_faq.md",
            audience="Affected employees",
            review_required=("HR review", "benefits confirmation"),
        )

    # -- checklist -----------------------------------------------------------
    def _checklist(self, report: Any) -> GeneratedDocument:
        L = self._header("Notice Day Checklist")
        L += ["## Before notice day", ""]
        obligations = sorted(
            [o for o in report.obligations if o.due_date],
            key=lambda o: o.due_date,
        )
        sep = report.separation_date
        for o in obligations:
            when = o.due_date.isoformat()
            flag = " **(MISSED)**" if o.missed else ""
            L.append(f"- [ ] {when} — {o.title} *({o.authority})*{flag}")
        L += [
            "",
            "## Notice day",
            "",
            "- [ ] Final paychecks physically available before the first meeting "
            "begins — wages and vested vacation are due at separation, not on the "
            "next payday",
            "- [ ] Packets assembled and checked against the employee list",
            "- [ ] Managers briefed on talking points; HR assigned to each meeting",
            "- [ ] Private rooms booked",
            "- [ ] IT briefed on access timing — coordinate with meeting times, "
            "not before them",
            "- [ ] Support available for employees who need a moment before "
            "leaving",
            "- [ ] Escalation path identified for distress or threatened claims",
            "",
            "## Same day, after notifications",
            "",
            "- [ ] Confirm every scheduled meeting occurred; follow up on absences "
            "and employees on leave through the agreed route",
            "- [ ] Communicate to the remaining team using the prepared messaging",
            "- [ ] Log anything unusual said in a meeting",
            "",
            f"## Following the separation date ({sep})",
            "",
            "- [ ] COBRA election notices issued within the statutory window",
            "- [ ] Track the consideration and revocation periods per employee",
            "- [ ] Do not process any release until its revocation period expires",
            "- [ ] Preserve the complete decision record — selection criteria, "
            "scores, impact analysis, and approvals",
        ]
        return GeneratedDocument(
            doc_type="notice_day_checklist",
            title="Notice day checklist",
            body="\n".join(L),
            filename="notice_day_checklist.md",
            audience="HR and project team",
        )

    # -- HR summary ----------------------------------------------------------
    def _hr_summary(
        self, report: Any, cut: pd.DataFrame, pay: Any
    ) -> GeneratedDocument:
        L = self._header("Internal Summary — HR and Project Team", [PRIVILEGE_BANNER])
        L += [
            f"- **Separation date:** {report.separation_date}",
            f"- **Employees affected:** {len(cut)}",
            f"- **Cal-WARN triggered:** {'yes' if report.warn_triggered else 'no'}",
            f"- **Obligations tracked:** {len(report.obligations)}",
            f"- **Missed deadlines:** {len(report.missed_deadlines)}",
            "",
        ]
        totals = getattr(getattr(pay, "report", None), "totals", None)
        if totals:
            L += [
                "## Cost",
                "",
                f"- Severance: ${totals.get('severance_gross', 0):,.0f}",
                f"- Vacation payout: ${totals.get('vacation_payout', 0):,.0f}",
                f"- Total employer cost: "
                f"${totals.get('total_employer_cost', 0):,.0f}",
                "",
            ]
        L += [
            "## Distribution by department",
            "",
            "| Department | Affected |",
            "|---|---|",
        ]
        if "department" in cut.columns:
            for dept, n in cut["department"].astype("string").fillna(
                "(unassigned)"
            ).value_counts().items():
                L.append(f"| {dept} | {n} |")
        L += [
            "",
            "## Reminders",
            "",
            "- The decision record is discoverable in its entirety, including "
            "scenarios modeled and not pursued",
            "- Do not circulate the adverse impact analysis outside the "
            "counsel-directed group",
            "- Final pay is due at separation and does not wait on a signature",
            "- Track revocation periods; a release is not effective until its "
            "period expires",
        ]
        return GeneratedDocument(
            doc_type="hr_summary",
            title="Internal HR summary",
            body="\n".join(L),
            filename="hr_summary.md",
            audience="HR and project team",
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    from .adverse_impact import AdverseImpactAnalyzer
    from .ca_compliance import ComplianceConfig, ComplianceEngine
    from .selection_criteria import SelectionEngine, load_plan
    from .severance_pay import PayConfig, SeveranceFormula, SeverancePayEngine
    from .workforce_data import load_workforce_csv

    ap = argparse.ArgumentParser(description="Generate draft RIF documents.")
    ap.add_argument("csv_path")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--separation-date", required=True)
    ap.add_argument("--notice-date", default=None)
    ap.add_argument("--leave-policy", default="separate", choices=["separate", "combined"])
    ap.add_argument("--employer", default="")
    ap.add_argument("--decisional-unit", default="")
    ap.add_argument("--override-by", default="")
    ap.add_argument("--override-reason", default="")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args(argv)

    plan = load_plan(args.plan)
    ingest = load_workforce_csv(
        args.csv_path,
        as_of=str(plan.as_of_date) if plan.as_of_date else args.separation_date,
    )
    selection = SelectionEngine(plan).run(ingest.data)
    impact = AdverseImpactAnalyzer().run(selection.scores, scenario=plan.plan_name)
    comp = ComplianceEngine(ComplianceConfig(
        proposed_separation_date=args.separation_date,
        notice_date=args.notice_date,
        service_coordination="lwdb",
        lwdb_email="board@example.gov", lwdb_phone="(555) 555-0100",
        employer_contact_email="hr@example.com",
        employer_contact_phone="(555) 555-0199",
    )).run(selection.scores, impact=impact, selection=selection)
    pay = SeverancePayEngine(PayConfig(
        separation_date=args.separation_date,
        leave_policy=args.leave_policy,
        formula=SeveranceFormula(),
    )).run(selection.cut_list)

    docs = DocumentGenerator(DocumentConfig(
        employer_name=args.employer,
        decisional_unit=args.decisional_unit,
        counsel_override_by=args.override_by,
        counsel_override_reason=args.override_reason,
    )).generate(compliance=comp, selection=selection, pay=pay, scores=selection.scores)

    print(docs.summary_markdown())

    if args.outdir:
        paths = docs.write(args.outdir)
        print(f"\nWrote {len(paths)} file(s) to {args.outdir}")

    return 1 if docs.blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
