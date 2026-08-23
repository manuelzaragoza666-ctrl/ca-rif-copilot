"""
task_tracker.py
===============

Execution & Task Tracker for the California RIF Copilot (box 9).

Turns the obligations, documents, and payments produced by the other boxes into
a dated, owned task list — statutory deadlines, notice-day logistics, and a
per-employee track covering delivery, acknowledgment, final pay, and the
consideration and revocation periods.

Three things this tracker refuses to do
---------------------------------------
**It will not let a statutory deadline be edited.** Dates derived from Labor
Code section 1401, section 201, or OWBPA are marked immovable. Moving the
separation date moves them, which is the correct way to change them; opening
the task and typing a new date is not. A tracker that lets someone quietly push
a WARN deadline is worse than a spreadsheet, because it looks authoritative.

**It will not close a task before its date arrives** where the date is the
point. A release cannot be processed until its revocation period expires, so
the task that says so cannot be marked complete early — not by an administrator
in a hurry, not to clear a board.

**It will not generate follow-up reminders during a consideration period.**
OWBPA gives an employee 45 days to consider a release, and a system that
prompts someone to chase a signature on day three converts a statutory
protection into a pressure campaign. Follow-up tasks appear after the period
ends, and the tracker says why.

Binding
-------
The board is bound to the approval fingerprint from box 8. If the plan changes,
the task list generated against the old version is stale and says so, rather
than quietly tracking work toward a plan nobody approved.

Usage
-----
    from .task_tracker import TaskBoard, TrackerConfig

    board = TaskBoard.build(
        compliance=compliance, pay=pay, documents=docs,
        package=package, config=TrackerConfig(...),
    )
    board.complete("STAT-FINAL_PAY", by="Payroll", evidence="Register #4471")
    print(board.to_markdown())
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .workforce_data import Severity

__all__ = [
    "Task",
    "TaskBoard",
    "TrackerConfig",
    "TaskError",
    "Acknowledgment",
    "STATUSES",
]

__version__ = "1.0.0"


class TaskError(ValueError):
    """Raised when a task action is not permitted."""


STATUSES = ("not_started", "in_progress", "blocked", "complete", "not_applicable")

#: Categories, in the order they appear on the board.
CATEGORIES = ("statutory", "approval", "document", "logistics", "per_employee")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrackerConfig:
    notice_day: dt.date | None = None
    #: Default owners by category, used when a task has no specific owner.
    owners: dict[str, str] = field(default_factory=lambda: {
        "statutory": "HR Director",
        "approval": "HR Business Partner",
        "document": "HR Business Partner",
        "logistics": "HR Operations",
        "per_employee": "Assigned manager",
    })
    #: Days before a due date at which a task starts showing as due soon.
    due_soon_days: int = 7
    #: OWBPA consideration days, for scheduling the post-period follow-up.
    consideration_days: int = 45
    revocation_days: int = 7
    #: Generate a per-employee track. Off for a purely programme-level board.
    per_employee_tasks: bool = True

    def owner_for(self, category: str) -> str:
        return self.owners.get(category, "Unassigned")


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


@dataclass
class Task:
    id: str
    title: str
    category: str
    due_date: dt.date | None = None
    owner: str = ""
    authority: str = ""
    description: str = ""
    employee_id: str | None = None
    depends_on: tuple[str, ...] = ()
    #: A statutory date. Editable only by changing the underlying facts.
    immovable: bool = False
    #: Completion requires a reference to evidence (a document, a payroll run).
    evidence_required: bool = False
    #: Cannot be completed before this date, because the waiting is the point.
    not_before: dt.date | None = None
    status: str = "not_started"
    completed_at: str | None = None
    completed_by: str = ""
    evidence_ref: str = ""
    notes: str = ""

    def is_overdue(self, today: dt.date) -> bool:
        return (
            self.due_date is not None
            and self.status not in ("complete", "not_applicable")
            and self.due_date < today
        )

    def is_due_soon(self, today: dt.date, window: int) -> bool:
        return (
            self.due_date is not None
            and self.status not in ("complete", "not_applicable")
            and today <= self.due_date <= today + dt.timedelta(days=window)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "category": self.category,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "owner": self.owner, "authority": self.authority,
            "employee_id": self.employee_id,
            "depends_on": "|".join(self.depends_on),
            "immovable": self.immovable,
            "evidence_required": self.evidence_required,
            "not_before": self.not_before.isoformat() if self.not_before else None,
            "status": self.status, "completed_at": self.completed_at,
            "completed_by": self.completed_by, "evidence_ref": self.evidence_ref,
            "notes": self.notes, "description": self.description,
        }


@dataclass
class Acknowledgment:
    """A per-employee receipt the employer needs to be able to prove."""

    employee_id: str
    item: str                 # notice | packet | release | final_pay
    received: bool = False
    received_at: str | None = None
    recorded_by: str = ""
    reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "employee_id": self.employee_id, "item": self.item,
            "received": self.received, "received_at": self.received_at,
            "recorded_by": self.recorded_by, "reference": self.reference,
        }


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


class TaskBoard:
    """A dated, owned task list derived from the pipeline outputs."""

    def __init__(
        self,
        tasks: Sequence[Task] = (),
        config: TrackerConfig | None = None,
        fingerprint: str = "",
        separation_date: dt.date | None = None,
        scenario: str = "",
    ) -> None:
        self.config = config or TrackerConfig()
        self._tasks: dict[str, Task] = {t.id: t for t in tasks}
        self.fingerprint = fingerprint
        self.separation_date = separation_date
        self.scenario = scenario
        self.acknowledgments: list[Acknowledgment] = []
        self.log: list[dict[str, Any]] = []
        self.findings: list[dict[str, Any]] = []

    # -- construction -------------------------------------------------------
    @classmethod
    def build(
        cls,
        compliance: Any = None,
        pay: Any = None,
        documents: Any = None,
        package: Any = None,
        selection: Any = None,
        config: TrackerConfig | None = None,
    ) -> "TaskBoard":
        cfg = config or TrackerConfig()
        report = getattr(compliance, "report", compliance)
        sep = None
        if report is not None and getattr(report, "separation_date", None):
            sep = pd.Timestamp(report.separation_date).date()

        board = cls(
            config=cfg,
            fingerprint=getattr(package, "fingerprint", ""),
            separation_date=sep,
            scenario=getattr(package, "scenario", "") or getattr(report, "scenario", ""),
        )

        if report is None:
            board._note(
                Severity.ERROR, "NO_COMPLIANCE_INPUT",
                "No compliance analysis supplied, so no statutory tasks could be "
                "derived. The board would look complete while tracking nothing "
                "that matters.",
            )
            return board

        board._add_statutory(report)
        board._add_logistics(report, sep)
        board._add_document_tasks(documents)
        board._add_approval_tasks(package)
        if cfg.per_employee_tasks:
            board._add_per_employee(pay, selection, sep, report)
        board._check_overdue_at_build()
        return board

    # -- generators ---------------------------------------------------------
    def _note(self, severity: str, code: str, message: str) -> None:
        self.findings.append({"severity": severity, "code": code, "message": message})

    def _add(self, task: Task) -> Task:
        if not task.owner:
            task.owner = self.config.owner_for(task.category)
        self._tasks[task.id] = task
        return task

    def _add_statutory(self, report: Any) -> None:
        for o in getattr(report, "obligations", []):
            self._add(Task(
                id=f"STAT-{o.code}",
                title=o.title,
                category="statutory",
                due_date=o.due_date,
                authority=o.authority,
                description=o.description,
                immovable=True,
                evidence_required=True,
                notes=(
                    "Deadline derived from statute. To change it, change the "
                    "underlying dates and regenerate — it cannot be edited here."
                ),
                status="not_started",
            ))
        if not getattr(report, "obligations", []):
            self._note(
                Severity.WARNING, "NO_STATUTORY_TASKS",
                "The compliance analysis produced no dated obligations. That is "
                "unusual for a separation of any size; verify box 5 ran fully.",
            )

    def _add_logistics(self, report: Any, sep: dt.date | None) -> None:
        notice_day = self.config.notice_day or sep
        pre = (notice_day - dt.timedelta(days=3)) if notice_day else None

        items = [
            ("PREP-PAYROLL",
             "Final paychecks prepared and physically available",
             pre,
             "Final wages and vested vacation are due at the moment of "
             "separation, not on the next payday. Checks must exist before the "
             "first meeting begins.",
             "Lab. Code §§ 201, 203", True),
            ("PREP-PACKETS", "Separation packets assembled and checked", pre,
             "One packet per employee, checked against the cut list.", "", True),
            ("PREP-MANAGERS", "Managers briefed on talking points", pre,
             "Every notifying manager has read the script, including the list of "
             "things not to say.", "", False),
            ("PREP-ROOMS", "Private rooms booked and schedule set", pre,
             "One private space per concurrent meeting.", "", False),
            ("PREP-IT", "IT briefed on access timing", pre,
             "Access changes coordinate with meeting times, not before them. "
             "Cutting access early tells people before their manager does.",
             "", False),
            ("PREP-SUPPORT", "Support arrangements confirmed", pre,
             "Someone available for employees who need a moment, and an "
             "escalation path for distress or a threatened claim.", "", False),
            ("DAY-NOTIFY", "Conduct notification meetings", notice_day,
             "HR present at each meeting.", "", False),
            ("DAY-DELIVER-PAY", "Deliver final pay at separation", notice_day,
             "Handed over in the meeting, not mailed later.",
             "Lab. Code § 201", True),
            ("DAY-REMAINING", "Communicate to the remaining team", notice_day,
             "Use the prepared messaging; do not discuss individuals.", "", False),
            ("POST-LOG", "Log any unusual statements or incidents", notice_day,
             "Anything said in a meeting that departed from the script, and any "
             "threatened claim, recorded the same day.", "", True),
        ]
        for tid, title, due, desc, auth, evidence in items:
            self._add(Task(
                id=tid, title=title, category="logistics", due_date=due,
                description=desc, authority=auth, evidence_required=evidence,
            ))

        self._tasks["DAY-NOTIFY"].depends_on = (
            "PREP-PAYROLL", "PREP-PACKETS", "PREP-MANAGERS"
        )
        self._tasks["DAY-DELIVER-PAY"].depends_on = ("PREP-PAYROLL",)

    def _add_document_tasks(self, documents: Any) -> None:
        if documents is None:
            self._add(Task(
                id="DOC-GENERATE", title="Generate separation documents",
                category="document",
                description="Run box 7 once the compliance gate and approval "
                            "chain permit.",
            ))
            return
        if getattr(documents, "blocked", False):
            self._add(Task(
                id="DOC-UNBLOCK",
                title="Resolve blockers preventing document generation",
                category="document",
                description="; ".join(getattr(documents, "blockers", []))[:400],
                evidence_required=True,
            ))
            return

        incomplete = getattr(documents, "incomplete", [])
        if incomplete:
            self._add(Task(
                id="DOC-PLACEHOLDERS",
                title=f"Fill {len(incomplete)} document(s) with open placeholders",
                category="document",
                description="Placeholders are facts the generator would not "
                            "guess. Fill them before counsel review, not after.",
                evidence_required=True,
            ))
        self._add(Task(
            id="DOC-COUNSEL-REVIEW",
            title="Employment counsel reviews every draft document",
            category="document",
            description="Every generated document is a draft. Counsel reviews "
                        "before anything is distributed.",
            depends_on=("DOC-PLACEHOLDERS",) if incomplete else (),
            evidence_required=True,
        ))

    def _add_approval_tasks(self, package: Any) -> None:
        if package is None:
            return
        for code in getattr(package, "blocker_codes", ()):
            self._add(Task(
                id=f"APPR-{code}",
                title=f"Resolve or clear: {code}",
                category="approval",
                description="Open compliance blocker. Data blockers are fixed at "
                            "the source; legal-judgment blockers are cleared by "
                            "recorded counsel sign-off.",
                evidence_required=True,
            ))

    def _add_per_employee(
        self, pay: Any, selection: Any, sep: dt.date | None, report: Any
    ) -> None:
        register = getattr(pay, "register", None)
        cut = getattr(selection, "cut_list", None)
        source = register if register is not None and len(register) else cut
        if source is None or not len(source):
            return

        cfg = self.config
        consider_end = (
            sep + dt.timedelta(days=cfg.consideration_days) if sep else None
        )
        revoke_end = (
            consider_end + dt.timedelta(days=cfg.revocation_days)
            if consider_end else None
        )

        for _, row in source.iterrows():
            emp = row.get("employee_id")
            if pd.isna(emp):
                continue
            emp = str(emp)

            self._add(Task(
                id=f"EMP-{emp}-NOTIFY", title=f"Notify {emp}",
                category="per_employee", due_date=sep, employee_id=emp,
                description="Notification meeting with HR present.",
            ))
            self._add(Task(
                id=f"EMP-{emp}-PACKET",
                title=f"Deliver packet and record acknowledgment — {emp}",
                category="per_employee", due_date=sep, employee_id=emp,
                evidence_required=True,
                depends_on=(f"EMP-{emp}-NOTIFY",),
                description="Separation letter, EDD notice, DE 2320, HIPP "
                            "notice, and the agreement if offered. Record that "
                            "it was received.",
            ))
            self._add(Task(
                id=f"EMP-{emp}-FINALPAY",
                title=f"Confirm final pay delivered — {emp}",
                category="per_employee", due_date=sep, employee_id=emp,
                authority="Lab. Code §§ 201, 203", evidence_required=True,
                immovable=True,
                description="Wages and vested vacation, delivered at separation "
                            "and not conditioned on signing anything.",
            ))
            self._add(Task(
                id=f"EMP-{emp}-COBRA",
                title=f"COBRA election notice issued — {emp}",
                category="per_employee",
                due_date=sep + dt.timedelta(days=44) if sep else None,
                employee_id=emp, authority="29 U.S.C. § 1166",
                evidence_required=True, immovable=True,
            ))

            if consider_end:
                # Deliberately no reminder tasks inside the consideration window.
                self._add(Task(
                    id=f"EMP-{emp}-CONSIDER",
                    title=f"Consideration period ends — {emp}",
                    category="per_employee", due_date=consider_end,
                    employee_id=emp, immovable=True,
                    authority="29 U.S.C. § 626(f)(1)(F)",
                    description=(
                        f"The employee has {cfg.consideration_days} days to "
                        f"decide. No follow-up task is scheduled inside this "
                        f"window by design: prompting for a signature during the "
                        f"statutory consideration period undermines the "
                        f"voluntariness the period exists to protect."
                    ),
                ))
                self._add(Task(
                    id=f"EMP-{emp}-FOLLOWUP",
                    title=f"Follow up on unreturned agreement — {emp}",
                    category="per_employee", due_date=consider_end,
                    not_before=consider_end, employee_id=emp,
                    depends_on=(f"EMP-{emp}-CONSIDER",),
                    description="Only after the consideration period has run. "
                                "Mark not_applicable if already returned.",
                ))
            if revoke_end:
                self._add(Task(
                    id=f"EMP-{emp}-REVOKE",
                    title=f"Revocation period expires; release effective — {emp}",
                    category="per_employee", due_date=revoke_end,
                    not_before=revoke_end, employee_id=emp, immovable=True,
                    authority="29 U.S.C. § 626(f)(1)(G)",
                    evidence_required=True,
                    description=(
                        "The release is not effective until this passes and it "
                        "cannot be waived. Severance must not be processed "
                        "before this date."
                    ),
                ))

            for item in ("notice", "packet", "final_pay"):
                self.acknowledgments.append(Acknowledgment(employee_id=emp, item=item))

    def _check_overdue_at_build(self) -> None:
        today = dt.date.today()
        overdue = [t for t in self._tasks.values() if t.is_overdue(today)]
        statutory = [t for t in overdue if t.immovable]
        if statutory:
            self._note(
                Severity.ERROR, "STATUTORY_DEADLINE_PASSED",
                f"{len(statutory)} statutory deadline(s) are already past: "
                f"{', '.join(t.id for t in statutory[:4])}. These cannot be met "
                f"by working faster — the dates have to move, which means the "
                f"separation date has to move. Escalate to counsel.",
            )
        elif overdue:
            self._note(
                Severity.WARNING, "TASKS_OVERDUE",
                f"{len(overdue)} task(s) are past their due date at the time the "
                f"board was built.",
            )

    # -- access -------------------------------------------------------------
    @property
    def tasks(self) -> tuple[Task, ...]:
        return tuple(self._tasks.values())

    def get(self, task_id: str) -> Task:
        if task_id not in self._tasks:
            raise TaskError(f"Unknown task {task_id!r}.")
        return self._tasks[task_id]

    def overdue(self, today: dt.date | None = None) -> list[Task]:
        today = today or dt.date.today()
        return sorted(
            [t for t in self._tasks.values() if t.is_overdue(today)],
            key=lambda t: t.due_date or dt.date.max,
        )

    def due_soon(self, today: dt.date | None = None) -> list[Task]:
        today = today or dt.date.today()
        return sorted(
            [t for t in self._tasks.values()
             if t.is_due_soon(today, self.config.due_soon_days)],
            key=lambda t: t.due_date or dt.date.max,
        )

    def blocked_tasks(self) -> list[Task]:
        out = []
        for t in self._tasks.values():
            if t.status in ("complete", "not_applicable"):
                continue
            unmet = [
                d for d in t.depends_on
                if d in self._tasks
                and self._tasks[d].status not in ("complete", "not_applicable")
            ]
            if unmet:
                out.append(t)
        return out

    # -- mutation -----------------------------------------------------------
    def complete(
        self, task_id: str, by: str, evidence: str = "", notes: str = "",
        today: dt.date | None = None,
    ) -> Task:
        task = self.get(task_id)
        today = today or dt.date.today()

        if not by.strip():
            raise TaskError("Completing a task requires naming who completed it.")

        if task.evidence_required and not evidence.strip():
            raise TaskError(
                f"{task_id} requires a reference to evidence — a document, a "
                f"payroll run, a signed acknowledgment. A task marked done with "
                f"nothing behind it is worse than one left open, because it "
                f"stops anyone looking."
            )

        if task.not_before and today < task.not_before:
            raise TaskError(
                f"{task_id} cannot be completed before {task.not_before}. The "
                f"waiting period is the substance of this task, not a formality "
                f"around it."
            )

        unmet = [
            d for d in task.depends_on
            if d in self._tasks
            and self._tasks[d].status not in ("complete", "not_applicable")
        ]
        if unmet:
            raise TaskError(
                f"{task_id} depends on {', '.join(unmet)}, which are not "
                f"complete."
            )

        task.status = "complete"
        task.completed_at = dt.datetime.now().isoformat(timespec="seconds")
        task.completed_by = by
        task.evidence_ref = evidence
        if notes:
            task.notes = (task.notes + " " + notes).strip()
        self._log("completed", task_id, by, evidence)
        return task

    def set_status(self, task_id: str, status: str, by: str, notes: str = "") -> Task:
        if status not in STATUSES:
            raise TaskError(f"Unknown status {status!r}.")
        if status == "complete":
            raise TaskError("Use complete() so evidence and dependencies are checked.")
        task = self.get(task_id)
        task.status = status
        if notes:
            task.notes = (task.notes + " " + notes).strip()
        self._log(f"status:{status}", task_id, by, notes)
        return task

    def reschedule(self, task_id: str, new_due: dt.date, by: str, reason: str) -> Task:
        task = self.get(task_id)
        if task.immovable:
            raise TaskError(
                f"{task_id} carries a statutory deadline ({task.authority or 'statute'}) "
                f"and cannot be rescheduled here. Change the underlying dates and "
                f"regenerate the board. A tracker that lets a WARN deadline be "
                f"typed over is worse than a spreadsheet, because it looks "
                f"authoritative."
            )
        if not reason.strip():
            raise TaskError("Rescheduling requires a reason.")
        old = task.due_date
        task.due_date = new_due
        self._log("rescheduled", task_id, by, f"{old} -> {new_due}: {reason}")
        return task

    def record_acknowledgment(
        self, employee_id: str, item: str, by: str, reference: str = "",
    ) -> Acknowledgment:
        for a in self.acknowledgments:
            if a.employee_id == employee_id and a.item == item:
                a.received = True
                a.received_at = dt.datetime.now().isoformat(timespec="seconds")
                a.recorded_by = by
                a.reference = reference
                self._log("acknowledged", f"{employee_id}:{item}", by, reference)
                return a
        ack = Acknowledgment(
            employee_id=employee_id, item=item, received=True,
            received_at=dt.datetime.now().isoformat(timespec="seconds"),
            recorded_by=by, reference=reference,
        )
        self.acknowledgments.append(ack)
        self._log("acknowledged", f"{employee_id}:{item}", by, reference)
        return ack

    def _log(self, action: str, target: str, actor: str, detail: str = "") -> None:
        self.log.append({
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "action": action, "target": target, "actor": actor, "detail": detail,
        })

    # -- staleness ----------------------------------------------------------
    def check_version(self, package: Any) -> tuple[bool, str]:
        """Confirm the board still matches the approved plan."""
        fp = getattr(package, "fingerprint", "")
        if not self.fingerprint or not fp:
            return True, ""
        if fp != self.fingerprint:
            return False, (
                f"This board was built against version {self.fingerprint} but "
                f"the current plan is {fp}. Task assignments, dates, and the "
                f"per-employee track may no longer match who is actually "
                f"affected. Rebuild the board."
            )
        return True, ""

    # -- summary ------------------------------------------------------------
    def summary(self, today: dt.date | None = None) -> dict[str, Any]:
        today = today or dt.date.today()
        by_status: dict[str, int] = {}
        for t in self._tasks.values():
            by_status[t.status] = by_status.get(t.status, 0) + 1
        acks = len(self.acknowledgments)
        received = sum(1 for a in self.acknowledgments if a.received)
        done = by_status.get("complete", 0) + by_status.get("not_applicable", 0)
        return {
            "scenario": self.scenario,
            "fingerprint": self.fingerprint,
            "separation_date": (
                self.separation_date.isoformat() if self.separation_date else None
            ),
            "total_tasks": len(self._tasks),
            "by_status": by_status,
            "percent_complete": round(100 * done / len(self._tasks), 1) if self._tasks else 0.0,
            "overdue": len(self.overdue(today)),
            "overdue_statutory": len([t for t in self.overdue(today) if t.immovable]),
            "due_soon": len(self.due_soon(today)),
            "blocked": len(self.blocked_tasks()),
            "acknowledgments_received": received,
            "acknowledgments_total": acks,
        }

    # -- output -------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self._tasks:
            return pd.DataFrame(columns=list(Task("", "", "").to_dict()))
        df = pd.DataFrame([t.to_dict() for t in self._tasks.values()])
        cat_order = {c: i for i, c in enumerate(CATEGORIES)}
        df["_c"] = df["category"].map(cat_order).fillna(9)
        df["_d"] = pd.to_datetime(df["due_date"], errors="coerce")
        return df.sort_values(["_c", "_d", "id"]).drop(columns=["_c", "_d"]).reset_index(drop=True)

    def acknowledgments_dataframe(self) -> pd.DataFrame:
        if not self.acknowledgments:
            return pd.DataFrame(
                columns=["employee_id", "item", "received", "received_at",
                         "recorded_by", "reference"]
            )
        return pd.DataFrame([a.to_dict() for a in self.acknowledgments])

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "tasks": [t.to_dict() for t in self._tasks.values()],
            "acknowledgments": [a.to_dict() for a in self.acknowledgments],
            "log": self.log,
            "findings": self.findings,
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    def to_markdown(self, today: dt.date | None = None) -> str:
        today = today or dt.date.today()
        s = self.summary(today)
        L: list[str] = []
        L.append("# Execution Task Board")
        L.append("")
        L.append(
            "> **Privileged and confidential — prepared at the direction of "
            "counsel.**"
        )
        L.append("")
        L.append(f"**Scenario:** {s['scenario'] or '(unnamed)'}  ")
        L.append(f"**Plan version:** `{s['fingerprint'] or '(unbound)'}`  ")
        L.append(f"**Separation date:** {s['separation_date']}  ")
        L.append(f"**Progress:** {s['percent_complete']}% "
                 f"({s['by_status'].get('complete', 0)} of {s['total_tasks']})  ")
        L.append("")

        if s["overdue_statutory"]:
            L.append(f"## {s['overdue_statutory']} statutory deadline(s) have passed")
            L.append("")
            L.append(
                "These cannot be met by working faster. The dates move only if "
                "the separation date moves. Escalate to counsel before doing "
                "anything else."
            )
            L.append("")
            for t in self.overdue(today):
                if t.immovable:
                    L.append(f"- **{t.due_date}** — {t.title} *({t.authority})*")
            L.append("")

        overdue = [t for t in self.overdue(today) if not t.immovable]
        if overdue:
            L.append(f"## Overdue ({len(overdue)})")
            L.append("")
            for t in overdue:
                L.append(f"- **{t.due_date}** — {t.title} — {t.owner}")
            L.append("")

        soon = self.due_soon(today)
        if soon:
            L.append(f"## Due in the next {self.config.due_soon_days} days ({len(soon)})")
            L.append("")
            for t in soon:
                L.append(f"- **{t.due_date}** — {t.title} — {t.owner}")
            L.append("")

        blocked = self.blocked_tasks()
        if blocked:
            L.append(f"## Blocked by dependencies ({len(blocked)})")
            L.append("")
            for t in blocked:
                unmet = [d for d in t.depends_on
                         if self._tasks.get(d) and
                         self._tasks[d].status not in ("complete", "not_applicable")]
                L.append(f"- {t.title} — waiting on {', '.join(unmet)}")
            L.append("")

        df = self.to_dataframe()
        for cat in CATEGORIES:
            sub = df.loc[df["category"] == cat]
            if sub.empty:
                continue
            label = cat.replace("_", " ").title()
            L.append(f"## {label} ({len(sub)})")
            L.append("")
            if cat == "per_employee" and len(sub) > 20:
                done = int((sub["status"] == "complete").sum())
                L.append(f"{len(sub)} per-employee task(s), {done} complete. "
                         f"See the CSV for the full list.")
                L.append("")
                continue
            L.append("| Due | Task | Owner | Status | Authority |")
            L.append("|---|---|---|---|---|")
            for _, r in sub.iterrows():
                due = r["due_date"] or "—"
                lock = " 🔒" if r["immovable"] else ""
                L.append(
                    f"| {due}{lock} | {r['title']} | {r['owner']} | "
                    f"{r['status']} | {r['authority'] or '—'} |"
                )
            L.append("")

        if self.acknowledgments:
            L.append("## Acknowledgments")
            L.append("")
            L.append(
                f"{s['acknowledgments_received']} of {s['acknowledgments_total']} "
                f"recorded. These are what the employer relies on to prove "
                f"delivery."
            )
            L.append("")

        if self.findings:
            L.append("## Findings")
            L.append("")
            for f in self.findings:
                L.append(f"- **[{f['severity']}] {f['code']}** — {f['message']}")
            L.append("")

        L.append("---")
        L.append(
            "_🔒 marks a statutory deadline, which cannot be rescheduled here. "
            "No follow-up task is scheduled inside an OWBPA consideration "
            "period: prompting for a signature during the window the statute "
            "provides for deliberation undermines the voluntariness it exists "
            "to protect._"
        )
        return "\n".join(L)

    def write(self, outdir: str | Path, stem: str = "tasks") -> dict[str, Path]:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        paths = {
            "board_md": outdir / f"{stem}_board.md",
            "tasks": outdir / f"{stem}.csv",
            "acknowledgments": outdir / f"{stem}_acknowledgments.csv",
            "log": outdir / f"{stem}_activity_log.csv",
            "json": outdir / f"{stem}.json",
        }
        self.to_dataframe().to_csv(paths["tasks"], index=False)
        self.acknowledgments_dataframe().to_csv(paths["acknowledgments"], index=False)
        pd.DataFrame(self.log or [], columns=["timestamp", "action", "target",
                                              "actor", "detail"]).to_csv(
            paths["log"], index=False
        )
        self.to_json(paths["json"])
        paths["board_md"].write_text(self.to_markdown(), encoding="utf-8")
        return paths
