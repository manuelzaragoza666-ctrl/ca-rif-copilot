"""
approvals.py
============

Workflow & Approvals for the California RIF Copilot (box 8).

Structured sign-off — HR, then Legal, then Executive — before a reduction can be
executed, with each approval bound to the specific state of the plan it was
given for.

Why binding matters
-------------------
An approval that is not bound to what was approved is decoration. If Legal signs
off on a 17-person cut list and someone then adds two people, changes a
weighting, or moves the separation date, the signature is still sitting there
looking valid. The most common way approval workflows fail is not that people
skip them — it is that the thing approved and the thing executed drift apart and
nobody notices.

So every submission is fingerprinted over the decision-relevant content: who is
on the cut list, what they scored, what they are being paid, what compliance
found, and the key plan parameters. Approvals record that fingerprint. If the
underlying plan changes, the fingerprint changes, prior approvals are marked
superseded, and the chain has to be walked again. Nothing silently carries over.

The ledger is append-only. Revoking an approval writes a revocation record; it
does not delete the approval. The history of who approved what, when, and on the
basis of which version is the artifact that matters if this is ever examined.

What a stage can clear
----------------------
Compliance blockers from box 5 are not all the same. Legal can clear an adverse
impact finding, because assessing it is legal work. Nobody can clear a missing
pay rate by signing a form. Each stage declares which blocker codes it is
competent to clear, and the data-completeness blockers appear in no stage's
list.

Usage
-----
    from .approvals import ApprovalLedger, ApprovalPackage, default_policy

    package = ApprovalPackage.from_pipeline(
        plan=plan, selection=selection, impact=impact,
        compliance=compliance, pay=pay, scenario="Scenario A",
    )
    ledger = ApprovalLedger(policy=default_policy())
    ledger.submit(package, submitted_by="M. Chen", role="HR Business Partner")

    ledger.approve("hr", "M. Chen", "HR Business Partner", comment="Reviewed.")
    ledger.approve("legal", "R. Alvarez", "Employment Counsel",
                   clears=["ADVERSE_IMPACT_INDICATED"],
                   comment="Criteria confirmed job-related.")
    ledger.approve("executive", "J. Park", "CFO")

    ledger.is_fully_approved(package)
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .workforce_data import Severity

__all__ = [
    "ApprovalStage",
    "ApprovalPolicy",
    "ApprovalPackage",
    "ApprovalRecord",
    "ApprovalLedger",
    "ApprovalStatus",
    "default_policy",
    "ApprovalError",
]

__version__ = "1.0.0"


class ApprovalError(ValueError):
    """Raised when an approval action is not permitted."""


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalStage:
    key: str
    name: str
    #: Roles competent to sign at this stage. Empty means any role.
    required_roles: tuple[str, ...] = ()
    #: How many distinct approvers this stage needs.
    min_approvers: int = 1
    #: Compliance blocker codes this stage may clear. A stage cannot clear a
    #: blocker outside its competence, and no stage can clear a data blocker.
    can_clear: tuple[str, ...] = ()
    description: str = ""


#: Blockers that no stage may clear by signing. These mean a field is blank or
#: uncomputable; a signature does not fill it in.
UNCLEARABLE_CODES: frozenset[str] = frozenset({
    "NO_ESTABLISHMENT_COLUMN", "NO_ESTABLISHMENTS", "FINAL_PAY_UNCOMPUTABLE",
    "NO_PAY_DATA", "NO_TENURE_DATA", "INCOMPLETE_REGISTER",
    "LEAVE_POLICY_UNDECLARED", "SB617_COORDINATION_UNDECLARED",
    "SB617_LWDB_CONTACT_MISSING", "SB617_EMPLOYER_CONTACT_MISSING",
    "NO_SELECTION", "SEPARATION_BEFORE_NOTICE",
})


@dataclass
class ApprovalPolicy:
    stages: tuple[ApprovalStage, ...]
    #: Approvals older than this are stale: facts move, and a sign-off given
    #: against a two-month-old picture is not a sign-off against today's.
    validity_days: int = 30
    #: Whether the person who submitted the package may also approve it.
    allow_self_approval: bool = False
    #: Whether one person may sign more than one stage.
    allow_same_person_multiple_stages: bool = False

    def stage(self, key: str) -> ApprovalStage | None:
        for s in self.stages:
            if s.key == key:
                return s
        return None

    def order(self, key: str) -> int:
        for i, s in enumerate(self.stages):
            if s.key == key:
                return i
        return -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [
                {
                    "key": s.key, "name": s.name,
                    "required_roles": list(s.required_roles),
                    "min_approvers": s.min_approvers,
                    "can_clear": list(s.can_clear),
                }
                for s in self.stages
            ],
            "validity_days": self.validity_days,
            "allow_self_approval": self.allow_self_approval,
            "allow_same_person_multiple_stages": self.allow_same_person_multiple_stages,
        }


def default_policy() -> ApprovalPolicy:
    """HR → Legal → Executive, matching the architecture diagram."""
    return ApprovalPolicy(
        stages=(
            ApprovalStage(
                key="hr", name="HR review",
                required_roles=("HR Business Partner", "HR Director", "CHRO"),
                can_clear=("REVIEW_QUEUE_NOT_EMPTY", "TIE_AT_CUT_BOUNDARY"),
                description=(
                    "Confirms the selection was applied as documented, the review "
                    "queue is resolved, and the data behind it is sound."
                ),
            ),
            ApprovalStage(
                key="legal", name="Legal review",
                required_roles=("Employment Counsel", "General Counsel",
                                "Outside Counsel"),
                can_clear=(
                    "ADVERSE_IMPACT_INDICATED", "SELECTED_ON_PROTECTED_LEAVE",
                    "SELECTED_UNION_MEMBERS", "SELECTED_VISA_HOLDERS",
                    "SELECTION_ERRORS", "WARN_NOTICE_DATE_PASSED",
                    "NEAR_WARN_THRESHOLD", "OWBPA_DECISIONAL_UNIT_REQUIRED",
                ),
                description=(
                    "Assesses adverse impact, protected-status exposure, WARN "
                    "obligations, and release enforceability. The only stage "
                    "that can clear a legal-judgment blocker."
                ),
            ),
            ApprovalStage(
                key="executive", name="Executive authorization",
                required_roles=("CEO", "CFO", "COO", "President",
                                "Business Unit Leader"),
                description=(
                    "Authorizes execution. Cannot clear compliance blockers — "
                    "business urgency is not a legal determination."
                ),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# The package being approved
# ---------------------------------------------------------------------------


@dataclass
class ApprovalPackage:
    """A snapshot of everything an approver is signing off on."""

    scenario: str
    fingerprint: str
    created_at: str
    summary: dict[str, Any]
    open_blockers: tuple[str, ...] = ()
    blocker_codes: tuple[str, ...] = ()
    content: dict[str, Any] = field(default_factory=dict)

    # -- construction ----------------------------------------------------
    @classmethod
    def from_pipeline(
        cls,
        scenario: str = "",
        plan: Any = None,
        selection: Any = None,
        impact: Any = None,
        compliance: Any = None,
        pay: Any = None,
    ) -> "ApprovalPackage":
        content: dict[str, Any] = {"scenario": scenario}

        # Who is affected, and on what basis. Sorted so row order cannot change
        # the fingerprint while the substance stays the same.
        cut = getattr(selection, "cut_list", None)
        if cut is not None and len(cut):
            people = []
            for _, r in cut.iterrows():
                people.append({
                    "employee_id": str(r.get("employee_id")),
                    "score": _round_or_none(r.get("retention_score")),
                    "job_title": _str_or_none(r.get("job_title")),
                    "department": _str_or_none(r.get("department")),
                })
            content["cut_list"] = sorted(people, key=lambda p: p["employee_id"] or "")

        if plan is not None:
            content["plan"] = {
                "cost_savings_target": getattr(plan, "cost_savings_target", None),
                "burden_multiplier": getattr(plan, "burden_multiplier", None),
                "min_comparison_group_size": getattr(
                    plan, "min_comparison_group_size", None
                ),
                "departments": sorted(
                    [
                        {
                            "department": d,
                            "mode": dp.mode,
                            "criteria": sorted(
                                [
                                    {"name": c.name, "weight": round(c.weight, 6),
                                     "column": c.source_column}
                                    for c in dp.normalized_criteria
                                ],
                                key=lambda c: c["name"],
                            ),
                        }
                        for d, dp in getattr(plan, "departments", {}).items()
                    ],
                    key=lambda d: d["department"],
                ),
            }

        if impact is not None:
            rep = getattr(impact, "report", impact)
            content["impact"] = {
                "verdicts": dict(sorted(rep.class_verdicts().items())),
                "indicated": sorted(
                    f"{c.protected_class}:{c.unit}" for c in rep.indicated
                ),
            }

        blockers: tuple[str, ...] = ()
        codes: tuple[str, ...] = ()
        if compliance is not None:
            rep = getattr(compliance, "report", compliance)
            gate = getattr(compliance, "gate", getattr(rep, "gate", None))
            blockers = tuple(getattr(gate, "blockers", ()))
            codes = tuple(sorted({
                f.code for f in getattr(rep, "findings", [])
                if getattr(f, "severity", None) == Severity.ERROR
            }))
            content["compliance"] = {
                "separation_date": getattr(rep, "separation_date", None),
                "notice_date": getattr(rep, "notice_date", None),
                "warn_triggered": getattr(rep, "warn_triggered", None),
                "blocker_codes": list(codes),
            }

        if pay is not None:
            rep = getattr(pay, "report", pay)
            totals = getattr(rep, "totals", {})
            content["pay"] = {
                k: totals.get(k)
                for k in ("severance_gross", "vacation_payout",
                          "total_employer_cost", "median_weeks")
            }

        fingerprint = _fingerprint(content)
        summary = {
            "scenario": scenario,
            "affected": len(content.get("cut_list", [])),
            "separation_date": content.get("compliance", {}).get("separation_date"),
            "warn_triggered": content.get("compliance", {}).get("warn_triggered"),
            "total_employer_cost": content.get("pay", {}).get("total_employer_cost"),
            "impact_verdicts": content.get("impact", {}).get("verdicts", {}),
            "open_blockers": len(blockers),
        }
        return cls(
            scenario=scenario,
            fingerprint=fingerprint,
            created_at=dt.datetime.now().isoformat(timespec="seconds"),
            summary=summary,
            open_blockers=blockers,
            blocker_codes=codes,
            content=content,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "fingerprint": self.fingerprint,
            "created_at": self.created_at,
            "summary": self.summary,
            "open_blockers": list(self.open_blockers),
            "blocker_codes": list(self.blocker_codes),
        }

    def diff(self, other: "ApprovalPackage") -> list[str]:
        """Human-readable description of what changed between two packages."""
        changes: list[str] = []
        a, b = other.content, self.content

        old_ids = {p["employee_id"] for p in a.get("cut_list", [])}
        new_ids = {p["employee_id"] for p in b.get("cut_list", [])}
        added, removed = new_ids - old_ids, old_ids - new_ids
        if added:
            changes.append(f"{len(added)} employee(s) added to the cut list: "
                           f"{', '.join(sorted(added)[:6])}")
        if removed:
            changes.append(f"{len(removed)} employee(s) removed from the cut list: "
                           f"{', '.join(sorted(removed)[:6])}")

        if a.get("plan") != b.get("plan"):
            changes.append("Selection plan parameters changed (target, weights, "
                           "or department configuration)")
        if a.get("compliance", {}).get("separation_date") != b.get(
            "compliance", {}
        ).get("separation_date"):
            changes.append(
                f"Separation date changed from "
                f"{a.get('compliance', {}).get('separation_date')} to "
                f"{b.get('compliance', {}).get('separation_date')}"
            )
        if a.get("impact", {}).get("verdicts") != b.get("impact", {}).get("verdicts"):
            changes.append("Adverse impact verdicts changed")
        if a.get("pay", {}).get("total_employer_cost") != b.get("pay", {}).get(
            "total_employer_cost"
        ):
            changes.append("Total employer cost changed")
        if not changes:
            changes.append("Content differs but no tracked field explains it")
        return changes


def _round_or_none(v: Any) -> float | None:
    return None if v is None or pd.isna(v) else round(float(v), 4)


def _str_or_none(v: Any) -> str | None:
    return None if v is None or pd.isna(v) else str(v)


def _fingerprint(content: dict[str, Any]) -> str:
    blob = json.dumps(content, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApprovalRecord:
    action: str                 # submitted | approved | rejected | revoked | superseded
    stage: str | None
    actor: str
    role: str
    timestamp: str
    fingerprint: str
    comment: str = ""
    clears: tuple[str, ...] = ()
    conditions: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action, "stage": self.stage, "actor": self.actor,
            "role": self.role, "timestamp": self.timestamp,
            "fingerprint": self.fingerprint, "comment": self.comment,
            "clears": list(self.clears), "conditions": self.conditions,
        }


@dataclass
class ApprovalStatus:
    fingerprint: str
    stages: dict[str, dict[str, Any]]
    complete: bool
    blocked_reason: str = ""
    next_stage: str | None = None
    cleared_codes: tuple[str, ...] = ()
    uncleared_codes: tuple[str, ...] = ()
    stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "fingerprint": self.fingerprint, "stages": self.stages,
            "complete": self.complete, "blocked_reason": self.blocked_reason,
            "next_stage": self.next_stage,
            "cleared_codes": list(self.cleared_codes),
            "uncleared_codes": list(self.uncleared_codes),
            "stale": self.stale,
        }


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class ApprovalLedger:
    """Append-only record of submissions and approval decisions."""

    def __init__(self, policy: ApprovalPolicy | None = None) -> None:
        self.policy = policy or default_policy()
        self._records: list[ApprovalRecord] = []
        self._packages: dict[str, ApprovalPackage] = {}
        self._current: str | None = None

    # -- history -----------------------------------------------------------
    @property
    def records(self) -> tuple[ApprovalRecord, ...]:
        return tuple(self._records)

    def _append(self, record: ApprovalRecord) -> ApprovalRecord:
        self._records.append(record)
        return record

    # -- submission --------------------------------------------------------
    def submit(
        self, package: ApprovalPackage, submitted_by: str, role: str = "",
        comment: str = "",
    ) -> ApprovalRecord:
        """Register a package for approval.

        If a different package was already in flight, its approvals are marked
        superseded rather than deleted — the history of what was approved and
        then changed is the point.
        """
        if not submitted_by.strip():
            raise ApprovalError("A submission must name who submitted it.")

        if self._current and self._current != package.fingerprint:
            prior = self._packages.get(self._current)
            changes = package.diff(prior) if prior else ["content changed"]
            self._append(ApprovalRecord(
                action="superseded", stage=None, actor=submitted_by, role=role,
                timestamp=_now(), fingerprint=self._current,
                comment=(
                    f"Superseded by {package.fingerprint}. "
                    + "; ".join(changes)
                ),
            ))

        self._packages[package.fingerprint] = package
        self._current = package.fingerprint
        return self._append(ApprovalRecord(
            action="submitted", stage=None, actor=submitted_by, role=role,
            timestamp=_now(), fingerprint=package.fingerprint, comment=comment,
        ))

    # -- decisions ---------------------------------------------------------
    def approve(
        self, stage: str, actor: str, role: str = "", comment: str = "",
        clears: Sequence[str] = (), conditions: str = "",
    ) -> ApprovalRecord:
        self._validate_actor(stage, actor, role)
        st = self._require_stage(stage)
        self._require_prior_stages(stage)

        clears = tuple(clears)
        illegal = [c for c in clears if c in UNCLEARABLE_CODES]
        if illegal:
            raise ApprovalError(
                f"{', '.join(illegal)} cannot be cleared by approval. These "
                f"indicate missing or uncomputable data — signing a form does "
                f"not supply a pay rate. Fix the underlying data and resubmit."
            )
        outside = [c for c in clears if c not in st.can_clear]
        if outside:
            raise ApprovalError(
                f"The {st.name} stage is not competent to clear "
                f"{', '.join(outside)}. Stages may only clear blockers within "
                f"their remit; check whether Legal review is the right stage."
            )
        if clears and not comment.strip():
            raise ApprovalError(
                "Clearing a compliance blocker requires a written basis. An "
                "unexplained clearance is the record a plaintiff will read aloud."
            )

        return self._append(ApprovalRecord(
            action="approved", stage=stage, actor=actor, role=role,
            timestamp=_now(), fingerprint=self._require_current(),
            comment=comment, clears=clears, conditions=conditions,
        ))

    def reject(
        self, stage: str, actor: str, role: str = "", comment: str = "",
    ) -> ApprovalRecord:
        self._validate_actor(stage, actor, role, check_self=False)
        self._require_stage(stage)
        if not comment.strip():
            raise ApprovalError("A rejection must state its reason.")
        return self._append(ApprovalRecord(
            action="rejected", stage=stage, actor=actor, role=role,
            timestamp=_now(), fingerprint=self._require_current(), comment=comment,
        ))

    def revoke(
        self, stage: str, actor: str, role: str = "", comment: str = "",
    ) -> ApprovalRecord:
        """Withdraw a previously given approval. Appends; never deletes."""
        if not comment.strip():
            raise ApprovalError("A revocation must state its reason.")
        return self._append(ApprovalRecord(
            action="revoked", stage=stage, actor=actor, role=role,
            timestamp=_now(), fingerprint=self._require_current(), comment=comment,
        ))

    # -- validation --------------------------------------------------------
    def _require_current(self) -> str:
        if not self._current:
            raise ApprovalError("Nothing has been submitted for approval yet.")
        return self._current

    def _require_stage(self, stage: str) -> ApprovalStage:
        st = self.policy.stage(stage)
        if st is None:
            raise ApprovalError(f"Unknown approval stage {stage!r}.")
        return st

    def _require_prior_stages(self, stage: str) -> None:
        idx = self.policy.order(stage)
        status = self.status()
        for earlier in self.policy.stages[:idx]:
            if not status.stages[earlier.key]["approved"]:
                raise ApprovalError(
                    f"{self.policy.stage(stage).name} cannot be given before "
                    f"{earlier.name} is complete. The order exists so that "
                    f"executive authorization is not the thing that surfaces a "
                    f"legal problem."
                )

    def _validate_actor(
        self, stage: str, actor: str, role: str, check_self: bool = True
    ) -> None:
        if not actor.strip():
            raise ApprovalError("An approval must name the approver.")
        st = self._require_stage(stage)
        fp = self._require_current()

        if st.required_roles and role not in st.required_roles:
            raise ApprovalError(
                f"Role {role!r} is not authorized for {st.name}. Expected one "
                f"of: {', '.join(st.required_roles)}."
            )

        if check_self and not self.policy.allow_self_approval:
            submitters = {
                r.actor for r in self._records
                if r.action == "submitted" and r.fingerprint == fp
            }
            if actor in submitters:
                raise ApprovalError(
                    f"{actor} submitted this package and cannot also approve it. "
                    f"Self-approval defeats the purpose of the chain."
                )

        if not self.policy.allow_same_person_multiple_stages:
            for r in self._records:
                if (
                    r.action == "approved" and r.fingerprint == fp
                    and r.actor == actor and r.stage != stage
                ):
                    raise ApprovalError(
                        f"{actor} already approved at the {r.stage} stage. One "
                        f"person signing multiple stages is not independent "
                        f"review."
                    )

    # -- status ------------------------------------------------------------
    def status(self, package: ApprovalPackage | None = None) -> ApprovalStatus:
        fp = package.fingerprint if package else (self._current or "")
        now = dt.datetime.now()

        stages: dict[str, dict[str, Any]] = {}
        cleared: set[str] = set()
        stale = False

        for st in self.policy.stages:
            approvals: list[ApprovalRecord] = []
            rejected: ApprovalRecord | None = None
            for r in self._records:
                if r.fingerprint != fp or r.stage != st.key:
                    continue
                if r.action == "approved":
                    approvals.append(r)
                elif r.action == "rejected":
                    rejected = r
                elif r.action == "revoked":
                    approvals = [a for a in approvals if a.actor != r.actor]

            distinct = {a.actor for a in approvals}
            expired = []
            for a in approvals:
                age = (now - dt.datetime.fromisoformat(a.timestamp)).days
                if age > self.policy.validity_days:
                    expired.append(a.actor)
                    stale = True

            approved = (
                len(distinct) >= st.min_approvers
                and rejected is None
                and not expired
            )
            if approved:
                for a in approvals:
                    cleared.update(a.clears)

            stages[st.key] = {
                "name": st.name,
                "approved": approved,
                "approvers": sorted(distinct),
                "required": st.min_approvers,
                "rejected_by": rejected.actor if rejected else None,
                "rejection_reason": rejected.comment if rejected else "",
                "expired_approvers": expired,
                "conditions": [a.conditions for a in approvals if a.conditions],
            }

        complete = all(s["approved"] for s in stages.values()) and bool(fp)
        next_stage = next(
            (st.key for st in self.policy.stages if not stages[st.key]["approved"]),
            None,
        )

        pkg = package or self._packages.get(fp)
        open_codes = set(pkg.blocker_codes) if pkg else set()
        uncleared = sorted(open_codes - cleared)

        blocked = ""
        if not fp:
            blocked = "Nothing submitted."
        elif any(s["rejected_by"] for s in stages.values()):
            rej = next(s for s in stages.values() if s["rejected_by"])
            blocked = f"Rejected by {rej['rejected_by']}: {rej['rejection_reason']}"
        elif stale:
            blocked = (
                f"One or more approvals are older than "
                f"{self.policy.validity_days} days and must be renewed."
            )
        elif uncleared:
            blocked = (
                f"{len(uncleared)} compliance blocker(s) remain uncleared: "
                f"{', '.join(uncleared)}"
            )
        elif not complete:
            blocked = f"Awaiting {stages[next_stage]['name']}." if next_stage else ""

        return ApprovalStatus(
            fingerprint=fp, stages=stages,
            complete=complete and not uncleared and not stale,
            blocked_reason=blocked, next_stage=next_stage,
            cleared_codes=tuple(sorted(cleared)), uncleared_codes=tuple(uncleared),
            stale=stale,
        )

    def is_fully_approved(self, package: ApprovalPackage | None = None) -> bool:
        if package is not None and package.fingerprint != self._current:
            return False
        return self.status(package).complete

    def verify(self, package: ApprovalPackage) -> tuple[bool, list[str]]:
        """Check a package against the ledger. Returns (ok, problems)."""
        problems: list[str] = []
        if not self._current:
            return False, ["Nothing has been submitted for approval."]
        if package.fingerprint != self._current:
            prior = self._packages.get(self._current)
            changes = package.diff(prior) if prior else ["content changed"]
            problems.append(
                f"The plan has changed since it was submitted for approval "
                f"(approved {self._current}, current {package.fingerprint}). "
                + "; ".join(changes)
                + ". Prior approvals do not carry over; resubmit and re-approve."
            )
            return False, problems
        status = self.status(package)
        if not status.complete:
            problems.append(status.blocked_reason or "Approval chain incomplete.")
        return not problems, problems

    # -- clearance for downstream boxes --------------------------------------
    def clearance(self, package: ApprovalPackage | None = None) -> dict[str, Any]:
        """What box 7 needs: whether to generate, and on whose authority."""
        status = self.status(package)
        legal = [
            r for r in self._records
            if r.action == "approved" and r.stage == "legal"
            and r.fingerprint == status.fingerprint
        ]
        return {
            "approved": status.complete,
            "fingerprint": status.fingerprint,
            "cleared_codes": list(status.cleared_codes),
            "legal_approver": legal[-1].actor if legal else "",
            "legal_basis": legal[-1].comment if legal else "",
            "legal_date": legal[-1].timestamp[:10] if legal else "",
            "blocked_reason": status.blocked_reason,
        }

    # -- output -------------------------------------------------------------
    def to_dataframe(self) -> pd.DataFrame:
        if not self._records:
            return pd.DataFrame(
                columns=["timestamp", "action", "stage", "actor", "role",
                         "fingerprint", "clears", "conditions", "comment"]
            )
        rows = []
        for r in self._records:
            d = r.to_dict()
            d["clears"] = "|".join(d["clears"])
            rows.append(d)
        df = pd.DataFrame(rows)
        return df[["timestamp", "action", "stage", "actor", "role", "fingerprint",
                   "clears", "conditions", "comment"]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.to_dict(),
            "current_fingerprint": self._current,
            "packages": {k: v.to_dict() for k, v in self._packages.items()},
            "records": [r.to_dict() for r in self._records],
            "status": self.status().to_dict(),
        }

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
        return payload

    @classmethod
    def from_json(cls, path: str | Path) -> "ApprovalLedger":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        policy = default_policy()
        ledger = cls(policy=policy)
        for r in raw.get("records", []):
            ledger._records.append(ApprovalRecord(
                action=r["action"], stage=r.get("stage"), actor=r["actor"],
                role=r.get("role", ""), timestamp=r["timestamp"],
                fingerprint=r["fingerprint"], comment=r.get("comment", ""),
                clears=tuple(r.get("clears", ())),
                conditions=r.get("conditions", ""),
            ))
        for fp, p in raw.get("packages", {}).items():
            ledger._packages[fp] = ApprovalPackage(
                scenario=p.get("scenario", ""), fingerprint=fp,
                created_at=p.get("created_at", ""), summary=p.get("summary", {}),
                open_blockers=tuple(p.get("open_blockers", ())),
                blocker_codes=tuple(p.get("blocker_codes", ())),
            )
        ledger._current = raw.get("current_fingerprint")
        return ledger

    def to_markdown(self) -> str:
        status = self.status()
        L: list[str] = []
        L.append("# Approval Record")
        L.append("")
        L.append(
            "> **Privileged and confidential — prepared at the direction of "
            "counsel.**"
        )
        L.append("")
        pkg = self._packages.get(status.fingerprint)
        if pkg:
            L.append(f"**Scenario:** {pkg.scenario or '(unnamed)'}  ")
            L.append(f"**Version fingerprint:** `{status.fingerprint}`  ")
            L.append(f"**Affected employees:** {pkg.summary.get('affected')}  ")
            L.append(f"**Separation date:** {pkg.summary.get('separation_date')}  ")
            L.append("")

        L.append(f"## Status: {'APPROVED' if status.complete else 'NOT APPROVED'}")
        L.append("")
        if status.blocked_reason:
            L.append(status.blocked_reason)
            L.append("")

        L.append("| Stage | Status | Approvers | Cleared |")
        L.append("|---|---|---|---|")
        for st in self.policy.stages:
            s = status.stages[st.key]
            mark = "approved" if s["approved"] else (
                f"rejected by {s['rejected_by']}" if s["rejected_by"] else "pending"
            )
            cleared = [
                c for r in self._records
                if r.stage == st.key and r.action == "approved"
                and r.fingerprint == status.fingerprint
                for c in r.clears
            ]
            L.append(
                f"| {s['name']} | {mark} | {', '.join(s['approvers']) or '—'} | "
                f"{', '.join(cleared) or '—'} |"
            )
        L.append("")

        if status.uncleared_codes:
            L.append("### Uncleared compliance blockers")
            L.append("")
            for c in status.uncleared_codes:
                note = (
                    " — cannot be cleared by approval; fix the underlying data"
                    if c in UNCLEARABLE_CODES else ""
                )
                L.append(f"- `{c}`{note}")
            L.append("")

        L.append("## History")
        L.append("")
        L.append("| When | Action | Stage | Actor | Fingerprint | Note |")
        L.append("|---|---|---|---|---|---|")
        for r in self._records:
            note = r.comment[:80] + ("…" if len(r.comment) > 80 else "")
            L.append(
                f"| {r.timestamp} | {r.action} | {r.stage or '—'} | {r.actor} | "
                f"`{r.fingerprint}` | {note} |"
            )
        L.append("")
        L.append("---")
        L.append(
            "_Approvals are bound to a content fingerprint. If the cut list, "
            "plan parameters, dates, or costs change, the fingerprint changes "
            "and prior approvals are superseded rather than carried forward. "
            "This ledger is append-only; revocations are recorded, not erased._"
        )
        return "\n".join(L)


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")
