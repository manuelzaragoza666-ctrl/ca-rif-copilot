"""
California RIF Copilot
======================

A modular system for planning a reduction in force under California law.

Modules map to the boxes in the system architecture:

    workforce_data      box 1   ingest, validate, normalize
    scenario_simulator  box 2   compare scenarios
    selection_criteria  box 3   score employees, build a cut list
    adverse_impact      box 4   four-fifths, Fisher's exact, per unit
    ca_compliance       box 5   Cal-WARN, SB 617, final pay, OWBPA; the gate
    severance_pay       box 6   severance, vacation payout, withholding
    document_generator  box 7   notices, letters, disclosures — all drafts
    approvals           box 8   HR -> Legal -> Exec, bound to a fingerprint
    task_tracker        box 9   dated tasks and acknowledgments
    audit_reporting     box 10  hash-chained decision record
    pipeline            --      orchestrator

Typical use runs the whole chain::

    from rif_copilot import PipelineConfig, run_pipeline

    result = run_pipeline(PipelineConfig(
        roster_csv="roster.csv",
        plan_yaml="rif_plan.yaml",
        separation_date="2026-10-30",
        leave_policy="separate",
    ))
    print(result.summary_markdown())

Individual boxes can also be used directly; see the module docstrings.

This is screening and drafting support. It is not legal, tax, or payroll
advice. Every determination is an input to a lawyer's analysis, and statutes
change.
"""

from __future__ import annotations

__version__ = "1.0.0"

from .adverse_impact import AdverseImpactAnalyzer, AdverseImpactResult
from .approvals import (
    ApprovalError,
    ApprovalLedger,
    ApprovalPackage,
    ApprovalPolicy,
    default_policy,
)
from .audit_reporting import AuditLog, AuditPackage, RetentionPolicy
from .ca_compliance import ComplianceConfig, ComplianceEngine, ComplianceResult
from .document_generator import DocumentConfig, DocumentGenerator, DocumentSet
from .pipeline import PipelineConfig, PipelineResult, run_pipeline
from .scenario_simulator import (
    CostAssumptions,
    Scenario,
    ScenarioSimulator,
    load_scenarios,
)
from .selection_criteria import (
    RifPlan,
    SelectionConfigError,
    SelectionEngine,
    SelectionResult,
    load_plan,
    plan_from_dict,
)
from .severance_pay import (
    PayConfig,
    SeveranceFormula,
    SeverancePayEngine,
    TaxAssumptions,
)
from .task_tracker import TaskBoard, TaskError, TrackerConfig
from .workforce_data import (
    IngestConfig,
    IngestResult,
    Severity,
    ValidationReport,
    load_workforce_csv,
    load_workforce_dataframe,
)

__all__ = [
    "__version__",
    # box 1
    "load_workforce_csv", "load_workforce_dataframe", "IngestConfig",
    "IngestResult", "ValidationReport", "Severity",
    # box 2
    "ScenarioSimulator", "Scenario", "CostAssumptions", "load_scenarios",
    # box 3
    "SelectionEngine", "SelectionResult", "RifPlan", "load_plan",
    "plan_from_dict", "SelectionConfigError",
    # box 4
    "AdverseImpactAnalyzer", "AdverseImpactResult",
    # box 5
    "ComplianceEngine", "ComplianceConfig", "ComplianceResult",
    # box 6
    "SeverancePayEngine", "PayConfig", "SeveranceFormula", "TaxAssumptions",
    # box 7
    "DocumentGenerator", "DocumentConfig", "DocumentSet",
    # box 8
    "ApprovalLedger", "ApprovalPackage", "ApprovalPolicy", "ApprovalError",
    "default_policy",
    # box 9
    "TaskBoard", "TrackerConfig", "TaskError",
    # box 10
    "AuditPackage", "AuditLog", "RetentionPolicy",
    # orchestrator
    "run_pipeline", "PipelineConfig", "PipelineResult",
]
