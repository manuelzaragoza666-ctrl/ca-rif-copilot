# Document & Communication Generator — California RIF Copilot (Box 7)

Produces the paperwork a reduction requires — Cal-WARN notices, separation
letters, the OWBPA decisional-unit disclosure, manager talking points, an
employee FAQ, and a notice-day checklist — from the outputs of boxes 3 through 6.

## Files

| File | Purpose |
|---|---|
| `document_generator.py` | The generator. |
| `test_document_generator.py` | 57 tests. |

## Quick start

```python
from document_generator import DocumentConfig, DocumentGenerator

cfg = DocumentConfig(
    employer_name="Acme Inc.",
    employer_address="1 Market St, San Francisco, CA 94105",
    signatory_name="Dana Reyes", signatory_title="VP, People",
    lwdb_name="SF Workforce Board", lwdb_email="board@example.gov",
    lwdb_phone="(555) 555-0100", service_coordination="lwdb",
    decisional_unit="All Engineering employees at the SF HQ establishment",
)
docs = DocumentGenerator(cfg).generate(
    compliance=compliance_result, selection=selection_result,
    pay=pay_result, scores=selection_result.scores,
)

docs.blocked        # True if the gate refused
docs.documents      # list of GeneratedDocument
docs.write("./out") # one subdirectory per document type
```

## The gate

This module generates nothing while Box 5 reports unresolved blockers. That is
the reason Box 5 was built first: a notice produced from a non-compliant
scenario is worse than no notice, because it creates a dated artifact
memorializing the defect.

Two blocker classes are treated differently, because they are not the same
problem:

**Legal-judgment blockers** — adverse impact indicated, employees on protected
leave, union members whose CBA hasn't been checked, an unresolved selection tie.
These turn on an assessment a lawyer makes. Recorded counsel sign-off clears
them: `counsel_override_by` plus `counsel_override_reason`, both required. The
sign-off is stamped into **every generated document** and into the manifest, and
cannot be stripped.

**Data-completeness blockers** — a missing pay rate, an uncomputable severance
figure, an undetermined WARN establishment, an undeclared leave policy. These
are **non-overridable**. No amount of legal judgment fills in a blank field; the
document would simply be wrong. A test confirms counsel sign-off does not clear
them.

The alternative designs were both worse. A gate with no override gets bypassed —
people copy the templates into Word and the tool stops seeing what ships. A gate
with a plain `--force` flag is decoration. Requiring a named person and a written
reason that follows the documents is the version that survives contact with a
deadline.

## What it will not produce

**A signable release of claims.** A release is a contract, and an ADEA release
carries OWBPA requirements whose failure is invisible until it's litigated — the
release stays enforceable as to everything *except* the age claim, so the
employer pays for a release it doesn't receive.

What it produces instead is a skeleton for counsel: the computed severance
figures, the required OWBPA elements enumerated, a list of terms California
prohibits, and every judgment call left as a visible placeholder. It is marked
SKELETON FOR COUNSEL and the generator will not remove that marking.

**Anything that isn't a draft.** Every document carries a DRAFT banner.

**A decisional unit it wasn't given.** The OWBPA disclosure needs one, defining
it is a legal judgment about the scope of the decision, and it is frequently
*not* the same as the comparison groups used for scoring. If it's undefined, the
disclosure isn't generated and the report says why.

**Invented facts.** Unfillable values become `[[PLACEHOLDER]]` tokens, which are
detected, counted, and listed in the summary. Final wages stay a placeholder for
the same reason they do in boxes 5 and 6 — payroll supplies that number.

## Documents

| Type | Audience | Notes |
|---|---|---|
| `warn_notice_employee` | Affected employees | All four SB 617 disclosures; counsel review checklist appended |
| `warn_notice_agency` | EDD, LWDB, local officials | Three separate notices |
| `separation_letter` | Individual | One per employee, with computed figures |
| `owbpa_disclosure` | Employees 40+ | Job titles and ages only, no names |
| `severance_agreement_draft` | Counsel only | Skeleton, never signable |
| `manager_script` | Managers | See below |
| `employee_faq` | Affected employees | Real EDD/Covered California/CalFresh resources |
| `notice_day_checklist` | HR | Built from Box 5's dated obligations |
| `hr_summary` | HR | Internal |

## Two documents worth reading closely

**The manager script** carries an explicit "do not say" list, because most of
the legal exposure in a RIF comes from what a manager improvises in the room.
Each item on it has generated litigation: references to age or retirement or
"your next chapter" (the single most common source of an age claim), references
to health or leave or pregnancy, asserting a performance reason when the
criteria were mixed, discussing other employees' status, promises about rehire
or references, negotiating terms, and "I fought for you" — which undermines the
process and invites the employee to go looking for the real reason. It also
covers what to do when someone becomes distressed and what to say when someone
threatens to sue.

**The separation letter** states plainly that final wages and vacation are paid
regardless of whether the employee signs anything. That's Labor Code § 206.5,
and it's the sentence most likely to be quietly dropped when someone edits the
template.

## Integration

- **Reads** Box 5's `ComplianceResult` (mandatory — running without it is
  refused outright), Box 3's selection, Box 6's payroll register, and the full
  scored population for the OWBPA disclosure.
- **Feeds** Box 8 (Approvals) — the manifest is what gets routed for sign-off —
  and Box 10 (Audit), which should retain the manifest, the override record, and
  the generated set.

## Output

`docs.write(outdir)` produces `00_SUMMARY.md`, `00_manifest.csv`, and one
subdirectory per document type. A blocked run still writes the summary
explaining what stopped it and generates no document files.
