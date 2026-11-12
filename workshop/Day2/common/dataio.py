"""common/dataio.py - loads synthetic finance records for analysis.

WHY THIS FILE EXISTS
--------------------
Session 6's parallel workers analyse real (synthetic) records instead of
asking the LLM to imagine them. This module owns file access:

    data/budgets/department_a.json        -> load_budget()
    data/invoices/pending_invoices.json   -> load_invoices()
    data/reimbursements/pending_claims.json -> load_claims()
    data/reference/compliance_rules.json  -> load_compliance_rules()

Keeping the readers here means the graph nodes contain *analysis logic*
only, and the same loaders can be reused by the API version in Session 9.

Every function returns plain dicts/lists - no LangChain types - so the
loaders are trivially testable without a model running.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from common import config


def _read_json(relative_path: str) -> Any:
    """Read one JSON file from the Day2 data folder.

    Parameters
    ----------
    relative_path : str
        Path relative to Day2/data, e.g. "budgets/department_a.json".

    Returns
    -------
    The parsed JSON (dict or list).

    Raises
    ------
    FileNotFoundError
        With the full path, so a missing data file is obvious.
    """
    path: Path = config.DATA_DIR / relative_path
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Day 2 expects the sample data folder to exist next to the code."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_budget(department_slug: str = "department_a") -> dict:
    """Load one department budget record.

    Parameters
    ----------
    department_slug : str
        File stem inside data/budgets/ (default "department_a").

    Returns
    -------
    dict with keys: department, cost_centre, annual_budget, quarterly_plan,
    actual_spend_to_date, committed_unpaid, ...
    """
    return _read_json(f"budgets/{department_slug}.json")


def load_invoices() -> dict:
    """Load pending invoices plus the purchase orders they must match.

    Returns
    -------
    dict with keys: invoices (list), purchase_orders (list), generated_on.
    """
    return _read_json("invoices/pending_invoices.json")


def load_claims() -> dict:
    """Load pending employee reimbursement claims.

    Returns
    -------
    dict with keys: claims (list), generated_on.
    """
    return _read_json("reimbursements/pending_claims.json")


def load_vendors() -> list[dict]:
    """Load the synthetic vendor master (same file the tools use)."""
    return _read_json("reference/vendors.json")["vendors"]


def load_compliance_rules() -> list[dict]:
    """Load the rule definitions used by the compliance worker.

    Returns
    -------
    list[dict]
        Each rule has rule_id, description, source, applies_to and a
        `check` name that maps to a Python function in D2P3.py.
    """
    return _read_json("reference/compliance_rules.json")["rules"]


# ==========================================================================
# ANALYSIS HELPERS - pure Python, no LLM, no side effects
# ==========================================================================
def analyse_budget(budget: dict) -> dict:
    """Compute utilisation metrics for one budget record.

    Parameters
    ----------
    budget : dict
        One record from load_budget().

    Returns
    -------
    dict with computed fields:
        utilisation_pct  - (actual / annual) * 100
        remaining        - annual - actual
        forecast         - actual + committed_unpaid
        status           - GREEN / AMBER / RED per Budget Guidelines 4.x

    WHY DETERMINISTIC?
    ------------------
    Arithmetic must be exact. An LLM can round 6,50,000/10,00,000 to the
    wrong percentage; this function cannot. The LLM only narrates results
    later (Session 6 aggregator).
    """
    annual = float(budget["annual_budget"])
    actual = float(budget["actual_spend_to_date"])
    committed = float(budget.get("committed_unpaid", 0))

    pct = (actual / annual * 100.0) if annual else 0.0
    remaining = annual - actual
    forecast = actual + committed

    # Thresholds from Department Budget Guidelines section 4 (simplified).
    if pct > 90:
        status = "RED"
    elif pct > 75:
        status = "AMBER"
    else:
        status = "GREEN"

    return {
        "department": budget.get("department", "unknown"),
        "annual_budget": annual,
        "actual_spend": actual,
        "committed_unpaid": committed,
        "utilisation_pct": round(pct, 1),
        "remaining": remaining,
        "forecast": forecast,
        "status": status,
    }


def analyse_invoices(invoice_data: dict) -> dict:
    """Find missing fields, duplicates and PO overruns in pending invoices.

    Parameters
    ----------
    invoice_data : dict
        Result of load_invoices().

    Returns
    -------
    dict with:
        total_pending   - sum of invoice totals
        count           - number of invoices
        issues          - list of human-readable finding strings

    CHECKS PERFORMED
    ----------------
    1. Required fields present (po_reference, gstin - Invoice Guidelines 2)
    2. Duplicate invoice numbers (Guidelines 4.4)
    3. Invoice total vs PO approved amount (Vendor Payment Policy 3.2)
    4. GRN missing (Procurement Policy 5.2)
    """
    invoices = invoice_data.get("invoices", [])
    purchase_orders = {
        po["po_id"]: po for po in invoice_data.get("purchase_orders", [])
    }

    issues: list[str] = []
    total = 0.0
    seen_ids: set[str] = set()

    for inv in invoices:
        inv_id = inv.get("invoice_id", "?")
        total += float(inv.get("total_amount", 0))

        # 1. Required fields ------------------------------------------------
        if not inv.get("po_reference"):
            issues.append(
                f"{inv_id}: missing PO reference "
                f"(invoice total Rs. {inv.get('total_amount', 0):,.0f})"
            )
        if not inv.get("gstin"):
            issues.append(f"{inv_id}: missing vendor GSTIN - input tax credit at risk")

        # 2. Duplicate invoice number --------------------------------------
        if inv_id in seen_ids:
            issues.append(f"{inv_id}: DUPLICATE invoice number - possible double payment")
        seen_ids.add(inv_id)

        # 3. PO amount comparison ------------------------------------------
        po_id = inv.get("po_reference")
        if po_id and po_id in purchase_orders:
            approved = float(purchase_orders[po_id]["approved_amount"])
            inv_total = float(inv.get("total_amount", 0))
            if inv_total > approved:
                issues.append(
                    f"{inv_id}: total Rs. {inv_total:,.0f} exceeds "
                    f"{po_id} approved Rs. {approved:,.0f}"
                )

        # 4. GRN check ------------------------------------------------------
        if not inv.get("grn_received"):
            issues.append(f"{inv_id}: goods/service receipt (GRN) not recorded")

    return {
        "count": len(invoices),
        "total_pending": total,
        "issues": issues,
    }


def analyse_claims(claim_data: dict) -> dict:
    """Summarise pending reimbursement claims and flag policy problems.

    Parameters
    ----------
    claim_data : dict
        Result of load_claims().

    Returns
    -------
    dict with:
        total_amount, count, by_status (dict), issues (list)

    The `policy_issue` field already stored on each claim is echoed into
    the report; the function also re-checks the receipt rule (Rs. 500+)
    so the logic is visible to students.
    """
    claims = claim_data.get("claims", [])
    issues: list[str] = []
    by_status: dict[str, int] = {}
    total = 0.0

    for claim in claims:
        amount = float(claim.get("amount", 0))
        total += amount
        status = claim.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

        # Re-apply rule 3.1 of the reimbursement policy ourselves.
        if amount >= 500 and not claim.get("receipts_attached"):
            issues.append(
                f"{claim.get('claim_id')}: Rs. {amount:,.0f} without receipts "
                "(policy requires receipts >= Rs. 500)"
            )
        # Echo the stored policy issue so nothing already flagged is lost.
        if claim.get("policy_issue"):
            issues.append(f"{claim.get('claim_id')}: {claim['policy_issue']}")

    return {
        "count": len(claims),
        "total_amount": total,
        "by_status": by_status,
        "issues": issues,
    }


def analyse_compliance(
    invoice_data: dict,
    claim_data: dict,
    rules: list[dict] | None = None,
) -> dict:
    """Apply the compliance rules to invoices and reimbursement claims.

    Parameters
    ----------
    invoice_data : dict  - from load_invoices()
    claim_data   : dict  - from load_claims()
    rules        : list  - optional override; defaults to the JSON rules

    Returns
    -------
    dict with:
        checked - how many rules ran
        passed  - number of rules with zero findings
        findings - list of "RULE-ID: message" strings

    HOW RULES MAP TO CODE
    ---------------------
    Each rule in compliance_rules.json carries a `check` name. The
    CHECKS dict below maps that name to a small lambda. Adding a new rule
    therefore needs one JSON entry and one lambda - no graph changes.
    """
    if rules is None:
        rules = load_compliance_rules()

    invoices = invoice_data.get("invoices", [])
    claims = claim_data.get("claims", [])

    # ---- the actual check implementations (see rule `check` names) -------
    def invoice_above_25k_requires_po(inv: dict) -> str | None:
        if float(inv.get("total_amount", 0)) >= 25000 and not inv.get("po_reference"):
            return f"{inv['invoice_id']}: >= Rs. 25,000 but no PO reference"
        return None

    def invoice_requires_gstin(inv: dict) -> str | None:
        if not inv.get("gstin"):
            return f"{inv['invoice_id']}: GSTIN missing"
        return None

    def invoice_above_25k_requires_grn(inv: dict) -> str | None:
        if float(inv.get("total_amount", 0)) >= 25000 and not inv.get("grn_received"):
            return f"{inv['invoice_id']}: >= Rs. 25,000 but GRN not recorded"
        return None

    def claim_above_500_requires_receipt(claim: dict) -> str | None:
        if float(claim.get("amount", 0)) >= 500 and not claim.get("receipts_attached"):
            return f"{claim['claim_id']}: receipt missing for Rs. {claim['amount']}"
        return None

    def claim_within_30_days(claim: dict) -> str | None:
        # The synthetic file already stores the finding when a claim is late;
        # here we only confirm the flag so the rule shows as "checked".
        if claim.get("policy_issue") and "30 days" in claim["policy_issue"]:
            return f"{claim['claim_id']}: submitted after the 30-day limit"
        return None

    CHECKS = {
        "invoice_above_25k_requires_po": invoice_above_25k_requires_po,
        "invoice_requires_gstin": invoice_requires_gstin,
        "invoice_above_25k_requires_grn": invoice_above_25k_requires_grn,
        "claim_above_500_requires_receipt": claim_above_500_requires_receipt,
        "claim_within_30_days": claim_within_30_days,
    }

    findings: list[str] = []
    passed = 0

    for rule in rules:
        check = CHECKS.get(rule["check"])
        if check is None:
            continue  # unknown check name in JSON - skip safely

        # Choose the record set the rule applies to.
        records = invoices if rule["applies_to"] == "invoice" else claims
        rule_hits: list[str] = []
        for record in records:
            hit = check(record)
            if hit:
                rule_hits.append(f"{rule['rule_id']}: {hit}")

        if rule_hits:
            findings.extend(rule_hits)
        else:
            passed += 1

    return {"checked": len(rules), "passed": passed, "findings": findings}
