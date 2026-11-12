"""common/tools.py - custom finance tools for the FinDesk agent.

WHY THIS FILE EXISTS
--------------------
RAG retrieves *knowledge* from documents. Tools perform *operations*:
calculations and lookups that an LLM must never do by itself (LLMs are
weak at arithmetic and can hallucinate vendor records). Session 4 binds
these tools to the chat model, which then decides WHEN to call them.

HOW A TOOL IS DEFINED
---------------------
@tool                                          <- registers the function
def gst_calculator(amount: float, rate: float) -> str:
    'One-line description THE MODEL READS to decide when to call it.'
    ...                                        <- plain Python: exact maths

The decorator converts the function into a LangChain StructuredTool:
* the docstring becomes the tool description shown to the model,
* type hints become the JSON schema of the arguments,
* the model returns a tool call, ToolNode executes the function, and the
  string result is fed back to the model as a ToolMessage.

DATA
----
Vendor records, GST rates and claim rules are loaded from JSON files in
data/reference/ once at import time - business data stays out of code.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.tools import tool

from common import config

# --------------------------------------------------------------------------
# Load reference data ONCE at import time (not on every tool call).
# --------------------------------------------------------------------------
_REFERENCE_DIR: Path = config.DATA_DIR / "reference"


def _load_json(name: str) -> dict:
    """Read a JSON file from data/reference/ and return it as a dict."""
    path = _REFERENCE_DIR / name
    # encoding="utf-8" keeps any ₹ / non-ASCII characters intact.
    return json.loads(path.read_text(encoding="utf-8"))


_VENDORS: list[dict] = _load_json("vendors.json")["vendors"]
_GST_RATES: dict[str, float] = _load_json("gst_rates.json")["rates_percent"]


# ==========================================================================
# 1. CALCULATOR TOOLS - exact arithmetic, no LLM involved
# ==========================================================================
@tool
def expense_calculator(amounts: str) -> str:
    """Add up a list of expense amounts and return the total in rupees.

    Use this when the employee lists several expenses and asks for the
    total, e.g. "add up 1200, 450 and 800".

    Parameters
    ----------
    amounts : str
        Comma-separated numbers, e.g. "1200, 450, 800".

    Returns
    -------
    str
        "Total expense: Rs. 2,450 (3 items)" or an error message.
    """
    try:
        # Split on commas, strip spaces, convert every part to float.
        # A generator expression inside sum() adds them exactly.
        values = [float(part.strip()) for part in amounts.split(",") if part.strip()]
        if not values:
            return "No amounts provided. Send numbers separated by commas."
        total = sum(values)
        # :,.0f formats with thousands separators and no decimals:
        # 2450 -> "2,450"
        return f"Total expense: Rs. {total:,.0f} ({len(values)} items)"
    except ValueError:
        return "Could not parse the amounts. Use e.g. '1200, 450, 800'."


@tool
def gst_calculator(base_amount: float, rate_name: str = "standard") -> str:
    """Calculate GST on a base amount using an illustrative rate slabs.

    Use this when the employee asks for GST, CGST/SGST or IGST on an
    amount. Rates are the illustrative training slabs, not live tax advice.

    Parameters
    ----------
    base_amount : float
        Taxable value in rupees (the amount before tax).
    rate_name : str
        One of: essential (5%), standard (12%), goods_default (18%),
        luxury (28%), services_default (18%). Defaults to "standard".

    Returns
    -------
    str
        Breakdown with total, CGST and SGST (intra-state split).
    """
    # Look up the rate; unknown names fall back to the default slab.
    rate = _GST_RATES.get(rate_name, _GST_RATES["standard"])
    gst = base_amount * rate / 100.0
    total = base_amount + gst
    # Intra-state supply splits GST equally into CGST and SGST.
    half = gst / 2.0
    return (
        f"Base amount : Rs. {base_amount:,.2f}\n"
        f"GST rate    : {rate}% ({rate_name})\n"
        f"GST amount  : Rs. {gst:,.2f}\n"
        f"CGST / SGST : Rs. {half:,.2f} each (intra-state split)\n"
        f"Total       : Rs. {total:,.2f}"
    )


@tool
def budget_utilisation(budget: float, spent: float) -> str:
    """Calculate budget utilisation percentage and remaining balance.

    Use this whenever the employee gives a budget and an amount spent and
    asks for utilisation, remaining balance or percentage used.

    Parameters
    ----------
    budget : float
        Total budget in rupees.
    spent : float
        Amount already spent in rupees.

    Returns
    -------
    str
        Utilisation %, remaining balance and a green/amber/red status
        based on the Department Budget Guidelines thresholds.
    """
    if budget <= 0:
        return "Budget must be greater than zero."
    # Exact formula from Department Budget Guidelines 3.1:
    # utilisation = (spent / budget) * 100
    pct = spent / budget * 100.0
    remaining = budget - spent

    # Status bands from Department Budget Guidelines section 4 (simplified
    # to absolute utilisation rather than elapsed-time target).
    if pct > 90:
        status = "RED - mandatory budget review required"
    elif pct > 75:
        status = "AMBER - monitor closely"
    else:
        status = "GREEN - within limits"

    return (
        f"Budget utilisation : {pct:.1f}%\n"
        f"Remaining balance  : Rs. {remaining:,.0f}\n"
        f"Status             : {status}"
    )


@tool
def invoice_total(taxable_value: float, gst_amount: float) -> str:
    """Add taxable value and GST to get the invoice grand total.

    Use this when the employee asks for the total payable on an invoice
    given its taxable value and GST amount.

    Parameters
    ----------
    taxable_value : float
        Amount before tax in rupees.
    gst_amount : float
        GST charged in rupees.

    Returns
    -------
    str
        Grand total of the invoice.
    """
    total = taxable_value + gst_amount
    return (
        f"Invoice total: Rs. {total:,.2f} "
        f"(taxable {taxable_value:,.2f} + GST {gst_amount:,.2f})"
    )


# ==========================================================================
# 2. LOOKUP TOOL - reads the synthetic vendor master
# ==========================================================================
@tool
def vendor_lookup(vendor_name: str) -> str:
    """Look up a vendor in the company vendor master by (partial) name.

    Use this when the employee asks about a vendor's GSTIN, payment terms,
    approval status or contact. Matching is case-insensitive and works on
    part of the name (e.g. "acme" finds "Acme Office Supplies").

    Parameters
    ----------
    vendor_name : str
        Full or partial vendor name.

    Returns
    -------
    str
        Vendor details, or a not-found message listing known vendors.
    """
    needle = vendor_name.strip().lower()
    for vendor in _VENDORS:
        if needle in vendor["name"].lower():
            # approved is stored as a boolean in JSON - render as Yes/No.
            approved = "Yes" if vendor["approved"] else "No"
            return (
                f"Vendor      : {vendor['name']} ({vendor['vendor_id']})\n"
                f"GSTIN       : {vendor['gstin']}\n"
                f"Category    : {vendor['category']}\n"
                f"Terms       : {vendor['payment_terms']}\n"
                f"Approved    : {approved}\n"
                f"Contact     : {vendor['contact']}"
            )
    names = ", ".join(v["name"] for v in _VENDORS)
    return f"No vendor matched '{vendor_name}'. Known vendors: {names}."


# --------------------------------------------------------------------------
# Single list the graph binds to the model and passes to ToolNode.
# Order is the order the model sees in the tool catalogue.
# --------------------------------------------------------------------------
FINANCE_TOOLS = [
    expense_calculator,
    gst_calculator,
    budget_utilisation,
    invoice_total,
    vendor_lookup,
]
