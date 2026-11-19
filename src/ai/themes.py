"""Complaint themes and the mapping from CFPB's own issue categories (used as a check)."""
from __future__ import annotations

THEMES = {
    "payment_hardship": "Borrower cannot afford or is struggling to make payments (job loss, "
                        "income drop, medical, forbearance ending, asking for relief).",
    "servicing_error": "Servicer mistakes or poor handling: misapplied/lost payments, wrong "
                       "statements, unreachable staff, credit-reporting errors, payment-process problems.",
    "escrow": "Escrow account, property tax or insurance payments, escrow shortage or "
              "payment-increase disputes.",
    "modification_denial": "Loan modification, workout or forbearance requests denied, delayed "
                           "or mishandled.",
    "foreclosure": "Foreclosure, eviction, sale or default-related actions threatened or taken.",
    "origination_other": "Applying for a loan or refinance, closing, appraisal, fees, "
                         "or anything else not covered above.",
}
THEME_LIST = list(THEMES)


def cfpb_theme(issue: str | None, sub_issue: str | None = None) -> str:
    """Rule-based theme from CFPB's own Issue / Sub-issue text. A sanity check, not ground truth."""
    t = f"{issue or ''} | {sub_issue or ''}".lower()
    if "escrow" in t:
        return "escrow"
    if "foreclos" in t or "eviction" in t:
        return "foreclosure"
    if "struggling to pay" in t or "hardship" in t or "can't pay" in t:
        return "payment_hardship"
    if "modif" in t or "loss mitigation" in t or "forbearance" in t:
        return "modification_denial"
    if ("payment process" in t or "servicing" in t or "credit report" in t
            or "credit reporting" in t or "communication" in t or "collect" in t
            or "closing your account" in t):
        return "servicing_error"
    return "origination_other"
