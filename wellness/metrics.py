"""Deterministic financial health indicators. The LLM never does this math."""
from .data_gen import LIABILITY_TYPES


def _div(a, b):
    return round(a / b, 4) if b else None


def _status(value, good, warn, higher_is_better=True):
    if value is None:
        return "unknown"
    if not higher_is_better:
        value, good, warn = -value, -good, -warn
    return "good" if value >= good else "watch" if value >= warn else "risk"


def compute_metrics(p):
    inc, spend, debt = p["monthly_income"], p["avg_monthly_spending"], p["monthly_debt_payments"]
    cards = [a for a in p["accounts"] if a["type"] == "credit_card"]
    mtg = sum(a.get("monthly_payment", 0) for a in p["accounts"] if a["type"] == "mortgage")
    out = {
        "emergency_fund_months": _div(p["liquid_cash"], spend + debt),
        "debt_to_income": _div(debt, inc),
        "savings_rate": _div(inc - spend - debt, inc),
        "credit_utilization": _div(sum(a["balance"] for a in cards),
                                   sum(a["credit_limit"] for a in cards)),
        "net_worth": p["net_worth"],
        "debt_to_asset": _div(p["total_liabilities"], p["total_assets"]),
        "housing_cost_ratio": _div(mtg, inc),
        "liquidity_ratio": _div(p["liquid_cash"], p["total_liabilities"]),
    }
    status = {
        "emergency_fund_months": _status(out["emergency_fund_months"], 6, 3),
        "debt_to_income": _status(out["debt_to_income"], 0.2, 0.36, False),
        "savings_rate": _status(out["savings_rate"], 0.2, 0.1),
        "credit_utilization": _status(out["credit_utilization"], 0.1, 0.3, False),
        "net_worth": "good" if p["net_worth"] > 0 else "risk",
        "debt_to_asset": _status(out["debt_to_asset"], 0.4, 0.7, False),
        "housing_cost_ratio": _status(out["housing_cost_ratio"], 0.28, 0.35, False),
        "liquidity_ratio": _status(out["liquidity_ratio"], 0.1, 0.03),
    }
    return {"values": out, "status": status}
