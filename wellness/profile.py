"""Consolidate all accounts into one unified financial profile."""
from .data_gen import ASSET_TYPES, LIABILITY_TYPES

LIQUID = {"checking", "savings"}


def build_profile(customer):
    accts = customer["accounts"]
    by_type = {}
    for a in accts:
        by_type[a["type"]] = round(by_type.get(a["type"], 0) + a["balance"], 2)
    assets = round(sum(a["balance"] for a in accts if a["type"] in ASSET_TYPES), 2)
    liabilities = round(sum(a["balance"] for a in accts if a["type"] in LIABILITY_TYPES), 2)
    tx = customer["transactions"]
    months = sorted({t["date"][:7] for t in tx})
    n = max(len(months), 1)
    spend = -sum(t["amount"] for t in tx if t["amount"] < 0 and t["category"] != "debt_payments")
    debt = -sum(t["amount"] for t in tx if t["category"] == "debt_payments")
    return {
        "customer_id": customer["customer_id"],
        "monthly_income": customer["monthly_income"],
        "total_assets": assets,
        "total_liabilities": liabilities,
        "net_worth": round(assets - liabilities, 2),
        "liquid_cash": round(sum(a["balance"] for a in accts if a["type"] in LIQUID), 2),
        "balances_by_type": by_type,
        "avg_monthly_spending": round(spend / n, 2),
        "monthly_debt_payments": round(debt / n, 2),
        "accounts": accts,
    }
