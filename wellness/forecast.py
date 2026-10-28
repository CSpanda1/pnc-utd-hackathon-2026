"""Simple cash-balance projection."""


def forecast_cash(p, months=12, extra_monthly_payment=0.0, start_cash=None):
    cash = p["liquid_cash"] if start_cash is None else start_cash
    net = p["monthly_income"] - p["avg_monthly_spending"] - p["monthly_debt_payments"] - extra_monthly_payment
    path = []
    for m in range(1, months + 1):
        cash = round(cash + net, 2)
        path.append({"month": m, "cash": cash})
    return {"monthly_net_cash_flow": round(net, 2), "projection": path,
            "min_cash": min(x["cash"] for x in path),
            "goes_negative": any(x["cash"] < 0 for x in path)}
