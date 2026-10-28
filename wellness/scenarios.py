"""What-if simulators."""
from .data_gen import _amortized_payment
from .forecast import forecast_cash


def car_purchase(p, price=30000, down=5000, apr=0.07, term_months=60):
    loan = price - down
    pay = _amortized_payment(loan, apr, term_months)
    cash_after = round(p["liquid_cash"] - down, 2)
    fc = forecast_cash(p, 12, pay, cash_after)
    monthly_out = p["avg_monthly_spending"] + p["monthly_debt_payments"] + pay
    return {"loan_amount": round(loan, 2), "monthly_payment": pay,
            "total_interest": round(pay * term_months - loan, 2),
            "cash_after_down_payment": cash_after,
            "emergency_fund_months_after": round(cash_after / monthly_out, 2) if monthly_out else None,
            "new_monthly_net_cash_flow": fc["monthly_net_cash_flow"],
            "cash_goes_negative_within_12m": fc["goes_negative"]}
