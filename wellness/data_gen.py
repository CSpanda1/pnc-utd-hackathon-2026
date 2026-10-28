"""Seeded synthetic data generator. Liabilities store a positive balance owed."""
import argparse
import json
import random
from datetime import date

ASSET_TYPES = {"checking", "savings", "cd", "ira", "brokerage"}
LIABILITY_TYPES = {"credit_card", "auto_loan", "personal_loan", "mortgage"}


def _amortized_payment(principal, apr, months):
    r = apr / 12
    return round(principal / months if r == 0 else principal * r / (1 - (1 + r) ** -months), 2)


def generate_customer(seed=42, customer_id="CUST-0001", months=6):
    """Return one customer: accounts, monthly income/expense transactions."""
    rng = random.Random(seed)
    income = round(rng.uniform(4500, 9500), 2)
    accounts = [
        {"id": "chk-1", "type": "checking", "name": "Everyday Checking",
         "balance": round(rng.uniform(1500, 6000), 2)},
        {"id": "sav-1", "type": "savings", "name": "High-Yield Savings",
         "balance": round(rng.uniform(2000, 25000), 2), "apy": 0.04},
        {"id": "cd-1", "type": "cd", "name": "12-Month CD",
         "balance": round(rng.choice([5000, 10000, 15000]), 2), "apy": 0.045},
        {"id": "ira-1", "type": "ira", "name": "Roth IRA",
         "balance": round(rng.uniform(8000, 90000), 2)},
        {"id": "brk-1", "type": "brokerage", "name": "Brokerage",
         "balance": round(rng.uniform(0, 30000), 2)},
    ]
    for i in range(rng.randint(1, 3)):
        bal = round(rng.uniform(200, 6000), 2)
        limit = round(bal + rng.choice([2000, 5000, 8000, 12000]), 2)
        accounts.append({"id": f"cc-{i+1}", "type": "credit_card", "name": f"Credit Card {i+1}",
                         "balance": bal, "credit_limit": limit, "apr": 0.2399,
                         "min_payment": round(max(35, bal * 0.02), 2)})
    auto = round(rng.uniform(6000, 28000), 2)
    accounts.append({"id": "auto-1", "type": "auto_loan", "name": "Auto Loan", "balance": auto,
                     "apr": 0.069, "monthly_payment": _amortized_payment(auto, 0.069, 48)})
    pl = round(rng.uniform(1500, 9000), 2)
    accounts.append({"id": "pl-1", "type": "personal_loan", "name": "Personal Loan", "balance": pl,
                     "apr": 0.105, "monthly_payment": _amortized_payment(pl, 0.105, 36)})
    mort = round(rng.uniform(150000, 380000), 2)
    accounts.append({"id": "mtg-1", "type": "mortgage", "name": "Mortgage", "balance": mort,
                     "apr": 0.065, "monthly_payment": _amortized_payment(mort, 0.065, 360)})
    debt_pmts = sum(a.get("monthly_payment", a.get("min_payment", 0))
                    for a in accounts if a["type"] in LIABILITY_TYPES)
    transactions = []
    today = date.today().replace(day=1)
    for m in range(months):
        y, mo = today.year, today.month - m
        while mo <= 0:
            mo += 12
            y -= 1
        d = lambda day: date(y, mo, day).isoformat()
        transactions.append({"date": d(1), "category": "income", "amount": income})
        transactions.append({"date": d(2), "category": "debt_payments", "amount": -round(debt_pmts, 2)})
        for cat, lo, hi in [("groceries", 400, 800), ("utilities", 150, 350), ("dining", 100, 500),
                            ("transport", 100, 300), ("subscriptions", 30, 120),
                            ("shopping", 100, 600)]:
            transactions.append({"date": d(15), "category": cat, "amount": -round(rng.uniform(lo, hi), 2)})
    return {"customer_id": customer_id, "name": "Synthetic Customer", "monthly_income": income,
            "accounts": accounts, "transactions": transactions}


def generate_dataset(n=1, seed=42):
    return [generate_customer(seed + i, f"CUST-{i+1:04d}") for i in range(n)]


def main():
    p = argparse.ArgumentParser(description="Generate synthetic customers as JSON")
    p.add_argument("-n", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("-o", "--output", default="-")
    a = p.parse_args()
    text = json.dumps(generate_dataset(a.n, a.seed), indent=2)
    if a.output == "-":
        print(text)
    else:
        with open(a.output, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
