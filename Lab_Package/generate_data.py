"""Generate deterministic synthetic profile/account snapshot files.

This script creates the baseline workshop inputs (profiles.csv and accounts.csv),
then calls generate_history.py to derive dated transactions/statements.
"""
import csv, random
from pathlib import Path

def generate_data():
    """Write synthetic profile/account snapshots and regenerate history artifacts."""
    # Fixed seed keeps the workshop dataset reproducible across runs.
    rng=random.Random(20261107)
    out=Path(__file__).parent/'data';out.mkdir(exist_ok=True)
    accounts=[]; profiles=[]
    for i in range(1,31):
        cid=f'C{i:03d}'
        # Profile-level assumptions used by downstream indicators/simulations.
        income=round(rng.uniform(3500,14500),2)
        expenses=round(income*rng.uniform(.48,.94),2)
        profiles.append(dict(customer_id=cid,monthly_income=income,monthly_non_debt_expenses=expenses,credit_limit=round(rng.uniform(5000,30000),2)))
        for typ,lo,hi in [('checking',300,9000),('savings',1000,28000),('cd',0,14000),('ira',1000,85000),('credit_card',1000,11000),('auto_loan',0,30000),('personal_loan',0,14000),('mortgage',0,330000)]:
            amount=round(rng.uniform(lo,hi),2)
            # Optional account types are sometimes zeroed to keep the population diverse.
            if typ in ('cd','auto_loan','personal_loan','mortgage') and rng.random()<.30: amount=0
            payment=0
            apr=0
            # Payment/APR rules are intentionally simplified workshop assumptions.
            if typ=='credit_card': payment=round(max(30,amount*.03),2);apr=22.5
            elif typ=='auto_loan' and amount:payment=round(amount/48,2);apr=7.5
            elif typ=='personal_loan' and amount:payment=round(amount/36,2);apr=11.0
            elif typ=='mortgage' and amount:payment=round(amount*.0065,2);apr=6.5
            if amount: accounts.append(dict(customer_id=cid,account_type=typ,balance=amount,monthly_payment=payment,apr_pct=apr))
    for name,rows in [('profiles.csv',profiles),('accounts.csv',accounts)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print('Generated',len(profiles),'synthetic profiles and',len(accounts),'synthetic accounts')
    # Regenerate derived ledger history so snapshots and history stay aligned.
    from generate_history import generate_history
    print(generate_history(out))

if __name__ == '__main__':
    generate_data()
