"""Illustrative financial metrics, NOT underwriting or regulated advice."""
import csv
from datetime import date
from decimal import Decimal
from functools import lru_cache
import json
from pathlib import Path
DATA=Path(__file__).parent/'data'
ASSETS={'checking','savings','cd','ira'}
DEBTS={'credit_card','auto_loan','personal_loan','mortgage'}

def _read(filename):
    with (DATA/filename).open(newline='') as f: return list(csv.DictReader(f))

def get_customer_profile(customer_id):
    rows=[r for r in _read('profiles.csv') if r['customer_id']==customer_id]
    if not rows: raise ValueError('Unknown synthetic customer ID')
    return rows[0]

def get_accounts(customer_id):
    get_customer_profile(customer_id)
    return [r for r in _read('accounts.csv') if r['customer_id']==customer_id]


@lru_cache(maxsize=6)
def _cached_history(path, modified_ns, size):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return tuple(csv.DictReader(stream))


def _history_rows(filename):
    path = DATA / filename
    stat = path.stat()
    return _cached_history(str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _owned_account(customer_id, account_id):
    owned = get_accounts(customer_id)  # Also validates customer even when no filter.
    if account_id and not any(a.get('account_id') == account_id for a in owned):
        raise ValueError('Account not available for this customer')


def _iso_day(value):
    if value is None:
        return None
    parsed = date.fromisoformat(value)
    if parsed.isoformat() != value:
        raise ValueError('Use an ISO date: YYYY-MM-DD')
    return value


def get_transactions(customer_id, account_id=None, start_date=None, end_date=None, limit=50, offset=0):
    """Paginated ledger entries; authorization is scoped by the supplied customer.

    API callers must derive customer_id from the authenticated session.
    """
    _owned_account(customer_id, account_id)
    start_date, end_date = _iso_day(start_date), _iso_day(end_date)
    if start_date and end_date and start_date > end_date:
        raise ValueError('Start date must not follow end date')
    if type(limit) is not int or type(offset) is not int or not 1 <= limit <= 200 or offset < 0:
        raise ValueError('Use limit 1–200 and offset >= 0')
    rows = [r for r in _history_rows('transactions.csv') if r['customer_id'] == customer_id
            and (not account_id or r['account_id'] == account_id)
            and (not start_date or r['posted_date'] >= start_date)
            and (not end_date or r['posted_date'] <= end_date)]
    rows.sort(key=lambda r: (r['posted_date'], r['transaction_id']), reverse=True)
    items = [{**r, **{k: float(r[k]) for k in ('amount', 'balance_change', 'balance_after')}}
             for r in rows[offset:offset + limit]]
    return {'items': items, 'total': len(rows), 'limit': limit, 'offset': offset}


def get_account_history(customer_id, account_id=None):
    """Monthly statements for the authenticated customer's listed accounts."""
    _owned_account(customer_id, account_id)
    fields = ('opening_balance', 'balance_increases', 'balance_decreases', 'closing_balance')
    return [{**r, **{k: float(r[k]) for k in fields}, 'transaction_count': int(r['transaction_count'])}
            for r in _history_rows('monthly_statements.csv') if r['customer_id'] == customer_id
            and (not account_id or r['account_id'] == account_id)]


def summarize_transaction_history(customer_id):
    """Cent-exact monthly activity; exclude internal and external transfers from spending."""
    get_customer_profile(customer_id)
    manifest = json.loads((DATA / 'history_manifest.json').read_text(encoding='utf-8'))
    records = [r for r in _history_rows('transactions.csv') if r['customer_id'] == customer_id]
    fields = ('paycheck_income', 'spending', 'debt_payments', 'debt_interest', 'asset_interest',
              'fees', 'external_transfers_in', 'external_transfers_out')
    monthly = {}
    for statement in get_account_history(customer_id):
        month = statement['month']
        row = monthly.setdefault(month, {k: 0 for k in fields} | {
            'month': month, 'transaction_count': 0, 'closing_assets': 0,
            'closing_debts': 0, 'closing_liquid_assets': 0})
        closing = round(statement['closing_balance'] * 100)
        row['closing_assets' if statement['account_type'] in ASSETS else 'closing_debts'] += closing
        if statement['account_type'] in {'checking', 'savings'}:
            row['closing_liquid_assets'] += closing
    for transaction in records:
        row = monthly[transaction['posted_date'][:7]]
        amount = int(Decimal(transaction['amount']) * 100)
        row['transaction_count'] += 1
        kind, event = transaction['account_type'], transaction['transaction_type']
        if event == 'paycheck': row['paycheck_income'] += amount
        if transaction['flow_class'] == 'expense': row['spending'] += amount
        if kind in DEBTS and event == 'payment': row['debt_payments'] += amount
        if event == 'interest_accrual':
            row['asset_interest' if kind in ASSETS else 'debt_interest'] += amount
        if event == 'fee': row['fees'] += amount
        if transaction['flow_class'] == 'external_transfer':
            row['external_transfers_in' if Decimal(transaction['balance_change']) > 0 else 'external_transfers_out'] += amount
    rows = []
    for month in sorted(monthly):
        row = monthly[month]
        converted = {k: (v / 100 if k not in {'month', 'transaction_count'} else v) for k, v in row.items()}
        converted['income_less_spending_and_charges'] = (
            row['paycheck_income'] - row['spending'] - row['debt_interest'] - row['fees']) / 100
        rows.append(converted)
    totals = {key: sum(monthly[m][key] for m in monthly) / 100 for key in fields}
    return {'customer_id': customer_id, 'history_start': manifest['history_start'],
            'as_of_date': manifest['as_of_date'], 'months': rows, 'totals': totals,
            'transaction_count': len(records),
            'definitions': 'Spending counts deposit-account expenses and card purchases once. Debt payments and internal transfers '
                'are excluded from spending; interest/fees are shown separately. Asset accrual is non-cash. '
                'External transfers are snapshot-calibration flows with unlisted accounts. '
                'Income less spending and charges is an accrual activity measure, not checking cash flow.'}


def get_history_evidence(customer_id):
    history = summarize_transaction_history(customer_id)
    return {key: history[key] for key in ('history_start', 'as_of_date', 'totals', 'transaction_count', 'definitions')} | {
        'recent_months': history['months'][-3:]}

def calculate_indicators(customer_id):
    p=get_customer_profile(customer_id); a=get_accounts(customer_id)
    balances={t:sum(float(x['balance']) for x in a if x['account_type']==t) for t in ASSETS|DEBTS}
    monthly_income=float(p['monthly_income']); nondebt=float(p['monthly_non_debt_expenses'])
    debt_payments=sum(float(x['monthly_payment']) for x in a if x['account_type'] in DEBTS)
    total_outflow=nondebt+debt_payments
    liquid=balances['checking']+balances['savings']
    total_assets=sum(balances[t] for t in ASSETS)
    total_debts=sum(balances[t] for t in DEBTS)
    credit_limit=float(p['credit_limit'])
    return {
        'customer_id':customer_id,
        'monthly_income':round(monthly_income,2),
        'monthly_non_debt_expenses':round(nondebt,2),
        'monthly_debt_payments':round(debt_payments,2),
        'monthly_cash_surplus':round(monthly_income-total_outflow,2),
        'net_worth_partial':round(total_assets-total_debts,2),
        'liquid_assets':round(liquid,2),
        'emergency_fund_months':round(liquid/total_outflow,2) if total_outflow>0 else None,
        'debt_payment_to_income_pct':round(100*debt_payments/monthly_income,2) if monthly_income>0 else None,
        'credit_card_utilization_pct':round(100*balances['credit_card']/credit_limit,2) if credit_limit>0 else None,
        'savings_rate_pct':round(100*(monthly_income-total_outflow)/monthly_income,2) if monthly_income>0 else None,
        'liquid_assets_to_debt_pct':round(100*liquid/total_debts,2) if total_debts>0 else None,
        'credit_card_balance':round(balances['credit_card'],2),
        'financial_data_scope':'Partial balance-sheet: excludes home values, external assets, taxes and market changes'
    }

def simulate_purchase(customer_id, purchase_price, down_payment=0, apr_pct=8, term_months=60):
    if not 0<=down_payment<=purchase_price or purchase_price<0: raise ValueError('Invalid purchase/down payment')
    if term_months<1 or apr_pct<0: raise ValueError('Invalid loan terms')
    m=calculate_indicators(customer_id)
    principal=purchase_price-down_payment; rate=apr_pct/1200
    payment=principal/term_months if rate==0 else principal*rate/(1-(1+rate)**(-term_months))
    remaining_liquid=m['liquid_assets']-down_payment
    outflow=m['monthly_non_debt_expenses']+m['monthly_debt_payments']+payment
    return {'customer_id':customer_id,'assumptions':{'purchase_price':purchase_price,'down_payment':down_payment,'apr_pct':apr_pct,'term_months':term_months},'new_monthly_payment':round(payment,2),'projected_monthly_surplus':round(m['monthly_cash_surplus']-payment,2),'remaining_liquid_assets':round(remaining_liquid,2),'projected_emergency_fund_months':round(remaining_liquid/outflow,2) if outflow>0 else None,'caveat':'Excludes taxes, insurance, maintenance, loan fees and future changes; not lending advice.'}

if __name__=='__main__':
    import json
    print(json.dumps(calculate_indicators('C001'),indent=2))
    print(json.dumps(simulate_purchase('C001',30000,5000),indent=2))
