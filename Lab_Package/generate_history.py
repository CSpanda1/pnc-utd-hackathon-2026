"""Deterministic, cent-exact synthetic ledger anchored to existing closing balances.

No real transactions, investment returns, or banking accounting rules are modeled.
Run directly to extend existing profiles/accounts without regenerating their values.
"""
import argparse
import calendar
import csv
from datetime import date
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path
import random

SEED = 20261108
ASSETS = {'checking', 'savings', 'cd', 'ira'}
YIELDS = {'checking': 0, 'savings': 2.5, 'cd': 4.25, 'ira': 3.5}


def cents(value):
    return int((Decimal(str(value)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def dollars(value):
    return f'{value / 100:.2f}'


def interest(balance, annual_pct):
    return int((Decimal(balance) * Decimal(str(annual_pct)) / 1200).quantize(Decimal('1'), rounding=ROUND_HALF_UP))


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_history(data_dir=None, end=date(2026, 10, 31), months=24):
    data_dir = Path(data_dir or Path(__file__).parent / 'data')
    if not 1 <= months <= 120 or end.day != calendar.monthrange(end.year, end.month)[1]:
        raise ValueError('Use 1–120 months and a month-end date')
    with (data_dir / 'profiles.csv').open(newline='', encoding='utf-8') as stream:
        profiles = list(csv.DictReader(stream))
    with (data_dir / 'accounts.csv').open(newline='', encoding='utf-8') as stream:
        accounts = list(csv.DictReader(stream))
    end_index = end.year * 12 + end.month - 1
    periods = [date(index // 12, index % 12 + 1, 1)
               for index in range(end_index - months + 1, end_index + 1)]
    start = periods[0]
    rng = random.Random(SEED)
    transactions, statements, enriched_accounts = [], [], []
    event_serial = 0

    for profile in profiles:
        cid = profile['customer_id']
        owned = [dict(a) for a in accounts if a['customer_id'] == cid]
        by_type = {a['account_type']: a for a in owned}
        if len(by_type) != len(owned) or 'checking' not in by_type:
            raise ValueError('History generator expects one row per customer/type and a checking account')
        for kind, account in by_type.items():
            account['account_id'] = f'A-{cid}-{kind.upper()}'
        account_events = {kind: [] for kind in by_type}
        opening = {}
        savings_payroll = [0] * months
        savings_expenses = [0] * months

        def add(kind, month_index, day, transaction_type, amount, sign, category,
                flow_class, transfer_id='', description=None):
            nonlocal event_serial
            if amount <= 0:
                return
            event_serial += 1
            period = periods[month_index]
            account_events[kind].append(dict(
                customer_id=cid, account_id=by_type[kind]['account_id'], account_type=kind,
                posted_date=period.replace(day=min(day, calendar.monthrange(period.year, period.month)[1])).isoformat(),
                transaction_type=transaction_type, category=category,
                description=description or transaction_type.replace('_', ' ').capitalize(),
                amount_cents=amount, change_cents=sign * amount, flow_class=flow_class,
                transfer_id=transfer_id, serial=event_serial))

        def paired(kind, month_index, day, transaction_type, amount, sign, category):
            link = f'X-{cid}-{kind}-{month_index:02d}-{transaction_type}'
            add(kind, month_index, day, transaction_type, amount, sign, category, 'internal_transfer', link)
            checking_type = ('transfer_out' if sign > 0 else 'deposit') if kind in ASSETS else 'payment'
            add('checking', month_index, day, checking_type,
                amount, -sign if kind in ASSETS else -1, category, 'internal_transfer', link,
                f'Synthetic linked {kind.replace("_", " ")} {transaction_type.replace("_", " ")}')

        # Savings/CD/IRA accrue monthly interest after contributions. Back-solve
        # opening cents; a residual cent is included in the final contribution.
        for kind in ['savings', 'cd', 'ira']:
            if kind not in by_type:
                continue
            final = cents(by_type[kind]['balance'])
            deposits = [max(1, int(final * rng.uniform(.0015, .004))) for _ in periods]
            withdrawals = [int(final * .001) if kind == 'savings' and m % 3 == 0 else 0 for m in range(months)]
            def ending(initial):
                value = initial
                for deposit, withdrawal in zip(deposits, withdrawals):
                    value += deposit - withdrawal
                    value += interest(value, YIELDS[kind])
                return value
            low, high = 0, final
            while low < high:
                mid = (low + high + 1) // 2
                if ending(mid) <= final:
                    low = mid
                else:
                    high = mid - 1
            opening[kind] = value = low
            for m in range(months):
                deposit = deposits[m]
                if kind == 'savings':
                    savings_payroll[m] = deposit // 2
                    add(kind, m, 1, 'paycheck', savings_payroll[m], 1, 'payroll', 'income',
                        description='Synthetic split-payroll deposit to savings')
                    deposit -= savings_payroll[m]
                paired(kind, m, 5, 'deposit', deposit, 1, 'savings_contribution' if kind == 'savings' else 'investment_contribution')
                value += deposits[m]
                if withdrawals[m]:
                    if kind == 'savings' and m % 6 == 0:
                        savings_expenses[m] = withdrawals[m]
                        add(kind, m, 22, 'payment', withdrawals[m], -1, 'other_spending', 'expense')
                    else:
                        paired(kind, m, 22, 'withdrawal', withdrawals[m], -1, 'savings_withdrawal')
                    value -= withdrawals[m]
                accrued = interest(value, YIELDS[kind])
                add(kind, m, 31, 'interest_accrual', accrued, 1, 'asset_interest', 'non_cash')
                value += accrued
                if m == months - 1 and final > value:
                    paired(kind, m, 31, 'deposit_rounding', final - value, 1, 'investment_contribution')
                    value = final

        # Loans: positive liability balances; interest/fees increase the balance,
        # payments reduce it. Principal components sum exactly to opening-final.
        for kind in ['auto_loan', 'personal_loan', 'mortgage']:
            if kind not in by_type:
                continue
            account = by_type[kind]
            final = cents(account['balance'])
            principal = [max(1, int(cents(account['monthly_payment']) * rng.uniform(.18, .32))) for _ in periods]
            opening[kind] = value = final + sum(principal)
            for m in range(months):
                accrued = interest(value, account['apr_pct'])
                fee = 500 if m % 7 == 3 else 0
                add(kind, m, 20, 'interest_accrual', accrued, 1, 'debt_interest', 'non_cash')
                add(kind, m, 20, 'fee', fee, 1, 'servicing_fee', 'non_cash')
                paired(kind, m, 21, 'payment', principal[m] + accrued + fee, -1, 'debt_payment')
                value -= principal[m]

        income_base, expense_base = cents(profile['monthly_income']), cents(profile['monthly_non_debt_expenses'])
        if 'credit_card' in by_type:
            opening['credit_card'] = int(cents(by_type['credit_card']['balance']) * .98)
        card_value = opening.get('credit_card', 0)
        for m in range(months):
            # Small generated trend and seasonal variation; not real seasonality.
            factor = .96 + .04 * (m / max(months - 1, 1))
            income = int(income_base * factor * rng.uniform(.98, 1.02))
            seasonal = 1.12 if periods[m].month in {11, 12} else 1.0
            spending = int(expense_base * factor * seasonal * rng.uniform(.94, 1.06))
            checking_income = income - savings_payroll[m]
            add('checking', m, 1, 'paycheck', checking_income // 2, 1, 'payroll', 'income')
            add('checking', m, 15, 'paycheck', checking_income - checking_income // 2, 1, 'payroll', 'income')
            card_spend = int(spending * .22) if 'credit_card' in by_type else 0
            direct = spending - card_spend - savings_expenses[m]
            portions = [int(direct * fraction) for fraction in [.45, .20, .15]]
            portions.append(direct - sum(portions))
            for day, category, amount in zip([3, 10, 18, 25], ['housing', 'utilities', 'groceries', 'other_spending'], portions):
                add('checking', m, day, 'withdrawal' if category == 'groceries' else 'payment', amount, -1, category, 'expense')
            if card_spend:
                pieces = [card_spend // 4] * 3
                pieces.append(card_spend - sum(pieces))
                for day, category, amount in zip([4, 11, 19, 25], ['groceries', 'transport', 'dining', 'shopping'], pieces):
                    add('credit_card', m, day, 'purchase', amount, 1, category, 'expense')
                accrued = interest(card_value, by_type['credit_card']['apr_pct'])
                fee = 1500 if m % 9 == 2 else 0
                add('credit_card', m, 26, 'interest_accrual', accrued, 1, 'debt_interest', 'non_cash')
                add('credit_card', m, 26, 'fee', fee, 1, 'card_fee', 'non_cash')
                final = cents(by_type['credit_card']['balance'])
                target = final if m == months - 1 else int(final * (.98 + .02 * (m + 1) / months))
                payment = card_value + card_spend + accrued + fee - target
                if payment <= 0:
                    raise ValueError('Generated card payment must be positive')
                paired('credit_card', m, 28, 'payment', payment, -1, 'debt_payment')
                card_value = target

        # Checking residual flows represent cash management with unlisted external
        # accounts. They are explicit external transfers, never payroll or spending.
        final = cents(by_type['checking']['balance'])
        opening['checking'] = value = int(final * .9)
        for m, period in enumerate(periods):
            events = sorted([e for e in account_events['checking'] if e['posted_date'].startswith(period.strftime('%Y-%m'))],
                            key=lambda e: (e['posted_date'], e['serial']))
            running, minimum = value, value
            for event in events:
                running += event['change_cents']
                minimum = min(minimum, running)
            target = final if m == months - 1 else max(10000, int(final * rng.uniform(.8, 1.2)))
            funding = max(0, target - running, 10000 - minimum)
            # Serial ordering on day 1 must place funding before other events.
            if funding:
                add('checking', m, 1, 'deposit', funding, 1, 'external_cash_management', 'external_transfer',
                    description='Synthetic funding from an unlisted external account')
                account_events['checking'][-1]['serial'] = -m - 1
            withdrawal = running + funding - target
            add('checking', m, 31, 'withdrawal', withdrawal, -1, 'external_cash_management', 'external_transfer',
                description='Synthetic withdrawal to an unlisted external account')
            value = target

        for kind, account in by_type.items():
            events = sorted(account_events[kind], key=lambda e: (e['posted_date'], e['serial']))
            balance = opening[kind]
            for m, period in enumerate(periods):
                month_events = [e for e in events if e['posted_date'].startswith(period.strftime('%Y-%m'))]
                initial = balance
                increases = decreases = 0
                for event in month_events:
                    balance += event['change_cents']
                    if balance < 0:
                        raise ValueError(f'Negative balance: {cid} {kind} {event}')
                    increases += max(0, event['change_cents'])
                    decreases += max(0, -event['change_cents'])
                    transactions.append(dict(transaction_id=f'T{len(transactions)+1:07d}',
                        **{key: event[key] for key in ['customer_id', 'account_id', 'account_type', 'posted_date',
                           'transaction_type', 'category', 'description']}, amount=dollars(event['amount_cents']),
                        balance_change=dollars(event['change_cents']), balance_after=dollars(balance),
                        flow_class=event['flow_class'], transfer_id=event['transfer_id']))
                statements.append(dict(customer_id=cid, account_id=account['account_id'], account_type=kind,
                    month=period.strftime('%Y-%m'), opening_balance=dollars(initial),
                    balance_increases=dollars(increases), balance_decreases=dollars(decreases),
                    closing_balance=dollars(balance), transaction_count=len(month_events)))
            if balance != cents(account['balance']):
                raise ValueError(f'Closing balance mismatch: {cid} {kind}')
            account.update(opening_balance=dollars(opening[kind]), history_start=start.isoformat(),
                           as_of_date=end.isoformat(), annual_yield_pct=YIELDS.get(kind, 0))
            enriched_accounts.append(account)

    # Transaction IDs encode stable ledger order; use date + ID for running balances.
    write_csv(data_dir / 'accounts.csv', enriched_accounts)
    write_csv(data_dir / 'transactions.csv', transactions)
    write_csv(data_dir / 'monthly_statements.csv', statements)
    manifest = dict(schema_version=2, seed=SEED, history_start=start.isoformat(), as_of_date=end.isoformat(),
        months=months, customers=len(profiles), accounts=len(enriched_accounts), transactions=len(transactions),
        monthly_statements=len(statements), currency='USD',
        balance_convention='Assets and liabilities are positive; balance_change increases or decreases that account balance.',
        calibration='Opening balances and explicit external cash-management transfers are calibrated to preserve supplied ending balances.',
        limitations='Synthetic monthly accrual, no daily compounding, IRA market returns, taxes, contribution eligibility, or actual banking fees modeled.')
    (data_dir / 'history_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end', type=date.fromisoformat, default=date(2026, 10, 31))
    parser.add_argument('--months', type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(generate_history(end=args.end, months=args.months), indent=2))
