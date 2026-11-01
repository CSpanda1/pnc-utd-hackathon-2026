"""Ledger/accounting invariants, access scope, pagination, and reproducibility."""
from collections import defaultdict
import csv
from decimal import Decimal
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest

import financial_tools as tools
from generate_history import generate_history, cents


class TransactionHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.accounts = tools._read('accounts.csv')
        cls.transactions = tools._read('transactions.csv')
        cls.statements = tools._read('monthly_statements.csv')

    def test_every_account_has_24_reconciled_months_and_terminal_snapshot(self):
        grouped, monthly = defaultdict(list), defaultdict(list)
        for row in self.transactions:
            grouped[row['account_id']].append(row)
            monthly[(row['account_id'], row['posted_date'][:7])].append(row)
        self.assertEqual(len(grouped), len(self.accounts))
        self.assertEqual(len({r['transaction_id'] for r in self.transactions}), len(self.transactions))
        for account in self.accounts:
            running = cents(account['opening_balance'])
            rows = sorted(grouped[account['account_id']], key=lambda r: (r['posted_date'], r['transaction_id']))
            for row in rows:
                running += cents(row['balance_change'])
                self.assertEqual(running, cents(row['balance_after']))
                self.assertGreaterEqual(running, 0)
                self.assertEqual(abs(cents(row['balance_change'])), cents(row['amount']))
                self.assertEqual(row['customer_id'], account['customer_id'])
                self.assertEqual(row['account_type'], account['account_type'])
                self.assertGreaterEqual(row['posted_date'], '2024-11-01')
                self.assertLessEqual(row['posted_date'], '2026-10-31')
            self.assertEqual(running, cents(account['balance']))
            statements = sorted([s for s in self.statements if s['account_id'] == account['account_id']], key=lambda r: r['month'])
            self.assertEqual(len(statements), 24)
            previous = cents(account['opening_balance'])
            for statement in statements:
                rows = monthly[(account['account_id'], statement['month'])]
                self.assertEqual(cents(statement['opening_balance']), previous)
                closing = previous + sum(cents(r['balance_change']) for r in rows)
                self.assertEqual(closing, cents(statement['closing_balance']))
                self.assertEqual(len(rows), int(statement['transaction_count']))
                self.assertEqual(sum(max(0, cents(r['balance_change'])) for r in rows), cents(statement['balance_increases']))
                self.assertEqual(sum(max(0, -cents(r['balance_change'])) for r in rows), cents(statement['balance_decreases']))
                previous = closing

    def test_transfer_pairs_are_same_customer_date_and_amount_and_balance_neutral(self):
        groups = defaultdict(list)
        for row in self.transactions:
            if row['transfer_id']:
                groups[row['transfer_id']].append(row)
            self.assertEqual(bool(row['transfer_id']), row['flow_class'] == 'internal_transfer')
        for pair in groups.values():
            self.assertEqual(len(pair), 2)
            for field in ['customer_id', 'posted_date', 'amount']:
                self.assertEqual(pair[0][field], pair[1][field])
            self.assertEqual(sum(cents(r['balance_change']) * (1 if r['account_type'] in tools.ASSETS else -1) for r in pair), 0)

    def test_expected_activity_for_every_type(self):
        by_type = defaultdict(set)
        for row in self.transactions:
            by_type[row['account_type']].add(row['transaction_type'])
        self.assertTrue({'deposit', 'paycheck', 'withdrawal', 'payment'} <= by_type['checking'])
        self.assertTrue({'deposit', 'paycheck', 'withdrawal', 'payment', 'interest_accrual'} <= by_type['savings'])
        for kind in ['cd', 'ira']:
            self.assertTrue({'deposit', 'interest_accrual'} <= by_type[kind])
        self.assertTrue({'purchase', 'fee', 'interest_accrual', 'payment'} <= by_type['credit_card'])
        for kind in ['auto_loan', 'personal_loan', 'mortgage']:
            self.assertTrue({'fee', 'interest_accrual', 'payment'} <= by_type[kind])

    def test_filters_pagination_and_account_ownership(self):
        account = tools.get_accounts('C001')[0]['account_id']
        result = tools.get_transactions('C001', account, '2026-10-01', '2026-10-31', limit=2)
        self.assertEqual(len(result['items']), 2)
        self.assertTrue(all(r['account_id'] == account and r['posted_date'].startswith('2026-10') for r in result['items']))
        next_page = tools.get_transactions('C001', account, '2026-10-01', '2026-10-31', limit=2, offset=2)
        self.assertNotEqual(result['items'][0]['transaction_id'], next_page['items'][0]['transaction_id'])
        with self.assertRaises(ValueError): tools.get_transactions('C001', 'A-C002-CHECKING')
        with self.assertRaises(ValueError): tools.get_account_history('C001', 'A-C002-CHECKING')
        with self.assertRaises(ValueError): tools.get_transactions('C001', start_date='2026-11-01', end_date='2026-10-01')
        with self.assertRaises(ValueError): tools.get_transactions('C001', limit=201)
        with self.assertRaises(ValueError): tools.get_transactions('UNKNOWN')

    def test_summary_excludes_transfer_double_counting(self):
        summary = tools.summarize_transaction_history('C001')
        rows = [r for r in self.transactions if r['customer_id'] == 'C001']
        expected_spending = sum(cents(r['amount']) for r in rows if r['flow_class'] == 'expense')
        self.assertEqual(cents(summary['totals']['spending']), expected_spending)
        expected_payments = sum(cents(r['amount']) for r in rows if r['account_type'] in tools.DEBTS and r['transaction_type'] == 'payment')
        self.assertEqual(cents(summary['totals']['debt_payments']), expected_payments)
        self.assertEqual(len(summary['months']), 24)
        self.assertEqual(summary['months'][-1]['closing_liquid_assets'], tools.calculate_indicators('C001')['liquid_assets'])

    def test_reproducible_generation_preserves_original_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for filename in ['profiles.csv', 'accounts.csv']:
                shutil.copyfile(tools.DATA / filename, directory / filename)
            before = {a['account_id']: (a['balance'], a['monthly_payment'], a['apr_pct']) for a in self.accounts}
            generate_history(directory)
            filenames = ['accounts.csv', 'transactions.csv', 'monthly_statements.csv', 'history_manifest.json']
            hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in filenames}
            generate_history(directory)
            self.assertEqual(hashes, {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in filenames})
            with (directory / 'accounts.csv').open(newline='') as stream:
                after = {a['account_id']: (a['balance'], a['monthly_payment'], a['apr_pct']) for a in csv.DictReader(stream)}
            self.assertEqual(before, after)


if __name__ == '__main__':
    unittest.main()
