import sqlite3
from contextlib import closing
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from backend.main import COOKIE, Settings, create_app, digest

HEADERS = {'Origin': 'http://localhost:3000', 'X-Workshop-CSRF': '1'}
PASSWORD = 'workshop-password-2026'


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / 'test.sqlite3'
        self.app = create_app(Settings(self.db))
        self.client = TestClient(self.app, headers=HEADERS)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.temp.cleanup()

    def register(self, email='alice@example.com'):
        return self.client.post('/auth/register', json={'email': email, 'password': PASSWORD})

    def test_anonymous_access_denied(self):
        for endpoint in ['/me', '/financials', '/transactions']:
            self.assertEqual(self.client.get(endpoint).status_code, 401)
        self.assertEqual(self.client.post('/scenario', json={}).status_code, 401)
        self.assertEqual(self.client.post('/narration', json={}).status_code, 401)

    def test_registration_cookie_hash_and_logout_revocation(self):
        response = self.register()
        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()['customer_id'], 'C001')
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertIn('SameSite=strict', response.headers['set-cookie'])
        token = self.client.cookies[COOKIE]
        with closing(sqlite3.connect(self.db)) as db, db:
            stored_hash = db.execute('SELECT password_hash FROM users').fetchone()[0]
            stored_session = db.execute('SELECT token_hash FROM sessions').fetchone()[0]
        self.assertTrue(stored_hash.startswith('$argon2id$'))
        self.assertNotIn(PASSWORD, stored_hash)
        self.assertEqual(stored_session, digest(token))
        self.assertEqual(self.client.post('/auth/logout', json={}).status_code, 204)
        self.client.cookies.set(COOKIE, token)
        self.assertEqual(self.client.get('/me').status_code, 401)

    def test_login_errors_and_rotation(self):
        self.register()
        old = self.client.cookies[COOKIE]
        response = self.client.post('/auth/login', json={'email': 'alice@example.com', 'password': 'incorrect-password'})
        self.assertEqual(response.status_code, 401)
        response = self.client.post('/auth/login', json={'email': 'ALICE@example.com', 'password': PASSWORD})
        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(old, self.client.cookies[COOKIE])
        self.client.cookies.set(COOKIE, old)
        self.assertEqual(self.client.get('/me').status_code, 401)

    def test_cross_user_isolation_and_rejected_customer_override(self):
        self.register()
        alice = self.client.cookies[COOKIE]
        response = self.register('bob@example.com')
        self.assertEqual(response.json()['customer_id'], 'C002')
        self.assertEqual(self.client.get('/financials').json()['indicators']['customer_id'], 'C002')
        self.assertEqual(self.client.post('/scenario', json={'customer_id': 'C001'}).status_code, 422)
        self.assertEqual(self.client.post('/narration', json={'evidence': {}}).status_code, 422)
        # Registering another user rotated the previous browser session.
        self.client.cookies.set(COOKIE, alice)
        self.assertEqual(self.client.get('/me').status_code, 401)

    def test_csrf_and_cross_origin_rejected(self):
        for headers in [{'Origin': 'https://evil.example'}, {'X-Workshop-CSRF': ''}, {'Origin': ''}]:
            response = self.client.post('/auth/register', json={'email': 'alice@example.com', 'password': PASSWORD}, headers=headers)
            self.assertEqual(response.status_code, 403)

    def test_expired_session_rejected(self):
        self.register()
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute('UPDATE sessions SET expires_at=0')
        self.assertEqual(self.client.get('/financials').status_code, 401)

    def test_validation_and_zero_apr(self):
        self.register()
        invalid = [{'purchase_price': -1}, {'down_payment': 31000}, {'apr_pct': 36},
                   {'term_months': 60.5}, {'term_months': True}, {'apr_pct': 'NaN'}]
        for body in invalid:
            with self.subTest(body=body):
                self.assertEqual(self.client.post('/scenario', json=body).status_code, 422)
        response = self.client.post('/scenario', json={'purchase_price': 12000, 'down_payment': 0, 'apr_pct': 0, 'term_months': 12})
        self.assertEqual(response.json()['new_monthly_payment'], 1000)

    def test_rate_limiting(self):
        body = {'email': 'unknown@example.com', 'password': PASSWORD}
        for _ in range(10):
            self.assertEqual(self.client.post('/auth/login', json=body).status_code, 401)
        response = self.client.post('/auth/login', json=body)
        self.assertEqual(response.status_code, 429)
        self.assertIn('retry-after', response.headers)

    def test_narration_uses_server_evidence(self):
        self.register()
        with patch('backend.main.narrate', return_value='Synthetic explanation') as narrate:
            response = self.client.post('/narration', json={})
        self.assertEqual(response.status_code, 200)
        evidence, provider = narrate.call_args.args
        self.assertEqual(evidence['indicators']['customer_id'], 'C001')
        self.assertEqual(evidence['scenario']['customer_id'], 'C001')
        self.assertEqual(provider, 'none')
        self.assertEqual(len(evidence['transaction_history']['recent_months']), 3)

    def test_transaction_filters_and_customer_scope(self):
        self.register()
        account = 'A-C001-CHECKING'
        response = self.client.get('/transactions', params={'account_id': account,
            'start_date': '2026-10-01', 'end_date': '2026-10-31', 'limit': 2})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data['items']), 2)
        self.assertTrue(all(r['customer_id'] == 'C001' and r['account_id'] == account
                            and r['posted_date'].startswith('2026-10') for r in data['items']))
        self.assertEqual(self.client.get('/transactions', params={'account_id': 'A-C002-CHECKING'}).status_code, 404)
        for params in [{'customer_id': 'C002'}, {'limit': 201}, {'offset': -1},
                       {'start_date': 'invalid'}, {'start_date': '2026-10-31', 'end_date': '2026-10-01'}]:
            with self.subTest(params=params):
                self.assertEqual(self.client.get('/transactions', params=params).status_code, 422)
        financials = self.client.get('/financials').json()
        self.assertEqual(len(financials['history']['months']), 24)
        self.assertTrue(all(r['customer_id'] == 'C001' for r in financials['account_history']))

    def test_duplicate_email_and_capacity(self):
        self.register()
        self.assertEqual(self.register().status_code, 409)
        with closing(sqlite3.connect(self.db)) as db, db:
            db.executemany('INSERT INTO users(email,password_hash,customer_id) VALUES (?,?,?)',
                           [(f'user{i}@example.com', 'unused', f'C{i:03d}') for i in range(2,31)])
        self.assertEqual(self.register('extra@example.com').status_code, 409)

    def test_https_cookie_configuration(self):
        app = create_app(Settings(Path(self.temp.name) / 'secure.sqlite3', 'https://workshop.example', True))
        with TestClient(app, base_url='https://workshop.example', headers={**HEADERS, 'Origin': 'https://workshop.example'}) as client:
            response = client.post('/auth/register', json={'email': 'secure@example.com', 'password': PASSWORD})
            self.assertIn('Secure', response.headers['set-cookie'])


if __name__ == '__main__':
    unittest.main()
