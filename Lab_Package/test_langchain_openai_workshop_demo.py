import unittest
from unittest.mock import patch

from langchain_openai_workshop_demo import run_demo


class _FakeChain:
    def __init__(self, answer: str):
        self.answer = answer

    def invoke(self, payload):
        return f"{self.answer} | route={payload['route']}"


class LangChainWorkshopDemoTests(unittest.TestCase):
    def test_indicator_route_runs_and_returns_answer(self):
        with patch('langchain_openai_workshop_demo.build_langchain_chain', return_value=_FakeChain('indicator answer')):
            result = run_demo(
                customer_id='C001',
                question='How does my monthly surplus look right now?',
                purchase_price=30000,
                down_payment=5000,
                apr_pct=8,
                term_months=60,
            )
        self.assertEqual(result['route'], 'indicators')
        self.assertIsNotNone(result['monthly_cash_surplus'])
        self.assertIsNone(result['scenario_payment'])
        self.assertIn('indicator answer', result['answer'])

    def test_scenario_route_runs_and_returns_scenario_payment(self):
        with patch('langchain_openai_workshop_demo.build_langchain_chain', return_value=_FakeChain('scenario answer')):
            result = run_demo(
                customer_id='C001',
                question='What if I buy a car with 8% APR?',
                purchase_price=32000,
                down_payment=6000,
                apr_pct=8,
                term_months=60,
            )
        self.assertEqual(result['route'], 'scenario')
        self.assertIsNotNone(result['scenario_payment'])
        self.assertIn('scenario answer', result['answer'])


if __name__ == '__main__':
    unittest.main()
