import unittest
from unittest.mock import patch

from langgraph_openai_workshop_demo import run_demo


class LangGraphWorkshopDemoTests(unittest.TestCase):
    def test_indicator_route_runs_and_returns_answer(self):
        with patch('langgraph_openai_workshop_demo.narrate', return_value='indicator answer'):
            result = run_demo(
                customer_id='C001',
                question='How does my monthly surplus look right now?',
                purchase_price=30000,
                down_payment=5000,
                apr_pct=8,
                term_months=60,
            )
        self.assertEqual(result['route'], 'indicators')
        self.assertIn('indicators', result)
        self.assertIn('history_evidence', result)
        self.assertEqual(result['answer'], 'indicator answer')

    def test_scenario_route_runs_and_returns_scenario(self):
        with patch('langgraph_openai_workshop_demo.narrate', return_value='scenario answer'):
            result = run_demo(
                customer_id='C001',
                question='What if I buy a car with 8% APR?',
                purchase_price=32000,
                down_payment=6000,
                apr_pct=8,
                term_months=60,
            )
        self.assertEqual(result['route'], 'scenario')
        self.assertIn('scenario', result)
        self.assertEqual(result['answer'], 'scenario answer')


if __name__ == '__main__':
    unittest.main()
