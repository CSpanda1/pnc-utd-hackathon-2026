import unittest

from review_financial_indicators import generate_100_indicators


class ReviewFinancialIndicatorsTests(unittest.TestCase):
    def test_generates_exactly_100_indicators(self):
        report = generate_100_indicators()
        self.assertEqual(report["indicator_count"], 100)
        self.assertEqual(len(report["indicators"]), 100)

    def test_contains_expected_portfolio_metrics(self):
        report = generate_100_indicators(limit_customers=10)
        indicators = report["indicators"]
        for key in [
            "monthly_cash_surplus_mean",
            "monthly_cash_surplus_p90",
            "pct_high_dti_gt40",
            "avg_spending_to_income_history_ratio",
            "median_external_transfer_dependency_ratio",
        ]:
            self.assertIn(key, indicators)

    def test_limit_customers_applies(self):
        report = generate_100_indicators(limit_customers=5)
        self.assertEqual(len(report["customer_ids_reviewed"]), 5)


if __name__ == "__main__":
    unittest.main()
