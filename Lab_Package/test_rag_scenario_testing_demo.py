import unittest

from rag_scenario_testing_demo import run_rag_scenario_testing


class RagScenarioTestingDemoTests(unittest.TestCase):
    def test_offline_rag_report_shape(self):
        report = run_rag_scenario_testing(customer_id="C001", top_k=3, provider="none")
        self.assertEqual(report["customer_id"], "C001")
        self.assertEqual(report["provider"], "none")
        self.assertEqual(report["tests_run"], 3)
        self.assertEqual(len(report["results"]), 3)
        for row in report["results"]:
            self.assertIn("retrieval", row)
            self.assertEqual(len(row["retrieval"]), 3)
            self.assertIn("coverage", row)
            self.assertIn("answer", row)

    def test_coverage_ratio_bounds(self):
        report = run_rag_scenario_testing(customer_id="C001", top_k=4, provider="none")
        for row in report["results"]:
            ratio = row["coverage"]["coverage_ratio"]
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 1.0)


if __name__ == "__main__":
    unittest.main()
