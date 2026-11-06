import unittest
from unittest.mock import patch

from monte_carlo_simulation_demo import MonteCarloConfig, run_demo, run_monte_carlo


class MonteCarloSimulationDemoTests(unittest.TestCase):
    def test_simulation_shape_and_probabilities(self):
        result = run_monte_carlo(
            customer_id="C001",
            config=MonteCarloConfig(trials=100, months=12, seed=7),
        )
        stats = result["results"]
        self.assertIn("end_liquid_p50", stats)
        self.assertGreaterEqual(stats["prob_any_negative_surplus_month"], 0.0)
        self.assertLessEqual(stats["prob_any_negative_surplus_month"], 1.0)
        self.assertGreaterEqual(stats["prob_liquid_below_zero"], 0.0)
        self.assertLessEqual(stats["prob_liquid_below_zero"], 1.0)

    def test_dual_provider_narration_paths(self):
        config = MonteCarloConfig(trials=50, months=6, seed=9)
        with (
            patch("monte_carlo_simulation_demo.generate_with_openai", return_value="openai summary"),
            patch("monte_carlo_simulation_demo.generate_with_huggingface", return_value="hf summary"),
        ):
            result = run_demo(customer_id="C001", config=config, providers=["openai", "huggingface"])
        self.assertEqual(result["narrations"]["openai"]["status"], "ok")
        self.assertEqual(result["narrations"]["huggingface"]["status"], "ok")
        self.assertIn("openai summary", result["narrations"]["openai"]["output"])
        self.assertIn("hf summary", result["narrations"]["huggingface"]["output"])


if __name__ == "__main__":
    unittest.main()
