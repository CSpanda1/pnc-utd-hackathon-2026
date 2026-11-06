import unittest
from unittest.mock import patch

from markov_chain_simulation_demo import MarkovConfig, STATES, run_demo, run_markov_chain


class MarkovChainSimulationDemoTests(unittest.TestCase):
    def test_simulation_shape_and_transition_rows(self):
        result = run_markov_chain(
            customer_id="C001",
            config=MarkovConfig(trials=200, months=12, seed=7),
        )
        self.assertIn(result["initial_state"], STATES)
        matrix = result["transition_matrix"]
        for state in STATES:
            self.assertIn(state, matrix)
            row_sum = sum(matrix[state][target] for target in STATES)
            self.assertAlmostEqual(row_sum, 1.0, places=6)
        prob = result["results"]["prob_ever_stress_or_worse"]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_dual_provider_narration_paths(self):
        with (
            patch("markov_chain_simulation_demo.generate_with_openai", return_value="openai markov summary"),
            patch("markov_chain_simulation_demo.generate_with_huggingface", return_value="hf markov summary"),
        ):
            result = run_demo(
                customer_id="C001",
                config=MarkovConfig(trials=100, months=6, seed=9),
                providers=["openai", "huggingface"],
            )
        self.assertEqual(result["narrations"]["openai"]["status"], "ok")
        self.assertEqual(result["narrations"]["huggingface"]["status"], "ok")
        self.assertIn("openai markov summary", result["narrations"]["openai"]["output"])
        self.assertIn("hf markov summary", result["narrations"]["huggingface"]["output"])


if __name__ == "__main__":
    unittest.main()
