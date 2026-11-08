import json

from wellness.data_gen import generate_customer, generate_dataset
from wellness.graph import level3
from wellness.levels import level1, level2
from wellness.llm import MockLLM
from wellness.metrics import compute_metrics
from wellness.profile import build_profile
from wellness.scenarios import car_purchase
from wellness.validation import validate

C = generate_customer(1)
P = build_profile(C)


def test_dataset_deterministic_and_all_account_types():
    assert generate_dataset(2, 5) == generate_dataset(2, 5)
    types = {a["type"] for a in C["accounts"]}
    assert {"checking", "savings", "cd", "ira", "credit_card", "auto_loan",
            "personal_loan", "mortgage"} <= types


def test_profile_net_worth():
    assert abs(P["net_worth"] - (P["total_assets"] - P["total_liabilities"])) < 0.01


def test_metrics_at_least_six():
    assert len(compute_metrics(P)["values"]) >= 6


def test_car_purchase():
    r = car_purchase(P, 30000, 5000, 0.0, 50)
    assert r["monthly_payment"] == 500 and r["loan_amount"] == 25000


def test_validation_flags_bad():
    assert not validate("simulate_car_purchase", {"loan_amount": -1})[0]
    assert validate("x", {"a": 1})[0]


def test_levels_run_offline():
    q = "If I buy a $30,000 car with $5,000 down, do I have enough for emergencies?"
    for fn in (level1, level2, level3):
        assert fn(MockLLM(), C, q)
    assert "monthly_payment" in level3(MockLLM(), C, q)
    json.dumps(P)


class RouterLLM:
    def __init__(self, tool, args):
        self.call = json.dumps({"tool": tool, "args": args})

    def chat(self, messages):
        assert "ROLE:ROUTER" in messages[0]["content"]
        return self.call


def test_coach_percent_apr_regression():
    for apr in (7, 0.07):
        answer = level2(RouterLLM("simulate_car_purchase", {
            "price": 30000, "down": 5000, "apr": apr,
        }), C, "Can I afford a $30,000 car with $5,000 down with 7% interest?")
        assert "$495.03" in answer
        assert "$4,701.80" in answer
        assert "60-month" in answer
        assert "7.00%" in answer


def test_coach_health_status_and_formatting():
    answer = level2(RouterLLM("calculate_health_metrics", {}), C, "Financial health?")
    metrics = compute_metrics(P)
    assert "Missing assets" in answer
    for key, status in metrics["status"].items():
        assert f"({status})" in answer
    assert ("Positive indicators" in answer) == ("good" in metrics["status"].values())
    positive = answer.split("**Positive indicators**", 1)[-1].split("**Needs attention**", 1)[0] if "Positive indicators" in answer else ""
    assert "(watch)" not in positive
    assert f"{metrics['values']['credit_utilization']:.2%}" in answer


def test_coach_rejects_invalid_financing():
    for value in (float('nan'), float('inf'), 60.5, 0):
        answer = level2(RouterLLM("simulate_car_purchase", {"term_months": value}), C, "Buy a car")
        assert "understand" in answer or "valid car financing" in answer


def test_coach_handles_bad_router_json():
    class BrokenLLM:
        def chat(self, messages):
            return "No JSON here"
    assert "rephrase" in level2(BrokenLLM(), C, "Show my accounts")


def test_general_coaching_works_without_model_routing():
    class UnusableRouter:
        def chat(self, messages):
            raise AssertionError("General coaching should not need model routing")
    for question in ("How can I be smart about my finances?",
                     "How can I improve my finances?", "What is my financial health?"):
        answer = level2(UnusableRouter(), C, question)
        assert "financial health snapshot" in answer
        assert "Suggested next steps" in answer
        assert "Please check the inputs" not in answer


def test_health_tool_ignores_model_supplied_arguments():
    for args in ({"customer_id": "wrong", "question": "help"}, ["invalid"], None):
        answer = level2(RouterLLM("calculate_health_metrics", args), C, "Help me budget")
        assert "Suggested next steps" in answer
        assert f"${P['net_worth']:,.2f}" in answer


def test_forecast_horizon_validation():
    for months in (0, -1, 121, 2.5, float('nan'), "twelve"):
        answer = level2(RouterLLM("forecast_balances", {"months": months}), C, "Forecast")
        assert "1 to 120 whole months" in answer


def test_forecast_filters_unsupported_arguments():
    class ForecastLLM:
        def chat(self, messages):
            if "ROLE:ROUTER" in messages[0]["content"]:
                return json.dumps({"tool": "forecast_balances", "args": {
                    "months": 3, "customer_id": "invented"}})
            evidence = json.loads(messages[-1]["content"].split("EVIDENCE: ")[1])
            assert len(evidence["projection"]) == 3
            return "Forecast available"
    assert level2(ForecastLLM(), C, "Forecast my balance") == "Forecast available"
