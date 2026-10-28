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
