"""Markov-chain financial state simulation with OpenAI/Hugging Face narration.

This script models customer financial-state transitions over time using a
discrete Markov chain, then asks one or more LLM providers to explain the
risk trajectory in plain language.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass

from financial_tools import calculate_indicators, simulate_purchase


STATES = ("stable", "watch", "stress", "critical")


BASE_TRANSITIONS: dict[str, dict[str, float]] = {
    "stable": {"stable": 0.78, "watch": 0.18, "stress": 0.03, "critical": 0.01},
    "watch": {"stable": 0.20, "watch": 0.58, "stress": 0.18, "critical": 0.04},
    "stress": {"stable": 0.06, "watch": 0.26, "stress": 0.50, "critical": 0.18},
    "critical": {"stable": 0.03, "watch": 0.12, "stress": 0.30, "critical": 0.55},
}


@dataclass(frozen=True)
class MarkovConfig:
    """Configuration for one Markov-chain simulation run."""
    trials: int = 2000
    months: int = 24
    seed: int = 42
    include_purchase: bool = False
    purchase_price: float = 30000
    down_payment: float = 5000
    apr_pct: float = 8.0
    term_months: int = 60


def _pct_to_ratio(value: float) -> float:
    """Normalize values that may be represented as percent (e.g., 42) or ratio (0.42)."""
    return value / 100.0 if value > 1.0 else value


def _normalize_row(row: dict[str, float]) -> dict[str, float]:
    """Ensure each transition row is a proper probability distribution."""
    total = sum(row.values())
    if total <= 0:
        return {"stable": 0.0, "watch": 0.0, "stress": 0.0, "critical": 1.0}
    return {state: max(0.0, prob / total) for state, prob in row.items()}


def _shift_mass(row: dict[str, float], from_state: str, to_state: str, amount: float) -> None:
    """Move transition probability mass between states while preserving non-negativity."""
    if amount <= 0:
        return
    available = row.get(from_state, 0.0)
    moved = min(available, amount)
    row[from_state] = available - moved
    row[to_state] = row.get(to_state, 0.0) + moved


def _derive_initial_state(indicators: dict) -> str:
    """Map customer indicators to a starting financial state."""
    surplus = float(indicators["monthly_cash_surplus"])
    emergency = float(indicators["emergency_fund_months"])
    dti = _pct_to_ratio(float(indicators["debt_payment_to_income_pct"]))
    util = _pct_to_ratio(float(indicators["credit_card_utilization_pct"]))

    if surplus >= 400 and emergency >= 3 and dti < 0.28 and util < 0.45:
        return "stable"
    if surplus >= 100 and emergency >= 1.5 and dti < 0.40 and util < 0.70:
        return "watch"
    if surplus >= -250 and emergency >= 0.5 and dti < 0.55 and util < 0.90:
        return "stress"
    return "critical"


def _risk_adjustment(indicators: dict, scenario_payment: float) -> float:
    """Compute a bounded stress drift used to tilt the transition matrix."""
    income = max(1.0, float(indicators["monthly_income"]))
    dti = _pct_to_ratio(float(indicators["debt_payment_to_income_pct"]))
    util = _pct_to_ratio(float(indicators["credit_card_utilization_pct"]))
    emergency = float(indicators["emergency_fund_months"])
    payment_shock = scenario_payment / income

    dti_risk = max(0.0, dti - 0.28) * 0.35
    util_risk = max(0.0, util - 0.55) * 0.20
    emergency_risk = max(0.0, 2.0 - emergency) * 0.05
    shock_risk = payment_shock * 0.90
    return min(0.20, dti_risk + util_risk + emergency_risk + shock_risk)


def build_transition_matrix(indicators: dict, scenario_payment: float) -> tuple[dict[str, dict[str, float]], float]:
    """Create a risk-adjusted transition matrix and return matrix plus applied drift."""
    drift = _risk_adjustment(indicators, scenario_payment)
    matrix = {state: dict(row) for state, row in BASE_TRANSITIONS.items()}

    # Apply asymmetric stress drift so higher-risk states become more likely under pressure.
    _shift_mass(matrix["stable"], "stable", "watch", drift)
    _shift_mass(matrix["stable"], "watch", "stress", drift * 0.40)

    _shift_mass(matrix["watch"], "watch", "stress", drift)
    _shift_mass(matrix["watch"], "stable", "watch", drift * 0.30)
    _shift_mass(matrix["watch"], "stress", "critical", drift * 0.35)

    _shift_mass(matrix["stress"], "watch", "stress", drift * 0.30)
    _shift_mass(matrix["stress"], "stress", "critical", drift)

    _shift_mass(matrix["critical"], "watch", "stress", drift * 0.40)
    _shift_mass(matrix["critical"], "stress", "critical", drift * 0.50)

    normalized = {state: _normalize_row(row) for state, row in matrix.items()}
    return normalized, round(drift, 6)


def _sample_next_state(row: dict[str, float], rng: random.Random) -> str:
    """Sample the next state from one transition row."""
    x = rng.random()
    cumulative = 0.0
    for state in STATES:
        cumulative += row[state]
        if x <= cumulative:
            return state
    return STATES[-1]


def _checkpoint_months(months: int) -> list[int]:
    """Return month checkpoints used for intermediate probability snapshots."""
    marks = {1, 3, 6, 12, months}
    return sorted(m for m in marks if 1 <= m <= months)


def run_markov_chain(customer_id: str, config: MarkovConfig) -> dict:
    """Simulate state transitions for many trials and aggregate risk metrics."""
    indicators = calculate_indicators(customer_id)
    scenario_payment = 0.0
    if config.include_purchase:
        # Optional purchase adds payment shock, which shifts transition probabilities.
        scenario = simulate_purchase(
            customer_id=customer_id,
            purchase_price=config.purchase_price,
            down_payment=config.down_payment,
            apr_pct=config.apr_pct,
            term_months=config.term_months,
        )
        scenario_payment = float(scenario["new_monthly_payment"])

    initial_state = _derive_initial_state(indicators)
    transition_matrix, risk_drift = build_transition_matrix(indicators, scenario_payment)
    checkpoints = _checkpoint_months(config.months)

    rng = random.Random(config.seed)
    terminal_counts = {state: 0 for state in STATES}
    checkpoint_counts = {month: {state: 0 for state in STATES} for month in checkpoints}
    ever_stress_or_worse = 0
    months_in_critical_sum = 0

    for _ in range(config.trials):
        state = initial_state
        had_stress_or_worse = state in {"stress", "critical"}
        months_in_critical = 1 if state == "critical" else 0

        for month in range(1, config.months + 1):
            # Standard time-homogeneous Markov process: state_{t+1} depends only on state_t.
            state = _sample_next_state(transition_matrix[state], rng)
            if state in {"stress", "critical"}:
                had_stress_or_worse = True
            if state == "critical":
                months_in_critical += 1
            if month in checkpoint_counts:
                checkpoint_counts[month][state] += 1

        terminal_counts[state] += 1
        if had_stress_or_worse:
            ever_stress_or_worse += 1
        months_in_critical_sum += months_in_critical

    checkpoint_probabilities = {
        str(month): {
            state: round(checkpoint_counts[month][state] / config.trials, 4)
            for state in STATES
        }
        for month in checkpoints
    }

    return {
        "customer_id": customer_id,
        "trials": config.trials,
        "months": config.months,
        "seed": config.seed,
        "initial_state": initial_state,
        "assumptions": {
            "include_purchase": config.include_purchase,
            "scenario_payment": round(scenario_payment, 2),
            "risk_drift": risk_drift,
        },
        "transition_matrix": transition_matrix,
        "results": {
            "terminal_state_probabilities": {
                state: round(terminal_counts[state] / config.trials, 4) for state in STATES
            },
            "checkpoint_state_probabilities": checkpoint_probabilities,
            "prob_ever_stress_or_worse": round(ever_stress_or_worse / config.trials, 4),
            "expected_months_in_critical": round(months_in_critical_sum / config.trials, 4),
        },
    }


def _build_prompt(simulation: dict) -> str:
    """Build a provider-agnostic narration prompt from Markov outputs."""
    return (
        "You are an educational financial wellness assistant.\n"
        "Use ONLY the Markov-chain simulation JSON below.\n"
        "Do not invent facts or provide lending/underwriting decisions.\n"
        "Explain state-transition risk, downside probability, and one practical budgeting action.\n"
        "Return exactly 5 bullet points and include one caveat on synthetic assumptions.\n\n"
        f"SIMULATION:\n{json.dumps(simulation, indent=2)}"
    )


def generate_with_openai(prompt: str) -> str:
    """Generate narration via OpenAI Responses API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to run OpenAI narration.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    response = OpenAI(api_key=api_key).responses.create(model=model, input=prompt)
    return response.output_text


def generate_with_huggingface(prompt: str) -> str:
    """Generate narration via local Hugging Face model wrapper."""
    from local_hf import generate_local

    return generate_local(prompt)


def run_demo(customer_id: str, config: MarkovConfig, providers: list[str]) -> dict:
    """Run Markov simulation once, then request narration from selected providers."""
    simulation = run_markov_chain(customer_id, config)
    prompt = _build_prompt(simulation)
    narrations: dict[str, dict] = {}

    for provider in providers:
        if provider == "none":
            narrations["none"] = {
                "status": "ok",
                "output": "No model call requested. Use simulation JSON directly.",
            }
            continue
        try:
            if provider == "openai":
                output = generate_with_openai(prompt)
            elif provider == "huggingface":
                output = generate_with_huggingface(prompt)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
            narrations[provider] = {"status": "ok", "output": output}
        except Exception as exc:
            # Preserve simulation output even if one narration provider fails.
            narrations[provider] = {"status": "error", "error": str(exc)}

    return {"simulation": simulation, "narrations": narrations}


def _parse_providers(value: str) -> list[str]:
    """Parse and validate comma-separated providers from CLI input."""
    raw = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not raw:
        return ["openai", "huggingface"]
    allowed = {"openai", "huggingface", "none"}
    invalid = [provider for provider in raw if provider not in allowed]
    if invalid:
        raise ValueError(f"Unsupported providers: {', '.join(invalid)}")
    return raw


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer-id", default="C001")
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-purchase", action="store_true")
    parser.add_argument("--purchase-price", type=float, default=30000)
    parser.add_argument("--down-payment", type=float, default=5000)
    parser.add_argument("--apr-pct", type=float, default=8.0)
    parser.add_argument("--term-months", type=int, default=60)
    parser.add_argument(
        "--providers",
        default="openai,huggingface",
        help="Comma-separated: openai,huggingface,none",
    )
    args = parser.parse_args()

    if args.trials <= 0:
        raise ValueError("--trials must be > 0")
    if args.months <= 0:
        raise ValueError("--months must be > 0")

    config = MarkovConfig(
        trials=args.trials,
        months=args.months,
        seed=args.seed,
        include_purchase=args.include_purchase,
        purchase_price=args.purchase_price,
        down_payment=args.down_payment,
        apr_pct=args.apr_pct,
        term_months=args.term_months,
    )
    result = run_demo(customer_id=args.customer_id, config=config, providers=_parse_providers(args.providers))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
