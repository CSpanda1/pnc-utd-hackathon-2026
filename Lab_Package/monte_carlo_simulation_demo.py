"""Monte Carlo scenario demo with optional OpenAI and Hugging Face narration.

This script runs a synthetic cashflow Monte Carlo simulation for one customer,
then asks one or more LLM providers to explain the risk profile.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from statistics import mean

from financial_tools import calculate_indicators, simulate_purchase


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for one Monte Carlo experiment."""
    trials: int = 1000
    months: int = 24
    seed: int = 42
    income_volatility: float = 0.08
    expense_volatility: float = 0.10
    include_purchase: bool = False
    purchase_price: float = 30000
    down_payment: float = 5000
    apr_pct: float = 8.0
    term_months: int = 60


def _percentile(values: list[float], pct: float) -> float:
    """Return a linearly interpolated percentile for a numeric sample."""
    if not values:
        return 0.0
    if pct <= 0:
        return float(min(values))
    if pct >= 100:
        return float(max(values))
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def run_monte_carlo(customer_id: str, config: MonteCarloConfig) -> dict:
    """Simulate monthly cashflow paths and summarize downside/central outcomes."""
    indicators = calculate_indicators(customer_id)
    base_income = float(indicators["monthly_income"])
    base_expenses = float(indicators["monthly_non_debt_expenses"])
    base_debt = float(indicators["monthly_debt_payments"])
    starting_liquid = float(indicators["liquid_assets"])

    scenario_payment = 0.0
    if config.include_purchase:
        # Inject optional loan payment shock into each simulated month.
        scenario = simulate_purchase(
            customer_id=customer_id,
            purchase_price=config.purchase_price,
            down_payment=config.down_payment,
            apr_pct=config.apr_pct,
            term_months=config.term_months,
        )
        scenario_payment = float(scenario["new_monthly_payment"])

    rng = random.Random(config.seed)
    end_liquid_assets: list[float] = []
    avg_monthly_surplus_per_trial: list[float] = []
    any_negative_month_count = 0
    liquid_below_zero_count = 0

    for _ in range(config.trials):
        # Down payment is applied once at simulation start, then monthly cashflow evolves stochastically.
        liquid = starting_liquid - (config.down_payment if config.include_purchase else 0.0)
        monthly_surplus_values: list[float] = []
        had_negative_month = False
        had_liquid_below_zero = liquid < 0

        for _month in range(config.months):
            # Income and expenses are shocked independently around the baseline with Gaussian noise.
            income = base_income * max(0.0, 1.0 + rng.gauss(0.0, config.income_volatility))
            expenses = base_expenses * max(0.0, 1.0 + rng.gauss(0.0, config.expense_volatility))
            debt = base_debt + scenario_payment
            surplus = income - expenses - debt
            monthly_surplus_values.append(surplus)
            liquid += surplus
            had_negative_month = had_negative_month or surplus < 0
            had_liquid_below_zero = had_liquid_below_zero or liquid < 0

        if had_negative_month:
            any_negative_month_count += 1
        if had_liquid_below_zero:
            liquid_below_zero_count += 1

        end_liquid_assets.append(liquid)
        avg_monthly_surplus_per_trial.append(mean(monthly_surplus_values))

    return {
        "customer_id": customer_id,
        "trials": config.trials,
        "months": config.months,
        "seed": config.seed,
        "assumptions": {
            "income_volatility": config.income_volatility,
            "expense_volatility": config.expense_volatility,
            "include_purchase": config.include_purchase,
            "scenario_payment": round(scenario_payment, 2),
        },
        "baseline": {
            "monthly_income": round(base_income, 2),
            "monthly_non_debt_expenses": round(base_expenses, 2),
            "monthly_debt_payments": round(base_debt, 2),
            "starting_liquid_assets": round(starting_liquid, 2),
        },
        "results": {
            "end_liquid_p10": round(_percentile(end_liquid_assets, 10), 2),
            "end_liquid_p50": round(_percentile(end_liquid_assets, 50), 2),
            "end_liquid_p90": round(_percentile(end_liquid_assets, 90), 2),
            "avg_monthly_surplus_p10": round(_percentile(avg_monthly_surplus_per_trial, 10), 2),
            "avg_monthly_surplus_p50": round(_percentile(avg_monthly_surplus_per_trial, 50), 2),
            "avg_monthly_surplus_p90": round(_percentile(avg_monthly_surplus_per_trial, 90), 2),
            "prob_any_negative_surplus_month": round(any_negative_month_count / config.trials, 4),
            "prob_liquid_below_zero": round(liquid_below_zero_count / config.trials, 4),
        },
    }


def _build_prompt(simulation: dict) -> str:
    """Build a provider-agnostic narration prompt from simulation JSON."""
    return (
        "You are an educational financial wellness assistant.\n"
        "Use ONLY the simulation JSON below. Do not invent facts or provide lending/underwriting advice.\n"
        "Explain what the Monte Carlo results imply for resilience, downside risk, and budgeting actions.\n"
        "Return 5 concise bullet points and include one explicit caveat about synthetic data assumptions.\n\n"
        f"SIMULATION:\n{json.dumps(simulation, indent=2)}"
    )


def generate_with_openai(prompt: str) -> str:
    """Generate narration via OpenAI Responses API."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY to run the OpenAI narration.")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    response = OpenAI(api_key=api_key).responses.create(model=model, input=prompt)
    return response.output_text


def generate_with_huggingface(prompt: str) -> str:
    """Generate narration via local Hugging Face model wrapper."""
    from local_hf import generate_local

    return generate_local(prompt)


def run_demo(customer_id: str, config: MonteCarloConfig, providers: list[str]) -> dict:
    """Run simulation once, then request narration from selected providers."""
    simulation = run_monte_carlo(customer_id, config)
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
            # Keep simulation output even if a provider is unavailable or misconfigured.
            narrations[provider] = {"status": "error", "error": str(exc)}

    return {"simulation": simulation, "narrations": narrations}


def _parse_providers(value: str) -> list[str]:
    """Parse and validate comma-separated providers from CLI input."""
    raw = [part.strip().lower() for part in value.split(",") if part.strip()]
    if not raw:
        return ["openai", "huggingface"]
    allowed = {"openai", "huggingface", "none"}
    bad = [provider for provider in raw if provider not in allowed]
    if bad:
        raise ValueError(f"Unsupported providers: {', '.join(bad)}")
    return raw


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer-id", default="C001")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--months", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--income-volatility", type=float, default=0.08)
    parser.add_argument("--expense-volatility", type=float, default=0.10)
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

    config = MonteCarloConfig(
        trials=args.trials,
        months=args.months,
        seed=args.seed,
        income_volatility=args.income_volatility,
        expense_volatility=args.expense_volatility,
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
