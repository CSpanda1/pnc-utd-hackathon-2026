"""Approved tools an agent may call. Each returns JSON-serializable data."""
from .forecast import forecast_cash
from .metrics import compute_metrics
from .profile import build_profile
from .scenarios import car_purchase


def make_tools(customer):
    prof = build_profile(customer)
    return {
        "get_accounts": (lambda: prof["accounts"], "List normalized accounts."),
        "get_profile": (lambda: {k: v for k, v in prof.items() if k != "accounts"},
                        "Unified totals: assets, liabilities, net worth."),
        "calculate_health_metrics": (lambda: compute_metrics(prof), "Health indicators + status."),
        "forecast_balances": (lambda months=12: forecast_cash(prof, int(months)),
                              "Project cash balance. args: months"),
        "simulate_car_purchase": (lambda price=30000, down=5000, apr=0.07, term_months=60:
                                  car_purchase(prof, float(price), float(down), float(apr), int(term_months)),
                                  "Car purchase what-if. args: price, down, apr, term_months"),
    }


def tool_descriptions(tools):
    return "\n".join(f"- {n}: {d}" for n, (_, d) in tools.items())
