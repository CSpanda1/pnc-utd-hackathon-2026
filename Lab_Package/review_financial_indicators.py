"""Generate 100 portfolio-level financial indicators across customers.

This script computes aggregate metrics over the synthetic customer population.
It is for workshop observability/analysis only, not underwriting.
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from financial_tools import calculate_indicators, summarize_transaction_history

DATA = Path(__file__).parent / "data"


def _list_customer_ids(limit: int | None = None) -> list[str]:
    with (DATA / "profiles.csv").open(newline="", encoding="utf-8") as stream:
        rows = sorted({row["customer_id"] for row in csv.DictReader(stream)})
    return rows if limit is None else rows[:limit]


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * pct
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _build_metric_vectors(customer_ids: list[str]) -> dict[str, list[float]]:
    vectors: dict[str, list[float]] = {
        "monthly_income": [],
        "monthly_non_debt_expenses": [],
        "monthly_debt_payments": [],
        "monthly_cash_surplus": [],
        "liquid_assets": [],
        "emergency_fund_months": [],
        "debt_payment_to_income_pct": [],
        "credit_card_utilization_pct": [],
        "savings_rate_pct": [],
        "liquid_assets_to_debt_pct": [],
    }
    history_ratios: dict[str, list[float]] = {
        "spending_to_income_history_ratio": [],
        "debt_interest_share_of_spending": [],
        "external_transfer_dependency_ratio": [],
        "fees_to_income_ratio": [],
    }
    counters = {
        "negative_monthly_surplus": 0,
        "emergency_fund_lt1": 0,
        "emergency_fund_lt3": 0,
        "high_dti_gt40": 0,
        "high_card_util_gt70": 0,
        "negative_savings_rate": 0,
        "positive_monthly_surplus": 0,
        "positive_net_worth_partial": 0,
        "any_external_transfers": 0,
    }
    transaction_counts: list[float] = []
    for customer_id in customer_ids:
        ind = calculate_indicators(customer_id)
        history = summarize_transaction_history(customer_id)
        totals = history["totals"]
        for key in vectors:
            value = ind.get(key)
            if isinstance(value, (int, float)):
                vectors[key].append(float(value))

        if ind.get("monthly_cash_surplus", 0) < 0:
            counters["negative_monthly_surplus"] += 1
        if ind.get("monthly_cash_surplus", 0) > 0:
            counters["positive_monthly_surplus"] += 1
        if ind.get("net_worth_partial", 0) > 0:
            counters["positive_net_worth_partial"] += 1
        if isinstance(ind.get("emergency_fund_months"), (int, float)) and ind["emergency_fund_months"] < 1:
            counters["emergency_fund_lt1"] += 1
        if isinstance(ind.get("emergency_fund_months"), (int, float)) and ind["emergency_fund_months"] < 3:
            counters["emergency_fund_lt3"] += 1
        if isinstance(ind.get("debt_payment_to_income_pct"), (int, float)) and ind["debt_payment_to_income_pct"] > 40:
            counters["high_dti_gt40"] += 1
        if isinstance(ind.get("credit_card_utilization_pct"), (int, float)) and ind["credit_card_utilization_pct"] > 70:
            counters["high_card_util_gt70"] += 1
        if isinstance(ind.get("savings_rate_pct"), (int, float)) and ind["savings_rate_pct"] < 0:
            counters["negative_savings_rate"] += 1

        paycheck = float(totals["paycheck_income"])
        spending = float(totals["spending"])
        debt_interest = float(totals["debt_interest"])
        fees = float(totals["fees"])
        external_total = float(totals["external_transfers_in"]) + float(totals["external_transfers_out"])
        if external_total > 0:
            counters["any_external_transfers"] += 1
        history_ratios["spending_to_income_history_ratio"].append(_safe_ratio(spending, paycheck))
        history_ratios["debt_interest_share_of_spending"].append(_safe_ratio(debt_interest, spending))
        history_ratios["external_transfer_dependency_ratio"].append(_safe_ratio(external_total, paycheck))
        history_ratios["fees_to_income_ratio"].append(_safe_ratio(fees, paycheck))
        transaction_counts.append(float(history["transaction_count"]))
    return {"vectors": vectors, "history_ratios": history_ratios, "counters": counters, "transaction_counts": transaction_counts}


def _distribution_indicators(name: str, values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            f"{name}_mean": None,
            f"{name}_median": None,
            f"{name}_min": None,
            f"{name}_max": None,
            f"{name}_p10": None,
            f"{name}_p25": None,
            f"{name}_p75": None,
            f"{name}_p90": None,
        }
    return {
        f"{name}_mean": round(statistics.mean(values), 4),
        f"{name}_median": round(statistics.median(values), 4),
        f"{name}_min": round(min(values), 4),
        f"{name}_max": round(max(values), 4),
        f"{name}_p10": round(_percentile(values, 0.10), 4),
        f"{name}_p25": round(_percentile(values, 0.25), 4),
        f"{name}_p75": round(_percentile(values, 0.75), 4),
        f"{name}_p90": round(_percentile(values, 0.90), 4),
    }


def generate_100_indicators(limit_customers: int | None = None) -> dict:
    customer_ids = _list_customer_ids(limit_customers)
    if not customer_ids:
        raise ValueError("No customers found in profiles.csv")
    bundle = _build_metric_vectors(customer_ids)
    vectors = bundle["vectors"]
    history_ratios = bundle["history_ratios"]
    counters = bundle["counters"]
    transaction_counts = bundle["transaction_counts"]
    total = len(customer_ids)

    indicators: dict[str, float | int | None] = {}
    for metric_name, values in vectors.items():
        indicators.update(_distribution_indicators(metric_name, values))

    # 80 indicators from 10 metrics x 8 distribution stats.
    # Add 20 cross-customer portfolio indicators for a total of 100.
    indicators.update({
        "customer_count_reviewed": total,
        "pct_negative_monthly_surplus": round(100 * counters["negative_monthly_surplus"] / total, 4),
        "pct_positive_monthly_surplus": round(100 * counters["positive_monthly_surplus"] / total, 4),
        "pct_emergency_fund_lt1_month": round(100 * counters["emergency_fund_lt1"] / total, 4),
        "pct_emergency_fund_lt3_months": round(100 * counters["emergency_fund_lt3"] / total, 4),
        "pct_high_dti_gt40": round(100 * counters["high_dti_gt40"] / total, 4),
        "pct_high_card_util_gt70": round(100 * counters["high_card_util_gt70"] / total, 4),
        "pct_negative_savings_rate": round(100 * counters["negative_savings_rate"] / total, 4),
        "pct_positive_net_worth_partial": round(100 * counters["positive_net_worth_partial"] / total, 4),
        "pct_customers_with_external_transfers": round(100 * counters["any_external_transfers"] / total, 4),
        "avg_transaction_count_per_customer": round(statistics.mean(transaction_counts), 4),
        "median_transaction_count_per_customer": round(statistics.median(transaction_counts), 4),
        "avg_spending_to_income_history_ratio": round(statistics.mean(history_ratios["spending_to_income_history_ratio"]), 6),
        "median_spending_to_income_history_ratio": round(statistics.median(history_ratios["spending_to_income_history_ratio"]), 6),
        "avg_debt_interest_share_of_spending": round(statistics.mean(history_ratios["debt_interest_share_of_spending"]), 6),
        "median_debt_interest_share_of_spending": round(statistics.median(history_ratios["debt_interest_share_of_spending"]), 6),
        "avg_external_transfer_dependency_ratio": round(statistics.mean(history_ratios["external_transfer_dependency_ratio"]), 6),
        "median_external_transfer_dependency_ratio": round(statistics.median(history_ratios["external_transfer_dependency_ratio"]), 6),
        "avg_fees_to_income_ratio": round(statistics.mean(history_ratios["fees_to_income_ratio"]), 6),
        "median_fees_to_income_ratio": round(statistics.median(history_ratios["fees_to_income_ratio"]), 6),
    })
    if len(indicators) != 100:
        raise RuntimeError(f"Expected exactly 100 indicators, generated {len(indicators)}")

    return {
        "customer_ids_reviewed": customer_ids,
        "indicator_count": len(indicators),
        "indicators": indicators,
        "definitions": {
            "distribution_metrics": "mean/median/min/max/p10/p25/p75/p90 over customer-level indicators",
            "history_ratios": "ratios computed from synthetic transaction-history totals per customer",
            "usage_note": "For educational workshop analysis only; not lending, underwriting, or investment advice.",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-customers", type=int, default=None,
                        help="Optional cap on customer population before cross-customer aggregation.")
    args = parser.parse_args()
    report = generate_100_indicators(limit_customers=args.limit_customers)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
