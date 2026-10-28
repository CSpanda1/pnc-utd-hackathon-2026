"""Validation & safety layer: numeric checks, missing data, evidence."""
import math


def _numbers(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield obj


def validate(tool, result):
    """Return (ok, issues)."""
    issues = []
    if result in (None, {}, []):
        issues.append("empty result")
    for n in _numbers(result):
        if math.isnan(n) or math.isinf(n):
            issues.append("non-finite number")
            break
    if isinstance(result, dict):
        if tool == "simulate_car_purchase":
            if result.get("loan_amount", 0) < 0:
                issues.append("down payment exceeds price")
        vals = result.get("values")
        if vals and any(v is None for v in vals.values()):
            issues.append("some metrics missing data")
    return (not issues, issues)
