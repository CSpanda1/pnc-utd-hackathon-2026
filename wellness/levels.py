"""Level 1 (single LLM) and Level 2 (tool-using agent)."""
import json
import math
import re

from .tools import make_tools, tool_descriptions

COACH = ("ROLE:COACH You are a friendly financial coach. Use ONLY the numbers in EVIDENCE. "
         "Never do your own arithmetic. Mention uncertainty. This is not financial advice.")


def level1(llm, customer, question):
    """Dump raw data into the prompt. Simple, but the model may miscalculate or hallucinate."""
    data = json.dumps({"accounts": customer["accounts"],
                       "income": customer["monthly_income"]})
    return llm.chat([{"role": "system", "content": COACH},
                     {"role": "user", "content": f"{question}\nEVIDENCE: {data}"}])


def parse_tool_call(text):
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def level2(llm, customer, question):
    """Route questions to approved tools and present computed evidence."""
    tools = make_tools(customer)
    # Common coaching requests should work even when a small local model
    # cannot produce a valid tool call.
    if _is_general_coaching(question):
        return _health_answer(tools["calculate_health_metrics"][0]())
    router = ("ROLE:ROUTER Pick ONE tool for the question. "
              "For general financial guidance or improving finances, choose calculate_health_metrics. "
              "get_accounts, get_profile, and calculate_health_metrics take no arguments: use {}. "
              "Only include documented argument names. Reply only with JSON "
              '{"tool": name, "args": {...}}. Tools:\n' + tool_descriptions(tools))
    try:
        call = parse_tool_call(llm.chat([{"role": "system", "content": router},
                                         {"role": "user", "content": question}]))
    except (ValueError, TypeError):
        return "I couldn't understand that request. Please rephrase it."
    if not isinstance(call, dict):
        return "I couldn't understand that request. Please rephrase it."

    if call.get("tool") not in tools:
        return "Sorry, I can't answer that with my approved tools."

    tool_name = call["tool"]
    args = call.get("args")
    # These tools already capture the selected customer; model-supplied
    # customer IDs, questions, or profile objects are unnecessary.
    if tool_name in {"get_accounts", "get_profile", "calculate_health_metrics"}:
        args = {}
    elif args is None:
        args = {}

    if not isinstance(args, dict):
        return "Sorry, I couldn't understand the tool arguments."

    if tool_name == "forecast_balances":
        try:
            months = float(args.get("months", 12))
            if not math.isfinite(months) or not months.is_integer() or not 1 <= months <= 120:
                raise ValueError("Invalid forecast horizon")
        except (ValueError, TypeError, OverflowError):
            return "Please choose a forecast length from 1 to 120 whole months."
        args = {"months": int(months)}

    # Handle missing optional car purchase parameters
    if tool_name == "simulate_car_purchase":
        defaults = {
            "price": 30000,
            "down": 5000,
            "apr": 0.07,
            "term_months": 60
        }

        for key, default in defaults.items():
            if args.get(key) is None:
                args[key] = default

        # Validate numeric inputs
        try:
            price = float(args["price"])
            down = float(args["down"])
            # An explicit percent in the question takes precedence over model units.
            rate = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|percent\b)", question, re.I)
            apr = float(rate.group(1)) / 100 if rate else float(args["apr"])
            if not rate and apr > 1:
                apr /= 100
            term_value = float(args["term_months"])
            if not all(math.isfinite(x) for x in (price, down, apr, term_value)):
                raise ValueError("Non-finite financing input")
            if not term_value.is_integer():
                raise ValueError("Loan term must be whole months")
            term = int(term_value)

            if not (0 <= down <= price and 0 <= apr <= 1 and term > 0):
                return "Please provide valid car financing details."

        except (ValueError, TypeError, OverflowError):
            return "I couldn't understand the car financing numbers."

        args = {
            "price": price,
            "down": down,
            "apr": apr,
            "term_months": term
        }

    try:
        result = tools[tool_name][0](**args)
    except (TypeError, ValueError, OverflowError):
        return "Please check the inputs for that request and try again."

    if tool_name == "calculate_health_metrics":
        return _health_answer(result)
    if tool_name == "simulate_car_purchase":
        return _car_answer(result, args)
    return llm.chat([{"role": "system", "content": COACH},
                     {"role": "user", "content": f"{question}\nEVIDENCE: {json.dumps(result)}"}])


def _is_general_coaching(question):
    q = question.lower()
    # Leave specific purchases, account requests, and forecasts to the router.
    if re.search(r"\b(car|buy|purchase|forecast|projection|accounts?)\b", q):
        return False
    return bool(re.search(
        r"financial health|financial wellness|smart about my (?:finances|money)|"
        r"(?:improve|manage|organize) my (?:finances|money)|"
        r"(?:financial|money) (?:advice|tips)|how am i doing financially",
        q,
    ))


def _health_answer(result):
    lines = ["Here is your financial health snapshot using the app's indicator thresholds."]
    for status, title in (("good", "Positive indicators"), ("watch", "Needs attention"),
                          ("risk", "Higher-risk indicators"), ("unknown", "Unavailable indicators")):
        entries = []
        for key, value in result["values"].items():
            if result["status"][key] != status:
                continue
            if value is None:
                display = "Unavailable"
            elif key == "net_worth":
                display = f"${value:,.2f}"
            elif key == "emergency_fund_months":
                display = f"{value:.2f} months"
            else:
                display = f"{value:.2%}"
            entries.append(f"- {key.replace('_', ' ').capitalize()}: {display} ({status})")
        if entries:
            lines.extend([f"\n**{title}**", *entries])
    actions = []
    status = result["status"]
    if status["emergency_fund_months"] in {"watch", "risk"}:
        actions.append("Review your monthly budget and choose an affordable recurring transfer to emergency savings.")
    if status["credit_utilization"] in {"watch", "risk"}:
        actions.append("Review your credit card balances and APRs to plan balance reductions while keeping required payments current.")
    if status["debt_to_income"] in {"watch", "risk"}:
        actions.append("Review existing monthly debt payments before taking on another loan; use the car simulator to compare scenarios.")
    if status["savings_rate"] in {"watch", "risk"}:
        actions.append("Review recurring spending for changes that could leave more income available each month.")
    if status["net_worth"] == "risk":
        actions.append("Check that the asset list is complete, especially any property associated with your loans.")
    if not actions:
        actions.append("Review any unavailable indicators, confirm your account data is complete, and use the forecast to check upcoming obligations.")
    lines.extend(["\n**Suggested next steps**", *[f"- {a}" for a in actions[:3]]])
    lines.append("\nThis snapshot covers recorded financial accounts only. Missing assets, such as a home or car, can understate net worth. A negative net worth alone does not establish a cash-flow shortfall.")
    return "\n".join(lines)


def _car_answer(result, args):
    lines = [f"Assuming a ${args['price']:,.2f} car, ${args['down']:,.2f} down, "
             f"{args['apr']:.2%} APR, and a {args['term_months']}-month loan "
             "(defaults used for any omitted details):",
             f"- Loan amount: ${result['loan_amount']:,.2f}",
             f"- Monthly payment: ${result['monthly_payment']:,.2f}",
             f"- Total interest: ${result['total_interest']:,.2f}",
             f"- Cash after down payment: ${result['cash_after_down_payment']:,.2f}",
             f"- Monthly cash flow after loan payment: ${result['new_monthly_net_cash_flow']:,.2f}"]
    coverage = result['emergency_fund_months_after']
    if coverage is not None:
        lines.append(f"- Emergency fund after purchase: {coverage:.2f} months")
    if result['cash_after_down_payment'] < 0:
        lines.append("Your recorded cash does not cover the down payment.")
    elif result['cash_goes_negative_within_12m']:
        lines.append("The projection shows a negative cash balance within 12 months.")
    else:
        lines.append("The projection stays nonnegative over 12 months, based on recorded income and spending.")
    if coverage is not None and coverage < 3:
        lines.append("This would leave less than three months of emergency coverage.")
    lines.append("Affordability remains uncertain: taxes, fees, insurance, maintenance, and changes in income or spending are excluded.")
    return "\n\n".join(lines)
