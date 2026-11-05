"""LangChain + OpenAI + LangSmith observability workshop demo."""
import argparse
import json
import os
from typing import Literal

from financial_tools import calculate_indicators, get_history_evidence, simulate_purchase
from langsmith_observability import configure_langsmith


def route_question(question: str) -> Literal['indicators', 'scenario']:
    text = question.lower()
    wants_scenario = any(token in text for token in (
        'car', 'vehicle', 'purchase', 'loan', 'apr', 'term', 'down payment', 'what if'
    ))
    return 'scenario' if wants_scenario else 'indicators'


def build_tool_calling_components(model: str):
    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError('Install optional packages: python -m pip install langchain langchain-openai langsmith') from exc

    @tool
    def get_indicators(customer_id: str) -> dict:
        """Get deterministic synthetic household indicators for a customer."""
        return calculate_indicators(customer_id)

    @tool
    def get_history(customer_id: str) -> dict:
        """Get summarized synthetic transaction history evidence."""
        return get_history_evidence(customer_id)

    @tool
    def run_vehicle_scenario(customer_id: str, purchase_price: float, down_payment: float, apr_pct: float, term_months: int) -> dict:
        """Run a deterministic what-if vehicle purchase simulation."""
        return simulate_purchase(customer_id, purchase_price, down_payment, apr_pct, term_months)

    tools = [get_indicators, get_history, run_vehicle_scenario]
    tool_map = {t.name: t for t in tools}
    llm = ChatOpenAI(model=model, temperature=0.2).bind_tools(tools, tool_choice='auto')
    return llm, tool_map, HumanMessage, SystemMessage, ToolMessage, AIMessage


def run_tool_calling_inference(question: str, route: Literal['indicators', 'scenario'], customer_id: str,
                               assumptions: dict[str, float | int], model: str) -> tuple[str, dict]:
    llm, tool_map, HumanMessage, SystemMessage, ToolMessage, AIMessage = build_tool_calling_components(model)
    system = (
        'You are an educational financial wellness assistant. '
        'Always call tools to gather evidence before drafting an answer. '
        'Use only synthetic tool outputs; do not invent facts, guarantees, eligibility, or underwriting outcomes. '
        'Return exactly three concise grounded considerations.'
    )
    user = (
        f'Question: {question}\n'
        f'Route: {route}\n'
        f'Customer ID: {customer_id}\n'
        f'Scenario assumptions JSON: {json.dumps(assumptions)}\n'
        'First gather evidence via tools, then answer.'
    )
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    gathered: dict = {}
    run_config = {
        'run_name': 'workshop-langchain-openai-tool-loop',
        'tags': ['workshop', 'langchain', 'openai', 'tool-calling'],
        'metadata': {'customer_id': customer_id, 'route': route},
    }
    for _ in range(6):
        response = llm.invoke(messages, config=run_config)
        messages.append(response)
        if not isinstance(response, AIMessage) or not response.tool_calls:
            return str(response.content), gathered
        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_output = tool_map[tool_name].invoke(tool_call['args'], config=run_config)
            if tool_name == 'get_indicators':
                gathered['indicators'] = tool_output
            elif tool_name == 'get_history':
                gathered['transaction_history'] = tool_output
            elif tool_name == 'run_vehicle_scenario':
                gathered['scenario'] = tool_output
            messages.append(ToolMessage(content=json.dumps(tool_output), tool_call_id=tool_call['id']))
    from langchain_openai import ChatOpenAI
    final_llm = ChatOpenAI(model=model, temperature=0.2)
    final_response = final_llm.invoke(messages, config=run_config)
    return str(final_response.content), gathered


def run_demo(customer_id: str, question: str, purchase_price: float, down_payment: float,
             apr_pct: float, term_months: int, model: str, project: str | None = None) -> dict:
    ls = configure_langsmith(project)
    route = route_question(question)
    assumptions: dict[str, float | int] = {
        'purchase_price': purchase_price,
        'down_payment': down_payment,
        'apr_pct': apr_pct,
        'term_months': term_months,
    }
    answer, evidence = run_tool_calling_inference(question, route, customer_id, assumptions, model)
    return {
        'route': route,
        'customer_id': customer_id,
        'monthly_cash_surplus': (evidence.get('indicators') or {}).get('monthly_cash_surplus'),
        'scenario_payment': (evidence.get('scenario') or {}).get('new_monthly_payment'),
        'langsmith_project': ls['project'],
        'answer': answer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--customer-id', default='C001')
    parser.add_argument('--question', default='If I buy a $30,000 car with $5,000 down at 8% APR for 60 months, what changes?')
    parser.add_argument('--purchase-price', type=float, default=30000)
    parser.add_argument('--down-payment', type=float, default=5000)
    parser.add_argument('--apr-pct', type=float, default=8)
    parser.add_argument('--term-months', type=int, default=60)
    parser.add_argument('--model', default=os.getenv('OPENAI_MODEL', 'gpt-4o'))
    parser.add_argument('--langsmith-project', default=os.getenv('LANGSMITH_PROJECT', 'pnc-utd-workshop-observability'))
    args = parser.parse_args()

    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY before running this demo')

    result = run_demo(
        customer_id=args.customer_id,
        question=args.question,
        purchase_price=args.purchase_price,
        down_payment=args.down_payment,
        apr_pct=args.apr_pct,
        term_months=args.term_months,
        model=args.model,
        project=args.langsmith_project,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
