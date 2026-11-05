"""LangGraph + OpenAI + LangSmith observability workshop demo."""
import argparse
import json
import os
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from financial_tools import calculate_indicators, get_history_evidence, simulate_purchase
from langsmith_observability import configure_langsmith


class WorkshopState(TypedDict, total=False):
    customer_id: str
    question: str
    route: Literal['indicators', 'scenario']
    scenario_assumptions: dict[str, float | int]
    indicators: dict
    scenario: dict
    history_evidence: dict
    answer: str
    trace: list[str]
    model: str
    langsmith_project: str


def _append_trace(state: WorkshopState, marker: str) -> list[str]:
    return [*state.get('trace', []), marker]


def route_question(state: WorkshopState) -> WorkshopState:
    question = state['question'].lower()
    wants_scenario = any(token in question for token in (
        'car', 'vehicle', 'purchase', 'loan', 'apr', 'term', 'down payment', 'what if'
    ))
    route: Literal['indicators', 'scenario'] = 'scenario' if wants_scenario else 'indicators'
    return {'route': route, 'trace': _append_trace(state, f'router:{route}')}


def indicator_node(state: WorkshopState) -> WorkshopState:
    return {'indicators': calculate_indicators(state['customer_id']),
            'trace': _append_trace(state, 'indicators:calculated')}


def scenario_node(state: WorkshopState) -> WorkshopState:
    assumptions = state.get('scenario_assumptions', {
        'purchase_price': 30000,
        'down_payment': 5000,
        'apr_pct': 8,
        'term_months': 60,
    })
    return {
        'scenario': simulate_purchase(
            state['customer_id'],
            assumptions['purchase_price'],
            assumptions['down_payment'],
            assumptions['apr_pct'],
            assumptions['term_months'],
        ),
        'trace': _append_trace(state, 'scenario:simulated'),
    }


def history_node(state: WorkshopState) -> WorkshopState:
    return {'history_evidence': get_history_evidence(state['customer_id']),
            'trace': _append_trace(state, 'history:loaded')}


def narration_node(state: WorkshopState) -> WorkshopState:
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
    llm = ChatOpenAI(model=state['model'], temperature=0.2).bind_tools(tools, tool_choice='auto')
    assumptions = state.get('scenario_assumptions') or {
        'purchase_price': 30000,
        'down_payment': 5000,
        'apr_pct': 8,
        'term_months': 60,
    }
    messages = [
        SystemMessage(content=(
            'You are an educational financial wellness assistant. '
            'Always call tools to gather evidence before drafting an answer. '
            'Use only synthetic tool outputs; do not invent facts, guarantees, underwriting outcomes, or eligibility claims. '
            'Return exactly three concise grounded considerations.'
        )),
        HumanMessage(content=(
            f"Question: {state['question']}\n"
            f"Route: {state['route']}\n"
            f"Customer ID: {state['customer_id']}\n"
            f"Scenario assumptions JSON: {json.dumps(assumptions)}\n"
            'First gather evidence via tools, then answer.'
        )),
    ]
    run_config = {
        'run_name': 'workshop-langgraph-node-narration',
        'tags': ['workshop', 'langgraph', 'openai', 'tool-calling'],
        'metadata': {'customer_id': state['customer_id'], 'route': state['route'],
                     'langsmith_project': state['langsmith_project']},
    }
    answer = ''
    for _ in range(6):
        response = llm.invoke(messages, config=run_config)
        messages.append(response)
        if not isinstance(response, AIMessage) or not response.tool_calls:
            answer = str(response.content)
            break
        for tool_call in response.tool_calls:
            tool_output = tool_map[tool_call['name']].invoke(tool_call['args'], config=run_config)
            messages.append(ToolMessage(content=json.dumps(tool_output), tool_call_id=tool_call['id']))
    if not answer:
        # Fallback: synthesize from collected tool outputs without further tool calls.
        final_llm = ChatOpenAI(model=state['model'], temperature=0.2)
        answer = str(final_llm.invoke(messages, config=run_config).content)
    return {'answer': answer, 'trace': _append_trace(state, 'narration:openai')}


def _route_selector(state: WorkshopState) -> Literal['indicators', 'scenario']:
    return state['route']


def build_graph():
    builder = StateGraph(WorkshopState)
    builder.add_node('router', route_question)
    builder.add_node('indicators', indicator_node)
    builder.add_node('scenario', scenario_node)
    builder.add_node('history', history_node)
    builder.add_node('narration', narration_node)
    builder.add_edge(START, 'router')
    builder.add_conditional_edges('router', _route_selector, {'indicators': 'indicators', 'scenario': 'scenario'})
    builder.add_edge('indicators', 'history')
    builder.add_edge('scenario', 'history')
    builder.add_edge('history', 'narration')
    builder.add_edge('narration', END)
    return builder.compile()


def run_demo(customer_id: str, question: str, purchase_price: float, down_payment: float,
             apr_pct: float, term_months: int, model: str, project: str | None = None):
    ls = configure_langsmith(project)
    graph = build_graph()
    result = graph.invoke(
        {
            'customer_id': customer_id,
            'question': question,
            'model': model,
            'langsmith_project': ls['project'],
            'scenario_assumptions': {
                'purchase_price': purchase_price,
                'down_payment': down_payment,
                'apr_pct': apr_pct,
                'term_months': term_months,
            },
            'trace': [],
        },
        config={'run_name': 'workshop-langgraph-openai', 'tags': ['workshop', 'langgraph', 'openai'],
                'metadata': {'customer_id': customer_id, 'langsmith_project': ls['project']}},
    )
    return {
        'route': result['route'],
        'trace': result.get('trace', []),
        'customer_id': result['customer_id'],
        'monthly_cash_surplus': result.get('indicators', {}).get('monthly_cash_surplus'),
        'scenario_payment': result.get('scenario', {}).get('new_monthly_payment'),
        'langsmith_project': ls['project'],
        'answer': result['answer'],
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
