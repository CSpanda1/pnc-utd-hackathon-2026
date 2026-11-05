"""LangGraph + OpenAI workshop demo with grounded synthetic financial evidence.

This script demonstrates a small multi-node graph:
1) route the participant question to indicators or scenario analysis
2) collect deterministic financial evidence from financial_tools.py
3) ask the OpenAI model for a concise educational explanation
"""
import argparse
import json
import os
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from financial_tools import calculate_indicators, get_history_evidence, simulate_purchase
from llm_demo import narrate


class WorkshopState(TypedDict, total=False):
    """Shared LangGraph state passed between nodes.

    Required inputs:
    - customer_id: synthetic customer key (for example C001)
    - question: participant prompt used for routing and narration

    Optional fields are populated by downstream nodes.
    """
    customer_id: str
    question: str
    route: Literal['indicators', 'scenario']
    scenario_assumptions: dict[str, float | int]
    indicators: dict
    scenario: dict
    history_evidence: dict
    answer: str
    trace: list[str]


def _append_trace(state: WorkshopState, marker: str) -> list[str]:
    """Append a trace marker while preserving existing trace entries."""
    return [*state.get('trace', []), marker]


def route_question(state: WorkshopState) -> WorkshopState:
    """Route classroom prompts to either baseline indicators or scenario simulation."""
    question = state['question'].lower()
    wants_scenario = any(token in question for token in (
        'car', 'vehicle', 'purchase', 'loan', 'apr', 'term', 'down payment', 'what if'
    ))
    route: Literal['indicators', 'scenario'] = 'scenario' if wants_scenario else 'indicators'
    return {'route': route, 'trace': _append_trace(state, f'router:{route}')}


def indicator_node(state: WorkshopState) -> WorkshopState:
    """Calculate deterministic point-in-time indicators from snapshot data."""
    return {
        'indicators': calculate_indicators(state['customer_id']),
        'trace': _append_trace(state, 'indicators:calculated'),
    }


def scenario_node(state: WorkshopState) -> WorkshopState:
    """Run the what-if purchase simulation from supplied assumptions."""
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
    """Load summarized transaction history evidence for grounded narration."""
    return {
        'history_evidence': get_history_evidence(state['customer_id']),
        'trace': _append_trace(state, 'history:loaded'),
    }


def narration_node(state: WorkshopState) -> WorkshopState:
    """Generate the final OpenAI explanation from validated evidence only."""
    evidence = {
        'workshop_question': state['question'],
        'indicators': state.get('indicators') or calculate_indicators(state['customer_id']),
        'scenario': state.get('scenario'),
        'transaction_history': state['history_evidence'],
    }
    answer = narrate(evidence, 'openai')
    return {'answer': answer, 'trace': _append_trace(state, 'narration:openai')}


def _route_selector(state: WorkshopState) -> Literal['indicators', 'scenario']:
    """Conditional edge selector used by LangGraph after router node."""
    return state['route']


def build_graph():
    """Build and compile the workshop graph topology."""
    builder = StateGraph(WorkshopState)
    builder.add_node('router', route_question)
    builder.add_node('indicators', indicator_node)
    builder.add_node('scenario', scenario_node)
    builder.add_node('history', history_node)
    builder.add_node('narration', narration_node)
    builder.add_edge(START, 'router')
    builder.add_conditional_edges('router', _route_selector, {
        'indicators': 'indicators',
        'scenario': 'scenario',
    })
    # Both routes converge on history evidence before narration.
    builder.add_edge('indicators', 'history')
    builder.add_edge('scenario', 'history')
    builder.add_edge('history', 'narration')
    builder.add_edge('narration', END)
    return builder.compile()


def run_demo(customer_id: str, question: str, purchase_price: float, down_payment: float, apr_pct: float, term_months: int):
    """Execute one end-to-end inference pass for a workshop prompt."""
    graph = build_graph()
    initial_state: WorkshopState = {
        'customer_id': customer_id,
        'question': question,
        'scenario_assumptions': {
            'purchase_price': purchase_price,
            'down_payment': down_payment,
            'apr_pct': apr_pct,
            'term_months': term_months,
        },
        'trace': [],
    }
    return graph.invoke(initial_state)


def main():
    """CLI entry point for workshop demonstrations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--customer-id', default='C001')
    parser.add_argument('--question', default='If I buy a $30,000 car with $5,000 down at 8% APR for 60 months, what changes?')
    parser.add_argument('--purchase-price', type=float, default=30000)
    parser.add_argument('--down-payment', type=float, default=5000)
    parser.add_argument('--apr-pct', type=float, default=8)
    parser.add_argument('--term-months', type=int, default=60)
    args = parser.parse_args()

    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY before running the LangGraph OpenAI demo')

    result = run_demo(
        customer_id=args.customer_id,
        question=args.question,
        purchase_price=args.purchase_price,
        down_payment=args.down_payment,
        apr_pct=args.apr_pct,
        term_months=args.term_months,
    )
    summary = {
        'route': result['route'],
        'trace': result.get('trace', []),
        'customer_id': result['customer_id'],
        'monthly_cash_surplus': result.get('indicators', {}).get('monthly_cash_surplus'),
        'scenario_payment': result.get('scenario', {}).get('new_monthly_payment'),
        'answer': result['answer'],
    }
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
