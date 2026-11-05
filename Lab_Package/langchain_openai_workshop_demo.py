"""LangChain + OpenAI workshop demo with grounded synthetic financial evidence.

This example mirrors the LangGraph demo intent, but uses a LangChain prompt+LLM
pipeline (LCEL) for inference:
1) route the participant question to indicators or scenario analysis
2) gather deterministic evidence from financial_tools.py
3) invoke an OpenAI model through LangChain for concise narration
"""
import argparse
import json
import os
from typing import Literal, TypedDict

from financial_tools import calculate_indicators, get_history_evidence, simulate_purchase


class WorkshopResult(TypedDict):
    route: Literal['indicators', 'scenario']
    trace: list[str]
    customer_id: str
    monthly_cash_surplus: float | None
    scenario_payment: float | None
    answer: str


def route_question(question: str) -> Literal['indicators', 'scenario']:
    """Route prompts to scenario mode when they ask for a what-if purchase analysis."""
    text = question.lower()
    wants_scenario = any(token in text for token in (
        'car', 'vehicle', 'purchase', 'loan', 'apr', 'term', 'down payment', 'what if'
    ))
    return 'scenario' if wants_scenario else 'indicators'


def gather_evidence(customer_id: str, route: Literal['indicators', 'scenario'],
                    assumptions: dict[str, float | int]) -> dict:
    """Collect validated tool evidence for the selected route."""
    indicators = calculate_indicators(customer_id)
    scenario = simulate_purchase(
        customer_id,
        assumptions['purchase_price'],
        assumptions['down_payment'],
        assumptions['apr_pct'],
        assumptions['term_months'],
    ) if route == 'scenario' else None
    history = get_history_evidence(customer_id)
    return {
        'indicators': indicators,
        'scenario': scenario,
        'transaction_history': history,
    }


def build_langchain_chain(model: str | None = None):
    """Build the LangChain prompt -> OpenAI chat model -> string output pipeline."""
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError('Install optional packages: python -m pip install langchain langchain-openai') from exc

    prompt = ChatPromptTemplate.from_messages([
        ('system',
         'You are an educational financial wellness assistant. Use ONLY provided synthetic evidence. '
         'Do not invent facts, guarantees, eligibility, credit decisions, or underwriting outcomes. '
         'Provide three concise grounded considerations and distinguish assumptions from observations.'),
        ('human',
         'Participant question: {question}\n'
         'Route: {route}\n'
         'Evidence JSON:\n{evidence_json}\n'
         'Return a compact workshop-friendly explanation.')
    ])
    llm = ChatOpenAI(model=model or os.getenv('OPENAI_MODEL', 'gpt-4o'), temperature=0.2)
    return prompt | llm | StrOutputParser()


def run_demo(customer_id: str, question: str, purchase_price: float, down_payment: float,
             apr_pct: float, term_months: int, model: str | None = None) -> WorkshopResult:
    """Execute one end-to-end LangChain inference pass for a workshop prompt."""
    route = route_question(question)
    trace = [f'router:{route}']
    assumptions: dict[str, float | int] = {
        'purchase_price': purchase_price,
        'down_payment': down_payment,
        'apr_pct': apr_pct,
        'term_months': term_months,
    }
    evidence = gather_evidence(customer_id, route, assumptions)
    trace.append('evidence:collected')
    chain = build_langchain_chain(model=model)
    answer = chain.invoke({
        'question': question,
        'route': route,
        'evidence_json': json.dumps(evidence),
    })
    trace.append('narration:langchain-openai')
    return {
        'route': route,
        'trace': trace,
        'customer_id': customer_id,
        'monthly_cash_surplus': evidence['indicators'].get('monthly_cash_surplus'),
        'scenario_payment': (evidence['scenario'] or {}).get('new_monthly_payment'),
        'answer': answer,
    }


def main():
    """CLI entry point for workshop demonstrations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--customer-id', default='C001')
    parser.add_argument('--question', default='If I buy a $30,000 car with $5,000 down at 8% APR for 60 months, what changes?')
    parser.add_argument('--purchase-price', type=float, default=30000)
    parser.add_argument('--down-payment', type=float, default=5000)
    parser.add_argument('--apr-pct', type=float, default=8)
    parser.add_argument('--term-months', type=int, default=60)
    parser.add_argument('--model', default=None)
    args = parser.parse_args()

    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('Set OPENAI_API_KEY before running the LangChain OpenAI demo')

    result = run_demo(
        customer_id=args.customer_id,
        question=args.question,
        purchase_price=args.purchase_price,
        down_payment=args.down_payment,
        apr_pct=args.apr_pct,
        term_months=args.term_months,
        model=args.model,
    )
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
