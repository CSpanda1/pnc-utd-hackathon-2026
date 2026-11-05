"""RAG-based scenario testing demo for the workshop.

This script demonstrates a lightweight Retrieval-Augmented Generation flow:
1) build a synthetic finance knowledge base for one customer
2) retrieve top-k evidence chunks for scenario questions
3) optionally call OpenAI to generate grounded answers
4) score retrieval coverage against expected keywords per test
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass

from financial_tools import calculate_indicators, get_accounts, summarize_transaction_history, simulate_purchase


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str


def _tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _tf(tokens: list[str]) -> Counter:
    return Counter(tokens)


def _idf(chunks: list[Chunk]) -> dict[str, float]:
    docs = [_tokenize(chunk.text) for chunk in chunks]
    doc_count = len(docs)
    df = Counter()
    for terms in docs:
        df.update(set(terms))
    return {term: math.log((1 + doc_count) / (1 + freq)) + 1.0 for term, freq in df.items()}


def _vector(counter: Counter, idf: dict[str, float]) -> dict[str, float]:
    if not counter:
        return {}
    norm_base = sum(counter.values())
    vec = {term: (count / norm_base) * idf.get(term, 0.0) for term, count in counter.items()}
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm == 0:
        return {}
    return {term: value / norm for term, value in vec.items()}


def _cosine(v1: dict[str, float], v2: dict[str, float]) -> float:
    if not v1 or not v2:
        return 0.0
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(value * v2.get(term, 0.0) for term, value in v1.items())


def build_customer_knowledge_base(customer_id: str) -> list[Chunk]:
    indicators = calculate_indicators(customer_id)
    accounts = get_accounts(customer_id)
    history = summarize_transaction_history(customer_id)
    chunks: list[Chunk] = []
    chunks.append(Chunk(
        id="policy-guardrails",
        source="policy",
        text=(
            "This is a synthetic financial wellness workshop dataset. "
            "Do not give underwriting or eligibility outcomes. "
            "Internal transfers and debt payments should not be double-counted as spending."
        ),
    ))
    chunks.append(Chunk(
        id="snapshot-indicators",
        source="snapshot",
        text=(
            f"Customer {customer_id} snapshot indicators: "
            f"monthly_income {indicators['monthly_income']}, "
            f"monthly_non_debt_expenses {indicators['monthly_non_debt_expenses']}, "
            f"monthly_debt_payments {indicators['monthly_debt_payments']}, "
            f"monthly_cash_surplus {indicators['monthly_cash_surplus']}, "
            f"liquid_assets {indicators['liquid_assets']}, "
            f"emergency_fund_months {indicators['emergency_fund_months']}, "
            f"debt_payment_to_income_pct {indicators['debt_payment_to_income_pct']}, "
            f"credit_card_utilization_pct {indicators['credit_card_utilization_pct']}."
        ),
    ))
    for idx, account in enumerate(accounts, start=1):
        chunks.append(Chunk(
            id=f"account-{idx}",
            source="accounts",
            text=(
                f"Account {idx}: type {account['account_type']}, "
                f"balance {account['balance']}, monthly_payment {account['monthly_payment']}, apr_pct {account['apr_pct']}."
            ),
        ))
    totals = history["totals"]
    chunks.append(Chunk(
        id="history-totals",
        source="history",
        text=(
            f"24-month history totals: paycheck_income {totals['paycheck_income']}, "
            f"spending {totals['spending']}, debt_payments {totals['debt_payments']}, "
            f"debt_interest {totals['debt_interest']}, fees {totals['fees']}, "
            f"external_transfers_in {totals['external_transfers_in']}, "
            f"external_transfers_out {totals['external_transfers_out']}."
        ),
    ))
    for idx, month in enumerate(history["months"][-6:], start=1):
        chunks.append(Chunk(
            id=f"recent-month-{idx}",
            source="history",
            text=(
                f"Recent month {month['month']}: paycheck_income {month['paycheck_income']}, "
                f"spending {month['spending']}, debt_payments {month['debt_payments']}, "
                f"debt_interest {month['debt_interest']}, fees {month['fees']}, "
                f"closing_liquid_assets {month['closing_liquid_assets']}."
            ),
        ))
    return chunks


def retrieve(query: str, chunks: list[Chunk], top_k: int = 4) -> list[tuple[Chunk, float]]:
    idf = _idf(chunks)
    query_vec = _vector(_tf(_tokenize(query)), idf)
    scored: list[tuple[Chunk, float]] = []
    for chunk in chunks:
        chunk_vec = _vector(_tf(_tokenize(chunk.text)), idf)
        scored.append((chunk, _cosine(query_vec, chunk_vec)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def generate_answer_with_openai(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Set OPENAI_API_KEY to use OpenAI generation")
    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    context = "\n\n".join([f"[{chunk.id}] {chunk.text}" for chunk, _ in retrieved])
    prompt = (
        "You are an educational financial wellness assistant. Use only the retrieved context.\n"
        "Do not invent facts or provide underwriting decisions.\n"
        "Question:\n"
        f"{question}\n\n"
        "Retrieved context:\n"
        f"{context}\n\n"
        "Return a concise scenario analysis in 4-6 bullet points."
    )
    response = OpenAI().responses.create(model=model, input=prompt)
    return response.output_text


def generate_answer_offline(question: str, retrieved: list[tuple[Chunk, float]]) -> str:
    lines = [f"- Retrieved [{chunk.id}] (score={score:.4f}) from {chunk.source}." for chunk, score in retrieved]
    return (
        "Offline RAG mode (no model call). Use retrieved evidence below for scenario reasoning:\n"
        f"Question: {question}\n" + "\n".join(lines)
    )


def scenario_test_cases() -> list[dict]:
    return [
        {
            "name": "vehicle-purchase-impact",
            "question": "If I buy a $30,000 car with $5,000 down at 8% APR for 60 months, what happens to surplus and reserves?",
            "expected_terms": ["monthly_cash_surplus", "liquid_assets", "debt_payments"],
            "scenario": {"purchase_price": 30000, "down_payment": 5000, "apr_pct": 8, "term_months": 60},
        },
        {
            "name": "high-spending-pressure",
            "question": "Do recent months show spending pressure relative to income and fees?",
            "expected_terms": ["spending", "paycheck_income", "fees"],
            "scenario": None,
        },
        {
            "name": "debt-burden-review",
            "question": "How elevated are debt payments and card utilization for this customer?",
            "expected_terms": ["debt_payment_to_income_pct", "credit_card_utilization_pct", "monthly_debt_payments"],
            "scenario": None,
        },
    ]


def _coverage_score(retrieved_text: str, expected_terms: list[str]) -> dict:
    lower = retrieved_text.lower()
    hits = [term for term in expected_terms if term.lower() in lower]
    return {
        "matched_terms": hits,
        "expected_terms": expected_terms,
        "coverage_ratio": round(len(hits) / len(expected_terms), 4) if expected_terms else 1.0,
    }


def run_rag_scenario_testing(customer_id: str, top_k: int = 4, provider: str = "none") -> dict:
    chunks = build_customer_knowledge_base(customer_id)
    report = []
    for case in scenario_test_cases():
        retrieved = retrieve(case["question"], chunks, top_k=top_k)
        retrieved_text = "\n".join(chunk.text for chunk, _ in retrieved)
        coverage = _coverage_score(retrieved_text, case["expected_terms"])
        scenario_output = None
        if case["scenario"] is not None:
            scenario_output = simulate_purchase(customer_id, **case["scenario"])
        if provider == "openai":
            answer = generate_answer_with_openai(case["question"], retrieved)
        else:
            answer = generate_answer_offline(case["question"], retrieved)
        report.append(
            {
                "name": case["name"],
                "question": case["question"],
                "retrieval": [
                    {"chunk_id": chunk.id, "source": chunk.source, "score": round(score, 6)}
                    for chunk, score in retrieved
                ],
                "coverage": coverage,
                "scenario_output": scenario_output,
                "answer": answer,
            }
        )
    avg_coverage = sum(item["coverage"]["coverage_ratio"] for item in report) / len(report)
    return {
        "customer_id": customer_id,
        "provider": provider,
        "knowledge_chunks": len(chunks),
        "top_k": top_k,
        "tests_run": len(report),
        "avg_retrieval_coverage": round(avg_coverage, 4),
        "results": report,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer-id", default="C001")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--provider", choices=["none", "openai"], default="none")
    args = parser.parse_args()
    result = run_rag_scenario_testing(customer_id=args.customer_id, top_k=args.top_k, provider=args.provider)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
