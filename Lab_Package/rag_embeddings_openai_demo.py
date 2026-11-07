"""RAG demo using OpenAI embeddings + OpenAI generation.

This script shows a compact end-to-end flow:
1) build a synthetic customer knowledge base
2) embed chunks with OpenAI embeddings
3) retrieve top-k chunks by cosine similarity
4) generate a grounded answer with OpenAI Responses API
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any

from financial_tools import calculate_indicators, get_accounts, summarize_transaction_history, simulate_purchase


@dataclass(frozen=True)
class Chunk:
    """Retrievable evidence unit for RAG."""

    id: str
    source: str
    text: str


def build_customer_knowledge_base(customer_id: str) -> list[Chunk]:
    """Build a small retrieval corpus from deterministic customer evidence."""

    indicators = calculate_indicators(customer_id)
    accounts = get_accounts(customer_id)
    history = summarize_transaction_history(customer_id)

    chunks: list[Chunk] = [
        Chunk(
            id="policy-guardrails",
            source="policy",
            text=(
                "This is synthetic educational data. Do not provide underwriting decisions or eligibility outcomes. "
                "Do not double-count debt payments or internal transfers as additional spending."
            ),
        ),
        Chunk(
            id="snapshot-indicators",
            source="snapshot",
            text=(
                f"Customer {customer_id} indicators: monthly_income {indicators['monthly_income']}, "
                f"monthly_non_debt_expenses {indicators['monthly_non_debt_expenses']}, "
                f"monthly_debt_payments {indicators['monthly_debt_payments']}, "
                f"monthly_cash_surplus {indicators['monthly_cash_surplus']}, liquid_assets {indicators['liquid_assets']}, "
                f"emergency_fund_months {indicators['emergency_fund_months']}, "
                f"debt_payment_to_income_pct {indicators['debt_payment_to_income_pct']}, "
                f"credit_card_utilization_pct {indicators['credit_card_utilization_pct']}."
            ),
        ),
    ]

    for idx, account in enumerate(accounts, start=1):
        chunks.append(
            Chunk(
                id=f"account-{idx}",
                source="accounts",
                text=(
                    f"Account {idx}: type {account['account_type']}, balance {account['balance']}, "
                    f"monthly_payment {account['monthly_payment']}, apr_pct {account['apr_pct']}."
                ),
            )
        )

    totals = history["totals"]
    chunks.append(
        Chunk(
            id="history-totals",
            source="history",
            text=(
                f"24-month totals: paycheck_income {totals['paycheck_income']}, spending {totals['spending']}, "
                f"debt_payments {totals['debt_payments']}, debt_interest {totals['debt_interest']}, fees {totals['fees']}."
            ),
        )
    )
    for idx, month in enumerate(history["months"][-6:], start=1):
        chunks.append(
            Chunk(
                id=f"recent-month-{idx}",
                source="history",
                text=(
                    f"Month {month['month']}: paycheck_income {month['paycheck_income']}, spending {month['spending']}, "
                    f"debt_payments {month['debt_payments']}, fees {month['fees']}, "
                    f"closing_liquid_assets {month['closing_liquid_assets']}."
                ),
            )
        )

    return chunks


def build_openai_client() -> Any:
    """Create OpenAI client and fail fast when key is missing."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running this demo.")
    from openai import OpenAI

    return OpenAI(api_key=api_key)


def _l2_normalize(vector: list[float]) -> list[float]:
    """Normalize vector to unit length for cosine similarity."""

    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [v / norm for v in vector]


def embed_texts(client: Any, texts: list[str], model: str) -> list[list[float]]:
    """Request embeddings for a batch of texts using OpenAI embeddings API."""

    response = client.embeddings.create(model=model, input=texts)
    # OpenAI returns one embedding object per input in the same order.
    return [_l2_normalize(item.embedding) for item in response.data]


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity (dot product after normalization)."""

    if len(v1) != len(v2):
        raise ValueError("Embedding dimension mismatch")
    return sum(a * b for a, b in zip(v1, v2))


def retrieve_with_embeddings(
    client: Any,
    query: str,
    chunks: list[Chunk],
    embedding_model: str,
    top_k: int = 4,
) -> list[tuple[Chunk, float]]:
    """Embed query/chunks, then return top-k chunks by cosine similarity."""

    chunk_vectors = embed_texts(client, [chunk.text for chunk in chunks], embedding_model)
    query_vector = embed_texts(client, [query], embedding_model)[0]

    scored = [(chunk, cosine_similarity(query_vector, chunk_vec)) for chunk, chunk_vec in zip(chunks, chunk_vectors)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def generate_grounded_answer(
    client: Any,
    question: str,
    retrieved: list[tuple[Chunk, float]],
    generation_model: str,
) -> str:
    """Generate answer constrained to retrieved evidence."""

    context = "\n\n".join([f"[{chunk.id}] {chunk.text}" for chunk, _ in retrieved])
    prompt = (
        "You are an educational financial wellness assistant.\n"
        "Use only the retrieved context below. Do not invent facts.\n"
        "Do not give underwriting or lending eligibility decisions.\n"
        "Return 4-6 concise bullet points.\n\n"
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context}"
    )
    response = client.responses.create(model=generation_model, input=prompt)
    return response.output_text


def run_demo(
    customer_id: str,
    question: str,
    top_k: int,
    embedding_model: str,
    generation_model: str,
) -> dict:
    """Run full embeddings-based RAG pipeline and return structured output."""

    client = build_openai_client()
    chunks = build_customer_knowledge_base(customer_id)
    retrieved = retrieve_with_embeddings(client, question, chunks, embedding_model, top_k=top_k)

    # Optional deterministic tool output can be included to contrast with the model narrative.
    scenario = simulate_purchase(customer_id, purchase_price=30000, down_payment=5000, apr_pct=8.0, term_months=60)
    answer = generate_grounded_answer(client, question, retrieved, generation_model)

    return {
        "customer_id": customer_id,
        "question": question,
        "embedding_model": embedding_model,
        "generation_model": generation_model,
        "top_k": top_k,
        "knowledge_chunks": len(chunks),
        "retrieval": [
            {"chunk_id": chunk.id, "source": chunk.source, "score": round(score, 6)} for chunk, score in retrieved
        ],
        "scenario_example": {
            "new_monthly_payment": scenario["new_monthly_payment"],
            "projected_monthly_surplus": scenario["projected_monthly_surplus"],
        },
        "answer": answer,
    }


def main() -> None:
    """CLI entrypoint for running embeddings-based RAG demo."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--customer-id", default="C001")
    parser.add_argument(
        "--question",
        default="If I buy a $30,000 car with $5,000 down, what changes in my monthly resilience?",
    )
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--embedding-model", default=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--generation-model", default=os.getenv("OPENAI_MODEL", "gpt-4o"))
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")

    result = run_demo(
        customer_id=args.customer_id,
        question=args.question,
        top_k=args.top_k,
        embedding_model=args.embedding_model,
        generation_model=args.generation_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
