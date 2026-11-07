import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag_embeddings_openai_demo import retrieve_with_embeddings, run_demo


class _FakeEmbeddings:
    def create(self, model, input):
        # Deterministic tiny vectors based on text statistics for test-only behavior.
        data = []
        for text in input:
            length = float(len(text))
            vowels = float(sum(ch.lower() in "aeiou" for ch in text))
            digits = float(sum(ch.isdigit() for ch in text))
            data.append(SimpleNamespace(embedding=[length, vowels, digits]))
        return SimpleNamespace(data=data)


class _FakeResponses:
    def create(self, model, input):
        return SimpleNamespace(output_text="Grounded summary from fake OpenAI response.")


class _FakeClient:
    def __init__(self):
        self.embeddings = _FakeEmbeddings()
        self.responses = _FakeResponses()


class RagEmbeddingsOpenAIDemoTests(unittest.TestCase):
    def test_retrieve_with_embeddings_returns_top_k(self):
        from rag_embeddings_openai_demo import Chunk

        client = _FakeClient()
        chunks = [
            Chunk(id="a", source="snapshot", text="monthly surplus and income details"),
            Chunk(id="b", source="history", text="fees and debt interest over months"),
            Chunk(id="c", source="policy", text="synthetic data guardrails"),
        ]
        top = retrieve_with_embeddings(client, query="surplus trend", chunks=chunks, embedding_model="fake", top_k=2)
        self.assertEqual(len(top), 2)
        self.assertGreaterEqual(top[0][1], top[1][1])

    def test_run_demo_shape(self):
        with patch("rag_embeddings_openai_demo.build_openai_client", return_value=_FakeClient()):
            result = run_demo(
                customer_id="C001",
                question="How does a vehicle purchase affect my resilience?",
                top_k=3,
                embedding_model="fake-embed-model",
                generation_model="fake-gen-model",
            )
        self.assertEqual(result["customer_id"], "C001")
        self.assertEqual(result["top_k"], 3)
        self.assertEqual(len(result["retrieval"]), 3)
        self.assertIn("answer", result)
        self.assertIn("Grounded summary", result["answer"])


if __name__ == "__main__":
    unittest.main()
