"""LLM providers: offline mock, local open-weight (Ollama), commercial (OpenAI-compatible)."""
import json
import os
import re
import urllib.request


def _post(url, body, headers=None):
    req = urllib.request.Request(url, json.dumps(body).encode(),
                                 {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


class MockLLM:
    """Deterministic stand-in so every exercise runs with no model or API key."""
    name = "mock"

    def chat(self, messages):
        system, user = messages[0]["content"], messages[-1]["content"]
        if "ROLE:ROUTER" in system:
            q = user.lower()
            if "car" in q or "buy" in q or "what if" in q:
                nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*", q)]
                args = {}
                if len(nums) >= 2:
                    args = {"price": nums[0], "down": nums[1]}
                return json.dumps({"tool": "simulate_car_purchase", "args": args})
            if "forecast" in q or "future" in q:
                return json.dumps({"tool": "forecast_balances", "args": {}})
            if "account" in q:
                return json.dumps({"tool": "get_accounts", "args": {}})
            return json.dumps({"tool": "calculate_health_metrics", "args": {}})
        return "Based on the verified figures: " + user.split("EVIDENCE:", 1)[-1].strip()[:600]


class OllamaLLM:
    """Open-weight model running locally (https://ollama.com). e.g. llama3.2, qwen2.5."""

    def __init__(self, model="llama3.2", host="http://localhost:11434"):
        self.name, self.model, self.host = f"ollama:{model}", model, host

    def chat(self, messages):
        r = _post(f"{self.host}/api/chat", {"model": self.model, "messages": messages, "stream": False})
        return r["message"]["content"]


class OpenAILLM:
    """Commercial API via OpenAI-compatible chat completions. Needs OPENAI_API_KEY."""

    def __init__(self, model="gpt-4o-mini", base_url="https://api.openai.com/v1"):
        self.name, self.model, self.base_url = f"openai:{model}", model, base_url

    def chat(self, messages):
        key = os.environ["OPENAI_API_KEY"]
        r = _post(f"{self.base_url}/chat/completions", {"model": self.model, "messages": messages},
                  {"Authorization": "Bearer " + key})
        return r["choices"][0]["message"]["content"]


def get_llm(spec="mock"):
    """spec: 'mock', 'ollama[:model]', or 'openai[:model]'."""
    kind, _, model = spec.partition(":")
    if kind == "mock":
        return MockLLM()
    if kind == "ollama":
        return OllamaLLM(model or "llama3.2")
    if kind == "openai":
        return OpenAILLM(model or "gpt-4o-mini")
    raise ValueError(f"unknown LLM spec: {spec}")
