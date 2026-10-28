# Workshop: Building a Financial Wellness AI (60 min)

**Challenge:** help people see their whole financial picture (checking, savings, CDs, IRAs, auto/personal loans,
credit cards, mortgages), understand it, and feel confident about their next move.
Everything uses **synthetic data** only.

## Learning outcomes
Consolidate accounts into one profile · compute 6+ health indicators · have an LLM explain (never calculate) ·
compare an open-weight local model with a commercial API model · build agent tools · understand orchestration &
validation · leave with a prototype or plan · reuse the synthetic data generator.

## Setup (before the session)
```bash
pip install -r requirements.txt
python -m wellness.data_gen -n 5 -o customers.json   # synthetic data, seeded
python -m pytest                                      # sanity check
# Optional open model:   ollama pull llama3.2
# Optional commercial:   export OPENAI_API_KEY=...
```

## Agenda
| Min | Segment | Hands-on |
|---|---|---|
| 0-5 | Challenge, ideas & architecture | Discuss the reference architecture below |
| 5-15 | Synthetic data & unified profile | Run `wellness/data_gen.py`, read `profile.py`; add a field/account |
| 15-25 | Health indicators | `metrics.py` has 8 indicators; add one more (e.g. subscription burden) |
| 25-35 | Level 1 vs Level 2 | `python -m wellness.cli --level 1 "..."` vs `--level 2`; try `--llm ollama:llama3.2` and `--llm openai:gpt-4o-mini` and compare |
| 35-47 | Level 3 orchestration | Read `graph.py`; add a node or route; break validation on purpose |
| 47-57 | Build time | Prototype or write an implementation plan |
| 57-60 | Share-out & next steps | |

## Reference architecture
```
Conversational UI (Streamlit / React / Gradio)
  -> Supervisor / Intent Router   (classify, choose tools, manage state)
      -> Account Data Agent       (tools.get_accounts / get_profile)
      -> Financial Health Agent   (tools.calculate_health_metrics)
      -> Forecasting Agent        (tools.forecast_balances)
      -> Scenario Agent           (tools.simulate_car_purchase)
  -> Validation & Safety Layer    (validation.py: numeric checks, evidence, missing data)
  -> Coach / Explanation Agent    (LLM explains findings, uncertainty, next steps)
```

## Example flow
*"If I buy a $30,000 car with $5,000 down, will I still have enough money for emergencies?"*
The supervisor recognizes a simulation request -> retrieves balances -> computes metrics -> runs the amortization and
cash-flow simulation -> validation checks the output -> the coach explains the **verified** result.

## Three implementation levels
- **Level 1 – Basic:** one LLM gets the data and answers (`levels.level1`). Fast to build; risks wrong math and hallucination.
- **Level 2 – Tool-using agent:** the LLM selects approved Python tools to fetch evidence and compute (`levels.level2`).
- **Level 3 – Orchestrated agents:** a workflow with shared state, conditional routing, validation and possibly several
  LLM agents (`graph.level3`, built on **LangGraph** when installed, with a plain-Python fallback).

**Recommendation:** start with Level 1 and 2; move to Level 3 only if your use case justifies the complexity.

## Rules of thumb
1. Code does the arithmetic; the LLM explains it.
2. Only expose approved tools; validate before explaining.
3. State uncertainty and missing data; no real customer data; output is not financial advice.
4. The default `mock` LLM works offline, so you are never blocked by models or keys.

## Open vs. commercial model comparison
Ask the same 3 questions with `--llm ollama:<model>` and `--llm openai:<model>`. Compare: correct tool choice,
valid JSON, faithfulness to the evidence, tone, latency, cost and privacy.

## Stretch ideas
Streamlit chat UI · LangGraph multi-agent with a critic · Monte Carlo forecast · debt-payoff (avalanche vs snowball)
simulator · goal-based coach · anomaly/subscription detection.
