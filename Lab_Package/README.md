# PNC × UT Dallas: AI Financial Wellness Workshop

**Deliverables:** 15-slide instructor PowerPoint (speaker notes included), progressive and EDA notebooks, 30-customer/196-account synthetic snapshots with 24 months of transactions and statements, deterministic financial tools, Streamlit and Next.js/FastAPI demonstrations, LLM integration, and unit tests. No real account data.

## Quick start
1. Extract the ZIP to a directory.
2. Install Python 3.10+; run `python -m pip install -r requirements.txt`.
3. Run `jupyter notebook` (or open the `.ipynb` with VS Code/JupyterLab). Install `notebook` separately if needed.
4. Execute cells top to bottom from the extracted directory. Basic sections do **not** need API keys or LangGraph.
5. Optional: `python -m pip install langgraph` for the graph. Restart kernel after installation.
6. Optional local LLM: use the already downloaded Hugging Face models as described below. Set `ENABLE_HF=1` for the notebook or select `huggingface` in Streamlit. Foundry Local (`foundry`) and Ollama are also available through the provider selector.
7. Optional cloud LLM: set `OPENAI_API_KEY` and an available `OPENAI_MODEL`. For tool calling additionally set `RUN_OPENAI_TOOLS=1`. Costs apply.
8. `python -m unittest test_financial_tools.py`; `streamlit run app.py` for standalone UI.

## Already downloaded Hugging Face models

The instructor machine's generative models are cached in WSL at `/home/muthu/.cache/huggingface/hub`. Run Python, Jupyter, and Streamlit **inside WSL** to use that cache. The Windows cache contains different models. Do not point a Windows Python process at the WSL cache and expect CUDA/dependencies to be shared.

From WSL, activate your existing inference environment, then run:

```bash
cd /mnt/c/Users/muthu/OneDrive/Documents/GitHub/codex-ai-assisted-coding-demo/PNC_UT_Dallas_Hackathon_Instructor_Lab_Package
# Only if your environment lacks the required packages:
python -m pip install -r requirements-hf.txt
export HF_HUB_OFFLINE=1
export ENABLE_HF=1
export HF_MODEL=Qwen/Qwen3-VL-2B-Instruct
LLM_PROVIDER=huggingface python llm_demo.py
python -m streamlit run app.py
```

For the progressive notebook, select the same WSL environment as its kernel. `ENABLE_HF` guards the optional inference cell; `LLM_PROVIDER` selects the CLI provider. The Streamlit selector controls its provider independently.

`local_hf.py` uses `local_files_only=True` for config, tokenizer, and weights, with remote code disabled. Missing or incomplete snapshots fail with an error instead of downloading weights. Loaded weights are reused between requests. Restart the process/kernel to release GPU memory or switch models cleanly.

### Foundry Local provider (qwen2.5-0.5b)

If Foundry Local is installed, you can use its OpenAI-compatible local endpoint from the same narration path:

```bash
foundry server start --port 5273 --idle-timeout 0
foundry model run qwen2.5-0.5b
set LLM_PROVIDER=foundry
set FOUNDRY_BASE_URL=http://127.0.0.1:5273/v1
set FOUNDRY_MODEL=qwen2.5-0.5b
python llm_demo.py
```

Notes:
- `LLM_PROVIDER=foundry` works in `llm_demo.py`, Streamlit (`app.py`), and `web_demo` backend narration.
- `FOUNDRY_API_KEY` is optional for local use; default is `foundry-local`.
- Use `foundry server status` to confirm the active local endpoint if you run a nondefault port.

| Setting | Default / purpose |
| --- | --- |
| `HF_MODEL` | `Qwen/Qwen3-VL-2B-Instruct`; cached repo ID or absolute snapshot directory |
| `HF_HUB_CACHE` | Optional custom cache directory; standard Hugging Face cache settings also apply |
| `HF_DEVICE` | `auto`: CUDA when available, otherwise CPU; can force `cuda` or `cpu` |
| `HF_QUANTIZATION` | `auto`: NF4 4-bit on CUDA, unquantized float32 on CPU; override with `4bit` or `none` |
| `HF_MAX_NEW_TOKENS` | `384`, bounded to 1–2048; increase if an explanation is cut off |

The default is the smallest cached instruction model found on this machine. Other cached choices include `Qwen/Qwen2.5-VL-3B-Instruct`, Qwen3-VL 4B/8B, and `mistralai/Mistral-7B-Instruct-v0.3`. Vision-language models receive text-only evidence in this lab. Qwen3.5 4B/9B require a Transformers version supporting their architecture (5.2+). Larger models need more memory; CPU inference is slower and uses float32 RAM. Base GPT-2/BERT and embedding models are not interchangeable with instruction models for this task. Install a PyTorch build suitable for your existing CUDA setup rather than replacing a working installation unnecessarily.

Run all lightweight checks with `python -m unittest discover -p 'test_*.py'`. Tests mock model inference so the core checks need no model packages, GPU, or network.

## Next.js + FastAPI alternative

The [`web_demo`](web_demo/README.md) adds registration, sign-in/out, a Next.js dashboard, authenticated FastAPI endpoints, per-user synthetic profiles, vehicle simulation, and optional cached Hugging Face narration. Follow its two-terminal setup and security walkthrough to showcase a separate frontend/backend architecture alongside Streamlit.

## LangGraph + OpenAI workshop inference example

Use `langgraph_openai_workshop_demo.py` for a runnable graph-based example that routes a workshop question, calls deterministic finance tools, and asks OpenAI for a grounded narrative from evidence.

```bash
# Windows PowerShell:
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_MODEL="gpt-4o"
python langgraph_openai_workshop_demo.py --customer-id C001 --question "If I buy a $30,000 car with $5,000 down, what changes?"
```

The script prints JSON including route trace, key computed values, and the final model answer. It demonstrates LangGraph orchestration for workshop inference while keeping all calculations in `financial_tools.py`.

## LangChain + OpenAI workshop inference example

Use `langchain_openai_workshop_demo.py` for a runnable LangChain LCEL pipeline demo that routes a workshop question, gathers deterministic evidence, and calls OpenAI through `langchain-openai`.

```bash
# Optional packages (if missing):
python -m pip install langchain langchain-openai

# Windows PowerShell:
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_MODEL="gpt-4o"
python langchain_openai_workshop_demo.py --customer-id C001 --question "If I buy a $30,000 car with $5,000 down, what changes?"
```

The script returns JSON with route, execution trace, key computed values, and final narration so you can compare LangChain and LangGraph workshop patterns side-by-side.

## LangSmith observability demos (LangChain + LangGraph)

Use these scripts when you want trace-level observability in LangSmith:
- `langchain_openai_langsmith_demo.py`
- `langgraph_openai_langsmith_demo.py`

```bash
# Optional packages (if missing):
python -m pip install langchain langchain-openai langsmith

# Windows PowerShell:
$env:OPENAI_API_KEY="sk-..."
$env:OPENAI_MODEL="gpt-4o"
$env:LANGSMITH_API_KEY="lsv2_pt_..."
$env:LANGSMITH_PROJECT="pnc-utd-workshop-observability"

python langchain_openai_langsmith_demo.py --customer-id C001 --question "How does my current surplus look?"
python langgraph_openai_langsmith_demo.py --customer-id C001 --question "If I buy a $30,000 car with $5,000 down, what changes?"
```

Both scripts enforce `LANGSMITH_TRACING=true`, use OpenAI tool-calling for evidence retrieval, and attach route/customer metadata with traceable run names/tags so workshop runs are easy to filter in LangSmith. In the LangSmith UI you should see Tool Calls instead of "No tool calls".

## Exploratory data analysis notebook

Open [`PNC_Synthetic_Financial_Wellness_EDA.ipynb`](PNC_Synthetic_Financial_Wellness_EDA.ipynb) in JupyterLab for a comprehensive analysis of the synthetic profiles and accounts. Install `python -m pip install -r requirements-eda.txt` in the selected kernel environment, then run all cells. It requires no API key, model download, or GPU.

The notebook includes data dictionaries and provenance hashes, quality checks, reconciliations against `financial_tools.py`, customer and account distributions, ownership patterns, cash flow, partial balance sheets, correlations, descriptive segments, outliers, debt concentration, vehicle sensitivity, income/expense stress scenarios, and a customer drill-down. Saved outputs and charts are included. Set `EXPORT_RESULTS=True` to additionally export CSV summaries and chart PNGs under `eda_outputs/`; source CSVs are never overwritten. Threshold examples are editable teaching assumptions.

## Slide pacing
Slide 1–2: 0–8 min; slides 3–5: 8–18; slides 6–8: 18–28; slides 9–11: 28–40; slides 12–13: 40–55; slides 14–15: 55–60. The slide deck has speaker notes with presenter prompts.

## Data definitions and limitations

### Transaction history

The dataset now includes **21,225 transactions and 4,704 monthly statements**, covering November 2024–October 2026 for every listed account. See [`data/README.md`](data/README.md) for the schema, posting conventions, generation rules, and reconciliation equations. Stable account IDs connect all three account-level files; every final statement reconciles to the original account balance.

Both notebooks and both frontends include the history. Streamlit provides monthly activity, account balance, and paginated transaction tabs. The Next.js dashboard provides monthly activity, account statements, account/date filters, and paginated transaction entries through an authenticated API. LLM evidence includes compact historical totals and the latest three months rather than raw ledger dumps. Profile-based indicators remain snapshot calculations; the history is a separate generated activity view.

Checking and savings include deposits/payroll, withdrawals, and payments; CDs/IRAs include contributions and interest; card and loan accounts include their respective purchases, fees, interest, and payments. Internal transfers are paired and excluded from spending. Snapshot-calibrated external transfers remain explicitly separate. IRA interest is a simplified teaching model, not a market-return or retirement-eligibility simulation.

Run `python generate_history.py` to reproduce history using the existing snapshots. `python generate_data.py` regenerates both snapshots and history. Run `python -m unittest discover -p 'test_*.py'` to check all core functionality. Restart app processes and rerun notebook cells after regeneration.
Assets in the dataset: checking, savings, CD, IRA. Liabilities: credit card, auto loan, personal loan, mortgage. Deposits and investment accounts are **not** all liquid. Liquidity includes checking and savings only. `net_worth_partial` is deliberately incomplete, because property values, other holdings and taxes are absent. The simplified loan simulator omits ownership costs. This dataset and teaching lab must not be used for underwriting, trading, investment advice or production decisions.

## Data generator script documentation

The package includes two deterministic generator scripts. If you meant `generate_date.py`, use `generate_data.py`.

### `generate_data.py`

Creates baseline synthetic snapshot files, then regenerates history by calling `generate_history(...)`.

**What it writes**
- `data/profiles.csv`
- `data/accounts.csv`
- `data/transactions.csv` and `data/monthly_statements.csv` (via `generate_history.py`)
- `data/history_manifest.json` (via `generate_history.py`)

**Generation behavior**
- Uses fixed seed `20261107` for reproducibility.
- Produces 30 customers (`C001`–`C030`).
- Samples monthly income and non-debt expenses with bounded random ranges.
- Always creates checking, savings, IRA, and credit-card rows.
- May omit zero-balance CD / auto-loan / personal-loan / mortgage rows.
- Uses simplified synthetic APR/payment assumptions for debt accounts.

**Run**

```bash
python generate_data.py
```

### `generate_history.py`

Builds deterministic, cent-exact monthly ledger history anchored to the current `profiles.csv` and `accounts.csv` closing balances.

**What it validates**
- Month count must be `1..120`.
- `--end` must be a month-end date.
- Each customer must have one row per account type and a checking account.
- Running balances cannot go negative.
- Final generated closing balance must match `accounts.csv` for every account.

**What it writes**
- Rewrites `data/accounts.csv` with added metadata:
  - `account_id`, `opening_balance`, `history_start`, `as_of_date`, `annual_yield_pct`
- `data/transactions.csv`
- `data/monthly_statements.csv`
- `data/history_manifest.json` (seed, date bounds, counts, schema metadata)

**CLI usage**

```bash
python generate_history.py
python generate_history.py --end 2026-10-31 --months 24
```

**Key modeling notes**
- Uses fixed seed `20261108`.
- Assets and liabilities are both stored as positive balances.
- `balance_change` is signed change in the specific account.
- Loan/card interest and fees are synthetic accrual events.
- External cash-management transfers are synthetic calibration flows that preserve supplied ending balances.

## Generate 100 indicators across customers

Use `review_financial_indicators.py` to generate **100 different cross-customer financial indicators** (portfolio-level metrics), not a per-customer list.

```bash
python review_financial_indicators.py
```

Optional:

```bash
python review_financial_indicators.py --limit-customers 20
```

Output is JSON with:
- exactly 100 aggregate indicators
- customer population used for aggregation
- metric definitions (distribution and history-ratio groups)

Complete reference (all 100 names + definitions): [`FINANCIAL-INDICATORS-100.md`](FINANCIAL-INDICATORS-100.md)

## Suggested participant exercises
A. Grounded conversational coach; B. at least six health indicators; C. vehicle or income-shock what-if; D. bounded function-call router. Test numeric fidelity and unknown/unauthorized customer IDs.

## Reference docs
- https://developers.openai.com/api/docs/guides/function-calling
- https://docs.langchain.com/oss/python/langgraph/overview
- https://learn.microsoft.com/azure/foundry-local/reference/reference-cli
- https://docs.ollama.com/
- https://docs.streamlit.io/
- https://www.consumerfinance.gov/consumer-tools/educator-tool/financial-well-being-resources/
