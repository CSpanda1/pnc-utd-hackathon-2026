# pnc-utd-hackathon-2026
This repo contains workshop materials to help Hackathon participants on the PNC Financial Wellness AI challenge.

- **Workshop guide (60 min):** [docs/WORKSHOP.md](docs/WORKSHOP.md)
- **Toolkit:** `wellness/` – synthetic data generator, unified profile, health metrics, forecast, scenario simulator,
  agent tools, LLM adapters (mock / Ollama open-weight / OpenAI commercial), Level 1-3 agents, LangGraph orchestration.

```bash
pip install -r requirements.txt
python -m wellness.data_gen -n 5 -o customers.json
python -m wellness.cli --level 3 "If I buy a \$30,000 car with \$5,000 down, will I have enough for emergencies?"
python -m pytest
```


**********************************************************
To run the Streamlit app, the command is: python -m streamlit run smartmoney_app.py
