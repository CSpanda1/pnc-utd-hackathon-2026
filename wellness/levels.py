"""Level 1 (single LLM) and Level 2 (tool-using agent)."""
import json

from .tools import make_tools, tool_descriptions

COACH = ("ROLE:COACH You are a friendly financial coach. Use ONLY the numbers in EVIDENCE. "
         "Never do your own arithmetic. Mention uncertainty. This is not financial advice.")


def level1(llm, customer, question):
    """Dump raw data into the prompt. Simple, but the model may miscalculate or hallucinate."""
    data = json.dumps({"accounts": customer["accounts"],
                       "income": customer["monthly_income"]})
    return llm.chat([{"role": "system", "content": COACH},
                     {"role": "user", "content": f"{question}\nEVIDENCE: {data}"}])


def parse_tool_call(text):
    start, end = text.find("{"), text.rfind("}")
    return json.loads(text[start:end + 1])


def level2(llm, customer, question):
    """LLM picks one approved tool; Python computes; LLM explains the tool output."""
    tools = make_tools(customer)
    router = ("ROLE:ROUTER Pick ONE tool for the question. Reply only with JSON "
              '{"tool": name, "args": {...}}. Tools:\n' + tool_descriptions(tools))
    call = parse_tool_call(llm.chat([{"role": "system", "content": router},
                                     {"role": "user", "content": question}]))
    if call.get("tool") not in tools:
        return "Sorry, I can't answer that with my approved tools."
    result = tools[call["tool"]][0](**call.get("args", {}))
    return llm.chat([{"role": "system", "content": COACH},
                     {"role": "user", "content": f"{question}\nEVIDENCE: {json.dumps(result)}"}])
