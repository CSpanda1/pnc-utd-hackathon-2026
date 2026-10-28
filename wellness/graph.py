"""Level 3: orchestrated workflow. Uses LangGraph when installed, else a minimal fallback runner."""
import json
from typing import Any, Dict, TypedDict

from .levels import COACH, parse_tool_call
from .tools import make_tools, tool_descriptions
from .validation import validate


class State(TypedDict, total=False):
    question: str
    tool: str
    args: dict
    result: Any
    issues: list
    answer: str


def build_nodes(llm, customer):
    tools = make_tools(customer)

    def supervisor(s: State) -> Dict:
        router = ("ROLE:ROUTER Classify the question and pick ONE tool. Reply only JSON "
                  '{"tool": name, "args": {...}}. Tools:\n' + tool_descriptions(tools))
        call = parse_tool_call(llm.chat([{"role": "system", "content": router},
                                         {"role": "user", "content": s["question"]}]))
        return {"tool": call.get("tool"), "args": call.get("args", {})}

    def run_tool(s: State) -> Dict:
        if s["tool"] not in tools:
            return {"result": None}
        return {"result": tools[s["tool"]][0](**s["args"])}

    def validator(s: State) -> Dict:
        ok, issues = validate(s["tool"], s["result"])
        return {"issues": issues}

    def coach(s: State) -> Dict:
        return {"answer": llm.chat([{"role": "system", "content": COACH},
                                    {"role": "user", "content": f"{s['question']}\nEVIDENCE: "
                                     f"{json.dumps(s['result'])}"}])}

    def refuse(s: State) -> Dict:
        return {"answer": "I couldn't verify that result: " + "; ".join(s["issues"])}

    return supervisor, run_tool, validator, coach, refuse


def _route(s):
    return "coach" if not s["issues"] else "refuse"


def build_graph(llm, customer):
    supervisor, run_tool, validator, coach, refuse = build_nodes(llm, customer)
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError:
        class Fallback:
            def invoke(self, s):
                s = dict(s)
                for node in (supervisor, run_tool, validator):
                    s.update(node(s))
                s.update((coach if _route(s) == "coach" else refuse)(s))
                return s
        return Fallback()
    g = StateGraph(State)
    for name, fn in [("supervisor", supervisor), ("run_tool", run_tool), ("validator", validator),
                     ("coach", coach), ("refuse", refuse)]:
        g.add_node(name, fn)
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", "run_tool")
    g.add_edge("run_tool", "validator")
    g.add_conditional_edges("validator", _route, {"coach": "coach", "refuse": "refuse"})
    g.add_edge("coach", END)
    g.add_edge("refuse", END)
    return g.compile()


def level3(llm, customer, question):
    return build_graph(llm, customer).invoke({"question": question})["answer"]
