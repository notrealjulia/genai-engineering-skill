"""Agentic RAG: (1) a simple router, and (2) a bounded agent loop that chooses tools.

Based on the pattern described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book has
no code for this pattern. Uses a provider-neutral JSON action protocol to keep control flow
visible. In production, use your provider's native tool calling instead.
"""
import time
from dataclasses import dataclass
from typing import Callable

from agents_common import parse_json
from llm_client import call_llm


@dataclass
class Tool:
    name: str
    description: str          # be specific: the agent chooses tools from this text alone
    fn: Callable[[str], str]  # takes a query string, returns text for the agent


# --- (1) Routing: one classification call, then a fixed path. Start here. ---------------

def route(question: str, tools: list[Tool]) -> str:
    names = [t.name for t in tools] + ["none"]
    menu = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    choice = call_llm(
        f"Pick the single best data source for the question, or 'none' if no lookup is "
        f"needed.\n{menu}\n- none: answerable without looking anything up\n\n"
        f"Reply with only the name.\n\nQuestion: {question}").strip().lower()
    return choice if choice in names else tools[0].name  # invalid choice: safe default


# --- (2) Agent loop with hard limits and a full trace ---------------------------------

SYSTEM = """You answer questions by gathering information with tools.
Tools:
{menu}

Respond with ONLY one JSON object per turn, either:
{{"thought": "...", "action": "<tool name>", "input": "<query>"}}
or, when you have enough information (or no lookup is needed):
{{"thought": "...", "action": "final_answer", "input": "<answer, citing which tool results you used>"}}"""


def agentic_rag(question: str, tools: list[Tool], max_steps: int = 6,
                max_seconds: float = 60.0) -> dict:
    by_name = {t.name: t for t in tools}
    menu = "\n".join(f"- {t.name}: {t.description}" for t in tools)
    messages = [{"role": "system", "content": SYSTEM.format(menu=menu)},
                {"role": "user", "content": question}]
    trace, start = [], time.monotonic()

    for step in range(1, max_steps + 1):
        if time.monotonic() - start > max_seconds:
            return {"answer": None, "stop_reason": "time_budget", "trace": trace}
        raw = call_llm(messages)
        messages.append({"role": "assistant", "content": raw})
        try:
            action = parse_json(raw)  # None on garbage -> TypeError below
            name, arg = action["action"], str(action.get("input", ""))
        except (KeyError, TypeError, AttributeError):
            observation = 'Invalid response. Reply with ONE JSON object as specified.'
            trace.append({"step": step, "error": "unparseable", "raw": raw[:200]})
        else:
            if name == "final_answer":
                trace.append({"step": step, "action": name})
                return {"answer": arg, "stop_reason": "answered", "trace": trace}
            if name not in by_name:
                observation = f"Unknown tool {name!r}. Available: {list(by_name)}"
            else:
                try:
                    observation = by_name[name].fn(arg)[:4000]  # cap observation size
                except Exception as e:  # tool failure is fed back so the agent can recover
                    observation = f"Tool {name} failed: {e}. Try another tool or query."
            trace.append({"step": step, "action": name, "input": arg,
                          "thought": action.get("thought", ""), "observation": observation[:200]})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    return {"answer": None, "stop_reason": "step_budget", "trace": trace}


if __name__ == "__main__":
    import re
    policies = {"refund": "Annual plans: refunds within 30 days. Monthly plans: no refunds.",
                "retention": "Customer data is kept 24 months after account closure."}
    tools = [Tool("policy_search", "Company policies on refunds and data retention",
                  lambda q: "\n".join(v for k, v in policies.items() if k in q.lower()) or "No match."),
             Tool("calculator", "Evaluate an arithmetic expression like '30 * 3'",
                  # never eval raw model output: allow digits and operators only
                  lambda q: str(eval(q)) if re.fullmatch(r"[\d\s+\-*/().]+", q)
                  else "Error: only arithmetic is allowed.")]
    print(route("How long do you keep my data after I close my account?", tools))
    result = agentic_rag("What is the annual-plan refund window in days, times 3?", tools)
    print(result["stop_reason"], "->", result["answer"])
    for t in result["trace"]:
        print(t)
