"""Multi-Agent Collaboration, supervisor style: (1) a fixed hand-off pipeline (start here),
and (2) a supervisor that decides which specialist to delegate to next.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0): (1) generalizes the book's
research -> write -> review supervisor. (2) is an addition: dynamic delegation with a
shared notes document, a delegation budget and a loop breaker.
"""
import json
from dataclasses import dataclass
from typing import Callable

from agents_common import parse_json
from llm_client import call_llm


@dataclass
class Agent:
    name: str
    description: str        # what the supervisor reads when choosing whom to delegate to
    system_prompt: str
    run: Callable[[str, str], str] | None = None  # custom (system, task) -> text, e.g. a ReAct agent

    def __call__(self, task: str) -> str:
        if self.run:
            return self.run(self.system_prompt, task)
        return call_llm([{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": task}])


# --- (1) Fixed pipeline: each agent gets the task plus ALL earlier outputs ---------------

def pipeline(task: str, agents: list[Agent]) -> dict:
    outputs = {}
    for agent in agents:
        # Pass every earlier output, not just the last one: each hand-off is a chance to
        # lose a constraint the first agent found.
        context = "\n\n".join(f"## From {n}\n{o}" for n, o in outputs.items())
        outputs[agent.name] = agent(f"Task: {task}\n\n{context}".strip())
    return {"answer": outputs[agents[-1].name], "outputs": outputs}


# --- (2) Dynamic supervisor -------------------------------------------------------------

SUPERVISOR = """You coordinate specialist agents. You never do the work yourself.
Specialists:
{menu}

Shared notes so far (results from specialists):
{notes}

Reply with ONLY one JSON object, either:
{{"delegate": "<specialist name>", "task": "<self-contained instructions, including any constraints from the notes>"}}
or, when the notes contain a complete result:
{{"final": "<the final response for the user>"}}"""


def supervisor(task: str, agents: list[Agent], max_delegations: int = 6) -> dict:
    by_name = {a.name: a for a in agents}
    menu = "\n".join(f"- {a.name}: {a.description}" for a in agents)
    notes, trace, seen = [], [], set()

    for _ in range(max_delegations + 1):  # +1: the final turn needs no delegation
        notes_text = "\n\n".join(f"[{n['agent']}] {n['result']}" for n in notes) or "(none)"
        raw = call_llm(SUPERVISOR.format(menu=menu, notes=notes_text) + f"\n\nUser task: {task}")
        decision = parse_json(raw)
        if isinstance(decision, dict) and isinstance(decision.get("final"), str):
            return {"answer": decision["final"], "stop_reason": "done", "notes": notes, "trace": trace}
        if not (isinstance(decision, dict) and decision.get("delegate") in by_name
                and isinstance(decision.get("task"), str)):
            trace.append({"error": "invalid supervisor decision", "raw": raw[:200]})
            return {"answer": None, "stop_reason": "invalid_decision", "notes": notes, "trace": trace}
        if len(trace) >= max_delegations:
            break
        key = json.dumps([decision["delegate"], decision["task"]])
        if key in seen:  # the same hand-off again = delegation loop, escalate instead
            trace.append({"error": "repeated delegation", **decision})
            return {"answer": None, "stop_reason": "delegation_loop", "notes": notes, "trace": trace}
        seen.add(key)
        try:
            result = by_name[decision["delegate"]](decision["task"])
        except Exception as e:  # a failing specialist is reported, so the supervisor can adapt
            result = f"FAILED: {type(e).__name__}: {e}"
        notes.append({"agent": decision["delegate"], "task": decision["task"], "result": result})
        trace.append({"agent": decision["delegate"], "task": decision["task"][:200],
                      "result": result[:200]})

    return {"answer": None, "stop_reason": "delegation_budget", "notes": notes, "trace": trace}


if __name__ == "__main__":
    team = [
        Agent("researcher", "Gathers key facts and data points on a topic.",
              "You are a research specialist. List key facts and data points, concisely."),
        Agent("writer", "Turns research notes into a clear, structured response.",
              "You are a writing specialist. Write a clear, well-structured response "
              "using only the research provided."),
        Agent("reviewer", "Checks a draft for accuracy, clarity and completeness.",
              "You are a reviewer. Return the corrected final version of the draft."),
    ]
    task = "Write a brief comparison of PostgreSQL vs. MySQL for a startup."
    print(pipeline(task, team)["answer"])
    result = supervisor(task, team)
    print(result["stop_reason"], [t.get("agent") for t in result["trace"]])
    print(result["answer"])
