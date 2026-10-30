"""ReAct loop: Thought -> Action -> Observation, repeated until the agent can answer.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0), including the chapter exercise:
an iteration cap with a circuit breaker that returns a fixed "could not find an answer"
message. Also added: a required, logged Thought per step, observation truncation and a
repeated-action breaker. Provider-neutral JSON actions; use native tool calling in
production.
"""
import json

from agents_common import Tool, execute_tool_call, parse_json, tool_menu
from llm_client import call_llm

GAVE_UP = "I could not find an answer within the allowed steps."

SYSTEM = """Solve the task by alternating reasoning and tool use.
Tools:
{menu}

Each turn, reply with ONLY one JSON object:
{{"thought": "<what you know so far, what is still missing, why this action>",
  "action": "<tool name>", "arguments": {{...}}}}
When you have enough information:
{{"thought": "<why you can answer now>", "action": "final_answer", "arguments": {{"answer": "..."}}}}
Base each thought on the previous observations. Don't repeat a search you already ran."""


def react(question: str, tools: list[Tool], max_iterations: int = 5,
          max_observation_chars: int = 2000, confirm=None) -> dict:
    messages = [{"role": "system", "content": SYSTEM.format(menu=tool_menu(tools))},
                {"role": "user", "content": question}]
    trace, done_actions = [], set()

    for step in range(1, max_iterations + 1):
        raw = call_llm(messages)
        messages.append({"role": "assistant", "content": raw})
        reply = parse_json(raw)
        if not (isinstance(reply, dict) and isinstance(reply.get("action"), str)):
            trace.append({"step": step, "error": "unparseable", "raw": raw[:200]})
            messages.append({"role": "user", "content": "Invalid response. Reply with ONE JSON "
                             "object with thought, action and arguments."})
            continue
        thought, action = reply.get("thought", ""), reply["action"]
        args = reply.get("arguments", {})

        if action == "final_answer":
            answer = args.get("answer") if isinstance(args, dict) else None
            if isinstance(answer, str) and answer.strip():
                trace.append({"step": step, "thought": thought, "action": action})
                return {"answer": answer, "stop_reason": "answered", "trace": trace}
            observation = "final_answer needs arguments {\"answer\": \"...\"}."
        else:
            key = json.dumps([action, args], sort_keys=True, default=str)
            if key in done_actions:  # reasoning that ignores observations repeats itself
                observation = ("You already ran this exact action; its result is above. "
                               "Use it, try something different, or give the final answer.")
            else:
                done_actions.add(key)
                _, observation = execute_tool_call(action, args, tools, confirm=confirm,
                                                   max_chars=max_observation_chars)
        trace.append({"step": step, "thought": thought, "action": action, "arguments": args,
                      "observation": observation[:200]})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    # Circuit breaker: a fixed, honest message instead of an unbounded loop or a guess.
    return {"answer": GAVE_UP, "stop_reason": "max_iterations", "trace": trace}


if __name__ == "__main__":
    facts = {"france": "France population: about 68 million (2024).",
             "germany": "Germany population: about 84 million (2024)."}
    search = Tool("search", "Look up facts about a country. Query with the country name.",
                  lambda query: next((v for k, v in facts.items() if k in query.lower()),
                                     "No results."),
                  {"type": "object", "required": ["query"],
                   "properties": {"query": {"type": "string"}}})
    result = react("How many more people live in Germany than in France?", [search])
    for t in result["trace"]:
        print(t)
    print(result["stop_reason"], "->", result["answer"])
