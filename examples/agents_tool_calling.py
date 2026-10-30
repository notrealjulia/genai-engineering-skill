"""Tool Calling: the model emits structured calls, your code validates and executes them,
results go back to the model, repeat until it answers.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0). Uses a provider-neutral JSON
protocol so the control flow is visible; in production use your provider's native tool
calling (same schemas, see agents_common.Tool) and keep the execution layer below.
Added: argument validation, confirmation for side effects, a call budget and a
repeated-call circuit breaker.
"""
import json

from agents_common import Tool, execute_tool_call, parse_json, tool_menu
from llm_client import call_llm

SYSTEM = """You can call tools to get information or take actions.
Tools:
{menu}

Respond with ONLY one JSON object per turn, either:
{{"tool_calls": [{{"name": "<tool>", "arguments": {{...}}}}]}}   (one or more independent calls)
or, when you can answer:
{{"answer": "<final answer for the user>"}}"""


def tool_calling_agent(query: str, tools: list[Tool], confirm=None, max_turns: int = 6,
                       max_tool_calls: int = 10, max_repeats: int = 2) -> dict:
    messages = [{"role": "system", "content": SYSTEM.format(menu=tool_menu(tools))},
                {"role": "user", "content": query}]
    trace, seen, n_calls = [], {}, 0

    for turn in range(1, max_turns + 1):
        raw = call_llm(messages)
        messages.append({"role": "assistant", "content": raw})
        reply = parse_json(raw)
        if isinstance(reply, dict) and isinstance(reply.get("answer"), str):
            return {"answer": reply["answer"], "stop_reason": "answered", "trace": trace}
        calls = reply.get("tool_calls") if isinstance(reply, dict) else None
        if not isinstance(calls, list) or not calls:
            trace.append({"turn": turn, "error": "unparseable", "raw": raw[:200]})
            messages.append({"role": "user", "content": "Invalid response. Reply with ONE JSON "
                             "object: either tool_calls or answer."})
            continue

        results = []
        for call in calls:
            if n_calls >= max_tool_calls:  # budget across the whole request, not per turn
                return {"answer": None, "stop_reason": "tool_call_budget", "trace": trace}
            n_calls += 1
            name = call.get("name") if isinstance(call, dict) else None
            args = call.get("arguments", {}) if isinstance(call, dict) else None
            # Circuit breaker: the same call over and over means the model is stuck.
            key = json.dumps([name, args], sort_keys=True, default=str)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] > max_repeats:
                trace.append({"turn": turn, "tool": name, "args": args, "error": "repeated call"})
                return {"answer": None, "stop_reason": "repeated_call", "trace": trace}
            ok, output = execute_tool_call(name, args, tools, confirm=confirm)
            trace.append({"turn": turn, "tool": name, "args": args, "ok": ok,
                          "output": output[:200]})
            results.append({"name": name, "ok": ok, "result": output})
        messages.append({"role": "user", "content": "Tool results:\n" + json.dumps(results)})

    return {"answer": None, "stop_reason": "turn_budget", "trace": trace}


if __name__ == "__main__":
    orders = {"C-42": [{"id": "O-1", "status": "shipped"}, {"id": "O-2", "status": "processing"}]}
    tools = [
        Tool("get_customer_orders", "List a customer's orders with their status. "
             "Use for any question about a specific customer's orders.",
             lambda customer_id: orders.get(customer_id, []),
             {"type": "object", "required": ["customer_id"],
              "properties": {"customer_id": {"type": "string", "description": "e.g. C-42"}}}),
        Tool("cancel_order", "Cancel an order that has not shipped yet.",
             lambda order_id: f"Order {order_id} cancelled.",
             {"type": "object", "required": ["order_id"],
              "properties": {"order_id": {"type": "string"}}},
             side_effects=True),
    ]
    ask = lambda name, args: input(f"Allow {name}({args})? [y/N] ").strip().lower() == "y"
    result = tool_calling_agent("Cancel customer C-42's order that hasn't shipped.", tools,
                                confirm=ask)
    print(result["stop_reason"], "->", result["answer"])
    for t in result["trace"]:
        print(t)
