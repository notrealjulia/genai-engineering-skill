"""Shared agent building blocks: tool definitions, defensive JSON parsing, argument
validation and a safe tool-execution layer. Stdlib only.

Tools use a JSON-Schema subset ({"type": "object", "properties": ..., "required": ...}), the
same shape as native tool-calling APIs, so definitions port directly to your provider.
"""
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str        # the model picks tools from this text alone: say when to use it
    fn: Callable[..., Any]  # called with keyword args validated against `parameters`
    parameters: dict = field(default_factory=lambda: {"type": "object", "properties": {}})
    side_effects: bool = False  # True = writes/sends/deletes: requires explicit confirmation


def tool_menu(tools: list[Tool]) -> str:
    return "\n".join(f"- {t.name}: {t.description}\n  parameters: {json.dumps(t.parameters)}"
                     for t in tools)


def parse_json(raw: str):
    """Parse the first JSON object/array in an LLM reply (tolerates ```json fences and
    chatter around it). Returns None if nothing parses: callers decide what that means."""
    text = re.sub(r"```(?:json)?", "", raw or "")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text, i)[0]
            except json.JSONDecodeError:
                continue
    return None


_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
          "array": list, "object": dict}


def validate_args(args, schema: dict) -> list[str]:
    """Return a list of problems (empty = valid). Never trust model-generated arguments."""
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    props, errors = schema.get("properties", {}), []
    errors += [f"missing required argument {r!r}" for r in schema.get("required", []) if r not in args]
    for key, value in args.items():
        if key not in props:
            errors.append(f"unknown argument {key!r}")
            continue
        expected = _TYPES.get(props[key].get("type"))
        # bool is a subclass of int in Python; don't let True pass as an integer
        if expected and (not isinstance(value, expected) or
                         (isinstance(value, bool) and props[key].get("type") != "boolean")):
            errors.append(f"{key!r} must be {props[key]['type']}")
        elif "enum" in props[key] and value not in props[key]["enum"]:
            errors.append(f"{key!r} must be one of {props[key]['enum']}")
    return errors


def execute_tool_call(name: str, args, tools: list[Tool], confirm=None,
                      max_chars: int = 4000) -> tuple[bool, str]:
    """The execution layer: the model proposes, this code disposes.
    Returns (ok, text). Errors come back as text so the model can correct itself.
    `confirm(name, args) -> bool` gates side-effecting tools; with no confirm they're denied.
    """
    by_name = {t.name: t for t in tools}  # the allowlist: only registered tools ever run
    if name not in by_name:
        return False, f"Unknown tool {name!r}. Available: {sorted(by_name)}"
    tool = by_name[name]
    errors = validate_args(args, tool.parameters)
    if errors:
        return False, f"Invalid arguments for {name}: {'; '.join(errors)}"
    if tool.side_effects and not (confirm and confirm(name, args)):
        return False, f"Call to {name} was not approved. Do not retry it; tell the user."
    try:
        result = tool.fn(**args)
    except Exception as e:  # tool failure is an observation, not a crash
        return False, f"Tool {name} failed: {type(e).__name__}: {e}"
    text = result if isinstance(result, str) else json.dumps(result, default=str)
    if len(text) > max_chars:  # observation overload pushes earlier reasoning out of context
        text = text[:max_chars] + f"\n...[truncated {len(text) - max_chars} chars]"
    return True, text
