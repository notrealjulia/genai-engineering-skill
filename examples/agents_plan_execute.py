"""Plan and Execute: a planner decomposes the task into steps, an executor runs them one at a
time, and after each step a replanner keeps, revises or ends the plan. Then synthesize.

Based on the pattern described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book has
no code for this pattern.
"""
from agents_common import parse_json
from llm_client import call_llm

PLAN_PROMPT = """Break the task into at most {max_steps} ordered steps. Each step must be doable
in 1-3 actions with ONLY these capabilities:
{capabilities}

Reply with ONLY JSON: {{"steps": ["step 1", "step 2", ...]}}

Task: {task}"""

REPLAN_PROMPT = """Task: {task}

Completed steps and results:
{done}

Remaining plan:
{remaining}

Given the results, reply with ONLY one JSON object:
{{"action": "continue"}}  (remaining plan still makes sense)
{{"action": "replan", "steps": ["...", ...]}}  (new remaining steps, e.g. a step failed or results changed the approach)
{{"action": "finish"}}  (results already answer the task)"""


def _valid_steps(obj, max_steps: int) -> list[str] | None:
    steps = obj.get("steps") if isinstance(obj, dict) else None
    if (isinstance(steps, list) and 0 < len(steps) <= max_steps
            and all(isinstance(s, str) and s.strip() for s in steps)):
        return [s.strip() for s in steps]
    return None


def default_executor(step: str, context: str) -> str:
    # In practice: a ReAct agent with tools, e.g. lambda s, c: react(f"{c}\n\nDo: {s}", tools)["answer"]
    return call_llm(f"Results so far:\n{context or '(none)'}\n\nDo this step and report the "
                    f"result concisely: {step}")


def plan_and_execute(task: str, capabilities: str, executor=default_executor,
                     approve=None, max_plan_steps: int = 6, max_executions: int = 10,
                     max_replans: int = 3) -> dict:
    # 1. Plan. The planner sees the capabilities so it can't plan steps nobody can do.
    plan = _valid_steps(parse_json(call_llm(PLAN_PROMPT.format(
        max_steps=max_plan_steps, capabilities=capabilities, task=task))), max_plan_steps)
    result = {"answer": None, "initial_plan": plan, "done": [], "replans": 0}
    if plan is None:
        return {**result, "stop_reason": "invalid_plan"}
    if approve and not approve(plan):  # optional human review before anything runs
        return {**result, "stop_reason": "plan_rejected"}

    # 2. Execute step by step, re-planning on evidence.
    remaining, done = list(plan), result["done"]
    while remaining:
        if len(done) >= max_executions:
            return {**result, "stop_reason": "execution_budget"}
        step = remaining.pop(0)
        context = "\n".join(f"- {d['step']}: {d['result']}" for d in done)
        try:
            output = executor(step, context)
        except Exception as e:  # don't cascade: let the replanner route around the failure
            output = f"FAILED: {type(e).__name__}: {e}"
        done.append({"step": step, "result": output})
        if not remaining and not output.startswith("FAILED"):
            break  # plan complete and nothing failed: skip a needless replanning call
        decision = parse_json(call_llm(REPLAN_PROMPT.format(
            task=task, done="\n".join(f"- {d['step']}: {d['result']}" for d in done),
            remaining="\n".join(f"- {s}" for s in remaining) or "(none)")))
        action = decision.get("action") if isinstance(decision, dict) else None
        if action == "finish":
            break
        if action == "replan":
            new = _valid_steps(decision, max_plan_steps)
            if new is None or result["replans"] >= max_replans:
                # replanning more than executing = task is too exploratory for this pattern
                return {**result, "stop_reason": "replan_failed"}
            remaining, result["replans"] = new, result["replans"] + 1
        # "continue" or an unparseable decision: keep the current plan (safe default)

    # 3. Synthesize from the step results only.
    results = "\n".join(f"- {d['step']}: {d['result']}" for d in done)
    result["answer"] = call_llm(f"Task: {task}\n\nStep results:\n{results}\n\n"
                                f"Write the final answer using only these results. "
                                f"Say clearly if any step failed.")
    return {**result, "stop_reason": "completed"}


if __name__ == "__main__":
    out = plan_and_execute("Compare the pricing models of three popular CI/CD services.",
                           capabilities="- general knowledge (no web access)",
                           approve=lambda p: print("Plan:", *p, sep="\n  ") or True)
    print(out["stop_reason"], "| replans:", out["replans"])
    print(out["answer"])
