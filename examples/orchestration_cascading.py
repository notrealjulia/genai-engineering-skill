"""Cascading: answer with the cheapest model first and escalate to a bigger tier only when
a quality gate rejects the answer.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0). Extended with: a combined gate
(free deterministic checks first, then an LLM judge whose JSON verdict is validated), any
number of tiers with a hard cap, an optional start tier so a Model Router can skip the
cheap tier, cost accounting against "always use the top tier", and a per-tier trace.
The __main__ block is a small version of the chapter exercise (cascading economics).
"""
import re

from agents_common import parse_json
from llm_client import call_llm
from orchestration_common import Tier, default_tiers

# Deterministic gate: the cheapest check there is. Catches obvious failures for free.
HEDGES = re.compile(r"\b(i'?m not (sure|certain)|i am not (sure|certain)|i don'?t know|"
                    r"i do not know|not enough information|i cannot (answer|determine))\b", re.I)

JUDGE = """You are a strict reviewer. Does the ANSWER fully and correctly answer the QUESTION?
Fail answers that are wrong, partial, vague or that dodge the question.
Reply with ONLY JSON: {{"pass": true or false, "reason": "<one sentence>"}}

QUESTION: {question}

ANSWER: {answer}"""


def deterministic_gate(answer: str, min_chars: int = 0) -> str | None:
    """Return a failure reason, or None. Raise min_chars only for questions that need
    detail: a correct "4" to "what is 2+2?" must not escalate."""
    if not answer.strip():
        return "empty answer"
    if HEDGES.search(answer):
        return "hedging language"
    if len(answer.strip()) < min_chars:
        return "suspiciously short"
    return None


def llm_judge(question: str, answer: str, model: str) -> tuple[bool, str]:
    verdict = parse_json(call_llm(JUDGE.format(question=question, answer=answer), model=model))
    # Only a real boolean counts: "true"/"false" strings or a missing key mean "unknown".
    # Unknown fails the gate: an escalation costs money, a bad answer costs trust.
    if not (isinstance(verdict, dict) and isinstance(verdict.get("pass"), bool)):
        return False, "unparseable judge verdict"
    return verdict["pass"], str(verdict.get("reason", ""))


def cascade(question: str, tiers: list[Tier] | None = None, start_tier: int = 0,
            max_tiers: int = 3, gate=None) -> dict:
    """gate(question, answer) -> (passed, reason) replaces the LLM judge: domain checks
    (recompute a number, look a fact up) beat a judge against confidently wrong models."""
    tiers = tiers or default_tiers()
    judge = tiers[0]  # the judge runs on the cheapest tier, or it eats the savings
    gate_cost = 0.0 if gate else judge.cost  # a custom gate makes no LLM call
    gate = gate or (lambda q, a: llm_judge(q, a, judge.model))
    attempts = tiers[start_tier:start_tier + max_tiers]  # cap tiers: latency stacks per tier
    if not attempts:
        raise ValueError(f"start_tier {start_tier} is out of range for {len(tiers)} tiers")

    trace, cost = [], 0.0
    for tier in attempts:
        answer = call_llm(question, model=tier.model).strip()
        cost += tier.cost
        reason = deterministic_gate(answer)
        passed = reason is None
        if passed:  # only pay for the judge when the free checks pass
            passed, reason = gate(question, answer)
            cost += gate_cost
        trace.append({"tier": tier.name, "model": tier.model, "passed": passed,
                      "reason": reason, "answer": answer})
        if passed:
            break
    # The last tier's answer ships even if it failed the gate, flagged for human review.
    return {"answer": answer, "tier": tier.name, "escalated": len(trace) > 1,
            "stop_reason": "accepted" if passed else "final_tier_unverified",
            "needs_review": not passed, "cost": cost, "baseline_cost": tiers[-1].cost,
            "trace": trace}


if __name__ == "__main__":
    # (query, regex a correct answer must match, or None if open-ended). Known answers let
    # you check the GATE, not just the models: the chapter exercise asks whether the gate
    # made the right call, because a gate that passes wrong answers looks like savings.
    queries = [("What is the capital of Australia?", r"Canberra"),
               ("Convert 72 degrees Fahrenheit to Celsius.", r"22\.2"),
               ("A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. "
                "How much is the ball?", r"0\.05|5 cents"),
               ("How many times does the letter r appear in 'strawberry'?", r"\b(3|three)\b"),
               ("Explain the CAP theorem in distributed systems in three sentences.", None),
               ("Write a Python regex that matches only valid IPv4 addresses (each octet "
                "0-255) and explain each part.", None)]
    results, gate_calls = [cascade(q) for q, _ in queries], []
    for (q, expected), r in zip(queries, results):
        print(f"[{r['tier']:5}] escalated={r['escalated']!s:5} review={r['needs_review']!s:5} "
              f"cost={r['cost']:5.1f} | {q[:50]}")
        for t in r["trace"]:
            right = ""
            if expected:  # was the gate's verdict on this tier's answer correct?
                correct = bool(re.search(expected, t["answer"], re.I))
                gate_calls.append(t["passed"] == correct)
                right = " | gate right" if t["passed"] == correct else (
                    f" | GATE WRONG: {'passed a wrong' if t['passed'] else 'rejected a correct'} answer")
            print(f"        {t['tier']}: passed={t['passed']} ({t['reason'][:70]}){right}")
    total, baseline = sum(r["cost"] for r in results), sum(r["baseline_cost"] for r in results)
    rate = sum(r["escalated"] for r in results) / len(results)
    print(f"\nescalation rate {rate:.0%} | cost {total:.1f} vs {baseline:.1f} top-tier only "
          f"| savings {1 - total / baseline:.0%} | gate right {sum(gate_calls)}/{len(gate_calls)}")
    if rate > 0.5:  # above 50-60%, the cheap tier or the gate is the problem
        print("WARNING: escalation rate above 50%: check the gate or the cheap model.")
    if not all(gate_calls):  # savings are only real if the gate is right
        print("WARNING: the gate made wrong calls: add a domain check or a stronger judge.")
