"""Model Router: decide upfront which model tier a query needs, then call only that tier.
Rule-based fast path first (free), then a cheap LLM complexity score (1-5) mapped to tiers.

Based on the pattern described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book has
no code for this pattern. The ML-classifier variant (train on queries labelled with the
tier that handled them well) plugs in as another `classify` step.
"""
import re
from collections import Counter

from agents_common import parse_json
from llm_client import call_llm
from orchestration_common import Tier, default_tiers

# Fast path: obvious cases skip the router call, so routing never costs more than it saves.
# Rules are brittle (users phrase things you didn't anticipate): keep them few and obvious.
SIMPLE = re.compile(r"^\s*(translate|what is the capital of|define|convert)\b", re.I)
COMPLEX = re.compile(r"```|\b(step[- ]by[- ]step|trade-?offs?|prove|architect|debug)\b", re.I)

RATE = """Rate how much capability a model needs to answer this request well, from 1 to 5:
1 = factual lookup or simple formatting; 2 = short, simple instruction;
3 = moderate reasoning, summarization or explanation; 4 = multi-step reasoning, code or
careful analysis; 5 = expert-level, ambiguous or high-stakes.
Judge the reasoning required, not the length of the request.
Reply with ONLY JSON: {{"complexity": <1-5>, "reason": "<short reason>"}}

Request: {query}"""


def _score(value) -> int | None:
    """Accept 4 or "4"; reject bools (True is an int in Python), floats and out-of-range."""
    if isinstance(value, str) and value.strip().isdigit():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= 5:
        return value
    return None


def route_model(query: str, tiers: list[Tier] | None = None) -> dict:
    tiers = tiers or default_tiers()
    if COMPLEX.search(query):
        return {"tier": tiers[-1], "method": "rule", "score": None, "reason": "complex keyword"}
    if SIMPLE.search(query):
        return {"tier": tiers[0], "method": "rule", "score": None, "reason": "simple keyword"}

    reply = parse_json(call_llm(RATE.format(query=query), model=tiers[0].model))
    score = _score(reply.get("complexity")) if isinstance(reply, dict) else None
    if score is None:
        # Misrouting a hard query to a cheap model is the costly failure: fail upward.
        return {"tier": tiers[-1], "method": "fallback", "score": None,
                "reason": "unparseable complexity rating", "router_cost": tiers[0].cost}
    index = min(len(tiers) - 1, (score - 1) * len(tiers) // 5)  # spread 1-5 over the tiers
    return {"tier": tiers[index], "method": "llm", "score": score,
            "reason": str(reply.get("reason", "")), "router_cost": tiers[0].cost}


def routed_answer(query: str, tiers: list[Tier] | None = None) -> dict:
    tiers = tiers or default_tiers()
    decision = route_model(query, tiers)
    tier = decision["tier"]
    answer = call_llm(query, model=tier.model)
    # Log every decision: misrouting is invisible to the user, so you must find it in logs.
    return {**decision, "tier": tier.name, "model": tier.model, "answer": answer,
            "cost": decision.get("router_cost", 0.0) + tier.cost, "baseline_cost": tiers[-1].cost}


if __name__ == "__main__":
    queries = ["Translate 'good morning' into Spanish.",
               "What is the capital of Canada?",
               "Summarize in one sentence why the sky is blue.",
               "Our Postgres primary hits 100% CPU every night at 2am. List the likely "
               "causes and how to confirm each one.",
               "What are the trade-offs between microservices and a monolith for a "
               "fintech startup handling 10,000 transactions per second?",
               "Is 2^61 - 1 a prime number? Explain how you would check."]
    results = [routed_answer(q) for q in queries]
    for q, r in zip(queries, results):
        print(f"[{r['tier']:5}] {r['method']:8} score={r['score']} cost={r['cost']:5.1f} "
              f"| {q[:55]}\n        {r['reason']} -> {r['answer'][:90]!r}")
    # If one tier gets ~95% of traffic, the router is doing little useful work.
    print("\ntraffic per tier:", dict(Counter(r["tier"] for r in results)))
    total, baseline = sum(r["cost"] for r in results), sum(r["baseline_cost"] for r in results)
    print(f"cost {total:.1f} vs {baseline:.1f} top-tier only | savings {1 - total / baseline:.0%}")
