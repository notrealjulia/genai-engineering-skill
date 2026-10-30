"""Prompt chaining with explicit JSON contracts, validation, a gate, and context passing.

Pipeline: extract facts -> (gate) -> summarize using facts + original article.
Adapted and extended from GenAI Patterns (S. Pandey, CC BY-NC 4.0).
"""
import json

from agents_common import parse_json as _parse_json
from llm_client import call_llm


class ChainStepError(Exception):
    """Raised when a step's output breaks its contract; tells you WHICH step failed."""


def parse_json(raw: str, step: str):
    parsed = _parse_json(raw)  # tolerates ```json fences and prose around the JSON
    if parsed is None:
        raise ChainStepError(f"[{step}] invalid JSON:\n{raw}")
    return parsed


def step_extract(article: str, n: int = 5) -> list[str]:
    """Step 1. Contract: a JSON list of 1..n non-empty fact strings."""
    raw = call_llm(
        f"Extract up to {n} of the most important facts from the article.\n"
        'Return ONLY a JSON array of strings, e.g. ["fact one", "fact two"].\n\n'
        f"ARTICLE:\n{article}"
    )
    facts = parse_json(raw, "extract")
    if not (isinstance(facts, list) and facts and all(isinstance(f, str) and f.strip() for f in facts)):
        raise ChainStepError(f"[extract] expected non-empty list of strings, got: {facts!r}")
    return facts[:n]


def step_summarize(facts: list[str], article: str) -> str:
    """Step 2. Gets the facts AND the original article, to avoid context loss."""
    facts_block = "\n".join(f"- {f}" for f in facts)
    return call_llm(
        "Write a 2-sentence executive summary based on these key facts. "
        "Use the article only to check accuracy, not to add new points.\n\n"
        f"KEY FACTS:\n{facts_block}\n\nARTICLE:\n{article}"
    ).strip()


def extract_then_summarize(article: str) -> dict:
    facts = step_extract(article)
    # Gate: deterministic control flow between steps (a single prompt can't do this).
    if len(facts) < 2:
        return {"facts": facts, "summary": None, "status": "skipped: too little content"}
    summary = step_summarize(facts, article)
    # Return intermediates. They are your debugging inspection points.
    return {"facts": facts, "summary": summary, "status": "ok"}


if __name__ == "__main__":
    article = ("Acme Corp reported Q3 revenue of $12M, up 40% year over year, driven by its "
               "new analytics product. Operating costs rose 15% as the company hired 60 engineers. "
               "CEO Dana Lee said Acme expects to reach profitability in Q2 next year, and the "
               "company raised its full-year revenue guidance to $45M.")
    print(json.dumps(extract_then_summarize(article), indent=2))
