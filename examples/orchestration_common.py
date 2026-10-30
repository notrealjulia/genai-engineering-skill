"""Shared building block for the Routing & Orchestration examples: model tiers.

A tier is a model plus its relative cost per call, so routers and cascades can report
what a request cost compared with sending everything to the top tier.
"""
from dataclasses import dataclass

import llm_client


@dataclass(frozen=True)
class Tier:
    name: str
    model: str
    cost: float  # relative cost per call (cheapest tier = 1). Set from YOUR price sheet.


# Cheapest first. Cost ratios are rough list-price ratios when written: prices change,
# so check your provider's current pricing before trusting a savings number.
DEFAULT_TIERS = {
    "openai": [Tier("small", "gpt-4o-mini", 1.0), Tier("large", "gpt-4o", 17.0)],
    "anthropic": [Tier("small", "claude-haiku-5-5", 1.0), Tier("mid", "claude-sonnet-5-5", 3.0),
                  Tier("large", "claude-opus-5-5", 5.0)],
}


def default_tiers() -> list[Tier]:
    return DEFAULT_TIERS[getattr(llm_client, "PROVIDER", "openai")]
