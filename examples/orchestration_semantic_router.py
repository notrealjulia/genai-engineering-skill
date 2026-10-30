"""Semantic Router: classify intent by embedding similarity to example utterances per route,
with a confidence threshold, an ambiguity margin, and an LLM fallback for unclear messages.

Based on the pattern described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book has
no code for this pattern. Reuses rag_common.VectorIndex: utterances are embedded once,
each message costs one embedding call, and the LLM runs only when embeddings are unsure.
"""
from dataclasses import dataclass

from llm_client import call_llm
from orchestration_common import default_tiers
from rag_common import Chunk, VectorIndex

OTHER = "other"  # forced into the closest route, an unmatched message is always misrouted


@dataclass
class Route:
    name: str
    description: str        # used by the LLM fallback
    utterances: list[str]   # 5-20 phrasings: diverse in wording, consistent in intent


class SemanticRouter:
    def __init__(self, routes: list[Route], threshold: float = 0.5, margin: float = 0.05,
                 llm_fallback: bool = True, embed_fn=None):
        """threshold and margin depend on the embedding model: calibrate both on real
        user messages, not on the clean examples you wrote."""
        self.routes = {r.name: r for r in routes}
        self.threshold, self.margin, self.llm_fallback = threshold, margin, llm_fallback
        self.index = VectorIndex([Chunk(f"{r.name}#{i}", u, r.name)
                                  for r in routes for i, u in enumerate(r.utterances)],
                                 embed_fn=embed_fn)

    def route(self, message: str) -> dict:
        best = {}  # a route scores as its closest utterance; results arrive best-first
        for score, chunk in self.index.search(message, k=len(self.index.chunks)):
            best.setdefault(chunk.source, round(score, 3))
        ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)
        (top, s1), s2 = ranked[0], ranked[1][1] if len(ranked) > 1 else -1.0
        decision = {"scores": dict(ranked)}
        if s1 >= self.threshold and s1 - s2 >= self.margin:
            return {**decision, "route": top, "method": "embedding"}
        why = "low_confidence" if s1 < self.threshold else "ambiguous"
        if not self.llm_fallback:
            return {**decision, "route": OTHER, "method": why}
        return {**decision, "route": self.classify_llm(message), "method": f"llm ({why})"}

    def classify_llm(self, message: str) -> str:
        """Slower, smarter classifier for the unclear minority of messages."""
        menu = "\n".join(f"- {r.name}: {r.description}" for r in self.routes.values())
        label = call_llm(
            f"Classify the message into exactly one category, or '{OTHER}' if none fits.\n"
            f"{menu}\n- {OTHER}: anything else\n\nReply with only the category name.\n\n"
            f"Message: {message}", model=default_tiers()[0].model)
        label = label.strip().strip("'\"`.").lower()
        return label if label in self.routes else OTHER  # never trust an unlisted label


if __name__ == "__main__":
    routes = [
        Route("billing", "invoices, charges, refunds, payment methods", [
            "Why was I charged twice this month?", "Can I get a refund?",
            "How do I update my credit card?", "Where can I download my invoice?",
            "I was billed after I cancelled"]),
        Route("tech_support", "errors, crashes, bugs, login or performance problems", [
            "The app crashes when I open it", "I can't log in to my account",
            "I'm getting error 500 on the dashboard", "Sync has been stuck for hours",
            "The export button does nothing"]),
        Route("sales", "pricing, plans, demos, upgrades, enterprise deals", [
            "How much does the enterprise plan cost?", "Can I book a demo?",
            "Do you offer discounts for nonprofits?", "I want to upgrade to more seats",
            "What's included in the Pro plan?"]),
    ]
    router = SemanticRouter(routes)
    handlers = {"billing": "-> billing agent", "tech_support": "-> support agent",
                "sales": "-> sales team", OTHER: "-> human triage queue"}
    for msg in ["I think you charged my card two times", "the page goes blank after login",
                "can someone show me the product before we buy 50 licenses?",
                "Why was I charged for something that crashed?",  # two intents: router picks one
                "what's the weather in Paris tomorrow?"]:
        d = router.route(msg)
        print(f"{d['route']:12} {d['method']:22} {handlers[d['route']]:22} | {msg}\n"
              f"             scores={d['scores']}")
