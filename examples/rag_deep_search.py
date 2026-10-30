"""Deep search: decompose -> search -> assess gaps -> follow-up queries -> synthesize,
with hard budgets and a diminishing-returns stop. Every round is logged.

Based on the pattern described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book has
no code for this pattern.
"""
from agents_common import parse_json
from llm_client import call_llm
from rag_common import Chunk, format_context


def _json(raw: str, default):
    parsed = parse_json(raw)  # tolerates ```json fences, which models add often
    return parsed if isinstance(parsed, type(default)) else default


def decompose(question: str, max_parts: int = 4) -> list[str]:
    parts = _json(call_llm(
        f"Break this question into at most {max_parts} self-contained search queries that "
        f"together cover every part of it. Return ONLY a JSON array of strings.\n\n{question}"),
        [])
    return [p for p in parts if isinstance(p, str)][:max_parts] or [question]


def assess(question: str, evidence: list[Chunk]) -> dict:
    """Is the evidence sufficient? What's missing? Any contradictions?"""
    result = _json(call_llm(
        "Given the question and the evidence gathered so far, return ONLY JSON:\n"
        '{"sufficient": true|false, "gaps": ["follow-up search query", ...], '
        '"contradictions": ["description", ...]}\n'
        "Gap queries should target missing information and may follow references in the "
        f"evidence.\n\nQUESTION: {question}\n\nEVIDENCE:\n{format_context(evidence)}"),
        {"sufficient": False, "gaps": [], "contradictions": []})
    return {"sufficient": bool(result.get("sufficient")),
            "gaps": [g for g in result.get("gaps", []) if isinstance(g, str)],
            "contradictions": result.get("contradictions", [])}


def deep_search(question: str, search_fn, max_rounds: int = 3, max_searches: int = 10,
                k: int = 4) -> dict:
    """search_fn(query, k) -> [(score, Chunk)]; can route to any source."""
    queries, evidence, seen, trace, searches = decompose(question), [], set(), [], 0
    assessment = {"sufficient": False, "gaps": [], "contradictions": []}
    stop_reason = "max_rounds"

    for round_no in range(1, max_rounds + 1):
        new = 0
        for q in queries:
            if searches >= max_searches:
                break
            searches += 1
            for _, chunk in search_fn(q, k):
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    evidence.append(chunk)
                    new += 1
        assessment = assess(question, evidence)
        trace.append({"round": round_no, "queries": queries, "new_chunks": new,
                      "sufficient": assessment["sufficient"], "gaps": assessment["gaps"]})
        if assessment["sufficient"]:
            stop_reason = "sufficient"
            break
        if new == 0:
            stop_reason = "no_new_information"  # diminishing returns
            break
        if searches >= max_searches:
            stop_reason = "search_budget"
            break
        queries = assessment["gaps"] or [question]

    answer = call_llm(
        "Answer the question using ONLY the evidence. Cite sources like [1]. "
        "If sources contradict each other, present both sides. Say explicitly which parts "
        "of the question the evidence does not cover.\n\n"
        f"QUESTION: {question}\n\nKNOWN CONTRADICTIONS: {assessment['contradictions']}\n\n"
        f"EVIDENCE:\n{format_context(evidence)}")
    return {"answer": answer, "stop_reason": stop_reason, "searches": searches,
            "sources": sorted({c.source for c in evidence}), "trace": trace}


if __name__ == "__main__":
    from rag_common import VectorIndex, chunk_text
    docs = {"sla.md": "The Enterprise plan includes a 99.9% uptime SLA. Service credits for "
                      "missed SLAs are described in credits.md.",
            "credits.md": "If monthly uptime falls below the SLA, Enterprise customers receive a "
                          "10% service credit per 0.1% of shortfall, capped at 50%.",
            "refunds.md": "Refunds are available within 30 days of purchase for annual plans."}
    index = VectorIndex([c for s, t in docs.items() for c in chunk_text(t, s)])
    result = deep_search("What uptime does Enterprise get, and what happens if it's missed?",
                         lambda q, k: index.search(q, k), k=1)  # k=1 forces follow-up rounds
    print(result["stop_reason"], result["searches"], result["sources"])
    print(result["answer"])
    for t in result["trace"]:
        print(t)
