"""Grounded generation: out-of-domain check, numbered citations, citation validation,
and LLM verification that each cited claim is supported.

Based on the techniques described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book
has no code for this pattern.
"""
import re

from agents_common import parse_json
from llm_client import call_llm
from rag_common import Chunk, format_context

NO_ANSWER = "I don't have enough information to answer that."

PROMPT = """Answer the question using ONLY the numbered sources below.
After every sentence, cite the source(s) that support it, like [1] or [1][3].
Do not include anything the sources don't support. If the sources answer the question
negatively (e.g. "X is not allowed"), that IS an answer: give it, with its citation.
Only if the sources don't address the question at all, reply exactly: "{no_answer}"

{context}

Question: {question}"""


def split_claims(answer: str) -> list[tuple[str, list[int]]]:
    """Split into sentences and the citation numbers attached to each."""
    sentences = re.split(r"(?<=[.!?\]])\s+(?=[A-Z])", answer.strip())
    return [(s, [int(n) for n in re.findall(r"\[(\d+)\]", s)]) for s in sentences if s]


def verify_claim(claim: str, sources: list[Chunk]) -> bool:
    """LLM check: is the claim fully supported by its cited sources? (The verifier is the
    ceiling of this technique: evaluate it, or use a stronger model than the generator.)"""
    text = "\n\n".join(c.text for c in sources)
    verdict = parse_json(call_llm(
        "Is the CLAIM fully supported by the SOURCES? Reply ONLY with JSON: "
        '{"supported": true|false}\n\n'
        f"SOURCES:\n{text}\n\nCLAIM: {re.sub(r'\[\d+\]', '', claim).strip()}"))
    # unparseable verdict counts as unverified; `is True` so the string "false" doesn't pass
    return isinstance(verdict, dict) and verdict.get("supported") is True


def grounded_answer(question: str, results: list[tuple[float, Chunk]], min_score: float = 0.35,
                    verify: bool = True) -> dict:
    # 1. Out-of-domain detection: weak retrieval -> honest refusal, not a guess.
    #    Calibrate min_score on answerable AND unanswerable eval queries.
    if not results or results[0][0] < min_score:
        return {"answer": NO_ANSWER, "status": "out_of_domain", "citations": {}, "issues": []}

    chunks = [c for _, c in results]
    answer = call_llm(PROMPT.format(no_answer=NO_ANSWER, context=format_context(chunks),
                                    question=question)).strip()
    if NO_ANSWER in answer:
        return {"answer": NO_ANSWER, "status": "declined", "citations": {}, "issues": []}

    # 2. Validate citations exist and every claim has one; 3. verify support.
    issues = []
    for claim, refs in split_claims(answer):
        if not refs:
            issues.append({"claim": claim, "problem": "uncited"})
        elif any(r < 1 or r > len(chunks) for r in refs):
            issues.append({"claim": claim, "problem": f"cites non-existent source {refs}"})
        elif verify and not verify_claim(claim, [chunks[r - 1] for r in refs]):
            issues.append({"claim": claim, "problem": "not supported by cited source"})

    cited = sorted({r for _, refs in split_claims(answer) for r in refs if 1 <= r <= len(chunks)})
    return {
        "answer": answer,
        "status": "ok" if not issues else "needs_review",  # route needs_review: regenerate/flag
        "citations": {r: chunks[r - 1].source for r in cited},
        "issues": issues,
    }


if __name__ == "__main__":
    from rag_common import VectorIndex, chunk_text
    docs = {"refunds.md": "Refunds are available within 30 days of purchase for annual plans. "
                          "Monthly plans are not refundable.",
            "sla.md": "The Enterprise plan includes a 99.9% uptime SLA."}
    index = VectorIndex([c for s, t in docs.items() for c in chunk_text(t, s)])
    for q in ["What is the refund window for annual plans?",
              "Can I get a refund on a monthly plan?",  # correct answer is "no", not a refusal
              "Who is the CEO?"]:                        # out-of-domain: must refuse
        print(grounded_answer(q, index.search(q, 2)))
