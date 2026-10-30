"""Self-consistency: parallel CoT samples, answer normalization, majority vote, confidence.

Extends the book's pattern description (GenAI Patterns, S. Pandey, CC BY-NC 4.0).
"""
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from llm_client import call_llm
from prompting_chain_of_thought import extract_answer

PROMPT = ("{question}\n\nThink step by step. Then give only the final answer on its own "
          "line, starting with 'ANSWER:'. For numeric answers give just the number.")


def normalize(answer: str) -> str:
    """Map equivalent answers to one key: '42', '42.0', 'The answer is 42.' -> '42'.

    Task-specific! Adapt for labels, units, yes/no, etc. Bad normalization silently
    breaks the vote.
    """
    a = answer.strip().lower().rstrip(".")
    a = re.sub(r"^(the answer is|answer:)\s*", "", a)
    num = re.fullmatch(r"[$€£]?\s*(-?[\d,]*\.?\d+)\s*%?", a)
    if num:
        value = float(num.group(1).replace(",", ""))
        return str(int(value)) if value.is_integer() else str(value)
    return a


def self_consistency(question: str, n: int = 7, temperature: float = 0.7) -> dict:
    prompt = PROMPT.format(question=question)
    # Parallel calls: latency ~ one call; cost still scales with n.
    with ThreadPoolExecutor(max_workers=n) as pool:
        outputs = list(pool.map(lambda _: call_llm(prompt, temperature=temperature), range(n)))

    answers = [normalize(a) for a in (extract_answer(o) for o in outputs) if a]
    if not answers:
        raise ValueError("No sample produced an extractable answer")

    counts = Counter(answers).most_common()
    top_votes = counts[0][1]
    winners = [ans for ans, c in counts if c == top_votes]
    return {
        "answer": winners[0],
        "tie": len(winners) > 1,        # decide a tie policy: re-sample, judge, or escalate
        "agreement": top_votes / n,     # cheap confidence signal; route low values for review
        "votes": dict(counts),
        "unparsed": n - len(answers),
    }


if __name__ == "__main__":
    result = self_consistency(
        "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
        "How much does the ball cost in cents?"
    )
    print(result)  # expect answer '5'
