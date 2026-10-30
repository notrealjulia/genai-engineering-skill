"""Chain-of-Thought: zero-shot and few-shot variants with reliable answer extraction.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0).
"""
import re

from llm_client import call_llm

ANSWER_MARKER = "ANSWER:"


def extract_answer(text: str) -> str | None:
    """Return the text after the LAST answer marker (the model may mention it mid-reasoning)."""
    matches = re.findall(rf"{ANSWER_MARKER}\s*(.+)", text)
    return matches[-1].strip() if matches else None


def chain_of_thought(question: str) -> dict:
    """Zero-shot CoT. Returns both the reasoning and the answer, for auditing."""
    full = call_llm(
        f"{question}\n\nThink step by step. Then give only the final answer on its own "
        f"line, starting with '{ANSWER_MARKER}'."
    )
    answer = extract_answer(full)
    if answer is None:
        # Fail loudly rather than silently returning the whole response as "the answer".
        raise ValueError(f"No '{ANSWER_MARKER}' line found in model output:\n{full}")
    return {"answer": answer, "reasoning": full}


# Few-shot CoT: worked examples teach both the reasoning style and the output format.
FEW_SHOT_COT_EXAMPLES = [
    {
        "q": "A train travels 60 km/h for 2.5 hours, then 80 km/h for 1 hour. Total distance?",
        "reasoning": "First leg: 60 * 2.5 = 150 km. Second leg: 80 * 1 = 80 km. "
                     "Total: 150 + 80 = 230 km.",
        "a": "230 km",
    },
    {
        "q": "A shirt costs $40 and is discounted 25%, then taxed 10%. Final price?",
        "reasoning": "Discount: 40 * 0.25 = 10, so 30. Tax: 30 * 0.10 = 3, so 33.",
        "a": "$33",
    },
]


def few_shot_chain_of_thought(question: str) -> dict:
    messages = [{"role": "system",
                 "content": f"Solve the problem step by step, then end with '{ANSWER_MARKER} <answer>'."}]
    for ex in FEW_SHOT_COT_EXAMPLES:
        messages.append({"role": "user", "content": ex["q"]})
        messages.append({"role": "assistant",
                         "content": f"{ex['reasoning']}\n{ANSWER_MARKER} {ex['a']}"})
    messages.append({"role": "user", "content": question})
    full = call_llm(messages)
    answer = extract_answer(full)
    if answer is None:
        raise ValueError(f"No '{ANSWER_MARKER}' line found in model output:\n{full}")
    return {"answer": answer, "reasoning": full}


if __name__ == "__main__":
    result = chain_of_thought(
        "If a store has 3 shelves with 8 books each, and 5 books are removed, how many remain?"
    )
    print(result["answer"])  # 19
