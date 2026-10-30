"""Few-shot classification with examples kept as data and validated output.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0).
"""
from llm_client import call_llm

LABELS = {"positive", "negative", "neutral"}

# Keep examples as data: easy to review, version, balance, and later optimize.
# One per label minimum, varied length and wording, including a tricky mixed case.
EXAMPLES = [
    ("The food was absolutely incredible, best meal I've had in years.", "positive"),
    ("Waited 45 minutes and the order was wrong. Never coming back.", "negative"),
    ("The restaurant is located on Main Street and opens at 11am.", "neutral"),
    ("Pricey, and the music was too loud, but honestly I'd go again for that pasta.", "positive"),
]


def build_messages(text: str) -> list[dict]:
    messages = [{
        "role": "system",
        "content": "Classify the sentiment of the review. "
                   f"Reply with exactly one word: {', '.join(sorted(LABELS))}.",
    }]
    for example_input, label in EXAMPLES:  # examples as prior conversation turns
        messages.append({"role": "user", "content": example_input})
        messages.append({"role": "assistant", "content": label})
    messages.append({"role": "user", "content": text})
    return messages


def few_shot_classify(text: str, retries: int = 1) -> str:
    """Return a label from LABELS, retrying once on invalid output, then failing loudly."""
    for _ in range(retries + 1):
        raw = call_llm(build_messages(text))
        label = raw.strip().strip(".").lower()
        if label in LABELS:
            return label
    raise ValueError(f"Model returned invalid label {raw!r}; expected one of {LABELS}")


if __name__ == "__main__":
    print(few_shot_classify("The service was slow but the dessert made up for it."))
