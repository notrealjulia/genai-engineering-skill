"""Minimal prompt-optimization harness (framework-free, illustrative).

NOT from the book (it has no code for this pattern). Shows the four components:
pipeline (prompt template), dataset (with held-out split), evaluator, optimizer
(here: search over candidate instructions and few-shot subsets). For real projects,
use DSPy, AdalFlow, PromptWizard, or similar.
"""
import itertools
import random

from llm_client import call_llm

# 1. DATASET: inputs with expected outputs. Use 50-100+ in practice.
DATASET = [
    ("Refund still not processed after 3 weeks.", "billing"),
    ("App crashes when I open settings.", "bug"),
    ("Can you add dark mode?", "feature_request"),
    ("I was charged twice this month.", "billing"),
    ("Login button does nothing on Android.", "bug"),
    ("Would love an export-to-CSV option.", "feature_request"),
    ("Why is my invoice higher than my plan price?", "billing"),
    ("Search returns no results even for exact titles.", "bug"),
    ("Please support SSO with Okta.", "feature_request"),
    ("My card was declined but I see a pending charge.", "billing"),
]
LABELS = sorted({label for _, label in DATASET})

# 2. PIPELINE: a template whose instruction text and examples are the tunable "parameters".
CANDIDATE_INSTRUCTIONS = [
    "Classify the support ticket.",
    f"Classify the support ticket into exactly one of: {', '.join(LABELS)}. Reply with the label only.",
    f"You route support tickets. Choose one label from {LABELS}. Output only the label, lowercase.",
]


def run_pipeline(instruction: str, examples: list, text: str) -> str:
    messages = [{"role": "system", "content": instruction}]
    for x, y in examples:
        messages += [{"role": "user", "content": x}, {"role": "assistant", "content": y}]
    messages.append({"role": "user", "content": text})
    return call_llm(messages).strip().lower()


# 3. EVALUATOR: exact match here; swap for a scoring function or LLM-as-Judge.
def evaluate(instruction: str, examples: list, data: list) -> float:
    return sum(run_pipeline(instruction, examples, x) == y for x, y in data) / len(data)


# 4. OPTIMIZER: exhaustive search over instruction x few-shot subset, scored on TRAIN only.
def optimize(seed: int = 0, n_shots: int = 2, max_subsets: int = 3) -> dict:
    rng = random.Random(seed)
    data = DATASET[:]
    rng.shuffle(data)
    split = int(len(data) * 0.6)
    train, test = data[:split], data[split:]  # held-out test guards against overfitting

    subsets = [[]] + list(itertools.combinations(train, n_shots))[:max_subsets]
    results = []
    for instruction in CANDIDATE_INSTRUCTIONS:
        for shots in subsets:
            # Exclude the shots themselves from scoring to avoid leaking the answers.
            eval_set = [d for d in train if d not in shots]
            results.append((evaluate(instruction, list(shots), eval_set), instruction, list(shots)))

    results.sort(key=lambda r: r[0], reverse=True)
    best_score, best_instruction, best_shots = results[0]
    return {
        "train_score": best_score,
        "test_score": evaluate(best_instruction, best_shots, test),  # the number to trust
        "instruction": best_instruction,
        "shots": best_shots,
        "candidates_tried": len(results),
    }


if __name__ == "__main__":
    print(optimize())
