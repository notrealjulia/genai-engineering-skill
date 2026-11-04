"""Self-Check: before serving an answer, measure how sure the model was and verify its
claims, then serve it, send it for review, or abstain.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0). The book's code asks the model to
review its own answer (VERDICT: PASS/FAIL). Extended with: token-probability confidence
when the provider returns logprobs; a sampling-consistency fallback when it doesn't
(Anthropic's API doesn't), which the book notes needs no logprobs; consistency as a second
opinion only when logprobs look low; claim-by-claim verification with a validated JSON
verdict; and an explicit decision with a reason.
"""
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from agents_common import parse_json
from llm_client import call_llm, call_llm_logprobs
from prompting_self_consistency import normalize

ANSWER = """{context}Question: {question}

Put the short answer alone on the first line (a name, number or phrase). Then add at most
two sentences of explanation."""

VERIFY = """Check every factual claim in the ANSWER{against}.
Verdict per claim: "supported", "contradicted" or "unsure". Be strict: a claim you can't
confirm is "unsure". Reply with ONLY JSON:
{{"claims": [{{"claim": "<claim>", "verdict": "supported|contradicted|unsure"}}]}}
{context}
QUESTION: {question}
ANSWER: {answer}"""

VERDICTS = {"supported", "contradicted", "unsure"}


def key(answer: str) -> str:  # compare short answers, not their explanations
    return normalize(answer.strip().splitlines()[0] if answer.strip() else "")


def token_confidence(logprobs: list[float]) -> float:
    # The weakest token, not the mean: one shaky digit or name is where a hallucination
    # hides, and the mean dilutes it among confident filler tokens.
    return math.exp(min(logprobs))


def consistency(prompt: str, answer: str, n: int = 5, temperature: float = 1.0,
                model=None) -> float:
    """Share of n resampled answers that agree with `answer`. Works on any provider.
    Costs n extra calls (in parallel, so latency stays about one call)."""
    with ThreadPoolExecutor(max_workers=n) as pool:
        samples = list(pool.map(lambda _: call_llm(prompt, temperature=temperature,
                                                    model=model), range(n)))
    return Counter(key(s) for s in samples)[key(answer)] / n


def verify_claims(question: str, answer: str, context: str = "", model=None) -> list[dict]:
    """With context, claims are checked against it (a real check). Without, the model checks
    its own knowledge: it shares the generator's blind spots, so treat it as weak."""
    raw = call_llm(VERIFY.format(question=question, answer=answer, context=context and
                                 f"\nCONTEXT:\n{context}\n", against=context and
                                 " against the CONTEXT"), model=model)
    reply = parse_json(raw)
    claims = reply.get("claims") if isinstance(reply, dict) else None
    if not isinstance(claims, list) or not claims:
        return [{"claim": "(verifier reply unparseable)", "verdict": "unsure"}]
    return [c if isinstance(c, dict) and c.get("verdict") in VERDICTS
            else {"claim": str(c)[:200], "verdict": "unsure"} for c in claims]


def self_check(question: str, context: str = "", threshold: float = 0.8, n: int = 5,
               verify: bool = True, model=None, verify_model=None) -> dict:
    prompt = ANSWER.format(question=question, context=context and f"Context:\n{context}\n\n")
    answer, logprobs = call_llm_logprobs(prompt, model=model)
    if logprobs:
        method, confidence = "logprobs", token_confidence(logprobs)
        if confidence < threshold:  # rare-but-correct tokens score low: get a second opinion
            method, confidence = "logprobs+consistency", consistency(prompt, answer, n,
                                                                      model=model)
    else:
        method, confidence = "consistency", consistency(prompt, answer, n, model=model)
    claims = verify_claims(question, answer, context, verify_model or model) if verify else []
    verdicts = {c["verdict"] for c in claims}
    if "contradicted" in verdicts:
        decision, reason = "abstain", "a claim was contradicted"
    elif confidence < threshold:
        decision, reason = "review", f"confidence {confidence:.2f} < {threshold}"
    elif "unsure" in verdicts:
        decision, reason = "review", "a claim could not be verified"
    else:
        decision, reason = "serve", "confident and verified"
    # High confidence is not correctness: a confidently wrong answer passes every check
    # here. For facts you can look up, look them up instead.
    return {"answer": answer, "decision": decision, "reason": reason, "method": method,
            "confidence": confidence, "claims": claims}


if __name__ == "__main__":
    for q in ["What is the capital of Australia?",
              "What is the sum of the digits of 2 to the power of 20?"]:
        r = self_check(q)
        print(f"{r['decision']:7} {r['method']:21} conf={r['confidence']:.2f} | {q}")
        print("   ", r["answer"].splitlines()[0], "|", r["reason"])
