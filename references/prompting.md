# Prompting Patterns

Five patterns for getting better output from a single model before reaching for retrieval,
agents or fine-tuning. Examples are in `examples/prompting_*.py`.

## Contents
- Decision guide
- Chain-of-Thought (beginner)
- Few-Shot Prompting (beginner)
- Prompt Chaining (intermediate)
- Self-Consistency (intermediate)
- Prompt Optimization (advanced)
- Reviewing an existing prompt

---

## Decision guide

Diagnose the symptom first, then pick the pattern:

| Symptom | Likely fix |
|---|---|
| Output format, tone or labels drift between calls | **Few-Shot** (show, don't describe) |
| Confidently wrong on multi-step reasoning (math, logic, constraints, tracing code) | **Chain-of-Thought** |
| One prompt doing several jobs, each done poorly; hard to tell which part failed | **Prompt Chaining** |
| Right answer *sometimes*; correctness matters more than cost | **Self-Consistency** on top of CoT |
| Stable, high-volume task with a metric; manual tweaking has plateaued or a model upgrade broke prompts | **Prompt Optimization** |

Typical escalation: clear zero-shot instructions → few-shot → CoT (or few-shot CoT) →
chaining → self-consistency → automated optimization. Skip steps when the symptom clearly
points elsewhere, but never jump to an expensive pattern without a baseline measurement.

**Beyond the book:** Many current models have built-in extended reasoning ("thinking"
modes). For those, explicit "think step by step" prompting adds less, and enabling the
model's native reasoning is usually the better first move for reasoning-heavy tasks. The
CoT guidance below still applies to models without it, and to cases where you need the
reasoning visible in the output.

---

## Chain-of-Thought (CoT)

**What:** Ask the model to write out intermediate reasoning steps before the final answer.
It doesn't teach new skills. It changes the generation pattern so the model uses its
own output as working memory, and each step constrains the next.

**Variants**
- *Zero-shot CoT*: append an instruction like "Think step by step, then give your final
  answer after `ANSWER:`." No examples needed.
- *Few-shot CoT*: provide worked input → reasoning → answer examples. More reliable,
  because the model gets a concrete template. Usually the strongest form.
- *Auto-CoT*: keep a store of verified reasoning traces and select relevant ones per input.
  More infrastructure, but scales across diverse problem types.

**Use when:** the task has dependent steps (math, multi-hop QA, code analysis, planning,
interacting constraints), the model "should" know how but keeps getting it wrong, or you
need an auditable trace for debugging.

**Avoid when:** simple retrieval, straightforward classification, or tasks the model
already handles well. It just burns tokens, and over-reasoning can make simple answers worse.

**Pitfalls**
- *Plausible but wrong chains.* An early error gets faithfully built upon, and the thorough
  look makes it harder to spot.
- *Unfaithful reasoning.* The written steps can be a post-hoc narrative, not the actual
  computation. Treat the trace as a useful signal, not ground truth.
- *Over-reasoning* on easy inputs: second-guessing, irrelevant edge cases.
- *Phrasing sensitivity.* "Think step by step", "show your work" and "break this down" can
  give different quality. Test the variants.

**Trade-offs:** responses are roughly 3–10× longer, so higher latency and cost. Output
format varies more, so you need a reliable answer-extraction step (a fixed `ANSWER:`
marker, or structured output). For latency-sensitive production, consider using CoT
offline during development and distilling what you learn into a more direct prompt.

**Pairs with:** Self-Consistency (vote across chains), ReAct (interleave reasoning with
tool calls), Prompt Optimization (find the best CoT phrasing and examples automatically).

**Example:** `examples/prompting_chain_of_thought.py`

---

## Few-Shot Prompting

**What:** Put a few input → output examples before the real query. The model infers the
format, style, granularity and edge-case handling from the examples (in-context learning).
This is programming by demonstration rather than specification. A few good examples can
replace paragraphs of instructions.

**Use when:** you need consistent formatting (exact JSON keys, fixed labels),
classification with a fixed label set (especially subjective categories), style or voice
matching, or information extraction.

**Avoid when:** zero-shot already works reliably, or the context window can't fit the
examples alongside the real input.

**Choosing examples (this matters more than how many)**
- 3–5 diverse, representative examples beat 10 similar ones.
- Cover the real input range, including length and at least one tricky or edge case.
- For classification, include at least one example per label, and keep labels balanced.
- Vary incidental features so the model can't latch onto shortcuts (e.g. every
  "positive" example containing the word "excellent").

**Pitfalls**
- *Biased or unrepresentative examples.* Behavior follows the examples more than most
  people expect.
- *Order effects.* Later examples weigh more. Don't put the easiest one last, and shuffle
  or deliberately order if behavior is inconsistent.
- *Surface-pattern overfitting* (see above).
- *Too many examples* crowd out the input. Returns diminish quickly.

**Trade-offs:** every example is paid on every request (5 × 200 tokens = 1,000 extra
tokens per call). The examples must be maintained as requirements change. You trade
flexibility on unusual inputs for consistency on typical ones. Curating examples can be
expensive for expert domains.

**Beyond the book:** for strict formats, combine few-shot with the provider's structured
output or JSON-schema features, and validate the result against the allowed values. With
providers that support prompt caching, put static examples in a cached prefix to cut their
recurring cost.

**Pairs with:** CoT (few-shot CoT: input → reasoning → output triples), Prompt Chaining
(targeted examples per step), Prompt Optimization (automated example selection).

**Example:** `examples/prompting_few_shot.py`

---

## Prompt Chaining

**What:** Split a complex task into a pipeline of focused LLM calls, where each step's output
is the next step's input. Each step can have its own prompt, examples, model, and even
non-LLM logic (database lookups, rule-based filters, formatting).

**Use when:** the task has naturally separable stages (if you can draw it as boxes and
arrows), stages have different reliability needs (exhaustive extraction vs. lossy
summary), you need conditional branching between steps, or you want cheap models on easy
steps and strong models on hard ones.

**Avoid when:** the task is genuinely atomic (translate a paragraph, answer a simple
question). Splitting only adds latency.

**Design rules**
- Define an explicit **contract** between steps, preferably JSON with a known shape, and
  validate it at each boundary.
- Decide **what context each step needs**. Pass the original input forward when later steps
  depend on it, because steps only see what you give them.
- Add **gates**: check intermediate output and branch, retry or stop early.
- Decide how **partial failure** is handled: retry the failed step, or restart the chain.

**Pitfalls:** error propagation (step 1 misses something and nothing downstream recovers
it), format coupling between steps that breaks silently, cumulative latency (N steps ≈ N
round trips, plus retries), and context loss.

**Trade-offs:** you trade simplicity for control and debuggability, since every
intermediate output is an inspection point. Total tokens go up, but cost per *correct*
output can go down if failures fall. Slower to build at first, faster to iterate
afterwards, because steps can be changed and tested independently.

**Pairs with:** CoT within individual steps, Few-Shot per step, Prompt Optimization per
step (smaller, more tractable searches).

**Example:** `examples/prompting_chaining.py`

---

## Self-Consistency

**What:** Sample the same prompt N times at temperature > 0, extract each final answer,
and take the majority vote. Errors tend to differ across samples while correct answers
converge, so randomness becomes the source of the diversity the vote needs. This is most
effective on top of CoT, since without reasoning, direct answers often don't vary enough.

Why it helps: five chained steps at 90% accuracy each are only about 59% correct end to
end. One sample is a fragile bet.

**Use when:** the answer is short and comparable (a number, multiple choice, yes/no, a fixed
label), the task is reasoning-heavy, and correctness matters more than cost or latency.

**Avoid when:** output is open-ended (emails, summaries have no "majority"), or the model is
already right nearly every time.

**Pitfalls**
- *Systematic bias isn't fixed.* If the model misunderstands the problem, you get a
  confident majority for the wrong answer. Voting fixes random errors only.
- *Answer normalization.* "42", "the answer is 42", "forty-two" and "42.0" must count as
  one answer. Bad normalization silently breaks the technique.
- *Sample count:* 5–10 is a reasonable start. Tune it against your accuracy curve.
- *Temperature:* too low and the samples are identical, too high and they're individually
  unreliable.

**Trade-offs:** cost scales linearly with N. Run the samples **in parallel** so latency
stays close to a single call. It cannot raise the model's ceiling, only filter noise from a
distribution that already contains the right answer. Expect some extra plumbing for
extraction, normalization, voting and ties.

**Beyond the book:** the agreement ratio (votes for the winner ÷ N) is a cheap confidence
signal. Route low-agreement cases to a stronger model or a human.

**Pairs with:** CoT (source of diverse paths), Reflection (sequential critique as an
alternative or follow-up), LLM-as-Judge (pick the best-argued answer instead of the
most common).

**Example:** `examples/prompting_self_consistency.py`

---

## Prompt Optimization

**What:** Treat prompts as code: versioned, benchmarked and tuned by systematic search
against a metric instead of by intuition. Four components:
1. **Pipeline:** prompt template(s) with tunable text and slots (single prompt or chain).
2. **Dataset:** representative inputs with expected outputs or annotations. 50–100
   examples can be enough to drive meaningful optimization.
3. **Evaluator:** exact match, a custom scoring function, or LLM-as-Judge with a rubric.
   It turns "feels better" into "0.87 vs 0.73".
4. **Optimizer:** explores variations by mutating text, having the LLM propose fixes from
   failure analysis, or searching over few-shot example subsets.

Frameworks: **DSPy** (compiles declarative signatures into optimized prompts), **AdalFlow**
(training loop with gradient-like textual feedback), **PromptWizard** (iterative
self-improvement). The principle matters more than the tool.

**Use when:** the task is stable, has measurable quality criteria and enough volume to
justify setup. It's especially valuable after model upgrades (re-run instead of
re-tuning by hand) and for multi-prompt pipelines where steps interact.

**Avoid when:** you're still figuring out the task, format or evaluation criteria. The
results will be invalid by the time they converge. One-off tasks don't justify the setup.

**Pitfalls**
- *Overfitting the eval set.* Keep a held-out test set, ensure diversity, and validate on
  production data.
- *Evaluator quality is the ceiling.* A noisy or misaligned evaluator means you optimize
  the wrong thing. Fix evaluation first.
- *Gaming the judge.* Optimizers can find phrasings that make an LLM judge lenient
  without real quality gains.
- *Compute cost.* A run can mean hundreds to thousands of calls. Budget for it.

**Trade-offs:** significant upfront investment, and optimized prompts may read oddly to
humans (harder to maintain). You take on a dependency on the eval pipeline staying
current. In return you get reproducibility, quantitative version comparison and a
systematic response to model changes.

**Pairs with:** LLM-as-Judge (scalable evaluator for open-ended output), Few-Shot (example
selection is a prime search dimension), CoT (let the optimizer test whether and how CoT
helps).

**Example:** `examples/prompting_optimization.py`, a minimal framework-free harness
(**beyond the book**, since the book has no code for this pattern) showing the
dataset / evaluator / candidate search / held-out-test loop. Use DSPy or similar for real
search.

---

## Reviewing an existing prompt

When asked to review or improve a prompt or LLM call, check:
1. Is there a measurable definition of a correct output, and any test inputs? If not,
   propose creating 20+ before changing anything.
2. Is the instruction unambiguous about format? Would examples or structured output
   remove the ambiguity?
3. Is one prompt doing several jobs? Would a chain with validated contracts help?
4. Does the task need multi-step reasoning, and is reasoning (CoT or native thinking)
   enabled?
5. Is the output parsed robustly and validated, with explicit failure handling?
6. What does it cost per call and per *correct* output, and does that fit the budget?
