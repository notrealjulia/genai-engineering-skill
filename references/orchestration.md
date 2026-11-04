# Routing & Orchestration Patterns

Three patterns for deciding *where* a request goes: which model answers it (Model Router,
Cascading) and which handler, tool or pipeline serves it (Semantic Router). Examples are in
`examples/orchestration_*.py`. Model tiers with relative costs are defined in
`examples/orchestration_common.py`.

## Contents
- Decision guide
- Cascading (intermediate)
- Model Router (intermediate)
- Semantic Router (intermediate)
- Combining the patterns
- Reviewing an existing routing setup
- Evaluating routing

---

## Decision guide

**First measure what you have.** Routing only pays off if you know the cost and quality of
sending everything to one model. Without a labelled set of real queries you can't tell a
saving from a quality loss.

| Symptom | Likely fix |
|---|---|
| Every request pays frontier-model prices, but much of the traffic is simple | **Model Router** |
| You can't tell upfront which queries are hard; the cheap model is usually right but sometimes wrong | **Cascading** |
| Requests need different handlers, tools, data or pipelines depending on intent | **Semantic Router** |
| Keyword rules misroute ("why was I charged for something that crashed") | **Semantic Router** |
| An LLM classification call adds a second or more to every request | **Semantic Router** (LLM only as fallback) |
| Simple queries are slow because they wait on a large model | **Model Router** |
| A cascade escalates more than 50–60% of queries | Fix the quality gate or the cheap tier (see Cascading) |
| Messages that fit no category land in the wrong handler | An explicit `other` route with a threshold (Semantic Router) |

Recommended build order: one model, measured on an eval set → a **Semantic Router** once the
application serves several distinct intents → a **Model Router** when API cost or latency
becomes a real problem → **Cascading** for the ambiguous middle the router can't classify
confidently. Each layer adds a component to build, monitor and retune.

**Don't route when:** all queries have roughly the same complexity, you only have one model,
cost is small relative to the value of each request, the price gap between tiers is small
(for cascading, a gap of **10× or more** is a strong indicator), or every single request must get top quality.
With only a few intents and low traffic, one LLM classification call is fine; build the
semantic router when latency or cost of that call starts to matter.

---

## Cascading

**What:** Decide *after* seeing an answer instead of before. The cheapest tier answers
first; a **quality gate** checks the answer; if it fails, the query goes to the next tier,
and so on up to the most capable model, whose answer ships regardless (flag it for review if
it also fails the gate). Gate options, cheapest first:
- *Deterministic:* hedging phrases ("I'm not sure", "not enough information"), empty or
  suspiciously short answers for questions that need detail. Free; catches obvious failures.
- *Logprobs:* low average token probability means the model was unsure. Cheap (comes with the
  generation), but models can be confidently wrong.
- *Self-check:* ask the model to rate its own confidence (1–5). One extra cheap call;
  imperfect but a useful signal.
- *LLM-as-Judge:* a separate call, often on the cheap tier, asks whether the answer is
  complete and correct. More robust than self-assessment, more expensive than logprobs.
The best gates combine signals, e.g. no hedging + high logprobs + a passing judge.

**Use when:** traffic is heavy-tailed (many easy queries, few hard ones), the cheapest and
most expensive tiers differ in cost by **10× or more**, hard queries can tolerate extra
latency, you can assess quality without a human, and an occasional transparent retry won't
confuse users.

**Avoid when:** nearly every query needs the top model, multi-tier latency is unacceptable,
you can't build a reliable gate for your domain, or the tiers cost about the same.

**Pitfalls**
- *Gate too lenient:* bad answers ship and trust erodes. Start strict (escalate too often),
  then loosen as data shows what the cheap model handles.
- *Gate too strict:* you pay for the cheap attempt *and* the expensive one, so total cost
  exceeds just using the expensive model. Watch the escalation rate: above **50–60%**, the
  cheap model isn't capable enough or the gate is too aggressive.
- *Latency stacking:* a query that falls through every tier pays every generation plus every
  gate. Cap the tiers (**three** is typical); for latency-critical apps, generate on the
  first two tiers in parallel.
- *Confidently wrong cheap models:* high logprobs, plausible answer, wrong. The hardest
  failure to catch. Judges help but cost more; **domain checks** (recompute the number, look
  the fact up in a database) are more reliable where they exist.
- *Inconsistent style* across tiers. If it matters, run every answer through one formatting
  step.

**Trade-offs:** large savings on easy traffic and built-in quality control, paid for with
complexity, slower hard queries and a gate that needs constant tuning. **Savings are
nonlinear:** when the cheap tier settles **80%** of traffic, the bill drops by about
**70–80%** versus the top model alone (gate overhead included); at **40%** most of that
disappears, because most queries pay for two attempts. The gate is make-or-break and
never set-and-forget. With a response cache in front, only novel queries reach the cascade,
so expect a *higher* escalation rate than in offline tests.

**Beyond the book:** a judge on the same cheap model as the generator shares its blind
spots, and can approve its own tier's wrong answer with full confidence (the book
suggests the cheap tier for the judge; that keeps cost down but doesn't remove the
blind spot). The chapter exercise already asks you to **measure the gate, not only the
models**: on labelled queries, count how often it passes a wrong answer and how often it
rejects a right one. A gate that passes wrong answers shows up as savings, not as errors.
Prefer domain checks for anything checkable. Parse the judge's verdict as JSON and treat a
missing or non-boolean `pass` (including the string `"true"`) as a failure. Not every
provider's API returns logprobs, so check before designing a gate around them.

**Pairs with:** Model Router (a pre-filter: obviously complex queries skip the cheap tier and
start higher), Small Language Models (the better the first tier, the more you save;
fine-tuning it for your domain increases the value of cascading; see the Cost & Performance
chapter), Self-Check (a natural gate; see the Safety & Guardrails chapter), LLM-as-Judge
(Evaluation chapter).

**Example:** `examples/orchestration_cascading.py` (deterministic gate first, validated
LLM-judge verdict on the cheapest tier, pluggable domain-check gate, tier cap, `start_tier`
for a router pre-filter, unverified top-tier answers flagged, cost vs top-tier baseline;
the `__main__` block is a small version of the chapter exercise and checks the gate's calls
against known answers). Chapter exercise: run 20 mixed queries with gpt-4o-mini and gpt-4o
as tiers and record the tier used, the total cost, whether the gate was right, and the
savings against gpt-4o for everything.

---

## Model Router

**What:** Decide *before* generating. A classifier scores each request's difficulty and
sends it to a tier: fast (lookups, formatting, simple instructions), standard (moderate
reasoning, summarization, most conversation) or premium (complex analysis, nuanced writing,
multi-step reasoning). Classifier options:
- *Rules:* keywords ("translate", "summarize") or token counts. Easy to build and debug, but
  brittle: users phrase things your rules didn't anticipate.
- *ML classifier:* logistic regression, random forest or a small neural net trained on queries
  labelled with the tier that should handle them. More robust, but the labels have to come from
  somewhere: usually a stretch of routing all traffic to the top model and grading which
  queries actually needed it.
- *LLM rating:* a cheap model rates complexity 1–5. Needs no training data and works
  surprisingly well; one cheap call is usually small next to the expensive calls it avoids.
Message length is not a proxy: short questions can be deep and long ones simple.

**Use when:** the traffic mixes very different complexities, LLM spend is significant,
latency matters (a small model can answer in **200 ms** where a large one takes **2 s**), you
have several models at different prices, and occasional misrouting is tolerable. At volume,
sending ~**70%** of traffic to a cheap model can decide whether the product is sustainable.

**Avoid when:** all queries are similar, you have one model, cost doesn't matter relative to
the value per request, or every request needs guaranteed top quality.

**Pitfalls**
- *Misrouting hard queries to cheap models:* the main failure. The cheap model answers
  confidently and wrongly, and the user never learns a better model would have been right.
  Build feedback loops to detect it.
- *Overhead exceeding savings:* an LLM router on very cheap queries can raise total cost.
  Use a rule-based fast path for the obvious cases, and cache routing decisions for similar
  queries.
- *Tiers that don't match traffic:* if **95%** of requests land in one tier, the router does
  little. Look at your real query distribution before designing tiers.
- *Stale routing:* models improve and prices change. A model too weak six months ago may be
  fine now. Review routing regularly.
- *Jarring quality shifts* within one session when consecutive queries hit different tiers.

**Trade-offs:** lower average cost and faster simple requests vs another component to
maintain, quality loss on misrouted queries, and ongoing tuning. Classifier accuracy drives
the value: catching **90%** of simple queries saves far more than catching **60%**, so invest
in labelled data. Behind a cache, the router sees mostly novel, harder queries, so production
accuracy can differ from your test set.

**Beyond the book:** when the router can't produce a valid rating (bad JSON, a score outside
1–5, a boolean where a number should be), **fail upward** to the top tier: an unnecessary
expensive call costs money, a misrouted hard query costs a wrong answer nobody notices. Run
the router itself on the cheapest tier, and log every decision (method, score, tier).
Misrouting is invisible to users, so the logs are the only place you can find it.

**Pairs with:** Cascading (route the obvious cases, cascade the ambiguous middle), Semantic
Router (intent first, then the model for that intent), Small Language Models (the premise of
the pattern; Cost & Performance chapter).

**Example:** `examples/orchestration_model_router.py` (rule fast path in both directions,
validated 1–5 LLM rating on the cheap tier, scores spread over any number of tiers, fail-up
fallback, per-tier traffic and cost report)

---

## Semantic Router

**What:** Classify intent by meaning, using embeddings instead of a generative call. Define
routes; give each **5–20** example utterances; embed them once. For each message, embed it,
compare it by cosine similarity against the stored vectors, and dispatch to the best route's
handler, tool or pipeline. An embedding takes **single-digit milliseconds** versus hundreds of
milliseconds to seconds for an LLM classification, and routing runs on every request: a
**2-second** classifier puts a 2-second delay in front of everything. **Hybrid:** above a
confidence threshold, route directly; below it, ask an LLM to classify.

**Use when:** there's a known, stable set of intents needing different processing, you need
**sub-100 ms** routing at scale, you can write good example utterances, and per-request cost
matters.

**Avoid when:** intents overlap so much that humans would disagree, there are hundreds of
fine-grained intents (similarity stops discriminating), intents change constantly (constant
re-embedding), or the right route depends on conversation history rather than the current
message.

**Pitfalls**
- *Overlapping intents* ("update my payment information": billing or account management?)
  cause inconsistent routing. Merge the categories or make their examples more distinct.
- *Poor utterances:* too narrow and similar, and valid messages don't match; too generic, and
  routes overlap. Aim for diverse wording with consistent intent.
- *Embedding blind spots:* some models handle short messages or jargon badly. Test on real
  user messages, not your clean examples.
- *No "unknown" route:* the router always picks the closest route, which is wrong for
  out-of-scope messages. Add an `other` fallback below a confidence threshold.
- *Static routes:* new features create new intents. Periodically check accuracy on real
  traffic and update the routes.

**Trade-offs:** millisecond, low-cost, meaning-based routing vs the upfront work of defining
routes and examples, choosing (maybe fine-tuning) an embedding model, and maintenance as
intents evolve. Edge cases will be misclassified: where misrouting is costly (a refund
request sent to sales), add a threshold and human review for low-confidence cases.
Multi-intent messages ("cancel my subscription, and also I'm seeing an error") get one route;
handle them in the application or split them first. The embedding model sets the ceiling: test
several early.

**Beyond the book:** use a second signal besides the threshold: the **margin** between the
best and second-best route. A message that scores high on two routes is ambiguous even when
both scores clear the threshold, so send it to the LLM fallback too. Score each route by its
*closest* utterance rather than an average, so one well-phrased example can cover a phrasing
the others miss. Calibrate the threshold and margin per embedding model on real messages:
good values differ between models. Validate the LLM fallback's label against the route list
and map anything else to `other`.

**Pairs with:** Model Router (intent here, model choice there), Cascading (unclear messages
go to progressively more capable classifiers instead of a forced guess), Tool Calling (each
route exposes a focused tool set instead of every tool; Agents chapter), Agentic RAG (routing
to one data source is the simplest form; RAG chapter).

**Example:** `examples/orchestration_semantic_router.py` (utterances embedded once with
`rag_common.VectorIndex`, closest-utterance scoring, threshold + margin, validated LLM
fallback, `other` route to a human queue)

---

## Combining the patterns

A typical layered setup: the **Semantic Router** picks the intent and its handler; within it,
the **Model Router** picks a tier (obvious cases by rule, the rest by a cheap rating); for the
ambiguous middle, **Cascading** starts at the routed tier and escalates on a failed gate
(`cascade(query, start_tier=...)` in the example). Put a response cache in front of all of it,
and remember the cache changes the difficulty mix the routers see.

---

## Reviewing an existing routing setup

1. Is there a labelled set of real queries, and a baseline of "everything to one model" for
   cost and quality?
2. Do the tiers really differ in price (10×+ is a strong signal for cascading) and does traffic spread across
   them, rather than ~95% landing in one?
3. Is there a cheap fast path, so routing never costs more than it saves?
4. What happens on an invalid routing decision? Does it fail toward quality (top tier, or
   `other`) and get logged?
5. Cascading: what does the gate combine? Is its accuracy measured (wrong answers passed,
   right answers rejected), and is the escalation rate monitored against the 50–60% warning
   line?
6. Is the number of tiers capped, and is worst-case latency (every tier plus every gate)
   acceptable?
7. Semantic routing: 5–20 diverse utterances per route, an `other` route, a calibrated
   threshold and margin, and a plan for multi-intent messages?
8. Is every decision logged (route, method, score, tier, cost) so misrouting can be found?
9. Is routing reviewed when models, prices or product features change?

## Evaluating routing

- **Routing accuracy:** on 50–100+ labelled real messages (including out-of-scope and
  multi-intent ones), measure accuracy per route and the confusion between routes. Overlap
  shows up as pairs that swap often.
- **Model routing and cascading:** report cost and latency per **correct** answer (Cost is
  per correct output) against the single-model baseline, the share of traffic per tier, the
  escalation rate, and quality on the queries routed to cheap tiers. That's where misrouting
  hides.
- **Gate accuracy** (from the chapter exercise): for cascades, score the gate itself: wrong answers it
  passed vs right answers it rejected. Use LLM-as-Judge or exact answers (Evaluation chapter).
- Re-run after changing models, prices, tiers, utterances, thresholds or the cache.
