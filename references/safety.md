# Safety & Guardrails Patterns

Two patterns for keeping an LLM application inside its rules and flagging answers that may
be wrong: Guardrails (policy checks around the model) and Self-Check (confidence and claim
verification before an answer is served). Examples are in `examples/safety_*.py`; shared
detectors (injection rules, PII redaction, untrusted-data wrapper, upload validation) are
in `examples/safety_common.py`.

## Contents
- Decision guide
- Guardrails (intermediate), with Output Structuring and Prompt Templating
- Untrusted data: documents, files and tool outputs
- Self-Check (advanced)
- Reviewing an existing system for safety
- Evaluating guardrails and self-check (the chapter exercises)

---

## Decision guide

A system prompt is a polite note, not a control: prompt injection can override it, and
the model drifts without any attacker. Safety has to be structural, in code around the
model.

| Symptom | Pattern |
|---|---|
| Users get the bot to ignore its instructions, reveal its prompt or change role | **Guardrails**: input layer |
| A retrieved page, uploaded file, CSV cell, email or tool result makes the model do something nobody asked for | **Guardrails**: retrieval layer, see *Untrusted data* |
| Customer emails, phone numbers or card numbers show up in answers | **Guardrails**: retrieval layer (redact before the context) + output layer |
| The model proposes destructive or out-of-scope tool calls | **Guardrails**: execution layer (see Tool Calling in the Agents chapter) |
| Answers give medical, legal or financial advice without caveats | **Guardrails**: output layer |
| A broken or hostile upload crashes the pipeline or reaches the model | Validate the file first (*Untrusted data*) |
| Fluent, confident answers that are wrong (invented citations, numbers, dosages) | **Self-Check**; ground in sources first if you can (RAG chapter) |
| You need to prove to auditors which controls ran on each request | **Guardrails** with a logged decision per layer |
| High-volume generated text must be exactly right and on-brand | **Output Structuring** or **Prompt Templating** |

Build order: least privilege and execution guardrails on anything that acts → input and
output rules (free) → retrieval-layer screening once untrusted content enters the context
→ an LLM classifier where rules miss too much → Self-Check for factual answers users act
on. A small internal tool for a trusted team can stop early with lightweight guardrails; a
consumer product needs all four layers.

**When not to:** creative and brainstorming features, where "hallucination is a feature";
and Self-Check when you can verify against a database in real time: look the fact up.

---

## Guardrails

**What:** Independent processing layers around the model, like middleware. Each one
intercepts data at one point, checks it against your policy, and **passes**, **modifies**
(redacts, rewrites, adds a disclaimer) or **rejects** it. Four insertion points:
- *Input* (user → model): injection attempts, toxic or out-of-scope requests.
- *Retrieval* (knowledge base → model): filter and redact documents before they enter the
  context window.
- *Execution* (model → tools/APIs): proposed calls stay within allowed parameters and
  never do destructive operations unchecked.
- *Output* (model → user): hallucinated claims, inappropriate content, leaked data.

The layers run outside the model, so they can use rules, regexes, classifier models or a
second LLM call.

**Use when:** the application is user-facing and arbitrary output is unacceptable; it
handles personal, financial or health data; you need auditable controls (regulated
industries); and always, as execution guardrails, when the model can act (APIs, database
writes, emails): "a security incident waiting to happen" otherwise.

**Avoid when:** never avoid them entirely; scale them down. For a small trusted internal
team, lightweight rules may suffice.

**Pitfalls**
- *Too rigid:* a filter that blocks anything mentioning a sensitive topic blocks real use
  (a medical bot refusing symptoms because "pain" tripped a filter). Calibrate on real user
  data. The rules in the example block "Can I ignore the setup instructions in the box?",
  a legitimate question. Flag or route borderline inputs instead of blocking them (see
  *Evaluating*).
- *One-time setup:* injection techniques keep evolving; maintain the rules like any other
  security infrastructure.
- *Correlated failure:* an LLM guarding an LLM can be fooled by the same injection.
  Combine rule-based and model-based checks. **Beyond the book:** run the rules first: they
  are free and can't be talked out of a verdict, and an input they reject never reaches a
  model an attacker could steer.
- *Latency:* each layer adds a step; chained LLM checks can **double or triple** response
  time. Run independent checks in parallel and use small, fast classifiers.
- *Fail-open parsing (Beyond the book):* a classifier reply that doesn't parse must not
  count as "safe". Validate the verdict (JSON, allowed labels) and choose explicitly:
  fail closed for consumer apps, fail open (with logging) where blocking costs more.
- *Leaks through the output (Beyond the book):* check the output for system-prompt text
  and PII, and allowlist URLs and image links: a markdown image pointing at an attacker's
  server can carry data out in its query string.

**Trade-offs:** more stages, configuration and failure modes to test and maintain. A
real tension between safety and utility: tighter filters, more false positives, so
thresholds are an ongoing decision. A classifier on every input and output **doubles
inference cost at minimum**; rule checks are essentially free but catch fewer edge cases.
Users notice "three seconds instead of one".

**Pairs with:** Self-Check (guardrails catch what *should not* be said, self-check what
*might not be true*), Grounded Generation (verifiability built into the output, see the
RAG chapter), LLM-as-Judge (pre-screen outputs, send only borderline ones to humans), Tool
Calling (the execution layer).

**Example:** `examples/safety_guardrails.py` (rules before a cheap-tier LLM classifier with
a validated, fail-closed JSON verdict; retrieval screening and PII redaction with a
public-contact allowlist; output checks for system-prompt leaks, PII and an optional
advice disclaimer; a per-layer decision trace). The execution layer is
`agents_common.execute_tool_call` (allowlist, argument validation, confirmation).

### Output Structuring

Separate content from format: assemble the facts with low-hallucination methods (database
queries, OCR, structured API calls, template lookups), then let the model organize and
phrase them. The model is an editor, not a researcher. Use it for catalog-scale content
where the data exists in structured form, and for domains with low tolerance for error
but high expectations of readability (financial reports, medical information, legal
summaries, technical documentation). *Pitfall:* the model can still distort meaning
while reformatting: name the fields that must appear verbatim, use a structured output
that requires every input field, and remember that garbage in is polished garbage out.

### Prompt Templating

Constrain output to predefined templates. Offline, a model drafts a finite set of
templates with placeholders and a human approves each once; online, the system picks a
template and fills the placeholders by plain string replacement, **with no model at
inference time**. That rules out injection and makes every message traceable to an
approved template. Use it for high-volume personalized content (emails, notifications,
product descriptions) where a bad output is costly, for auditability, and for strict brand
guidelines. *Pitfalls:* combinatorial explosion as personalization axes multiply,
mechanical-sounding text without enough variants, and the review step as a bottleneck for
new products or markets.

---

## Untrusted data: documents, files and tool outputs

**Beyond the book** (the book's retrieval guardrail covers redaction; this section extends
it to injection). The most dangerous injections don't come from the user. Any text the
model reads is a channel: retrieved documents, web pages, emails, PDFs, **CSV cells**,
code comments, and **tool outputs** (an API response, a search result, a scraped page).
The user never sees the instruction; the model follows it with the user's privileges.

The risk is highest when one agent has all three of: **private data**, **untrusted
content**, and a way to **send data out** (email, HTTP, links in its output). Remove one
of the three and an obeyed injection can't do much.

What to do, in order of how much it protects:
1. **Least privilege.** The model can only misuse what it can reach: read-only tools by
   default, confirmation on side effects, no secrets or network in code sandboxes (see
   Code Execution in the Agents chapter).
2. **Validate files before anything reads them.** Allowlist file types, cap sizes, check
   the bytes (no binary in a `.txt`, valid UTF-8), and for tables check that it parses, the
   expected columns exist and every row has the same width. Reject with an error the user
   can act on, *before* any LLM call.
3. **Screen untrusted text like user input.** Run the injection rules (and, if needed,
   the classifier) on documents and tool results; quarantine what matches instead of
   passing it on with a warning.
4. **Mark data as data.** Wrap untrusted text in tags with its source, escape anything
   that could close the wrapper early, and tell the model in the system prompt that tagged
   text is never instructions. This lowers the hit rate; it is not a boundary.
5. **Show the model as little as needed.** A schema and a few truncated sample rows, not
   the whole file; summaries of tool results, not raw pages.
6. **Check the output** for leaks and unexpected links (see Guardrails pitfalls).

**Example:** `safety_common.validate_upload()` and `agents_common.inspect_csv()` (file
checks), `safety_guardrails.retrieval_guardrail()` (screening, redaction, wrapping),
`safety_common.wrap_untrusted()`.

---

## Self-Check

**What:** Before serving an answer, estimate how sure the model was and verify its claims,
then serve, flag for review or abstain. Models sound equally confident when they are wrong
(a plausible court citation with a realistic case number; a fabricated dosage). This is a
structural property of next-token prediction, not a bug the next model fixes. Signals,
from cheapest:
- *Token logprobs:* the probability of each generated token (`p("Paris") = 0.92`). Watch
  the tokens that matter, the values more than the keys: "capital: Nairobi" can be
  confident while the digits of "population: 4,397,073" are not. Flag below a threshold.
- *Perplexity:* token probabilities aggregated over the sequence; unusually high values
  against your baseline go to review.
- *Sampling consistency:* generate the same answer several times at non-zero temperature
  (the book uses five). Consistent answers suggest well-represented knowledge; different
  answers mean the model is guessing. **Needs no logprobs**, so it works on any API.
- *Claim verification:* a second pass critiques the answer against criteria (the book's
  code: `VERDICT: PASS/FAIL`).
- *A trained classifier* on probability features of verified-correct vs hallucinated
  outputs: fast and domain-tuned, but needs labelled data.

**Use when:** the application produces factual claims users might act on (legal, medical,
financial), the facts are too diverse for real-time lookup, and you have a plan for
flagged answers (human review, a retrieval-based fallback, or telling the user the system
isn't confident).

**Avoid when:** creative or brainstorming output; or a ground-truth source exists: check
against it instead.

**Pitfalls**
- *Thresholds aren't universal:* rare but correct tokens (unusual names, jargon) have low
  probabilities. Calibrate per domain on labelled data.
- *Cost of consistency:* five samples is **5× the cost** and, unless run in parallel,
  roughly 5× the latency. Compromise from the book: sample only when logprobs flag
  uncertainty.
- *False security:* high confidence isn't correctness. Self-check catches uncertainty, not
  confident mistakes, which are common where the training data itself is wrong.
- *Over-filtering:* aggressive thresholds flag correct answers and erode trust; monitor on
  production data. In our run of the exercise with gpt-4o-mini, the checks caught all 5
  wrong answers but also flagged 1 of 5 correct ones; with gpt-4o the verifier
  "contradicted" a correct letter count.
- *Self-verification is correlated (Beyond the book):* a model checking its own knowledge
  shares the generator's blind spots. Verify against retrieved context when you have it,
  or use a different model as the verifier.

**Beyond the book:** Anthropic's API doesn't return logprobs, and some OpenAI reasoning
models don't either: check before designing around them. The fallback is sampling
consistency plus claim verification, which needs only text. Compare *short answers*, not
whole replies (ask for the answer alone on the first line and normalize it), or every
sample "disagrees" on wording.

**Trade-offs:** logprob monitoring is nearly free when the API returns them; consistency
multiplies cost linearly; a classifier needs labelled data and maintenance. Extra calls
add latency: for interactive use, run checks asynchronously and show a confidence
indicator after the first response. Self-check detects hallucination *sometimes*; it
doesn't fix it. Critical applications combine it with retrieval grounding and human
review.

**Pairs with:** Guardrails (the output layer acts on the confidence signal), Grounded
Generation (citations say where a claim came from; confidence says how sure the model
was), LLM-as-Judge (an independent second opinion that catches different errors),
Self-Consistency in the Prompting chapter (the same sampling, used to pick an answer).

**Example:** `examples/safety_self_check.py` (logprob confidence via
`llm_client.call_llm_logprobs()`, consistency only when logprobs look low, a consistency
fallback when the provider has no logprobs, claim-by-claim verification with a validated
JSON verdict, a serve/review/abstain decision with its reason)

---

## Reviewing an existing system for safety

1. What can the model *do*? List every tool and data source; is each one least-privilege,
   with confirmation on side effects?
2. Is safety enforced in code, or only asked for in the system prompt?
3. Which of the four layers exist, and is every decision logged with its reason?
4. Are documents, uploads, CSV cells, emails and tool outputs screened and marked as data,
   not just user input? Does any agent combine private data, untrusted content and a way
   to send data out?
5. Are uploaded files validated (type, size, encoding, schema) before any model or tool
   reads them, with a clear error instead of a crash?
6. Do rules run before LLM checks? Are classifier verdicts validated, and is the
   fail-open / fail-closed choice explicit?
7. Is PII redacted before it enters the context, not only on the way out? Are public
   contact details exempt?
8. Does the output layer catch system-prompt leaks and unexpected links or images?
9. Are detection and false-positive rates measured on a labelled set, and re-measured when
   rules, prompts or models change?
10. For factual answers: is there a confidence signal that works on this provider, and a
    defined path for flagged answers?

## Evaluating guardrails and self-check

The chapter's two exercises are runnable in `examples/safety_exercises.py` (and covered,
with scripted models, by the mock tests):
- *Input guardrail:* 10 injection attempts and 10 benign inputs, including benign ones
  close to the line ("ignore the setup instructions"). Report **detection rate** and
  **false-positive rate**, for rules alone and rules + classifier. Our run: rules 80%
  detection / 10% false positives; with a gpt-4o-mini classifier 100% / 20%. Blocking 2 of
  10 benign messages is too high for most customer-facing bots: refused customers leave
  or escalate. Block only clear-cut cases (rule hits, confident classifier labels), and
  flag or route borderline ones (a stricter path with no tools, a clarifying question, or a
  human) instead of refusing.
- *Self-check:* 10 factual questions with known answers. Report how many wrong answers it
  catches and how many correct ones it flags. Use a model that gets some wrong (a cheap
  tier): a strong model that answers all ten correctly leaves nothing to catch.

**Beyond the book:** grow both sets from real traffic (blocked inputs users complained
about, attacks from logs, answers users corrected), keep injections that arrive through
documents and tool outputs in the set, and track both rates over time: a guardrail
change that raises detection but doubles false positives is a regression for most
products.
