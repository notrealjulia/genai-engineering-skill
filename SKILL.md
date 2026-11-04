---
name: genai-engineering
description: Use when designing, building, reviewing or debugging LLM features: prompts (chain-of-thought, few-shot, chaining, self-consistency, prompt optimization), RAG (chunking, embeddings, hybrid/BM25 search, reranking, citations, deep search, agentic RAG), agents (tool calling, ReAct, code execution, plan-and-execute, multi-agent), routing (model routers, cascading, semantic routers) and safety (guardrails, prompt injection via users, documents, files or tool outputs, PII, self-check). Also when output is inconsistent, badly formatted, hallucinated or wrong, an agent loops or picks the wrong tool, or LLM costs run high.
---

# GenAI Engineering

A pattern catalog and decision guide for building production systems with LLMs. Use it to
pick the *simplest pattern that meets the quality bar*, explain the trade-off to the user,
and implement it with the example code in `examples/`.

## How to work with this skill

1. **Clarify the task before picking a pattern.** What is the input, what is a correct
   output, how will we know it is correct, and what are the latency and cost budgets?
   If there is no way to measure correctness yet, say so. Suggest a small eval set
   (20–50 labelled examples) before anything gets optimized.
2. **Start simple and escalate only on evidence.** Every pattern adds tokens, latency or
   moving parts. Recommend the next step up only when the current one demonstrably fails.
3. **Name the pattern and its trade-off** when you recommend or implement one, so the user
   learns the vocabulary and can make the call themselves.
4. **Open the relevant reference file** for details, pitfalls and pairings, and adapt the
   matching example rather than writing from scratch.

## Topic map

| Question is about… | Read |
|---|---|
| Writing or fixing a prompt; wrong format, inconsistent or incorrect answers; reasoning accuracy; splitting a task into steps; systematically tuning prompts | `references/prompting.md` |
| Grounding answers in private or recent data; chunking, embeddings, vector, keyword or hybrid search; poor retrieval; reranking; citations and hallucinations; multi-document or multi-source questions | `references/rag.md` |
| Letting the model call tools or APIs, run code, loop over observations, plan multi-step work or coordinate several agents; agents that loop, misuse tools or cost too much | `references/agents.md` |
| Sending each request to the right model or handler: cheap vs expensive models, escalating on low-quality answers, intent classification; LLM costs too high, simple queries too slow, requests reaching the wrong handler | `references/orchestration.md` |
| Keeping the model inside its rules: prompt injection (from users, documents, uploaded files, CSV cells or tool outputs), validating uploads, PII leaks, unsafe advice, policy checks; flagging confident but wrong answers (logprobs, consistency, claim checks) | `references/safety.md` |

## Example code

`examples/` holds runnable, provider-neutral Python, named `<topic>_<pattern>.py`. Every
example calls `call_llm()` (RAG examples also `embed()`, code execution the native
tool-calling `call_llm_tool()`, self-check `call_llm_logprobs()`) from `examples/llm_client.py`,
which supports OpenAI and Anthropic for chat, and OpenAI or local sentence-transformers for
embeddings. Switch with the `LLM_PROVIDER` / `EMBED_PROVIDER` env vars, or replace those
functions for any other provider. When adapting
an example into the user's codebase, use their existing client and model configuration
instead of copying `llm_client.py`.

## Principles that apply everywhere

- **Measure, don't guess.** "Feels better" is not evidence. Compare variants on the same
  test inputs.
- **Constrain output at the API level when possible.** Prefer the provider's structured
  output / JSON schema / tool-calling features over regex parsing of free text, and
  validate whatever comes back.
- **Fail loudly.** Validate every LLM output that downstream code depends on (allowed
  labels, required JSON fields), and decide explicitly what happens on failure: retry,
  fallback or surface the error.
- **Cost is per *correct* output, not per call.** A pipeline that costs 3× per call but
  rarely needs a retry can be cheaper overall than a single prompt that often fails.

## Source

Patterns condensed from *GenAI Patterns* (Draft, 2026) by Sangam Pandey,
genaipatterns.dev, licensed CC BY-NC 4.0. Examples adapted and extended from the book's
snippets. Notes marked **Beyond the book** are additions.

To add a chapter to this skill, follow `AUTHORING.md`.
