# Authoring guide: adding a chapter to this skill

Instructions for Claude (or a human) adding the next chapter from the *GenAI Patterns*
book. The goal is that every chapter matches the existing ones (`references/prompting.md`,
`references/rag.md`) in structure, depth and code quality.

Use it like: *"Add chapter 3 (Agents) to the genai-engineering skill following
AUTHORING.md. The book text is at <path>."*

## Inputs

- Book text: extract with `pdftotext -layout book.pdf book.txt`, then read only the chapter
  in question (headers look like `CHAPTER 03 -- AGENTS`, with two hyphens). **Never copy the book into the
  skill folder.** It's licensed CC BY-NC 4.0, and the skill should hold condensed guidance,
  not the text.
- Read `references/rag.md` and one example (e.g. `examples/rag_grounded.py`) first, as the
  model to match.

## Size budgets

Every question loads `SKILL.md` plus one reference file, so size is paid on every use, and
a long file gets skimmed. Hard caps, checked before finishing any change:

| File | Target | Hard cap |
|---|---|---|
| `SKILL.md` description | | 1024 characters |
| `SKILL.md` body | under 80 lines | 100 lines |
| `references/<topic>.md` | 200–350 lines | 375 lines and 3,200 words |
| `examples/<topic>_<pattern>.py` | 40–120 lines | 150 lines |
| `examples/<topic>_common.py` | | 150 lines |
| One **Beyond the book** note | 2–4 lines | 6 lines |

Check with `wc -l SKILL.md references/*.md examples/*.py` and `wc -w references/*.md`.

- **Cut before you add.** A change that would break a cap has to pay for itself first:
  merge overlapping pitfalls, drop narrative and repeated advice, shorten examples. Never
  cut the book's concrete numbers to make room.
- Fixes from an eval (step 5) go in the shortest form that closes the gap, usually one
  pitfall bullet or one review-checklist question, not a new section.
- If the description runs short of room for a new chapter, shorten older chapters' terms;
  don't go over 1024.
- A file that can't fit its cap without losing guidance should become two files. Ask the
  user before splitting.
- Report the sizes in the chapter summary, and any file above its target.

## 1. Reference file: `references/<topic>.md`

Structure, in this order:
1. Title + one line saying what the chapter covers + where the examples live.
2. `## Contents` listing each pattern with its difficulty (beginner/intermediate/advanced).
3. `## Decision guide`: a **symptom → pattern** table, plus a recommended build or
   escalation order, ending with when *not* to use the advanced patterns.
4. One `## <Pattern>` section per pattern, always with these fields:
   **What** · **Use when** · **Avoid when** · **Pitfalls** · **Trade-offs** · **Pairs with** · **Example**
5. `## Reviewing an existing <topic> system`: a numbered checklist.
6. Optional evaluation section if the chapter implies one.

Writing rules:
- **Condense and paraphrase.** Keep every concrete number from the book (latencies,
  thresholds, sample counts, percentages). They're what make guidance actionable.
- Write for decisions: favour "use X when…, because…" over narrative.
- Additions not in the book (newer practice, provider features, corrections) go in a
  clearly labelled **Beyond the book:** paragraph. Never blend them silently.
- Cross-reference other chapters' patterns by name (e.g. "see Tool Calling in the Agents
  chapter").
- Stay within the size budgets (see *Size budgets* below).

## 2. Examples: `examples/<topic>_<pattern>.py`

One file per pattern, **including patterns the book gives no code for** (write your own,
and say so in the docstring).

Every example must:
- Start with a docstring: what it does, plus attribution: either "Adapted from GenAI
  Patterns (S. Pandey, CC BY-NC 4.0)" or "Based on the pattern described in … the book has
  no code for this pattern."
- Call models only through `llm_client` (`call_llm()`, `call_llm_tool()`, `embed()`). No
  provider SDK calls in examples.
- Be stdlib + numpy only, unless the pattern really needs a library. Then import it
  inside the function that uses it.
- Production habits, visible in the code:
  - parse LLM output defensively (JSON with fallbacks) and **validate** it. Always parse
    LLM JSON with the shared, fence-tolerant `parse_json()` in `examples/agents_common.py`.
    **Never use plain `json.loads`.** Real models wrap JSON in ```json fences and add prose
    around it, so `json.loads` silently falls back to its default on most real replies.
  - fail loudly or degrade explicitly, and never silently pass bad output downstream
  - **hard limits** on loops (steps, calls, time) and a `stop_reason` in the result
  - return intermediate results and a trace for debugging
  - short comments explaining the *why* of each design choice, linked to a pitfall
    from the reference
- Stay short (see *Size budgets*). Shared helpers go in `<topic>_common.py`.

## 3. Test before finishing

Write a test script **outside the skill folder** that replaces `llm_client` with a mock
(a fake module in `sys.modules` providing `call_llm`, `embed` and `EMBED_MODEL`), then
covers each example's happy path **and** its failure paths (bad JSON, invalid labels, budget
exhaustion, tool errors). All must pass.

Mock replies must look like real model output, not just clean JSON: include **fenced JSON**
(```json … ```), **extra prose around the JSON** ("Sure! Here it is: {...}"), and
**booleans returned as strings** (`"false"`). Clean-JSON mocks hid fallbacks that fired on
almost every real reply.

After the mock tests pass, run every new example's `__main__` block once against the real
API (if a key is set). Report any example that **degrades silently** (empty expansions,
every answer flagged, refusals where the sources answer, loops that stop early with no
gaps), not just the ones that crash. A green exit code isn't a working example.

## 4. Wire it up

- Add a row to the **Topic map** in `SKILL.md`.
- Extend the `description` in `SKILL.md`'s frontmatter with the new chapter's key terms
  (e.g. "agents, tool calling, ReAct, multi-agent"), keeping it under ~1024 characters.
- Don't change existing chapters unless asked.

## 5. Evaluate with and without the skill

Mock tests prove the examples run; this step checks the skill makes answers *better*.

**Add one eval prompt for the new chapter** in
`~/Projects/genai-engineering-workspace/evals/<chapter>-<scenario>/` (`prompt.md`,
`case.yaml`, one file per checklist item in `graders/`). **Never put evals in the skill
folder:** agents that list the skill see the grader file names, which leak the checklist.
- The prompt describes **symptoms, not diagnoses**, the way a real user would ("it often
  says it can't find things that are in the handbook", not "it refuses negative answers").
- The checklist is what an **experienced AI engineer** would expect, in their words, not
  phrases from our reference file. 5–8 items. Include items a good answer without the
  skill could miss (security, measurement, failure handling); drop items every answer
  passes, since they can't tell the two apart.

**Run it 2+2**, together with the negative prompt `negative-large-csv-log-counts` (a plain
Python question the skill should not change): two agents told "a skill is available at
<skill path>; use it if relevant", and two told not to read the skill folder. Every agent
gets the same prompt, must not call a real model API, and saves `answer.md`, its code, and
(with-skill only) `skill_usage.txt` to `iteration-<N>/<case>/<config>/run-<n>/outputs/`.

**Grade blind.** Copy each pair's outputs (without `skill_usage.txt`) into
`blind-<N>/<case>/pair-<n>/A` and `/B` in random order, and keep the A/B mapping outside
the workspace. A fresh grader agent per prompt reads only its `blind-<N>/<case>/` folder,
scores every answer against the checklist (substance, not keywords; it reads the code),
and picks the better answer per pair. Reveal the mapping only after all grading is done.

**Report in the chapter summary:** checklist passes per run with and without the skill,
the grader's overall pick per pair, tokens and time per configuration, whether the skill
was consulted on the negative prompt, and any caveat (leaks, non-discriminating items,
bugs the grader found). Two runs per side is a small sample: treat a one-item
difference as a hint, not a result.
