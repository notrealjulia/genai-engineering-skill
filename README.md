# genai-engineering: a Claude Code skill

A pattern catalog and decision guide for building production systems with LLMs, packaged
as a [Claude Code skill](https://docs.claude.com/en/docs/claude-code/skills). When you
design, build, review or debug LLM-powered code, Claude loads the skill and uses it to
pick the *simplest pattern that meets the quality bar*, name the trade-off, and adapt a
runnable example instead of writing from scratch.

## What's covered

| Chapter | Reference | Patterns (one runnable example each) |
|---|---|---|
| 1. Prompting | `references/prompting.md` | Chain-of-thought, few-shot, prompt chaining, self-consistency, prompt optimization |
| 2. RAG | `references/rag.md` | Basic RAG, semantic indexing, hybrid retrieval (BM25 + vectors, RRF, HyDE), retrieval refinement (filter, dedupe, rerank, compress), grounded generation (citations, claim verification), deep search, agentic RAG |
| 3. Agents | `references/agents.md` | Tool calling, ReAct, code execution, plan-and-execute, multi-agent (pipeline and supervisor) |
| 4. Routing & Orchestration | `references/orchestration.md` | Cascading (quality-gated escalation), model router, semantic router |

Each reference has a symptom → pattern decision guide, and per pattern: what it is, when to
use and avoid it, pitfalls, trade-offs, pairings, and a review checklist. More chapters are
in progress; `AUTHORING.md` describes how new ones are added.

## Install

The repository root *is* the skill folder (it contains `SKILL.md`). Clone it into your
Claude Code skills directory. Claude Code picks it up in new sessions.

**macOS / Linux**
```bash
git clone https://github.com/notrealjulia/genai-engineering-skill.git ~/.claude/skills/genai-engineering
```

**Windows (PowerShell)**
```powershell
git clone https://github.com/notrealjulia/genai-engineering-skill.git "$env:USERPROFILE\.claude\skills\genai-engineering"
```

To use it in one project only, clone it into that project's `.claude/skills/genai-engineering`
instead. Update with `git pull` in the skill folder.

Check it's loaded: start `claude` and ask *"what skills are available?"*, or just ask a
question like *"my RAG answers cite the wrong document, what should I check?"*.

## Running the examples

The examples in `examples/` are plain Python (3.10+), standard library plus numpy, and
call models only through `examples/llm_client.py`. That file is the one place to swap in
another provider.

```bash
pip install numpy openai        # default: OpenAI for chat and embeddings
pip install anthropic           # optional: Claude for chat
pip install sentence-transformers  # optional: local embeddings, cross-encoder reranking
```

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (none) | Required for the default chat and embedding provider |
| `ANTHROPIC_API_KEY` | (none) | Required when `LLM_PROVIDER=anthropic` |
| `LLM_PROVIDER` | `openai` | `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-4o` / `claude-sonnet-5-5` | Chat model override |
| `EMBED_PROVIDER` | `openai` | `openai` or `local` (sentence-transformers). Anthropic has no embeddings API, so RAG examples with Claude still need one of these |
| `EMBED_MODEL` | `text-embedding-3-small` / `all-MiniLM-L6-v2` | Embedding model override |

Run any example from the `examples/` folder:

```bash
cd examples
export OPENAI_API_KEY=...          # PowerShell: $env:OPENAI_API_KEY = "..."
python rag_grounded.py
```

Notes:
- `agents_tool_calling.py` asks for `[y/N]` approval before a side-effecting tool call, so
  run it in an interactive terminal.
- `agents_code_execution.py` runs model-generated Python in a subprocess with limits. That
  is **not** a security sandbox. Read the warning in the file before using it on untrusted
  input.
- All examples have been run against the real OpenAI API (gpt-4o, text-embedding-3-small).
  The Anthropic chat path is implemented but hasn't been tested end to end yet.

## Attribution

The pattern content is condensed and adapted from **[GenAI Patterns](https://genaipatterns.dev)**
(Draft, 2026) by **Sangam Pandey**, licensed
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Changes made: the book text is condensed and paraphrased into decision-oriented reference
files; additions not in the book are marked **Beyond the book**; the book's code snippets
are rewritten and extended, and new examples are written for patterns the book has no code
for (each example's docstring says which).

This repository is licensed CC BY-NC 4.0 as well. See [LICENSE](LICENSE).
