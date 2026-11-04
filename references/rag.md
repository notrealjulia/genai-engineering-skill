# RAG Patterns

Seven patterns for grounding LLM answers in your own data. Examples are in
`examples/rag_*.py`. Shared building blocks (chunking, BM25 and vector indexes, context
formatting) are in `examples/rag_common.py`.

## Contents
- Decision guide
- Basic RAG (beginner)
- Semantic Indexing (intermediate)
- Hybrid Retrieval (intermediate)
- Retrieval Refinement (intermediate)
- Grounded Generation (advanced)
- Deep Search (advanced)
- Agentic RAG (advanced)
- Evaluating a RAG system
- Reviewing an existing RAG pipeline

---

## Decision guide

**First, debug the right stage.** Every RAG failure is either *retrieval* (the right chunk
never reached the prompt) or *generation* (it was there and the model misused it or
ignored it). Log the retrieved chunks for every query and check which one it was before
changing anything.

| Symptom | Likely fix |
|---|---|
| Model doesn't know your private or recent data | **Basic RAG** |
| Keyword search misses paraphrases ("cancel subscription" vs "account termination"), or the corpus is multilingual | **Semantic Indexing** |
| Vector search misses exact terms (error codes, product IDs, names), or users' casual wording differs from expert docs | **Hybrid Retrieval** (BM25 + vectors; then query expansion or HyDE) |
| The right chunk is retrieved but ranked low, or chunks are noisy, duplicated or outdated | **Retrieval Refinement** (rerank, filter, dedupe, compress) |
| Answers invent details, cite things that don't exist, or answer out-of-scope questions | **Grounded Generation** |
| Questions need synthesis across many documents ("compare", "trend", "summarize all") | **Deep Search** |
| Queries need different sources (docs, SQL, web, APIs), or sometimes no retrieval at all | **Agentic RAG** (start with routing) |

Recommended build order (the book's own advice): Basic RAG → hybrid search as the
production default → refinement (reranking). Add Deep Search or Agentic RAG only when an evaluation shows the
simpler patterns are insufficient. **Out-of-domain detection and source citations are
table stakes for any RAG system serving real users.**

**Beyond the book:** if the whole corpus fits comfortably in the model's context window
(and it's small and fairly static), consider skipping retrieval and putting it all in the
prompt, with prompt caching. Measure the cost and quality against RAG before building
an index.

---

## Basic RAG

**What:** Two pipelines. *Indexing* (offline): split documents into self-contained chunks
of a few hundred tokens and store them in a searchable index. *Query* (online): retrieve
the top-k chunks for the question, put them in the prompt, and instruct the model to
answer only from them. The separation gives you an audit trail: show sources, and log
retrieval separately from generation.

**Use when:** answers exist in a corpus the model wasn't trained on (internal docs, help
articles, case files, clinical guidelines) and the latency budget can absorb retrieval
(~100–500 ms).

**Avoid / not enough when:** the answer requires multi-step inference not written down in
any single document. RAG is not a reasoning pattern. Add CoT, or see Deep Search.

**Pitfalls**
- *Chunk size.* Too large wastes context and crowds out other chunks. Too small loses
  meaning ("see the table above" in a different chunk). Tune per document type.
- *Poor relevance filtering.* Irrelevant chunks are noise the model may anchor on. Use a
  minimum score threshold, which needs calibrating.
- *Context overflow / lost in the middle.* Too many chunks hit token limits, and models
  lose track of information buried mid-context. Focused beats voluminous.

**Trade-offs:** at least two operations per query (search plus generation) adds
latency. Caching frequent queries helps but needs invalidation. Index freshness is ongoing
work: stale indexes give confidently outdated answers, and the ingestion pipeline is often
more work than the query side.

**Pairs with:** Semantic Indexing (the highest-impact retrieval upgrade), Tool Calling
(fetch live data instead of static indexes).

**Example:** `examples/rag_basic.py`

---

## Semantic Indexing

**What:** Embed chunks into dense vectors (typically 384–1536 dimensions) where nearness
means similar meaning. Embed the query with the **same model** and return the nearest
chunks (cosine similarity, approximate nearest-neighbour search in a vector DB). This
handles synonyms, paraphrases and cross-lingual matches without rules.

**Use when:** users ask in natural language with different vocabulary from the docs
(nearly always true for customer-facing search), the corpus is multilingual, or you keep
patching keyword search with synonym lists and boost rules.

**Pitfalls**
- *Domain mismatch.* General embedding models can struggle with legal, medical or code
  text. Test on your own queries, not leaderboards.
- *Dimensionality.* Higher dimensions mean more nuance plus more storage and compute.
  Lower dimensions are cheaper but may conflate concepts.
- *Semantic drift.* Surface-level confusion ("Python exceptions" retrieving snake facts).
  Always inspect results during development, and use thresholds and reranking.
- *Cold start.* Without query logs you can't validate the vectors. Build a small set of
  query → expected document pairs and measure recall before launch.

**Trade-offs:** more expensive than keyword search at every layer (embedding, storage,
search). Not justified for small corpora where keyword search works. **Model lock-in:**
switching embedding models means re-embedding everything, since vectors from different
models are incompatible. Store the raw text and model name alongside the vectors, and keep
a re-runnable indexing pipeline.

Embed documents once at index time and store the vectors alongside the text;
re-embed only chunks that change. **Beyond the book:** some embedding models expect
different prefixes or input types for queries vs documents, so check the model's docs.

**Pairs with:** Basic RAG (swap it into the retrieval step), Hybrid Retrieval (most
production systems end up combining vectors with BM25).

**Example:** `examples/rag_semantic_index.py`

---

## Hybrid Retrieval

**What:** Close the vocabulary gap between how users ask and how documents are written.
There are four techniques, in order of recommended adoption:

1. **Hybrid search:** run BM25 (exact keywords) and vector search in parallel and combine
   the scores with weight alpha (0 = pure keyword, 1 = pure vector, production is often
   0.5–0.7). **Make this your production default.** It's cheap and consistently helps.
2. **Query expansion:** rewrite the query into 3–4 phrasings, search each and merge.
   Helps when queries are short (3–5 words) or ambiguous.
3. **HyDE (Hypothetical Document Embeddings):** have the LLM write a hypothetical answer,
   embed *that*, and search with it. Its vocabulary resembles the documents'. Use for
   severe jargon mismatch.
4. **Graph-based retrieval:** link chunks to entities, parents and related chunks, and
   expand matches along the graph. Use when questions span documents and the corpus has
   strong entity structure and cross-references.

**Use when:** relevant documents often fall outside the top-k, or users rephrase repeatedly
before getting a good answer.

**Pitfalls**
- *HyDE with a confidently wrong hypothetical* retrieves the wrong concept, which is
  risky in specialist domains where the LLM lacks knowledge.
- *Expansion drift* pulls in off-topic results (5 queries × 10 results = 50 candidates).
- *Alpha needs per-domain tuning.* Set-and-forget loses most of the benefit.
- *Graphs* need entity extraction, relation mapping and upkeep, which adds complexity
  for flat corpora.

**Trade-offs:** each technique adds latency or complexity (HyDE: an extra LLM call;
expansion: multiplied searches; hybrid: two indexes; graph: a knowledge graph to
maintain). Let the eval set decide: going from 90% to 93% may not be worth +500 ms, but
at 60% these techniques are essential.

**Beyond the book:** when combining result lists whose scores aren't comparable (BM25
vs cosine, or several expanded queries), **Reciprocal Rank Fusion (RRF)** is a robust,
tuning-free alternative to weighted score fusion. Both are in the example.

**Pairs with:** Semantic Indexing (better index), Retrieval Refinement (rerank the larger
candidate set).

**Example:** `examples/rag_hybrid.py`

---

## Retrieval Refinement

**What:** A quality-control stage between retrieval and generation, because vector
similarity measures topical closeness, not whether a chunk *answers the question*.
- **Reranking** (most impactful): retrieve broadly (20–50), score each (query, chunk)
  pair with a cross-encoder, and keep the top 3–5. It often yields +10–25 points in
  relevance metrics.
- **Contextual compression:** a small model extracts only the query-relevant sentences
  (e.g. 500 → 80 tokens).
- **Disambiguation:** drop chunks about a different entity with the same name (Mercury the
  planet vs the database), using metadata like source, category or date.
- **Metadata filtering:** hard constraints such as recency, source authority or minimum
  score.
- Plus **deduplication** of near-identical passages.

Typical order: retrieve broad → metadata filter → dedupe → rerank → compress → generate.

**Use when:** retrieval accuracy is under ~80% after tuning embeddings and chunking (then
rerank, at ~100–300 ms), chunks are large (500+ tokens, so compress), or there are
freshness, source-quality or entity-ambiguity issues (filter).

**Avoid when:** retrieval is already accurate, the total latency budget is under ~200 ms,
or the corpus is tiny.

**Pitfalls:** reranker bias (prefers long or formal text, weak on specialist jargon, and
can make things *worse* on a specialized corpus). Compression can drop meaning-changing
context ("unlike the previous approach…"). Over-filtering can remove the only answering
chunk (strict recency drops old-but-valid docs). Stacked steps are hard to debug.

**Trade-offs:** reranking adds latency and a model to host or call. Compression is an
irreversible relevance decision made before the LLM sees the content. You need quality
measured at each stage. **Add steps one at a time and measure each independently.**

**Beyond the book:** an LLM can stand in for a cross-encoder as a reranker (score chunks
0–10 in one call). It's simpler to start with but slower and costlier at scale. Dedicated
rerankers (open-source cross-encoders or hosted rerank APIs) are the production choice.

**Pairs with:** Hybrid Retrieval (better candidates in), Grounded Generation (cleaner
chunks make citations accurate).

**Example:** `examples/rag_refinement.py`

---

## Grounded Generation

**What:** Make answers verifiably grounded, and make the system admit when it can't answer.
- **Out-of-domain detection** (first line of defense): if the best retrieval score is
  below a threshold, or a classifier says the question is out of scope, answer "I don't
  have enough information" instead of letting the model fill gaps from training data.
- **Citations**, at increasing precision: *source-level* (list the documents used) →
  *claim-level* (inline references per claim, each verified to support it) →
  *token-level attribution* (research-grade, expensive).
- **Corrective RAG (CRAG):** a lightweight evaluator grades the retrieved set. High means
  generate. Ambiguous means supplement with web search. Low means fall back or decline.
- **Self-RAG:** generate, then self-critique each claim against the sources (supported?
  complete? anything added?), and regenerate, retrieve more or flag low confidence.
- **Pipeline guardrails** at four stages: input (validate and classify) → retrieval (chunk
  relevance) → generation (grounding instructions) → output (unsupported claims, format).

**Use when:** *always* implement out-of-domain detection and source-level citations for real
users. Add CRAG when the knowledge base has known gaps and a supplementary source exists.
Claim-level citation is a good precision/practicality balance for most applications.
Reserve Self-RAG for high-stakes domains (legal, medical, financial compliance).

**Pitfalls:** an over-aggressive OOD threshold refuses answerable questions, while one set
too lax lets hallucinations through (tune on both error types). The verifier is the
ceiling, since unreliable verification gives false confidence. Self-RAG loops can fail to
converge, so cap iterations and define a fallback. CRAG's web results are unvetted and
need their own filtering.

**Beyond the book:** *negative answer read as no answer.* When the source says "X is not
allowed" and the user asks "can I do X?", a strict refusal instruction can make the model
return the refusal message, as if the sources were silent. Say explicitly that a negative answer IS an answer, and that the
refusal is only for sources that don't address the question at all (see the prompts in
`rag_basic.py` and `rag_grounded.py`).

**Trade-offs:** every check adds latency (Self-RAG can 2–3× generation time).
Helpfulness vs safety is a **product decision**: strict refusal is trustworthy but
unhelpful. Implementation and evaluation effort is high. Invest in proportion to how much
users treat the system as a source of truth.

**Pairs with:** Retrieval Refinement (clean inputs), Deep Search (multi-hop synthesis needs
verification most).

**Example:** `examples/rag_grounded.py` (OOD threshold, numbered citations, citation
validity check, LLM claim-support verification)

---

## Deep Search

**What:** Replace one retrieve-then-generate pass with a loop: **decompose** the question
into sub-questions → **search** → **assess** the evidence (enough? contradictions?
references to follow?) → **identify gaps** → follow-up queries informed by what was found
→ repeat → **synthesize**. Exit conditions: everything answered, budget hit (rounds,
calls, time), or diminishing returns (no new information). Contradictions are reported,
not silently resolved. Sub-questions can target different sources.

**Use when:** questions require synthesis across documents or sources ("compare",
"analyze the trend", "relationship between", "summarize all"), or the corpus is large and
heterogeneous.

**Avoid when:** simple questions dominate. Offer it as an explicit "deep research" mode, or
trigger it only for detected complex queries. Latency is roughly 5–30 s vs ~200 ms.

**Pitfalls:** scope creep and runaway cost without strict budgets. Poor decomposition
misses a dimension that no iteration will recover, so expose the sub-questions for
inspection. Contradiction detection is hard. Staleness compounds across rounds, so carry
dates and prefer newer sources.

**Trade-offs:** cost is roughly rounds × sub-queries (5 × 3 = 15 retrievals plus
reasoning calls). Latency needs UX support (progress, streaming, "researching…"). It's
hard to debug, so log every round. It's closer to building an agent than a search
pipeline.

**Pairs with:** Basic RAG (handles the simple queries), Grounded Generation (verify the
synthesis), Hybrid Retrieval (iterating over bad retrievals doesn't help, so improve each
round).

**Example:** `examples/rag_deep_search.py`

---

## Agentic RAG

**What:** An agent controls retrieval. It decides *whether* to retrieve, *which* tools to use
(vector search, web, SQL, APIs, code search), *what* to query, evaluates the results,
reformulates, and decides when it has enough. **Routing** is the simpler form: classify
the query once and send it down one fixed path.

**Use when:** query types need fundamentally different retrieval strategies, data lives in
systems that can't be unified into one index, or queries are unpredictable. If you can
enumerate the query types, **routing is simpler** and usually enough.

**Don't start here:** go Basic → Hybrid → Refinement first, then consider agent control.

**Pitfalls:** poor tool choice (clear, specific tool descriptions are essential).
**Unbounded exploration**: put hard limits on tool calls, tokens and wall-clock time.
Combinatorial error paths, so every tool failure needs a recovery. Blind spots, where the
agent over-uses one tool (test across diverse query types). Observability: trace every
decision and tool call.

**Trade-offs:** flexibility costs predictability (500 ms with one call vs 20 s with eight).
The LLM is in the hot path for every decision, so cost scales with query complexity. Tool
maintenance is ongoing, and the agent is only as good as its tools. Testing must cover tool
choice, decomposition, failures, budgets and the final answers.

In production, implement the loop with your provider's **native tool calling** rather
than parsing JSON actions from text, as the book's own agent code does (Agents chapter).
The example uses a provider-neutral JSON protocol to keep the control flow visible.

**Pairs with:** Semantic Router (a faster router by embeddings; Routing & Orchestration
chapter), Tool Calling (the mechanism), Deep Search (a mode within it for multi-hop
questions), Basic RAG (one of its tools).

**Example:** `examples/rag_agentic.py` (a router plus a bounded agent loop with a trace)

---

## Evaluating a RAG system

Measure retrieval and generation **separately**:
- **Retrieval:** build 20–50+ queries with known relevant chunk or document IDs. Measure
  recall@k (did the right chunk make the top-k?) and MRR or rank position. Compare
  keyword, vector and hybrid on the same set.
- **Out-of-domain threshold:** include unanswerable queries. For each candidate
  threshold, measure correct refusals vs wrong refusals and pick the balance (sweep e.g.
  0.5 → 0.9).
- **Beyond the book:** include questions whose correct answer is **"no"** (e.g. "can I
  get a refund on a monthly plan?" against "monthly plans are not refundable"). Count a
  refusal on these as an error. Otherwise over-refusal looks like good grounding.
- **Generation:** faithfulness (every claim supported by the cited chunk), answer
  correctness, and citation validity. LLM-as-Judge is usual for these (Evaluation
  chapter).
- Re-run the suite on every change to chunking, the embedding model, alpha, the reranker
  or prompts.

`examples/rag_common.py` includes `recall_at_k()` for the retrieval side.

## Reviewing an existing RAG pipeline

1. Is there a retrieval eval set, and is retrieval logged separately from generation?
2. Chunking: is the size suited to the document type? Is there overlap? Do chunks keep
   their source and metadata, and are they self-contained?
3. Are embeddings computed once and persisted, with the model name stored?
4. Is it hybrid (BM25 + vectors)? Are exact identifiers searchable?
5. Is there a relevance threshold and an "I don't know" path?
6. Are sources cited, and are the citations validated?
7. Is the index kept in sync with the source of truth (updates and deletions)?
8. Are top-k and context size bounded, with the most relevant chunks placed first?
9. For agentic or deep variants: are there hard limits on steps, tokens and time, and a
   full trace?
