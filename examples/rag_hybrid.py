"""Hybrid retrieval: BM25 + vector fusion (weighted alpha or RRF), query expansion, HyDE.

Based on the techniques described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book
has no code for this pattern.
"""
from agents_common import parse_json
from llm_client import call_llm
from rag_common import BM25Index, Chunk, VectorIndex


def _minmax(results: list[tuple[float, Chunk]]) -> dict[str, float]:
    """BM25 and cosine scores are on different scales: normalize to 0..1 before mixing."""
    if not results:
        return {}
    scores = [s for s, _ in results]
    lo, hi = min(scores), max(scores)
    return {c.id: (s - lo) / (hi - lo) if hi > lo else 1.0 for s, c in results}


def hybrid_search(query: str, bm25: BM25Index, vectors: VectorIndex, k: int = 5,
                  alpha: float = 0.6, candidates: int = 20) -> list[tuple[float, Chunk]]:
    """Weighted fusion. alpha=0 -> pure keyword, 1 -> pure vector. Tune alpha per domain."""
    kw, vec = bm25.search(query, candidates), vectors.search(query, candidates)
    kw_n, vec_n = _minmax(kw), _minmax(vec)
    by_id = {c.id: c for _, c in kw + vec}
    fused = [(alpha * vec_n.get(i, 0.0) + (1 - alpha) * kw_n.get(i, 0.0), c) for i, c in by_id.items()]
    return sorted(fused, key=lambda x: x[0], reverse=True)[:k]


def rrf(result_lists: list[list[tuple[float, Chunk]]], k: int = 5, c: int = 60):
    """Reciprocal Rank Fusion: combines ranked lists using ranks only (no score tuning).
    Robust default for merging BM25 + vector, or results of several expanded queries."""
    scores, by_id = {}, {}
    for results in result_lists:
        for rank, (_, chunk) in enumerate(results, 1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (c + rank)
            by_id[chunk.id] = chunk
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(s, by_id[i]) for i, s in ranked]


def expand_query(query: str, n: int = 3) -> list[str]:
    """LLM rewrites a short/casual query into n alternative phrasings (original kept first)."""
    raw = call_llm(
        f"Rewrite this search query into {n} alternative phrasings that use the vocabulary "
        "documentation would likely use. Stay faithful to the original intent.\n"
        f"Return ONLY a JSON array of strings.\n\nQuery: {query}", temperature=0.3)
    parsed = parse_json(raw)  # models often wrap JSON in ```json fences
    # degrade gracefully on bad output: the original query still runs
    variants = [v for v in parsed if isinstance(v, str)][:n] if isinstance(parsed, list) else []
    return [query] + variants


def multi_query_search(query: str, search_fn, k: int = 5) -> list[tuple[float, Chunk]]:
    return rrf([search_fn(q, k * 2) for q in expand_query(query)], k=k)


def hyde_search(query: str, vectors: VectorIndex, k: int = 5) -> list[tuple[float, Chunk]]:
    """HyDE: embed a hypothetical answer (closer to doc vocabulary) instead of the question.
    Risk: a confidently wrong hypothetical retrieves the wrong concept; combine with the
    plain query via RRF as a safety net."""
    hypothetical = call_llm(
        f"Write a short, plausible passage from technical documentation that answers:\n{query}")
    return rrf([vectors.search(hypothetical, k * 2), vectors.search(query, k * 2)], k=k)


if __name__ == "__main__":
    from rag_common import chunk_text
    docs = {"ci.md": "Resolving intermittent CI/CD pipeline failures: retry flaky steps, pin "
                     "dependency versions, and check runner resource limits.",
            "errors.md": "ERR-504 means the upstream gateway timed out."}
    chunks = [c for s, t in docs.items() for c in chunk_text(t, s)]
    bm25, vectors = BM25Index(chunks), VectorIndex(chunks)
    print(hybrid_search("what does ERR-504 mean", bm25, vectors, k=2))   # keyword side catches the code
    print(multi_query_search("fix a flaky deploy",
                             lambda q, k: hybrid_search(q, bm25, vectors, k), k=2))
