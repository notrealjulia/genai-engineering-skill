"""Retrieval refinement: metadata filter -> dedupe -> rerank -> compress.

Based on the techniques described in GenAI Patterns (S. Pandey, CC BY-NC 4.0); the book
has no code for this pattern. Add steps one at a time and measure each.
"""
from datetime import date

from agents_common import parse_json
from llm_client import call_llm
from rag_common import Chunk, tokenize

Results = list[tuple[float, Chunk]]


def filter_metadata(results: Results, min_date: str | None = None,
                    allowed_sources: set[str] | None = None) -> Results:
    """Hard constraints. Beware over-filtering: chunks without a date are KEPT, not dropped."""
    out = []
    for s, c in results:
        updated = c.metadata.get("updated")
        if min_date and updated and date.fromisoformat(updated) < date.fromisoformat(min_date):
            continue
        if allowed_sources and c.source not in allowed_sources:
            continue
        out.append((s, c))
    return out


def dedupe(results: Results, threshold: float = 0.85) -> Results:
    """Drop near-duplicates (Jaccard similarity of token sets), keeping the higher-ranked one."""
    kept, kept_sets = [], []
    for s, c in results:
        toks = set(tokenize(c.text))
        if all(len(toks & k) / max(len(toks | k), 1) < threshold for k in kept_sets):
            kept.append((s, c))
            kept_sets.append(toks)
    return kept


def rerank_llm(query: str, results: Results, top_k: int = 3) -> Results:
    """LLM as reranker: one call scores every (query, chunk) pair 0-10 for whether it
    ANSWERS the question (not just topical overlap). Fine to start with; for scale use a
    cross-encoder (see rerank_cross_encoder)."""
    if not results:
        return []
    listing = "\n\n".join(f"<chunk id={i}>\n{c.text}\n</chunk>" for i, (_, c) in enumerate(results))
    raw = call_llm(
        "Score how well each chunk helps answer the question, 0 (irrelevant) to 10 "
        "(directly answers it). Return ONLY a JSON object mapping chunk id to score, "
        f'e.g. {{"0": 7, "1": 2}}.\n\nQuestion: {query}\n\n{listing}')
    try:
        scores = {int(k): float(v) for k, v in parse_json(raw).items()}
    except (ValueError, TypeError, AttributeError):
        return results[:top_k]  # reranker failed: fall back to the original order
    ranked = sorted(range(len(results)), key=lambda i: scores.get(i, -1), reverse=True)
    return [(scores.get(i, 0.0), results[i][1]) for i in ranked[:top_k]]


def rerank_cross_encoder(query: str, results: Results, top_k: int = 3,
                         model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> Results:
    """Dedicated cross-encoder (pip install sentence-transformers). Check it on YOUR corpus:
    general rerankers can make specialized domains worse."""
    from sentence_transformers import CrossEncoder
    scores = CrossEncoder(model).predict([(query, c.text) for _, c in results])
    ranked = sorted(zip(scores, (c for _, c in results)), key=lambda x: x[0], reverse=True)
    return [(float(s), c) for s, c in ranked[:top_k]]


def compress(query: str, results: Results) -> Results:
    """Keep only query-relevant sentences per chunk; drop chunks with none.
    Irreversible: the generator never sees what's removed. Ask the model to keep qualifiers."""
    out = []
    for s, c in results:
        extract = call_llm(
            "Copy verbatim only the sentences from the passage that are relevant to the "
            "question, keeping any qualifying context (e.g. 'unlike X', 'except when'). "
            f"If none are relevant, reply NONE.\n\nQuestion: {query}\n\nPassage:\n{c.text}").strip()
        if extract and extract.upper() != "NONE":
            out.append((s, Chunk(c.id, extract, c.source, {**c.metadata, "compressed": True})))
    return out


def refine(query: str, results: Results, top_k: int = 3, min_date: str | None = None,
           use_compression: bool = False) -> Results:
    """Typical order: broad retrieve (20-50) -> filter -> dedupe -> rerank -> compress."""
    refined = rerank_llm(query, dedupe(filter_metadata(results, min_date=min_date)), top_k)
    return compress(query, refined) if use_compression else refined


if __name__ == "__main__":
    docs = [Chunk("refunds.md#0", "Refunds are available within 30 days of purchase for annual "
                  "plans. Monthly plans are not refundable.", "refunds.md", {"updated": "2026-02-01"}),
            Chunk("refunds-copy.md#0", "Refunds are available within 30 days of purchase for annual "
                  "plans. Monthly plans are not refundable.", "refunds-copy.md"),  # near-duplicate
            Chunk("refunds-2022.md#0", "Refunds are available within 14 days for all plans.",
                  "refunds-2022.md", {"updated": "2022-05-01"}),                    # stale
            Chunk("sla.md#0", "The Enterprise plan includes a 99.9% uptime SLA.", "sla.md")]
    results = [(1.0, c) for c in docs]  # pretend a broad first-stage retrieval returned these
    for s, c in refine("Can I get a refund on a monthly plan?", results, top_k=2,
                       min_date="2024-01-01", use_compression=True):
        print(s, c.id, "|", c.text)
