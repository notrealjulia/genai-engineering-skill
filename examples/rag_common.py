"""Shared RAG building blocks: chunks, chunking, BM25 and vector indexes, context formatting,
retrieval evaluation. Dependency-light (numpy only) so the patterns stay visible.
In production, swap the indexes for a search engine / vector DB with the same interface.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

import llm_client


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)  # e.g. {"updated": "2026-03-01", "type": "policy"}


# --- Chunking -------------------------------------------------------------------------

def chunk_text(text: str, source: str, max_words: int = 200, overlap: int = 40,
               metadata: dict | None = None) -> list[Chunk]:
    """Paragraph-aware chunking: pack whole paragraphs up to max_words; split long
    paragraphs with a sliding window. Overlap carries context across boundaries.
    Word counts approximate tokens (~0.75 words/token); tune per document type.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, current = [], []
    for para in paragraphs:
        words = para.split()
        if len(words) > max_words:  # long paragraph: flush, then window it
            if current:
                pieces.append(current)
                current = []
            step = max_words - overlap
            for i in range(0, len(words), step):
                pieces.append(words[i:i + max_words])
                if i + max_words >= len(words):
                    break
        elif len(current) + len(words) > max_words:
            pieces.append(current)
            current = current[-overlap:] + words if overlap else words
        else:
            current += words
    if current:
        pieces.append(current)
    return [Chunk(id=f"{source}#{i}", text=" ".join(p), source=source,
                  metadata=dict(metadata or {})) for i, p in enumerate(pieces)]


# --- Keyword index (BM25) -------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    # Keeps identifiers like "ERR-504" or "v2.1" intact; that's what keyword search is for.
    return re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower())


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[float, Chunk]]:
        terms = tokenize(query)
        scored = []
        for doc, length, chunk in zip(self.docs, self.lengths, self.chunks):
            s = 0.0
            for t in terms:
                if t in doc:
                    tf = doc[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (
                        tf + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            if s > 0:
                scored.append((s, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# --- Vector index ---------------------------------------------------------------------

class VectorIndex:
    """Embeds chunks ONCE at build time; queries embed only the query.
    Stores the embedding model name: vectors from different models are incompatible.
    """

    def __init__(self, chunks: list[Chunk], embed_fn=None, model_name: str | None = None):
        self.embed_fn = embed_fn or llm_client.embed
        self.model_name = model_name or llm_client.EMBED_MODEL
        self.chunks = chunks
        self.matrix = self._normalize(np.array(self.embed_fn([c.text for c in chunks]), dtype=float))

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=-1, keepdims=True)
        return m / np.where(norms == 0, 1, norms)

    def search_vector(self, vector, k: int = 5) -> list[tuple[float, Chunk]]:
        q = self._normalize(np.array(vector, dtype=float))
        scores = self.matrix @ q  # cosine similarity (rows are unit-normalized)
        top = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in top]

    def search(self, query: str, k: int = 5) -> list[tuple[float, Chunk]]:
        return self.search_vector(self.embed_fn([query])[0], k)


# --- Prompt context & evaluation ------------------------------------------------------

def format_context(chunks: list[Chunk]) -> str:
    """Numbered blocks so the model can cite [1], [2], ... and we can map citations back."""
    return "\n\n".join(f"[{i}] (source: {c.source})\n{c.text}" for i, c in enumerate(chunks, 1))


def recall_at_k(search_fn, eval_set: list[tuple[str, set[str]]], k: int = 5) -> float:
    """eval_set: [(query, {relevant chunk ids})]. Fraction of queries with >=1 hit in top-k."""
    hits = 0
    for query, relevant in eval_set:
        ids = {c.id for _, c in search_fn(query, k)}
        hits += bool(ids & relevant)
    return hits / len(eval_set)
