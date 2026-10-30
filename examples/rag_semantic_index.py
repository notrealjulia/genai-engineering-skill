"""Semantic indexing: embed once, persist vectors + model name, search by meaning.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0). Fixes the common mistake of
re-embedding the whole corpus on every query, and adds persistence + a relevance threshold.
"""
import json
import os
import tempfile

import numpy as np

import llm_client
from rag_common import Chunk, VectorIndex, chunk_text


def save_index(index: VectorIndex, path: str) -> None:
    """Persist raw text + metadata + vectors + model name so you can reload or re-embed later."""
    np.save(f"{path}.npy", index.matrix)
    with open(f"{path}.json", "w") as f:
        json.dump({"model": index.model_name,
                   "chunks": [c.__dict__ for c in index.chunks]}, f)


def load_index(path: str, embed_fn=None) -> VectorIndex:
    with open(f"{path}.json") as f:
        data = json.load(f)
    current_model = llm_client.EMBED_MODEL
    if embed_fn is None and data["model"] != current_model:
        # Vectors from different models live in different spaces: mixing them is silently wrong.
        raise ValueError(f"Index built with {data['model']!r}, but EMBED_MODEL is "
                         f"{current_model!r}. Re-embed the corpus (raw text is stored).")
    index = VectorIndex.__new__(VectorIndex)  # skip re-embedding
    index.embed_fn = embed_fn or llm_client.embed
    index.model_name = data["model"]
    index.chunks = [Chunk(**c) for c in data["chunks"]]
    index.matrix = np.load(f"{path}.npy")
    return index


def semantic_search(index: VectorIndex, query: str, k: int = 3, min_score: float = 0.3):
    """Drop results below a similarity threshold. Calibrate min_score on an eval set:
    good values differ per embedding model."""
    return [(s, c) for s, c in index.search(query, k) if s >= min_score]


if __name__ == "__main__":
    texts = {
        "account.md": "Account Termination Policy: you can close your account at any time "
                      "from Settings > Billing.",
        "deploys.md": "Resolving intermittent CI/CD pipeline failures: add retries and pin versions.",
    }
    chunks = [c for src, t in texts.items() for c in chunk_text(t, src)]
    path = os.path.join(tempfile.gettempdir(), "kb_index")
    index = VectorIndex(chunks)  # one-time embedding cost
    save_index(index, path)
    index = load_index(path)     # later: reload without re-embedding
    for score, chunk in semantic_search(index, "how do I cancel my subscription?"):
        print(f"{score:.3f}  {chunk.source}")  # matches account.md despite no shared words
