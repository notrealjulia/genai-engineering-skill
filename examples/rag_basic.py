"""Basic RAG: index -> retrieve top-k -> answer ONLY from context, with sources.

Adapted from GenAI Patterns (S. Pandey, CC BY-NC 4.0). Uses BM25 instead of the book's
keyword-overlap retrieval; any retriever with .search(query, k) works.
"""
from llm_client import call_llm
from rag_common import BM25Index, chunk_text, format_context

NO_ANSWER = "I don't have enough information in the provided documents to answer that."

# Models tend to read "X is not allowed" as "no information"; say that a "no" IS an answer.
PROMPT = """Answer the question using ONLY the documents below.
If the documents answer the question negatively (e.g. "X is not allowed"), that IS an
answer: give it. Only if the documents don't address the question at all, reply exactly:
"{no_answer}"

{context}

Question: {question}"""


def basic_rag(question: str, retriever, top_k: int = 3) -> dict:
    results = retriever.search(question, k=top_k)
    chunks = [c for _, c in results]
    if not chunks:  # nothing retrieved: don't let the model answer from training data
        return {"answer": NO_ANSWER, "sources": [], "retrieved": []}
    answer = call_llm(PROMPT.format(no_answer=NO_ANSWER, context=format_context(chunks),
                                    question=question)).strip()
    return {
        "answer": answer,
        "sources": sorted({c.source for c in chunks}),                   # source-level citation
        "retrieved": [(round(s, 3), c.id) for s, c in results],          # log retrieval separately
    }


if __name__ == "__main__":
    docs = {
        "retention-policy.md": "Customer data is retained for 24 months after account closure, "
                               "then permanently deleted.",
        "refunds.md": "Refunds are available within 30 days of purchase for annual plans. "
                      "Monthly plans are not refundable.",
    }
    chunks = [c for src, text in docs.items() for c in chunk_text(text, src)]
    index = BM25Index(chunks)  # indexing pipeline: run once, offline
    print(basic_rag("How long is customer data retained?", index))
    print(basic_rag("Can I get a refund on a monthly plan?", index))  # correct answer is "no"
