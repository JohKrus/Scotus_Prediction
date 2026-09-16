"""Shared RAG utilities — FAISS, BM25, cross-encoder reranking."""

from __future__ import annotations

import logging
import time

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

from scotus_v3 import config, models

log = logging.getLogger(__name__)

_cross_encoder: CrossEncoder | None = None


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        cfg = config.load()["retrieval"]
        _cross_encoder = CrossEncoder(cfg["cross_encoder"], device="cpu")
    return _cross_encoder


def build_faiss_index(
    docs: list[Document],
    batch_size: int = 30,
) -> FAISS | None:
    """Build a FAISS vector store with batched embedding and retry logic."""
    if not docs:
        return None

    embedding_model = models.get_embedding_model()
    vector_store: FAISS | None = None

    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        for attempt in range(1, 4):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embedding_model)
                else:
                    vector_store.add_documents(batch)
                break
            except Exception as e:
                err = str(e).lower()
                if "rate" in err or "429" in err:
                    wait = min(30 * attempt, 90)
                    log.warning("Rate limit hit, waiting %ds …", wait)
                    time.sleep(wait)
                elif attempt == 3:
                    log.error("FAISS batch failed after 3 retries: %s", e)
                    break
                else:
                    time.sleep(5)
        time.sleep(0.5)

    return vector_store


def hybrid_retrieve(
    docs: list[Document],
    query: str,
    top_k: int = 10,
) -> list[Document]:
    """Retrieve relevant documents via BM25 + FAISS, deduplicated."""
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = top_k

    bm25_docs = bm25.invoke(query)

    faiss_store = build_faiss_index(docs)
    if faiss_store:
        faiss_docs = faiss_store.as_retriever(search_kwargs={"k": top_k}).invoke(query)
    else:
        faiss_docs = []

    # Deduplicate
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in bm25_docs + faiss_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
            if len(unique) >= top_k:
                break

    return unique


def rerank(
    docs: list[Document],
    query: str,
    top_n: int | None = None,
) -> list[Document]:
    """Rerank documents with the cross-encoder model."""
    if not docs:
        return []

    top_n = top_n or config.load()["retrieval"]["top_n_reranked"]
    encoder = _get_cross_encoder()

    pairs = [[query, doc.page_content] for doc in docs]
    scores = encoder.predict(pairs)

    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


def retrieve_and_rerank(
    typed_retrievers: dict[str, object],
    queries: list[str],
    top_k_per_type: int | None = None,
    top_n_reranked: int | None = None,
    rerank_query: str = "",
) -> list[Document]:
    """Full retrieval pipeline: multi-query + multi-type + dedup + rerank."""
    cfg = config.load()["retrieval"]
    top_k_per_type = top_k_per_type or cfg["top_k_per_type"]
    top_n_reranked = top_n_reranked or cfg["top_n_reranked"]

    all_docs: list[Document] = []
    for query in queries:
        for doc_type, retriever in typed_retrievers.items():
            if not retriever:
                continue
            try:
                results = retriever.invoke(query)
                for doc in results:
                    doc.metadata["retrieved_doc_type"] = doc_type
                all_docs.extend(results)
            except Exception as e:
                log.warning("Retrieval error for %s: %s", doc_type, e)

    # Deduplicate
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in all_docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)

    if not unique:
        return []

    return rerank(unique, rerank_query or queries[0], top_n_reranked)
