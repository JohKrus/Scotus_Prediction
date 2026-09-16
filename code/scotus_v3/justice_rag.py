"""Per-justice FAISS vector stores built from authored opinions.

Each justice gets their own RAG index containing all opinions they have
written (majority, concurrences, dissents). Temporal filtering at query
time prevents data leakage by only returning opinions from terms strictly
before the prediction term.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder

from scotus_v3 import config, models
from scotus_v3.opinion_parser import OpinionSegment

log = logging.getLogger(__name__)


class JusticeRAGStore:
    """Manages a single justice's opinion FAISS index."""

    def __init__(self, justice_name: str, base_dir: Path | None = None):
        self.justice_name = justice_name
        self.base_dir = base_dir or config.justice_indices_dir()
        self.store_dir = self.base_dir / justice_name.replace(" ", "_")
        self.faiss_index: FAISS | None = None
        self.documents: list[Document] = []  # all chunks with metadata
        self.metadata: dict = {
            "justice": justice_name,
            "terms_indexed": [],
            "total_segments": 0,
            "total_chunks": 0,
            "opinion_types": {},
        }

    def build_index(
        self,
        segments: list[OpinionSegment],
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """Build FAISS index from opinion segments."""
        cfg = config.load()
        rag_cfg = cfg.get("justice_rag", {})
        chunk_size = chunk_size or rag_cfg.get("chunk_size", 1500)
        chunk_overlap = chunk_overlap or rag_cfg.get("chunk_overlap", 300)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
        )

        docs: list[Document] = []
        terms_seen: set[int] = set()
        type_counts: dict[str, int] = {}

        for seg in segments:
            if not seg.text or len(seg.text) < 200:
                continue

            chunks = splitter.split_text(seg.text)
            terms_seen.add(seg.term)
            type_counts[seg.opinion_type] = type_counts.get(seg.opinion_type, 0) + 1

            for idx, chunk_text in enumerate(chunks):
                if len(chunk_text.strip()) < 100:
                    continue
                docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "justice": self.justice_name,
                        "docket": seg.docket,
                        "term": seg.term,
                        "opinion_type": seg.opinion_type,
                        "case_name": seg.case_name,
                        "joining_justices": seg.joining_justices,
                        "source": "opinion",
                        "chunk_index": idx,
                    },
                ))

        if not docs:
            log.warning("No chunks produced for %s", self.justice_name)
            return

        self.documents = docs
        self.metadata.update({
            "terms_indexed": sorted(terms_seen),
            "total_segments": len(segments),
            "total_chunks": len(docs),
            "opinion_types": type_counts,
        })

        # Build FAISS index with batched embedding
        log.info("  Building FAISS index for %s: %d chunks from %d segments",
                 self.justice_name, len(docs), len(segments))

        embedding_model = models.get_embedding_model()
        batch_size = 30

        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            for attempt in range(1, 4):
                try:
                    if self.faiss_index is None:
                        self.faiss_index = FAISS.from_documents(batch, embedding_model)
                    else:
                        self.faiss_index.add_documents(batch)
                    break
                except Exception as e:
                    err = str(e).lower()
                    if "rate" in err or "429" in err:
                        wait = min(30 * attempt, 90)
                        log.warning("Rate limit, waiting %ds ...", wait)
                        time.sleep(wait)
                    elif attempt == 3:
                        log.error("FAISS batch failed for %s: %s", self.justice_name, e)
                        break
                    else:
                        time.sleep(5)
            time.sleep(0.5)

        log.info("  %s: FAISS index built (%d chunks)", self.justice_name, len(docs))

    def save(self) -> None:
        """Persist FAISS index and metadata to disk."""
        if self.faiss_index is None:
            log.warning("No index to save for %s", self.justice_name)
            return

        self.store_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        self.faiss_index.save_local(str(self.store_dir))

        # Save metadata
        meta_path = self.store_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        # Save document metadata (for temporal filtering at query time)
        docs_meta_path = self.store_dir / "documents_metadata.json"
        docs_meta = [
            {
                "term": doc.metadata["term"],
                "docket": doc.metadata["docket"],
                "opinion_type": doc.metadata["opinion_type"],
                "case_name": doc.metadata["case_name"],
            }
            for doc in self.documents
        ]
        with open(docs_meta_path, "w", encoding="utf-8") as f:
            json.dump(docs_meta, f, indent=2)

        log.info("  %s: index saved to %s", self.justice_name, self.store_dir)

    def load(self) -> bool:
        """Load persisted index. Returns True if successful."""
        faiss_path = self.store_dir / "index.faiss"
        if not faiss_path.exists():
            return False

        try:
            embedding_model = models.get_embedding_model()
            self.faiss_index = FAISS.load_local(
                str(self.store_dir),
                embedding_model,
                allow_dangerous_deserialization=True,
            )

            # Load metadata
            meta_path = self.store_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)

            log.info("  %s: loaded index (%d chunks)", self.justice_name,
                     self.metadata.get("total_chunks", "?"))
            return True
        except Exception as e:
            log.error("Failed to load index for %s: %s", self.justice_name, e)
            return False

    def query(
        self,
        question: str,
        max_term: int,
        top_k: int = 8,
        opinion_types: list[str] | None = None,
    ) -> list[Document]:
        """Query this justice's opinion index with temporal filtering.

        Only returns chunks from terms strictly before max_term.
        Retrieves top_k * 3 candidates, then post-filters and reranks.
        """
        if self.faiss_index is None:
            return []

        # Over-retrieve to allow for filtering
        retrieve_k = min(top_k * 3, self.metadata.get("total_chunks", top_k * 3))
        try:
            candidates = self.faiss_index.similarity_search(question, k=retrieve_k)
        except Exception as e:
            log.error("Query failed for %s: %s", self.justice_name, e)
            return []

        # Temporal filter: only opinions from terms before predict_term
        filtered = []
        for doc in candidates:
            doc_term = doc.metadata.get("term", 0)
            if doc_term >= max_term:
                continue
            if opinion_types and doc.metadata.get("opinion_type") not in opinion_types:
                continue
            filtered.append(doc)

        if not filtered:
            return []

        # Rerank with cross-encoder
        try:
            cfg = config.load()["retrieval"]
            encoder = CrossEncoder(cfg["cross_encoder"], device="cpu")
            pairs = [[question, doc.page_content] for doc in filtered]
            scores = encoder.predict(pairs)
            scored = sorted(zip(scores, filtered), key=lambda x: x[0], reverse=True)
            return [doc for _, doc in scored[:top_k]]
        except Exception as e:
            log.warning("Reranking failed for %s, returning unranked: %s", self.justice_name, e)
            return filtered[:top_k]

    def get_summary(self, max_term: int | None = None) -> str:
        """Summary of indexed opinions for this justice."""
        types = self.metadata.get("opinion_types", {})
        terms = self.metadata.get("terms_indexed", [])

        if max_term:
            terms = [t for t in terms if t < max_term]

        if not terms:
            return f"{self.justice_name}: no opinions indexed"

        type_parts = []
        for t, c in sorted(types.items()):
            type_parts.append(f"{c} {t}")

        return (
            f"{self.justice_name}: {', '.join(type_parts)} "
            f"across terms {min(terms)}-{max(terms)}"
        )


class JusticeRAGManager:
    """Manages all per-justice RAG stores."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or config.justice_indices_dir()
        self.stores: dict[str, JusticeRAGStore] = {}

    def build_all(
        self,
        terms: list[int] | None = None,
    ) -> None:
        """Parse all opinion PDFs and build per-justice indices."""
        from scotus_v3.opinion_parser import parse_all_opinions

        justice_segments = parse_all_opinions(terms)

        for justice_name, segments in justice_segments.items():
            # Skip Per Curiam — not individual justice reasoning
            if justice_name == "Per Curiam":
                log.info("Skipping Per Curiam opinions (not individual reasoning)")
                continue

            store = JusticeRAGStore(justice_name, self.base_dir)
            store.build_index(segments)
            store.save()
            self.stores[justice_name] = store

        log.info("Built indices for %d justices", len(self.stores))

    def load_all(self) -> None:
        """Load all persisted justice indices from disk."""
        if not self.base_dir.exists():
            log.warning("Justice indices directory does not exist: %s", self.base_dir)
            return

        for store_dir in sorted(self.base_dir.iterdir()):
            if not store_dir.is_dir():
                continue
            justice_name = store_dir.name.replace("_", " ")
            store = JusticeRAGStore(justice_name, self.base_dir)
            if store.load():
                self.stores[justice_name] = store

        log.info("Loaded %d justice RAG indices", len(self.stores))

    def query_justice(
        self,
        justice_name: str,
        query: str,
        predict_term: int,
        top_k: int = 8,
    ) -> list[Document]:
        """Query a specific justice's opinions, respecting temporal cutoff."""
        if justice_name not in self.stores:
            log.debug("No RAG store for %s", justice_name)
            return []

        return self.stores[justice_name].query(
            question=query,
            max_term=predict_term,
            top_k=top_k,
        )

    def multi_query_justice(
        self,
        justice_name: str,
        queries: list[str],
        predict_term: int,
        top_k: int = 8,
    ) -> list[Document]:
        """Query with multiple queries, deduplicate, and return top results.

        This is critical for avoiding "no results" — a single query may miss
        relevant opinions, but multiple complementary queries cast a wider net.
        """
        if justice_name not in self.stores:
            return []

        all_docs: list[Document] = []
        seen_content: set[str] = set()

        for query in queries:
            docs = self.stores[justice_name].query(
                question=query,
                max_term=predict_term,
                top_k=top_k,
            )
            for doc in docs:
                # Deduplicate by content
                content_key = doc.page_content[:200]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    all_docs.append(doc)

        # Rerank the combined results
        if len(all_docs) > top_k:
            try:
                cfg = config.load()["retrieval"]
                encoder = CrossEncoder(cfg["cross_encoder"], device="cpu")
                # Use the first query as reranking reference
                pairs = [[queries[0], doc.page_content] for doc in all_docs]
                scores = encoder.predict(pairs)
                scored = sorted(zip(scores, all_docs), key=lambda x: x[0], reverse=True)
                return [doc for _, doc in scored[:top_k]]
            except Exception:
                return all_docs[:top_k]

        return all_docs

    def format_justice_context(
        self,
        justice_name: str,
        query: str | list[str],
        predict_term: int,
        top_k: int = 8,
    ) -> str:
        """Query and format opinion excerpts for use in prompts.

        Accepts a single query string or a list of queries for multi-query retrieval.
        """
        if isinstance(query, list):
            docs = self.multi_query_justice(justice_name, query, predict_term, top_k)
        else:
            docs = self.query_justice(justice_name, query, predict_term, top_k)

        if not docs:
            return "No relevant prior opinions found for this legal question."

        parts = []
        for doc in docs:
            meta = doc.metadata
            op_type = meta.get("opinion_type", "opinion").upper()
            case = meta.get("case_name", "Unknown case")
            term = meta.get("term", "?")
            header = f"[{op_type} in {case}, Term {term}]"
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def get_justice_summary(self, justice_name: str, predict_term: int) -> str:
        """Get a summary of a justice's indexed opinions."""
        if justice_name not in self.stores:
            return f"{justice_name}: no opinions indexed"
        return self.stores[justice_name].get_summary(max_term=predict_term)

    def available_justices(self) -> list[str]:
        """List all justices with loaded RAG stores."""
        return sorted(self.stores.keys())


# --- CLI entry point ---

def build_indices(terms: list[int] | None = None) -> None:
    """Build all per-justice FAISS indices from scraped opinions."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    manager = JusticeRAGManager()
    manager.build_all(terms)
    print(f"\nBuilt indices for {len(manager.stores)} justices:")
    for name, store in sorted(manager.stores.items()):
        print(f"  {store.get_summary()}")


if __name__ == "__main__":
    build_indices()
