import os
import pickle
import re
import logging
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

from src.ingestion.schemas import Chunk

logger = logging.getLogger(__name__)

ENGLISH_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "is",
    "are",
    "be",
    "by",
    "with",
    "as",
}


class SparseBM25Index:
    """Manages sparse BM25 keyword index for exact lexical searching."""

    def __init__(self, storage_path: str = "data/sparse_index.pkl"):
        self.storage_path = storage_path
        self.bm25: BM25Okapi = None
        self.indexed_chunks: List[Chunk] = []

    def _tokenize(self, text: str) -> List[str]:
        """Convert text into lowercase tokens, stripping whitespace and common stopwords."""
        if not text:
            return []
        text = re.sub(r"\s+", " ", text.lower()).strip()
        tokens = re.findall(r"\b[a-z0-9]+(?:-[a-z0-9]+)*\b", text)
        return [t for t in tokens if len(t) > 1 and t not in ENGLISH_STOPWORDS]

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Build and update the BM25 lexical index with new document chunks."""
        if not chunks:
            return

        existing_ids = {chunk.id for chunk in self.indexed_chunks}
        new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]

        if not new_chunks:
            logger.info("All provided chunks are already present in sparse index.")
            return

        self.indexed_chunks.extend(new_chunks)
        logger.info(
            f"Indexing {len(new_chunks)} new chunks (Total: {len(self.indexed_chunks)}) into BM25 index."
        )

        corpus_tokenized = [self._tokenize(chunk.page_content) for chunk in self.indexed_chunks]
        self.bm25 = BM25Okapi(corpus_tokenized)
        self.save_index()

    def search(
        self, query: str, top_k: int = 5, source_path: str = None
    ) -> List[Dict[str, Any]]:
        """Search BM25 index for keyword matches and return scored chunk dictionary list.
        """
        if not self.bm25 or not self.indexed_chunks:
            logger.warning("Sparse index not initialized. Ingest documents first.")
            return []

        tokenized_query = self._tokenize(query)
        raw_scores = self.bm25.get_scores(tokenized_query)

        results = [
            {"chunk": self.indexed_chunks[idx], "sparse_score": float(score)}
            for idx, score in enumerate(raw_scores)
            if score > 0.0
        ]

        if source_path:
            results = [
                r for r in results if r["chunk"].metadata.source_path == source_path
            ]

        results.sort(key=lambda x: x["sparse_score"], reverse=True)
        return results[:top_k]

    def retrieve(self, query: str, top_k: int = 10) -> List[Chunk]:
        """Retrieve raw Chunk objects matching query for compatibility with unified retriever interface."""
        search_results = self.search(query, top_k=top_k)
        return [item["chunk"] for item in search_results]

    def save_index(self) -> None:
        """Serialize current BM25 model and chunks to disk via pickle."""
        try:
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            with open(self.storage_path, "wb") as f:
                pickle.dump({"chunks": self.indexed_chunks, "model": self.bm25}, f)
            logger.info(f"Saved sparse index to {self.storage_path}")
        except Exception as e:
            logger.error(f"Failed to save sparse index: {e}")

    def load_index(self) -> None:
        """Deserialize stored BM25 model and chunks from disk if present."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as f:
                    data = pickle.load(f)
                    self.indexed_chunks = data["chunks"]
                    self.bm25 = data["model"]
                    logger.info(f"Loaded sparse index with {len(self.indexed_chunks)} chunks.")
            except Exception as e:
                logger.error(f"Failed to load sparse index from pickle file: {e}")
                self.bm25 = None
                self.indexed_chunks = []
        else:
            logger.info(
                f"No existing sparse index file found at '{self.storage_path}'. Starting fresh."
            )
            self.bm25 = None
            self.indexed_chunks = []