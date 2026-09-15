import logging
import numpy as np
from typing import List, Callable
from src.ingestion.schemas import Chunk

logger = logging.getLogger(__name__)


class ChunkDeduplicator:
    """Removes semantically duplicate chunks based on vector cosine similarity."""

    @staticmethod
    def calculate_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two 1D numpy vector arrays."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    def deduplicate(
        self,
        chunks: List[Chunk],
        embedding_fn: Callable[[List[str]], List[List[float]]],
        threshold: float = 0.95,
    ) -> List[Chunk]:
        """Filter out chunks whose embedding similarity with an already accepted chunk exceeds threshold."""
        if not chunks:
            return []

        # Generate dense embeddings for all chunk text payloads in bulk
        texts = [chunk.page_content for chunk in chunks]
        embeddings_list = embedding_fn(texts)
        embeddings = [np.array(vec) for vec in embeddings_list]

        unique_chunks: List[Chunk] = []
        seen_vectors: List[np.ndarray] = []
        removed_count = 0

        for idx, current_vector in enumerate(embeddings):
            is_duplicate = False

            # Compare current chunk vector against recently accepted vectors (up to last 50)
            check_window = seen_vectors[-50:] if len(seen_vectors) > 50 else seen_vectors
            for seen_vector in check_window:
                similarity = self.calculate_cosine_similarity(current_vector, seen_vector)
                if similarity > threshold:
                    is_duplicate = True
                    removed_count += 1
                    break

            if not is_duplicate:
                unique_chunks.append(chunks[idx])
                seen_vectors.append(current_vector)

        if removed_count > 0:
            logger.info(f"Deduplicated {removed_count} chunks (threshold={threshold})")

        return unique_chunks
