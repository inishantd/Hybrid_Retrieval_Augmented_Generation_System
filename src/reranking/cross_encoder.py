import logging
from typing import List, Dict, Any
import torch
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class DocumentReranker:
    """Reranks retrieved candidate chunks using a cross-attention CrossEncoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Loading CrossEncoder reranker model '{model_name}' on device: '{self.device}'"
        )
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self, query: str, hybrid_results: List[Dict[str, Any]], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Compute pairwise relevance scores between query and document text payloads.
        Returns top_n reranked chunk dictionaries.
        """
        if not hybrid_results:
            return []

        # Build [query, chunk_text] input pairs for the cross-encoder
        pairs = [[query, res["chunk"].page_content] for res in hybrid_results]

        # Predict relevance logits across all candidate pairs
        scores = self.model.predict(pairs)

        reranked_results = [
            {
                "chunk": hybrid_results[idx]["chunk"],
                "rerank_score": float(score),
                "previous_sources": hybrid_results[idx].get("sources", []),
            }
            for idx, score in enumerate(scores)
        ]

        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results[:top_n]
