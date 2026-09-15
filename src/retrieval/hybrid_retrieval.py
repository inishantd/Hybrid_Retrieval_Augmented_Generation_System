from typing import List, Dict, Any
from src.retrieval.sparse import SparseBM25Index
from src.retrieval.dense import DenseVectorIndex
from src.ingestion.schemas import Chunk, ChunkMetadata


class HybridRetriever:
    """Fuses Sparse (BM25) and Dense (Vector) search results using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, sparse_index: SparseBM25Index, dense_index: DenseVectorIndex):
        self.sparse_index = sparse_index
        self.dense_index = dense_index

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        source_path: str = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute parallel sparse and dense queries, then fuse candidate rankings.
        RRF Formula: score(d) = sum(1 / (rrf_k + rank(d))) across all search strategies.
        """
        # Fetch top_k * 2 candidates from each index to ensure sufficient overlap
        candidate_k = top_k * 2
        sparse_results = self.sparse_index.search(query, top_k=candidate_k, source_path=source_path)
        dense_results = self.dense_index.search(query, top_k=candidate_k, source_path=source_path)

        rrf_scores: Dict[str, Dict[str, Any]] = {}

        # Process Sparse (BM25) search rankings
        for rank, res in enumerate(sparse_results):
            chunk: Chunk = res["chunk"]
            chunk_id = chunk.id

            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = {"chunk": chunk, "score": 0.0, "sources": ["sparse"]}

            rrf_scores[chunk_id]["score"] += 1.0 / (rrf_k + (rank + 1))

        # Process Dense (ChromaDB Vector) search rankings
        for rank, res in enumerate(dense_results):
            chunk_id = res["chunk_id"]

            if chunk_id not in rrf_scores:
                meta = res["metadata"]
                metadata = ChunkMetadata(
                    source_path=meta["source_path"],
                    file_type=meta["file_type"],
                    chunk_index=meta["chunk_index"],
                    parent_document_id=meta["parent_document_id"],
                )
                chunk = Chunk(id=chunk_id, page_content=res["text"], metadata=metadata)
                rrf_scores[chunk_id] = {"chunk": chunk, "score": 0.0, "sources": ["dense"]}
            else:
                rrf_scores[chunk_id]["sources"].append("dense")

            rrf_scores[chunk_id]["score"] += 1.0 / (rrf_k + (rank + 1))

        # 3. Sort fused candidates by combined RRF score descending
        fused_results = list(rrf_scores.values())
        fused_results.sort(key=lambda x: x["score"], reverse=True)

        return fused_results[:top_k]
