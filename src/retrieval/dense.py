import logging
from typing import List, Dict, Any
import chromadb
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

from src.ingestion.schemas import Chunk
from src.config import config

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """Custom embedding function wrapper for ChromaDB using local SentenceTransformer."""

    def __init__(self, model_name: str):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model '{model_name}' on device: '{self.device}'")
        self.model = SentenceTransformer(model_name, device=self.device)

    def __call__(self, input: Documents) -> Embeddings:
        """Encode text documents into dense float vectors for ChromaDB indexing."""
        embeddings = self.model.encode(input, convert_to_numpy=True)
        return embeddings.tolist()


class DenseVectorIndex:
    """Manages dense vector storage and semantic HNSW search in ChromaDB."""

    def __init__(self, collection_name: str = "internal_docs"):
        self.client = chromadb.PersistentClient(path=config.CHROMA_STORAGE_DIR)
        self.embedding_fn = LocalSentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_chunks(self, chunks: List[Chunk]) -> None:
        """Upsert document chunks into the ChromaDB vector collection."""
        if not chunks:
            logger.info("Empty chunk list provided to dense vector index. Ingestion skipped.")
            return

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.page_content for chunk in chunks]
        metadatas = [
            {
                "source_path": chunk.metadata.source_path,
                "file_type": chunk.metadata.file_type,
                "chunk_index": chunk.metadata.chunk_index,
                "parent_document_id": chunk.metadata.parent_document_id,
            }
            for chunk in chunks
        ]

        logger.info(f"Upserting {len(chunks)} chunks into ChromaDB collection...")
        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def search(
        self, query: str, top_k: int = 5, source_path: str = None
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB for top-k nearest semantic neighbor chunks.
        """
        query_kwargs = {"query_texts": [query], "n_results": top_k}
        if source_path:
            query_kwargs["where"] = {"source_path": source_path}

        results = self.collection.query(**query_kwargs)

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        formatted_results = []
        for idx in range(len(results["ids"][0])):
            distance = results["distances"][0][idx] if results["distances"] else 1.0
            # ChromaDB cosine space returns Cosine Distance; convert to Similarity (1.0 - Distance)
            similarity_score = float(1.0 - distance)

            formatted_results.append(
                {
                    "chunk_id": results["ids"][0][idx],
                    "text": results["documents"][0][idx],
                    "metadata": results["metadatas"][0][idx],
                    "dense_score": similarity_score,
                }
            )

        formatted_results.sort(key=lambda x: x["dense_score"], reverse=True)
        return formatted_results
