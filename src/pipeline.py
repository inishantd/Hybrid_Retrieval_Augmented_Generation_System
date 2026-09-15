import logging
from typing import Dict, Any

from src.retrieval.sparse import SparseBM25Index
from src.retrieval.dense import DenseVectorIndex
from src.retrieval.hybrid_retrieval import HybridRetriever

from src.reranking.cross_encoder import DocumentReranker
from src.generation.generator import GroundedGenerator
from src.generation.verifier import CitationVerifier

from src.config import config


logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(self):
        logger.info("Initializing RAG pipeline...")

        # Retrieval
        # NOTE: storage_path must match DocumentIngestionPipeline's

        self.sparse_index = SparseBM25Index(storage_path=str(config.SPARSE_INDEX_PATH))
        self.sparse_index.load_index()

        self.dense_index = DenseVectorIndex()

        self.hybrid_retriever = HybridRetriever(
            sparse_index=self.sparse_index,
            dense_index=self.dense_index,
        )

        # Reranking
        self.reranker = DocumentReranker()

        # Generation
        self.generator = GroundedGenerator()

        # Verification
        self.verifier = CitationVerifier()

        logger.info("RAG pipeline initialized successfully.")

    def query(self, user_query: str, source_path: str = None) -> Dict[str, Any]:
        """Run a user query through the complete RAG pipeline.

        source_path is optional and defaults to None (search every
        indexed document -- unchanged default behavior). Pass a document's
        metadata.source_path to scope retrieval to just that document
        (e.g. right after it's uploaded), without disabling hybrid
        retrieval or touching reranking/generation/verification.
        """

        if not user_query or not user_query.strip():
            return {
                "answer": "Please provide a valid question.",
                "is_context_sufficient": False,
                "citation_verification": {
                    "is_valid": False,
                    "flagged_issues": {
                        "query": "EMPTY_QUERY"
                    },
                    "validated_indices": [],
                },
            }

        user_query = user_query.strip()


        # 1. Hybrid Retrieval


        hybrid_results = self.hybrid_retriever.retrieve(
            query=user_query,
            top_k=config.RETRIEVAL_TOP_K,
            source_path=source_path,
        )

        if not hybrid_results:
            return {
                "answer": (
                    "I am sorry, but the internal documentation "
                    "does not provide sufficient context to answer "
                    "this question."
                ),
                "is_context_sufficient": False,
                "citation_verification": {
                    "is_valid": False,
                    "flagged_issues": {
                        "context": "NO_RETRIEVAL_RESULTS"
                    },
                    "validated_indices": [],
                },
                "retrieval_results": [],
                "reranked_results": [],
            }


        #  Cross-Encoder Reranking


        reranked_results = self.reranker.rerank(
            query=user_query,
            hybrid_results=hybrid_results,
            top_n=config.RERANK_TOP_N,
        )


        # Grounded Generation


        llm_output = self.generator.generate_answer(
            query=user_query,
            top_chunks=reranked_results,
        )

        generated_answer = llm_output.get(
            "answer",
            "",
        )

        context_sufficient = llm_output.get(
            "is_context_sufficient",
            False,
        )


        # Citation Verification


        citation_result = self.verifier.verify_citations(
            llm_answer=generated_answer,
            top_chunks=reranked_results,
        )

        # Final result


        return {
            "query": user_query,
            "answer": generated_answer,
            "is_context_sufficient": context_sufficient,
            "citation_verification": citation_result,
            "retrieval_results": hybrid_results,
            "reranked_results": reranked_results,
        }