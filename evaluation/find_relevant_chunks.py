import json
from pathlib import Path

from src.config import config
from src.retrieval.sparse import SparseBM25Index
from src.retrieval.dense import DenseVectorIndex
from src.retrieval.hybrid_retrieval import HybridRetriever
from src.reranking.cross_encoder import DocumentReranker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_DATASET = PROJECT_ROOT / "data" / "golden_dataset.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "golden_candidates.json"


def main():

    print("=" * 80)
    print("GOLDEN DATASET CANDIDATE GENERATION")
    print("=" * 80)

    # Load golden questions

    with open(GOLDEN_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} questions")

    # Initialize retrieval components

    sparse = SparseBM25Index(
        storage_path=str(config.SPARSE_INDEX_PATH)
    )

    sparse.load_index()

    dense = DenseVectorIndex()

    hybrid = HybridRetriever(
        sparse_index=sparse,
        dense_index=dense
    )

    reranker = DocumentReranker(
        model_name=config.RERANKER_MODEL_NAME
    )

    # Generate candidates

    all_candidates = []

    for number, item in enumerate(dataset, start=1):

        query = item["query"]

        print()
        print("=" * 80)
        print(f"[{number}/{len(dataset)}] {item['id']}")
        print(query)
        print("=" * 80)

        # Retrieve more candidates than production top-k.
        hybrid_results = hybrid.retrieve(
            query,
            top_k=20
        )

        # Rerank candidates.
        reranked_results = reranker.rerank(
            query,
            hybrid_results,
            top_n=20
        )

        candidates = []

        for rank, result in enumerate(reranked_results, start=1):

            chunk = result["chunk"]

            candidate = {
                "rank": rank,
                "chunk_id": chunk.id,
                "rerank_score": result["rerank_score"],
                "previous_sources": result.get(
                    "previous_sources",
                    []
                ),
                "chunk_index": chunk.metadata.chunk_index,
                "text": chunk.page_content,
            }

            candidates.append(candidate)

            print()
            print(f"RANK {rank}")
            print(f"CHUNK ID: {chunk.id}")
            print(f"CHUNK INDEX: {chunk.metadata.chunk_index}")
            print(f"SOURCES: {result.get('previous_sources', [])}")
            print(f"RERANK SCORE: {result['rerank_score']:.4f}")
            print("-" * 60)
            print(chunk.page_content[:500].replace("\n", " "))
            print("-" * 60)

        all_candidates.append(
            {
                "id": item["id"],
                "query": query,
                "type": item["type"],
                "candidates": candidates,
                "relevant_chunk_ids": []
            }
        )

    # Save candidates

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            all_candidates,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 80)
    print("CANDIDATE GENERATION COMPLETED")
    print("=" * 80)
    print(f"Output: {OUTPUT_FILE}")
    print()
    print(
        "Review the candidates and fill "
        "'relevant_chunk_ids'."
    )


if __name__ == "__main__":
    main()