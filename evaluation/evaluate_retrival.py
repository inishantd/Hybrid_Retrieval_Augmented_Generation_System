import json
import math
from pathlib import Path
from typing import List, Dict, Any, Set

from src.config import config
from src.retrieval.sparse import SparseBM25Index
from src.retrieval.dense import DenseVectorIndex
from src.retrieval.hybrid_retrieval import HybridRetriever
from src.reranking.cross_encoder import DocumentReranker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_FILE = PROJECT_ROOT / "data" / "golden_candidates.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "retrieval_evaluation.json"

TOP_K = 20

# METRICS

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Recall@K:

        Number of relevant chunks retrieved in top K
        ------------------------------------------------
        Total number of relevant chunks
    """

    if not relevant_ids:
        return 0.0

    retrieved_top_k = set(retrieved_ids[:k])

    hits = retrieved_top_k.intersection(relevant_ids)

    return len(hits) / len(relevant_ids)


def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
) -> float:
    """
    Reciprocal Rank:

        1 / rank of first relevant result

    If no relevant result is found:

        0.0
    """

    if not relevant_ids:
        return 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids,
        start=1,
    ):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Discounted Cumulative Gain@K.

    Binary relevance:

        relevant   = 1
        irrelevant = 0
    """

    score = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids[:k],
        start=1,
    ):

        if chunk_id in relevant_ids:

            score += 1.0 / math.log2(rank + 1)

    return score


def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
) -> float:
    """
    Normalized DCG@K.
    """

    if not relevant_ids:
        return 0.0

    actual_dcg = dcg_at_k(
        retrieved_ids,
        relevant_ids,
        k,
    )

    # Ideal ranking:
    # all relevant chunks appear first.
    ideal_ids = list(relevant_ids)

    ideal_dcg = dcg_at_k(
        ideal_ids,
        relevant_ids,
        k,
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def calculate_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
) -> Dict[str, float]:

    relevant_set = set(relevant_ids)

    return {
        "recall@1": recall_at_k(
            retrieved_ids,
            relevant_set,
            1,
        ),

        "recall@3": recall_at_k(
            retrieved_ids,
            relevant_set,
            3,
        ),

        "recall@5": recall_at_k(
            retrieved_ids,
            relevant_set,
            5,
        ),

        "recall@10": recall_at_k(
            retrieved_ids,
            relevant_set,
            10,
        ),

        "recall@20": recall_at_k(
            retrieved_ids,
            relevant_set,
            20,
        ),

        "mrr": reciprocal_rank(
            retrieved_ids,
            relevant_set,
        ),

        "ndcg@5": ndcg_at_k(
            retrieved_ids,
            relevant_set,
            5,
        ),

        "ndcg@10": ndcg_at_k(
            retrieved_ids,
            relevant_set,
            10,
        ),

        "ndcg@20": ndcg_at_k(
            retrieved_ids,
            relevant_set,
            20,
        ),
    }


def average_metrics(
    results: List[Dict[str, float]],
) -> Dict[str, float]:

    if not results:
        return {}

    metric_names = results[0].keys()

    return {
        metric: sum(
            result[metric]
            for result in results
        ) / len(results)

        for metric in metric_names
    }

# RETRIEVAL HELPERS

def sparse_retrieve(
    sparse: SparseBM25Index,
    query: str,
) -> List[str]:

    results = sparse.search(
        query,
        top_k=TOP_K,
    )

    return [
        result["chunk"].id
        for result in results
    ]


def dense_retrieve(
    dense: DenseVectorIndex,
    query: str,
) -> List[str]:

    results = dense.search(
        query,
        top_k=TOP_K,
    )

    return [
        result["chunk_id"]
        for result in results
    ]


def hybrid_retrieve(
    hybrid: HybridRetriever,
    query: str,
) -> List[str]:

    results = hybrid.retrieve(
        query,
        top_k=TOP_K,
    )

    return [
        result["chunk"].id
        for result in results
    ]


def hybrid_rerank(
    hybrid: HybridRetriever,
    reranker: DocumentReranker,
    query: str,
) -> List[str]:

    hybrid_results = hybrid.retrieve(
        query,
        top_k=TOP_K,
    )

    reranked_results = reranker.rerank(
        query,
        hybrid_results,
        top_n=TOP_K,
    )

    return [
        result["chunk"].id
        for result in reranked_results
    ]

# EVALUATE ONE SYSTEM

def evaluate_system(
    name: str,
    dataset: List[Dict[str, Any]],
    retrieval_function,
) -> Dict[str, Any]:

    print()
    print("=" * 80)
    print(f"EVALUATING: {name}")
    print("=" * 80)

    query_results = []
    metric_results = []

    for index, item in enumerate(
        dataset,
        start=1,
    ):

        query = item["query"]

        relevant_ids = item.get(
            "relevant_chunk_ids",
            [],
        )

        retrieved_ids = retrieval_function(
            query
        )

        metrics = calculate_metrics(
            retrieved_ids,
            relevant_ids,
        )

        metric_results.append(metrics)

        query_results.append(
            {
                "id": item["id"],
                "query": query,
                "relevant_chunk_ids": relevant_ids,
                "retrieved_chunk_ids": retrieved_ids,
                "metrics": metrics,
            }
        )

        print(
            f"[{index:02d}/{len(dataset)}] "
            f"{item['id']} | "
            f"Recall@5={metrics['recall@5']:.3f} | "
            f"MRR={metrics['mrr']:.3f}"
        )

    summary = average_metrics(
        metric_results
    )

    return {
        "system": name,
        "summary": summary,
        "queries": query_results,
    }


# MAIN

def main():

    print("=" * 80)
    print("ENTERPRISE HYBRID RAG - RETRIEVAL EVALUATION")
    print("=" * 80)

    # Load golden dataset

    with open(
        GOLDEN_FILE,
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(file)

    print(
        f"Golden questions loaded: {len(dataset)}"
    )

    # Validate ground truth

    questions_without_ground_truth = [
        item["id"]
        for item in dataset
        if not item.get("relevant_chunk_ids")
    ]

    if questions_without_ground_truth:

        raise ValueError(
            "The following questions have no "
            f"relevant_chunk_ids: "
            f"{questions_without_ground_truth}"
        )

    print(
        "Ground-truth validation: PASSED"
    )

    # Load Sparse index

    print()
    print("Loading BM25 index...")

    sparse = SparseBM25Index(
        storage_path=str(
            config.SPARSE_INDEX_PATH
        )
    )

    sparse.load_index()

    print(
        f"Sparse chunks loaded: "
        f"{len(sparse.indexed_chunks)}"
    )

    # Load Dense index

    print()
    print("Loading ChromaDB...")

    dense = DenseVectorIndex()

    print(
        "Dense index loaded."
    )

    # Hybrid

    hybrid = HybridRetriever(
        sparse_index=sparse,
        dense_index=dense,
    )

    # Cross Encoder

    print()
    print("Loading Cross-Encoder...")

    reranker = DocumentReranker(
        model_name=config.RERANKER_MODEL_NAME
    )

    print(
        "Cross-Encoder loaded."
    )

    # SPARSE

    sparse_result = evaluate_system(
        "Sparse BM25",
        dataset,
        lambda query: sparse_retrieve(
            sparse,
            query,
        ),
    )

    # DENSE

    dense_result = evaluate_system(
        "Dense Vector",
        dataset,
        lambda query: dense_retrieve(
            dense,
            query,
        ),
    )

    # HYBRID

    hybrid_result = evaluate_system(
        "Hybrid RRF",
        dataset,
        lambda query: hybrid_retrieve(
            hybrid,
            query,
        ),
    )

    # HYBRID + RERANKER

    reranked_result = evaluate_system(
        "Hybrid + Cross-Encoder",
        dataset,
        lambda query: hybrid_rerank(
            hybrid,
            reranker,
            query,
        ),
    )


    # SAVE RESULTS


    report = {
        "dataset": {
            "num_questions": len(dataset),
            "top_k": TOP_K,
        },

        "systems": [
            sparse_result,
            dense_result,
            hybrid_result,
            reranked_result,
        ],
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # PRINT FINAL TABLE

    print()
    print("=" * 110)
    print("FINAL RETRIEVAL RESULTS")
    print("=" * 110)

    header = (
        f"{'System':<28}"
        f"{'R@1':>8}"
        f"{'R@5':>8}"
        f"{'R@10':>8}"
        f"{'R@20':>8}"
        f"{'MRR':>8}"
        f"{'NDCG@5':>10}"
        f"{'NDCG@10':>10}"
        f"{'NDCG@20':>10}"
    )

    print(header)
    print("-" * 110)

    for system in report["systems"]:

        summary = system["summary"]

        print(
            f"{system['system']:<28}"
            f"{summary['recall@1']:>8.3f}"
            f"{summary['recall@5']:>8.3f}"
            f"{summary['recall@10']:>8.3f}"
            f"{summary['recall@20']:>8.3f}"
            f"{summary['mrr']:>8.3f}"
            f"{summary['ndcg@5']:>10.3f}"
            f"{summary['ndcg@10']:>10.3f}"
            f"{summary['ndcg@20']:>10.3f}"
        )

    print("-" * 110)

    print()
    print(
        f"Detailed report saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 110)
    print("EVALUATION COMPLETED")
    print("=" * 110)


if __name__ == "__main__":
    main()