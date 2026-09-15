from src.retrieval.hybrid_retrieval import HybridRetriever
from src.ingestion.schemas import Chunk, ChunkMetadata, FileType


def make_chunk(chunk_id, text, chunk_index):
    meta = ChunkMetadata(
        source_path="sample_data/doc.pdf",
        file_type=FileType.PDF,
        chunk_index=chunk_index,
        parent_document_id="doc-1",
    )
    chunk = Chunk(id=chunk_id, page_content=text, metadata=meta)
    return chunk


class StubSparseIndex:
    """Duck-types SparseBM25Index.search() without needing a real BM25 model."""

    def __init__(self, results):
        self._results = results

    def search(self, query, top_k=10, source_path=None):
        return self._results


class StubDenseIndex:
    """Duck-types DenseVectorIndex.search() without downloading any model."""

    def __init__(self, results):
        self._results = results

    def search(self, query, top_k=10, source_path=None):
        return self._results


def test_rrf_fuses_and_ranks_by_combined_score():
    chunk_a = make_chunk("a", "chunk A content", 0)
    chunk_b = make_chunk("b", "chunk B content", 1)

    # Sparse ranks: a (rank 1), b (rank 2)
    sparse = StubSparseIndex([
        {"chunk": chunk_a, "sparse_score": 5.0},
        {"chunk": chunk_b, "sparse_score": 3.0},
    ])

    # Dense ranks: b (rank 1), c (rank 2, sparse-unseen)
    dense = StubDenseIndex([
        {
            "chunk_id": "b",
            "text": "chunk B content",
            "metadata": {
                "source_path": "sample_data/doc.pdf",
                "file_type": "pdf",
                "chunk_index": 1,
                "parent_document_id": "doc-1",
            },
            "dense_score": 0.9,
        },
        {
            "chunk_id": "c",
            "text": "chunk C content",
            "metadata": {
                "source_path": "sample_data/doc.pdf",
                "file_type": "pdf",
                "chunk_index": 2,
                "parent_document_id": "doc-1",
            },
            "dense_score": 0.5,
        },
    ])

    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)
    results = retriever.retrieve("some query", top_k=3, rrf_k=60)

    ids_in_order = [r["chunk"].id for r in results]
    # b appears in both lists (rank 1 in each) so it must fuse to the
    # highest combined RRF score and come first.
    assert ids_in_order[0] == "b"
    assert set(ids_in_order) == {"a", "b", "c"}

    scores = {r["chunk"].id: r["score"] for r in results}
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["c"]
    # a was sparse-rank-1, c was dense-rank-2 -> a should outrank c.
    assert scores["a"] > scores["c"]

    sources = {r["chunk"].id: set(r["sources"]) for r in results}
    assert sources["a"] == {"sparse"}
    assert sources["b"] == {"sparse", "dense"}
    assert sources["c"] == {"dense"}


def test_retrieve_respects_top_k():
    chunk_a = make_chunk("a", "content a", 0)
    chunk_b = make_chunk("b", "content b", 1)
    chunk_c = make_chunk("c", "content c", 2)

    sparse = StubSparseIndex([
        {"chunk": chunk_a, "sparse_score": 3.0},
        {"chunk": chunk_b, "sparse_score": 2.0},
        {"chunk": chunk_c, "sparse_score": 1.0},
    ])
    dense = StubDenseIndex([])

    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)
    results = retriever.retrieve("query", top_k=2)

    assert len(results) == 2


def test_retrieve_with_no_results_returns_empty_list():
    retriever = HybridRetriever(sparse_index=StubSparseIndex([]), dense_index=StubDenseIndex([]))
    assert retriever.retrieve("anything", top_k=10) == []


class RecordingSparseIndex:
    """Records the source_path it was called with, to verify it gets
    forwarded from HybridRetriever.retrieve() down to the sparse index."""

    def __init__(self):
        self.last_source_path = "not called"

    def search(self, query, top_k=10, source_path=None):
        self.last_source_path = source_path
        return []


class RecordingDenseIndex:
    def __init__(self):
        self.last_source_path = "not called"

    def search(self, query, top_k=10, source_path=None):
        self.last_source_path = source_path
        return []


def test_retrieve_forwards_source_path_scope_to_both_indexes():
    """Regression test for scoping search to a just-uploaded document:
    HybridRetriever.retrieve(source_path=...) must reach BOTH the sparse
    and dense index's search() calls unchanged, and default to None
    (search everything) when not given."""
    sparse = RecordingSparseIndex()
    dense = RecordingDenseIndex()
    retriever = HybridRetriever(sparse_index=sparse, dense_index=dense)

    retriever.retrieve("query", top_k=5, source_path="sample_data/new_upload.pdf")
    assert sparse.last_source_path == "sample_data/new_upload.pdf"
    assert dense.last_source_path == "sample_data/new_upload.pdf"

    retriever.retrieve("query", top_k=5)
    assert sparse.last_source_path is None
    assert dense.last_source_path is None
