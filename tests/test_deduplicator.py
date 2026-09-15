import numpy as np

from src.ingestion.deduplicator import ChunkDeduplicator
from src.ingestion.schemas import Chunk, ChunkMetadata, FileType


def make_chunk(text, chunk_index):
    meta = ChunkMetadata(
        source_path="sample_data/doc.pdf",
        file_type=FileType.PDF,
        chunk_index=chunk_index,
        parent_document_id="doc-1",
    )
    return Chunk(page_content=text, metadata=meta)


def test_cosine_similarity_identical_vectors_is_one():
    v = np.array([1.0, 2.0, 3.0])
    assert ChunkDeduplicator.calculate_cosine_similarity(v, v) == 1.0


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert ChunkDeduplicator.calculate_cosine_similarity(a, b) == 0.0


def test_cosine_similarity_zero_vector_is_zero_not_nan():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 1.0])
    assert ChunkDeduplicator.calculate_cosine_similarity(a, b) == 0.0


def test_deduplicate_removes_near_duplicate_chunk():
    chunks = [
        make_chunk("Employees must be punctual.", 0),
        make_chunk("Employees must be punctual!!", 1),  # near-duplicate embedding
        make_chunk("The company was founded in 1990.", 2),  # distinct
    ]

    # Stub embedding function: chunk 0 and 1 map to nearly identical
    # vectors (similarity > 0.95); chunk 2 maps to an orthogonal vector.
    vectors = {
        0: [1.0, 0.0, 0.0],
        1: [0.99, 0.01, 0.0],
        2: [0.0, 1.0, 0.0],
    }

    def stub_embedding_fn(texts):
        # deduplicate() passes texts in the same order as `chunks`
        return [vectors[i] for i in range(len(texts))]

    deduped = ChunkDeduplicator().deduplicate(chunks, embedding_fn=stub_embedding_fn, threshold=0.95)

    assert len(deduped) == 2
    kept_indices = {c.metadata.chunk_index for c in deduped}
    assert kept_indices == {0, 2}


def test_deduplicate_empty_input_returns_empty_list():
    result = ChunkDeduplicator().deduplicate([], embedding_fn=lambda texts: [], threshold=0.95)
    assert result == []
