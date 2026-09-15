from src.retrieval.sparse import SparseBM25Index
from src.ingestion.schemas import Chunk, ChunkMetadata, FileType


def make_chunk(chunk_id, text, parent_id, chunk_index):
    meta = ChunkMetadata(
        source_path=f"sample_data/{parent_id}.txt",
        file_type=FileType.TXT,
        chunk_index=chunk_index,
        parent_document_id=parent_id,
    )
    return Chunk(id=chunk_id, page_content=text, metadata=meta)


def test_reindexing_without_loading_first_overwrites_previous_chunks(tmp_path):
    """Documents the bug: an index_chunks() call on a FRESH
    SparseBM25Index (no load_index() call) wipes out whatever was
    already persisted on disk, because indexed_chunks starts empty and
    save_index() overwrites the pickle unconditionally."""
    storage_path = str(tmp_path / "sparse_index.pkl")

    first_run = SparseBM25Index(storage_path=storage_path)
    first_run.index_chunks([make_chunk("doc1-0", "first document content", "doc1", 0)])

    # Simulate DocumentIngestionPipeline's old (buggy) behavior: a new
    # SparseBM25Index instance is created for the second ingest, and
    # load_index() is never called before index_chunks().
    second_run_buggy = SparseBM25Index(storage_path=storage_path)
    second_run_buggy.index_chunks([make_chunk("doc2-0", "second document content", "doc2", 0)])

    reloaded = SparseBM25Index(storage_path=storage_path)
    reloaded.load_index()
    assert len(reloaded.indexed_chunks) == 1  # doc1's chunk is gone


def test_reindexing_after_loading_first_accumulates_chunks(tmp_path):
    """This is the fix applied in src/ingest.py: calling load_index()
    before index_chunks() makes a second ingestion run ADD to the
    existing index instead of replacing it."""
    storage_path = str(tmp_path / "sparse_index.pkl")

    first_run = SparseBM25Index(storage_path=storage_path)
    first_run.index_chunks([make_chunk("doc1-0", "first document content", "doc1", 0)])

    second_run_fixed = SparseBM25Index(storage_path=storage_path)
    second_run_fixed.load_index()  # <-- the fix
    second_run_fixed.index_chunks([make_chunk("doc2-0", "second document content", "doc2", 0)])

    reloaded = SparseBM25Index(storage_path=storage_path)
    reloaded.load_index()
    ids = {c.id for c in reloaded.indexed_chunks}
    assert ids == {"doc1-0", "doc2-0"}


def test_search_with_source_path_scopes_to_that_document_only(tmp_path):
    """Regression test for scoping search to a just-uploaded document:
    when the corpus has a third, unrelated chunk to keep BM25 IDF
    meaningful, search(source_path=...) for doc2 must return only doc2's
    chunk even though doc1 also mentions "report"."""
    storage_path = str(tmp_path / "sparse_index.pkl")

    index = SparseBM25Index(storage_path=storage_path)
    index.index_chunks(
        [
            make_chunk("doc1-0", "annual report on company finances", "doc1", 0),
            make_chunk("doc2-0", "incident report from the security team", "doc2", 0),
            make_chunk("doc3-0", "unrelated content about gardening tips", "doc3", 0),
        ]
    )

    unscoped = index.search("report", top_k=10)
    assert {r["chunk"].id for r in unscoped} == {"doc1-0", "doc2-0"}

    scoped = index.search("report", top_k=10, source_path="sample_data/doc2.txt")
    assert {r["chunk"].id for r in scoped} == {"doc2-0"}
