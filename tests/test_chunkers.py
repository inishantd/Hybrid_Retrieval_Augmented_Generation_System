import pytest

from src.ingestion.chunkers import ChunkingEngine
from src.ingestion.schemas import Document, DocumentMetadata, FileType


def make_document(text, file_type=FileType.TXT, source_path="sample_data/doc.txt"):
    return Document(
        page_content=text,
        metadata=DocumentMetadata(source_path=source_path, file_type=file_type),
    )


def test_fixed_size_chunk_respects_chunk_size_and_produces_sequential_index():
    text = "x" * 100
    doc = make_document(text)
    chunks = ChunkingEngine.fixed_size_chunk(doc, chunk_size=30, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 30
    assert [c.metadata.chunk_index for c in chunks] == list(range(len(chunks)))


def test_fixed_size_chunk_rejects_overlap_greater_or_equal_to_size():
    doc = make_document("some content")
    with pytest.raises(ValueError):
        ChunkingEngine.fixed_size_chunk(doc, chunk_size=10, chunk_overlap=10)


def test_fixed_size_chunk_blank_document_returns_no_chunks():
    doc = make_document("   ")
    chunks = ChunkingEngine.fixed_size_chunk(doc, chunk_size=100, chunk_overlap=10)
    assert chunks == []


def test_recursive_chunk_never_exceeds_chunk_size():
    text = ("This is a sentence. " * 200).strip()
    doc = make_document(text)
    chunks = ChunkingEngine.recursive_chunk(doc, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 200


def test_structure_aware_markdown_chunk_keeps_header_context():
    text = (
        "# Introduction\n\nSome intro text.\n\n"
        "## Details\n\nMore detailed text here."
    )
    doc = make_document(text, file_type=FileType.MARKDOWN, source_path="doc.md")
    chunks = ChunkingEngine.structure_aware_markdown_chunk(doc)

    assert len(chunks) == 2
    assert chunks[0].page_content.startswith("# Introduction")
    assert chunks[1].page_content.startswith("## Details")


def test_production_chunk_routes_markdown_to_structure_aware():
    text = "# Title\n\n" + ("Body text. " * 40)
    doc = make_document(text, file_type=FileType.MARKDOWN, source_path="doc.md")
    chunks = ChunkingEngine.production_chunk(doc, chunk_size=200, chunk_overlap=20)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert len(chunk.page_content) <= 200


def test_production_chunk_routes_pdf_to_plain_recursive():
    text = "Some plain PDF-extracted text. " * 50
    doc = make_document(text, file_type=FileType.PDF, source_path="doc.pdf")
    chunks = ChunkingEngine.production_chunk(doc, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.metadata.custom_attributes.get("chunking_strategy") == "recursive" for c in chunks)


def test_production_chunk_routes_html_to_structure_aware():
    """Regression test: production_chunk's routing check does
    `"markdown" in str(file_type).lower()` / `str(file_type).lower() in {"md", "html"}`.
    str(FileType.HTML) is "FileType.HTML", so str(...).lower() is
    "filetype.html" -- which matches NEITHER check. HTML documents
    therefore silently fall through to plain recursive_chunk instead of
    the structure-aware chunker the docstring promises. This test
    documents the intended behavior and currently FAILS against the bug.
    """
    text = "# Title\n\n" + ("Body text. " * 40)
    doc = make_document(text, file_type=FileType.HTML, source_path="doc.html")
    chunks = ChunkingEngine.production_chunk(doc, chunk_size=200, chunk_overlap=20)

    strategies = {c.metadata.custom_attributes.get("chunking_strategy") for c in chunks}
    assert "structure_recursive" in strategies, (
        "HTML documents should route through structure_aware_recursive_chunk, "
        f"but got strategies={strategies}. See src/ingestion/chunkers.py "
        "production_chunk()'s file_type routing check."
    )