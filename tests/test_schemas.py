import pytest
from pydantic import ValidationError

from src.ingestion.schemas import Chunk, ChunkMetadata, Document, DocumentMetadata, FileType


def make_doc_metadata(**overrides):
    defaults = dict(source_path="sample_data/doc.pdf", file_type=FileType.PDF)
    defaults.update(overrides)
    return DocumentMetadata(**defaults)


def make_chunk_metadata(**overrides):
    defaults = dict(
        source_path="sample_data/doc.pdf",
        file_type=FileType.PDF,
        chunk_index=0,
        parent_document_id="doc-1",
    )
    defaults.update(overrides)
    return ChunkMetadata(**defaults)


def test_document_requires_nonempty_content():
    with pytest.raises(ValidationError):
        Document(page_content="", metadata=make_doc_metadata())


def test_document_generates_default_id():
    doc = Document(page_content="hello world", metadata=make_doc_metadata())
    assert isinstance(doc.id, str) and len(doc.id) > 0


def test_document_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Document(
            page_content="hello",
            metadata=make_doc_metadata(),
            unexpected_field="nope",
        )


def test_document_metadata_requires_source_path():
    with pytest.raises(ValidationError):
        DocumentMetadata(source_path="", file_type=FileType.PDF)


def test_chunk_metadata_requires_non_negative_chunk_index():
    with pytest.raises(ValidationError):
        make_chunk_metadata(chunk_index=-1)


def test_chunk_metadata_valid_case():
    meta = make_chunk_metadata(chunk_index=3, parent_document_id="doc-42")
    assert meta.chunk_index == 3
    assert meta.parent_document_id == "doc-42"


def test_chunk_requires_nonempty_content():
    with pytest.raises(ValidationError):
        Chunk(page_content="", metadata=make_chunk_metadata())


def test_chunk_valid_roundtrip():
    chunk = Chunk(page_content="some real chunk text", metadata=make_chunk_metadata())
    assert chunk.page_content == "some real chunk text"
    assert chunk.metadata.chunk_index == 0
