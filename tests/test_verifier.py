import pytest

from src.generation.verifier import CitationVerifier
from src.ingestion.schemas import Chunk, ChunkMetadata, FileType


def make_chunk(text, chunk_index=0):
    meta = ChunkMetadata(
        source_path="sample_data/doc.pdf",
        file_type=FileType.PDF,
        chunk_index=chunk_index,
        parent_document_id="doc-1",
    )
    return Chunk(page_content=text, metadata=meta)


def test_empty_answer_is_invalid():
    result = CitationVerifier.verify_citations("", top_chunks=[{"chunk": make_chunk("some text")}])
    assert result["is_valid"] is False
    assert "answer" in result["flagged_issues"]


def test_empty_context_is_invalid():
    result = CitationVerifier.verify_citations("An answer [1].", top_chunks=[])
    assert result["is_valid"] is False
    assert "context" in result["flagged_issues"]


def test_valid_citation_is_accepted():
    chunks = [{"chunk": make_chunk("Employees get 20 days of leave.")}]
    result = CitationVerifier.verify_citations("Leave policy is generous [1].", chunks)

    assert result["is_valid"] is True
    assert result["validated_indices"] == [1]
    assert result["flagged_issues"] == {}


def test_out_of_range_citation_is_flagged():
    chunks = [{"chunk": make_chunk("Only one chunk here.")}]
    result = CitationVerifier.verify_citations("This cites something [5].", chunks)

    assert result["is_valid"] is False
    assert "[5]" in result["flagged_issues"]
    assert result["validated_indices"] == []


def test_citation_pointing_at_blank_chunk_is_flagged():
    chunks = [{"chunk": make_chunk(" ")}]  # whitespace-only content
    result = CitationVerifier.verify_citations("Cites the only chunk [1].", chunks)

    assert result["is_valid"] is False
    assert "[1]" in result["flagged_issues"]


def test_mixed_valid_and_invalid_citations():
    chunks = [{"chunk": make_chunk("Real content here.")}]
    result = CitationVerifier.verify_citations("Valid [1] and invalid [2].", chunks)

    assert result["is_valid"] is False
    assert result["validated_indices"] == [1]
    assert "[2]" in result["flagged_issues"]
    assert "[1]" not in result["flagged_issues"]
