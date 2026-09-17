import pytest

from src.ingestion.parsers import DocumentParserRouter
from src.ingestion.schemas import FileType


def test_sanitize_removes_null_bytes():
    assert "\x00" not in DocumentParserRouter.sanitize_unicode_string("hello\x00world")


def test_sanitize_normalizes_pdf_bullet_character():
    raw = "item one" + "\uf0b7" + "item two"
    result = DocumentParserRouter.sanitize_unicode_string(raw)
    assert "\uf0b7" not in result
    assert "- item two" in result


def test_sanitize_collapses_blank_lines():
    result = DocumentParserRouter.sanitize_unicode_string("line one\n\n\n   \nline two")
    assert result == "line one\nline two"


def test_sanitize_empty_input_returns_empty_string():
    assert DocumentParserRouter.sanitize_unicode_string("") == ""
    assert DocumentParserRouter.sanitize_unicode_string(None) == ""


def test_content_hash_is_deterministic():
    h1 = DocumentParserRouter.calculate_content_hash("some content")
    h2 = DocumentParserRouter.calculate_content_hash("some content")
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest


def test_content_hash_differs_for_different_content():
    h1 = DocumentParserRouter.calculate_content_hash("content A")
    h2 = DocumentParserRouter.calculate_content_hash("content B")
    assert h1 != h2


def test_extract_title_uses_markdown_h1(tmp_path):
    path = tmp_path / "doc.md"
    title = DocumentParserRouter.extract_document_title(path, "# My Real Title\n\nBody text")
    assert title == "My Real Title"


def test_extract_title_falls_back_to_filename_stem(tmp_path):
    path = tmp_path / "quarterly-report.txt"
    title = DocumentParserRouter.extract_document_title(path, "No heading here, just text.")
    assert title == "quarterly-report"


def test_process_file_raises_for_missing_file(tmp_path):
    router = DocumentParserRouter()
    with pytest.raises(FileNotFoundError):
        router.process_file(tmp_path / "does_not_exist.txt")


def test_process_file_raises_for_unsupported_extension(tmp_path):
    path = tmp_path / "data.xyz"
    path.write_text("some content")
    router = DocumentParserRouter()
    with pytest.raises(ValueError):
        router.process_file(path)


def test_process_file_parses_plain_txt(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Employees must be punctual and professional.")

    router = DocumentParserRouter()
    document = router.process_file(path)

    assert document.metadata.file_type == FileType.TXT
    assert "punctual" in document.page_content
    assert document.metadata.content_hash is not None
    assert document.metadata.document_title == "notes"


def test_process_file_raises_for_empty_txt(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n   ")  # whitespace only -> sanitizes to ""

    router = DocumentParserRouter()
    with pytest.raises(ValueError):
        router.process_file(path)


def test_same_file_gives_same_document_id(tmp_path):
    file = tmp_path / "doc.txt"
    file.write_text("Employees get 20 days of leave.")

    first = DocumentParserRouter().process_file(file)
    second = DocumentParserRouter().process_file(file)

    assert first.id == second.id