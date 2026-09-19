"""Tests for the document ingestion pipeline."""

import base64
from pathlib import Path

import pytest
from pypdf import PdfWriter

from enterprise_ai_agent.ingestion import (
    DocumentParsingError,
    DocumentType,
    EmptyDocumentError,
    PdfLoader,
    UnsupportedFileTypeError,
    load_document,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _write_sample_pdf(tmp_path: Path) -> Path:
    encoded = (FIXTURES_DIR / "sample.pdf.b64").read_text(encoding="ascii")
    path = tmp_path / "sample.pdf"
    path.write_bytes(base64.b64decode(encoded))
    return path


def _write_blank_pdf(tmp_path: Path) -> Path:
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)

    with path.open("wb") as file:
        writer.write(file)

    return path


def test_load_document_loads_txt(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello\nworld", encoding="utf-8")

    document = load_document(path)

    assert document.content == "hello\nworld"
    assert document.metadata.file_type == DocumentType.TEXT


def test_load_document_loads_markdown(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Title\n\nBody", encoding="utf-8")

    document = load_document(path)

    assert document.content == "# Title\n\nBody"
    assert document.metadata.file_type == DocumentType.MARKDOWN


def test_load_document_supports_markdown_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.markdown"
    path.write_text("content", encoding="utf-8")

    document = load_document(path)

    assert document.metadata.file_type == DocumentType.MARKDOWN


def test_pdf_loader_loads_sample_pdf(tmp_path: Path) -> None:
    path = _write_sample_pdf(tmp_path)

    document = PdfLoader().load(path)

    assert "Hello from pypdf" in document.content
    assert document.metadata.file_type == DocumentType.PDF


def test_document_metadata_is_populated(tmp_path: Path) -> None:
    path = tmp_path / "metadata.txt"
    path.write_text("metadata", encoding="utf-8")

    document = load_document(path)

    assert document.metadata.source == path.resolve().as_posix()
    assert document.metadata.file_name == "metadata.txt"
    assert document.metadata.file_type == DocumentType.TEXT
    assert document.metadata.file_size == path.stat().st_size


def test_load_document_uses_extension_for_uppercase_suffix(tmp_path: Path) -> None:
    path = tmp_path / "notes.TXT"
    path.write_text("content", encoding="utf-8")

    document = load_document(path)

    assert document.metadata.file_type == DocumentType.TEXT


def test_load_document_rejects_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_document(missing)


def test_load_document_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "notes.docx"
    path.write_text("content", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError):
        load_document(path)


def test_empty_txt_raises_empty_document_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_empty_markdown_raises_empty_document_error(tmp_path: Path) -> None:
    path = tmp_path / "empty.md"
    path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDocumentError):
        load_document(path)


def test_utf8_bom_txt_is_decoded(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes(b"\xef\xbb\xbfhello")

    document = load_document(path)

    assert document.content == "hello"


def test_corrupt_pdf_raises_parsing_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pdf"
    path.write_bytes(b"not a pdf")

    with pytest.raises(DocumentParsingError):
        load_document(path)


def test_pdf_without_extractable_text_raises_empty_document_error(tmp_path: Path) -> None:
    path = _write_blank_pdf(tmp_path)

    with pytest.raises(EmptyDocumentError):
        load_document(path)
