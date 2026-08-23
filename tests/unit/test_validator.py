"""Unit tests for document file validation and path sanitization."""

import pytest

from app.core.exceptions import ValidationError
from app.rag.ingestion.validator import (
    MAGIC_BYTES,
    detect_source_type,
    is_allowed_extension,
    sanitize_filename,
    validate_document_file,
)


def test_sanitize_filename():
    """Verify filename sanitization prevents path traversal and special characters."""
    assert sanitize_filename("../../etc/passwd") == "passwd"
    assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
    assert sanitize_filename("my document (1).pdf") == "my document (1).pdf"
    assert sanitize_filename("test\x00file.txt") == "testfile.txt"
    assert sanitize_filename("   ...clean_name.md...  ") == "clean_name.md"
    assert sanitize_filename("") == "unnamed_document.txt"


def test_extension_checks():
    """Verify supported extension detection."""
    assert is_allowed_extension("paper.pdf") is True
    assert is_allowed_extension("notes.md") is True
    assert is_allowed_extension("doc.markdown") is True
    assert is_allowed_extension("data.txt") is True
    assert is_allowed_extension("report.docx") is True
    assert is_allowed_extension("script.py") is False
    assert is_allowed_extension("app.exe") is False


def test_detect_source_type():
    """Verify source type mapping."""
    assert detect_source_type("research.pdf") == "pdf"
    assert detect_source_type("README.md") == "markdown"
    assert detect_source_type("notes.txt") == "text"
    assert detect_source_type("report.docx") == "docx"


def test_validate_document_file_success():
    """Verify successful validation for valid content."""
    name, stype = validate_document_file("notes.txt", b"Hello world text content")
    assert name == "notes.txt"
    assert stype == "text"

    pdf_content = MAGIC_BYTES["pdf"] + b" dummy pdf stream"
    pdf_name, pdf_stype = validate_document_file("research.pdf", pdf_content)
    assert pdf_name == "research.pdf"
    assert pdf_stype == "pdf"


def test_validate_document_file_empty():
    """Verify empty files are rejected."""
    with pytest.raises(ValidationError, match="File content is empty"):
        validate_document_file("empty.txt", b"")


def test_validate_document_file_oversize():
    """Verify oversized files are rejected."""
    content = b"x" * 1024
    with pytest.raises(ValidationError, match="exceeds maximum allowed limit"):
        validate_document_file("big.txt", content, max_size_bytes=512)


def test_validate_document_file_invalid_extension():
    """Verify unsupported extensions are rejected."""
    with pytest.raises(ValidationError, match="Unsupported file extension"):
        validate_document_file("payload.sh", b"echo 'attack'")


def test_validate_pdf_magic_bytes():
    """Verify corrupted PDF without magic bytes is rejected."""
    with pytest.raises(ValidationError, match="Invalid PDF format"):
        validate_document_file("corrupt.pdf", b"NOT_A_PDF_STREAM")
