"""Security tests: file upload validation and path traversal prevention.

Verifies that the file validation layer correctly rejects malicious uploads,
oversized files, disallowed extensions, bad magic bytes, and script content.
"""

import pytest
from app.rag.ingestion.validator import (
    sanitize_filename,
    validate_document_file,
    validate_no_executable_content,
)
from app.core.exceptions import ValidationError


# ── sanitize_filename ─────────────────────────────────────────────────────────

class TestSanitizeFilename:
    """Tests for path traversal prevention in sanitize_filename()."""

    def test_strips_unix_path_traversal(self):
        assert sanitize_filename("../../etc/passwd") == "passwd"

    def test_strips_windows_path_traversal(self):
        assert sanitize_filename("..\\..\\windows\\system32\\config") == "config"

    def test_strips_absolute_path_unix(self):
        assert sanitize_filename("/etc/passwd") == "passwd"

    def test_strips_null_bytes(self):
        result = sanitize_filename("doc\x00ument.txt")
        assert "\x00" not in result

    def test_strips_control_characters(self):
        result = sanitize_filename("file\x1fname.txt")
        assert "\x1f" not in result

    def test_empty_filename_returns_fallback(self):
        assert sanitize_filename("") == "unnamed_document.txt"

    def test_dots_only_returns_fallback(self):
        assert sanitize_filename("...") == "unnamed_document.txt"

    def test_normal_filename_preserved(self):
        result = sanitize_filename("my_document.pdf")
        assert result == "my_document.pdf"

    def test_long_filename_truncated_at_255(self):
        long_name = "a" * 300 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255

    def test_mixed_slash_traversal(self):
        """Mixed / and \\ traversal attempt."""
        result = sanitize_filename("/../../secret.txt")
        assert "/" not in result.split(".")[0]
        assert "secret.txt" == result or result.endswith("secret.txt")


# ── validate_document_file ────────────────────────────────────────────────────

class TestValidateDocumentFile:
    """Tests for validate_document_file() — size, extension, magic bytes."""

    def test_valid_markdown_accepted(self):
        content = b"# Hello\nThis is a valid markdown document."
        name, source_type = validate_document_file("report.md", content)
        assert name == "report.md"
        assert source_type == "markdown"

    def test_valid_txt_accepted(self):
        content = b"Plain text content here."
        name, source_type = validate_document_file("notes.txt", content)
        assert source_type == "text"

    def test_valid_pdf_accepted(self):
        content = b"%PDF-1.4 fake pdf content"
        name, source_type = validate_document_file("report.pdf", content)
        assert source_type == "pdf"

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError, match="empty"):
            validate_document_file("file.txt", b"")

    def test_oversized_file_rejected(self):
        big_content = b"x" * (5 * 1024 * 1024 + 1)  # 5MB + 1 byte
        with pytest.raises(ValidationError, match="size"):
            validate_document_file("big.txt", big_content, max_size_bytes=5 * 1024 * 1024)

    def test_disallowed_extension_rejected(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_document_file("malware.exe", b"content here")

    def test_zip_extension_rejected(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_document_file("archive.zip", b"PK\x03\x04content")

    def test_html_extension_rejected(self):
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_document_file("page.html", b"<html>content</html>")

    def test_fake_pdf_wrong_magic_bytes_rejected(self):
        """A .pdf file with non-PDF magic bytes must be rejected."""
        fake_content = b"This is not a PDF file, just text."
        with pytest.raises(ValidationError, match="magic bytes"):
            validate_document_file("fake.pdf", fake_content)

    def test_fake_docx_wrong_magic_bytes_rejected(self):
        """A .docx file that isn't a zip archive must be rejected."""
        fake_content = b"This is not a docx file at all."
        with pytest.raises(ValidationError, match="DOCX"):
            validate_document_file("fake.docx", fake_content)

    def test_path_traversal_in_filename_sanitized(self):
        """Path traversal in filename is sanitized before extension check."""
        content = b"# Safe markdown"
        # ../../etc/passwd doesn't have an allowed extension → rejected
        with pytest.raises(ValidationError, match="Unsupported"):
            validate_document_file("../../etc/passwd", content)

    def test_filename_with_null_byte_sanitized(self):
        """Null bytes in filename are stripped before validation."""
        content = b"Plain text content."
        # After sanitization, the file has no extension → rejected
        with pytest.raises(ValidationError):
            validate_document_file("file\x00.txt.exe", content)


# ── validate_no_executable_content ────────────────────────────────────────────

class TestValidateNoExecutableContent:
    """Tests for validate_no_executable_content() — shebang and script detection."""

    def test_clean_markdown_accepted(self):
        content = b"# Documentation\nThis explains the API."
        validate_no_executable_content(content, "doc.md")  # Must not raise

    def test_unix_shebang_rejected(self):
        content = b"#!/bin/bash\nrm -rf /"
        with pytest.raises(ValidationError, match="shebang"):
            validate_no_executable_content(content, "script.md")

    def test_python_shebang_rejected(self):
        content = b"#!/usr/bin/env python3\nimport os; os.system('rm -rf /')"
        with pytest.raises(ValidationError, match="shebang"):
            validate_no_executable_content(content, "exploit.txt")

    def test_php_script_rejected(self):
        content = b"<?php system('cat /etc/passwd'); ?>"
        with pytest.raises(ValidationError, match="executable"):
            validate_no_executable_content(content, "page.txt")

    def test_html_script_tag_rejected(self):
        content = b"<script>alert('xss')</script>"
        with pytest.raises(ValidationError, match="executable"):
            validate_no_executable_content(content, "page.txt")

    def test_windows_batch_rejected(self):
        content = b"@echo off\ndel /f /q C:\\*"
        with pytest.raises(ValidationError, match="executable"):
            validate_no_executable_content(content, "batch.txt")

    def test_normal_text_with_shebang_in_middle_accepted(self):
        """Shebang is only checked at start of file."""
        content = b"This is documentation.\n#!/bin/bash is a shell interpreter."
        validate_no_executable_content(content, "doc.txt")  # Must not raise
