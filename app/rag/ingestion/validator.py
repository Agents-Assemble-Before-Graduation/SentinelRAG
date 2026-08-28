"""Document upload validation and safe file path handling."""

import os
import re

from app.core.exceptions import ValidationError

# Allowed file extensions and their canonical source types
ALLOWED_EXTENSIONS = {
    ".pdf": "pdf",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "text",
    ".docx": "docx",
}

# Magic byte signatures for basic MIME validation
MAGIC_BYTES = {
    "pdf": b"%PDF-",
    "docx": b"PK\x03\x04",  # Zip file format used by docx
}


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent directory traversal and filesystem attacks."""
    if not filename:
        return "unnamed_document.txt"

    # Strip directory components (both Unix and Windows style)
    base_name = os.path.basename(filename.replace("\\", "/"))

    # Remove null bytes, control characters, and leading/trailing whitespace/dots
    cleaned = re.sub(r"[\x00-\x1f\x7f]", "", base_name).strip(" .")

    # If stripped completely, provide safe fallback
    if not cleaned:
        return "unnamed_document.txt"

    # Keep only safe alphanumeric, dash, underscore, dot, space, and parentheses
    safe_chars = re.sub(r"[^\w\.\-\s\(\)]", "_", cleaned)
    return safe_chars[:255]


def is_allowed_extension(filename: str) -> bool:
    """Check if the filename has a supported extension."""
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def detect_source_type(filename: str) -> str:
    """Determine canonical source type based on extension."""
    _, ext = os.path.splitext(filename.lower())
    return ALLOWED_EXTENSIONS.get(ext, "text")


def validate_document_file(
    filename: str,
    content: bytes,
    max_size_bytes: int = 50 * 1024 * 1024,
) -> tuple[str, str]:
    """Validate document content, size, and extension.

    Returns:
        Tuple of (sanitized_filename, source_type)

    Raises:
        ValidationError: If file violates security, size, or format requirements.
    """
    if not content:
        raise ValidationError("File content is empty.")

    if len(content) > max_size_bytes:
        max_mb = max_size_bytes / (1024 * 1024)
        raise ValidationError(f"File size exceeds maximum allowed limit ({max_mb:.1f} MB).")

    clean_name = sanitize_filename(filename)

    if not is_allowed_extension(clean_name):
        allowed_str = ", ".join(ALLOWED_EXTENSIONS.keys())
        raise ValidationError(
            f"Unsupported file extension for '{clean_name}'. Supported types are: {allowed_str}"
        )

    source_type = detect_source_type(clean_name)

    # Magic byte verification for binary formats
    if source_type == "pdf" and not content.startswith(MAGIC_BYTES["pdf"]):
        raise ValidationError("Invalid PDF format: file does not start with valid PDF magic bytes.")

    if source_type == "docx" and not content.startswith(MAGIC_BYTES["docx"]):
        raise ValidationError("Invalid DOCX format: file does not match Office Open XML structure.")

    return clean_name, source_type


def validate_no_executable_content(content: bytes, filename: str) -> None:
    """Reject content that appears to be an executable script.

    Checks for Unix/Windows shebang lines and known script signatures.
    This is a defense-in-depth measure in addition to extension and magic-byte checks.

    Args:
        content: Raw file bytes.
        filename: Sanitized filename for error messages.

    Raises:
        ValidationError: If executable or script content is detected.
    """
    # Unix/Linux/macOS shebang
    if content.startswith(b"#!"):
        raise ValidationError(
            f"Rejected '{filename}': file appears to be an executable script (shebang detected). "
            "Executable content is not permitted."
        )

    # Windows batch / PowerShell indicators
    content_start = content[:512].lower()
    disallowed_starts = (
        b"@echo off",
        b"set-executionpolicy",
        b"powershell",
        b"<script",
        b"<?php",
        b"import os;",
        b"import subprocess",
    )
    for sig in disallowed_starts:
        if sig in content_start:
            raise ValidationError(
                f"Rejected '{filename}': file contains executable or script content "
                f"({sig.decode(errors='replace')!r}). Executable content is not permitted."
            )

