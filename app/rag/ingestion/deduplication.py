"""Content hashing and deduplication utilities."""

import hashlib


def calculate_content_hash(content: bytes | str) -> str:
    """Calculate deterministic SHA-256 hash for document content deduplication."""
    if isinstance(content, str):
        content_bytes = content.strip().encode("utf-8")
    else:
        content_bytes = content

    return hashlib.sha256(content_bytes).hexdigest()
