"""
Utility helpers — shared functions used across the application.
Maps to the 'utils' directory in the target architecture.
"""

import os
import re
import mimetypes
from pathlib import Path
from typing import List, Optional
from app.core.config import settings


# ─── File Validation ─────────────────────────────────────────

ALLOWED_EXTENSIONS = set(settings.ALLOWED_EXTENSIONS)
MAX_SIZE_BYTES = settings.MAX_UPLOAD_MB * 1024 * 1024

MIME_MAP = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "html": "text/html",
    "htm": "text/html",
    "txt": "text/plain",
}


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return get_file_extension(filename) in ALLOWED_EXTENSIONS


def get_file_extension(filename: str) -> str:
    """Extract lowercase extension from filename."""
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def get_file_mime_type(filename: str) -> str:
    """Get MIME type from filename."""
    ext = get_file_extension(filename)
    return MIME_MAP.get(ext, mimetypes.guess_type(filename)[0] or "application/octet-stream")


def validate_file_size(size_bytes: int) -> tuple[bool, str]:
    """Validate file size against configured maximum."""
    if size_bytes > MAX_SIZE_BYTES:
        return False, f"File exceeds maximum size of {settings.MAX_UPLOAD_MB}MB"
    if size_bytes == 0:
        return False, "File is empty"
    return True, ""


def sanitize_filename(filename: str) -> str:
    """Remove potentially dangerous characters from filenames."""
    name = re.sub(r'[^\w\s\-.]', '', filename)
    name = re.sub(r'\s+', '_', name)
    return name[:255]  # Cap length


# ─── Text Utilities ──────────────────────────────────────────

def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


def word_count(text: str) -> int:
    """Count words in text."""
    return len(text.split()) if text else 0


def estimate_reading_time(text: str, wpm: int = 200) -> int:
    """Estimate reading time in minutes."""
    words = word_count(text)
    return max(1, round(words / wpm))


# ─── Format Helpers ──────────────────────────────────────────

def format_file_size(size_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"
