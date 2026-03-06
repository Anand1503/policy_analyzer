"""
File validation — MIME type and content-based file verification.
Rejects disguised executables, mismatched extensions, and oversized files.
"""

import io
import logging
from typing import Tuple, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Expected MIME types per extension
# ═══════════════════════════════════════════════════════════════

EXTENSION_MIME_MAP = {
    "pdf":  {"application/pdf"},
    "docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
             "application/zip"},  # DOCX is a ZIP container
    "doc":  {"application/msword", "application/octet-stream"},
    "html": {"text/html", "text/plain"},
    "txt":  {"text/plain"},
}

# Binary magic signatures
MAGIC_SIGNATURES = {
    b"%PDF":       "application/pdf",
    b"PK\x03\x04": "application/zip",         # DOCX/XLSX (ZIP container)
    b"\xd0\xcf\x11\xe0": "application/msword", # OLE2 (DOC)
}

# Dangerous: executable/binary signatures to reject
DANGEROUS_SIGNATURES = {
    b"MZ",           # Windows PE executable
    b"\x7fELF",      # Linux ELF executable
    b"\xca\xfe\xba\xbe",  # Java class / Mach-O fat binary
    b"\xfe\xed\xfa",      # Mach-O executable
}


def detect_mime_from_content(content: bytes) -> Optional[str]:
    """Detect MIME type from file magic bytes."""
    header = content[:16]
    for sig, mime in MAGIC_SIGNATURES.items():
        if header.startswith(sig):
            return mime
    # Check if it looks like text/HTML
    text_header = header.decode("utf-8", errors="ignore").strip().lower()
    if text_header.startswith("<!doctype") or text_header.startswith("<html"):
        return "text/html"
    # Try to decode as UTF-8 text
    try:
        content[:1024].decode("utf-8")
        return "text/plain"
    except (UnicodeDecodeError, ValueError):
        return "application/octet-stream"


def is_dangerous_content(content: bytes) -> bool:
    """Check if file content matches known executable/binary signatures."""
    header = content[:8]
    for sig in DANGEROUS_SIGNATURES:
        if header.startswith(sig):
            return True
    return False


def validate_upload_content(
    filename: str,
    content: bytes,
    declared_mime: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Full content-based file validation:
    1. Check extension is allowed
    2. Detect MIME from magic bytes
    3. Reject executables/binaries
    4. Verify extension matches content
    5. Check file size

    Returns:
        (is_valid, error_message)
    """
    # Extract extension
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if ext not in settings.ALLOWED_EXTENSIONS:
        logger.warning(f"[mime] Rejected upload: disallowed extension '.{ext}'")
        return False, f"File type '.{ext}' is not allowed. Allowed: {settings.ALLOWED_EXTENSIONS}"

    # Reject empty files
    if len(content) == 0:
        return False, "Empty file uploaded"

    # Check dangerous content
    if is_dangerous_content(content):
        logger.warning(f"[mime] REJECTED: executable/binary detected in '{filename}'")
        return False, "File appears to be an executable or binary. Upload rejected."

    # Detect actual MIME from content
    detected_mime = detect_mime_from_content(content)

    # Verify extension matches detected content
    expected_mimes = EXTENSION_MIME_MAP.get(ext, set())
    if expected_mimes and detected_mime and detected_mime not in expected_mimes:
        logger.warning(
            f"[mime] REJECTED: MIME mismatch for '{filename}' — "
            f"ext='.{ext}', detected='{detected_mime}', expected={expected_mimes}"
        )
        return False, (
            f"File content does not match extension '.{ext}'. "
            f"Detected content type: {detected_mime}"
        )

    # Size check (redundant with middleware, but defense in depth)
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if len(content) > max_bytes:
        return False, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit"

    logger.info(f"[mime] Validated: '{filename}' → {detected_mime}")
    return True, ""
