"""
Unified text extraction interface.

Routes file to the appropriate extractor based on file type,
with OCR fallback for scanned PDFs and images.
"""
import logging
from pathlib import Path

from typing import Optional

logger = logging.getLogger(__name__)


def extract_text(file_path: str, file_type: Optional[str] = None) -> str:
    """
    Unified text extraction from any supported file format.

    Supported: pdf, docx, doc, html, htm, txt, png, jpg, jpeg, tiff, bmp

    Args:
        file_path: Absolute path to the file.
        file_type: Optional file extension override. If None, detected from path.

    Returns:
        Extracted text as string.

    Raises:
        ValueError: If file type is unsupported.
        FileNotFoundError: If file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = (file_type or path.suffix.lstrip(".")).lower()
    logger.info(f"[extract] Routing '{path.name}' (type={ext})")

    if ext == "pdf":
        from app.extractors.pdf_extractor import extract_pdf
        text = extract_pdf(file_path)
        # OCR fallback for scanned PDFs
        if not text or len(text.strip()) < 50:
            logger.info("[extract] PDF text too short, trying OCR fallback...")
            from app.extractors.ocr_utils import extract_text_ocr
            text = extract_text_ocr(file_path)
        return text

    elif ext == "docx" or ext == "doc":
        from app.extractors.docx_extractor import extract_docx
        return extract_docx(file_path)

    elif ext in ("html", "htm"):
        from app.extractors.html_extractor import extract_html
        return extract_html(file_path)

    elif ext == "txt":
        from app.extractors.txt_extractor import extract_txt
        return extract_txt(file_path)

    elif ext in ("png", "jpg", "jpeg", "tiff", "bmp", "webp"):
        from app.extractors.ocr_utils import extract_text_ocr
        return extract_text_ocr(file_path)

    else:
        raise ValueError(f"Unsupported file type: '.{ext}'")
