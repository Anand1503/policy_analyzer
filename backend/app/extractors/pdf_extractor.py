"""
PDF text extraction using PyMuPDF with multiple fallback strategies.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import io

logger = logging.getLogger(__name__)

# Configure Tesseract-OCR binary path (Windows default install location)
import os
TESSERACT_CMD = os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_pdf(file_path: str, document_id: str) -> Dict[str, Any]:
    """
    Extract text from PDF using multiple strategies:
      1. Standard PyMuPDF text extraction
      2. Block-level extraction (catches some edge cases)
      3. HTML-based extraction (gets text from rendered content)
      4. pytesseract OCR on rendered page images (if available)
    """
    import fitz  # PyMuPDF

    logger.info(f"Extracting PDF: {file_path}")

    try:
        doc = fitz.open(file_path)
        num_pages = len(doc)
        pages_data = []
        full_text_parts = []
        extraction_method = "pymupdf"

        # ── Strategy 1: Standard text extraction ──────────────
        for page_num in range(num_pages):
            page = doc[page_num]
            page_text = page.get_text()
            pages_data.append({
                "page_no": page_num + 1,
                "text": page_text,
                "char_count": len(page_text),
            })
            full_text_parts.append(page_text)

        full_text = "\n\n".join(full_text_parts)
        total_chars = len(full_text.strip())

        # ── Strategy 2: Block-level extraction ────────────────
        if total_chars < 20:
            logger.info("Strategy 1 yielded little text, trying block-level extraction...")
            block_parts = []
            for page_num in range(num_pages):
                page = doc[page_num]
                blocks = page.get_text("blocks")
                for block in blocks:
                    if block[6] == 0:  # text block (not image)
                        block_parts.append(str(block[4]))
            block_text = "\n".join(block_parts).strip()
            if len(block_text) > total_chars:
                full_text = block_text
                total_chars = len(full_text)
                extraction_method = "pymupdf_blocks"
                logger.info(f"Block extraction got {total_chars} chars")

        # ── Strategy 3: HTML extraction ───────────────────────
        if total_chars < 20:
            logger.info("Trying HTML-based extraction...")
            import re
            html_parts = []
            for page_num in range(num_pages):
                page = doc[page_num]
                html = page.get_text("html")
                # Strip HTML tags to get raw text
                clean = re.sub(r"<[^>]+>", " ", html)
                clean = re.sub(r"\s+", " ", clean).strip()
                html_parts.append(clean)
            html_text = "\n\n".join(html_parts).strip()
            if len(html_text) > total_chars:
                full_text = html_text
                total_chars = len(full_text)
                extraction_method = "pymupdf_html"
                logger.info(f"HTML extraction got {total_chars} chars")

        # ── Strategy 4: OCR via pytesseract (if installed) ────
        if total_chars < 20:
            try:
                import pytesseract
                from PIL import Image

                # Point pytesseract to the installed binary
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

                logger.info(f"PDF is scanned. Running pytesseract OCR on page images (tesseract={TESSERACT_CMD})...")
                ocr_parts = []
                for page_num in range(num_pages):
                    page = doc[page_num]
                    # Render page at 2x zoom for better OCR quality
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    if img.mode != "L":
                        img = img.convert("L")
                    page_ocr = pytesseract.image_to_string(img, lang="eng")
                    ocr_parts.append(page_ocr.strip())

                ocr_text = "\n\n".join(ocr_parts).strip()
                if len(ocr_text) > total_chars:
                    full_text = ocr_text
                    total_chars = len(full_text)
                    extraction_method = "pytesseract_ocr"
                    # Rebuild pages_data from OCR
                    pages_data = [
                        {"page_no": i + 1, "text": ocr_parts[i], "char_count": len(ocr_parts[i])}
                        for i in range(len(ocr_parts))
                    ]
                    logger.info(f"OCR extraction got {total_chars} chars")
            except ImportError:
                logger.warning(
                    "pytesseract not installed — cannot OCR scanned PDF. "
                    "Install with: pip install pytesseract  "
                    "Also install Tesseract-OCR binary: https://github.com/tesseract-ocr/tesseract"
                )
            except Exception as e:
                logger.warning(f"OCR attempt failed: {e}")

        doc.close()

        # Rebuild pages_data if we used a different strategy
        if extraction_method not in ("pymupdf", "pytesseract_ocr"):
            pages_data = [{"page_no": 1, "text": full_text, "char_count": total_chars}]

        result = {
            "document_id": document_id,
            "filename": Path(file_path).name,
            "extracted_at": datetime.utcnow().isoformat(),
            "text": full_text.strip(),
            "pages": pages_data,
            "metadata": {
                "num_pages": num_pages,
                "is_scanned": total_chars < 20,
                "language": "en",
                "mime_type": "application/pdf",
                "extraction_method": extraction_method,
                "char_count": len(full_text.strip()),
            },
        }

        logger.info(
            f"PDF extraction complete: {num_pages} pages, "
            f"{len(full_text.strip())} chars, method={extraction_method}"
        )
        return result

    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract PDF: {str(e)}")
