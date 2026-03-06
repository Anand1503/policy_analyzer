"""
PDF text extraction using PyMuPDF with OCR fallback.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

from app.extractors.ocr_utils import (
    detect_scanned_pdf,
    ocr_pdf_with_tika,
    ocr_pdf_pages_with_tesseract,
    is_ocr_available
)

logger = logging.getLogger(__name__)

# Configuration
OCR_THRESHOLD = int(os.getenv("OCR_THRESHOLD_CHARS_PER_PAGE", "300"))


def extract_pdf(file_path: str, document_id: str) -> Dict[str, Any]:
    """
    Extract text from PDF file using PyMuPDF (fitz).
    Falls back to OCR if PDF appears to be scanned.
    
    Args:
        file_path: Path to PDF file
        document_id: Document UUID (as string)
        
    Returns:
        Extraction result dictionary with structure:
        {
            "document_id": str,
            "filename": str,
            "extracted_at": str (ISO timestamp),
            "text": str (full text),
            "pages": [{"page_no": int, "text": str, "char_count": int}],
            "metadata": {
                "num_pages": int,
                "is_scanned": bool,
                "language": str,
                "mime_type": str,
                "extraction_method": str,
                "char_count": int
            }
        }
    """
    import fitz  # PyMuPDF
    
    logger.info(f"Extracting PDF: {file_path}")
    
    try:
        doc = fitz.open(file_path)
        num_pages = len(doc)
        pages_data = []
        full_text_parts = []
        
        # Extract text from each page
        for page_num in range(num_pages):
            page = doc[page_num]
            page_text = page.get_text()
            
            pages_data.append({
                "page_no": page_num + 1,
                "text": page_text,
                "char_count": len(page_text)
            })
            full_text_parts.append(page_text)
        
        doc.close()
        
        # Combine all text
        full_text = "\n\n".join(full_text_parts)
        total_chars = len(full_text.strip())
        
        # Determine if OCR is needed
        is_scanned = detect_scanned_pdf(file_path, OCR_THRESHOLD)
        extraction_method = "pymupdf"
        
        # If scanned and OCR is available, try OCR
        if is_scanned and is_ocr_available() and total_chars < (num_pages * OCR_THRESHOLD):
            logger.warning(f"PDF appears to be scanned ({total_chars} chars for {num_pages} pages). Attempting OCR...")
            
            # Try Tika first (simpler, handles full PDF)
            ocr_text = ocr_pdf_with_tika(file_path)
            
            if ocr_text and len(ocr_text) > total_chars:
                logger.info(f"Tika OCR improved extraction: {len(ocr_text)} chars vs {total_chars} chars")
                full_text = ocr_text
                extraction_method = "tika_ocr"
                
                # Update pages with OCR text (split proportionally)
                # For simplicity, put all OCR text in one block
                pages_data = [{
                    "page_no": 1,
                    "text": ocr_text,
                    "char_count": len(ocr_text)
                }]
            else:
                # Try Tesseract per-page OCR as fallback
                logger.info("Trying Tesseract per-page OCR...")
                ocr_pages = ocr_pdf_pages_with_tesseract(file_path)
                
                if ocr_pages:
                    ocr_total = sum(len(p) for p in ocr_pages)
                    if ocr_total > total_chars:
                        logger.info(f"Tesseract OCR improved extraction: {ocr_total} chars vs {total_chars} chars")
                        pages_data = [
                            {
                                "page_no": i + 1,
                                "text": ocr_pages[i],
                                "char_count": len(ocr_pages[i])
                            }
                            for i in range(len(ocr_pages))
                        ]
                        full_text = "\n\n".join(ocr_pages)
                        extraction_method = "tesseract_ocr"
        
        # Build result
        result = {
            "document_id": document_id,
            "filename": Path(file_path).name,
            "extracted_at": datetime.utcnow().isoformat(),
            "text": full_text.strip(),
            "pages": pages_data,
            "metadata": {
                "num_pages": num_pages,
                "is_scanned": is_scanned,
                "language": "en",
                "mime_type": "application/pdf",
                "extraction_method": extraction_method,
                "char_count": len(full_text.strip())
            }
        }
        
        logger.info(f"PDF extraction complete: {num_pages} pages, {len(full_text)} chars, method={extraction_method}")
        return result
        
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract PDF: {str(e)}")
