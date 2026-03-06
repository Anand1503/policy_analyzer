"""
OCR utilities using Tika and pytesseract for scanned document processing.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Configuration
USE_OCR = os.getenv("USE_OCR", "true").lower() == "true"
OCR_LANG = os.getenv("OCR_LANG", "eng")
TIKA_SERVER_URL = os.getenv("TIKA_SERVER_URL", "http://localhost:9998")


def is_ocr_available() -> bool:
    """Check if OCR is enabled and available."""
    return USE_OCR


def ocr_pdf_with_tika(file_path: str) -> Optional[str]:
    """
    Extract text from PDF using Apache Tika (includes OCR).
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text or None if extraction fails
    """
    if not USE_OCR:
        logger.warning("OCR is disabled in configuration")
        return None
    
    try:
        from tika import parser
        
        logger.info(f"Running Tika OCR on {file_path}")
        
        # Tika automatically detects if OCR is needed
        parsed = parser.from_file(file_path, serverEndpoint=TIKA_SERVER_URL)
        
        if parsed and 'content' in parsed:
            text = parsed['content']
            if text:
                text = text.strip()
                logger.info(f"Tika OCR extracted {len(text)} characters")
                return text
        
        logger.warning("Tika returned no content")
        return None
        
    except ImportError:
        logger.error("Tika library not installed. Run: pip install tika")
        return None
    except Exception as e:
        logger.error(f"Tika OCR failed: {str(e)}")
        return None


def ocr_image_with_tesseract(image_path: str) -> Optional[str]:
    """
    Extract text from image using pytesseract.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text or None if extraction fails
    """
    if not USE_OCR:
        logger.warning("OCR is disabled in configuration")
        return None
    
    try:
        import pytesseract
        from PIL import Image
        
        logger.info(f"Running Tesseract OCR on {image_path}")
        
        # Open and preprocess image
        image = Image.open(image_path)
        
        # Convert to grayscale for better OCR
        if image.mode != 'L':
            image = image.convert('L')
        
        # Run OCR
        text = pytesseract.image_to_string(image, lang=OCR_LANG)
        
        if text:
            text = text.strip()
            logger.info(f"Tesseract OCR extracted {len(text)} characters")
            return text
        
        logger.warning("Tesseract returned no content")
        return None
        
    except ImportError:
        logger.error("pytesseract library not installed. Run: pip install pytesseract")
        return None
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {str(e)}")
        return None


def ocr_pdf_pages_with_tesseract(file_path: str) -> Optional[List[str]]:
    """
    Extract text from each page of PDF using Tesseract OCR.
    Converts PDF pages to images first.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        List of extracted text per page, or None if extraction fails
    """
    if not USE_OCR:
        logger.warning("OCR is disabled in configuration")
        return None
    
    try:
        import pytesseract
        import fitz  # PyMuPDF
        
        logger.info(f"Running Tesseract OCR on PDF pages: {file_path}")
        
        doc = fitz.open(file_path)
        page_texts = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Render page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Run OCR
            text = pytesseract.image_to_string(image, lang=OCR_LANG)
            page_texts.append(text.strip() if text else "")
            
            logger.debug(f"OCR page {page_num + 1}: {len(page_texts[-1])} chars")
        
        doc.close()
        logger.info(f"Tesseract OCR completed for {len(page_texts)} pages")
        return page_texts
        
    except ImportError as e:
        logger.error(f"Required library not installed: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"PDF page OCR failed: {str(e)}")
        return None


def detect_scanned_pdf(file_path: str, threshold_chars_per_page: int = 300) -> bool:
    """
    Heuristic to detect if a PDF is scanned (no embedded text).
    
    Args:
        file_path: Path to PDF file
        threshold_chars_per_page: Minimum characters per page to consider it text-based
        
    Returns:
        True if PDF appears to be scanned
    """
    try:
        import fitz
        
        doc = fitz.open(file_path)
        total_chars = 0
        total_pages = len(doc)
        
        if total_pages == 0:
            return False
        
        # Sample first few pages
        sample_pages = min(3, total_pages)
        
        for page_num in range(sample_pages):
            page = doc[page_num]
            text = page.get_text()
            total_chars += len(text.strip())
        
        doc.close()
        
        avg_chars = total_chars / sample_pages
        is_scanned = avg_chars < threshold_chars_per_page
        
        logger.info(f"PDF scan detection: avg {avg_chars:.0f} chars/page, threshold {threshold_chars_per_page}, scanned={is_scanned}")
        return is_scanned
        
    except Exception as e:
        logger.error(f"Scanned PDF detection failed: {str(e)}")
        return False
