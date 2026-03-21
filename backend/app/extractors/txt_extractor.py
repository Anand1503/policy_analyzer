"""
Plain text file extraction with encoding detection.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import chardet

logger = logging.getLogger(__name__)


def extract_txt(file_path: str, document_id: str) -> Dict[str, Any]:
    """
    Extract text from TXT file with encoding detection.
    
    Args:
        file_path: Path to TXT file
        document_id: Document UUID (as string)
        
    Returns:
        Extraction result dictionary
    """
    logger.info(f"Extracting TXT: {file_path}")
    
    try:
        # Try UTF-8 first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            encoding_used = 'utf-8'
        except UnicodeDecodeError:
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
            
            detected = chardet.detect(raw_data)
            encoding_used = detected['encoding'] or 'utf-8'
            confidence = detected['confidence']
            
            logger.info(f"Detected encoding: {encoding_used} (confidence: {confidence})")
            
            # Try detected encoding
            try:
                text = raw_data.decode(encoding_used, errors='ignore')
            except:
                # Final fallback to latin-1 (never fails)
                text = raw_data.decode('latin-1', errors='ignore')
                encoding_used = 'latin-1'
        
        # Normalize line endings
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # Count lines
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        result = {
            "document_id": document_id,
            "filename": Path(file_path).name,
            "extracted_at": datetime.utcnow().isoformat(),
            "text": text,
            "pages": None,  # TXT doesn't have pages
            "metadata": {
                "num_pages": 1,
                "is_scanned": False,
                "language": "en",
                "mime_type": "text/plain",
                "extraction_method": "text_read",
                "char_count": len(text),
                "line_count": len(lines),
                "non_empty_line_count": len(non_empty_lines),
                "encoding": encoding_used
            }
        }
        
        logger.info(f"TXT extraction complete: {len(text)} chars, {len(lines)} lines, encoding={encoding_used}")
        return result
        
    except Exception as e:
        logger.error(f"TXT extraction failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract TXT: {str(e)}")
