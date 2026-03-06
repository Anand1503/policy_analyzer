"""
HTML text extraction using BeautifulSoup.
"""

import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def extract_html(file_path: str, document_id: str) -> Dict[str, Any]:
    """
    Extract text from HTML file using BeautifulSoup.
    Removes scripts, styles, and extracts visible text.
    
    Args:
        file_path: Path to HTML file
        document_id: Document UUID (as string)
        
    Returns:
        Extraction result dictionary
    """
    from bs4 import BeautifulSoup
    
    logger.info(f"Extracting HTML: {file_path}")
    
    try:
        # Read HTML file
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # Parse HTML
        soup = BeautifulSoup(html_content, 'lxml')
        
        # Remove script and style elements
        for script in soup(["script", "style", "meta", "link"]):
            script.decompose()
        
        # Extract text
        text = soup.get_text(separator='\n', strip=True)
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Try to extract structured content
        sections = []
        
        # Extract headings
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            sections.append({
                "type": heading.name,
                "text": heading.get_text(strip=True)
            })
        
        # Extract title
        title = soup.title.string if soup.title else "Untitled"
        
        result = {
            "document_id": document_id,
            "filename": Path(file_path).name,
            "extracted_at": datetime.utcnow().isoformat(),
            "text": text,
            "pages": None,  # HTML doesn't have pages
            "metadata": {
                "num_pages": 1,
                "is_scanned": False,
                "language": "en",
                "mime_type": "text/html",
                "extraction_method": "beautifulsoup",
                "char_count": len(text),
                "title": title,
                "heading_count": len(sections),
                "headings": sections[:20]  # First 20 headings
            }
        }
        
        logger.info(f"HTML extraction complete: {len(text)} chars, {len(sections)} headings")
        return result
        
    except Exception as e:
        logger.error(f"HTML extraction failed: {str(e)}", exc_info=True)
        raise Exception(f"Failed to extract HTML: {str(e)}")
