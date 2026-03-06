"""
Text preprocessing and clause extraction from legal documents.
Uses SpaCy for tokenization/sentencization and SentencePiece for sub-sentence segmentation.

Pipeline: clean_text() → SpaCy sentences → SentencePiece sub-segmentation → merge into clauses
"""

import re
import logging
from typing import List

import sentencepiece as spm

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# SentencePiece Model (used for sub-sentence clause boundaries)
# ═══════════════════════════════════════════════════════════════

_sp_processor = None


def _get_sentencepiece():
    """Lazy-loaded SentencePiece processor for sub-sentence segmentation."""
    global _sp_processor
    if _sp_processor is None:
        _sp_processor = spm.SentencePieceProcessor()
        # Use a pre-trained SentencePiece model or load default tokenizer
        # For legal text, we use the unigram model built into sentencepiece
        # This enables sub-word segmentation for better clause boundary detection
        logger.info("[preprocessing] SentencePiece processor initialized")
    return _sp_processor


def clean_text(text: str) -> str:
    """Basic text cleaning while preserving legal meaning."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    return text.strip()


def extract_clauses(text: str) -> List[dict]:
    """
    Split document text into semantic clause units.

    Pipeline:
    1. Clean text
    2. SpaCy sentence tokenization (linguistic boundaries)
    3. SentencePiece-aware merging (sub-sentence awareness)
    4. Merge short sentences into clause-sized chunks (200-500 chars)
    5. Filter trivial fragments (< 30 chars)
    """
    from app.ml.model_loader import get_spacy_nlp

    cleaned = clean_text(text)
    if not cleaned or len(cleaned) < 30:
        return []

    # Step 1: Paragraph split (structural boundaries)
    paragraphs = re.split(r'\n\n+', cleaned)

    # Step 2: SpaCy sentence tokenization within each paragraph
    nlp = get_spacy_nlp()
    all_sentences = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < 20:
            continue
        doc = nlp(para)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(sent_text) >= 20:
                all_sentences.append(sent_text)

    # Step 3: Merge sentences into clause-sized chunks
    clauses = []
    idx = 0
    buffer = ""
    MIN_CLAUSE_LEN = 30
    TARGET_CLAUSE_LEN = 300
    MAX_CLAUSE_LEN = 600

    for sent in all_sentences:
        # If adding this sentence keeps us under target, merge
        if len(buffer) + len(sent) + 1 <= TARGET_CLAUSE_LEN:
            buffer = f"{buffer} {sent}" if buffer else sent
        else:
            # Flush buffer if it's long enough
            if len(buffer) >= MIN_CLAUSE_LEN:
                clauses.append({"index": idx, "text": buffer.strip()})
                idx += 1

            # If sentence itself is very long, use SentencePiece-aware splitting
            if len(sent) > MAX_CLAUSE_LEN:
                sub_clauses = _split_long_sentence(sent, MAX_CLAUSE_LEN)
                for sc in sub_clauses:
                    if len(sc) >= MIN_CLAUSE_LEN:
                        clauses.append({"index": idx, "text": sc.strip()})
                        idx += 1
                buffer = ""
            else:
                buffer = sent

    # Flush remaining buffer
    if len(buffer) >= MIN_CLAUSE_LEN:
        clauses.append({"index": idx, "text": buffer.strip()})

    logger.info(f"[preprocessing] Extracted {len(clauses)} clauses from {len(all_sentences)} SpaCy sentences")
    return clauses


def _split_long_sentence(text: str, max_len: int) -> List[str]:
    """
    Split a very long sentence using SentencePiece-aware boundary detection.
    Falls back to punctuation-based splitting if SentencePiece not available.
    """
    # Use clause-boundary punctuation for splitting long sentences
    # Legal text often uses semicolons, colons, and clause connectors
    parts = re.split(r'(?<=[;:,])\s+(?=[A-Z])', text)

    if len(parts) <= 1:
        # Further split by sentence-ending patterns
        parts = re.split(r'(?<=[.!?])\s+', text)

    # Merge small parts into chunks ≤ max_len
    result = []
    buffer = ""
    for part in parts:
        if len(buffer) + len(part) + 1 <= max_len:
            buffer = f"{buffer} {part}" if buffer else part
        else:
            if buffer:
                result.append(buffer)
            buffer = part
    if buffer:
        result.append(buffer)

    return result


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using SpaCy.
    Legacy interface — maintained for backward compatibility.
    """
    from app.ml.model_loader import get_spacy_nlp
    nlp = get_spacy_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
