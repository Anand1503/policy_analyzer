"""
Text preprocessing and clause extraction from legal documents.
Uses SpaCy for tokenization/sentencization.

Production optimizations:
- Enhanced text cleaning (removes legal noise, artifacts)
- Token-aware clause merging (targets 300-500 word-token range)
- Near-duplicate clause deduplication
- Improved handling of legal list structures (numbered/bulleted items)
"""

import re
import logging
from typing import List, Set

logger = logging.getLogger(__name__)

# ─── Clause Sizing Parameters ────────────────────────────────
# Target 300-500 tokens per clause (Legal-BERT max = 512)
# Approximate: 1 word ≈ 1.3 tokens for legal text → 230-380 words
MIN_CLAUSE_CHARS = 40         # Minimum chars to be a valid clause
TARGET_CLAUSE_WORDS = 280     # Target merge size (words)
MAX_CLAUSE_WORDS = 450        # Hard max before splitting (words)
SHORT_SENTENCE_WORDS = 15     # Sentences shorter than this are merged aggressively

# ─── Deduplication Threshold ─────────────────────────────────
# Jaccard similarity above this = duplicate clause
DEDUP_SIMILARITY_THRESHOLD = 0.85


def clean_text(text: str) -> str:
    """
    Enhanced text cleaning for legal documents.

    Removes:
    - Carriage returns, excessive whitespace
    - Page numbers and header/footer artifacts
    - Unicode noise characters
    - Excessive repetition of punctuation

    Preserves:
    - Legal structure (section numbers, bullet points)
    - Sentence boundaries
    - Paragraph breaks
    """
    # Normalize line endings
    text = re.sub(r'\r\n|\r', '\n', text)

    # Remove page number artifacts (e.g., "Page 1 of 10", "- 3 -")
    text = re.sub(r'(?im)^\s*[-–—]?\s*page\s+\d+\s*(of\s+\d+)?\s*[-–—]?\s*$', '', text)
    text = re.sub(r'(?m)^\s*\d+\s*$', '', text)  # Lines with only a number

    # Remove common header/footer patterns
    text = re.sub(r'(?im)^(last updated|effective date|policy version|©|copyright):.*$', '', text)

    # Normalize tabs and non-breaking spaces
    text = re.sub(r'[\t\xa0]', ' ', text)

    # Normalize repeated characters (e.g., "---", "===", "...")
    text = re.sub(r'[-=_]{3,}', ' ', text)
    text = re.sub(r'\.{4,}', '...', text)

    # Remove URLs (they add noise to summarization)
    text = re.sub(r'https?://\S+', '[URL]', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Collapse excessive blank lines (keep max 2 for paragraph structure)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Strip leading/trailing whitespace per line
    lines = [line.rstrip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def _word_count(text: str) -> int:
    """Fast word count."""
    return len(text.split())


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two text strings (word-level)."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _is_duplicate(candidate: str, existing: List[str]) -> bool:
    """Check if a clause is a near-duplicate of any existing clause."""
    for existing_clause in existing[-10:]:  # Only check recent clauses for speed
        if _jaccard_similarity(candidate, existing_clause) >= DEDUP_SIMILARITY_THRESHOLD:
            return True
    return False


def _split_long_sentence(text: str, max_words: int) -> List[str]:
    """
    Split a very long sentence at legal clause boundaries.
    Tries: semicolons, colons, comma+conjunction, then hard split by word count.
    """
    # Try splitting at legal clause boundaries (semicolons)
    parts = re.split(r'(?<=;)\s+', text)
    if len(parts) > 1:
        return _merge_splits(parts, max_words)

    # Try colon-based splits (usually introduces a list)
    parts = re.split(r'(?<=:)\s+', text)
    if len(parts) > 1:
        return _merge_splits(parts, max_words)

    # Try comma + coordinating conjunction
    parts = re.split(r',\s+(?=(?:and|or|but|provided|except|unless|if|when)\s)', text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return _merge_splits(parts, max_words)

    # Hard split by word count
    words = text.split()
    result = []
    for i in range(0, len(words), max_words):
        result.append(' '.join(words[i:i + max_words]))
    return result


def _merge_splits(parts: List[str], max_words: int) -> List[str]:
    """Merge small split parts back into chunks up to max_words."""
    result = []
    buffer = ""
    for part in parts:
        if _word_count(buffer) + _word_count(part) <= max_words:
            buffer = f"{buffer} {part}".strip() if buffer else part
        else:
            if buffer:
                result.append(buffer)
            buffer = part
    if buffer:
        result.append(buffer)
    return result


def extract_clauses(text: str) -> List[dict]:
    """
    Split document text into semantic clause units optimized for Legal-BERT.

    Pipeline:
    1. Clean text (enhanced)
    2. Split into paragraphs (structural boundaries)
    3. SpaCy sentence tokenization within each paragraph
    4. Merge short sentences into clause-sized chunks (300-500 tokens / ~230-380 words)
    5. Split oversized clauses
    6. Deduplicate near-identical clauses
    7. Filter trivial fragments

    Returns list of {index, text} dicts.
    """
    from app.ml.model_loader import get_spacy_nlp

    cleaned = clean_text(text)
    if not cleaned or len(cleaned) < MIN_CLAUSE_CHARS:
        logger.warning("[preprocessing] Text too short after cleaning")
        return []

    # Step 1: Split into paragraphs
    paragraphs = re.split(r'\n\n+', cleaned)

    # Step 2: SpaCy sentence tokenization within each paragraph
    nlp = get_spacy_nlp()
    all_sentences = []
    for para in paragraphs:
        para = para.strip()
        if len(para) < MIN_CLAUSE_CHARS:
            continue

        # Handle numbered/bulleted lists as separate items
        if re.match(r'^\s*(\d+[\.\)]\s+|\*\s+|-\s+|•\s+)', para):
            items = re.split(r'\n\s*(?=\d+[\.\)]\s+|\*\s+|-\s+|•\s+)', para)
            for item in items:
                item = item.strip()
                if _word_count(item) >= 5:
                    all_sentences.append(item)
            continue

        # Standard paragraph: use SpaCy for sentence boundaries
        doc = nlp(para)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if _word_count(sent_text) >= 4:
                all_sentences.append(sent_text)

    if not all_sentences:
        logger.warning("[preprocessing] No sentences extracted from text")
        return []

    # Step 3: Merge sentences into optimal clause chunks
    clauses_raw = []
    buffer = ""
    buffer_words = 0

    for sent in all_sentences:
        sent_words = _word_count(sent)
        total_words = buffer_words + sent_words

        if sent_words > MAX_CLAUSE_WORDS:
            # Flush buffer first
            if buffer and buffer_words >= 5:
                clauses_raw.append(buffer.strip())
            # Then split the long sentence
            sub_clauses = _split_long_sentence(sent, MAX_CLAUSE_WORDS)
            clauses_raw.extend(sc.strip() for sc in sub_clauses if _word_count(sc) >= 4)
            buffer = ""
            buffer_words = 0

        elif total_words <= TARGET_CLAUSE_WORDS:
            # Keep merging
            buffer = f"{buffer} {sent}".strip() if buffer else sent
            buffer_words = total_words

        else:
            # Flush buffer if it's substantial enough
            if buffer_words >= 5:
                clauses_raw.append(buffer.strip())
            buffer = sent
            buffer_words = sent_words

    # Flush remaining buffer
    if buffer and buffer_words >= 5:
        clauses_raw.append(buffer.strip())

    # Step 4: Filter by minimum length and deduplicate
    seen_clauses: List[str] = []
    clauses = []
    idx = 0

    for raw in clauses_raw:
        raw = raw.strip()
        if len(raw) < MIN_CLAUSE_CHARS:
            continue
        if _word_count(raw) < 5:
            continue
        # Deduplication check
        if _is_duplicate(raw, seen_clauses):
            logger.debug(f"[preprocessing] Skipping duplicate clause: {raw[:60]}...")
            continue
        seen_clauses.append(raw)
        clauses.append({"index": idx, "text": raw})
        idx += 1

    logger.info(
        f"[preprocessing] Extracted {len(clauses)} clauses "
        f"from {len(all_sentences)} sentences "
        f"(avg {sum(_word_count(c['text']) for c in clauses) // max(len(clauses), 1)} words/clause)"
    )
    return clauses


def _split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using SpaCy.
    Legacy interface — maintained for backward compatibility.
    """
    from app.ml.model_loader import get_spacy_nlp
    nlp = get_spacy_nlp()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]
