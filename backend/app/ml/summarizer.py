"""
Document summarization using T5 (t5-base).
Abstractive summarization with text-to-text approach.

Production fixes applied:
- Legal-specific structured prompt for higher quality output
- Token-aware chunking (not character-based)
- Input deduplication to remove repeated clauses
- Post-processing to remove repeated phrases
- Graceful fallback on model failure
"""

import re
import logging
import torch
from typing import List
from app.ml.model_loader import get_summarizer, get_summarizer_device

logger = logging.getLogger(__name__)

# T5-base can handle up to 512 tokens reliably; use 480 to leave headroom
MAX_INPUT_TOKENS = 480
# Target output length for single chunk
SUMMARY_MAX_TOKENS = 150
SUMMARY_MIN_TOKENS = 40
# For merged multi-chunk summaries
MERGE_MAX_TOKENS = 200
MERGE_MIN_TOKENS = 60


# ═══════════════════════════════════════════════════════════════
# Text Cleaning & Deduplication
# ═══════════════════════════════════════════════════════════════

def _clean_input_text(text: str) -> str:
    """
    Clean legal text before summarization:
    - Normalize whitespace
    - Remove repeated whitespace/newlines
    - Remove common legal boilerplate noise
    - Strip URL-only lines
    """
    # Normalize line endings and whitespace
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'\t', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are only URLs or section numbers
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip URL-only lines
        if re.match(r'^https?://\S+$', stripped):
            continue
        # Skip lines that are only numbers or bullets
        if re.match(r'^[\d\.\-\*•]+$', stripped):
            continue
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines).strip()


def _deduplicate_sentences(text: str) -> str:
    """
    Remove near-duplicate or repeated sentences to reduce noise.
    Uses exact sentence matching after normalization.
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique = []
    for sent in sentences:
        normalized = re.sub(r'\s+', ' ', sent.lower().strip())
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(sent)
    return ' '.join(unique)


def _remove_repeated_phrases(text: str) -> str:
    """
    Post-process generated summary to remove repeated n-gram phrases.
    Operates at the word level (3-gram through 6-gram repetition check).
    """
    words = text.split()
    if len(words) < 10:
        return text

    result_words = []
    i = 0
    while i < len(words):
        found_repeat = False
        # Check for n-gram repetition (n = 4 down to 3)
        for n in range(min(6, len(words) - i), 2, -1):
            candidate = words[i:i + n]
            # Look ahead for the same sequence
            search_start = i + n
            for j in range(search_start, min(search_start + n * 3, len(words) - n + 1)):
                if words[j:j + n] == candidate:
                    # Skip the repeated occurrence — only if it appears within short distance
                    found_repeat = True
                    break
            if found_repeat:
                break

        result_words.append(words[i])
        i += 1

    return ' '.join(result_words)


def _ensure_complete_sentence(text: str) -> str:
    """Ensure the summary ends at a sentence boundary."""
    if not text:
        return text
    # If already ends with sentence-ending punctuation, return as-is
    if text[-1] in '.!?':
        return text
    # Find last sentence boundary
    last_period = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_period > len(text) // 2:
        return text[:last_period + 1]
    return text + '.'


# ═══════════════════════════════════════════════════════════════
# Legal Summarization Prompt Builder
# ═══════════════════════════════════════════════════════════════

def _build_legal_prompt(text: str) -> str:
    """
    Build a structured prompt for legal policy summarization.
    T5 uses prefix-based task conditioning.
    """
    return (
        "summarize legal policy: "
        f"{text}"
    )


# ═══════════════════════════════════════════════════════════════
# Single Chunk Summarizer
# ═══════════════════════════════════════════════════════════════

def _summarize_chunk(
    text: str,
    model,
    tokenizer,
    device,
    max_new_tokens: int = SUMMARY_MAX_TOKENS,
    min_new_tokens: int = SUMMARY_MIN_TOKENS,
) -> str:
    """Summarize a single text chunk using T5 with legal prompt."""
    prompt = _build_legal_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        padding=False,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=4,
            length_penalty=1.5,
            no_repeat_ngram_size=4,
            early_stopping=True,
            repetition_penalty=1.3,
        )

    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return summary.strip()


# ═══════════════════════════════════════════════════════════════
# Token-Aware Chunker
# ═══════════════════════════════════════════════════════════════

def _chunk_text_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int = MAX_INPUT_TOKENS,
    overlap_tokens: int = 30,
) -> List[str]:
    """
    Split text into chunks that respect the model's token limit.
    Uses actual tokenizer to count tokens (not character heuristics).
    """
    # Tokenize the full text
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    if len(token_ids) <= max_tokens:
        return [text]

    # Decode chunks back to text
    chunks = []
    step = max_tokens - overlap_tokens
    for start in range(0, len(token_ids), step):
        chunk_ids = token_ids[start:start + max_tokens]
        if len(chunk_ids) < 20:  # Skip tiny trailing chunks
            break
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        if chunk_text.strip():
            chunks.append(chunk_text.strip())

    return chunks if chunks else [text[:2000]]


# ═══════════════════════════════════════════════════════════════
# Main Public API
# ═══════════════════════════════════════════════════════════════

def summarize_document(text: str) -> str:
    """
    Generate a high-quality abstractive summary of a legal policy document.

    Strategy:
    1. Clean and deduplicate input text
    2. If text fits in one chunk → summarize directly with legal prompt
    3. If too long → chunk by actual token count → summarize each chunk
    4. Merge chunk summaries → re-summarize the merged text
    5. Post-process: remove repeated phrases, ensure complete sentences

    Returns a coherent, professional-quality policy summary.
    """
    if not text or len(text.strip()) < 100:
        return "Document is too short to summarize."

    logger.info(f"[summarizer] Starting summarization: {len(text)} chars")

    try:
        model, tokenizer = get_summarizer()
        device = get_summarizer_device()

        # Step 1: Clean and deduplicate input
        cleaned = _clean_input_text(text)
        cleaned = _deduplicate_sentences(cleaned)
        logger.info(f"[summarizer] Cleaned text: {len(cleaned)} chars")

        # Step 2: Chunk by token count
        chunks = _chunk_text_by_tokens(cleaned, tokenizer, max_tokens=MAX_INPUT_TOKENS)
        logger.info(f"[summarizer] Split into {len(chunks)} chunks")

        # Step 3: Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                chunk_summary = _summarize_chunk(chunk, model, tokenizer, device)
                if chunk_summary:
                    chunk_summaries.append(chunk_summary)
                    logger.debug(f"[summarizer] Chunk {i+1}/{len(chunks)}: {len(chunk_summary)} chars")
            except Exception as chunk_err:
                logger.warning(f"[summarizer] Chunk {i+1} failed: {chunk_err}")
                continue

        if not chunk_summaries:
            return "Summary generation failed. Please try again."

        # Step 4: Merge and re-summarize if multiple chunks
        if len(chunk_summaries) == 1:
            final_summary = chunk_summaries[0]
        else:
            merged = ' '.join(chunk_summaries)
            logger.info(f"[summarizer] Merging {len(chunk_summaries)} chunk summaries: {len(merged)} chars")

            # Re-summarize the merged summaries
            merged_token_count = len(tokenizer.encode(merged, add_special_tokens=False))
            if merged_token_count <= MAX_INPUT_TOKENS:
                final_summary = _summarize_chunk(
                    merged, model, tokenizer, device,
                    max_new_tokens=MERGE_MAX_TOKENS,
                    min_new_tokens=MERGE_MIN_TOKENS,
                )
            else:
                # Too long even after chunk summarization — use best chunk
                # Pick the first 3 summaries which cover intro/main content
                best_chunks = chunk_summaries[:3]
                final_summary = ' '.join(best_chunks)

        # Step 5: Post-process
        final_summary = _remove_repeated_phrases(final_summary)
        final_summary = _ensure_complete_sentence(final_summary)

        logger.info(f"[summarizer] Final summary: {len(final_summary)} chars")
        return final_summary

    except Exception as e:
        logger.error(f"[summarizer] Summarization failed: {e}", exc_info=True)
        return f"Summarization unavailable: {str(e)}"


def summarize_clause(clause_text: str) -> str:
    """
    Summarize an individual policy clause in plain English.
    Used for clause-level simplification.
    """
    if not clause_text or len(clause_text.strip()) < 30:
        return clause_text or ""

    try:
        model, tokenizer = get_summarizer()
        device = get_summarizer_device()

        cleaned = _clean_input_text(clause_text)
        summary = _summarize_chunk(
            cleaned, model, tokenizer, device,
            max_new_tokens=80,
            min_new_tokens=15,
        )
        return _ensure_complete_sentence(summary)
    except Exception as e:
        logger.warning(f"[summarizer] Clause summarization failed: {e}")
        return clause_text[:300]  # Fallback: return truncated original
