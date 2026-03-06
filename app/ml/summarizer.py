"""
Document summarization using T5 (t5-base).
Abstractive summarization with text-to-text approach.

Replaces BART-CNN summarization pipeline.
T5 uses "summarize: " prefix for summarization tasks.
"""

import logging
import torch
from app.ml.model_loader import get_summarizer, get_summarizer_device

logger = logging.getLogger(__name__)

MAX_INPUT_LENGTH = 512  # T5-base token limit


def summarize_document(text: str) -> str:
    """
    Generate an abstractive summary of the full document using T5.

    Strategy:
    - If text fits within model context, summarize directly.
    - If too long, summarize in chunks and merge.
    - Uses T5 "summarize: " task prefix.
    """
    if not text or len(text.strip()) < 100:
        return "Document is too short to summarize."

    try:
        model, tokenizer = get_summarizer()
        device = get_summarizer_device()

        # T5 uses ~4 chars per token approximation
        max_chars = MAX_INPUT_LENGTH * 4  # ~2048 chars

        if len(text) <= max_chars:
            chunks = [text]
        else:
            # Split into overlapping chunks
            chunks = []
            stride = max_chars - 200  # 200-char overlap
            for i in range(0, len(text), stride):
                chunk = text[i:i + max_chars]
                if len(chunk) > 100:
                    chunks.append(chunk)

        # Summarize each chunk
        summaries = []
        for chunk in chunks:
            # T5 requires task prefix
            input_text = f"summarize: {chunk}"

            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
                padding=True,
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=150,
                    min_length=40,
                    num_beams=4,
                    length_penalty=2.0,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            summaries.append(summary)

        # If multiple chunks, merge and re-summarize
        if len(summaries) > 1:
            merged = " ".join(summaries)
            if len(merged) <= max_chars:
                input_text = f"summarize: {merged}"
                inputs = tokenizer(
                    input_text,
                    return_tensors="pt",
                    max_length=MAX_INPUT_LENGTH,
                    truncation=True,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    output_ids = model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=250,
                        min_length=60,
                        num_beams=4,
                        length_penalty=2.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True,
                    )
                return tokenizer.decode(output_ids[0], skip_special_tokens=True)
            else:
                return merged[:2000]  # Safety cap
        else:
            return summaries[0]

    except Exception as e:
        logger.error(f"Summarization failed: {e}", exc_info=True)
        return f"Summarization unavailable: {str(e)}"
