"""
Multi-Label Clause Classification Engine (Module 2).
Uses Legal-BERT (nlpaueb/legal-bert-base-uncased) for sequence classification.

Architecture:
  - No zero-shot classification (removed BART-MNLI).
  - Legal-BERT outputs logits for 10 legal taxonomy labels.
  - Sigmoid activation for multi-label classification.
  - Batch inference with truncation guards and OOM handling.
"""

import time
import logging
import torch
import torch.nn.functional as F
from typing import List, Dict, Any

from app.ml.model_loader import get_classifier, get_classifier_device, get_model_version
from app.core.exceptions import ClassificationError
from app.core.config import settings

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Legal Taxonomy (mandatory labels) — maps index → label
# ═══════════════════════════════════════════════════════════════

LEGAL_TAXONOMY: List[str] = [
    "DATA_COLLECTION",
    "DATA_SHARING",
    "USER_RIGHTS",
    "DATA_RETENTION",
    "SECURITY_MEASURES",
    "THIRD_PARTY_TRANSFER",
    "COOKIES_TRACKING",
    "CHILDREN_PRIVACY",
    "COMPLIANCE_REFERENCE",
    "LIABILITY_LIMITATION",
]

LABEL_DISPLAY_NAMES: Dict[str, str] = {
    "DATA_COLLECTION": "Data Collection",
    "DATA_SHARING": "Data Sharing",
    "USER_RIGHTS": "User Rights",
    "DATA_RETENTION": "Data Retention",
    "SECURITY_MEASURES": "Security Measures",
    "THIRD_PARTY_TRANSFER": "Third-Party Transfer",
    "COOKIES_TRACKING": "Cookies & Tracking",
    "CHILDREN_PRIVACY": "Children's Privacy",
    "COMPLIANCE_REFERENCE": "Compliance Reference",
    "LIABILITY_LIMITATION": "Liability Limitation",
}

DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_BATCH_SIZE = 16


# ═══════════════════════════════════════════════════════════════
# Core Classification (Legal-BERT Sequence Classification)
# ═══════════════════════════════════════════════════════════════

def classify_clauses_multi_label(
    clauses: List[Dict[str, Any]],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[Dict[str, Any]]:
    """
    Classify clauses using Legal-BERT multi-label sequence classification.
    Uses sigmoid activation on logits for independent per-label probabilities.
    """
    if not clauses:
        return []

    # MAX_CLAUSES guard
    max_clauses = settings.MAX_CLAUSES
    if len(clauses) > max_clauses:
        logger.warning(
            f"[classifier] Clause count {len(clauses)} exceeds MAX_CLAUSES={max_clauses}. "
            f"Processing first {max_clauses} only."
        )
        clauses = clauses[:max_clauses]

    try:
        model, tokenizer = get_classifier()
        device = get_classifier_device()
    except Exception as e:
        raise ClassificationError(f"Failed to load Legal-BERT classifier: {e}", step="model_load")

    model_version = get_model_version("classifier")
    results = []
    total_start = time.perf_counter()
    max_length = settings.CLASSIFIER_MAX_LENGTH

    logger.info(
        f"[classifier] START Legal-BERT classification: "
        f"{len(clauses)} clauses, threshold={confidence_threshold}, "
        f"batch_size={batch_size}, max_length={max_length}"
    )

    # ── Batch Inference ──────────────────────────────────────
    for batch_start in range(0, len(clauses), batch_size):
        batch = clauses[batch_start:batch_start + batch_size]
        texts = [c["text"] for c in batch]

        try:
            # Tokenize batch
            inputs = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

            # Forward pass (no gradient needed for inference)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # shape: (batch_size, num_labels)

            # Sigmoid for multi-label (each label independent)
            probabilities = torch.sigmoid(logits).cpu().numpy()

            for clause, probs in zip(batch, probabilities):
                scored_labels = []
                for idx, prob in enumerate(probs):
                    if idx < len(LEGAL_TAXONOMY):
                        scored_labels.append({
                            "label": LEGAL_TAXONOMY[idx],
                            "confidence": round(float(prob), 4),
                        })

                scored_labels.sort(key=lambda x: (-x["confidence"], x["label"]))

                filtered = [lbl for lbl in scored_labels if lbl["confidence"] >= confidence_threshold]
                if not filtered and scored_labels:
                    filtered = [scored_labels[0]]  # Always return at least the top label

                primary = filtered[0] if filtered else {"label": "UNKNOWN", "confidence": 0.0}

                results.append({
                    "clause_index": clause["index"],
                    "clause_text": clause["text"],
                    "labels": filtered,
                    "all_labels": scored_labels,
                    "primary_label": primary["label"],
                    "primary_confidence": primary["confidence"],
                    "model_version": model_version,
                })

        except RuntimeError as e:
            # OOM handling — reduce batch size and retry
            if "out of memory" in str(e).lower() and batch_size > 1:
                logger.warning(f"[classifier] OOM at batch_size={batch_size}, retrying batch_size=1")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                for single_clause in batch:
                    try:
                        inputs = tokenizer(
                            single_clause["text"],
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors="pt",
                        ).to(device)
                        with torch.no_grad():
                            logits = model(**inputs).logits
                        probs = torch.sigmoid(logits).cpu().numpy()[0]
                        scored = [{"label": LEGAL_TAXONOMY[i], "confidence": round(float(p), 4)}
                                  for i, p in enumerate(probs) if i < len(LEGAL_TAXONOMY)]
                        scored.sort(key=lambda x: (-x["confidence"], x["label"]))
                        filtered = [l for l in scored if l["confidence"] >= confidence_threshold]
                        if not filtered:
                            filtered = [scored[0]]
                        primary = filtered[0]
                        results.append({
                            "clause_index": single_clause["index"],
                            "clause_text": single_clause["text"],
                            "labels": filtered, "all_labels": scored,
                            "primary_label": primary["label"],
                            "primary_confidence": primary["confidence"],
                            "model_version": model_version,
                        })
                    except Exception as inner_e:
                        results.append({
                            "clause_index": single_clause["index"],
                            "clause_text": single_clause["text"],
                            "labels": [], "all_labels": [],
                            "primary_label": "UNKNOWN", "primary_confidence": 0.0,
                            "model_version": model_version, "error": str(inner_e),
                        })
            else:
                logger.error(f"[classifier] Batch failed (start={batch_start}): {e}", exc_info=True)
                for clause in batch:
                    results.append({
                        "clause_index": clause["index"], "clause_text": clause["text"],
                        "labels": [], "all_labels": [], "primary_label": "UNKNOWN",
                        "primary_confidence": 0.0, "model_version": model_version, "error": str(e),
                    })

        except Exception as e:
            logger.error(f"[classifier] Batch failed (start={batch_start}): {e}", exc_info=True)
            for clause in batch:
                results.append({
                    "clause_index": clause["index"], "clause_text": clause["text"],
                    "labels": [], "all_labels": [], "primary_label": "UNKNOWN",
                    "primary_confidence": 0.0, "model_version": model_version, "error": str(e),
                })

    # Clear GPU cache after inference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    elapsed = int((time.perf_counter() - total_start) * 1000)
    classified_count = sum(1 for r in results if r["primary_label"] != "UNKNOWN")
    avg_confidence = 0.0
    if classified_count > 0:
        avg_confidence = round(
            sum(r["primary_confidence"] for r in results if r["primary_label"] != "UNKNOWN")
            / classified_count, 4,
        )

    logger.info(f"[classifier] ✓ {len(results)} clauses classified ({elapsed}ms)")
    logger.info(f"[classifier] Average confidence: {avg_confidence}")
    logger.info(f"[classifier] COMPLETE — Legal-BERT sequence classification")

    return results


# ═══════════════════════════════════════════════════════════════
# Legacy Single-Label Interface (backward compatibility)
# ═══════════════════════════════════════════════════════════════

def classify_clauses(clauses: List[dict]) -> List[dict]:
    """Single-label classification — wraps multi-label, returns primary only."""
    if not clauses:
        return []

    multi_results = classify_clauses_multi_label(clauses, confidence_threshold=0.3)

    output = []
    for clause, result in zip(clauses, multi_results):
        output.append({
            **clause,
            "category": LABEL_DISPLAY_NAMES.get(
                result["primary_label"],
                result["primary_label"],
            ),
            "confidence": result["primary_confidence"],
        })

    return output
