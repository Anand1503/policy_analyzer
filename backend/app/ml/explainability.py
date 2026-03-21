"""
Explainable AI & Risk Justification Engine (Module 4).

Two explainability layers:
  1. Classification Explanation — SHAP token-level attribution + attention maps
  2. Risk Explanation — structured factor decomposition (rule-based)

Architecture:
  - SHAP Explainer wraps Legal-BERT for token importance scores
  - Attention maps extracted from Legal-BERT's attention weights
  - Risk explanations remain deterministic (template-based)
  - All explanations reproducible
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Label Indicator Terms (for fallback + risk analysis)
# ═══════════════════════════════════════════════════════════════

_LABEL_INDICATOR_TERMS: Dict[str, List[str]] = {
    "DATA_COLLECTION": [
        "collect", "gather", "obtain", "record", "receive", "acquire",
        "personal information", "personal data", "usage data", "device information",
    ],
    "DATA_SHARING": [
        "share", "disclose", "provide to", "make available", "distribute",
        "partner", "advertiser", "service provider",
    ],
    "USER_RIGHTS": [
        "right to access", "right to delete", "opt-out", "unsubscribe",
        "withdraw consent", "data portability", "rectification",
    ],
    "DATA_RETENTION": [
        "retain", "store", "keep", "retention period", "delete after",
        "archive", "preserve", "expiration",
    ],
    "SECURITY_MEASURES": [
        "encrypt", "ssl", "tls", "firewall", "secure", "protection",
        "safeguard", "access control", "authentication",
    ],
    "THIRD_PARTY_TRANSFER": [
        "transfer", "transmit", "cross-border", "international",
        "third party", "third-party", "outside", "foreign",
    ],
    "COOKIES_TRACKING": [
        "cookie", "tracking", "pixel", "beacon", "analytics",
        "google analytics", "fingerprint", "session",
    ],
    "CHILDREN_PRIVACY": [
        "child", "children", "minor", "under 13", "COPPA",
        "parental consent", "age verification",
    ],
    "COMPLIANCE_REFERENCE": [
        "GDPR", "CCPA", "HIPAA", "comply", "regulation",
        "applicable law", "jurisdiction", "legal basis",
    ],
    "LIABILITY_LIMITATION": [
        "liability", "limitation", "disclaim", "warranty", "indemnif",
        "arbitration", "waive", "forfeit",
    ],
}

_COMPILED_INDICATORS: Dict[str, List[re.Pattern]] = {
    label: [re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in terms]
    for label, terms in _LABEL_INDICATOR_TERMS.items()
}


# ═══════════════════════════════════════════════════════════════
# SHAP-Based Classification Explanation
# ═══════════════════════════════════════════════════════════════

def explain_classification_shap(
    clause_text: str,
    labels: List[Dict[str, Any]],
    clause_id: str = "",
) -> Dict[str, Any]:
    """
    Generate SHAP token-level attribution for Legal-BERT classification.

    Uses SHAP Partition explainer to determine which tokens most influenced
    each predicted label. Returns reproducible importance scores.

    Args:
        clause_text: The clause text to explain
        labels: List of {"label": str, "confidence_score": float}
        clause_id: Optional clause identifier

    Returns:
        Dict with token attributions and attention map
    """
    import shap
    import torch
    import numpy as np
    from app.ml.model_loader import get_classifier, get_classifier_device
    from app.ml.classifier import LEGAL_TAXONOMY

    try:
        model, tokenizer = get_classifier()
        device = get_classifier_device()

        # Define prediction function for SHAP
        def predict_fn(texts):
            inputs = tokenizer(
                list(texts), padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                logits = model(**inputs).logits
            return torch.sigmoid(logits).cpu().numpy()

        # Create SHAP explainer
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(predict_fn, masker, output_names=LEGAL_TAXONOMY)

        # Compute SHAP values
        shap_values = explainer([clause_text])

        # Extract token-level importance per label
        token_attributions = {}
        for i, label_info in enumerate(labels):
            label = label_info.get("label", "")
            if label in LEGAL_TAXONOMY:
                label_idx = LEGAL_TAXONOMY.index(label)
                values = shap_values.values[0][:, label_idx]
                tokens = shap_values.data[0]
                # Top contributing tokens
                top_indices = np.argsort(np.abs(values))[-10:][::-1]
                token_attributions[label] = [
                    {"token": str(tokens[j]), "importance": round(float(values[j]), 4)}
                    for j in top_indices if abs(values[j]) > 0.001
                ]

        # Extract attention map
        attention_map = _extract_attention_map(clause_text)

        return {
            "clause_id": clause_id,
            "method": "shap",
            "token_attributions": token_attributions,
            "attention_map": attention_map,
            "reproducible": True,
        }

    except Exception as e:
        logger.warning(f"[explainability] SHAP explanation failed, falling back to term-matching: {e}")
        return explain_classification(clause_text, labels, clause_id)


def _extract_attention_map(clause_text: str) -> List[Dict[str, float]]:
    """Extract attention weights from Legal-BERT for the given text."""
    import torch
    from app.ml.model_loader import get_classifier, get_classifier_device

    try:
        model, tokenizer = get_classifier()
        device = get_classifier_device()

        inputs = tokenizer(clause_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)

        # Average attention across all layers and heads
        # Shape: (num_layers, batch, num_heads, seq_len, seq_len)
        attentions = outputs.attentions
        avg_attention = torch.mean(torch.stack(attentions), dim=(0, 1, 2))  # (seq_len, seq_len)
        # CLS token attention to all other tokens
        cls_attention = avg_attention[0].cpu().numpy()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        attention_map = [
            {"token": tok, "attention_score": round(float(score), 4)}
            for tok, score in zip(tokens[1:], cls_attention[1:])  # Skip [CLS]
            if tok not in ("[SEP]", "[PAD]")
        ]

        return sorted(attention_map, key=lambda x: -x["attention_score"])[:15]

    except Exception as e:
        logger.debug(f"[explainability] Attention map extraction failed: {e}")
        return []


# ═══════════════════════════════════════════════════════════════
# Term-Matching Classification Explanation (Fallback)
# ═══════════════════════════════════════════════════════════════

def explain_classification(
    clause_text: str,
    labels: List[Dict[str, Any]],
    clause_id: str = "",
) -> Dict[str, Any]:
    """
    Fallback classification explanation using term-matching.
    Used when SHAP is unavailable or too slow.
    """
    lower_text = clause_text.lower()
    label_explanations = []

    for label_info in labels:
        label = label_info.get("label", "")
        confidence = label_info.get("confidence_score", label_info.get("confidence", 0))

        matched_terms = []
        patterns = _COMPILED_INDICATORS.get(label, [])
        for pattern in patterns:
            if pattern.search(lower_text):
                matched_terms.append(pattern.pattern.replace(r"\b", "").replace("\\b", ""))

        influence_score = min(len(matched_terms) / max(len(patterns), 1), 1.0)

        label_explanations.append({
            "label": label,
            "confidence": round(confidence, 4),
            "matched_terms": matched_terms,
            "term_count": len(matched_terms),
            "influence_score": round(influence_score, 4),
        })

    return {
        "clause_id": clause_id,
        "method": "term_matching",
        "label_explanations": label_explanations,
    }


# ═══════════════════════════════════════════════════════════════
# Risk Factor Decomposition (Rule-Based — unchanged)
# ═══════════════════════════════════════════════════════════════

def explain_risk(
    clause_id: str,
    risk_score: float,
    risk_level: str,
    risk_factors: List[str],
    debug: Optional[Dict[str, float]] = None,
    category: str = "",
    entities: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    Generate structured risk explanation for a single clause.
    Decomposes the risk score into contributing factors.
    """
    entity_types = list({e.get("label", "") for e in (entities or []) if e.get("label")})

    explanation = debug if debug else {
        "base_risk": risk_score / 100 if risk_score > 1 else risk_score,
        "entity_bonus": 0,
        "pattern_bonus": 0,
        "position_factor": 1.0,
    }

    justification = _generate_justification(
        category=category,
        risk_score=risk_score,
        risk_level=risk_level,
        risk_factors=risk_factors,
        entity_types=entity_types,
        explanation=explanation,
    )

    return {
        "clause_id": clause_id,
        "risk_score": round(risk_score, 2),
        "risk_level": risk_level,
        "factor_count": len(risk_factors),
        "factors": risk_factors,
        "entity_types": entity_types,
        "explanation": explanation,
        "justification": justification,
    }


def _generate_justification(
    category: str,
    risk_score: float,
    risk_level: str,
    risk_factors: List[str],
    entity_types: List[str],
    explanation: Dict[str, float],
) -> str:
    """Generate a plain-English justification understandable by non-technical users."""
    parts = []
    cat_display = (category or "policy").replace("_", " ").lower()

    # Opening statement — user-friendly risk description
    level_descriptions = {
        "CRITICAL": f"This clause about {cat_display} raises serious concerns and should be reviewed immediately. It may expose users to significant privacy or legal risks.",
        "HIGH": f"This {cat_display} clause contains language that could put users at a disadvantage. It warrants careful attention before agreeing to these terms.",
        "MEDIUM": f"This {cat_display} clause contains some potentially concerning language. While not immediately alarming, it is worth understanding what it means for your rights.",
        "LOW": f"This {cat_display} clause appears to be standard and poses minimal risk. The language used is generally in line with common policy practices.",
    }
    parts.append(level_descriptions.get(risk_level.upper(), f"This {cat_display} clause should be reviewed."))

    # Explain risk factors in plain language
    if risk_factors:
        friendly_factors = []
        for f in risk_factors[:4]:
            readable = f.replace("_", " ").lower().strip()
            friendly_factors.append(readable)
        parts.append(f"Specifically, it involves: {', '.join(friendly_factors)}.")

    # Mention sensitive data if detected
    if entity_types:
        entity_display = [e.replace("_", " ").lower() for e in entity_types[:4]]
        parts.append(f"It also references sensitive information like {', '.join(entity_display)}.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════
# Batch Processing
# ═══════════════════════════════════════════════════════════════

def generate_explanations_batch(
    clauses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate both classification and risk explanations for a batch of clauses.
    Uses SHAP for classification explanations when possible.
    """
    results = []

    for clause in clauses:
        clause_id = clause.get("clause_id", "")

        # Classification explanation (fast term-matching for batch processing)
        # SHAP is too slow for batch use (~10-30s per clause on CPU).
        # Use explain_classification_shap() for on-demand single-clause deep analysis.
        cls_explanation = explain_classification(
            clause_text=clause.get("clause_text", ""),
            labels=clause.get("classifications", []),
            clause_id=clause_id,
        )

        # Risk explanation (rule-based)
        risk_explanation = explain_risk(
            clause_id=clause_id,
            risk_score=clause.get("risk_score", 0),
            risk_level=clause.get("risk_level", "LOW"),
            risk_factors=clause.get("risk_factors", []),
            debug=clause.get("risk_debug"),
            category=clause.get("category", ""),
            entities=clause.get("entities"),
        )

        results.append({
            "clause_id": clause_id,
            "clause_index": clause.get("clause_index", 0),
            "category": clause.get("category", ""),
            "classification_explanation": cls_explanation,
            "risk_explanation": risk_explanation,
        })

    return results


# ═══════════════════════════════════════════════════════════════
# Legacy Interface (backward compat with run_pipeline)
# ═══════════════════════════════════════════════════════════════

def generate_explanations(scored_clauses: List[dict]) -> List[dict]:
    """
    Legacy wrapper used by AnalysisService.run_pipeline().
    Enriches scored clauses with human-readable explanations.
    """
    explained = []

    for clause in scored_clauses:
        category = clause.get("category", "Unknown")
        risk_level = clause.get("risk_level", "low")
        risk_score = clause.get("risk_score", 0)
        text = clause.get("text", "")

        # Generate justification using the rule-based engine
        justification = _generate_justification(
            category=category,
            risk_score=risk_score * 100 if risk_score <= 1 else risk_score,
            risk_level=risk_level.upper(),
            risk_factors=[],
            entity_types=[],
            explanation={"base_risk": risk_score, "entity_bonus": 0, "pattern_bonus": 0, "position_factor": 1.0},
        )

        # Enhance with text-specific details
        specifics = _extract_text_specifics(text)
        if specifics:
            justification = f"{justification} {specifics}"

        explained.append({
            **clause,
            "explanation": justification,
        })

    return explained


def _extract_text_specifics(text: str) -> str:
    """Extract specific concerning patterns from clause text."""
    specifics = []
    lower = text.lower()

    if any(t in lower for t in ["email", "phone", "address"]):
        specifics.append("Personal contact information referenced.")
    if any(t in lower for t in ["location", "gps"]):
        specifics.append("Location data may be involved.")
    if any(t in lower for t in ["biometric", "fingerprint", "facial"]):
        specifics.append("Biometric data may be collected.")
    if re.search(r"sell|monetize", lower):
        specifics.append("Potential data monetization terms detected.")
    if re.search(r"arbitrat|waive|forfeit", lower):
        specifics.append("Legal rights limitations detected.")
    if re.search(r"without.{0,20}(notice|consent)", lower):
        specifics.append("Changes without user consent may be permitted.")
    if re.search(r"(child|minor|under\s+1[0-3])", lower):
        specifics.append("Policy may involve data from minors.")

    return " ".join(specifics)
