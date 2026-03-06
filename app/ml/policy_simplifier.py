"""
Policy Simplifier — Template-based clause summarization.
Module 6: Converts technical legal clauses into plain-English summaries.

Design:
  - Purely computational (no DB access)
  - Deterministic (template-based, no ML randomness)
  - Risk-aware emphasis
  - Extractive approach (uses clause text, no hallucination)
  - Batch processing support (max 200 clauses)

Output per clause:
  {
      "clause_id": "...",
      "clause_index": 0,
      "plain_summary": "This clause explains that...",
      "risk_note": "This may increase your exposure due to..."
  }
"""

import logging
import re
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Maximum clause text length (chars) — ~800 tokens
MAX_CLAUSE_LENGTH = 3200
MAX_CLAUSES_PER_BATCH = 200


# ═══════════════════════════════════════════════════════════════
# Category Templates — deterministic, no hallucination
# ═══════════════════════════════════════════════════════════════

CATEGORY_TEMPLATES: Dict[str, Dict[str, str]] = {
    "DATA_COLLECTION": {
        "summary_prefix": "This clause describes how your personal data is collected",
        "risk_high": "Extensive data collection may expose more personal information than necessary.",
        "risk_medium": "Data collection scope should be reviewed to ensure it aligns with stated purposes.",
        "risk_low": "Data collection practices appear standard and limited to stated purposes.",
    },
    "DATA_SHARING": {
        "summary_prefix": "This clause explains how your data may be shared with other parties",
        "risk_high": "Broad data sharing with third parties significantly increases your privacy exposure.",
        "risk_medium": "Data sharing practices exist and should be reviewed for scope and safeguards.",
        "risk_low": "Data sharing appears limited and subject to appropriate controls.",
    },
    "USER_RIGHTS": {
        "summary_prefix": "This clause outlines your rights regarding your personal data",
        "risk_high": "User rights may be limited or difficult to exercise, reducing your control over personal data.",
        "risk_medium": "Some user rights are mentioned but may lack clear exercise procedures.",
        "risk_low": "User rights are clearly stated and appear accessible.",
    },
    "DATA_RETENTION": {
        "summary_prefix": "This clause addresses how long your data is stored",
        "risk_high": "Data retention periods are vague or excessively long, increasing exposure risk.",
        "risk_medium": "Retention practices are mentioned but specific timelines may be unclear.",
        "risk_low": "Data retention periods are clearly defined and reasonable.",
    },
    "SECURITY_MEASURES": {
        "summary_prefix": "This clause describes the security measures protecting your data",
        "risk_high": "Security commitments appear weak or vague, potentially leaving data inadequately protected.",
        "risk_medium": "Security measures are mentioned but lack specific technical details.",
        "risk_low": "Security measures appear robust with specific technical protections described.",
    },
    "THIRD_PARTY_TRANSFER": {
        "summary_prefix": "This clause covers the transfer of your data to third parties",
        "risk_high": "Third-party data transfers are extensive with limited safeguards described.",
        "risk_medium": "Third-party transfers occur and the scope of sharing should be examined.",
        "risk_low": "Third-party transfers appear controlled with appropriate protections.",
    },
    "COOKIES_TRACKING": {
        "summary_prefix": "This clause explains the use of cookies and tracking technologies",
        "risk_high": "Extensive tracking technologies are used, which may monitor your activity beyond what is necessary.",
        "risk_medium": "Tracking technologies are in use; review consent mechanisms.",
        "risk_low": "Cookie usage appears standard with appropriate opt-out options.",
    },
    "CHILDREN_PRIVACY": {
        "summary_prefix": "This clause addresses privacy protections for children",
        "risk_high": "Children's data may not be adequately protected, posing significant regulatory risk.",
        "risk_medium": "Children's privacy is mentioned but specific protections should be verified.",
        "risk_low": "Children's privacy protections appear compliant with applicable regulations.",
    },
    "COMPLIANCE_REFERENCE": {
        "summary_prefix": "This clause references regulatory compliance frameworks",
        "risk_high": "Compliance claims appear vague or unsubstantiated, requiring independent verification.",
        "risk_medium": "Regulatory frameworks are referenced but specific compliance details should be confirmed.",
        "risk_low": "Compliance references appear specific and verifiable.",
    },
    "LIABILITY_LIMITATION": {
        "summary_prefix": "This clause limits the organization's liability",
        "risk_high": "Liability limitations are broad, significantly reducing your ability to seek remedies.",
        "risk_medium": "Some liability limitations exist that may affect your rights.",
        "risk_low": "Liability terms appear balanced and standard.",
    },
}

DEFAULT_TEMPLATE = {
    "summary_prefix": "This clause addresses policy terms",
    "risk_high": "This clause presents elevated risk that warrants careful review.",
    "risk_medium": "This clause contains terms that should be reviewed for potential impact.",
    "risk_low": "This clause appears standard with no significant concerns identified.",
}


# ═══════════════════════════════════════════════════════════════
# Core Simplifier
# ═══════════════════════════════════════════════════════════════

def simplify_clauses(
    clauses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate plain-English summaries for a batch of clauses.

    Args:
        clauses: list of {clause_id, clause_index, clause_text,
                          category, risk_score, risk_level, labels}

    Returns:
        List of {clause_id, clause_index, plain_summary, risk_note}
    """
    if len(clauses) > MAX_CLAUSES_PER_BATCH:
        logger.warning(
            f"[simplifier] Truncating from {len(clauses)} to {MAX_CLAUSES_PER_BATCH} clauses"
        )
        clauses = clauses[:MAX_CLAUSES_PER_BATCH]

    results = []
    for clause in clauses:
        result = _simplify_single(clause)
        results.append(result)

    logger.info(f"[simplifier] Simplified {len(results)} clauses")
    return results


def _simplify_single(clause: Dict[str, Any]) -> Dict[str, Any]:
    """Simplify a single clause using template-based extraction."""

    clause_id = clause.get("clause_id", "")
    clause_index = clause.get("clause_index", 0)
    text = clause.get("clause_text", "")
    category = clause.get("category", "")
    risk_score = clause.get("risk_score", 0.0)
    risk_level = clause.get("risk_level", "low")

    # Truncate clause text
    if len(text) > MAX_CLAUSE_LENGTH:
        text = text[:MAX_CLAUSE_LENGTH] + "..."

    # Get template
    template = CATEGORY_TEMPLATES.get(category, DEFAULT_TEMPLATE)

    # Build plain summary
    plain_summary = _build_summary(template["summary_prefix"], text, category)

    # Build risk note
    risk_note = _build_risk_note(template, risk_score, risk_level)

    return {
        "clause_id": clause_id,
        "clause_index": clause_index,
        "plain_summary": plain_summary,
        "risk_note": risk_note,
    }


def _build_summary(prefix: str, text: str, category: str) -> str:
    """
    Build a plain-language summary (≤ 3 sentences).
    Uses extractive approach: template prefix + key phrase extraction from text.
    """
    # Extract key phrases from clause text
    key_phrases = _extract_key_phrases(text)

    if key_phrases:
        detail = ", ".join(key_phrases[:3])
        summary = f"{prefix}. Specifically, this covers: {detail}."
    else:
        # Fallback: use first sentence of clause text
        first_sentence = _get_first_sentence(text)
        if first_sentence and len(first_sentence) > 20:
            summary = f"{prefix}. In particular: \"{first_sentence}\""
        else:
            summary = f"{prefix}. Review the full clause text for specific details."

    # Cap at 3 sentences
    sentences = summary.split(". ")
    if len(sentences) > 3:
        summary = ". ".join(sentences[:3]) + "."

    return summary


def _build_risk_note(template: Dict[str, str], risk_score: float, risk_level: str) -> str:
    """Build a risk-aware note based on the clause's risk assessment."""
    level = risk_level.lower()
    if level in ("high", "critical") or risk_score >= 0.7:
        note = template.get("risk_high", DEFAULT_TEMPLATE["risk_high"])
        return f"⚠️ High Risk (score: {risk_score:.2f}). {note}"
    elif level == "medium" or risk_score >= 0.4:
        note = template.get("risk_medium", DEFAULT_TEMPLATE["risk_medium"])
        return f"⚡ Moderate Risk (score: {risk_score:.2f}). {note}"
    else:
        note = template.get("risk_low", DEFAULT_TEMPLATE["risk_low"])
        return f"✅ Low Risk (score: {risk_score:.2f}). {note}"


# ═══════════════════════════════════════════════════════════════
# Text Extraction Helpers
# ═══════════════════════════════════════════════════════════════

# Key concept patterns to extract from legal text
_KEY_PATTERNS = [
    r'(?:collect|gather|obtain|receive)\s+(?:your\s+)?(\w[\w\s,]+?)(?:\.|,|;|$)',
    r'(?:share|disclose|transfer|provide)\s+(?:your\s+)?(?:data|information)\s+(?:to|with)\s+(\w[\w\s,]+?)(?:\.|,|;|$)',
    r'(?:retain|store|keep)\s+(?:your\s+)?(?:data|information)\s+(?:for\s+)?(\w[\w\s,]+?)(?:\.|,|;|$)',
    r'(?:right to)\s+(\w[\w\s,]+?)(?:\.|,|;|$)',
    r'(?:cookies?|tracking|analytics)\s+(?:are|is|may be)\s+(?:used\s+)?(?:to|for)\s+(\w[\w\s,]+?)(?:\.|,|;|$)',
    r'(?:encrypt|protect|secure|safeguard)\s+(\w[\w\s,]+?)(?:\.|,|;|$)',
    r'(?:children|minors|under\s+\d+)\s+(\w[\w\s,]+?)(?:\.|,|;|$)',
    r'(?:liable|liability|responsible|damages)\s+(\w[\w\s,]+?)(?:\.|,|;|$)',
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _KEY_PATTERNS]


def _extract_key_phrases(text: str) -> List[str]:
    """Extract key phrases from clause text using pattern matching."""
    phrases = []
    for pattern in _COMPILED_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            phrase = match.strip().rstrip(",;")
            if 3 < len(phrase) < 80:
                phrases.append(phrase)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in phrases:
        p_lower = p.lower()
        if p_lower not in seen:
            seen.add(p_lower)
            unique.append(p)

    return unique[:5]  # Max 5 key phrases


def _get_first_sentence(text: str) -> str:
    """Extract the first meaningful sentence from text."""
    text = text.strip()
    # Match first sentence-ending punctuation
    match = re.match(r'^(.+?[.!?])\s', text)
    if match:
        sentence = match.group(1).strip()
        if len(sentence) > 120:
            return sentence[:120] + "..."
        return sentence
    # No clear sentence boundary — use first 120 chars
    return text[:120].strip() + ("..." if len(text) > 120 else "")
