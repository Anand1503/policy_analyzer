"""
Dynamic Risk Scoring Engine (Module 3).

Hybrid risk model — computes risk as:
    Risk = f(Label_Weight, Confidence, Entity_Sensitivity,
             Regulatory_Flags, Position_Importance)

Architecture:
    - Pure computational module: NO database, NO API logic
    - Reads all config from core/risk_config.py
    - Accepts pre-fetched data, returns scored results
    - Vectorized batch computation
    - Deterministic and explainable
"""

import re
import time
import logging
from typing import List, Dict, Any, Optional

from app.core.risk_config import (
    LABEL_RISK_WEIGHTS,
    DEFAULT_LABEL_WEIGHT,
    ENTITY_SENSITIVITY,
    CHILDREN_KEYWORDS,
    HOME_JURISDICTIONS,
    CONFIDENCE_FLOOR,
    CONFIDENCE_SCALE,
    HIGH_RISK_CLAUSE_WEIGHT,
    AGGREGATION_METHOD,
    score_to_level,
    position_weight,
)
from app.core.exceptions import RiskComputationError

logger = logging.getLogger(__name__)

# Compile children patterns once
_CHILDREN_PATTERNS = [re.compile(p, re.IGNORECASE) for p in CHILDREN_KEYWORDS]

# ─── High-Risk Text Patterns (supplementary rule-based) ──────
_RISK_PATTERNS = [
    (r"sell\s+(your|user|personal)\s+data", 0.15, "SELLS_USER_DATA"),
    (r"(indefinite|perpetual|unlimited)\s+(retention|storage)", 0.12, "INDEFINITE_RETENTION"),
    (r"(waive|forfeit|relinquish)\s+.*(right|claim)", 0.12, "WAIVES_RIGHTS"),
    (r"mandatory\s+arbitration", 0.10, "MANDATORY_ARBITRATION"),
    (r"(no|without)\s+(notice|notification|consent)", 0.12, "NO_CONSENT"),
    (r"(irrevocable|non.?exclusive|worldwide)\s+license", 0.10, "BROAD_LICENSE"),
]
_COMPILED_RISK_PATTERNS = [(re.compile(p, re.IGNORECASE), bonus, tag) for p, bonus, tag in _RISK_PATTERNS]


# ═══════════════════════════════════════════════════════════════
# Core Scoring Functions
# ═══════════════════════════════════════════════════════════════

def compute_clause_risks(
    clauses: List[Dict[str, Any]],
    total_clauses: int = 0,
) -> List[Dict[str, Any]]:
    """
    Compute risk scores for a batch of classified clauses.

    Args:
        clauses: List of dicts with keys:
            - clause_id (str): UUID
            - clause_index (int)
            - clause_text (str)
            - category (str): primary classification label
            - confidence (float): classification confidence
            - entities (list): NER entities [{text, label, start, end}]
            - classifications (list): multi-label [{label, confidence_score}]
        total_clauses: Total clause count for position weighting.

    Returns:
        List of dicts:
            - clause_id, risk_score (0-100), risk_level, risk_factors, explanation
    """
    if not clauses:
        return []

    if total_clauses <= 0:
        total_clauses = len(clauses)

    t0 = time.perf_counter()
    results = []

    for clause in clauses:
        try:
            result = _score_single_clause(clause, total_clauses)
            results.append(result)
        except Exception as e:
            logger.warning(
                f"[risk] Scoring failed for clause {clause.get('clause_id', '?')}: {e}"
            )
            results.append({
                "clause_id": clause.get("clause_id", ""),
                "clause_index": clause.get("clause_index", 0),
                "risk_score": 0.0,
                "risk_level": "LOW",
                "risk_factors": [],
                "explanation": f"Risk computation failed: {str(e)}",
            })

    elapsed = int((time.perf_counter() - t0) * 1000)
    logger.info(f"[risk] ✓ {len(results)} clause risks computed ({elapsed}ms)")

    return results


def compute_document_risk(clause_risks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate clause-level risks into a document-level score.

    Uses weighted mean where high-risk clauses count more.
    """
    if not clause_risks:
        return {
            "overall_risk_score": 0.0,
            "risk_level": "LOW",
            "total_high_risk_clauses": 0,
            "risk_distribution": {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0},
        }

    scores = []
    weights = []
    distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "CRITICAL": 0}

    for cr in clause_risks:
        score = cr.get("risk_score", 0)
        level = cr.get("risk_level", "LOW")
        distribution[level] = distribution.get(level, 0) + 1

        # Weight high-risk clauses more heavily
        w = HIGH_RISK_CLAUSE_WEIGHT if level in ("HIGH", "CRITICAL") else 1.0
        scores.append(score)
        weights.append(w)

    # Weighted mean
    total_weight = sum(weights)
    if total_weight > 0:
        overall = sum(s * w for s, w in zip(scores, weights)) / total_weight
    else:
        overall = 0.0

    overall = round(min(100.0, max(0.0, overall)), 2)
    risk_level = score_to_level(overall)

    return {
        "overall_risk_score": overall,
        "risk_level": risk_level,
        "total_high_risk_clauses": distribution.get("HIGH", 0) + distribution.get("CRITICAL", 0),
        "risk_distribution": distribution,
    }


# ═══════════════════════════════════════════════════════════════
# Single Clause Scorer (internal)
# ═══════════════════════════════════════════════════════════════

def _score_single_clause(clause: Dict[str, Any], total_clauses: int) -> Dict[str, Any]:
    """
    Score a single clause using the hybrid model:

    1. Base risk = label_weight × confidence_factor
    2. Entity bonus = sum of entity sensitivity multipliers
    3. Pattern bonus = regex-matched risk patterns
    4. Position factor = clause position in document
    5. Final = clamp(base + entity + pattern) × position_factor × 100
    """
    clause_id = clause.get("clause_id", "")
    clause_index = clause.get("clause_index", 0)
    text = clause.get("clause_text", "")
    category = clause.get("category", "")
    confidence = clause.get("confidence", 0.5)
    entities = clause.get("entities", []) or []
    classifications = clause.get("classifications", []) or []

    risk_factors = []

    # ── 1. Base Risk from Label Weight ───────────────────────
    # Use multi-label: sum weighted contributions from all labels
    if classifications:
        base_risk = 0.0
        for cls in classifications:
            lbl = cls.get("label", "")
            conf = cls.get("confidence_score", 0.5)
            lbl_weight = LABEL_RISK_WEIGHTS.get(lbl, DEFAULT_LABEL_WEIGHT)
            conf_factor = CONFIDENCE_FLOOR + (conf * CONFIDENCE_SCALE)
            base_risk = max(base_risk, lbl_weight * conf_factor)
            if lbl_weight >= 0.7:
                risk_factors.append(lbl)
    else:
        # Fallback to primary category
        lbl_weight = LABEL_RISK_WEIGHTS.get(category, DEFAULT_LABEL_WEIGHT)
        conf_factor = CONFIDENCE_FLOOR + (confidence * CONFIDENCE_SCALE)
        base_risk = lbl_weight * conf_factor
        if lbl_weight >= 0.7:
            risk_factors.append(category)

    # ── 2. Entity Sensitivity Bonus ──────────────────────────
    entity_bonus = 0.0
    entity_labels_seen = set()

    for entity in entities:
        label = entity.get("label", "")
        if label in entity_labels_seen:
            continue
        entity_labels_seen.add(label)

        if label == "MONETARY_VALUE":
            entity_bonus += ENTITY_SENSITIVITY.get("MONETARY_VALUE", 0)
            risk_factors.append("MONETARY_VALUE_PRESENT")

        elif label == "REGULATION":
            entity_bonus += ENTITY_SENSITIVITY.get("REGULATION", 0)
            risk_factors.append("REGULATION_REFERENCE")

        elif label == "JURISDICTION":
            entity_text = entity.get("text", "")
            if entity_text not in HOME_JURISDICTIONS:
                entity_bonus += ENTITY_SENSITIVITY.get("JURISDICTION_FOREIGN", 0)
                risk_factors.append("FOREIGN_JURISDICTION")

        elif label in ENTITY_SENSITIVITY:
            entity_bonus += ENTITY_SENSITIVITY.get(label, 0)

    # Children detection (pattern-based)
    for pattern in _CHILDREN_PATTERNS:
        if pattern.search(text):
            entity_bonus += ENTITY_SENSITIVITY.get("CHILDREN_RELATED", 0)
            risk_factors.append("CHILDREN_RELATED")
            break

    # ── 3. Pattern Bonus (supplementary rule-based) ──────────
    pattern_bonus = 0.0
    for compiled, bonus, tag in _COMPILED_RISK_PATTERNS:
        if compiled.search(text):
            pattern_bonus += bonus
            risk_factors.append(tag)

    # ── 4. Position Factor ───────────────────────────────────
    pos_factor = position_weight(clause_index, total_clauses)

    # ── 5. Final Score (0-100) ───────────────────────────────
    raw_score = (base_risk + entity_bonus + pattern_bonus) * pos_factor
    final_score = round(min(100.0, max(0.0, raw_score * 100)), 2)
    risk_level = score_to_level(final_score)

    # ── Explanation ──────────────────────────────────────────
    explanation = _build_explanation(category, risk_level, risk_factors, final_score)

    # Deduplicate risk factors
    risk_factors = list(dict.fromkeys(risk_factors))

    return {
        "clause_id": clause_id,
        "clause_index": clause_index,
        "risk_score": final_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "explanation": explanation,
        # Debug fields (optional, useful for explainability)
        "_debug": {
            "base_risk": round(base_risk, 4),
            "entity_bonus": round(entity_bonus, 4),
            "pattern_bonus": round(pattern_bonus, 4),
            "position_factor": round(pos_factor, 4),
        },
    }


def _build_explanation(
    category: str, risk_level: str, factors: List[str], score: float
) -> str:
    """Generate a plain-language explanation of the risk assessment."""
    if not factors:
        return f"This clause ({category}) has {risk_level.lower()} risk (score: {score})."

    factor_strs = []
    for f in factors[:3]:  # Top 3 factors
        readable = f.replace("_", " ").lower()
        factor_strs.append(readable)

    factors_text = ", ".join(factor_strs)
    return (
        f"This clause is rated {risk_level} risk (score: {score:.1f}/100). "
        f"Key factors: {factors_text}."
    )


# ═══════════════════════════════════════════════════════════════
# Legacy Interface (backward compatibility with run_pipeline)
# ═══════════════════════════════════════════════════════════════

def score_risks(classified_clauses: List[dict]) -> List[dict]:
    """
    Legacy wrapper used by AnalysisService.run_pipeline().
    Accepts Module 2 format, returns enriched clauses.
    """
    if not classified_clauses:
        return []

    # Adapt to new format
    adapted = []
    for c in classified_clauses:
        adapted.append({
            "clause_id": "",
            "clause_index": c.get("index", 0),
            "clause_text": c.get("text", ""),
            "category": c.get("category", ""),
            "confidence": c.get("confidence", 0.5),
            "entities": c.get("entities", []),
            "classifications": [],
        })

    risks = compute_clause_risks(adapted, total_clauses=len(adapted))

    # Merge back into original format
    output = []
    for clause, risk in zip(classified_clauses, risks):
        level_map = {"LOW": "low", "MEDIUM": "medium", "HIGH": "high", "CRITICAL": "critical"}
        output.append({
            **clause,
            "risk_score": risk["risk_score"] / 100.0,  # Legacy uses 0-1 scale
            "risk_level": level_map.get(risk["risk_level"], "low"),
            "explanation": risk["explanation"],
        })

    return output
