"""
Risk Configuration — centralized, configurable risk parameters.

All risk weights, entity multipliers, and level thresholds are defined here.
The risk_scorer.py module reads from this config; no hardcoded values in logic.

Designed for:
- Easy tuning without code changes
- Research-paper friendly parameter tables
- Future ML-based weight learning
"""

from typing import Dict


# ═══════════════════════════════════════════════════════════════
# Label Risk Weights (0.0 – 1.0)
# ═══════════════════════════════════════════════════════════════
# Base risk inherent to each classification label.
# Higher weight = the category is inherently riskier.

LABEL_RISK_WEIGHTS: Dict[str, float] = {
    "DATA_COLLECTION":       0.60,
    "DATA_SHARING":          0.80,
    "USER_RIGHTS":           0.30,
    "DATA_RETENTION":        0.65,
    "SECURITY_MEASURES":     0.40,
    "THIRD_PARTY_TRANSFER":  0.90,
    "COOKIES_TRACKING":      0.50,
    "CHILDREN_PRIVACY":      1.00,
    "COMPLIANCE_REFERENCE":  0.35,
    "LIABILITY_LIMITATION":  0.55,
}

# Display-friendly label names (for reports and UI)
LABEL_DISPLAY_NAMES: Dict[str, str] = {
    "DATA_COLLECTION":       "Data Collection",
    "DATA_SHARING":          "Data Sharing",
    "USER_RIGHTS":           "User Rights",
    "DATA_RETENTION":        "Data Retention",
    "SECURITY_MEASURES":     "Security Measures",
    "THIRD_PARTY_TRANSFER":  "Third-Party Transfer",
    "COOKIES_TRACKING":      "Cookies & Tracking",
    "CHILDREN_PRIVACY":      "Children's Privacy",
    "COMPLIANCE_REFERENCE":  "Compliance Reference",
    "LIABILITY_LIMITATION":  "Liability Limitation",
}

# Default weight for unknown/unmapped labels
DEFAULT_LABEL_WEIGHT: float = 0.50


# ═══════════════════════════════════════════════════════════════
# Entity Sensitivity Multipliers
# ═══════════════════════════════════════════════════════════════
# Additive bonus when a clause contains specific NER entity types.
# Applied on top of the base label weight.

ENTITY_SENSITIVITY: Dict[str, float] = {
    "MONETARY_VALUE":       0.10,
    "REGULATION":           0.20,
    "CHILDREN_RELATED":     0.30,   # Matched from text patterns
    "JURISDICTION_FOREIGN": 0.15,
    "DATE":                 0.05,
    "LEGAL_REFERENCE":      0.10,
    "DATA_TYPE":            0.05,
    "TIME_PERIOD":          0.05,
    "ORGANIZATION":         0.05,
}

# Regex patterns for children-related terms (used as entity detector)
CHILDREN_KEYWORDS = [
    r"\bchild(?:ren)?\b",
    r"\bminor[s]?\b",
    r"\bunder\s+(?:13|16|18)\b",
    r"\bCOPPA\b",
    r"\bparental\s+consent\b",
    r"\bage\s+(?:verification|restriction|limit)\b",
]

# Jurisdictions considered "foreign" (non-US baseline; configurable)
HOME_JURISDICTIONS = {"United States", "US", "USA"}


# ═══════════════════════════════════════════════════════════════
# Risk Level Thresholds (0–100 scale)
# ═══════════════════════════════════════════════════════════════
# Maps numeric risk score to categorical risk level.

RISK_THRESHOLDS = {
    "LOW":      (0, 30),
    "MEDIUM":   (31, 60),
    "HIGH":     (61, 80),
    "CRITICAL": (81, 100),
}


def score_to_level(score: float) -> str:
    """Convert a 0-100 risk score to a categorical level."""
    score = max(0.0, min(100.0, score))
    for level, (low, high) in RISK_THRESHOLDS.items():
        if low <= score <= high:
            return level
    return "CRITICAL"  # Anything above 100 caps to CRITICAL


# ═══════════════════════════════════════════════════════════════
# Confidence Weighting
# ═══════════════════════════════════════════════════════════════
# How much the classification confidence affects the final score.
# Score = base_weight * (CONFIDENCE_FLOOR + confidence * CONFIDENCE_SCALE)

CONFIDENCE_FLOOR: float = 0.3    # Minimum weight even at 0 confidence
CONFIDENCE_SCALE: float = 0.7    # Remaining weight scaled by confidence


# ═══════════════════════════════════════════════════════════════
# Position Importance (clause_index relative to total)
# ═══════════════════════════════════════════════════════════════
# Early clauses (definitions, scope) slightly less risky.
# Middle/late clauses (obligations, liabilities) slightly more.

POSITION_WEIGHT_EARLY: float = 0.90    # First 20% of clauses
POSITION_WEIGHT_MIDDLE: float = 1.00   # Middle 60%
POSITION_WEIGHT_LATE: float = 1.05     # Last 20%


def position_weight(clause_index: int, total_clauses: int) -> float:
    """Compute position-based weight multiplier."""
    if total_clauses <= 0:
        return 1.0
    ratio = clause_index / total_clauses
    if ratio < 0.20:
        return POSITION_WEIGHT_EARLY
    elif ratio > 0.80:
        return POSITION_WEIGHT_LATE
    return POSITION_WEIGHT_MIDDLE


# ═══════════════════════════════════════════════════════════════
# Aggregation Settings
# ═══════════════════════════════════════════════════════════════

# Document-level aggregation method: "weighted_mean" or "percentile_90"
AGGREGATION_METHOD: str = "weighted_mean"

# Weight for high-risk clauses in weighted aggregation
HIGH_RISK_CLAUSE_WEIGHT: float = 1.5


# ═══════════════════════════════════════════════════════════════
# Config Snapshot (Category 8: Research Reproducibility)
# ═══════════════════════════════════════════════════════════════

def get_config_snapshot() -> dict:
    """
    Capture all risk configuration parameters at analysis time.
    Stored in documents.config_snapshot for reproducibility.
    """
    return {
        "label_risk_weights": LABEL_RISK_WEIGHTS,
        "default_label_weight": DEFAULT_LABEL_WEIGHT,
        "entity_sensitivity": ENTITY_SENSITIVITY,
        "risk_thresholds": RISK_THRESHOLDS,
        "confidence_floor": CONFIDENCE_FLOOR,
        "confidence_scale": CONFIDENCE_SCALE,
        "position_weight_early": POSITION_WEIGHT_EARLY,
        "position_weight_middle": POSITION_WEIGHT_MIDDLE,
        "position_weight_late": POSITION_WEIGHT_LATE,
        "aggregation_method": AGGREGATION_METHOD,
        "high_risk_clause_weight": HIGH_RISK_CLAUSE_WEIGHT,
        "snapshot_version": "1.0",
    }
