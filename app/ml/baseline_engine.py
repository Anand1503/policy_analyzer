"""
Baseline Comparison Engine — Simple baselines for benchmarking.
Module 7: Provides keyword-only classifier and label-weight-only risk scorer.

Design:
  - Purely computational
  - Deterministic
  - Used as comparison point for hybrid model performance
"""

import logging
import re
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Keyword-Only Baseline Classifier
# ═══════════════════════════════════════════════════════════════

# Keyword → label mapping for baseline
_KEYWORD_LABEL_MAP: Dict[str, List[str]] = {
    "collect": ["DATA_COLLECTION"],
    "gather": ["DATA_COLLECTION"],
    "obtain": ["DATA_COLLECTION"],
    "share": ["DATA_SHARING"],
    "disclose": ["DATA_SHARING"],
    "transfer": ["DATA_SHARING", "THIRD_PARTY_TRANSFER"],
    "third party": ["THIRD_PARTY_TRANSFER", "DATA_SHARING"],
    "third-party": ["THIRD_PARTY_TRANSFER", "DATA_SHARING"],
    "retain": ["DATA_RETENTION"],
    "retention": ["DATA_RETENTION"],
    "store": ["DATA_RETENTION"],
    "delete": ["USER_RIGHTS", "DATA_RETENTION"],
    "erasure": ["USER_RIGHTS"],
    "right to": ["USER_RIGHTS"],
    "access your": ["USER_RIGHTS"],
    "opt out": ["USER_RIGHTS"],
    "opt-out": ["USER_RIGHTS"],
    "cookie": ["COOKIES_TRACKING"],
    "tracking": ["COOKIES_TRACKING"],
    "analytics": ["COOKIES_TRACKING"],
    "encrypt": ["SECURITY_MEASURES"],
    "security": ["SECURITY_MEASURES"],
    "protect": ["SECURITY_MEASURES"],
    "firewall": ["SECURITY_MEASURES"],
    "children": ["CHILDREN_PRIVACY"],
    "minor": ["CHILDREN_PRIVACY"],
    "under 13": ["CHILDREN_PRIVACY"],
    "coppa": ["CHILDREN_PRIVACY", "COMPLIANCE_REFERENCE"],
    "gdpr": ["COMPLIANCE_REFERENCE"],
    "ccpa": ["COMPLIANCE_REFERENCE"],
    "regulation": ["COMPLIANCE_REFERENCE"],
    "liable": ["LIABILITY_LIMITATION"],
    "liability": ["LIABILITY_LIMITATION"],
    "limitation": ["LIABILITY_LIMITATION"],
    "indemnif": ["LIABILITY_LIMITATION"],
    "consent": ["CONSENT"],
    "agree": ["CONSENT"],
    "permission": ["CONSENT"],
}


def keyword_baseline_classify(
    clauses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Classify clauses using keyword matching only.

    Args:
        clauses: [{clause_id, clause_text}]

    Returns:
        [{clause_id, labels, confidences}]
    """
    results = []
    for clause in clauses:
        text = clause.get("clause_text", "").lower()
        clause_id = clause.get("clause_id", "")

        matched_labels = set()
        for keyword, labels in _KEYWORD_LABEL_MAP.items():
            if keyword in text:
                matched_labels.update(labels)

        # Assign uniform confidence
        label_list = sorted(matched_labels)
        conf = 0.6 if label_list else 0.0

        results.append({
            "clause_id": clause_id,
            "labels": label_list if label_list else ["UNKNOWN"],
            "confidences": [conf] * len(label_list) if label_list else [0.0],
        })

    return results


# ═══════════════════════════════════════════════════════════════
# Label-Weight-Only Baseline Risk Scorer
# ═══════════════════════════════════════════════════════════════

# Label → base risk weight
_LABEL_RISK_WEIGHTS: Dict[str, float] = {
    "DATA_COLLECTION": 0.45,
    "DATA_SHARING": 0.65,
    "THIRD_PARTY_TRANSFER": 0.70,
    "DATA_RETENTION": 0.50,
    "USER_RIGHTS": 0.30,
    "SECURITY_MEASURES": 0.35,
    "COOKIES_TRACKING": 0.40,
    "CHILDREN_PRIVACY": 0.75,
    "COMPLIANCE_REFERENCE": 0.25,
    "LIABILITY_LIMITATION": 0.55,
    "CONSENT": 0.35,
    "UNKNOWN": 0.50,
}


def label_weight_baseline_risk(
    clauses: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Score risk using label weights only (no context, entities, or patterns).

    Args:
        clauses: [{clause_id, labels: [str]}]

    Returns:
        [{clause_id, risk_score, risk_level}]
    """
    results = []
    for clause in clauses:
        clause_id = clause.get("clause_id", "")
        labels = clause.get("labels", [])

        if not labels:
            score = 0.3
        else:
            weights = [_LABEL_RISK_WEIGHTS.get(l, 0.5) for l in labels]
            score = max(weights)

        level = (
            "critical" if score >= 0.85 else
            "high" if score >= 0.7 else
            "medium" if score >= 0.4 else "low"
        )

        results.append({
            "clause_id": clause_id,
            "risk_score": round(score, 4),
            "risk_level": level,
        })

    return results


# ═══════════════════════════════════════════════════════════════
# Baseline Comparison
# ═══════════════════════════════════════════════════════════════

def compare_with_baseline(
    hybrid_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Any],
    metric_type: str = "classification",
) -> Dict[str, Any]:
    """
    Compare hybrid model vs baseline metrics.

    Args:
        hybrid_metrics: Metrics from the full hybrid model
        baseline_metrics: Metrics from the baseline
        metric_type: "classification" or "risk"

    Returns:
        Comparison with absolute and percentage improvements
    """
    comparison = {"metric_type": metric_type}

    if metric_type == "classification":
        keys = ["f1_macro", "f1_micro", "precision_macro", "recall_macro", "accuracy"]
    else:
        keys = ["mae", "rmse", "correlation", "level_accuracy"]

    improvements = {}
    for key in keys:
        hybrid_val = hybrid_metrics.get(key, 0.0)
        baseline_val = baseline_metrics.get(key, 0.0)
        delta = hybrid_val - baseline_val

        # For MAE/RMSE, lower is better
        if key in ("mae", "rmse"):
            improvement_pct = ((baseline_val - hybrid_val) / max(abs(baseline_val), 1e-10)) * 100
        else:
            improvement_pct = (delta / max(abs(baseline_val), 1e-10)) * 100

        improvements[key] = {
            "hybrid": round(hybrid_val, 4),
            "baseline": round(baseline_val, 4),
            "delta": round(delta, 4),
            "improvement_pct": round(improvement_pct, 2),
        }

    comparison["improvements"] = improvements

    # Overall verdict
    if metric_type == "classification":
        f1_delta = improvements.get("f1_macro", {}).get("delta", 0)
        comparison["verdict"] = (
            "Hybrid model significantly outperforms baseline"
            if f1_delta > 0.1 else
            "Hybrid model moderately outperforms baseline"
            if f1_delta > 0.03 else
            "Models perform similarly"
            if f1_delta > -0.03 else
            "Baseline outperforms hybrid model"
        )
    else:
        mae_delta = improvements.get("mae", {}).get("delta", 0)
        comparison["verdict"] = (
            "Hybrid risk model significantly outperforms baseline"
            if mae_delta < -0.05 else
            "Hybrid risk model moderately outperforms baseline"
            if mae_delta < -0.01 else
            "Models perform similarly"
            if mae_delta < 0.01 else
            "Baseline outperforms hybrid model"
        )

    return comparison
