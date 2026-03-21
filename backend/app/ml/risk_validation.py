"""
Risk Validation — Risk scoring evaluation and ablation study.
Module 7: Compares predicted risk vs ground truth, supports component ablation.

Design:
  - Purely computational (no DB access)
  - Deterministic
  - Supports ablation: remove entity/position/pattern bonuses
"""

import math
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Core Validation
# ═══════════════════════════════════════════════════════════════

def validate_risk_scores(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compare predicted risk scores vs labeled ground truth.

    Args:
        predictions: [{clause_id, risk_score: float, risk_level: str}]
        ground_truth: [{clause_id, risk_score: float, risk_level: str}]

    Returns:
        MAE, RMSE, correlation, level accuracy, per-level breakdown
    """
    gt_map = {str(g["clause_id"]): g for g in ground_truth}
    pred_map = {str(p["clause_id"]): p for p in predictions}

    common_ids = set(gt_map.keys()) & set(pred_map.keys())
    if not common_ids:
        return {"error": "No overlapping clause IDs between predictions and ground truth"}

    gt_scores = [gt_map[cid]["risk_score"] for cid in common_ids]
    pred_scores = [pred_map[cid]["risk_score"] for cid in common_ids]
    n = len(gt_scores)

    # MAE
    mae = sum(abs(g - p) for g, p in zip(gt_scores, pred_scores)) / n

    # RMSE
    mse = sum((g - p) ** 2 for g, p in zip(gt_scores, pred_scores)) / n
    rmse = math.sqrt(mse)

    # Pearson correlation
    correlation = _pearson_correlation(gt_scores, pred_scores)

    # Risk level accuracy
    level_correct = sum(
        1 for cid in common_ids
        if gt_map[cid].get("risk_level", "").lower() == pred_map[cid].get("risk_level", "").lower()
    )
    level_accuracy = level_correct / n

    # Per-level breakdown
    level_breakdown = _compute_level_breakdown(
        [(gt_map[cid], pred_map[cid]) for cid in common_ids]
    )

    # Score distribution
    score_distribution = {
        "gt_mean": round(sum(gt_scores) / n, 4),
        "gt_std": round(_std(gt_scores), 4),
        "pred_mean": round(sum(pred_scores) / n, 4),
        "pred_std": round(_std(pred_scores), 4),
    }

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "correlation": round(correlation, 4),
        "level_accuracy": round(level_accuracy, 4),
        "total_samples": n,
        "level_breakdown": level_breakdown,
        "score_distribution": score_distribution,
    }


# ═══════════════════════════════════════════════════════════════
# Ablation Study
# ═══════════════════════════════════════════════════════════════

def run_ablation_study(
    clauses: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
    full_predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Run ablation study by comparing full model vs simplified variants.

    Tests impact of removing:
      - Entity bonus
      - Position factor
      - Pattern bonus

    Args:
        clauses: [{clause_id, clause_text, category, ...}]
        ground_truth: [{clause_id, risk_score, risk_level}]
        full_predictions: [{clause_id, risk_score, risk_level}] from full model

    Returns:
        Ablation results showing each component's contribution
    """
    gt_map = {str(g["clause_id"]): g for g in ground_truth}

    # Full model baseline
    full_metrics = validate_risk_scores(full_predictions, ground_truth)

    # Ablation variants
    ablation_results = {}

    # 1. Without entity bonus: simulate by reducing scores for clauses with entities
    no_entity = _ablate_component(full_predictions, clauses, "entity")
    no_entity_metrics = validate_risk_scores(no_entity, ground_truth)
    ablation_results["without_entity_bonus"] = {
        "mae": no_entity_metrics["mae"],
        "rmse": no_entity_metrics["rmse"],
        "correlation": no_entity_metrics["correlation"],
        "mae_delta": round(no_entity_metrics["mae"] - full_metrics["mae"], 4),
        "rmse_delta": round(no_entity_metrics["rmse"] - full_metrics["rmse"], 4),
    }

    # 2. Without position factor
    no_position = _ablate_component(full_predictions, clauses, "position")
    no_position_metrics = validate_risk_scores(no_position, ground_truth)
    ablation_results["without_position_factor"] = {
        "mae": no_position_metrics["mae"],
        "rmse": no_position_metrics["rmse"],
        "correlation": no_position_metrics["correlation"],
        "mae_delta": round(no_position_metrics["mae"] - full_metrics["mae"], 4),
        "rmse_delta": round(no_position_metrics["rmse"] - full_metrics["rmse"], 4),
    }

    # 3. Without pattern bonus
    no_pattern = _ablate_component(full_predictions, clauses, "pattern")
    no_pattern_metrics = validate_risk_scores(no_pattern, ground_truth)
    ablation_results["without_pattern_bonus"] = {
        "mae": no_pattern_metrics["mae"],
        "rmse": no_pattern_metrics["rmse"],
        "correlation": no_pattern_metrics["correlation"],
        "mae_delta": round(no_pattern_metrics["mae"] - full_metrics["mae"], 4),
        "rmse_delta": round(no_pattern_metrics["rmse"] - full_metrics["rmse"], 4),
    }

    return {
        "full_model": {
            "mae": full_metrics["mae"],
            "rmse": full_metrics["rmse"],
            "correlation": full_metrics["correlation"],
        },
        "ablation_results": ablation_results,
        "component_importance": _rank_components(ablation_results),
    }


def _ablate_component(
    predictions: List[Dict[str, Any]],
    clauses: List[Dict[str, Any]],
    component: str,
) -> List[Dict[str, Any]]:
    """
    Simulate removing a scoring component by applying a deterministic adjustment.

    The adjustment factors are calibrated to approximate the component's contribution.
    """
    clause_map = {str(c.get("clause_id", "")): c for c in clauses}
    adjusted = []

    for pred in predictions:
        cid = str(pred["clause_id"])
        clause = clause_map.get(cid, {})
        score = pred["risk_score"]

        if component == "entity":
            # Entities typically add 0.05-0.15 bonus
            entities = clause.get("entities", [])
            if entities:
                score = max(0, score - 0.10)
        elif component == "position":
            # Position factor adjusts ±0.05
            idx = clause.get("clause_index", 0)
            total = clause.get("total_clauses", 1)
            if total > 0 and idx < total * 0.2:
                score = max(0, score - 0.05)
        elif component == "pattern":
            # Pattern bonus adds 0.05-0.10
            text = clause.get("clause_text", "")
            if any(kw in text.lower() for kw in ["notwithstanding", "shall not", "irrevocable"]):
                score = max(0, score - 0.08)

        level = (
            "critical" if score >= 0.85 else
            "high" if score >= 0.7 else
            "medium" if score >= 0.4 else "low"
        )
        adjusted.append({
            "clause_id": cid,
            "risk_score": round(score, 4),
            "risk_level": level,
        })

    return adjusted


def _rank_components(ablation_results: Dict) -> List[Dict[str, Any]]:
    """Rank components by their impact (higher MAE delta = more important)."""
    ranked = []
    for component, metrics in ablation_results.items():
        ranked.append({
            "component": component.replace("without_", "").replace("_", " "),
            "mae_impact": abs(metrics["mae_delta"]),
            "rmse_impact": abs(metrics["rmse_delta"]),
        })
    ranked.sort(key=lambda x: x["mae_impact"], reverse=True)
    return ranked


# ═══════════════════════════════════════════════════════════════
# Math Helpers
# ═══════════════════════════════════════════════════════════════

def _pearson_correlation(x: List[float], y: List[float]) -> float:
    """Pearson correlation coefficient."""
    n = len(x)
    if n < 2:
        return 0.0
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x))
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y))
    if std_x * std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def _std(values: List[float]) -> float:
    """Standard deviation."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def _compute_level_breakdown(
    pairs: List[tuple],
) -> Dict[str, Dict[str, int]]:
    """Breakdown of prediction accuracy per risk level."""
    levels = ["low", "medium", "high", "critical"]
    breakdown = {}
    for level in levels:
        gt_count = sum(1 for gt, _ in pairs if gt.get("risk_level", "").lower() == level)
        pred_correct = sum(
            1 for gt, pred in pairs
            if gt.get("risk_level", "").lower() == level
            and pred.get("risk_level", "").lower() == level
        )
        breakdown[level] = {
            "total": gt_count,
            "correct": pred_correct,
            "accuracy": round(pred_correct / max(gt_count, 1), 4),
        }
    return breakdown
