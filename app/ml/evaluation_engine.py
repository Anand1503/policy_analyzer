"""
Evaluation Engine — Classification performance metrics.
Module 7: Computes precision, recall, F1, confusion matrix for labeled test data.

Design:
  - Purely computational (no DB access)
  - Deterministic (no ML randomness)
  - Operates on labeled test datasets
  - Does NOT modify production data
"""

import logging
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Core Evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_classification(
    predictions: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate classification performance against ground truth labels.

    Args:
        predictions: [{clause_id, labels: [str], confidences: [float]}]
        ground_truth: [{clause_id, labels: [str]}]

    Returns:
        Structured metrics dict with micro/macro averages and per-label breakdown.
    """
    # Index ground truth by clause_id
    gt_map = {str(g["clause_id"]): set(g.get("labels", [])) for g in ground_truth}
    pred_map = {str(p["clause_id"]): set(p.get("labels", [])) for p in predictions}

    # Collect all labels
    all_labels: Set[str] = set()
    for labels in gt_map.values():
        all_labels.update(labels)
    for labels in pred_map.values():
        all_labels.update(labels)

    sorted_labels = sorted(all_labels)

    # Per-label TP, FP, FN
    per_label: Dict[str, Dict[str, int]] = {}
    for label in sorted_labels:
        tp = fp = fn = 0
        for clause_id in set(list(gt_map.keys()) + list(pred_map.keys())):
            gt_has = label in gt_map.get(clause_id, set())
            pred_has = label in pred_map.get(clause_id, set())
            if gt_has and pred_has:
                tp += 1
            elif pred_has and not gt_has:
                fp += 1
            elif gt_has and not pred_has:
                fn += 1
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn}

    # Per-label metrics
    per_label_metrics = {}
    for label in sorted_labels:
        counts = per_label[label]
        p, r, f1 = _compute_prf(counts["tp"], counts["fp"], counts["fn"])
        per_label_metrics[label] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "support": counts["tp"] + counts["fn"],
            "true_positives": counts["tp"],
            "false_positives": counts["fp"],
            "false_negatives": counts["fn"],
        }

    # Micro averages (global TP/FP/FN)
    total_tp = sum(c["tp"] for c in per_label.values())
    total_fp = sum(c["fp"] for c in per_label.values())
    total_fn = sum(c["fn"] for c in per_label.values())
    micro_p, micro_r, micro_f1 = _compute_prf(total_tp, total_fp, total_fn)

    # Macro averages (average of per-label)
    all_p = [m["precision"] for m in per_label_metrics.values()]
    all_r = [m["recall"] for m in per_label_metrics.values()]
    all_f1 = [m["f1"] for m in per_label_metrics.values()]
    n = max(len(all_p), 1)

    macro_p = sum(all_p) / n
    macro_r = sum(all_r) / n
    macro_f1 = sum(all_f1) / n

    # Weighted averages (weighted by support)
    total_support = sum(m["support"] for m in per_label_metrics.values()) or 1
    weighted_p = sum(m["precision"] * m["support"] for m in per_label_metrics.values()) / total_support
    weighted_r = sum(m["recall"] * m["support"] for m in per_label_metrics.values()) / total_support
    weighted_f1 = sum(m["f1"] * m["support"] for m in per_label_metrics.values()) / total_support

    # Confusion matrix
    confusion = _build_confusion_matrix(pred_map, gt_map, sorted_labels)

    # Overall accuracy (clause-level exact match)
    exact_matches = sum(
        1 for cid in gt_map
        if pred_map.get(cid, set()) == gt_map[cid]
    )
    accuracy = exact_matches / max(len(gt_map), 1)

    return {
        "precision_micro": round(micro_p, 4),
        "recall_micro": round(micro_r, 4),
        "f1_micro": round(micro_f1, 4),
        "precision_macro": round(macro_p, 4),
        "recall_macro": round(macro_r, 4),
        "f1_macro": round(macro_f1, 4),
        "precision_weighted": round(weighted_p, 4),
        "recall_weighted": round(weighted_r, 4),
        "f1_weighted": round(weighted_f1, 4),
        "accuracy": round(accuracy, 4),
        "total_samples": len(gt_map),
        "total_labels": len(sorted_labels),
        "per_label_metrics": per_label_metrics,
        "confusion_matrix": confusion,
    }


def evaluate_compliance_detection(
    predicted: List[Dict[str, Any]],
    ground_truth: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate compliance detection accuracy.

    Args:
        predicted: [{article_id, status}]  — from compliance engine
        ground_truth: [{article_id, status}]  — labeled truth

    Returns:
        Coverage accuracy, gap detection precision, FP/FN rates
    """
    gt_map = {g["article_id"]: g["status"] for g in ground_truth}
    pred_map = {p["article_id"]: p["status"] for p in predicted}

    all_articles = set(list(gt_map.keys()) + list(pred_map.keys()))

    correct = 0
    tp_gaps = 0  # Correctly detected gaps (missing/partial)
    fp_gaps = 0  # Falsely flagged as gap
    fn_gaps = 0  # Missed gaps

    for article in all_articles:
        gt_status = gt_map.get(article, "unknown")
        pred_status = pred_map.get(article, "unknown")

        if gt_status == pred_status:
            correct += 1

        gt_is_gap = gt_status in ("missing", "partial")
        pred_is_gap = pred_status in ("missing", "partial")

        if gt_is_gap and pred_is_gap:
            tp_gaps += 1
        elif pred_is_gap and not gt_is_gap:
            fp_gaps += 1
        elif gt_is_gap and not pred_is_gap:
            fn_gaps += 1

    coverage_accuracy = correct / max(len(all_articles), 1)
    gap_precision = tp_gaps / max(tp_gaps + fp_gaps, 1)
    gap_recall = tp_gaps / max(tp_gaps + fn_gaps, 1)
    gap_f1 = 2 * gap_precision * gap_recall / max(gap_precision + gap_recall, 1e-10)

    return {
        "coverage_accuracy": round(coverage_accuracy, 4),
        "gap_detection_precision": round(gap_precision, 4),
        "gap_detection_recall": round(gap_recall, 4),
        "gap_detection_f1": round(gap_f1, 4),
        "true_positive_gaps": tp_gaps,
        "false_positive_gaps": fp_gaps,
        "false_negative_gaps": fn_gaps,
        "total_articles_evaluated": len(all_articles),
    }


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _compute_prf(tp: int, fp: int, fn: int) -> tuple:
    """Compute precision, recall, F1 from counts."""
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return precision, recall, f1


def _build_confusion_matrix(
    pred_map: Dict[str, set],
    gt_map: Dict[str, set],
    labels: List[str],
) -> Dict[str, Any]:
    """Build a label-level confusion matrix."""
    matrix = defaultdict(lambda: defaultdict(int))

    for clause_id in gt_map:
        gt_labels = gt_map[clause_id]
        pred_labels = pred_map.get(clause_id, set())

        for true_label in gt_labels:
            if true_label in pred_labels:
                matrix[true_label][true_label] += 1
            else:
                for pred_label in pred_labels:
                    matrix[true_label][pred_label] += 1
                if not pred_labels:
                    matrix[true_label]["_NO_PREDICTION_"] += 1

    # Convert to serializable dict
    return {
        "labels": labels,
        "matrix": {k: dict(v) for k, v in matrix.items()},
    }
