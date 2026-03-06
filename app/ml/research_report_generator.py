"""
Research Report Generator — Automated Markdown report generation.
Module 7: Produces publication-ready evaluation reports.

Design:
  - Purely computational (no DB access)
  - Deterministic output
  - Generates Markdown + graph-ready JSON
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Core Report Generator
# ═══════════════════════════════════════════════════════════════

def generate_research_report(
    dataset_name: str,
    model_version: str,
    classification_metrics: Optional[Dict[str, Any]] = None,
    risk_metrics: Optional[Dict[str, Any]] = None,
    compliance_metrics: Optional[Dict[str, Any]] = None,
    baseline_comparison: Optional[Dict[str, Any]] = None,
    ablation_results: Optional[Dict[str, Any]] = None,
    statistical_tests: Optional[Dict[str, Any]] = None,
    config_snapshot: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a complete research evaluation report.

    Returns:
        {
            "markdown": str,         # Full Markdown report
            "graph_data": dict,      # JSON-ready data for charts
            "summary": str,          # One-line summary
        }
    """
    ts = timestamp or datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    sections = []

    # Header
    sections.append(f"# Evaluation Report: {dataset_name}")
    sections.append("")
    sections.append(f"**Model Version**: `{model_version}`")
    sections.append(f"**Generated**: {ts}")
    sections.append(f"**Dataset**: {dataset_name}")
    sections.append("")

    # System description
    sections.append("## System Overview")
    sections.append("")
    sections.append(
        "The Intelligent Policy Analyzer is a modular legal AI system for privacy "
        "policy analysis. It performs clause-level classification, hybrid risk scoring, "
        "regulatory compliance mapping, and policy simplification using a pipeline of "
        "ML models and deterministic rule engines."
    )
    sections.append("")

    # Dataset summary
    sections.append("## Dataset Summary")
    sections.append("")
    if classification_metrics:
        sections.append(f"- **Total samples**: {classification_metrics.get('total_samples', 'N/A')}")
        sections.append(f"- **Total labels**: {classification_metrics.get('total_labels', 'N/A')}")
    if risk_metrics:
        sections.append(f"- **Risk samples**: {risk_metrics.get('total_samples', 'N/A')}")
    sections.append("")

    # 1. Classification Performance
    if classification_metrics:
        sections.extend(_classification_section(classification_metrics))

    # 2. Risk Scoring Performance
    if risk_metrics:
        sections.extend(_risk_section(risk_metrics))

    # 3. Compliance Performance
    if compliance_metrics:
        sections.extend(_compliance_section(compliance_metrics))

    # 4. Baseline Comparison
    if baseline_comparison:
        sections.extend(_baseline_section(baseline_comparison))

    # 5. Ablation Study
    if ablation_results:
        sections.extend(_ablation_section(ablation_results))

    # 6. Statistical Tests
    if statistical_tests:
        sections.extend(_statistical_section(statistical_tests))

    # Config snapshot
    if config_snapshot:
        sections.append("## Configuration Snapshot")
        sections.append("")
        sections.append("```json")
        import json
        sections.append(json.dumps(config_snapshot, indent=2, default=str))
        sections.append("```")
        sections.append("")

    # Graph-ready data
    graph_data = _build_graph_data(
        classification_metrics, risk_metrics,
        baseline_comparison, ablation_results,
    )

    # Summary
    summary = _build_one_line_summary(
        classification_metrics, risk_metrics, baseline_comparison,
    )

    markdown = "\n".join(sections)

    return {
        "markdown": markdown,
        "graph_data": graph_data,
        "summary": summary,
    }


# ═══════════════════════════════════════════════════════════════
# Section Builders
# ═══════════════════════════════════════════════════════════════

def _classification_section(metrics: Dict) -> List[str]:
    """Build classification performance section."""
    lines = [
        "## Classification Performance",
        "",
        "### Overall Metrics",
        "",
        "| Metric | Micro | Macro | Weighted |",
        "|--------|-------|-------|----------|",
        f"| Precision | {metrics.get('precision_micro', 0):.4f} | {metrics.get('precision_macro', 0):.4f} | {metrics.get('precision_weighted', 0):.4f} |",
        f"| Recall | {metrics.get('recall_micro', 0):.4f} | {metrics.get('recall_macro', 0):.4f} | {metrics.get('recall_weighted', 0):.4f} |",
        f"| F1-Score | {metrics.get('f1_micro', 0):.4f} | {metrics.get('f1_macro', 0):.4f} | {metrics.get('f1_weighted', 0):.4f} |",
        "",
        f"**Exact Match Accuracy**: {metrics.get('accuracy', 0):.4f}",
        "",
    ]

    # Per-label breakdown
    per_label = metrics.get("per_label_metrics", {})
    if per_label:
        lines.append("### Per-Label Breakdown")
        lines.append("")
        lines.append("| Label | Precision | Recall | F1 | Support |")
        lines.append("|-------|-----------|--------|----|---------|")
        for label, m in sorted(per_label.items()):
            lines.append(
                f"| {label} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['f1']:.4f} | {m['support']} |"
            )
        lines.append("")

    return lines


def _risk_section(metrics: Dict) -> List[str]:
    """Build risk scoring section."""
    lines = [
        "## Risk Scoring Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| MAE | {metrics.get('mae', 0):.4f} |",
        f"| RMSE | {metrics.get('rmse', 0):.4f} |",
        f"| Correlation | {metrics.get('correlation', 0):.4f} |",
        f"| Level Accuracy | {metrics.get('level_accuracy', 0):.4f} |",
        "",
    ]

    # Level breakdown
    level_breakdown = metrics.get("level_breakdown", {})
    if level_breakdown:
        lines.append("### Per-Level Accuracy")
        lines.append("")
        lines.append("| Level | Total | Correct | Accuracy |")
        lines.append("|-------|-------|---------|----------|")
        for level in ["low", "medium", "high", "critical"]:
            lb = level_breakdown.get(level, {})
            lines.append(
                f"| {level.capitalize()} | {lb.get('total', 0)} | "
                f"{lb.get('correct', 0)} | {lb.get('accuracy', 0):.4f} |"
            )
        lines.append("")

    return lines


def _compliance_section(metrics: Dict) -> List[str]:
    """Build compliance evaluation section."""
    return [
        "## Compliance Detection Performance",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Coverage Accuracy | {metrics.get('coverage_accuracy', 0):.4f} |",
        f"| Gap Detection Precision | {metrics.get('gap_detection_precision', 0):.4f} |",
        f"| Gap Detection Recall | {metrics.get('gap_detection_recall', 0):.4f} |",
        f"| Gap Detection F1 | {metrics.get('gap_detection_f1', 0):.4f} |",
        f"| True Positive Gaps | {metrics.get('true_positive_gaps', 0)} |",
        f"| False Positive Gaps | {metrics.get('false_positive_gaps', 0)} |",
        f"| False Negative Gaps | {metrics.get('false_negative_gaps', 0)} |",
        "",
    ]


def _baseline_section(comparison: Dict) -> List[str]:
    """Build baseline comparison section."""
    lines = [
        "## Baseline Comparison",
        "",
        f"**Verdict**: {comparison.get('verdict', 'N/A')}",
        "",
    ]

    improvements = comparison.get("improvements", {})
    if improvements:
        lines.append("| Metric | Hybrid | Baseline | Delta | Improvement |")
        lines.append("|--------|--------|----------|-------|-------------|")
        for metric_name, vals in improvements.items():
            lines.append(
                f"| {metric_name} | {vals['hybrid']:.4f} | {vals['baseline']:.4f} | "
                f"{vals['delta']:+.4f} | {vals['improvement_pct']:+.1f}% |"
            )
        lines.append("")

    return lines


def _ablation_section(ablation: Dict) -> List[str]:
    """Build ablation study section."""
    lines = [
        "## Ablation Study",
        "",
    ]

    full = ablation.get("full_model", {})
    lines.append(f"**Full Model**: MAE={full.get('mae', 0):.4f}, "
                 f"RMSE={full.get('rmse', 0):.4f}, "
                 f"Correlation={full.get('correlation', 0):.4f}")
    lines.append("")

    results = ablation.get("ablation_results", {})
    if results:
        lines.append("| Component Removed | MAE | RMSE | MAE Δ | RMSE Δ |")
        lines.append("|-------------------|-----|------|-------|--------|")
        for component, m in results.items():
            lines.append(
                f"| {component.replace('_', ' ').title()} | {m['mae']:.4f} | "
                f"{m['rmse']:.4f} | {m['mae_delta']:+.4f} | {m['rmse_delta']:+.4f} |"
            )
        lines.append("")

    importance = ablation.get("component_importance", [])
    if importance:
        lines.append("### Component Importance Ranking")
        lines.append("")
        for i, comp in enumerate(importance, 1):
            lines.append(f"{i}. **{comp['component'].title()}** — MAE impact: {comp['mae_impact']:.4f}")
        lines.append("")

    return lines


def _statistical_section(stats: Dict) -> List[str]:
    """Build statistical tests section."""
    lines = [
        "## Statistical Significance Tests",
        "",
        f"**Metric**: {stats.get('metric', 'N/A')} | "
        f"**Samples**: {stats.get('n_samples', 0)}",
        "",
    ]

    t_test = stats.get("paired_t_test", {})
    if t_test and "error" not in t_test:
        lines.append("### Paired t-Test")
        lines.append("")
        lines.append(f"- **t-statistic**: {t_test.get('t_statistic', 0):.4f}")
        lines.append(f"- **p-value**: {t_test.get('p_value', 1):.6f}")
        lines.append(f"- **Assessment**: {t_test.get('significance', 'N/A')}")
        lines.append(f"- **Reject H₀ at α=0.05**: {'Yes' if t_test.get('reject_h0_at_005') else 'No'}")
        lines.append("")

    hybrid_ci = stats.get("hybrid_confidence_interval", {})
    if hybrid_ci and "error" not in hybrid_ci:
        cl = hybrid_ci.get("confidence_level", 0.95) * 100
        lines.append(f"### Bootstrap Confidence Intervals ({cl:.0f}%)")
        lines.append("")
        lines.append(
            f"- **Hybrid**: {hybrid_ci.get('mean', 0):.4f} "
            f"[{hybrid_ci.get('ci_lower', 0):.4f}, {hybrid_ci.get('ci_upper', 0):.4f}]"
        )

    baseline_ci = stats.get("baseline_confidence_interval", {})
    if baseline_ci and "error" not in baseline_ci:
        lines.append(
            f"- **Baseline**: {baseline_ci.get('mean', 0):.4f} "
            f"[{baseline_ci.get('ci_lower', 0):.4f}, {baseline_ci.get('ci_upper', 0):.4f}]"
        )
        lines.append("")

    return lines


# ═══════════════════════════════════════════════════════════════
# Graph Data Builder
# ═══════════════════════════════════════════════════════════════

def _build_graph_data(
    classification_metrics: Optional[Dict] = None,
    risk_metrics: Optional[Dict] = None,
    baseline_comparison: Optional[Dict] = None,
    ablation_results: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Build JSON-ready data for visualization."""
    data = {}

    if classification_metrics:
        per_label = classification_metrics.get("per_label_metrics", {})
        data["classification_by_label"] = {
            "labels": list(per_label.keys()),
            "precision": [m["precision"] for m in per_label.values()],
            "recall": [m["recall"] for m in per_label.values()],
            "f1": [m["f1"] for m in per_label.values()],
        }
        data["classification_summary"] = {
            "metrics": ["Precision", "Recall", "F1"],
            "micro": [
                classification_metrics.get("precision_micro", 0),
                classification_metrics.get("recall_micro", 0),
                classification_metrics.get("f1_micro", 0),
            ],
            "macro": [
                classification_metrics.get("precision_macro", 0),
                classification_metrics.get("recall_macro", 0),
                classification_metrics.get("f1_macro", 0),
            ],
        }

    if risk_metrics:
        data["risk_distribution"] = risk_metrics.get("score_distribution", {})
        data["risk_level_accuracy"] = risk_metrics.get("level_breakdown", {})

    if baseline_comparison:
        improvements = baseline_comparison.get("improvements", {})
        data["baseline_comparison"] = {
            "metrics": list(improvements.keys()),
            "hybrid": [v["hybrid"] for v in improvements.values()],
            "baseline": [v["baseline"] for v in improvements.values()],
        }

    if ablation_results:
        results = ablation_results.get("ablation_results", {})
        data["ablation"] = {
            "components": [k.replace("without_", "") for k in results.keys()],
            "mae_delta": [v["mae_delta"] for v in results.values()],
            "rmse_delta": [v["rmse_delta"] for v in results.values()],
        }

    return data


def _build_one_line_summary(
    classification_metrics: Optional[Dict] = None,
    risk_metrics: Optional[Dict] = None,
    baseline_comparison: Optional[Dict] = None,
) -> str:
    """Build a one-line summary of evaluation results."""
    parts = []

    if classification_metrics:
        f1 = classification_metrics.get("f1_macro", 0)
        parts.append(f"F1-macro={f1:.4f}")

    if risk_metrics:
        mae = risk_metrics.get("mae", 0)
        parts.append(f"MAE={mae:.4f}")

    if baseline_comparison:
        verdict = baseline_comparison.get("verdict", "")
        if verdict:
            parts.append(verdict)

    return " | ".join(parts) if parts else "No metrics computed"
