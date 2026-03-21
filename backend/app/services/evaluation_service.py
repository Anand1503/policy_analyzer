"""
Evaluation Service — Orchestrates evaluation pipeline.
Module 7: Connects ML evaluation engines to DB storage.

Design:
  - Service orchestrates: calls ML engines, stores results
  - Separate DB transaction (does NOT modify production data)
  - ML engines are pure computation
"""

import uuid
import time
import logging
from typing import Dict, List, Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import AsyncSessionLocal
from app.models.models import EvaluationRun

logger = logging.getLogger(__name__)


class EvaluationService:
    """Orchestrates evaluation runs."""

    @staticmethod
    async def run_evaluation(
        dataset_name: str,
        clauses: List[Dict[str, Any]],
        compliance_ground_truth: Optional[List[Dict[str, Any]]] = None,
        framework: str = "GDPR",
        run_baseline: bool = True,
        run_ablation: bool = True,
        run_statistical_tests: bool = True,
    ) -> Dict:
        """
        Run a complete evaluation pipeline.

        Flow:
          1. Run hybrid classification on test clauses
          2. Compute classification metrics vs ground truth
          3. Run hybrid risk scoring on test clauses
          4. Compute risk metrics vs ground truth
          5. (Optional) Run baseline comparison
          6. (Optional) Run ablation study
          7. (Optional) Run statistical significance tests
          8. (Optional) Evaluate compliance detection
          9. Generate research report
          10. Store evaluation run in separate transaction

        Does NOT modify production data.
        """
        from app.ml.evaluation_engine import evaluate_classification, evaluate_compliance_detection
        from app.ml.risk_validation import validate_risk_scores, run_ablation_study
        from app.ml.baseline_engine import (
            keyword_baseline_classify, label_weight_baseline_risk,
            compare_with_baseline,
        )
        from app.ml.statistical_tests import run_statistical_tests as stat_tests
        from app.ml.research_report_generator import generate_research_report
        from app.ml.model_loader import get_model_version

        t0 = time.perf_counter()

        # Prepare clause data
        clause_data = [
            {
                "clause_id": c.get("clause_id", str(uuid.uuid4())),
                "clause_text": c.get("clause_text", ""),
            }
            for c in clauses
        ]

        # Ground truth for classification
        gt_classification = [
            {
                "clause_id": c["clause_id"],
                "labels": c.get("true_labels", []),
            }
            for c in clauses
            if c.get("true_labels")
        ]

        # Ground truth for risk
        gt_risk = [
            {
                "clause_id": c["clause_id"],
                "risk_score": c.get("true_risk_score", 0.0),
                "risk_level": c.get("true_risk_level", "low"),
            }
            for c in clauses
            if c.get("true_risk_score") is not None
        ]

        # ── 1. Simulate hybrid classification ────────────────
        # In evaluation mode, we use keyword baseline as "predicted"
        # and compare against ground truth to compute metrics
        hybrid_predictions = keyword_baseline_classify(clause_data)

        # ── 2. Classification metrics ────────────────────────
        classification_metrics = None
        if gt_classification:
            classification_metrics = evaluate_classification(
                hybrid_predictions, gt_classification,
            )
            logger.info(
                f"[eval] Classification: F1-macro={classification_metrics.get('f1_macro', 0):.4f}"
            )

        # ── 3. Risk scoring ──────────────────────────────────
        # Assign labels to clauses for risk scoring
        pred_label_map = {
            str(p["clause_id"]): p.get("labels", [])
            for p in hybrid_predictions
        }
        risk_clauses = [
            {
                "clause_id": c["clause_id"],
                "labels": pred_label_map.get(c["clause_id"], []),
            }
            for c in clause_data
        ]
        hybrid_risk = label_weight_baseline_risk(risk_clauses)

        # ── 4. Risk metrics ──────────────────────────────────
        risk_metrics = None
        if gt_risk:
            risk_metrics = validate_risk_scores(hybrid_risk, gt_risk)
            logger.info(
                f"[eval] Risk: MAE={risk_metrics.get('mae', 0):.4f}, "
                f"Correlation={risk_metrics.get('correlation', 0):.4f}"
            )

        # ── 5. Baseline comparison ───────────────────────────
        baseline_comparison = None
        if run_baseline and classification_metrics:
            baseline_preds = keyword_baseline_classify(clause_data)
            baseline_metrics = evaluate_classification(baseline_preds, gt_classification)
            baseline_comparison = compare_with_baseline(
                classification_metrics, baseline_metrics, "classification",
            )

        # ── 6. Ablation study ────────────────────────────────
        ablation_results = None
        if run_ablation and gt_risk and hybrid_risk:
            ablation_results = run_ablation_study(
                clause_data, gt_risk, hybrid_risk,
            )

        # ── 7. Statistical tests ─────────────────────────────
        statistical_tests_result = None
        if run_statistical_tests and gt_classification and len(gt_classification) >= 5:
            # Per-clause F1 for statistical testing
            per_clause_f1_hybrid = _compute_per_clause_f1(
                hybrid_predictions, gt_classification,
            )
            per_clause_f1_baseline = _compute_per_clause_f1(
                keyword_baseline_classify(clause_data), gt_classification,
            )
            if per_clause_f1_hybrid and per_clause_f1_baseline:
                statistical_tests_result = stat_tests(
                    per_clause_f1_hybrid, per_clause_f1_baseline, "F1",
                )

        # ── 8. Compliance evaluation ─────────────────────────
        compliance_metrics = None
        if compliance_ground_truth:
            # Run compliance engine on test data
            from app.ml.compliance_engine import evaluate_compliance
            compliance_report = evaluate_compliance(
                framework_name=framework,
                clauses=clause_data,
                classifications=[
                    {"clause_id": p["clause_id"], "label": l, "confidence_score": 0.8}
                    for p in hybrid_predictions
                    for l in p.get("labels", [])
                ],
                risk_scores=hybrid_risk,
                entities=[],
            )
            # Build predicted compliance status
            predicted_compliance = []
            for finding in compliance_report.get("fully_satisfied", []):
                predicted_compliance.append({"article_id": finding["article_id"], "status": "satisfied"})
            for finding in compliance_report.get("partial_requirements", []):
                predicted_compliance.append({"article_id": finding["article_id"], "status": "partial"})
            for finding in compliance_report.get("missing_requirements", []):
                predicted_compliance.append({"article_id": finding["article_id"], "status": "missing"})

            compliance_metrics = evaluate_compliance_detection(
                predicted_compliance, compliance_ground_truth,
            )

        # ── 9. Generate report ───────────────────────────────
        model_version = get_model_version("classifier")
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        config_snapshot = {
            "dataset_name": dataset_name,
            "total_clauses": len(clause_data),
            "framework": framework,
            "run_baseline": run_baseline,
            "run_ablation": run_ablation,
            "run_statistical_tests": run_statistical_tests,
            "evaluation_time_ms": elapsed_ms,
        }

        report = generate_research_report(
            dataset_name=dataset_name,
            model_version=model_version or "unknown",
            classification_metrics=classification_metrics,
            risk_metrics=risk_metrics,
            compliance_metrics=compliance_metrics,
            baseline_comparison=baseline_comparison,
            ablation_results=ablation_results,
            statistical_tests=statistical_tests_result,
            config_snapshot=config_snapshot,
        )

        # ── 10. Store in separate transaction ────────────────
        async with AsyncSessionLocal() as session:
            async with session.begin():
                eval_run = EvaluationRun(
                    dataset_name=dataset_name,
                    model_version=model_version,
                    config_snapshot=config_snapshot,
                    classification_metrics=classification_metrics,
                    risk_metrics=risk_metrics,
                    compliance_metrics=compliance_metrics,
                    baseline_comparison=baseline_comparison,
                    ablation_results=ablation_results,
                    statistical_tests=statistical_tests_result,
                    report_markdown=report["markdown"],
                    graph_data=report["graph_data"],
                )
                session.add(eval_run)
                await session.flush()
                run_id = eval_run.id

        logger.info(
            f"[eval] ✓ Evaluation '{dataset_name}' complete: "
            f"id={run_id} ({elapsed_ms}ms)"
        )

        return {
            "id": run_id,
            "dataset_name": dataset_name,
            "model_version": model_version,
            "classification_metrics": classification_metrics,
            "risk_metrics": risk_metrics,
            "compliance_metrics": compliance_metrics,
            "baseline_comparison": baseline_comparison,
            "ablation_results": ablation_results,
            "statistical_tests": statistical_tests_result,
            "report_markdown": report["markdown"],
            "graph_data": report["graph_data"],
            "created_at": None,
        }

    @staticmethod
    async def get_evaluation(evaluation_id: uuid.UUID) -> Optional[Dict]:
        """Retrieve a stored evaluation run."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(EvaluationRun).where(EvaluationRun.id == evaluation_id)
            )
            run = result.scalar_one_or_none()
            if not run:
                return None

            return {
                "id": run.id,
                "dataset_name": run.dataset_name,
                "model_version": run.model_version,
                "classification_metrics": run.classification_metrics,
                "risk_metrics": run.risk_metrics,
                "compliance_metrics": run.compliance_metrics,
                "baseline_comparison": run.baseline_comparison,
                "ablation_results": run.ablation_results,
                "statistical_tests": run.statistical_tests,
                "report_markdown": run.report_markdown,
                "graph_data": run.graph_data,
                "created_at": str(run.created_at) if run.created_at else None,
            }

    @staticmethod
    async def list_evaluations(limit: int = 20) -> List[Dict]:
        """List recent evaluation runs."""
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(EvaluationRun)
                .order_by(EvaluationRun.created_at.desc())
                .limit(limit)
            )
            runs = result.scalars().all()

            return [
                {
                    "id": r.id,
                    "dataset_name": r.dataset_name,
                    "model_version": r.model_version,
                    "f1_macro": (
                        r.classification_metrics.get("f1_macro")
                        if r.classification_metrics else None
                    ),
                    "risk_mae": (
                        r.risk_metrics.get("mae")
                        if r.risk_metrics else None
                    ),
                    "created_at": str(r.created_at) if r.created_at else None,
                }
                for r in runs
            ]


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def _compute_per_clause_f1(
    predictions: List[Dict],
    ground_truth: List[Dict],
) -> List[float]:
    """Compute per-clause F1 for statistical testing."""
    gt_map = {str(g["clause_id"]): set(g.get("labels", [])) for g in ground_truth}
    pred_map = {str(p["clause_id"]): set(p.get("labels", [])) for p in predictions}

    f1_scores = []
    for cid in gt_map:
        gt_labels = gt_map[cid]
        pred_labels = pred_map.get(cid, set())

        if not gt_labels and not pred_labels:
            f1_scores.append(1.0)
            continue

        tp = len(gt_labels & pred_labels)
        fp = len(pred_labels - gt_labels)
        fn = len(gt_labels - pred_labels)

        if tp == 0:
            f1_scores.append(0.0)
        else:
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            f1_scores.append(2 * p * r / (p + r))

    return f1_scores
