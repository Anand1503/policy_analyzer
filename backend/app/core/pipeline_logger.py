"""
Pipeline Performance Logger — centralized timing and performance tracking.

Provides:
- PipelineTimer: context manager for timing individual pipeline stages
- PipelineReport: aggregates timings and logs a structured summary
- Convenience decorators for ML inference functions

Usage:
    with PipelineTimer("embedding") as t:
        embeddings = model.encode(texts)
    # t.elapsed_ms is available after the block

    report = PipelineReport("document_analysis")
    report.record("upload", 120)
    report.record("extraction", 340)
    report.record("classification", 2100)
    report.log_summary()
"""

import time
import logging
from typing import Optional, Dict, List
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PipelineTimer:
    """
    Context manager that measures elapsed time for a pipeline stage.

    Usage:
        with PipelineTimer("classification") as timer:
            result = classify_clauses(clauses)
        logger.info(f"Classification took {timer.elapsed_ms}ms")
    """

    def __init__(self, stage_name: str, log_on_exit: bool = True):
        self.stage_name = stage_name
        self.log_on_exit = log_on_exit
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: int = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = int((self.end_time - self.start_time) * 1000)
        if self.log_on_exit:
            status = "✓" if exc_type is None else "✗"
            logger.info(
                f"[pipeline_timer] {status} {self.stage_name}: {self.elapsed_ms}ms"
            )
        return False  # Don't suppress exceptions


class PipelineReport:
    """
    Collects timing records for all pipeline stages and logs a structured summary.

    Usage:
        report = PipelineReport("full_analysis", document_id="abc123")
        report.record("upload", 150)
        report.record("extraction", 800)
        report.record("classification", 2200)
        report.record("risk_scoring", 45)
        report.record("summarization", 3100)
        report.log_summary()
    """

    def __init__(self, pipeline_name: str, document_id: str = ""):
        self.pipeline_name = pipeline_name
        self.document_id = document_id
        self.stages: List[Dict] = []
        self.start_time = time.perf_counter()

    def record(self, stage: str, elapsed_ms: int, metadata: Optional[dict] = None):
        """Record timing for a named stage."""
        self.stages.append({
            "stage": stage,
            "elapsed_ms": elapsed_ms,
            "metadata": metadata or {},
        })

    @property
    def total_ms(self) -> int:
        """Total elapsed time from pipeline creation."""
        return int((time.perf_counter() - self.start_time) * 1000)

    def log_summary(self):
        """Log a structured performance summary."""
        total = self.total_ms
        doc_label = f" [{self.document_id}]" if self.document_id else ""
        separator = "─" * 50

        logger.info(f"\n{separator}")
        logger.info(f"[pipeline_report] {self.pipeline_name}{doc_label}")
        logger.info(f"{'Stage':<25} {'Time (ms)':>10} {'% Total':>10}")
        logger.info(separator)

        for stage_info in self.stages:
            name = stage_info["stage"]
            ms = stage_info["elapsed_ms"]
            pct = f"{(ms / total * 100):.1f}%" if total > 0 else "n/a"
            logger.info(f"{name:<25} {ms:>10}ms {pct:>10}")

            # Log metadata if present
            for k, v in stage_info.get("metadata", {}).items():
                logger.info(f"  {k}: {v}")

        logger.info(separator)
        logger.info(f"{'TOTAL':<25} {total:>10}ms {'100%':>10}")
        logger.info(separator)

        # Identify bottleneck
        if self.stages:
            slowest = max(self.stages, key=lambda s: s["elapsed_ms"])
            logger.info(
                f"[pipeline_report] Bottleneck: '{slowest['stage']}' "
                f"({slowest['elapsed_ms']}ms, "
                f"{slowest['elapsed_ms'] / total * 100:.1f}% of total)"
            )

    def as_dict(self) -> dict:
        """Return timing data as a serializable dict."""
        return {
            "pipeline": self.pipeline_name,
            "document_id": self.document_id,
            "total_ms": self.total_ms,
            "stages": self.stages,
        }


@contextmanager
def timed_stage(report: PipelineReport, stage_name: str, **metadata):
    """
    Context manager that times a stage and records it in a PipelineReport.

    Usage:
        with timed_stage(report, "classification", clauses=len(clauses)):
            results = classify_clauses(clauses)
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = int((time.perf_counter() - t0) * 1000)
        report.record(stage_name, elapsed, metadata)
        logger.info(f"[{stage_name}] Completed in {elapsed}ms")
