"""
Prometheus metrics instrumentation.
Exports: /metrics endpoint, middleware for request counting, ML timing histograms.
"""

import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, PlainTextResponse

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# In-Memory Metrics Store (no external dependency needed)
# ═══════════════════════════════════════════════════════════════

class _MetricsStore:
    """Thread-safe in-memory Prometheus-compatible metrics."""

    def __init__(self):
        self._counters = {}
        self._histograms = {}

    def inc(self, name: str, labels: dict = None, value: float = 1):
        key = (name, self._label_key(labels))
        self._counters[key] = self._counters.get(key, 0) + value

    def observe(self, name: str, value: float, labels: dict = None):
        key = (name, self._label_key(labels))
        if key not in self._histograms:
            self._histograms[key] = {"count": 0, "sum": 0.0}
        self._histograms[key]["count"] += 1
        self._histograms[key]["sum"] += value

    def _label_key(self, labels: dict = None) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def export(self) -> str:
        """Export metrics in Prometheus text format."""
        lines = []
        lines.append("# Intelligent Policy Analyzer Metrics\n")

        for (name, label_key), value in sorted(self._counters.items()):
            lbl = f"{{{label_key}}}" if label_key else ""
            lines.append(f"{name}{lbl} {value}")

        for (name, label_key), data in sorted(self._histograms.items()):
            lbl = f"{{{label_key}}}" if label_key else ""
            lines.append(f"{name}_count{lbl} {data['count']}")
            lines.append(f"{name}_sum{lbl} {data['sum']:.4f}")
            if data["count"] > 0:
                avg = data["sum"] / data["count"]
                lines.append(f"{name}_avg{lbl} {avg:.4f}")

        return "\n".join(lines) + "\n"


# Global singleton
metrics = _MetricsStore()


# ═══════════════════════════════════════════════════════════════
# Pre-defined metric helpers
# ═══════════════════════════════════════════════════════════════

def track_classification(duration_seconds: float, clause_count: int):
    metrics.observe("classification_duration_seconds", duration_seconds,
                    {"module": "classifier"})
    metrics.inc("classification_clauses_total", value=clause_count)

def track_risk_scoring(duration_seconds: float, clause_count: int):
    metrics.observe("risk_duration_seconds", duration_seconds,
                    {"module": "risk_scorer"})
    metrics.inc("risk_clauses_total", value=clause_count)

def track_explanation(duration_seconds: float, clause_count: int):
    metrics.observe("explain_duration_seconds", duration_seconds,
                    {"module": "explainability"})

def track_model_load(model_name: str, success: bool):
    status = "success" if success else "failure"
    metrics.inc("model_load_total", {"model": model_name, "status": status})

def track_request(method: str, path: str, status_code: int, duration: float):
    metrics.observe("http_request_duration_seconds", duration,
                    {"method": method, "path": path})
    metrics.inc("http_requests_total",
                {"method": method, "path": path, "status": str(status_code)})


# ═══════════════════════════════════════════════════════════════
# Metrics Middleware
# ═══════════════════════════════════════════════════════════════

class MetricsMiddleware(BaseHTTPMiddleware):
    """Track HTTP request count and duration."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start

        # Skip metrics endpoint itself
        if request.url.path != "/metrics":
            track_request(
                request.method,
                request.url.path,
                response.status_code,
                duration,
            )
            metrics.inc("active_requests_served")

        return response


# ═══════════════════════════════════════════════════════════════
# /metrics endpoint handler
# ═══════════════════════════════════════════════════════════════

async def metrics_endpoint(request: Request):
    """Prometheus-compatible /metrics endpoint."""
    return PlainTextResponse(
        content=metrics.export(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
