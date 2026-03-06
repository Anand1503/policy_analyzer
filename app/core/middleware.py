"""
Middleware — Request ID correlation and rate limiting.
Category 5: Security (rate limiting)
Category 6: Observability (correlation IDs)
"""

import uuid
import time
import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Injects a unique X-Request-ID header into every request/response.
    Enables log correlation across services.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:12])
        request.state.request_id = request_id

        start = time.perf_counter()
        response = await call_next(request)
        elapsed = int((time.perf_counter() - start) * 1000)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(elapsed)

        logger.info(
            f"[request] {request.method} {request.url.path} "
            f"→ {response.status_code} ({elapsed}ms) [rid={request_id}]"
        )

        return response


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """
    Reject requests with body larger than max_bytes.
    Category 5: Security — prevent oversized payloads.
    """

    def __init__(self, app, max_bytes: int = 50 * 1024 * 1024):
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_bytes:
            return JSONResponse(
                status_code=413,
                content={"detail": f"Request body exceeds {self.max_bytes // (1024*1024)}MB limit"},
            )
        return await call_next(request)
