"""
In-memory rate limiter — per-user token bucket with configurable limits.
No external dependency required (no slowapi).
"""

import time
import logging
from collections import defaultdict
from typing import Optional
from fastapi import Request, HTTPException, Depends
from app.core.config import settings

logger = logging.getLogger(__name__)


class _TokenBucket:
    """Simple token bucket rate limiter."""

    def __init__(self, rate: float, capacity: int):
        self.rate = rate            # tokens per second
        self.capacity = capacity    # max burst
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self) -> bool:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_refill = now

        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False

    @property
    def retry_after(self) -> float:
        """Seconds until next token available."""
        if self.tokens >= 1.0:
            return 0.0
        return (1.0 - self.tokens) / self.rate


class RateLimiter:
    """
    Per-user rate limiter using token bucket algorithm.

    Usage:
        limiter = RateLimiter()

        @router.post("/classify/{doc_id}")
        async def classify(doc_id, _=Depends(limiter.limit("classify", 10, 60))):
            ...
    """

    def __init__(self):
        self._buckets: dict[str, _TokenBucket] = {}

    def _get_bucket(self, key: str, max_requests: int, window_seconds: int) -> _TokenBucket:
        if key not in self._buckets:
            rate = max_requests / window_seconds
            self._buckets[key] = _TokenBucket(rate, max_requests)
        return self._buckets[key]

    def limit(self, endpoint: str, max_requests: int = 10, window_seconds: int = 60):
        """
        FastAPI dependency that enforces rate limiting.

        Args:
            endpoint: Endpoint name (used as bucket namespace).
            max_requests: Max requests per window.
            window_seconds: Time window in seconds.
        """
        async def dependency(request: Request):
            # Extract user ID from request state (set by JWT auth)
            user_id = "anonymous"
            if hasattr(request.state, "user_id"):
                user_id = request.state.user_id
            else:
                # Fallback: extract from auth header
                auth = request.headers.get("authorization", "")
                if auth:
                    user_id = auth[-12:]  # Use last 12 chars of token as key

            bucket_key = f"{endpoint}:{user_id}"
            bucket = self._get_bucket(bucket_key, max_requests, window_seconds)

            if not bucket.consume():
                retry_after = int(bucket.retry_after) + 1
                logger.warning(
                    f"[rate_limit] 429 for {endpoint} user={user_id[:8]}... "
                    f"retry_after={retry_after}s"
                )
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded for {endpoint}. Try again in {retry_after}s.",
                    headers={"Retry-After": str(retry_after)},
                )

        return dependency


# Global singleton
rate_limiter = RateLimiter()
