"""
Retry utility — exponential backoff for transient failures.
Supports DB, ML, and ChromaDB transient errors.
"""

import asyncio
import functools
import logging
import time
from typing import Tuple, Type

logger = logging.getLogger(__name__)

# Transient error types that should be retried
_TRANSIENT_KEYWORDS = (
    "connection",
    "timeout",
    "temporary",
    "too many clients",
    "connection refused",
    "out of memory",
    "resource temporarily unavailable",
    "broken pipe",
)


def is_transient(exc: Exception) -> bool:
    """Check if an exception is likely transient and retryable."""
    msg = str(exc).lower()
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)


def retry_sync(
    max_retries: int = 3,
    backoff_base: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retry_on_transient_only: bool = True,
):
    """
    Synchronous retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds (doubles each retry).
        retryable_exceptions: Exception types to catch.
        retry_on_transient_only: If True, only retries transient errors.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if retry_on_transient_only and not is_transient(e):
                        logger.debug(f"[retry] Non-transient error, not retrying: {e}")
                        raise
                    if attempt < max_retries:
                        delay = backoff_base * (2 ** attempt)
                        logger.warning(
                            f"[retry] {func.__name__} attempt {attempt+1}/{max_retries} "
                            f"failed: {e}. Retrying in {delay}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"[retry] {func.__name__} exhausted {max_retries} retries"
                        )
            raise last_exc
        return wrapper
    return decorator


def retry_async(
    max_retries: int = 3,
    backoff_base: float = 1.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    retry_on_transient_only: bool = True,
):
    """
    Async retry decorator with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        backoff_base: Base delay in seconds (doubles each retry).
        retryable_exceptions: Exception types to catch.
        retry_on_transient_only: If True, only retries transient errors.
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exc = e
                    if retry_on_transient_only and not is_transient(e):
                        raise
                    if attempt < max_retries:
                        delay = backoff_base * (2 ** attempt)
                        logger.warning(
                            f"[retry] {func.__name__} attempt {attempt+1}/{max_retries} "
                            f"failed: {e}. Retrying in {delay}s..."
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            f"[retry] {func.__name__} exhausted {max_retries} retries"
                        )
            raise last_exc
        return wrapper
    return decorator
