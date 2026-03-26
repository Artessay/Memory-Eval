"""Retry utilities for transient API failures."""

import os
import random
import time
from typing import Callable, Optional, TypeVar


T = TypeVar("T")
STRING_RESPONSE_CHOICES_ERROR = "'str' object has no attribute 'choices'"


class RetryHandler:
    """Reusable exponential-backoff retry helper."""

    def __init__(
        self,
        max_retries: int = 8,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        jitter: float = 0.25,
        sleep_func: Callable[[float], None] = time.sleep,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self._sleep = sleep_func

    @classmethod
    def from_config(
        cls,
        max_retries: Optional[int] = None,
        base_delay: Optional[float] = None,
        max_delay: Optional[float] = None,
        jitter: Optional[float] = None,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> "RetryHandler":
        return cls(
            max_retries=int(
                max_retries
                if max_retries is not None
                else os.environ.get("MEMORY_EVAL_MAX_RETRIES", 8)
            ),
            base_delay=float(
                base_delay
                if base_delay is not None
                else os.environ.get("MEMORY_EVAL_RETRY_BASE_DELAY", 1.0)
            ),
            max_delay=float(
                max_delay
                if max_delay is not None
                else os.environ.get("MEMORY_EVAL_RETRY_MAX_DELAY", 30.0)
            ),
            jitter=float(
                jitter
                if jitter is not None
                else os.environ.get("MEMORY_EVAL_RETRY_JITTER", 0.25)
            ),
            sleep_func=sleep_func,
        )

    def is_retryable_error(self, exc: Exception) -> bool:
        try:
            from openai import (
                APIConnectionError,
                APIStatusError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
            )
        except ImportError:
            APIConnectionError = APITimeoutError = APIStatusError = InternalServerError = RateLimitError = ()

        if isinstance(
            exc,
            (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError),
        ):
            return True

        if APIStatusError and isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            return status_code == 429 or (status_code is not None and status_code >= 500)

        if isinstance(exc, AttributeError) and str(exc) == STRING_RESPONSE_CHOICES_ERROR:
            return True

        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return status_code == 429 or status_code >= 500

        return False

    def compute_delay(self, attempt: int) -> float:
        delay = min(self.base_delay * (2 ** attempt), self.max_delay)
        if self.jitter <= 0:
            return delay
        return min(delay + random.uniform(0.0, self.jitter), self.max_delay)

    def run(self, operation: Callable[[], T]) -> T:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not self.is_retryable_error(exc):
                    raise
                message = (
                    f"Operation failed with retryable error: {exc}.\n"
                    f"Retrying attempt {attempt + 1}/{self.max_retries}..."
                )
                print(message)
                self._sleep(self.compute_delay(attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry loop exited without returning or raising.")