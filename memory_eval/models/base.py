"""Abstract base class for all model backends."""

import os
import random
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, TypeVar


T = TypeVar("T")


class BaseModel(ABC):
    """Base class for all model backends."""

    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

        # Retry configuration with defaults and environment variable overrides
        self.max_retries = int(kwargs.pop("max_retries", os.environ.get("MEMORY_EVAL_MAX_RETRIES", 4)))
        self.retry_base_delay = float(
            kwargs.pop("retry_base_delay", os.environ.get("MEMORY_EVAL_RETRY_BASE_DELAY", 1.0))
        )
        self.retry_max_delay = float(
            kwargs.pop("retry_max_delay", os.environ.get("MEMORY_EVAL_RETRY_MAX_DELAY", 30.0))
        )
        self.retry_jitter = float(
            kwargs.pop("retry_jitter", os.environ.get("MEMORY_EVAL_RETRY_JITTER", 0.25))
        )

    def _sleep(self, delay_seconds: float) -> None:
        time.sleep(delay_seconds)

    def _is_retryable_error(self, exc: Exception) -> bool:
        try:
            from openai import APIConnectionError, APITimeoutError, APIStatusError, InternalServerError, RateLimitError
        except ImportError:
            APIConnectionError = APITimeoutError = APIStatusError = InternalServerError = RateLimitError = ()

        if isinstance(exc, (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)):
            return True

        if APIStatusError and isinstance(exc, APIStatusError):
            status_code = getattr(exc, "status_code", None)
            return status_code == 429 or (status_code is not None and status_code >= 500)

        status_code = getattr(exc, "status_code", None)
        if status_code is not None:
            return status_code == 429 or status_code >= 500

        return False

    def _compute_retry_delay(self, attempt: int) -> float:
        delay = min(self.retry_base_delay * (2 ** attempt), self.retry_max_delay)
        if self.retry_jitter <= 0:
            return delay
        jitter = random.uniform(0.0, self.retry_jitter)
        return min(delay + jitter, self.retry_max_delay)

    def _call_with_retry(self, operation: Callable[[], T]) -> T:
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_retries or not self._is_retryable_error(exc):
                    raise
                self._sleep(self._compute_retry_delay(attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("Retry loop exited without returning or raising.")

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Generate a response given a list of messages.

        Args:
            messages: A list of message dicts with 'role' and 'content'.
                      Content can be a string or a list of content parts
                      (for multimodal inputs).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.

        Returns:
            The generated text response.
        """

    def batch_generate(
        self,
        all_messages: List[List[Dict[str, Any]]],
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs,
    ) -> List[str]:
        """Generate responses for a batch of message lists."""
        return [
            self.generate(messages, max_tokens=max_tokens, temperature=temperature, **kwargs)
            for messages in all_messages
        ]
