"""Tests for retry utilities."""

from unittest.mock import MagicMock

import pytest

from memory_eval.utils.retry import RetryHandler


class RetryableError(Exception):
    status_code = 429


class NonRetryableError(Exception):
    status_code = 400


def test_retry_handler_retries_on_retryable_error():
    sleep_mock = MagicMock()
    handler = RetryHandler(
        max_retries=2,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
        sleep_func=sleep_mock,
    )
    operation = MagicMock(side_effect=[RetryableError(), "ok"])

    result = handler.run(operation)

    assert result == "ok"
    assert operation.call_count == 2
    sleep_mock.assert_called_once_with(0.0)


def test_retry_handler_does_not_retry_on_non_retryable_error():
    sleep_mock = MagicMock()
    handler = RetryHandler(
        max_retries=2,
        base_delay=0.0,
        max_delay=0.0,
        jitter=0.0,
        sleep_func=sleep_mock,
    )
    operation = MagicMock(side_effect=NonRetryableError())

    with pytest.raises(NonRetryableError):
        handler.run(operation)

    assert operation.call_count == 1
    sleep_mock.assert_not_called()


def test_retry_handler_uses_environment_defaults():
    handler = RetryHandler.from_config(
        max_retries=3,
        base_delay=1.5,
        max_delay=10.0,
        jitter=0.1,
    )

    assert handler.max_retries == 3
    assert handler.base_delay == 1.5
    assert handler.max_delay == 10.0
    assert handler.jitter == 0.1