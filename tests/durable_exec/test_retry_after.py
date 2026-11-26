"""Unit tests for `model_error_retry_after`, not VCR tests: it walks an exception chain in memory and makes no request, so there is nothing for a cassette to record."""

from __future__ import annotations

import pytest

from pydantic_ai.durable_exec import model_error_retry_after
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, ModelRateLimitError


def _rate_limited(retry_after: float | None) -> ModelHTTPError:
    return ModelHTTPError.for_category(
        ModelRateLimitError, status_code=429, model_name='test-model', retry_after=retry_after
    )


@pytest.mark.parametrize(
    ('retry_after', 'maximum', 'expected'),
    [
        pytest.param(7.0, 60.0, 7.0, id='retry-after'),
        pytest.param(3600.0, 60.0, 60.0, id='capped'),
        pytest.param(None, 60.0, None, id='no-retry-after'),
    ],
)
def test_model_error_retry_after(retry_after: float | None, maximum: float, expected: float | None) -> None:
    """A model error's `retry_after`, up to `maximum`, or `None` when it asked for no wait."""
    assert model_error_retry_after(_rate_limited(retry_after), maximum=maximum) == expected


def test_model_error_retry_after_finds_the_model_error_an_engine_error_was_raised_from() -> None:
    """An engine or a wrapper can raise its own error around the model error, from it or while handling it."""
    try:
        try:
            raise _rate_limited(7.0)
        except ModelAPIError as error:
            raise RuntimeError('step failed') from error
    except RuntimeError as error:
        assert model_error_retry_after(error, maximum=60.0) == 7.0

    try:
        try:
            raise _rate_limited(7.0)
        except ModelAPIError:
            raise RuntimeError('step failed')
    except RuntimeError as error:
        assert model_error_retry_after(error, maximum=60.0) == 7.0


def test_model_error_retry_after_stops_at_a_suppressed_context() -> None:
    """A wrapper that raised its own error `from None` hid the model error, so its wait no longer applies."""
    try:
        try:
            raise _rate_limited(7.0)
        except ModelAPIError:
            raise RuntimeError('step failed') from None
    except RuntimeError as error:
        assert model_error_retry_after(error, maximum=60.0) is None


def test_model_error_retry_after_ignores_errors_that_are_not_model_errors() -> None:
    """Without a model error in the chain, even one that loops back on itself, the engine's backoff applies."""
    first = RuntimeError('first')
    second = ValueError('second')
    first.__context__ = second
    second.__context__ = first
    assert model_error_retry_after(first, maximum=60.0) is None
