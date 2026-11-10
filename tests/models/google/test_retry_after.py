"""`retry_after` from the `google.rpc.RetryInfo` in a Gemini error body.

Not a VCR test: providers don't return a 429 on demand, and no single recording could pin the protobuf `Duration`
spellings, the body shapes without a `RetryInfo`, or the header precedence below. The adapter paths that read this
are covered by the `google-*` cases in `tests/models/test_error_categories.py`.
"""

from __future__ import annotations as _annotations

from typing import Any

import pytest

from pydantic_ai._model_errors import google_retry_after
from pydantic_ai.exceptions import ModelHTTPError

_RETRY_INFO = 'type.googleapis.com/google.rpc.RetryInfo'


def _body(*details: dict[str, Any]) -> dict[str, Any]:
    return {
        'error': {
            'code': 429,
            'message': 'Resource has been exhausted',
            'status': 'RESOURCE_EXHAUSTED',
            'details': list(details),
        }
    }


@pytest.mark.parametrize(
    ('body', 'headers', 'expected'),
    [
        pytest.param(_body({'@type': _RETRY_INFO, 'retryDelay': '14s'}), None, 14.0, id='seconds'),
        pytest.param(_body({'@type': _RETRY_INFO, 'retryDelay': '0.5s'}), None, 0.5, id='fractional-seconds'),
        pytest.param(_body({'@type': _RETRY_INFO, 'retryDelay': '1.000340012s'}), None, 1.000340012, id='nanoseconds'),
        pytest.param(
            _body({'@type': _RETRY_INFO, 'retryDelay': '14s'}), {'retry-after': '7'}, 7.0, id='retry-after-header-wins'
        ),
        pytest.param(
            _body({'@type': _RETRY_INFO, 'retryDelay': '14s'}),
            {'retry-after-ms': '7000'},
            7.0,
            id='retry-after-ms-header-wins',
        ),
        pytest.param(_body(), None, None, id='no-details'),
        pytest.param(_body({'@type': 'type.googleapis.com/google.rpc.Help'}), None, None, id='no-retry-info'),
        pytest.param(_body({'@type': _RETRY_INFO}), None, None, id='no-retry-delay'),
        pytest.param(_body({'@type': _RETRY_INFO, 'retryDelay': '14'}), None, None, id='no-second-suffix'),
        pytest.param(_body({'@type': _RETRY_INFO, 'retryDelay': 'soon'}), None, None, id='not-a-number'),
        pytest.param(_body({'@type': _RETRY_INFO, 'retryDelay': '-3s'}), None, None, id='negative'),
        pytest.param(_body({'@type': _RETRY_INFO, 'retryDelay': 'infs'}), None, None, id='infinite'),
        pytest.param(_body({'@type': _RETRY_INFO, 'retryDelay': 14}), None, None, id='not-a-string'),
    ],
)
def test_google_retry_after(body: dict[str, Any], headers: dict[str, str] | None, expected: float | None) -> None:
    """A protobuf `Duration` is the fallback, and a `Retry-After` header the error already read keeps its value."""
    error = ModelHTTPError(429, 'gemini-2.5-flash', body, headers=headers)
    assert google_retry_after(error) == expected
