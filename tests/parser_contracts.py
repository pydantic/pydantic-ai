"""Parsers of external input, each with the input it reads and the only exceptions it may raise.

A parser of what a model, a provider or a client sends is called with whatever arrives. An exception class its
caller doesn't expect escapes the caller's handling and ends the run instead of being reported, as the
`OverflowError` from a `Retry-After` date past the year 9999 did, until
https://github.com/pydantic/pydantic-ai/pull/10023. Each `ParserContract` names a parser, a Hypothesis strategy for
its input, the exceptions it may raise, and the inputs that once escaped it.
`tests/test_parser_contracts.py` calls each with drawn inputs and with those examples, and fails on any other
exception.

Register a parser here when it reads text or numbers a model, a provider or a client chose. The Pydantic AI Harness
parsers are in `tests/harness/test_parser_contracts.py`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TypeVar
from unittest.mock import Mock

import httpx2
from hypothesis import HealthCheck, settings, strategies as st

from pydantic_ai._model_errors import google_retry_after
from pydantic_ai._utils import bare_media_type, has_yaml_structured_suffix, is_text_like_media_type
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import BinaryContent

from .conftest import try_import

with try_import() as retries_available:
    from tenacity import RetryCallState

    from pydantic_ai.retries import wait_retry_after

with try_import() as evals_available:
    from pydantic_evals.reporting.render_numbers import (
        default_render_number,
        default_render_number_diff,
        default_render_percentage,
    )

T = TypeVar('T')

Call = Callable[[], object]
"""A parser applied to one input, ready to run."""


@dataclass(frozen=True)
class ParserContract:
    id: str
    calls: st.SearchStrategy[Call]
    """The parser applied to each input the strategy draws."""
    raises: tuple[type[Exception], ...]
    """The exceptions the parser may raise, which its callers handle."""
    examples: tuple[Call, ...]
    """The parser applied to the inputs that once raised something else, each kept since its fix."""


_long_run_examples = os.environ.get('PARSER_CONTRACT_EXAMPLES')
CONTRACT_SETTINGS = settings(
    max_examples=int(_long_run_examples) if _long_run_examples else 50,
    derandomize=_long_run_examples is None,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
"""A derandomized batch of 50 examples per contract, so a run is reproducible and fits the per-test budget.

Hunt for new failures with `PARSER_CONTRACT_EXAMPLES=1000 uv run pytest tests/test_parser_contracts.py
tests/harness/test_parser_contracts.py`, which draws 1000 randomized examples per contract. `cost_guards` bounds each
test's time, so Hypothesis's own check on slow drawing is off."""


def check(contract: ParserContract, call: Call) -> None:
    """Run `call`, letting through only an exception `contract` doesn't declare."""
    try:
        call()
    except contract.raises:
        pass


def contract(
    id: str,
    parse: Callable[[T], object],
    inputs: st.SearchStrategy[T],
    *,
    raises: tuple[type[Exception], ...] = (),
    examples: Sequence[T] = (),
) -> ParserContract:
    return ParserContract(
        id, inputs.map(lambda value: partial(parse, value)), raises, tuple(partial(parse, e) for e in examples)
    )


_HEADER_VALUES = st.one_of(
    st.text(),
    st.from_regex(r'\s?[+-]?\d{1,400}(\.\d*)?\s?', fullmatch=True),
    st.from_regex(r'[A-Za-z]{3}, \d{1,14} [A-Za-z]{3} \d{1,14} \d{1,3}:\d{1,3}:\d{1,3} [A-Z]{0,4}', fullmatch=True),
)
"""Header values: any text, numbers of any size and sign, and dates with fields of any size."""

_MEDIA_TYPES = st.one_of(
    st.text(), st.from_regex(r'\s?[A-Za-z]{1,10}/[A-Za-z0-9.+-]{0,20}(\s?;\s?[a-z]+=[^;]*){0,3}', fullmatch=True)
)
"""Media types: any text, and `type/subtype` with any parameters."""

_NUMBERS = st.one_of(st.integers(), st.floats())
"""Numbers, `inf`, `-inf` and `nan` included."""

_RETRY_AFTER_HEADERS = st.dictionaries(st.sampled_from(['retry-after', 'retry-after-ms']), _HEADER_VALUES)
_OVERFLOWING_DATE = {'retry-after': 'Wed, 21 Oct 999999999999 07:28:00 GMT'}


def _model_http_error_retry_after(headers: dict[str, str]) -> float | None:
    return ModelHTTPError(status_code=429, model_name='model', headers=headers).retry_after


def _google_retry_delay(delay: str) -> float | None:
    body = {'error': {'details': [{'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': delay}]}}
    return google_retry_after(ModelHTTPError(status_code=429, model_name='gemini', body=body))


def _binary_content_format(media_type: str) -> str:
    return BinaryContent(data=b'', media_type=media_type).format


CONTRACTS: list[ParserContract] = [
    contract(
        'model-http-error-retry-after',
        _model_http_error_retry_after,
        _RETRY_AFTER_HEADERS,
        examples=[_OVERFLOWING_DATE, {'retry-after': '9' * 400}],
    ),
    contract(
        'google-retry-delay',
        _google_retry_delay,
        st.one_of(st.text(), st.from_regex(r'[+-]?\d{1,400}(\.\d*)?s', fullmatch=True)),
        examples=['infs', 'nans', '-1s'],
    ),
    contract('bare-media-type', bare_media_type, _MEDIA_TYPES),
    contract('is-text-like-media-type', is_text_like_media_type, _MEDIA_TYPES),
    contract('has-yaml-structured-suffix', has_yaml_structured_suffix, _MEDIA_TYPES),
    contract('binary-content-format', _binary_content_format, _MEDIA_TYPES, raises=(ValueError,)),
    contract(
        'binary-content-from-data-uri',
        BinaryContent.from_data_uri,
        st.text().map(lambda rest: f'data:{rest}'),
        raises=(ValueError,),
        examples=['data:text/plain;base64,%ZZ', 'data:;base64,a'],
    ),
]

if retries_available():

    def _wait_retry_after(headers: dict[str, str]) -> float:
        request = httpx2.Request('GET', 'https://example.com')
        # Header values arrive as bytes; `httpx2` would refuse to encode a non-ASCII `str` value itself.
        raw_headers = [(name.encode(), value.encode()) for name, value in headers.items()]
        response = httpx2.Response(429, headers=raw_headers, request=request)
        state = Mock(spec=RetryCallState)
        state.outcome = Mock()
        state.outcome.exception.return_value = httpx2.HTTPStatusError('', request=request, response=response)
        return wait_retry_after(fallback_strategy=lambda _: 1.0)(state)

    CONTRACTS.append(
        contract(
            'wait-retry-after',
            _wait_retry_after,
            _RETRY_AFTER_HEADERS,
            examples=[_OVERFLOWING_DATE, {'retry-after': '9' * 400}],
        )
    )

if evals_available():
    CONTRACTS.extend(
        [
            contract('evals-render-number', default_render_number, _NUMBERS, examples=[float('inf'), float('nan')]),
            contract('evals-render-percentage', default_render_percentage, st.floats(), examples=[float('inf')]),
            contract(
                'evals-render-number-diff',
                lambda pair: default_render_number_diff(*pair),
                st.tuples(_NUMBERS, _NUMBERS),
                examples=[(1.5, float('inf')), (1.5, float('nan'))],
            ),
        ]
    )
