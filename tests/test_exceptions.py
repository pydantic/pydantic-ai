"""Tests for exception classes."""

import pickle
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_core import ErrorDetails

from pydantic_ai import ModelRetry, ToolFailed
from pydantic_ai.exceptions import (
    AgentRunError,
    ApprovalRequired,
    CallDeferred,
    ConcurrencyLimitExceeded,
    ContentFilterError,
    IncompleteToolCall,
    ModelAPIError,
    ModelHTTPError,
    ToolFailedError,
    ToolRetryError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import RetryPromptPart, ToolReturnPart


def test_tool_failed_pydantic_schema_accepts_instance() -> None:
    """The custom schema accepts Python instances and preserves its tagged JSON representation."""
    adapter = TypeAdapter(ToolFailed)
    error = ToolFailed('Disk full')

    assert adapter.validate_python(error) is error
    assert adapter.validate_json(adapter.dump_json(error)) == error
    assert adapter.json_schema() == {
        'properties': {
            'kind': {'const': 'tool-failed', 'title': 'Kind', 'type': 'string'},
            'message': {'title': 'Message', 'type': 'string'},
        },
        'required': ['message', 'kind'],
        'type': 'object',
    }


@pytest.mark.parametrize(
    'exc_factory',
    [
        lambda: ModelRetry('test'),
        lambda: ToolFailed('test'),
        lambda: CallDeferred(),
        lambda: ApprovalRequired(),
        lambda: UserError('test'),
        lambda: AgentRunError('test'),
        lambda: UnexpectedModelBehavior('test'),
        lambda: UsageLimitExceeded('test'),
        lambda: ModelAPIError('model', 'test message'),
        lambda: ModelHTTPError(500, 'model'),
        lambda: IncompleteToolCall('test'),
        lambda: ToolRetryError(RetryPromptPart(content='test', tool_name='test')),
    ],
    ids=[
        'ModelRetry',
        'ToolFailed',
        'CallDeferred',
        'ApprovalRequired',
        'UserError',
        'AgentRunError',
        'UnexpectedModelBehavior',
        'UsageLimitExceeded',
        'ModelAPIError',
        'ModelHTTPError',
        'IncompleteToolCall',
        'ToolRetryError',
    ],
)
def test_exceptions_hashable(exc_factory: Callable[[], Any]):
    """Test that all exception classes are hashable and usable as keys."""
    exc = exc_factory()

    # Does not raise TypeError
    _ = hash(exc)

    # Can be used in sets and dicts
    s = {exc}
    d = {exc: 'value'}

    assert exc in s
    assert d[exc] == 'value'


@pytest.mark.parametrize(
    'exc_factory,check_attrs',
    [
        (lambda: ModelRetry('retry msg'), {'message': 'retry msg'}),
        (lambda: ToolFailed('failed msg'), {'message': 'failed msg'}),
        (lambda: CallDeferred(), {'metadata': None}),
        (lambda: CallDeferred({'key': 'value'}), {'metadata': {'key': 'value'}}),
        (lambda: ApprovalRequired(), {'metadata': None}),
        (lambda: ApprovalRequired({'key': 'value'}), {'metadata': {'key': 'value'}}),
        (lambda: UserError('user error'), {'message': 'user error'}),
        (lambda: AgentRunError('agent error'), {'message': 'agent error'}),
        (
            lambda: UsageLimitExceeded('limit hit'),
            {
                'message': 'limit hit. Consider raising the limit, or see the docs on usage limits '
                'for budget-aware patterns: https://pydantic.dev/docs/ai/core-concepts/agent/#usage-limits'
            },
        ),
        (lambda: ConcurrencyLimitExceeded('too many'), {'message': 'too many'}),
        (lambda: UnexpectedModelBehavior('unexpected'), {'message': 'unexpected', 'body': None}),
        (
            lambda: UnexpectedModelBehavior('unexpected', 'response body'),
            {'message': 'unexpected', 'body': 'response body'},
        ),
        (lambda: ContentFilterError('filtered'), {'message': 'filtered', 'body': None}),
        (lambda: ModelAPIError('gpt-4', 'api failed'), {'model_name': 'gpt-4', 'message': 'api failed'}),
        (
            lambda: ModelHTTPError(500, 'gpt-4'),
            {'status_code': 500, 'model_name': 'gpt-4', 'body': None, 'headers': None},
        ),
        (
            lambda: ModelHTTPError(429, 'gpt-4', {'error': 'rate limit'}),
            {'status_code': 429, 'model_name': 'gpt-4', 'body': {'error': 'rate limit'}, 'headers': None},
        ),
        (
            lambda: ModelHTTPError(429, 'gpt-4', headers={'Retry-After': '60', 'X-Request-Id': 'abc'}),
            {
                'status_code': 429,
                'model_name': 'gpt-4',
                'body': None,
                'headers': {'retry-after': '60', 'x-request-id': 'abc'},
            },
        ),
        (
            lambda: ModelHTTPError(404, 'gpt-5x', suggested_model_id='openai:gpt-5'),
            {
                'status_code': 404,
                'model_name': 'gpt-5x',
                'body': None,
                'headers': None,
                'suggested_model_id': 'openai:gpt-5',
            },
        ),
        (lambda: IncompleteToolCall('incomplete'), {'message': 'incomplete', 'body': None}),
    ],
    ids=[
        'ModelRetry',
        'ToolFailed',
        'CallDeferred-no-metadata',
        'CallDeferred-with-metadata',
        'ApprovalRequired-no-metadata',
        'ApprovalRequired-with-metadata',
        'UserError',
        'AgentRunError',
        'UsageLimitExceeded',
        'ConcurrencyLimitExceeded',
        'UnexpectedModelBehavior-no-body',
        'UnexpectedModelBehavior-with-body',
        'ContentFilterError',
        'ModelAPIError',
        'ModelHTTPError-no-body',
        'ModelHTTPError-with-body',
        'ModelHTTPError-with-headers',
        'ModelHTTPError-with-model-suggestion',
        'IncompleteToolCall',
    ],
)
def test_exceptions_pickle_round_trip(exc_factory: Callable[[], Exception], check_attrs: dict[str, Any]):
    """Test that exception classes survive pickle round-trip with all attributes preserved."""
    exc = exc_factory()
    restored = pickle.loads(pickle.dumps(exc))

    assert type(restored) is type(exc)
    assert str(restored) == str(exc)
    for attr, expected in check_attrs.items():
        assert getattr(restored, attr) == expected


def test_tool_retry_error_pickle_round_trip():
    """Test that ToolRetryError survives pickle round-trip with tool_retry preserved."""
    part = RetryPromptPart(content='retry this', tool_name='my_tool')
    exc = ToolRetryError(part)
    restored = pickle.loads(pickle.dumps(exc))

    assert type(restored) is ToolRetryError
    assert str(restored) == str(exc)
    assert restored.tool_retry.content == 'retry this'
    assert restored.tool_retry.tool_name == 'my_tool'
    assert restored.tool_retry.tool_call_id == part.tool_call_id
    assert restored.tool_retry.timestamp == part.timestamp


def test_tool_failed_error_pickle_round_trip():
    """Test that ToolFailedError survives pickle round-trip with tool_failed preserved."""
    part = ToolReturnPart(content='tool failed', tool_name='my_tool', outcome='failed')
    exc = ToolFailedError(part)
    restored = pickle.loads(pickle.dumps(exc))

    assert type(restored) is ToolFailedError
    assert str(restored) == str(exc)
    assert restored.tool_failed.content == 'tool failed'
    assert restored.tool_failed.tool_name == 'my_tool'
    assert restored.tool_failed.tool_call_id == part.tool_call_id
    assert restored.tool_failed.timestamp == part.timestamp
    assert restored.tool_failed.outcome == 'failed'


def test_tool_failed_error_non_str_content():
    """ToolFailedError stringifies non-`str` content without the model-facing error wrapper."""
    part = ToolReturnPart(content={'code': 42, 'reason': 'disk full'}, tool_name='my_tool', outcome='failed')
    exc = ToolFailedError(part)

    assert str(exc) == part.model_response_str(wrap_if_error=False)
    restored = pickle.loads(pickle.dumps(exc))
    assert restored.tool_failed.content == {'code': 42, 'reason': 'disk full'}
    assert str(restored) == str(exc)


def test_tool_retry_error_str_with_string_content():
    """Test that ToolRetryError uses string content as message automatically."""
    part = RetryPromptPart(content='error from tool', tool_name='my_tool')
    error = ToolRetryError(part)
    assert str(error) == 'error from tool'


def test_tool_retry_error_str_with_error_details():
    """Test that ToolRetryError formats ErrorDetails automatically."""
    validation_error = ValidationError.from_exception_data(
        'Test', [{'type': 'string_type', 'loc': ('name',), 'input': 123}]
    )
    part = RetryPromptPart(content=validation_error.errors(include_url=False), tool_name='my_tool')
    error = ToolRetryError(part)

    assert str(error) == (
        "1 validation error for 'my_tool'\nname\n  Input should be a valid string [type=string_type, input_value=123]"
    )


def test_tool_retry_error_str_with_value_error_type():
    """Test that ToolRetryError handles value_error type without ctx.error.

    When ErrorDetails are serialized, the exception object in ctx is stripped.
    This test ensures we handle error types that normally require ctx.error.
    """
    # Simulate serialized ErrorDetails where ctx.error has been stripped
    error_details: list[ErrorDetails] = [
        {
            'type': 'value_error',
            'loc': ('field',),
            'msg': 'Value error, must not be foo',
            'input': 'foo',
        }
    ]
    part = RetryPromptPart(content=error_details, tool_name='my_tool')
    error = ToolRetryError(part)

    assert str(error) == (
        "1 validation error for 'my_tool'\nfield\n  Value error, must not be foo [type=value_error, input_value='foo']"
    )


def test_model_http_error_headers_normalized_to_lowercase():
    """Headers passed to ModelHTTPError are stored with lowercase keys.

    Providers return headers in various casings (e.g. httpx normalises to lowercase,
    but some SDKs may preserve server casing). Requiring callers to lowercase before
    access would be fragile, so we normalise on construction.
    """
    exc = ModelHTTPError(429, 'gpt-4', headers={'Retry-After': '60', 'X-Request-Id': 'abc'})
    assert exc.headers == {'retry-after': '60', 'x-request-id': 'abc'}
    # Access is case-insensitive only on the stored lowercase keys
    assert exc.headers is not None
    assert exc.headers.get('retry-after') == '60'


def test_model_http_error_headers_default_none():
    """headers defaults to None when not provided, keeping existing call-sites unchanged."""
    exc = ModelHTTPError(500, 'gpt-4')
    assert exc.headers is None


def test_model_http_error_headers_none_explicit():
    """Passing headers=None is equivalent to omitting it."""
    exc = ModelHTTPError(500, 'gpt-4', headers=None)
    assert exc.headers is None


def test_model_http_error_headers_does_not_change_message():
    """Adding headers must not alter the existing str() / message format.

    Several places in the test suite — and downstream user code — pattern-match
    on the message string, so this must stay stable.
    """
    without = ModelHTTPError(429, 'gpt-4')
    with_headers = ModelHTTPError(429, 'gpt-4', headers={'retry-after': '60'})
    assert str(without) == str(with_headers)
    assert without.message == with_headers.message


def test_model_http_error_retry_after_delta_seconds():
    """retry_after parses an integer delta-seconds Retry-After value."""
    exc = ModelHTTPError(429, 'gpt-4', headers={'retry-after': '42'})
    assert exc.retry_after == 42.0


def test_model_http_error_retry_after_missing():
    """retry_after returns None when no Retry-After header is present."""
    exc = ModelHTTPError(429, 'gpt-4', headers={'x-request-id': 'abc'})
    assert exc.retry_after is None


def test_model_http_error_retry_after_no_headers():
    """retry_after returns None when headers is None."""
    exc = ModelHTTPError(429, 'gpt-4')
    assert exc.retry_after is None


def test_model_http_error_retry_after_http_date():
    """retry_after parses an HTTP-date Retry-After value into a non-negative float.

    We can't assert the exact value without freezing time, so we just check it's
    a non-negative float (the date is far in the future).
    """
    # Wed, 01 Jan 2099 00:00:00 GMT — always in the future
    exc = ModelHTTPError(429, 'gpt-4', headers={'retry-after': 'Thu, 01 Jan 2099 00:00:00 GMT'})
    result = exc.retry_after
    assert result is not None
    assert result > 0


def test_model_http_error_retry_after_unparseable():
    """retry_after returns None for a Retry-After value it cannot parse."""
    exc = ModelHTTPError(429, 'gpt-4', headers={'retry-after': 'not-a-number-or-date'})
    assert exc.retry_after is None


def test_model_http_error_retry_after_negative():
    """retry_after returns None for a negative Retry-After value.

    Negative delta-seconds are not defined by RFC 9110 — a server that sends
    Retry-After: -1 is misbehaving, and we must not propagate a negative wait
    time to callers who would sleep for a negative duration.
    """
    exc = ModelHTTPError(429, 'gpt-4', headers={'retry-after': '-1'})
    assert exc.retry_after is None


def test_model_http_error_retry_after_overflow():
    """retry_after returns None for an astronomically large integer Retry-After.

    float(int(very_large_string)) raises OverflowError in Python when the integer
    cannot be represented as a finite float. The except clause must cover it so
    callers always receive None rather than an unhandled exception.
    """
    # 10^309 cannot be represented as a finite double
    huge = '1' + '0' * 309
    exc = ModelHTTPError(429, 'gpt-4', headers={'retry-after': huge})
    assert exc.retry_after is None


def test_model_http_error_retry_after_http_date_asctime():
    """retry_after handles the asctime HTTP-date format (RFC 9110 §5.6.7 obs-date).

    Python's parsedate_to_datetime returns a *naive* datetime for the asctime
    format because the string carries no timezone. Without the fix the subtraction
    from an aware datetime.now(UTC) raises TypeError which is caught and silently
    returns None — a false negative. The fix normalises the naive datetime to UTC
    before computing the wait, so a future asctime date yields a positive float.
    """
    # Far-future date so the wait is always positive regardless of when the test runs.
    exc = ModelHTTPError(429, 'gpt-4', headers={'retry-after': 'Sun Nov  6 08:49:37 2099'})
    result = exc.retry_after
    assert result is not None
    assert result > 0


def test_model_http_error_headers_provider_openai():
    """Headers from an openai.APIStatusError land on ModelHTTPError.

    This is a unit test — not a VCR test — because the header propagation path
    lives in our own _map_api_errors helper, not in recorded API behaviour.
    """
    openai = pytest.importorskip('openai', reason='openai extra not installed')
    import httpx

    from pydantic_ai.models.openai import _map_api_errors  # pyright: ignore[reportPrivateUsage]

    req = httpx.Request('POST', 'https://api.openai.com/v1/chat/completions')
    resp = httpx.Response(429, headers={'retry-after': '30', 'x-request-id': 'rid-1'}, request=req)
    sdk_exc = openai.RateLimitError('Rate limited', response=resp, body=None)

    with pytest.raises(ModelHTTPError) as exc_info:
        with _map_api_errors('gpt-4o'):
            raise sdk_exc

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.headers is not None
    assert exc.headers.get('retry-after') == '30'
    assert exc.headers.get('x-request-id') == 'rid-1'
    assert exc.retry_after == 30.0


@pytest.mark.parametrize(
    'model_name',
    ['gpt-5', 'claude-sonet-4-5'],
    ids=['requested-known-model', 'different-provider-model'],
)
def test_model_http_error_does_not_suggest_an_unusable_match(model_name: str):
    """Provider access errors and custom-endpoint IDs cannot produce a useful close match."""
    openai = pytest.importorskip('openai', reason='openai extra not installed')
    import httpx

    from pydantic_ai.models.openai import _map_api_errors  # pyright: ignore[reportPrivateUsage]

    req = httpx.Request('POST', 'https://example.com/v1/responses')
    resp = httpx.Response(404, request=req)
    sdk_exc = openai.NotFoundError(
        'Model unavailable',
        response=resp,
        body={'code': 'model_not_found', 'message': 'Model unavailable'},
    )

    with pytest.raises(ModelHTTPError) as exc_info:
        with _map_api_errors(model_name):
            raise sdk_exc

    assert exc_info.value.suggested_model_id is None
    assert 'Did you mean' not in str(exc_info.value)


def test_model_http_error_headers_provider_anthropic():
    """Headers from an anthropic.APIStatusError land on ModelHTTPError."""
    anthropic = pytest.importorskip('anthropic', reason='anthropic extra not installed')
    import httpx2

    from pydantic_ai.models.anthropic import _map_api_errors  # pyright: ignore[reportPrivateUsage]

    req = httpx2.Request('POST', 'https://api.anthropic.com/v1/messages')
    resp = httpx2.Response(
        429,
        headers={'retry-after': '10', 'anthropic-ratelimit-tokens-remaining': '0'},
        request=req,
    )
    sdk_exc = anthropic.RateLimitError(message='Rate limited', response=resp, body=None)

    with pytest.raises(ModelHTTPError) as exc_info:
        with _map_api_errors('claude-sonnet-4-5'):
            raise sdk_exc

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.headers is not None
    assert exc.headers.get('retry-after') == '10'
    assert exc.retry_after == 10.0


def test_model_http_error_headers_provider_bedrock():
    """Headers from a botocore.ClientError land on ModelHTTPError."""
    pytest.importorskip('botocore', reason='botocore (bedrock extra) not installed')
    from botocore.exceptions import ClientError

    from pydantic_ai.models.bedrock import _map_api_errors  # pyright: ignore[reportPrivateUsage]

    error_response: Any = {
        'Error': {'Code': 'ThrottlingException', 'Message': 'Too many requests'},
        'ResponseMetadata': {
            'HTTPStatusCode': 429,
            'HTTPHeaders': {'retry-after': '5', 'x-amzn-requestid': 'req-abc'},
        },
    }
    sdk_exc = ClientError(error_response, 'InvokeModel')

    with pytest.raises(ModelHTTPError) as exc_info:
        with _map_api_errors('amazon.nova-pro-v1:0'):
            raise sdk_exc

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.headers is not None
    assert exc.headers.get('retry-after') == '5'
    assert exc.retry_after == 5.0


def test_model_http_error_headers_provider_xai_no_headers():
    """xAI errors are gRPC-based: no HTTP response headers, so ModelHTTPError.headers is None."""
    grpc = pytest.importorskip('grpc', reason='grpcio (xai extra) not installed')

    from pydantic_ai.models.xai import _map_api_errors  # pyright: ignore[reportPrivateUsage]

    class _FakeRpcError(grpc.RpcError):
        def code(self) -> Any:  # grpc.StatusCode only known at runtime
            return grpc.StatusCode.RESOURCE_EXHAUSTED

        def details(self) -> str:
            return 'quota exceeded'

    with pytest.raises(ModelHTTPError) as exc_info:
        with _map_api_errors('grok-3'):
            raise _FakeRpcError()

    exc = exc_info.value
    assert exc.status_code == 429
    assert exc.headers is None
    assert exc.retry_after is None
