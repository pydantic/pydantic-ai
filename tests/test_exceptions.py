"""Tests for exception classes."""

import pickle
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from pydantic_ai import ModelRetry
from pydantic_ai.exceptions import (
    AgentRunError,
    ApprovalRequired,
    CallDeferred,
    ConcurrencyLimitExceeded,
    ContentFilterError,
    IncompleteToolCall,
    ModelAPIError,
    ModelHTTPError,
    ToolRetryError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
    UserError,
)
from pydantic_ai.messages import RetryPromptPart


@pytest.mark.parametrize(
    'exc_factory',
    [
        lambda: ModelRetry('test'),
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
        (lambda: CallDeferred(), {'metadata': None}),
        (lambda: CallDeferred({'key': 'value'}), {'metadata': {'key': 'value'}}),
        (lambda: ApprovalRequired(), {'metadata': None}),
        (lambda: ApprovalRequired({'key': 'value'}), {'metadata': {'key': 'value'}}),
        (lambda: UserError('user error'), {'message': 'user error'}),
        (lambda: AgentRunError('agent error'), {'message': 'agent error'}),
        (lambda: UsageLimitExceeded('limit hit'), {'message': 'limit hit'}),
        (lambda: ConcurrencyLimitExceeded('too many'), {'message': 'too many'}),
        (lambda: UnexpectedModelBehavior('unexpected'), {'message': 'unexpected', 'body': None}),
        (
            lambda: UnexpectedModelBehavior('unexpected', 'response body'),
            {'message': 'unexpected', 'body': 'response body'},
        ),
        (lambda: ContentFilterError('filtered'), {'message': 'filtered', 'body': None}),
        (lambda: ModelAPIError('gpt-4', 'api failed'), {'model_name': 'gpt-4', 'message': 'api failed'}),
        (lambda: ModelHTTPError(500, 'gpt-4'), {'status_code': 500, 'model_name': 'gpt-4', 'body': None}),
        (
            lambda: ModelHTTPError(429, 'gpt-4', {'error': 'rate limit'}),
            {'status_code': 429, 'model_name': 'gpt-4', 'body': {'error': 'rate limit'}},
        ),
        (lambda: IncompleteToolCall('incomplete'), {'message': 'incomplete', 'body': None}),
    ],
    ids=[
        'ModelRetry',
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
