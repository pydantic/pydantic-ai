"""Tests for exception classes."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic import ValidationError

from pydantic_ai import ModelRetry
from pydantic_ai.exceptions import (
    AgentRunError,
    ApprovalRequired,
    CallDeferred,
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

    # Should contain the full Pydantic error with input value
    assert 'my_tool' in str(error)
    assert 'name' in str(error)
    assert 'input_value=123' in str(error)
