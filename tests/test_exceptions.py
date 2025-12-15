"""Tests for exception classes."""

from collections.abc import Callable
from typing import Any

import pytest
from pydantic_core import ErrorDetails

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


def test_tool_retry_error_str():
    """Test that ToolRetryError has a meaningful string representation."""
    part = RetryPromptPart(content='Invalid query syntax', tool_name='sql_query')
    error = ToolRetryError(part)
    assert str(error) == "Tool 'sql_query' failed: Invalid query syntax"


def test_tool_retry_error_str_with_error_details():
    """Test that ToolRetryError handles list of ErrorDetails content."""
    error_details: list[ErrorDetails] = [
        {'type': 'missing', 'loc': ('field1',), 'msg': 'Field required', 'input': {}},
        {'type': 'string_type', 'loc': ('field2',), 'msg': 'Input should be a valid string', 'input': 123},
    ]
    part = RetryPromptPart(content=error_details, tool_name='my_tool')
    error = ToolRetryError(part)

    expected = """\
Tool 'my_tool' failed: 2 validation errors
  field1: Field required
  field2: Input should be a valid string"""
    assert str(error) == expected
