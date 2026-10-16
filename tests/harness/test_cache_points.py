"""Tests for the shared tail-reminder cache breakpoint helper."""

from __future__ import annotations

import pytest

from pydantic_ai.messages import (
    BinaryContent,
    CachePoint,
    ModelRequestPart,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness._cache_points import has_cache_point_target


@pytest.mark.parametrize(
    ('parts', 'expected'),
    [
        pytest.param([UserPromptPart('goal')], True, id='user-prompt'),
        pytest.param([ToolReturnPart('tool', 'ok', tool_call_id='c1')], True, id='tool-return'),
        pytest.param([RetryPromptPart('try again', tool_call_id='c1')], True, id='retry-prompt'),
        pytest.param(
            [UserPromptPart([BinaryContent(data=b'\x00', media_type='image/png')])], True, id='binary-content'
        ),
        pytest.param([UserPromptPart([CachePoint(), 'goal'])], True, id='text-after-cache-point'),
        pytest.param([UserPromptPart([TextContent('goal')])], True, id='text-content'),
        pytest.param([UserPromptPart([TextContent('')])], False, id='empty-text-content'),
        pytest.param([UserPromptPart(''), UserPromptPart([''])], False, id='empty-user-prompts'),
        pytest.param([UserPromptPart([CachePoint()])], False, id='cache-point-only'),
        pytest.param([SystemPromptPart('be terse')], False, id='system-prompt-only'),
    ],
)
def test_has_cache_point_target(parts: list[ModelRequestPart], expected: bool) -> None:
    assert has_cache_point_target(parts) is expected
