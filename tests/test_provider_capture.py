"""`provider_capture` reads each provider's request back in the voices the model receives it in."""

from __future__ import annotations

import pytest

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from .provider_capture import CAPTURED_PROVIDERS, CapturedProvider, capture

pytestmark = pytest.mark.anyio

HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[SystemPromptPart('SYSTEM_MARK'), UserPromptPart('USER_MARK')]),
    ModelResponse(parts=[TextPart('ASSISTANT_MARK'), ToolCallPart('fetch', {'url': 'x'}, tool_call_id='call_1')]),
    ModelRequest(parts=[ToolReturnPart('fetch', 'RESULT_MARK', tool_call_id='call_1')]),
]


@pytest.mark.parametrize('provider', CAPTURED_PROVIDERS)
async def test_each_voice_is_read_back(provider: CapturedProvider, allow_model_requests: None) -> None:
    captured = await capture(provider, HISTORY)
    assert {
        marker: captured.voices_of(marker) for marker in ('SYSTEM_MARK', 'USER_MARK', 'ASSISTANT_MARK', 'RESULT_MARK')
    } == {
        'SYSTEM_MARK': ['system'],
        'USER_MARK': ['user'],
        'ASSISTANT_MARK': ['assistant'],
        'RESULT_MARK': ['tool'],
    }
