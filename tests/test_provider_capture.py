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
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from .provider_capture import CAPTURED_PROVIDERS, CapturedProvider, capture

HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[SystemPromptPart('SYSTEM_MARK'), UserPromptPart('USER_MARK')]),
    ModelResponse(parts=[TextPart('ASSISTANT_MARK'), ToolCallPart('fetch', {'url': 'x'}, tool_call_id='call_1')]),
    ModelRequest(parts=[ToolReturnPart('fetch', {'text': 'RESULT_MARK "quoted" é', 'n': 1}, tool_call_id='call_1')]),
]


PARAMS = ModelRequestParameters(
    function_tools=[
        ToolDefinition(
            name='fetch',
            description='TOOL_MARK',
            parameters_json_schema={
                'type': 'object',
                'properties': {'url': {'type': 'string', 'description': 'ARG_MARK'}},
            },
        )
    ]
)


@pytest.mark.parametrize('provider', CAPTURED_PROVIDERS)
async def test_each_voice_is_read_back(provider: CapturedProvider, allow_model_requests: None) -> None:
    """Each provider's system, user, assistant, tool-result and tool-definition text comes back in its voice.

    Not a VCR test: `capture` answers the request offline, and what it checks is the request the model built.
    """
    captured = await capture(provider, HISTORY, PARAMS)
    markers = ('SYSTEM_MARK', 'USER_MARK', 'ASSISTANT_MARK', 'RESULT_MARK', 'TOOL_MARK', 'ARG_MARK')
    assert {marker: captured.voices_of(marker) for marker in markers} == {
        'SYSTEM_MARK': ['system'],
        'USER_MARK': ['user'],
        'ASSISTANT_MARK': ['assistant'],
        'RESULT_MARK': ['tool'],
        'TOOL_MARK': ['system'],
        'ARG_MARK': ['system'],
    }
