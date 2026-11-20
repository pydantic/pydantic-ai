"""A tool call ID from another provider that Anthropic rejects is sent as a hash of itself.

Anthropic requires `tool_use.id` (and so `tool_result.tool_use_id`) to match `^[a-zA-Z0-9_-]+$`.
Other providers' IDs needn't: Gemini's, as an OpenAI-compatible gateway returns them, carry a base64
thought signature with `+`, `/` and `=`. A history with such a call, say from a `FallbackModel` turn,
failed every later Anthropic request with `messages.N.content.0.tool_use.id: String should match
pattern '^[a-zA-Z0-9_-]+$'`, measured on `claude-haiku-5-5`; the same request with the ID made of
allowed characters is accepted.
"""

from __future__ import annotations as _annotations

import re

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from ...conftest import try_import
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

import pytest

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

GEMINI_TOOL_CALL_ID = 'call_5076186143709904954__thought__Eo0BCooBAWkUfRNz+1fttZAYkE8BYwJhG1k/gPE091hCYqFxkg0A='
ANTHROPIC_TOOL_CALL_ID = 'toolu_01A09q90qw90lq917835lq9'


def _history(tool_call_id: str) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content='Is my bus late?')]),
        ModelResponse(
            parts=[ToolCallPart(tool_name='ride_status', args={'ride': 1}, tool_call_id=tool_call_id)],
            model_name='gemini-3.1-flash-lite',
        ),
        ModelRequest(
            parts=[ToolReturnPart(tool_name='ride_status', content={'status': 'late'}, tool_call_id=tool_call_id)]
        ),
        ModelResponse(parts=[TextPart(content='It is 5 minutes late.')], model_name='gemini-3.1-flash-lite'),
    ]


def _sent_tool_ids(history: list[ModelMessage]) -> list[str]:
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='On its way.', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    )
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model)

    @agent.tool_plain
    def ride_status(ride: int) -> dict[str, str]:
        raise NotImplementedError  # pragma: no cover

    agent.run_sync('And now?', message_history=history)
    (kwargs,) = get_mock_chat_completion_kwargs(mock_client)
    return [
        block.get('id') or block['tool_use_id']
        for message in kwargs['messages']
        if isinstance(message['content'], list)
        for block in message['content']
        if block['type'] in ('tool_use', 'tool_result')
    ]


def test_foreign_tool_call_id_is_hashed(allow_model_requests: None) -> None:
    history = _history(GEMINI_TOOL_CALL_ID)
    tool_use_id, tool_result_id = _sent_tool_ids(history)

    assert re.fullmatch(r'[a-zA-Z0-9_-]+', tool_use_id)
    assert tool_result_id == tool_use_id
    # The same history maps to the same request.
    assert _sent_tool_ids(history) == [tool_use_id, tool_use_id]
    # The history itself keeps the original ID.
    assert history[1].parts[0].tool_call_id == GEMINI_TOOL_CALL_ID  # type: ignore[union-attr]


def test_valid_tool_call_id_is_sent_unchanged(allow_model_requests: None) -> None:
    assert _sent_tool_ids(_history(ANTHROPIC_TOOL_CALL_ID)) == [ANTHROPIC_TOOL_CALL_ID, ANTHROPIC_TOOL_CALL_ID]
