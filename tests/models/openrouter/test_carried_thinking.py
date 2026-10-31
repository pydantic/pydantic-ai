"""Claude through OpenRouter's Responses API accepts reasoning carried in a user turn.

`tests/test_thinking_placement.py` pins where `OpenAIResponsesModel` puts a `ThinkingPart` it can't send
natively when the profile sets `mimics_assistant_message_formatting`: in a `user` message of its own,
ahead of the assistant message it came from. This is the live half: the request is the one the code builds,
read off the wire rather than out of the recording, and answered with a real 200, with the carried reasoning
directly behind a tool result.
"""

from __future__ import annotations as _annotations

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..._inline_snapshot import snapshot
from ...conftest import RequestCapture, try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]

_TOOL_CALL_ID = 'call_treasury_duration_1'
# Reasoning another model produced: it can't ride Claude's native channel, so it is carried as text.
_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content='What is the modified duration of the 10-year Treasury?')]),
    ModelResponse(
        parts=[ToolCallPart(tool_name='treasury_duration', args={'tenor_years': 10}, tool_call_id=_TOOL_CALL_ID)],
        model_name='gpt-5.6',
        provider_name='openai',
    ),
    ModelRequest(parts=[ToolReturnPart(tool_name='treasury_duration', content='8.1', tool_call_id=_TOOL_CALL_ID)]),
    ModelResponse(
        parts=[
            ThinkingPart(
                content='The lookup came back with 8.1 for the 10-year, so a 1% move in yields is worth roughly 8% of price.',
                provider_name='openai',
            ),
            TextPart(content='About 8.1 years, so a 1% rate move is worth roughly 8% of its price.'),
        ],
        model_name='gpt-5.6',
        provider_name='openai',
    ),
]


async def test_openrouter_responses_claude_accepts_carried_thinking(
    allow_model_requests: None, openrouter_api_key: str, request_capture: RequestCapture
):
    model = OpenAIResponsesModel(
        'anthropic/claude-sonnet-4.5',
        provider=OpenRouterProvider(api_key=openrouter_api_key, http_client=request_capture.client),
    )
    agent = Agent(model)

    @agent.tool_plain
    def treasury_duration(tenor_years: int) -> str:
        """Modified duration of the on-the-run Treasury at the given tenor."""
        return '8.1'  # pragma: no cover

    result = await agent.run(
        'In one sentence: does a longer duration mean more or less interest-rate risk?', message_history=_HISTORY
    )

    assert request_capture.body('/responses')['input'] == snapshot(
        [
            {'role': 'user', 'content': 'What is the modified duration of the 10-year Treasury?'},
            {
                'name': 'treasury_duration',
                'arguments': '{"tenor_years":10}',
                'call_id': 'call_treasury_duration_1',
                'type': 'function_call',
            },
            {'type': 'function_call_output', 'call_id': 'call_treasury_duration_1', 'output': '8.1'},
            {
                'role': 'user',
                'content': """\
<assistant_thinking>
The lookup came back with 8.1 for the 10-year, so a 1% move in yields is worth roughly 8% of price.
</assistant_thinking>\
""",
            },
            {'role': 'assistant', 'content': 'About 8.1 years, so a 1% rate move is worth roughly 8% of its price.'},
            {
                'role': 'user',
                'content': 'In one sentence: does a longer duration mean more or less interest-rate risk?',
            },
        ]
    )
    assert result.output == snapshot(
        "A longer duration means more interest-rate risk, as the bond's price will be more sensitive to changes in interest rates."
    )
