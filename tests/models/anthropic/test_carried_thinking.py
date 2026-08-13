"""Anthropic accepts the wire shape a carried-over `ThinkingPart` produces.

`tests/test_foreign_thinking_placement.py` pins *where* the mapper puts an unsigned or foreign
`ThinkingPart` when the profile carries `mimics_assistant_message_formatting`: in a `user` message of its
own, ahead of the assistant turn it was produced in. That placement is only safe if Anthropic takes it,
and the hardest shape it can produce is this one — two consecutive `user` messages, the first holding a
`tool_result` block and the second plain text, which is what a tool-using agent hits on every step after
the first. Anthropic combines consecutive same-role turns, so the carried reasoning ends up directly
behind a tool result inside a single user message.

This is the live half of that pin: the request is built by `AnthropicModel` itself, read off the wire
rather than out of the recording, and answered with a real 200.
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
from ...conftest import try_import
from ..conftest import AnthropicModelFactory, RequestCapture

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

_QUESTION = 'What is the modified duration of the 10-year Treasury?'
_REASONING = (
    'Duration measures price sensitivity to rates, and the lookup came back with 8.1 for the 10-year, so a '
    '1% move in yields is worth roughly 8% of price.'
)
_ANSWER = 'About 8.1 years, so a 1% rate move is worth roughly 8% of its price.'
_FOLLOW_UP = 'In one sentence: does a longer duration mean more or less interest-rate risk?'
_TOOL_CALL_ID = 'toolu_01WjXqPrN8vKsRt2YbLmZdQe'

# The tool call resolves within the history, so the turn ahead of the reasoning renders as a user message
# holding a `tool_result` block. The `ThinkingPart` is unsigned and carries no `provider_name` — the shape
# left by storage round-trips, history processors, and other models in a `FallbackModel` chain — so it
# can't ride the native reasoning channel and is carried in the user turn instead.
_HISTORY: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
    ModelResponse(
        parts=[ToolCallPart(tool_name='treasury_duration', args={'tenor_years': 10}, tool_call_id=_TOOL_CALL_ID)]
    ),
    ModelRequest(parts=[ToolReturnPart(tool_name='treasury_duration', content='8.1', tool_call_id=_TOOL_CALL_ID)]),
    ModelResponse(parts=[ThinkingPart(content=_REASONING), TextPart(content=_ANSWER)]),
]


async def test_carried_thinking_accepted_after_tool_result(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
):
    """Anthropic answers a request whose carried reasoning follows a `tool_result` in a user turn of its own.

    The snapshot is the body as sent, so it also pins that the reasoning is *not* in the assistant turn:
    that turn carries only the answer text the user saw. The recorded 200 behind `result.output` is the
    point — the placement the mapper chose is one the API accepts.
    """
    model: AnthropicModel = anthropic_model('claude-sonnet-4-5', capture=True)
    agent = Agent(model)

    @agent.tool_plain
    def treasury_duration(tenor_years: int) -> str:
        """Modified duration of the on-the-run Treasury at the given tenor."""
        return '8.1'  # pragma: no cover

    result = await agent.run(_FOLLOW_UP, message_history=_HISTORY)

    assert request_capture.body('/v1/messages')['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [{'text': 'What is the modified duration of the 10-year Treasury?', 'type': 'text'}],
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe',
                        'type': 'tool_use',
                        'name': 'treasury_duration',
                        'input': {'tenor_years': 10},
                    }
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'tool_use_id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe',
                        'type': 'tool_result',
                        'content': [{'text': '8.1', 'type': 'text'}],
                        'is_error': False,
                    }
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'text': """\
<thinking>
Duration measures price sensitivity to rates, and the lookup came back with 8.1 for the 10-year, so a 1% move in yields is worth roughly 8% of price.
</thinking>\
""",
                        'type': 'text',
                    }
                ],
            },
            {
                'role': 'assistant',
                'content': [
                    {'text': 'About 8.1 years, so a 1% rate move is worth roughly 8% of its price.', 'type': 'text'}
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'In one sentence: does a longer duration mean more or less interest-rate risk?',
                        'type': 'text',
                    }
                ],
            },
        ]
    )
    assert result.output == snapshot(
        "A longer duration means more interest-rate risk, as the bond's price will be more sensitive to changes in interest rates."
    )
