"""A native tool call with no result block is dropped from the replayed payload.

Anthropic fails a whole request with `<tool> tool use with id ... was found without a corresponding
<tool>_tool_result block` when a `server_tool_use` or `mcp_tool_use` block goes unpaired. It makes one
exception, for the request that ends at the tool-result turn right after the call, where the result is
still in flight. That exception is what makes the bug survivable long enough to store: the turn is
accepted once, and every later request built from that history replays the unpaired call and fails.

Measured on `claude-sonnet-4-5` — the same history is accepted while the request ends at the
tool-result turn and rejected as soon as one more turn follows, with no reasoning or other content
involved. `_drop_unpaired_native_tool_calls` decides pairing on the blocks actually built, so a result
that never arrived and a result whose part didn't render both leave the call unpaired and both drop.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.messages import CachePoint, NativeToolSearchCallPart
from pydantic_ai.models import ModelRequestParameters

from ..._inline_snapshot import snapshot
from ...conftest import try_import
from ..conftest import AnthropicModelFactory, RequestCapture, message_shape

with try_import() as imports_successful:
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

_SEARCH_ID = 'srvtoolu_01EoSNE7k4dUJyGatASCV5qs'
_TOOL_CALL_ID = 'toolu_01WjXqPrN8vKsRt2YbLmZdQe'
_QUESTION = 'Look up the 10-year Treasury duration, then add 2 and 2.'
_FOLLOW_UP = 'In one sentence: does a longer duration mean more interest-rate risk?'

_SEARCH_CALL = NativeToolCallPart(
    tool_name='web_search',
    args={'query': '10-year Treasury modified duration'},
    tool_call_id=_SEARCH_ID,
    provider_name='anthropic',
)
_RENDERABLE_RESULT = [
    {
        'type': 'web_search_result',
        'url': 'https://example.com/treasury',
        'title': 'Treasury duration',
        'encrypted_content': 'encrypted',
    }
]


def _search_return(content: object) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        tool_name='web_search', content=content, tool_call_id=_SEARCH_ID, provider_name='anthropic'
    )


# The conversation continues past the tool-result turn, which is what turns a survivable in-flight call
# into a permanently unsendable history.
def _continued_history(*response_parts: object) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
        ModelResponse(
            parts=[
                *response_parts,  # pyright: ignore[reportArgumentType]
                ToolCallPart(tool_name='add', args={'a': 2, 'b': 2}, tool_call_id=_TOOL_CALL_ID),
            ]
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='add', content='4', tool_call_id=_TOOL_CALL_ID)]),
        ModelResponse(parts=[TextPart(content='It is 4.')]),
    ]


@dataclass
class Case:
    id: str
    history: list[ModelMessage]
    expected: list[tuple[str, list[str]]]
    follow_up: bool = True

    def __str__(self) -> str:
        return self.id


CASES = [
    Case(
        'result-never-arrived-drops-the-call',
        _continued_history(_SEARCH_CALL),
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['tool_use']),
                ('user', ['tool_result']),
                ('assistant', ['text']),
                ('user', ['text']),
            ]
        ),
    ),
    Case(
        # A history processor that trims a large search payload to a string leaves a return part with no
        # block shape here, so the result is silently skipped and the call is left unpaired on the wire.
        'unrenderable-result-drops-the-call',
        _continued_history(_SEARCH_CALL, _search_return('[search results trimmed]')),
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['tool_use']),
                ('user', ['tool_result']),
                ('assistant', ['text']),
                ('user', ['text']),
            ]
        ),
    ),
    Case(
        'paired-call-is-kept',
        _continued_history(_SEARCH_CALL, _search_return(_RENDERABLE_RESULT)),
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['server_tool_use', 'web_search_tool_result', 'tool_use']),
                ('user', ['tool_result']),
                ('assistant', ['text']),
                ('user', ['text']),
            ]
        ),
    ),
    Case(
        # Anthropic can deliver the result in the response after the one that called the tool, so pairing
        # is decided across the whole payload rather than within a turn.
        'result-in-a-later-response-is-kept',
        [
            ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
            ModelResponse(parts=[_SEARCH_CALL]),
            ModelResponse(parts=[_search_return(_RENDERABLE_RESULT), TextPart(content='It is 8.1.')]),
        ],
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['server_tool_use']),
                ('assistant', ['web_search_tool_result', 'text']),
                ('user', ['text']),
            ]
        ),
    ),
    Case(
        # Nothing else was in the turn, so dropping the call leaves no assistant message to send.
        'turn-holding-only-the-call-is-removed',
        [
            ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
            ModelResponse(parts=[_SEARCH_CALL]),
            ModelResponse(parts=[TextPart(content='It is 8.1.')]),
        ],
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['text']),
                ('user', ['text']),
            ]
        ),
    ),
    Case(
        # The result is still on its way, which is the one shape Anthropic takes an unpaired call in.
        # Dropping it here would break a pause-turn resume, whose whole point is to continue the call.
        'in-flight-call-is-kept',
        [
            ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
            ModelResponse(parts=[_SEARCH_CALL, ToolCallPart(tool_name='add', args={}, tool_call_id=_TOOL_CALL_ID)]),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content='4', tool_call_id=_TOOL_CALL_ID)]),
        ],
        follow_up=False,
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['server_tool_use', 'tool_use']),
                ('user', ['tool_result']),
            ]
        ),
    ),
    Case(
        # Bedrock rejects the in-flight shape the direct API tolerates, and a search is cheap to redo,
        # so tool search is the one native tool whose unpaired call drops even while in flight.
        'in-flight-tool-search-call-drops-anyway',
        [
            ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
            ModelResponse(
                parts=[
                    NativeToolSearchCallPart(
                        tool_call_id='srv_orphan', provider_name='anthropic', args={'queries': ['weather']}
                    ),
                    ToolCallPart(tool_name='add', args={}, tool_call_id=_TOOL_CALL_ID),
                ]
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content='4', tool_call_id=_TOOL_CALL_ID)]),
        ],
        follow_up=False,
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['tool_use']),
                ('user', ['tool_result']),
            ]
        ),
    ),
]


@pytest.mark.parametrize('case', CASES, ids=str)
async def test_drop_unpaired_native_tool_calls(case: Case):
    """The outbound payload never carries a native tool call without its result block.

    Asserted on the mapper's own output rather than through a cassette: Anthropic cassettes match on
    method and URI only, so a recorded request plays back green whether the call was dropped or not.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    history = [*case.history, *([ModelRequest(parts=[UserPromptPart(content=_FOLLOW_UP)])] if case.follow_up else [])]
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        history, ModelRequestParameters(), AnthropicModelSettings()
    )
    assert message_shape({'messages': [dict(message) for message in messages]}) == case.expected


async def test_dropped_call_keeps_the_cache_boundary():
    """A `CachePoint` authored on the dropped block lands on the block before it.

    Losing the breakpoint would silently re-process the tail on every later request, and moving it
    forward would cache content the user placed outside the boundary — both are invisible at runtime.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=[_QUESTION, CachePoint()])]),
        ModelResponse(parts=[_SEARCH_CALL]),
        ModelResponse(parts=[TextPart(content='It is 8.1.')]),
        ModelRequest(parts=[UserPromptPart(content=_FOLLOW_UP)]),
    ]
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        history, ModelRequestParameters(), AnthropicModelSettings()
    )
    assert [dict(message) for message in messages] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'Look up the 10-year Treasury duration, then add 2 and 2.',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                    }
                ],
            },
            {'role': 'assistant', 'content': [{'text': 'It is 8.1.', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [
                    {'text': 'In one sentence: does a longer duration mean more interest-rate risk?', 'type': 'text'}
                ],
            },
        ]
    )


async def test_dropped_call_hands_its_cache_boundary_to_the_block_before_it():
    """A breakpoint that landed *on* the dropped block moves back one block rather than vanishing.

    A `CachePoint` opening a user message has nothing to attach to there, so it attaches to the end of
    the previous message — which is the assistant turn holding the unpaired call. Dropping that block
    silently would take the breakpoint with it and re-process the tail on every later request.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
        ModelResponse(parts=[TextPart(content='Searching.'), _SEARCH_CALL]),
        ModelRequest(parts=[UserPromptPart(content=[CachePoint(), _FOLLOW_UP])]),
    ]
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        history, ModelRequestParameters(), AnthropicModelSettings()
    )
    assert [dict(message) for message in messages] == snapshot(
        [
            {
                'role': 'user',
                'content': [{'text': 'Look up the 10-year Treasury duration, then add 2 and 2.', 'type': 'text'}],
            },
            {
                'role': 'assistant',
                'content': [
                    {'text': 'Searching.', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}
                ],
            },
            {
                'role': 'user',
                'content': [
                    {'text': 'In one sentence: does a longer duration mean more interest-rate risk?', 'type': 'text'}
                ],
            },
        ]
    )


@pytest.mark.vcr
async def test_unpaired_native_tool_call_history_is_accepted(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
):
    """Anthropic answers a history that carries an unpaired native tool call.

    The live half: this exact history is rejected with `web_search tool use with id ... was found
    without a corresponding web_search_tool_result block` when the call is replayed, so the recorded
    200 is the assertion. The body read off the wire pins that the call is what went missing and the
    rest of the turn survived.
    """
    model: AnthropicModel = anthropic_model('claude-sonnet-4-5', capture=True)
    agent = Agent(model)

    @agent.tool_plain
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return '4'  # pragma: no cover

    result = await agent.run(_FOLLOW_UP, message_history=_continued_history(_SEARCH_CALL))

    body = request_capture.body('/v1/messages')
    assert message_shape(body) == snapshot(
        [
            ('user', ['text']),
            ('assistant', ['tool_use']),
            ('user', ['tool_result']),
            ('assistant', ['text']),
            ('user', ['text']),
        ]
    )
    assert result.output == snapshot(
        "Yes, a longer duration means more interest-rate risk because the bond's price will be more sensitive to changes in interest rates."
    )
