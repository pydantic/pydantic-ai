"""A native tool call with no result block is dropped from the replayed payload.

Anthropic fails a whole request with `<tool> tool use with id ... was found without a corresponding
<tool>_tool_result block` when a `server_tool_use` or `mcp_tool_use` block goes unpaired. It makes one
exception while every message after the call is a user turn containing only concurrent client-tool
results, where the native result is still in flight. That exception is what makes the bug survivable
long enough to store: the turn is accepted once, and a later non-result message makes replay fail.

Measured on `claude-sonnet-4-5` — the same history is accepted while every turn after the call
carries nothing but tool results, and rejected as soon as a turn with any other content follows, with
no reasoning involved. Pairing is decided on the blocks actually built, so a result that never
arrived and a result whose part didn't render both leave the call unpaired and both drop.
"""

from __future__ import annotations as _annotations

from dataclasses import dataclass

import pytest
from pydantic import JsonValue

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
from pydantic_ai.capabilities import NativeTool, ToolSearch
from pydantic_ai.messages import CachePoint, ModelResponsePart, NativeToolSearchCallPart, ThinkingPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.native_tools import MCPServerTool, WebSearchTool

from ..._inline_snapshot import snapshot
from ...conftest import RequestCapture, try_import
from ..conftest import AnthropicModelFactory, message_shape
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic.types.beta import BetaMCPToolResultBlock, BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

_SEARCH_ID = 'srvtoolu_01EoSNE7k4dUJyGatASCV5qs'
_TOOL_CALL_ID = 'toolu_01WjXqPrN8vKsRt2YbLmZdQe'
_MCP_CALL_ID = 'mcptoolu_01AbCdEfGhIjKlMnOpQrStUv'
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


_MCP_CALL = NativeToolCallPart(
    tool_name=f'{MCPServerTool.kind}:docs',
    args={'tool_name': 'ask_question', 'tool_args': {'question': 'What is the 10-year Treasury duration?'}},
    tool_call_id=_MCP_CALL_ID,
    provider_name='anthropic',
)
_MCP_RETURN = NativeToolReturnPart(
    tool_name=f'{MCPServerTool.kind}:docs',
    content={'content': [{'type': 'text', 'text': '8.1'}], 'is_error': False},
    tool_call_id=_MCP_CALL_ID,
    provider_name='anthropic',
)


def _search_return(content: object) -> NativeToolReturnPart:
    return NativeToolReturnPart(
        tool_name='web_search', content=content, tool_call_id=_SEARCH_ID, provider_name='anthropic'
    )


# The conversation continues past the tool-result turn, which is what turns a survivable in-flight call
# into a permanently unsendable history.
def _continued_history(*response_parts: ModelResponsePart) -> list[ModelMessage]:
    return [
        ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
        ModelResponse(
            parts=[
                *response_parts,
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
    expected_call_ids: list[str] | None = None

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
        expected_call_ids=[_SEARCH_ID, _TOOL_CALL_ID],
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['server_tool_use', 'tool_use']),
                ('user', ['tool_result']),
            ]
        ),
    ),
    Case(
        # Two tool-result turns can follow the call when the model called two client tools and their
        # results came back separately. Measured as accepted too, so the exemption reads the whole
        # suffix rather than a single turn.
        'in-flight-call-is-kept-behind-two-tool-result-turns',
        [
            ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
            ModelResponse(
                parts=[
                    _SEARCH_CALL,
                    ToolCallPart(tool_name='add', args={}, tool_call_id=_TOOL_CALL_ID),
                    ToolCallPart(tool_name='double', args={}, tool_call_id='toolu_01DoUbLeCaLLiDeNtIfIeR00'),
                ]
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content='4', tool_call_id=_TOOL_CALL_ID)]),
            ModelRequest(
                parts=[ToolReturnPart(tool_name='double', content='8', tool_call_id='toolu_01DoUbLeCaLLiDeNtIfIeR00')]
            ),
        ],
        follow_up=False,
        expected_call_ids=[_SEARCH_ID, _TOOL_CALL_ID, 'toolu_01DoUbLeCaLLiDeNtIfIeR00'],
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['server_tool_use', 'tool_use', 'tool_use']),
                ('user', ['tool_result']),
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
        expected_call_ids=[_TOOL_CALL_ID],
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['tool_use']),
                ('user', ['tool_result']),
            ]
        ),
    ),
    Case(
        # `mcp_tool_use` is the other block type Anthropic pairs, and it fails the same way: an
        # unpaired one is rejected with `mcp_tool_use with id ... was found without a corresponding
        # mcp_tool_result block`, measured live against a control with the call removed.
        'mcp-result-never-arrived-drops-the-call',
        _continued_history(_MCP_CALL),
        expected_call_ids=[_TOOL_CALL_ID],
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
        'in-flight-mcp-call-is-kept',
        [
            ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
            ModelResponse(parts=[_MCP_CALL, ToolCallPart(tool_name='add', args={}, tool_call_id=_TOOL_CALL_ID)]),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content='4', tool_call_id=_TOOL_CALL_ID)]),
        ],
        follow_up=False,
        expected_call_ids=[_MCP_CALL_ID, _TOOL_CALL_ID],
        expected=snapshot(
            [
                ('user', ['text']),
                ('assistant', ['mcp_tool_use', 'tool_use']),
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
    rendered_messages: list[JsonValue] = []
    for message in messages:
        role = message['role']
        assert isinstance(role, str)
        content = message['content']
        assert isinstance(content, list)

        rendered_blocks: list[JsonValue] = []
        for block in content:
            assert isinstance(block, dict)
            block_type = block.get('type')
            assert isinstance(block_type, str)
            rendered_blocks.append({'type': block_type})
        rendered_messages.append({'role': role, 'content': rendered_blocks})

    assert message_shape({'messages': rendered_messages}) == case.expected
    if case.expected_call_ids is not None:
        call_ids: list[str] = []
        for message in messages:
            content = message['content']
            assert isinstance(content, list)
            for block in content:
                if isinstance(block, dict) and isinstance(block_id := block.get('id'), str):
                    call_ids.append(block_id)
        assert call_ids == case.expected_call_ids


async def test_mcp_call_answered_by_its_replayed_result_is_kept():
    """A replayed MCP result pairs its call, so neither block is dropped.

    Asserted whole rather than through `message_shape`, which subscripts blocks: the result comes back
    as an SDK model, and it is the `tool_use_id` on that model that has to match the call's `id`.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        _continued_history(_MCP_CALL, _MCP_RETURN), ModelRequestParameters(), AnthropicModelSettings()
    )
    assert [dict(message) for message in messages][1] == snapshot(
        {
            'role': 'assistant',
            'content': [
                {
                    'id': 'mcptoolu_01AbCdEfGhIjKlMnOpQrStUv',
                    'type': 'mcp_tool_use',
                    'server_name': 'docs',
                    'name': 'ask_question',
                    'input': {'question': 'What is the 10-year Treasury duration?'},
                },
                BetaMCPToolResultBlock(
                    content=[BetaTextBlock(text='8.1', type='text')],
                    is_error=False,
                    tool_use_id='mcptoolu_01AbCdEfGhIjKlMnOpQrStUv',
                    type='mcp_tool_result',
                ),
                {'id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe', 'type': 'tool_use', 'name': 'add', 'input': {'a': 2, 'b': 2}},
            ],
        }
    )


async def test_dropped_call_keeps_the_cache_boundary():
    """A breakpoint authored earlier in the payload stays where it is when the call's turn goes away.

    The turn holding the unpaired call carries no breakpoint of its own here, so removing it must not
    disturb one: moving the boundary forward would cache content the user placed outside it, and
    losing it would silently re-process the tail on every later request.
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


_THINKING = ThinkingPart(content='Searching for the duration.', signature='sig', provider_name='anthropic')


async def test_dropped_call_skips_a_block_that_cannot_carry_the_boundary():
    """The breakpoint walks back past a `thinking` block, which takes no `cache_control`.

    An assistant turn that reasons before calling a server tool renders `[thinking, server_tool_use]`,
    so the block right before the dropped one is routinely one that would raise if handed the boundary.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
        ModelResponse(parts=[TextPart(content='Searching.'), _THINKING, _SEARCH_CALL], provider_name='anthropic'),
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
                    {'text': 'Searching.', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
                    {'thinking': 'Searching for the duration.', 'signature': 'sig', 'type': 'thinking'},
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


async def test_dropped_call_hands_the_boundary_back_a_message_when_its_turn_cannot_carry_it():
    """A turn left holding only a `thinking` block passes the breakpoint to the message before it.

    Nothing in the assistant turn can carry `cache_control` once the call is gone, and dropping the
    breakpoint instead would silently re-process the whole tail on every later request.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
        ModelResponse(parts=[_THINKING, _SEARCH_CALL], provider_name='anthropic'),
        ModelRequest(parts=[UserPromptPart(content=[CachePoint(), _FOLLOW_UP])]),
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
            {
                'role': 'assistant',
                'content': [{'thinking': 'Searching for the duration.', 'signature': 'sig', 'type': 'thinking'}],
            },
            {
                'role': 'user',
                'content': [
                    {'text': 'In one sentence: does a longer duration mean more interest-rate risk?', 'type': 'text'}
                ],
            },
        ]
    )


async def test_dropped_call_that_opens_the_payload_gives_up_its_boundary():
    """A breakpoint with nothing before it to hold it is lost rather than failing the request.

    Nothing precedes the dropped block, so there is no cacheable block anywhere to walk back to. The
    `CachePoint` that authored the boundary lived on the block going away, so a raise here would fail
    a request over a breakpoint the user can no longer see.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelResponse(parts=[_SEARCH_CALL], provider_name='anthropic'),
        ModelRequest(parts=[UserPromptPart(content=[CachePoint(), _FOLLOW_UP])]),
    ]
    _, messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        history, ModelRequestParameters(), AnthropicModelSettings()
    )
    assert [dict(message) for message in messages] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': 'In one sentence: does a longer duration mean more interest-rate risk?', 'type': 'text'}
                ],
            }
        ]
    )


async def test_dropped_call_hands_its_boundary_across_a_turn_that_cannot_carry_it():
    """The search for a carrier crosses message boundaries, not just the dropped block's own turn.

    The message before the dropped one holds a lone `thinking` block, which takes no `cache_control`,
    so the nearest block that can carry the boundary is two turns back.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
        ModelResponse(parts=[_THINKING], provider_name='anthropic'),
        ModelResponse(parts=[_SEARCH_CALL], provider_name='anthropic'),
        ModelRequest(parts=[UserPromptPart(content=[CachePoint(), _FOLLOW_UP])]),
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
            {
                'role': 'assistant',
                'content': [{'thinking': 'Searching for the duration.', 'signature': 'sig', 'type': 'thinking'}],
            },
            {
                'role': 'user',
                'content': [
                    {'text': 'In one sentence: does a longer duration mean more interest-rate risk?', 'type': 'text'}
                ],
            },
        ]
    )


async def test_dropped_call_skips_a_block_that_is_not_a_mapping():
    """A replayed MCP result renders as a Pydantic model, which subscripting would raise on.

    It shares an assistant turn with a native call whenever the model used both, so the boundary search
    walks over blocks that aren't `dict`s at all, not just ones that can't carry `cache_control`.
    """
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key='x'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
        ModelResponse(
            parts=[
                NativeToolReturnPart(
                    tool_name=f'{MCPServerTool.kind}:docs',
                    content={'content': [{'type': 'text', 'text': '8.1'}], 'is_error': False},
                    tool_call_id=_MCP_CALL_ID,
                    provider_name='anthropic',
                ),
                _SEARCH_CALL,
            ],
            provider_name='anthropic',
        ),
        ModelRequest(parts=[UserPromptPart(content=[CachePoint(), _FOLLOW_UP])]),
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
            {
                'role': 'assistant',
                'content': [
                    BetaMCPToolResultBlock(
                        content=[BetaTextBlock(text='8.1', type='text')],
                        is_error=False,
                        tool_use_id='mcptoolu_01AbCdEfGhIjKlMnOpQrStUv',
                        type='mcp_tool_result',
                    )
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


async def test_orphaned_tool_search_call_is_dropped_through_an_agent_run(allow_model_requests: None):
    """The tool-search drop reaches the wire through an agent run, not only through the mapper.

    Anthropic occasionally emits a `tool_search_tool_*` server tool use alongside a client `tool_use`
    and ends the turn before delivering the corresponding result block
    (https://github.com/anthropics/anthropic-sdk-python/issues/1325), so this is the shape a
    `ToolSearch()` run actually stores. Reported by @kclisp on PR #5143.
    """
    mock_client = MockAnthropic.create_mock(
        completion_message([BetaTextBlock(text='ok', type='text')], BetaUsage(input_tokens=5, output_tokens=5))
    )
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model, capabilities=[ToolSearch()])

    @agent.tool_plain
    def send_status(message: str) -> str:  # pragma: no cover
        return 'ok'

    @agent.tool_plain(defer_loading=True)
    def pay_rent() -> str:  # pragma: no cover
        return 'paid'

    history: list[ModelMessage] = [
        ModelRequest.user_text_prompt('pay rent and send status'),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    tool_call_id='srv_orphan',
                    provider_name='anthropic',
                    args={'queries': ['pay.*']},
                    provider_details={'strategy': 'regex'},
                ),
                ToolCallPart(tool_name='send_status', args={'message': 'looking'}, tool_call_id='cl_1'),
            ],
            provider_name='anthropic',
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='send_status', content='ok', tool_call_id='cl_1')]),
    ]
    await agent.run('continue', message_history=history)

    assert message_shape(get_mock_chat_completion_kwargs(mock_client)[0]) == snapshot(
        [('user', ['text']), ('assistant', ['tool_use']), ('user', ['tool_result', 'text'])]
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
    assert body['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [{'text': 'Look up the 10-year Treasury duration, then add 2 and 2.', 'type': 'text'}],
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe',
                        'type': 'tool_use',
                        'name': 'add',
                        'input': {'a': 2, 'b': 2},
                    }
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'tool_use_id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe',
                        'type': 'tool_result',
                        'content': [{'text': '4', 'type': 'text'}],
                        'is_error': False,
                    }
                ],
            },
            {'role': 'assistant', 'content': [{'text': 'It is 4.', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [
                    {'text': 'In one sentence: does a longer duration mean more interest-rate risk?', 'type': 'text'}
                ],
            },
        ]
    )
    assert result.output == snapshot(
        "Yes, a longer duration means more interest-rate risk because the bond's price will be more sensitive to changes in interest rates."
    )


# The server name on the block has to match a declared MCP server, so the live call names the one
# `test_anthropic_mcp_servers` already records against.
_LIVE_MCP_CALL = NativeToolCallPart(
    tool_name=f'{MCPServerTool.kind}:deepwiki',
    args={
        'tool_name': 'ask_question',
        'tool_args': {'question': 'What is pydantic-ai?', 'repoName': 'pydantic/pydantic-ai'},
    },
    tool_call_id='mcptoolu_01SAss3KEwASziHZoMR6HcZU',
    provider_name='anthropic',
)


@pytest.mark.vcr
async def test_in_flight_mcp_call_history_is_accepted(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
):
    """Anthropic answers a history carrying an unpaired `mcp_tool_use` whose result is in flight.

    The in-flight exemption is written against the block type rather than one tool, and a continued
    conversation is what was measured as rejected for `mcp_tool_use`. This records the other side for
    the same block type, so keeping it rests on a 200 rather than on the `server_tool_use` result.
    """
    model: AnthropicModel = anthropic_model('claude-sonnet-4-5', capture=True)
    agent = Agent(model, capabilities=[NativeTool(MCPServerTool(id='deepwiki', url='https://mcp.deepwiki.com/mcp'))])

    @agent.tool_plain
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return '4'  # pragma: no cover

    result = await agent.run(
        message_history=[
            ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
            ModelResponse(
                parts=[
                    _LIVE_MCP_CALL,
                    ToolCallPart(tool_name='add', args={'a': 2, 'b': 2}, tool_call_id=_TOOL_CALL_ID),
                ]
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content='4', tool_call_id=_TOOL_CALL_ID)]),
        ]
    )

    body = request_capture.body('/v1/messages')
    assert body['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [{'text': 'Look up the 10-year Treasury duration, then add 2 and 2.', 'type': 'text'}],
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'id': 'mcptoolu_01SAss3KEwASziHZoMR6HcZU',
                        'type': 'mcp_tool_use',
                        'server_name': 'deepwiki',
                        'name': 'ask_question',
                        'input': {'question': 'What is pydantic-ai?', 'repoName': 'pydantic/pydantic-ai'},
                    },
                    {
                        'id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe',
                        'type': 'tool_use',
                        'name': 'add',
                        'input': {'a': 2, 'b': 2},
                    },
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'tool_use_id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe',
                        'type': 'tool_result',
                        'content': [{'text': '4', 'type': 'text'}],
                        'is_error': False,
                    }
                ],
            },
        ]
    )
    assert result.output == snapshot("""\
I apologize, but I don't have the ability to look up the 10-year Treasury duration. The tools available to me are limited to asking questions about GitHub repositories and performing basic mathematical operations.

However, I can tell you that **2 + 2 = 4**.

Regarding the 10-year Treasury duration, typically the duration of a 10-year U.S. Treasury bond is approximately 8-9 years (it's less than the maturity because duration accounts for the present value of all cash flows including coupon payments). However, the exact duration varies based on the current yield and coupon rate. You would need to check current financial data sources like Bloomberg, the U.S. Treasury website, or financial news outlets for the most accurate current duration figure.\
""")


@pytest.mark.vcr
async def test_in_flight_native_tool_call_history_is_accepted(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
):
    """Anthropic answers a history whose unpaired call is still in flight, so keeping it is safe.

    The kept side of the same claim the drop rests on. Dropping a call here would break a pause-turn
    resume, whose whole point is to continue it, and the payload still goes out unpaired — so the
    recorded 200 is the assertion, with the body read off the wire pinning that the call survived.
    """
    model: AnthropicModel = anthropic_model('claude-sonnet-4-5', capture=True)
    agent = Agent(model, capabilities=[NativeTool(WebSearchTool())])

    @agent.tool_plain
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return '4'  # pragma: no cover

    result = await agent.run(
        message_history=[
            ModelRequest(parts=[UserPromptPart(content=_QUESTION)]),
            ModelResponse(
                parts=[
                    _SEARCH_CALL,
                    ToolCallPart(tool_name='add', args={'a': 2, 'b': 2}, tool_call_id=_TOOL_CALL_ID),
                ]
            ),
            ModelRequest(parts=[ToolReturnPart(tool_name='add', content='4', tool_call_id=_TOOL_CALL_ID)]),
        ]
    )

    body = request_capture.body('/v1/messages')
    assert body['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [{'text': 'Look up the 10-year Treasury duration, then add 2 and 2.', 'type': 'text'}],
            },
            {
                'role': 'assistant',
                'content': [
                    {
                        'id': 'srvtoolu_01EoSNE7k4dUJyGatASCV5qs',
                        'type': 'server_tool_use',
                        'name': 'web_search',
                        'input': {'query': '10-year Treasury modified duration'},
                    },
                    {
                        'id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe',
                        'type': 'tool_use',
                        'name': 'add',
                        'input': {'a': 2, 'b': 2},
                    },
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'tool_use_id': 'toolu_01WjXqPrN8vKsRt2YbLmZdQe',
                        'type': 'tool_result',
                        'content': [{'text': '4', 'type': 'text'}],
                        'is_error': False,
                    }
                ],
            },
        ]
    )
    assert result.output == snapshot("""\
Based on the search results, the duration of a 10-year Treasury note is approximately 8.95 years (this example was given when yields were at 1.30%). However, it's important to note that the exact duration varies depending on current market conditions, coupon rates, and yield levels.

The duration typically ranges between 7 to 9 years for 10-year Treasury securities based on the examples found in the search results.

And 2 + 2 = **4**\
""")
