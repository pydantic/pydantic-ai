"""Mid-conversation system prompts on Anthropic.

The Anthropic Messages API accepts a `{'role': 'system'}` entry inside `messages`, so an application
can add an operator instruction partway through a conversation without rewriting the top-level
`system` parameter (and with it the whole cached prefix). Pydantic AI renders a `SystemPromptPart`
outside the first `ModelRequest` that way when the model and client support it, and keeps the
`<system>`-tagged user rendering everywhere else.

The API accepts the entry only directly behind a user turn and directly ahead of an assistant turn
(or the end of the array), so two transforms keep it legal without giving up its authority: it slides
past user turns that ended up behind it, and then, if nothing legal precedes where it landed, gets a
minimal `.` user turn to follow. These tests pin both, in that order — deciding the anchor first put
it between a `tool_use` and its `tool_result` — plus both sides of the `supports_inline_system_prompts`
profile flag, the client-transport gate, and the cache breakpoint that now lands on the new message.
"""

from __future__ import annotations as _annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from pydantic_ai import (
    Agent,
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import ToolDefinition

from .._inline_snapshot import snapshot
from ..cassette_utils import single_request_body
from ..conftest import try_import

if TYPE_CHECKING:
    from vcr.cassette import Cassette

with try_import() as imports_successful:
    from anthropic import AsyncAnthropicBedrock, AsyncAnthropicFoundry
    from anthropic.types.beta import BetaTextBlock, BetaToolUseBlock, BetaUsage

    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import (
        _INLINE_SYSTEM_PROMPT_UNSUPPORTED_CLIENTS,  # pyright: ignore[reportPrivateUsage]
        AnthropicModel,
        AnthropicModelSettings,
    )
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

INSTRUCTION = 'From now on, every suggestion must include explicit type annotations.'
_CACHE_PREFIX = 'Stable harness state for the mid-conversation cache test.\n' + '\n'.join(
    f'Fact {i:04d}: workspace file {i % 37} has revision {i}, owner {i % 11}, and status verified.' for i in range(50)
)


@pytest.fixture
def rendered_requests(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """The `system` parameter and `messages` array the adapter renders, per request, as it goes out.

    `single_request_body` reads the request stored in the *cassette*, and VCR is configured to match
    on method, path and host only, so a rendering change still replays its recording and keeps
    asserting the recorded body — the trap `tests/AGENTS.md` calls out. Every test below therefore
    also asserts that what the adapter produces today equals what the recording captured, which is
    what makes the snapshots evidence about live behavior rather than about the file.

    Both halves are captured, not just `messages`: the top-level `system` parameter is the cache
    section this feature exists to leave alone, so a change that emptied it or folded the instruction
    into it would be exactly the regression worth catching, and asserting only the recorded
    `body['system']` would miss it.

    The lists are captured by reference, after `_map_message` but before the caching passes run over
    them, so what the assertions see includes the `cache_control` blocks those passes add — the wire,
    not an intermediate.
    """
    rendered: list[dict[str, Any]] = []
    map_message = AnthropicModel._map_message  # pyright: ignore[reportPrivateUsage]

    async def capture(self: AnthropicModel, *args: Any, **kwargs: Any) -> Any:
        system_prompt, anthropic_messages = await map_message(self, *args, **kwargs)
        rendered.append({'system': system_prompt, 'messages': cast('list[dict[str, Any]]', anthropic_messages)})
        return system_prompt, anthropic_messages

    monkeypatch.setattr(AnthropicModel, '_map_message', capture)
    return rendered


@pytest.fixture
def anthropic_bedrock_client() -> AsyncAnthropicBedrock:
    """An `AsyncAnthropicBedrock` client authenticated by bearer token, defaulted for replay.

    The SDK reads `AWS_BEARER_TOKEN_BEDROCK` into `api_key`, and refuses it alongside SigV4
    credentials, so the token is the only auth this passes. `botocore` only ships under the
    `bedrock` extra.
    """
    pytest.importorskip('botocore')

    return AsyncAnthropicBedrock(
        api_key=os.environ.get('AWS_BEARER_TOKEN_BEDROCK', 'test-bedrock-token'),
        aws_region=os.environ.get('AWS_DEFAULT_REGION', 'us-east-1'),
    )


def message_history() -> list[ModelMessage]:
    """A reviewed-once conversation that a new operator instruction is appended to.

    The trailing request carries only the `SystemPromptPart`; the agent's own user prompt merges
    into it, which is the shape `ctx.enqueue(SystemPromptPart(...))` produces as well.
    """
    return [
        ModelRequest(
            parts=[
                SystemPromptPart(content='You are a code reviewer.'),
                UserPromptPart(content='Review `def add(a, b): return a + b`.'),
            ]
        ),
        ModelResponse(parts=[TextPart(content='Looks fine.')]),
        ModelRequest(parts=[SystemPromptPart(content=INSTRUCTION)]),
    ]


def message_history_with_paired_instruction() -> list[ModelMessage]:
    """The same conversation, with the instruction arriving alongside a user prompt.

    The pairing is what puts a native `system` entry directly behind the request's own user turn.
    Without one, the entry gets the `.` anchor instead (see `..._without_user_turn`), so these are
    the histories where `_place_system_messages_before_generation` decides where it ends up.
    """
    history = message_history()
    history[-1] = ModelRequest(
        parts=[SystemPromptPart(content=INSTRUCTION), UserPromptPart(content='Review it again.')]
    )
    return history


@dataclass(frozen=True)
class Case:
    """A model on one side of the `supports_inline_system_prompts` flag, and the wire it produces."""

    id: str
    model: str
    expected_messages: list[dict[str, Any]]


CASES = [
    Case(
        id='inline-system-supported',
        model='claude-opus-4-8',
        expected_messages=snapshot(
            [
                {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
                {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
                {'role': 'user', 'content': [{'text': 'Review it again.', 'type': 'text'}]},
                {
                    'role': 'system',
                    'content': [
                        {
                            'text': 'From now on, every suggestion must include explicit type annotations.',
                            'type': 'text',
                        }
                    ],
                },
            ]
        ),
    ),
    Case(
        id='inline-system-unsupported',
        model='claude-sonnet-4-6',
        expected_messages=snapshot(
            [
                {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
                {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
                {
                    'role': 'user',
                    'content': [
                        {
                            'text': '<system>From now on, every suggestion must include explicit type annotations.</system>',
                            'type': 'text',
                        },
                        {'text': 'Review it again.', 'type': 'text'},
                    ],
                },
            ]
        ),
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CASES])
async def test_mid_conversation_system_prompt(
    allow_model_requests: None,
    anthropic_api_key: str,
    vcr: Cassette,
    rendered_requests: list[dict[str, Any]],
    case: Case,
):
    """A non-leading `SystemPromptPart` gets its own `system` entry only on models that accept the role.

    Both cases send the same history, so the snapshots show the only difference: the supported model
    gets a fourth `{'role': 'system'}` message, the unsupported one gets the instruction folded into the
    user turn as `<system>`-tagged text. The leading system prompt hoists to the top-level `system`
    parameter either way — the point of the feature is that adding the instruction leaves it untouched.

    Both recordings show the model acting on the instruction, so the annotated signature is asserted
    on the output: the two renderings are equivalent in effect, they differ in what they cost.
    """
    agent = Agent(AnthropicModel(case.model, provider=AnthropicProvider(api_key=anthropic_api_key)))

    result = await agent.run('Review it again.', message_history=message_history())
    assert 'def add(a: int, b: int) -> int:' in result.output

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == case.expected_messages


async def test_mid_conversation_system_prompt_takes_cache_breakpoint(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_requests: list[dict[str, Any]]
):
    """`anthropic_cache_messages` puts its breakpoint on the trailing `system` entry, and the API takes it.

    The setting caches the last content block of the last message, which the new entry now is, so this
    pins that a `cache_control` on a `system`-role block isn't rejected — the shape is only reachable
    since mid-conversation system prompts started getting their own message.
    """
    agent = Agent(
        AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key)),
        model_settings=AnthropicModelSettings(anthropic_cache_messages=True),
    )

    result = await agent.run('Review it again.', message_history=message_history())
    assert 'def add(a: int, b: int) -> int:' in result.output

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert body['messages'][-1] == snapshot(
        {
            'role': 'system',
            'content': [
                {
                    'text': 'From now on, every suggestion must include explicit type annotations.',
                    'type': 'text',
                    'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                }
            ],
        }
    )


async def test_enqueued_system_prompt_is_inside_following_cache_point(allow_model_requests: None):
    """An explicit cache boundary covers every enqueued part that precedes it."""
    responses = [
        completion_message(
            [BetaToolUseBlock(id='inject', input={}, name='inject', type='tool_use')],
            BetaUsage(input_tokens=5, output_tokens=2),
        ),
        completion_message([BetaTextBlock(text='done', type='text')], BetaUsage(input_tokens=10, output_tokens=2)),
    ]
    mock_client = MockAnthropic.create_mock(responses)
    agent = Agent(
        AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=mock_client)),
        deps_type=type(None),
    )

    @agent.tool
    def inject(ctx: RunContext[None]) -> str:
        ctx.enqueue(SystemPromptPart('S'), 'context', CachePoint())
        return 'injected'

    result = await agent.run('start')
    assert result.output == 'done'
    injected = next(
        message
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        and any(isinstance(part, SystemPromptPart) and part.content == 'S' for part in message.parts)
    )
    assert isinstance(injected.parts[0], SystemPromptPart)
    assert injected.parts[0].content == 'S'
    assert isinstance(injected.parts[1], UserPromptPart)
    assert injected.parts[1].content == ['context', CachePoint()]

    second_request = get_mock_chat_completion_kwargs(mock_client)[1]
    cache_controlled_blocks = [
        (message['role'], block['text'])
        for message in second_request['messages']
        for block in message['content']
        if block.get('cache_control')
    ]
    assert cache_controlled_blocks == [('system', 'S')]


async def test_enqueued_system_prompt_preserves_nonterminal_cache_boundary(allow_model_requests: None):
    """A cache marker cannot silently exclude an earlier instruction when later content follows it."""
    responses = [
        completion_message(
            [BetaToolUseBlock(id='inject', input={}, name='inject', type='tool_use')],
            BetaUsage(input_tokens=5, output_tokens=2),
        ),
        completion_message([BetaTextBlock(text='done', type='text')], BetaUsage(input_tokens=10, output_tokens=2)),
    ]
    mock_client = MockAnthropic.create_mock(responses)
    agent = Agent(
        AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=mock_client)),
        deps_type=type(None),
    )

    @agent.tool
    def inject(ctx: RunContext[None]) -> str:
        ctx.enqueue(SystemPromptPart('S'), 'before', CachePoint(), 'after')
        return 'injected'

    result = await agent.run('start')
    assert result.output == 'done'

    second_request = get_mock_chat_completion_kwargs(mock_client)[1]
    assert second_request['messages'][-1] == snapshot(
        {
            'role': 'user',
            'content': [
                {
                    'tool_use_id': 'inject',
                    'type': 'tool_result',
                    'content': [{'text': 'injected', 'type': 'text'}],
                    'is_error': False,
                },
                {'text': '<system>S</system>', 'type': 'text'},
                {'text': 'before', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
                {'text': 'after', 'type': 'text'},
            ],
        }
    )


def test_inline_system_cache_boundary_before_later_request_is_preserved():
    """A later request cannot be silently pulled inside an earlier explicit cache boundary."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('answer')]),
        ModelRequest(parts=[SystemPromptPart('S'), UserPromptPart(['context', CachePoint()])]),
        ModelRequest(parts=[UserPromptPart('later')]),
    ]

    _, anthropic_messages = asyncio.run(
        model._map_message(  # pyright: ignore[reportPrivateUsage]
            history,
            ModelRequestParameters(),
            AnthropicModelSettings(),
        )
    )
    assert anthropic_messages[-2:] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': '<system>S</system>', 'type': 'text'},
                    {'text': 'context', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
                ],
            },
            {'role': 'user', 'content': [{'text': 'later', 'type': 'text'}]},
        ]
    )


def test_inline_system_cache_boundary_survives_empty_response_before_later_request():
    """Rendered-empty responses cannot hide later content that would widen a cache boundary."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('answer')]),
        ModelRequest(parts=[SystemPromptPart('S'), UserPromptPart(['context', CachePoint()])]),
        ModelResponse(parts=[TextPart('')]),
        ModelRequest(parts=[UserPromptPart('later')]),
    ]

    _, anthropic_messages = asyncio.run(
        model._map_message(  # pyright: ignore[reportPrivateUsage]
            history,
            ModelRequestParameters(),
            AnthropicModelSettings(),
        )
    )
    assert anthropic_messages[-2:] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {'text': '<system>S</system>', 'type': 'text'},
                    {'text': 'context', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
                ],
            },
            {'role': 'user', 'content': [{'text': 'later', 'type': 'text'}]},
        ]
    )


def test_inline_system_cache_boundary_before_system_only_request_stays_native():
    """A later system-only request does not make an exact native cache boundary unrepresentable."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('first')]),
        ModelResponse(parts=[TextPart('answer')]),
        ModelRequest(parts=[SystemPromptPart('S'), UserPromptPart(['context', CachePoint()])]),
        ModelRequest(parts=[SystemPromptPart('T')]),
    ]

    _, anthropic_messages = asyncio.run(
        model._map_message(  # pyright: ignore[reportPrivateUsage]
            history,
            ModelRequestParameters(),
            AnthropicModelSettings(),
        )
    )
    assert anthropic_messages[-3:] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'context', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {
                        'text': 'S',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                    }
                ],
            },
            {'role': 'system', 'content': [{'text': 'T', 'type': 'text'}]},
        ]
    )


def test_inline_system_cache_boundary_cannot_split_tool_pair():
    """A cache boundary cannot be preserved between a tool call and its required result."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('start')]),
        ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='call_1')]),
        ModelRequest(parts=[SystemPromptPart('S'), UserPromptPart([CachePoint()])]),
        ModelRequest(parts=[ToolReturnPart(tool_name='lookup', content='result', tool_call_id='call_1')]),
    ]

    with pytest.raises(UserError, match='cannot be placed between an Anthropic tool call and its result'):
        asyncio.run(
            model._map_message(  # pyright: ignore[reportPrivateUsage]
                history,
                ModelRequestParameters(),
                AnthropicModelSettings(),
            )
        )


def test_inline_system_cache_boundary_before_merged_tool_result_raises():
    """History cleaning cannot hide a cache boundary that precedes an outstanding tool result."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('start')]),
        ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='call_1')]),
        ModelRequest(
            parts=[
                SystemPromptPart('S'),
                UserPromptPart([CachePoint()]),
                ToolReturnPart(tool_name='lookup', content='result', tool_call_id='call_1'),
            ]
        ),
    ]

    with pytest.raises(UserError, match='cannot be placed between an Anthropic tool call and its result'):
        asyncio.run(
            model._map_message(  # pyright: ignore[reportPrivateUsage]
                history,
                ModelRequestParameters(),
                AnthropicModelSettings(),
            )
        )


def test_inline_system_cache_boundary_after_merged_tool_result_is_preserved():
    """A tagged fallback keeps the required tool result first when the boundary follows it."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('start')]),
        ModelResponse(parts=[ToolCallPart(tool_name='lookup', args={}, tool_call_id='call_1')]),
        ModelRequest(
            parts=[
                SystemPromptPart('S'),
                ToolReturnPart(tool_name='lookup', content='result', tool_call_id='call_1'),
                UserPromptPart([CachePoint()]),
            ]
        ),
        ModelRequest(parts=[UserPromptPart('later')]),
    ]

    _, anthropic_messages = asyncio.run(
        model._map_message(  # pyright: ignore[reportPrivateUsage]
            history,
            ModelRequestParameters(),
            AnthropicModelSettings(),
        )
    )
    assert anthropic_messages[-2:] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'tool_use_id': 'call_1',
                        'type': 'tool_result',
                        'content': [{'text': 'result', 'type': 'text'}],
                        'is_error': False,
                    },
                    {
                        'text': '<system>S</system>',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                    },
                ],
            },
            {'role': 'user', 'content': [{'text': 'later', 'type': 'text'}]},
        ]
    )


def test_inline_system_cache_fallback_removes_the_matching_request_by_identity():
    """Fallback cannot remove an earlier request with identical rendered content."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(['context', CachePoint()])]),
        ModelResponse(parts=[TextPart('first answer')]),
        ModelRequest(parts=[SystemPromptPart('S'), UserPromptPart(['context', CachePoint()])]),
        ModelRequest(parts=[UserPromptPart('later')]),
    ]

    _, anthropic_messages = asyncio.run(
        model._map_message(  # pyright: ignore[reportPrivateUsage]
            history,
            ModelRequestParameters(),
            AnthropicModelSettings(),
        )
    )
    assert anthropic_messages == snapshot(
        [
            {
                'role': 'user',
                'content': [{'text': 'context', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}],
            },
            {'role': 'assistant', 'content': [{'text': 'first answer', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [
                    {'text': '<system>S</system>', 'type': 'text'},
                    {'text': 'context', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}},
                ],
            },
            {'role': 'user', 'content': [{'text': 'later', 'type': 'text'}]},
        ]
    )


async def test_inline_system_prompt_cache_prefix_is_reused(
    allow_model_requests: None,
    anthropic_api_key: str,
    vcr: Cassette,
    rendered_requests: list[dict[str, Any]],
):
    """A terminal breakpoint caches the enqueued instruction and the full stable prefix before it."""
    agent = Agent(AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key)))
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart('Reply with OK.'),
                UserPromptPart(_CACHE_PREFIX),
            ]
        ),
        ModelResponse(parts=[TextPart('OK')]),
        ModelRequest(
            parts=[
                SystemPromptPart('S'),
                UserPromptPart(['context', CachePoint()]),
            ]
        ),
    ]

    first = await agent.run(message_history=history)
    second = await agent.run(message_history=history)

    requests = vcr.requests  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    recorded_bodies: list[dict[str, Any]] = [
        json.loads(request.body)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
        for request in requests  # pyright: ignore[reportUnknownVariableType]
    ]
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']} for body in recorded_bodies]
    assert rendered_requests[0] == rendered_requests[1]
    assert rendered_requests[0]['messages'][-2:] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'context', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {
                        'text': 'S',
                        'type': 'text',
                        'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                    }
                ],
            },
        ]
    )
    cached_prefix_tokens = max(first.usage.cache_write_tokens, first.usage.cache_read_tokens)
    assert cached_prefix_tokens > 0
    assert second.usage.cache_read_tokens >= cached_prefix_tokens


def test_leading_cache_point_after_inline_system_prompt_preserves_boundary():
    """A marker before later user content keeps the tagged fallback's exact boundary."""
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))
    history = message_history()
    history[-1] = ModelRequest(
        parts=[SystemPromptPart(content=INSTRUCTION), UserPromptPart(content=[CachePoint(), 'Review it again.'])]
    )

    _, anthropic_messages = asyncio.run(
        model._map_message(  # pyright: ignore[reportPrivateUsage]
            history,
            ModelRequestParameters(),
            AnthropicModelSettings(),
        )
    )
    assert anthropic_messages[-1] == snapshot(
        {
            'role': 'user',
            'content': [
                {
                    'text': f'<system>{INSTRUCTION}</system>',
                    'type': 'text',
                    'cache_control': {'type': 'ephemeral', 'ttl': '5m'},
                },
                {'text': 'Review it again.', 'type': 'text'},
            ],
        }
    )


def test_cache_point_with_nothing_before_it_still_raises():
    """The error the message describes — no prior content anywhere — is the one case left raising.

    Falling back to the previous message must not turn this into a silent no-op: a `CachePoint` opening
    the very first turn has nothing to cache, and saying so is more useful than dropping it.
    """
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key='not-used'))

    with pytest.raises(UserError, match='CachePoint cannot be the first content in a user message'):
        asyncio.run(
            model._map_message(  # pyright: ignore[reportPrivateUsage]
                [ModelRequest(parts=[UserPromptPart(content=[CachePoint(), 'Hello.'])])],
                ModelRequestParameters(),
                AnthropicModelSettings(),
            )
        )


async def test_mid_conversation_system_prompt_without_user_turn(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_requests: list[dict[str, Any]]
):
    """Without a user turn to follow, the instruction gets a minimal one rather than degrading.

    Anthropic rejects a `system` entry that directly follows an assistant turn, so a system prompt
    that lands at the end of the history on its own — here, a run with no new user prompt, the shape
    `ctx.enqueue(SystemPromptPart(...))` produces — has nothing legal to sit behind. A `.` user turn
    is the cheapest thing that satisfies the rule, and it asserts nothing on the user's behalf.

    The alternative was the `<system>`-tagged rendering, and the recording it replaced is why we
    don't: the model read the tagged text as "a stated preference from you rather than a
    higher-privilege instruction". Here it just complies.
    """
    agent = Agent(AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key)))

    result = await agent.run(message_history=message_history())
    assert 'def add(a: int, b: int) -> int:' in result.output

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': '.', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
                ],
            },
        ]
    )


async def test_mid_conversation_system_prompt_before_another_request(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_requests: list[dict[str, Any]]
):
    """A system entry that a *user* turn would follow slides past it instead of degrading.

    `[user, system, user]` is rejected outright, and a second `ModelRequest` directly after the one
    carrying the instruction produces exactly that. `Model.request` is public and doesn't run the
    history cleaning that folds consecutive requests together, so the adapter can't assume every
    request is followed by a response.

    Moving the entry past the user turn costs nothing: it governs the same generation either way, and
    it stays an operator instruction rather than becoming text the model can overrule.

    Driven through `Model.request` rather than `Agent.run` precisely because the agent's history
    cleaning would merge the two requests.
    """
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            *message_history_with_paired_instruction(),
            ModelRequest(parts=[UserPromptPart(content='Review it once more.')]),
        ],
        None,
        ModelRequestParameters(),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    assert 'def add(a: int, b: int) -> int:' in reply.content

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Review it again.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Review it once more.', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
                ],
            },
        ]
    )


async def test_mid_conversation_system_prompt_kept_mid_history(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_requests: list[dict[str, Any]]
):
    """An instruction with a real response after it keeps the native entry, mid-history.

    This is the shape a stored conversation replays on every later turn — the entry sits between the
    user turn it arrived with and the assistant turn it governed, and stays there as the history
    grows past it. The cheap alternative (only allow the entry in trailing position) would
    re-render the whole conversation's instructions on every request, which is the cache churn the
    feature exists to avoid.
    """
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            *message_history_with_paired_instruction(),
            ModelResponse(parts=[TextPart(content='def add(a: int, b: int) -> int: return a + b')]),
            ModelRequest(parts=[UserPromptPart(content='Now do `def mul(a, b): return a * b`.')]),
        ],
        None,
        ModelRequestParameters(),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    assert 'def mul(a: int, b: int) -> int:' in reply.content

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert [message['role'] for message in body['messages']] == snapshot(
        ['user', 'assistant', 'user', 'system', 'assistant', 'user']
    )
    assert body['messages'][3] == snapshot(
        {
            'content': [
                {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
            ],
            'role': 'system',
        }
    )


async def test_mid_conversation_system_prompt_before_empty_response(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_requests: list[dict[str, Any]]
):
    """A `ModelResponse` that renders to nothing doesn't count as the assistant turn to follow.

    A response whose parts all drop out — here an empty `TextPart` — appends no assistant message,
    so a request after it lands directly behind the system entry and the API rejects the whole
    thing. Reading ahead in the message list would call this placement legal; only checking what was
    actually rendered gets it right, which is why the decision is a pass over the wire messages.

    This is the case that makes the pass necessary rather than merely convenient: the history here
    is byte-identical to `..._kept_mid_history` apart from the response being empty, and that one
    leaves its entry where it is.
    """
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            *message_history_with_paired_instruction(),
            ModelResponse(parts=[TextPart(content='')]),
            ModelRequest(parts=[UserPromptPart(content='Review it once more.')]),
        ],
        None,
        ModelRequestParameters(),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    assert 'def add(a: int, b: int) -> int:' in reply.content

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert [message['role'] for message in body['messages']] == snapshot(
        ['user', 'assistant', 'user', 'user', 'system']
    )
    assert body['messages'][-1] == snapshot(
        {
            'role': 'system',
            'content': [
                {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
            ],
        }
    )


async def test_two_mid_conversation_system_prompts_keep_their_order(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_requests: list[dict[str, Any]]
):
    """Two instructions that both have to move end up adjacent, in the order they were given.

    Each slides only past *user* turns, so the earlier one stops behind the later one rather than
    overtaking it — sliding past everything would invert them, and "ignore the previous instruction"
    cases would then resolve backwards. Consecutive `system` entries are a placement the API takes:
    the group as a whole still precedes the generation, and the recording shows both obeyed.
    """
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            *message_history_with_paired_instruction(),
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Also always state the time complexity.'),
                    UserPromptPart(content='Review it once more.'),
                ]
            ),
        ],
        None,
        ModelRequestParameters(),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    # Both instructions applied: the annotation on the parameter (the model is free to pick the
    # type), and the complexity note. Asserting the exact signature would pin a choice neither
    # instruction constrains.
    assert 'def add(a: int' in reply.content
    assert 'O(1)' in reply.content

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Review it again.', 'type': 'text'}]},
            {'role': 'user', 'content': [{'text': 'Review it once more.', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
                ],
            },
            {'role': 'system', 'content': [{'text': 'Also always state the time complexity.', 'type': 'text'}]},
        ]
    )


async def test_mid_conversation_system_prompt_anchor_keeps_tool_pair_intact(
    allow_model_requests: None, anthropic_api_key: str, vcr: Cassette, rendered_requests: list[dict[str, Any]]
):
    """The anchor never lands between a `tool_use` and the `tool_result` that answers it.

    A system-only request between the model's tool call and the request carrying the result is the one
    shape where anchoring at the position the instruction was authored at is wrong: the anchor goes in
    behind the `tool_use`, the entry then slides past the result, and the anchor is left splitting the
    pair. The API rejects that outright — `tool_use ids were found without tool_result blocks
    immediately after: call_1` — so it was a 400 rather than a degraded rendering.

    Deciding placement after the slide instead, on final positions, this history needs no anchor at
    all: the entry ends up behind the tool result, which is a user turn. The recorded 200 is the
    assertion that matters; the shape below is what earned it.
    """
    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(api_key=anthropic_api_key))

    response = await model.request(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='You are a code reviewer.'),
                    UserPromptPart(content='Look up the style guide, then review `def add(a, b): return a + b`.'),
                ]
            ),
            ModelResponse(parts=[ToolCallPart(tool_name='style_guide', args={}, tool_call_id='call_1')]),
            ModelRequest(parts=[SystemPromptPart(content=INSTRUCTION)]),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='style_guide', content='Prefer explicit signatures.', tool_call_id='call_1'
                    )
                ]
            ),
        ],
        None,
        ModelRequestParameters(
            function_tools=[
                ToolDefinition(
                    name='style_guide',
                    description='Fetch the project style guide.',
                    parameters_json_schema={'type': 'object', 'properties': {}},
                )
            ]
        ),
    )
    reply = response.parts[-1]
    assert isinstance(reply, TextPart)
    assert 'def add(a: int, b: int) -> int:' in reply.content

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert body['messages'] == snapshot(
        [
            {
                'role': 'user',
                'content': [
                    {
                        'text': 'Look up the style guide, then review `def add(a, b): return a + b`.',
                        'type': 'text',
                    }
                ],
            },
            {
                'role': 'assistant',
                'content': [{'id': 'call_1', 'input': {}, 'name': 'style_guide', 'type': 'tool_use'}],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'tool_use_id': 'call_1',
                        'type': 'tool_result',
                        'content': [{'text': 'Prefer explicit signatures.', 'type': 'text'}],
                        'is_error': False,
                    }
                ],
            },
            {
                'role': 'system',
                'content': [
                    {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
                ],
            },
        ]
    )


def test_only_foundry_is_excluded_by_the_transport_gate():
    """Exactly which transports fall back, asserted as a set rather than left to a comment.

    Anthropic publishes the feature for the Claude API, Amazon Bedrock and Google Cloud, so Microsoft
    Foundry is the only exclusion. Bedrock is recorded (`..._on_bedrock`) and the direct API is recorded
    by every other test here, but Vertex rests on the published list — we have no Vertex credentials, and
    a mocked test can only show what we send, never that the transport serves the role. Pinning the tuple
    is the honest half of that: it can't confirm Vertex works, but it does stop Vertex being added to or
    dropped from the exclusions without someone changing this line and thinking about why.
    """
    assert _INLINE_SYSTEM_PROMPT_UNSUPPORTED_CLIENTS == (AsyncAnthropicFoundry,)


async def test_mid_conversation_system_prompt_on_foundry(allow_model_requests: None):
    """Microsoft Foundry doesn't serve the `system` role, so a supported model still gets the wrap.

    This is the transport half of the gate on its own: the model name clears
    `_INLINE_SYSTEM_PROMPT_MODEL_PREFIXES`, and the instruction is still `<system>`-tagged because of
    the client it would be sent through. Bedrock, which used to stand in for this case, turned out to
    support the role (see `..._on_bedrock`), leaving Foundry as the only transport that falls back.

    Mocked rather than recorded because we have no Foundry credentials, so there is no live shape to
    capture — and the assertion is about what the adapter sends, which the captured kwargs show
    directly.
    """
    completion = completion_message(
        [BetaTextBlock(text='def add(a: int, b: int) -> int: return a + b', type='text')],
        BetaUsage(input_tokens=5, output_tokens=10),
    )
    # `spec=` is what makes the client-type gate see a Foundry client: it sets `__class__`, so the
    # `isinstance` check in `AnthropicModel.profile` matches and narrows the flag. The wrap asserted
    # below is then the shared one from `Model.prepare_messages` that every model without support
    # gets — the adapter has no `<system>` rendering of its own.
    foundry_client = MagicMock(spec=AsyncAnthropicFoundry)
    foundry_client.base_url = 'https://example.services.ai.azure.com/anthropic'
    foundry_client.beta.messages.create = AsyncMock(return_value=completion)

    model = AnthropicModel('claude-opus-4-8', provider=AnthropicProvider(anthropic_client=foundry_client))
    # The two halves of the gate, asserted where each is decided: the provider's profile sees the model
    # name and says yes, and the model's profile — which is the one that also sees the client — narrows
    # it back to no. That's what makes this a transport exclusion rather than a model one.
    provider_profile = AnthropicProvider.model_profile('claude-opus-4-8')
    assert provider_profile is not None
    assert provider_profile.get('supports_inline_system_prompts') is True
    assert model.profile.get('supports_inline_system_prompts') is False

    await Agent(model).run('Review it again.', message_history=message_history())

    call_kwargs = foundry_client.beta.messages.create.call_args.kwargs
    assert call_kwargs['system'] == 'You are a code reviewer.'
    assert call_kwargs['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}]},
            {'role': 'assistant', 'content': [{'text': 'Looks fine.', 'type': 'text'}]},
            {
                'role': 'user',
                'content': [
                    {
                        'text': '<system>From now on, every suggestion must include explicit type annotations.</system>',
                        'type': 'text',
                    },
                    {'text': 'Review it again.', 'type': 'text'},
                ],
            },
        ]
    )


async def test_mid_conversation_system_prompt_on_bedrock(
    allow_model_requests: None,
    anthropic_bedrock_client: AsyncAnthropicBedrock,
    vcr: Cassette,
    rendered_requests: list[dict[str, Any]],
):
    """Bedrock serves the `system` role, so a supported model gets the native entry there too.

    Anthropic publishes the feature for the Claude API, Amazon Bedrock and Google Cloud, and this is
    the Bedrock half of that recorded rather than assumed. It replaced a test that sent
    `us.anthropic.claude-sonnet-5` and concluded from the `<system>`-tagged result that Bedrock
    doesn't serve the role — but Sonnet 5 ignores the entry on every transport, so that test was
    measuring the model. Microsoft Foundry is the transport that still falls back.

    The Bedrock provider segment is stripped before the prefix check, so `us.anthropic.` in front of
    a supported model leaves both halves of the gate open.
    """
    model = AnthropicModel(
        'us.anthropic.claude-opus-4-8', provider=AnthropicProvider(anthropic_client=anthropic_bedrock_client)
    )
    assert model.profile.get('supports_inline_system_prompts') is True

    result = await Agent(model).run('Review it again.', message_history=message_history())
    assert 'def add(a: int' in result.output

    body = single_request_body(vcr)
    assert rendered_requests == [{'system': body['system'], 'messages': body['messages']}]
    assert body['system'] == 'You are a code reviewer.'
    assert body['messages'] == snapshot(
        [
            {'content': [{'text': 'Review `def add(a, b): return a + b`.', 'type': 'text'}], 'role': 'user'},
            {'content': [{'text': 'Looks fine.', 'type': 'text'}], 'role': 'assistant'},
            {'role': 'user', 'content': [{'text': 'Review it again.', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [
                    {'text': 'From now on, every suggestion must include explicit type annotations.', 'type': 'text'}
                ],
            },
        ]
    )
