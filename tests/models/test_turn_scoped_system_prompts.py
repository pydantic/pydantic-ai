"""Turn-scoped system prompts: `SystemPromptPart(..., scope='turn')`.

A turn-scoped prompt is a nudge for one model request — a reminder against instruction fade, a note
about the current moment — that stays in the message history like any other part. Two renderings:

* Where the API keeps it and stops rendering it itself (Anthropic's `clear_at: 'next_user_message'`),
  every copy is sent unchanged, current or not. Sending it unchanged is the point: on Claude Opus 5.5,
  Sonnet 5.5 and Fable 5.1, deleting a reminder on the next request invalidates every earlier thinking
  block, and the history stays append-only on the wire, so the cached prefix keeps matching.
* Everywhere else, it's sent while it's current and left out of every later request, rendered at the
  end of the request so automatic cache breakpoints can land before it.

The recorded tests pin both against the real APIs. The unit tests cover what a recording can't see:
the cache anchor on adapters where we can't record, orphaned parts after a failed step, and the core
projection on both sides of the `supports_turn_scoped_system_prompts` flag.
"""

from __future__ import annotations as _annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RunContext,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    capture_run_messages,
)
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, RequestCapture, try_import

if TYPE_CHECKING:
    from .conftest import AnthropicModelFactory

with try_import() as anthropic_available:
    from anthropic import AsyncAnthropicFoundry

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile
    from pydantic_ai.providers.anthropic import AnthropicProvider

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
    from pydantic_ai.providers.bedrock import BedrockProvider

pytestmark = pytest.mark.vcr

_TURN_SCOPED_BETA = 'mid-conversation-system-clear-at-2026-08-21'

SYSTEM_PROMPT = (
    'You are a terse weather assistant. Always use the get_weather tool for each city, one city per call, '
    'and call them one at a time in the order given, waiting for each result. '
) + ' '.join(f'Background fact {i}: the weather service code {i} means nothing important.' for i in range(300))
PROMPT = 'Get the weather in Paris, then in Oslo, then summarize in one line.'


@dataclass
class TurnReminder(AbstractCapability[Any]):
    """Adds a turn-scoped reminder to every model request, the way a harness capability does.

    The reminder goes into persistent history through `ctx.messages` and into this request through a
    new `request_context.messages` list, which is the contract `before_model_request` hooks follow.
    """

    text: str = 'Reminder: be terse. This is request {n}.'
    requests: int = field(default=0, init=False)

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.requests += 1
        reminder = ModelRequest(parts=[SystemPromptPart(self.text.format(n=self.requests), scope='turn')])
        ctx.messages.append(reminder)
        request_context.messages = [*request_context.messages, reminder]
        return request_context


def weather_agent(model: Any, **kwargs: Any) -> Agent[None, str]:
    agent = Agent(model, system_prompt=SYSTEM_PROMPT, capabilities=[TurnReminder()], **kwargs)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'{city}: 18C, cloudy'

    return agent


def turn_scoped_texts(messages: Sequence[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.scope == 'turn'
    ]


def without_cache_control(value: Any) -> Any:
    """`value` with every `cache_control` key removed, so wire prefixes compare on content alone."""
    if isinstance(value, dict):
        return {key: without_cache_control(item) for key, item in value.items() if key != 'cache_control'}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [without_cache_control(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


def message_breakpoints(body: dict[str, Any]) -> list[str]:
    """Where in `messages` a request carries `cache_control`, as `messages[i][j]` paths."""
    return [
        f'messages[{message_index}][{block_index}]'
        for message_index, message in enumerate(body['messages'])
        if isinstance(message['content'], list)
        for block_index, block in enumerate(message['content'])
        if 'cache_control' in block
    ]


def assert_append_only(bodies: Sequence[dict[str, Any]]) -> None:
    """Every request's `messages` starts with the previous request's, byte for byte bar breakpoints.

    That's what keeps the cached prefix and, on models with preserved thinking, every earlier thinking
    block valid: the prefix check compares the conversation as sent, and a breakpoint is the one thing
    it ignores.
    """
    for previous, current in zip(bodies, bodies[1:]):
        prefix = without_cache_control(previous['messages'])
        assert without_cache_control(current['messages'])[: len(prefix)] == prefix


@pytest.mark.moves_cache_prefix(
    reason='`anthropic_cache_messages` moves its breakpoint to the newest tool result each request; '
    'with breakpoints stripped the messages are append-only, which `assert_append_only` checks'
)
@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize('model_name', ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-fable-5-1'])
async def test_anthropic_turn_scoped_reminders_keep_thinking_and_cache(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
    model_name: str,
):
    """Every reminder is sent as a `clear_at` system entry and kept, so no thinking block is dropped.

    These are the models with preserved thinking. `drop_block` makes the API report any thinking block
    a history edit invalidated in the response's `input_transformations`, instead of rejecting it, and
    the run reports none. The beta header goes with every request, every reminder ever sent is still on
    the wire in the last one, and the wire grows append-only, which is also why each cache read covers
    the whole previous request: the reminder sits after the breakpoint and costs nothing once cleared.
    """
    settings: AnthropicModelSettings = {
        'anthropic_cache_messages': True,
        'anthropic_thinking': {
            'type': 'adaptive',
            'display': 'summarized',
            'block_binding': {'prefix_mismatch_behavior': 'drop_block'},
        },
    }
    agent = weather_agent(anthropic_model(model_name, capture=True), model_settings=settings)

    result = await agent.run(PROMPT)

    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert len(responses) >= 3
    assert all((response.provider_details or {}).get('input_transformations') is None for response in responses)
    assert any(isinstance(part, TextPart) for part in responses[-1].parts)

    bodies = request_capture.bodies('/v1/messages')
    assert len(bodies) == len(responses)
    assert all(_TURN_SCOPED_BETA in headers.get('anthropic-beta', '') for headers in request_capture.headers)
    assert_append_only(bodies)
    turn_entries = [
        message for message in bodies[-1]['messages'] if isinstance(message, dict) and 'clear_at' in message
    ]
    assert turn_entries == [
        {
            'role': 'system',
            'content': [{'text': f'Reminder: be terse. This is request {n}.', 'type': 'text'}],
            'clear_at': 'next_user_message',
        }
        for n in range(1, len(bodies) + 1)
    ]
    # The breakpoint lands on the tool result ahead of the reminder: `cache_control` on a turn-scoped
    # entry is a 400.
    assert [message_breakpoints(body) for body in bodies[1:]] == [
        [f'messages[{len(body["messages"]) - 2}][0]'] for body in bodies[1:]
    ]
    for previous, current in zip(responses, responses[1:]):
        assert current.usage.cache_read_tokens >= previous.usage.input_tokens - 30


@dataclass(frozen=True)
class FallbackCase:
    id: str
    settings: AnthropicModelSettings
    breakpoints: list[str]


FALLBACK_CASES = [
    FallbackCase(
        id='cache-messages', settings={'anthropic_cache_messages': True}, breakpoints=snapshot(['messages[2][0]'])
    ),
    FallbackCase(id='auto-cache', settings={'anthropic_cache': True}, breakpoints=snapshot(['messages[2][0]'])),
]


@pytest.mark.moves_cache_prefix(
    reason='a model that cannot clear a turn-scoped prompt is sent it only while current, by design'
)
@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in FALLBACK_CASES])
async def test_anthropic_turn_scoped_fallback_on_model_without_native_support(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
    case: FallbackCase,
):
    """Claude Haiku 4.5 has no `system` role: the reminder is sent while current, then left out.

    It renders as `<system>`-tagged text at the end of the request's user turn, and the breakpoint goes
    on the block before it, whether `anthropic_cache_messages` or `anthropic_cache` asked for one: with
    `anthropic_cache` the server would otherwise put its own breakpoint on the reminder, writing an
    entry the next request, which no longer sends it, can't read. With the breakpoint there, each cache
    read covers the previous request bar the reminder. No beta header, no `clear_at`, and no empty
    responses: the text never piles up after the tool results.
    """
    agent = weather_agent(anthropic_model('claude-haiku-4-5', capture=True), model_settings=case.settings)

    result = await agent.run(PROMPT)

    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert all(response.parts for response in responses)
    bodies = request_capture.bodies('/v1/messages')
    assert all(_TURN_SCOPED_BETA not in headers.get('anthropic-beta', '') for headers in request_capture.headers)
    for n, body in enumerate(bodies, start=1):
        texts = [
            block['text']
            for message in body['messages']
            for block in message['content']
            if isinstance(block, dict) and block.get('type') == 'text' and 'Reminder' in str(block.get('text'))
        ]
        assert texts == [f'<system>Reminder: be terse. This is request {n}.</system>']
        last_content = body['messages'][-1]['content']
        assert last_content[-1] == {'text': texts[0], 'type': 'text'}
    assert message_breakpoints(bodies[1]) == case.breakpoints
    for previous, current in zip(responses, responses[1:]):
        assert current.usage.cache_read_tokens >= previous.usage.input_tokens - 30
    # History keeps every reminder; only the wire leaves the finished ones out.
    assert turn_scoped_texts(result.all_messages()) == [
        f'Reminder: be terse. This is request {n}.' for n in range(1, len(bodies) + 1)
    ]


@pytest.mark.moves_cache_prefix(
    reason='a model that cannot clear a turn-scoped prompt is sent it only while current, by design'
)
@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.parametrize('api', ['chat', 'responses'])
async def test_openai_turn_scoped_reminder_sent_only_while_current(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture, api: str
):
    """OpenAI has no turn-scoped channel: each reminder goes out once, as a mid-conversation system message.

    GPT-5.6 takes explicit cache breakpoints, and this is the run that used to crash on Chat
    Completions when a harness reminder led with a `CachePoint`. Nothing here adds one, and each cache
    read still covers the previous request bar the reminder.
    """
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    if api == 'chat':
        model = OpenAIChatModel('gpt-5.6', provider=provider, settings={'openai_reasoning_effort': 'none'})
        path = '/chat/completions'
    else:
        model = OpenAIResponsesModel('gpt-5.6', provider=provider)
        path = '/responses'
    agent = weather_agent(model)

    result = await agent.run(PROMPT)

    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    bodies = request_capture.bodies(path)
    assert len(bodies) == len(responses) >= 3
    for n, body in enumerate(bodies, start=1):
        sent = body['messages'] if api == 'chat' else body['input']
        reminders = [item for item in sent if isinstance(item, dict) and 'Reminder' in str(item.get('content'))]
        assert reminders == [{'role': 'system', 'content': f'Reminder: be terse. This is request {n}.'}]
        assert sent[-1] == reminders[0]
    for previous, current in zip(responses, responses[1:]):
        assert current.usage.cache_read_tokens >= previous.usage.input_tokens - 30


def recording_function_model(seen: list[list[ModelMessage]], *, profile: ModelProfile | None = None) -> FunctionModel:
    """A model that calls `get_weather` twice, then answers, recording the history each request sends."""

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        if len(seen) < 3:
            return ModelResponse(parts=[ToolCallPart('get_weather', {'city': ['Paris', 'Oslo'][len(seen) - 1]})])
        return ModelResponse(parts=[TextPart('done')])

    return FunctionModel(respond, profile=profile)


async def test_superseded_reminders_are_left_out_without_native_support():
    """A model that can't clear a turn-scoped prompt gets each one only while it's current.

    It's moved to the end of its request and `<system>`-wrapped like any mid-conversation system prompt
    on a model without a `system` role, while the history keeps every reminder unchanged.
    """
    seen: list[list[ModelMessage]] = []
    agent = weather_agent(recording_function_model(seen))

    result = await agent.run(PROMPT)

    assert [[part.content for part in messages[-1].parts if isinstance(part, UserPromptPart)] for messages in seen] == [
        [PROMPT, '<system>Reminder: be terse. This is request 1.</system>'],
        ['<system>Reminder: be terse. This is request 2.</system>'],
        ['<system>Reminder: be terse. This is request 3.</system>'],
    ]
    assert all(len(turn_scoped_texts(messages)) == 0 for messages in seen)
    assert all(
        sum('Reminder' in str(getattr(part, 'content', '')) for message in messages for part in message.parts) == 1
        for messages in seen
    )
    assert turn_scoped_texts(result.all_messages()) == [
        'Reminder: be terse. This is request 1.',
        'Reminder: be terse. This is request 2.',
        'Reminder: be terse. This is request 3.',
    ]


async def test_native_support_sends_every_reminder():
    """A model whose API clears turn-scoped prompts itself is sent every one, unchanged and unmoved."""
    seen: list[list[ModelMessage]] = []
    profile = ModelProfile(supports_inline_system_prompts=True, supports_turn_scoped_system_prompts=True)
    agent = weather_agent(recording_function_model(seen, profile=profile))

    await agent.run(PROMPT)

    assert [turn_scoped_texts(messages) for messages in seen] == [
        ['Reminder: be terse. This is request 1.'],
        ['Reminder: be terse. This is request 1.', 'Reminder: be terse. This is request 2.'],
        [
            'Reminder: be terse. This is request 1.',
            'Reminder: be terse. This is request 2.',
            'Reminder: be terse. This is request 3.',
        ],
    ]


async def test_enqueued_turn_scoped_prompt():
    """`ctx.enqueue` delivers a turn-scoped prompt into the next request, and it's gone from the one after."""
    seen: list[list[ModelMessage]] = []
    agent = Agent(recording_function_model(seen))

    @agent.tool
    def get_weather(ctx: RunContext[None], city: str) -> str:
        ctx.enqueue(SystemPromptPart(f'The {city} result is cached; do not fetch it again.', scope='turn'))
        return f'{city}: 18C, cloudy'

    result = await agent.run(PROMPT)

    assert [
        [str(part.content) for message in messages for part in message.parts if isinstance(part, UserPromptPart)][1:]
        for messages in seen
    ] == [
        [],
        ['<system>The Paris result is cached; do not fetch it again.</system>'],
        ['<system>The Oslo result is cached; do not fetch it again.</system>'],
    ]
    assert turn_scoped_texts(result.all_messages()) == [
        'The Paris result is cached; do not fetch it again.',
        'The Oslo result is cached; do not fetch it again.',
    ]


@dataclass
class KeepLastRequest(AbstractCapability[Any]):
    """A compaction stand-in: replaces everything before the last request with a summary."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        summary = ModelRequest(parts=[UserPromptPart('Summary of the conversation so far.')])
        compacted: list[ModelMessage] = [
            summary,
            ModelResponse(parts=[TextPart('Noted.')]),
            request_context.messages[-1],
        ]
        ctx.messages[:] = compacted
        request_context.messages = list(compacted)
        return request_context


async def test_current_reminder_survives_compaction():
    """A compaction capability that keeps the current request keeps its turn-scoped prompt too.

    The reminder capability runs first, so compaction sees its reminder in the last request, the one
    it keeps; the request goes out with it, and the history still records it.
    """
    seen: list[list[ModelMessage]] = []
    agent = Agent(
        recording_function_model(seen),
        capabilities=[TurnReminder(), KeepLastRequest()],
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'{city}: 18C, cloudy'

    result = await agent.run(PROMPT)

    assert [
        [
            str(part.content)
            for part in messages[-1].parts
            if isinstance(part, UserPromptPart) and 'Reminder' in str(part.content)
        ]
        for messages in seen
    ] == [[f'<system>Reminder: be terse. This is request {n}.</system>'] for n in (1, 2, 3)]
    assert turn_scoped_texts(result.all_messages()) == ['Reminder: be terse. This is request 3.']


@dataclass
class ProviderDown(AbstractCapability[Any]):
    """Fails every model request after the reminder was added, before anything is sent."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        raise RuntimeError('provider down')


@pytest.mark.parametrize('stream', [False, True])
async def test_failed_step_drops_its_turn_scoped_prompt(stream: bool):
    """A step that raises takes its turn-scoped prompt with it, so a retry doesn't send it twice.

    Without that, resuming from the captured history would carry the failed step's reminder into the
    next request alongside the one that request adds for itself. (A stream that fails after it opened
    commits its partial response, which answers the request, so that case keeps the reminder.)
    """
    agent = Agent(TestModel(), capabilities=[TurnReminder(), ProviderDown()])

    with capture_run_messages() as messages:
        with pytest.raises(RuntimeError, match='provider down'):
            if stream:
                async with agent.run_stream('hello'):
                    pass  # pragma: no cover
            else:
                await agent.run('hello')

    assert [[type(part).__name__ for part in message.parts] for message in messages] == [['UserPromptPart']]


async def test_retry_without_response_drops_the_turn_scoped_prompt():
    """A retry that never got a response gives up on the step and its reminder.

    The model fails, and a capability turns the failure into a retry. The retry request gets a fresh
    reminder of its own, and the abandoned one is gone, from what the model is sent and from history.
    """
    seen: list[list[ModelMessage]] = []

    @dataclass
    class RetryOnError(AbstractCapability[Any]):
        async def on_model_request_error(
            self, ctx: RunContext[Any], *, request_context: ModelRequestContext, error: Exception
        ) -> ModelResponse:
            raise ModelRetry('try again')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        if len(seen) == 1:
            raise RuntimeError('provider down')
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), capabilities=[TurnReminder(), RetryOnError()])

    result = await agent.run('hello')

    assert [
        [
            str(part.content)
            for message in messages
            for part in message.parts
            if 'Reminder' in str(getattr(part, 'content', ''))
        ]
        for messages in seen
    ] == [
        ['<system>Reminder: be terse. This is request 1.</system>'],
        ['<system>Reminder: be terse. This is request 2.</system>'],
    ]
    assert turn_scoped_texts(result.all_messages()) == ['Reminder: be terse. This is request 2.']


async def test_fallback_attempts_keep_the_turn_scoped_prompt():
    """A fallback attempt is still the same step's turn, so the next model is sent the same reminder."""
    seen: list[list[ModelMessage]] = []

    def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        raise RuntimeError('provider down')

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('done')])

    model = FallbackModel(FunctionModel(fail), FunctionModel(respond), fallback_on=(RuntimeError,))
    agent = Agent(model, capabilities=[TurnReminder()])

    result = await agent.run('hello')

    assert [[str(part.content) for part in messages[-1].parts] for messages in seen] == [
        ['hello', '<system>Reminder: be terse. This is request 1.</system>'],
        ['hello', '<system>Reminder: be terse. This is request 1.</system>'],
    ]
    assert turn_scoped_texts(result.all_messages()) == ['Reminder: be terse. This is request 1.']


def test_turn_scoped_prompt_round_trips_through_serialization():
    """`scope` survives `ModelMessagesTypeAdapter`, which is what durable execution and storage use."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi'), SystemPromptPart('Be terse.', scope='turn')])
    ]

    assert ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages)) == messages


def tool_loop_history() -> list[ModelMessage]:
    """Two tool rounds, each with a reminder appended as its own request, the way `TurnReminder` writes them."""
    return [
        ModelRequest(parts=[UserPromptPart('Weather in Paris and Oslo?')]),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_1')]),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Paris: 18C', tool_call_id='call_1')]),
        ModelRequest(parts=[SystemPromptPart('Reminder 1.', scope='turn')]),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Oslo'}, tool_call_id='call_2')]),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Oslo: 18C', tool_call_id='call_2')]),
        ModelRequest(parts=[SystemPromptPart('Reminder 2.', scope='turn')]),
    ]


REMINDER_ONLY: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart('Reminder.', scope='turn')])]


def test_current_turn_scoped_prompts_move_to_the_last_request():
    """Turn-scoped prompts anywhere in the current turn go to the end of its last request.

    The agent merges a turn's requests before preparing them, so this is about `prepare_messages`
    called directly on a history that isn't merged: a request holding nothing but a moved prompt is
    left out, and one with other parts keeps them.
    """
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')]))
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hello'), SystemPromptPart('Reminder A.', scope='turn')]),
        ModelRequest(parts=[SystemPromptPart('Reminder B.', scope='turn')]),
        ModelRequest(parts=[UserPromptPart('Are you there?')]),
    ]

    assert model.prepare_messages(messages) == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Hello', timestamp=IsDatetime())]),
            ModelRequest(
                parts=[
                    UserPromptPart(content='Are you there?', timestamp=IsDatetime()),
                    UserPromptPart(content='<system>Reminder A.</system>', timestamp=IsDatetime()),
                    UserPromptPart(content='<system>Reminder B.</system>', timestamp=IsDatetime()),
                ],
                metadata={'__pydantic_ai__': {'turn_scoped_tail': 2}},
            ),
        ]
    )


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_turn_scoped_opt_out_through_the_profile():
    """Setting the profile flag to `False` on a model that has it opts out of the beta.

    For an account without access to it: the reminder is then sent only while it's current, as an
    ordinary `system` entry (the model still takes the role), without `clear_at` or the beta header.
    A unit test, not a recording: the claim is about the request, which the recorded tests already
    show the API accepting in both shapes.
    """
    provider = AnthropicProvider(api_key='not-used')
    model = AnthropicModel(
        'claude-opus-5-5', provider=provider, profile=AnthropicModelProfile(supports_turn_scoped_system_prompts=False)
    )
    native = AnthropicModel('claude-opus-5-5', provider=provider)
    foundry = AnthropicModel(
        'claude-opus-5-5',
        provider=AnthropicProvider(anthropic_client=AsyncAnthropicFoundry(api_key='x', base_url='https://example.com')),
    )
    assert model.profile.get('supports_inline_system_prompts') is True
    assert native.profile.get('supports_turn_scoped_system_prompts') is True
    # Microsoft Foundry serves no `system` role, so it can't serve a turn-scoped one either.
    assert foundry.profile.get('supports_turn_scoped_system_prompts') is False

    prepared = model.prepare_messages(tool_loop_history(), ModelRequestParameters())
    _, anthropic_messages = await model._map_message(prepared, ModelRequestParameters(), {})  # pyright: ignore[reportPrivateUsage]

    assert [message['role'] for message in anthropic_messages] == snapshot(
        ['user', 'assistant', 'user', 'assistant', 'user', 'system']
    )
    assert anthropic_messages[-1] == {'role': 'system', 'content': [{'text': 'Reminder 2.', 'type': 'text'}]}
    betas, _ = model._get_betas_and_extra_headers(  # pyright: ignore[reportPrivateUsage]
        {}, model.profile, prepared, ModelRequestParameters(), {}
    )
    assert _TURN_SCOPED_BETA not in betas


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_leading_turn_scoped_prompt_is_not_hoisted():
    """A turn-scoped prompt opening the first request is not part of the standing system prompt.

    The top-level `system` parameter goes out with every request, so hoisting it there would make it
    permanent; it's sent as a `clear_at` entry after the user turn instead.
    """
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(api_key='not-used'))
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart('You are terse.'),
                SystemPromptPart('Answer in French this turn.', scope='turn'),
                UserPromptPart('Hello'),
            ]
        )
    ]

    prepared = model.prepare_messages(messages, ModelRequestParameters())
    system_prompt, anthropic_messages = await model._map_message(  # pyright: ignore[reportPrivateUsage]
        prepared, ModelRequestParameters(), {}
    )

    assert system_prompt == 'You are terse.'
    assert anthropic_messages == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Hello', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [{'text': 'Answer in French this turn.', 'type': 'text'}],
                'clear_at': 'next_user_message',
            },
        ]
    )


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_bedrock_message_cache_point_goes_before_the_turn_scoped_prompt():
    """`bedrock_cache_messages` puts its cache point before the reminder, not after it.

    The Converse API has no `system` role in `messages`, so the reminder is `<system>`-tagged text that
    ends the last user turn and is gone from the next request; a cache point after it would write an
    entry that request can't read. A unit test because Bedrock isn't recorded in this suite: the claim
    is where the adapter puts the block.
    """
    model = BedrockConverseModel(
        'us.anthropic.claude-haiku-4-5-20251001-v1:0',
        provider=BedrockProvider(region_name='us-east-1', aws_access_key_id='x', aws_secret_access_key='x'),
    )
    settings: BedrockModelSettings = {'bedrock_cache_messages': True}

    prepared = model.prepare_messages(tool_loop_history(), ModelRequestParameters())
    _, bedrock_messages = await model._map_messages(prepared, ModelRequestParameters(), settings)  # pyright: ignore[reportPrivateUsage]

    assert bedrock_messages[-1]['content'] == snapshot(
        [
            {'toolResult': {'toolUseId': 'call_2', 'content': [{'text': 'Oslo: 18C'}], 'status': 'success'}},
            {'cachePoint': {'type': 'default'}},
            {'text': '<system>Reminder 2.</system>'},
        ]
    )
    assert 'Reminder 1.' not in str(bedrock_messages)

    # With nothing ahead of the reminder in its message, there's nothing to cache up to.
    only_reminder = model.prepare_messages(REMINDER_ONLY, ModelRequestParameters())
    _, bedrock_messages = await model._map_messages(only_reminder, ModelRequestParameters(), settings)  # pyright: ignore[reportPrivateUsage]
    assert bedrock_messages == snapshot([{'role': 'user', 'content': [{'text': '<system>Reminder.</system>'}]}])


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_openrouter_message_cache_control_goes_before_the_turn_scoped_prompt():
    """`openrouter_cache_messages` marks the message before the reminder, not the reminder.

    OpenRouter takes no `system` entries mid-conversation, so the reminder is its own `<system>`-tagged
    user message at the end, and it's gone from the next request. A unit test because the claim is
    which message the adapter marks, which a recording replayed by URL wouldn't catch.
    """
    model = OpenRouterModel('anthropic/claude-haiku-4.5', provider=OpenRouterProvider(api_key='not-used'))
    settings: OpenRouterModelSettings = {'openrouter_cache_messages': '5m'}

    prepared = model.prepare_messages(tool_loop_history(), ModelRequestParameters())
    openai_messages = await model._map_messages(prepared, ModelRequestParameters(), model_settings=settings)  # pyright: ignore[reportPrivateUsage]

    only_reminder = model.prepare_messages(REMINDER_ONLY, ModelRequestParameters())
    only_reminder_messages = await model._map_messages(only_reminder, ModelRequestParameters(), model_settings=settings)  # pyright: ignore[reportPrivateUsage]
    assert only_reminder_messages == snapshot([{'role': 'user', 'content': '<system>Reminder.</system>'}])
    assert openai_messages[-2:] == snapshot(
        [
            {
                'role': 'tool',
                'tool_call_id': 'call_2',
                'content': [{'type': 'text', 'text': 'Oslo: 18C', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}],
            },
            {'role': 'user', 'content': '<system>Reminder 2.</system>'},
        ]
    )
