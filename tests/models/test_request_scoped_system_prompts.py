"""Request-scoped system prompts: `SystemPromptPart(..., scope='request')`.

A request-scoped prompt is a nudge for one model request — a reminder against instruction fade, a note
about the current moment — that stays in the message history like any other part. Two renderings:

* Where the API keeps it and stops rendering it itself (Anthropic's `clear_at: 'next_user_message'`),
  every copy is sent unchanged, current or not. Sending it unchanged is the point: on Claude Opus 5.5,
  Sonnet 5.5 and Fable 5.1, deleting a reminder on the next request invalidates every earlier thinking
  block, and the history stays append-only on the wire, so the cached prefix keeps matching.
* Everywhere else, it's sent while it's current and left out of every later request, rendered at the
  end of the request so automatic cache breakpoints can land before it.

The recorded tests pin both against the real APIs. The unit tests cover what a recording can't see:
the cache anchor on adapters where we can't record, orphaned parts after a failed step, and the core
projection on both sides of the `supports_request_scoped_system_prompts` flag.
"""

from __future__ import annotations as _annotations

import json
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import JsonValue

from pydantic_ai import (
    Agent,
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryFeedbackPart,
    RetryInfo,
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
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequestPart
from pydantic_ai.models import (
    Model,
    ModelRequestContext,
    ModelRequestParameters,
)
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.profiles import ModelProfile
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from tests.function_model_prefix import REQUEST_SCOPED_PROMPT_DROPPED

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, RequestCapture, try_import
from .conftest import cache_breakpoints, message_shape

if TYPE_CHECKING:
    from .conftest import AnthropicModelFactory

with try_import() as anthropic_available:
    from anthropic import AsyncAnthropicBedrock, AsyncAnthropicFoundry, omit as OMIT
    from anthropic.types.beta import BetaMessageParam, BetaTextBlock, BetaToolUseBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.profiles.anthropic import AnthropicModelProfile
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .test_anthropic import (
        MockAnthropic,
        completion_message,
        get_mock_chat_completion_kwargs,
        mock_anthropic_client,
    )

with try_import() as openai_available:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings, OpenAIResponsesModel
    from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.providers.openrouter import OpenRouterProvider

with try_import() as vercel_ai_available:
    from pydantic_ai.ui.vercel_ai import VercelAIAdapter

with try_import() as ag_ui_available:
    from pydantic_ai.ui.ag_ui import AGUIAdapter

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
    """Adds a request-scoped reminder to every model request, the way a harness capability does.

    By default it enqueues the reminder, the documented pattern: content enqueued from
    `before_model_request` goes into this request and into history. With `two_writes`, it writes the
    reminder to both itself, into persistent history through `ctx.messages` and into this request through
    a new `request_context.messages` list, which works too and lands the reminder ahead of later hooks.
    """

    text: str = 'Reminder: be terse. This is request {n}.'
    two_writes: bool = False
    requests: int = field(default=0, init=False)

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self.requests += 1
        part = SystemPromptPart(self.text.format(n=self.requests), scope='request')
        if not self.two_writes:
            ctx.enqueue(part)
            return request_context
        reminder = ModelRequest(parts=[part])
        ctx.messages.append(reminder)
        request_context.messages = [*request_context.messages, reminder]
        return request_context


def weather_agent(model: Any, **kwargs: Any) -> Agent[None, str]:
    agent = Agent(model, system_prompt=SYSTEM_PROMPT, capabilities=[TurnReminder()], **kwargs)

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'{city}: 18C, cloudy'

    return agent


def request_scoped_texts(messages: Sequence[ModelMessage]) -> list[str]:
    return [
        part.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.scope == 'request'
    ]


def without_cache_control(value: Any) -> Any:
    """`value` with every `cache_control` key removed, so wire prefixes compare on content alone."""
    if isinstance(value, dict):
        return {key: without_cache_control(item) for key, item in value.items() if key != 'cache_control'}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [without_cache_control(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


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
async def test_anthropic_request_scoped_reminders_keep_thinking_and_cache(
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

    bodies: list[dict[str, Any]] = request_capture.bodies('/v1/messages')
    assert len(bodies) == len(responses)
    assert all(_TURN_SCOPED_BETA in headers.get('anthropic-beta', '') for headers in request_capture.headers)
    assert_append_only(bodies)
    turn_entries = [message for message in bodies[-1]['messages'] if 'clear_at' in message]
    assert turn_entries == [
        {
            'role': 'system',
            'content': [{'text': f'Reminder: be terse. This is request {n}.', 'type': 'text'}],
            'clear_at': 'next_user_message',
        }
        for n in range(1, len(bodies) + 1)
    ]
    # The breakpoint lands on the tool result ahead of the reminder: `cache_control` on a request-scoped
    # entry is a 400.
    assert [cache_breakpoints(body) for body in bodies[1:]] == [
        (None, [f'messages[{len(body["messages"]) - 2}].content[0]']) for body in bodies[1:]
    ]
    for previous, current in zip(responses, responses[1:]):
        assert current.usage.cache_read_tokens >= previous.usage.input_tokens - 30


@dataclass(frozen=True)
class FallbackCase:
    id: str
    settings: AnthropicModelSettings
    breakpoints: tuple[dict[str, JsonValue] | None, list[str]]


FALLBACK_CASES = [
    FallbackCase(
        id='cache-messages',
        settings={'anthropic_cache_messages': True},
        breakpoints=snapshot((None, ['messages[2].content[0]'])),
    ),
    # The automatic breakpoint would land on the reminder, so an explicit one goes on the block before it.
    FallbackCase(
        id='auto-cache',
        settings={'anthropic_cache': True},
        breakpoints=snapshot(({'type': 'ephemeral', 'ttl': '5m'}, ['messages[2].content[0]'])),
    ),
]


@pytest.mark.moves_cache_prefix(
    reason='a model that cannot clear a request-scoped prompt is sent it only while current, by design'
)
@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in FALLBACK_CASES])
async def test_anthropic_request_scoped_fallback_on_model_without_native_support(
    allow_model_requests: None,
    anthropic_model: AnthropicModelFactory,
    request_capture: RequestCapture,
    case: FallbackCase,
):
    """Claude Sonnet 5 doesn't honor a `system` entry in `messages`: the reminder is sent while current, then left out.

    It renders as `<system>`-tagged text at the end of the request's user turn, and the breakpoint goes
    on the block before it, whether `anthropic_cache_messages` or `anthropic_cache` asked for one: with
    `anthropic_cache` the server would otherwise put its own breakpoint on the reminder, writing an
    entry the next request, which no longer sends it, can't read. With the breakpoint there, each cache
    read covers the previous request bar the reminder. No beta header, no `clear_at`, and no empty
    responses: the text never piles up after the tool results.
    """
    agent = weather_agent(anthropic_model('claude-sonnet-5', capture=True), model_settings=case.settings)

    result = await agent.run(PROMPT)

    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    assert all(response.parts for response in responses)
    bodies: list[dict[str, Any]] = request_capture.bodies('/v1/messages')
    assert all(_TURN_SCOPED_BETA not in headers.get('anthropic-beta', '') for headers in request_capture.headers)
    for n, body in enumerate(bodies, start=1):
        texts = [
            block['text']
            for message in body['messages']
            for block in message['content']
            if block['type'] == 'text' and 'Reminder' in block['text']
        ]
        assert texts == [f'<system>Reminder: be terse. This is request {n}.</system>']
        last_content = body['messages'][-1]['content']
        assert last_content[-1] == {'text': texts[0], 'type': 'text'}
    assert cache_breakpoints(bodies[1]) == case.breakpoints
    for previous, current in zip(responses, responses[1:]):
        assert current.usage.cache_read_tokens >= previous.usage.input_tokens - 30
    # History keeps every reminder; only the wire leaves the finished ones out.
    assert request_scoped_texts(result.all_messages()) == [
        f'Reminder: be terse. This is request {n}.' for n in range(1, len(bodies) + 1)
    ]


@pytest.mark.moves_cache_prefix(
    reason='a model that cannot clear a request-scoped prompt is sent it only while current, by design'
)
@pytest.mark.skipif(not openai_available(), reason='openai not installed')
@pytest.mark.parametrize('api', ['chat', 'responses'])
async def test_openai_request_scoped_reminder_sent_only_while_current(
    allow_model_requests: None, openai_api_key: str, request_capture: RequestCapture, api: str
):
    """OpenAI has no request-scoped channel: each reminder goes out once, as a mid-conversation system message.

    GPT-5.6 takes explicit cache breakpoints, and this is the run that used to crash on Chat
    Completions when a harness reminder led with a `CachePoint`. Nothing here adds one, and each cache
    read still covers the previous request bar the reminder.
    """
    provider = OpenAIProvider(api_key=openai_api_key, http_client=request_capture.client)
    if api == 'chat':
        model = OpenAIChatModel(
            'gpt-5.6', provider=provider, settings=OpenAIChatModelSettings(openai_reasoning_effort='none')
        )
        path = '/chat/completions'
    else:
        model = OpenAIResponsesModel('gpt-5.6', provider=provider)
        path = '/responses'
    agent = weather_agent(model)

    result = await agent.run(PROMPT)

    responses = [message for message in result.all_messages() if isinstance(message, ModelResponse)]
    bodies: list[dict[str, Any]] = request_capture.bodies(path)
    assert len(bodies) == len(responses) >= 3
    for n, body in enumerate(bodies, start=1):
        sent = body['messages'] if api == 'chat' else body['input']
        reminders = [item for item in sent if 'Reminder' in str(item.get('content'))]
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


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_superseded_reminders_are_left_out_without_native_support():
    """A model that can't clear a request-scoped prompt gets each one only while it's current.

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
    assert all(len(request_scoped_texts(messages)) == 0 for messages in seen)
    assert all(
        sum(
            isinstance(part, UserPromptPart) and 'Reminder' in str(part.content)
            for message in messages
            for part in message.parts
        )
        == 1
        for messages in seen
    )
    assert request_scoped_texts(result.all_messages()) == [
        'Reminder: be terse. This is request 1.',
        'Reminder: be terse. This is request 2.',
        'Reminder: be terse. This is request 3.',
    ]


async def test_native_support_sends_every_reminder():
    """A model whose API clears request-scoped prompts itself is sent every one, unchanged and unmoved."""
    seen: list[list[ModelMessage]] = []
    profile = ModelProfile(supports_inline_system_prompts=True, supports_request_scoped_system_prompts=True)
    agent = weather_agent(recording_function_model(seen, profile=profile))

    await agent.run(PROMPT)

    assert [request_scoped_texts(messages) for messages in seen] == [
        ['Reminder: be terse. This is request 1.'],
        ['Reminder: be terse. This is request 1.', 'Reminder: be terse. This is request 2.'],
        [
            'Reminder: be terse. This is request 1.',
            'Reminder: be terse. This is request 2.',
            'Reminder: be terse. This is request 3.',
        ],
    ]


async def test_native_support_leaves_out_a_reminder_no_user_message_clears():
    """A `no_output` retry adds only system-voice feedback, so nothing would clear the previous reminder.

    A model that clears request-scoped prompts does so at the next user message, and the retry request has
    none, so the previous step's reminder is left out of it, and of every later request so the history's
    bytes stay append-only. One followed by a user message, here a tool result, is still sent for the API
    to clear. https://github.com/pydantic/pydantic-ai/issues/10935; the append-only property test pins the
    same case as `retry:no-output-retry+request-scoped`.
    """
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        if len(seen) == 1:
            return ModelResponse(parts=[TextPart('')])
        if len(seen) == 2:
            return ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Paris'})])
        return ModelResponse(parts=[TextPart('done')])

    profile = ModelProfile(supports_inline_system_prompts=True, supports_request_scoped_system_prompts=True)
    agent = weather_agent(FunctionModel(respond, profile=profile))

    await agent.run(PROMPT)

    assert [request_scoped_texts(messages) for messages in seen] == [
        ['Reminder: be terse. This is request 1.'],
        ['Reminder: be terse. This is request 2.'],
        ['Reminder: be terse. This is request 2.', 'Reminder: be terse. This is request 3.'],
    ]
    retry_request = seen[1][-1]
    assert isinstance(retry_request, ModelRequest)
    assert [type(part).__name__ for part in retry_request.parts] == ['SystemPromptPart', 'SystemPromptPart']

    # A request left with nothing but the reminder is left out with it.
    model = FunctionModel(respond, profile=profile)
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content=PROMPT)]),
        ModelRequest(parts=[SystemPromptPart('Reminder 1.', scope='request')]),
        ModelResponse(parts=[TextPart('')]),
        ModelRequest(parts=[RetryFeedbackPart(content='Please return text.', retry=RetryInfo(cause='no_output'))]),
    ]
    prepared = model.prepare_messages(history, ModelRequestParameters())
    assert [type(message).__name__ for message in prepared] == ['ModelRequest', 'ModelResponse', 'ModelRequest']
    assert request_scoped_texts(prepared) == []


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_reminder_enqueued_from_before_model_request_matches_two_writes():
    """`ctx.enqueue` from `before_model_request` delivers into the request being prepared, and into history.

    It's the one-call form of writing the reminder to `ctx.messages` and `request_context.messages`: the model
    is sent the same requests and the run records the same history either way. An enqueue from a hook used to
    wait for the next request, since the queue is drained ahead of the other hooks.
    """
    runs: list[tuple[list[list[ModelMessage]], list[ModelMessage]]] = []
    for two_writes in (False, True):
        seen: list[list[ModelMessage]] = []
        agent = Agent(recording_function_model(seen), capabilities=[TurnReminder(two_writes=two_writes)])

        @agent.tool_plain
        def get_weather(city: str) -> str:
            return f'{city}: 18C, cloudy'

        result = await agent.run(PROMPT)
        runs.append((seen, result.all_messages()))

    (enqueued_seen, enqueued_history), (written_seen, written_history) = runs
    assert [request_scoped_texts(messages) for messages in enqueued_seen] == [[], [], []]
    assert [
        [str(part.content) for part in messages[-1].parts if isinstance(part, UserPromptPart)]
        for messages in enqueued_seen
    ] == [
        [PROMPT, '<system>Reminder: be terse. This is request 1.</system>'],
        ['<system>Reminder: be terse. This is request 2.</system>'],
        ['<system>Reminder: be terse. This is request 3.</system>'],
    ]
    assert [[type(part) for message in messages for part in message.parts] for messages in enqueued_seen] == [
        [type(part) for message in messages for part in message.parts] for messages in written_seen
    ]
    assert request_scoped_texts(enqueued_history) == request_scoped_texts(written_history)
    assert [type(message) for message in enqueued_history] == [type(message) for message in written_history]


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_enqueued_request_scoped_prompt():
    """`ctx.enqueue` delivers a request-scoped prompt into the next request, and it's gone from the one after."""
    seen: list[list[ModelMessage]] = []
    agent = Agent(recording_function_model(seen))

    @agent.tool
    def get_weather(ctx: RunContext[Any], city: str) -> str:
        ctx.enqueue(SystemPromptPart(f'The {city} result is cached; do not fetch it again.', scope='request'))
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
    assert request_scoped_texts(result.all_messages()) == [
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
    """A compaction capability that keeps the current request keeps its request-scoped prompt too.

    The reminder capability writes its reminder itself ahead of compaction, so compaction sees it in the
    last request, the one it keeps; the request goes out with it, and the history still records it.
    """
    seen: list[list[ModelMessage]] = []
    agent = Agent(
        recording_function_model(seen),
        capabilities=[TurnReminder(two_writes=True), KeepLastRequest()],
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
    assert request_scoped_texts(result.all_messages()) == ['Reminder: be terse. This is request 3.']


@dataclass
class ProviderDown(AbstractCapability[Any]):
    """Fails every model request after the reminder was added, before anything is sent."""

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        raise RuntimeError('provider down')


@pytest.mark.parametrize('stream', [False, True])
async def test_failed_step_drops_its_request_scoped_prompt(stream: bool):
    """A step that raises takes its request-scoped prompt with it, so a retry doesn't send it twice.

    Without that, resuming from the captured history would carry the failed step's reminder into the
    next request alongside the one that request adds for itself. (A stream that fails after it opened
    commits its partial response, which answers the request, so that case keeps the reminder.)
    """
    agent = Agent(TestModel(), capabilities=[TurnReminder(two_writes=True), ProviderDown()])

    with capture_run_messages() as messages:
        with pytest.raises(RuntimeError, match='provider down'):
            if stream:
                async with agent.run_stream('hello'):
                    pass  # pragma: no cover
            else:
                await agent.run('hello')

    assert [[type(part).__name__ for part in message.parts] for message in messages] == [['UserPromptPart']]


async def test_failed_step_keeps_the_rest_of_a_request_holding_a_request_scoped_prompt():
    """Only the request-scoped part goes when the request it's in also holds other content.

    Here the prompt arrives in the trailing request of `message_history`, which the run resumes
    without a new prompt; the step fails, and the user prompt next to it stays.
    """
    agent = Agent(TestModel(), capabilities=[ProviderDown()])
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hello'), SystemPromptPart('Be brief.', scope='request')])
    ]

    with capture_run_messages() as messages:
        with pytest.raises(RuntimeError, match='provider down'):
            await agent.run(message_history=history)

    assert [[type(part).__name__ for part in message.parts] for message in messages] == [['UserPromptPart']]


async def test_retry_without_response_drops_the_request_scoped_prompt():
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
            if isinstance(part, UserPromptPart) and 'Reminder' in str(part.content)
        ]
        for messages in seen
    ] == [
        ['<system>Reminder: be terse. This is request 1.</system>'],
        ['<system>Reminder: be terse. This is request 2.</system>'],
    ]
    assert request_scoped_texts(result.all_messages()) == ['Reminder: be terse. This is request 2.']


@pytest.mark.moves_cache_prefix(reason=REQUEST_SCOPED_PROMPT_DROPPED)
async def test_resuming_a_suspended_response_defers_the_enqueued_reminder():
    """Resuming a provider-suspended turn sends the history ending in it unchanged, with nothing added after it.

    That request is a continuation of the paused one, so a reminder a `before_model_request` hook enqueues
    then waits for the next request instead of landing after the suspended response, which the resume
    request has to end with.
    """
    seen: list[list[ModelMessage]] = []
    agent = weather_agent(recording_function_model(seen))

    result = await agent.run(
        message_history=[
            ModelRequest(parts=[UserPromptPart(PROMPT)]),
            ModelResponse(parts=[TextPart('Checking.')], state='suspended'),
        ]
    )

    assert result.output == 'done'
    assert 'Reminder' not in str(seen[0])
    assert [
        [
            str(part.content)
            for part in messages[-1].parts
            if isinstance(part, UserPromptPart) and 'Reminder' in str(part.content)
        ]
        for messages in seen[1:]
    ] == [
        [
            '<system>Reminder: be terse. This is request 1.</system>',
            '<system>Reminder: be terse. This is request 2.</system>',
        ],
        ['<system>Reminder: be terse. This is request 3.</system>'],
    ]
    assert request_scoped_texts(result.all_messages()) == [
        f'Reminder: be terse. This is request {n}.' for n in (1, 2, 3)
    ]


async def test_fallback_attempts_keep_the_request_scoped_prompt():
    """A fallback attempt is still the same step's request, so the next model is sent the same reminder."""
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

    assert [
        [str(part.content) for part in messages[-1].parts if isinstance(part, UserPromptPart)] for messages in seen
    ] == [
        ['hello', '<system>Reminder: be terse. This is request 1.</system>'],
        ['hello', '<system>Reminder: be terse. This is request 1.</system>'],
    ]
    assert request_scoped_texts(result.all_messages()) == ['Reminder: be terse. This is request 1.']


def test_request_scoped_prompt_round_trips_through_serialization():
    """`scope` survives `ModelMessagesTypeAdapter`, which is what durable execution and storage use."""
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('hi'), SystemPromptPart('Be terse.', scope='request')])
    ]

    assert ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages)) == messages


def tool_loop_history() -> list[ModelMessage]:
    """Two tool rounds, each with a reminder appended as its own request, the way `TurnReminder` writes them."""
    return [
        ModelRequest(parts=[UserPromptPart('Weather in Paris and Oslo?')]),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_1')]),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Paris: 18C', tool_call_id='call_1')]),
        ModelRequest(parts=[SystemPromptPart('Reminder 1.', scope='request')]),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Oslo'}, tool_call_id='call_2')]),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'Oslo: 18C', tool_call_id='call_2')]),
        ModelRequest(parts=[SystemPromptPart('Reminder 2.', scope='request')]),
    ]


REMINDER_ONLY: list[ModelMessage] = [ModelRequest(parts=[SystemPromptPart('Reminder.', scope='request')])]

WEATHER_PARAMETERS = ModelRequestParameters(
    function_tools=[
        ToolDefinition(
            name='get_weather',
            parameters_json_schema={'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']},
        )
    ]
)


@contextmanager
def converse_request_bodies(model: BedrockConverseModel) -> Generator[list[dict[str, Any]]]:
    """The Converse request bodies as sent, read off boto's `before-send` event, which fires on cassette replay too.

    `request()` uses `ConverseStream` for a Claude model whose maximum output it sends as the default `maxTokens`.
    """
    captured: list[dict[str, Any]] = []

    def capture(request: Any, **_: Any) -> None:
        captured.append(json.loads(request.body))

    events = ['before-send.bedrock-runtime.Converse', 'before-send.bedrock-runtime.ConverseStream']
    for event in events:
        model.client.meta.events.register_last(event, capture)
    try:
        yield captured
    finally:
        for event in events:
            model.client.meta.events.unregister(event, capture)


async def send(model: Model, history: list[ModelMessage], settings: ModelSettings | None = None) -> None:
    """Send `history` as one request through the model's public path, as the agent graph does."""
    await model.request(model.prepare_messages(history, WEATHER_PARAMETERS), settings, WEATHER_PARAMETERS)


def test_current_request_scoped_prompts_move_to_the_last_request():
    """Request-scoped prompts anywhere in the current request go to the end of its last request.

    The agent merges adjacent requests before preparing them, so this is about `prepare_messages`
    called directly on a history that isn't merged: a request holding nothing but a moved prompt is
    left out, and one with other parts keeps them.
    """
    model = FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart('done')]))
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hello'), SystemPromptPart('Reminder A.', scope='request')]),
        ModelRequest(parts=[SystemPromptPart('Reminder B.', scope='request')]),
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
                metadata={'__pydantic_ai__': {'request_scoped_tail': 2}},
            ),
        ]
    )


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_request_scoped_opt_out_through_the_profile(
    allow_model_requests: None, anthropic_api_key: str, request_capture: RequestCapture
):
    """Setting the profile flag to `False` on a model that has it opts out of the beta.

    For an account without access to it: the reminder is then sent only while it's current, as an
    ordinary `system` entry (the model still takes the role), without `clear_at` or the beta header.
    """
    provider = AnthropicProvider(api_key=anthropic_api_key, http_client=request_capture.client)
    model = AnthropicModel(
        'claude-opus-5-5',
        provider=provider,
        profile=AnthropicModelProfile(supports_request_scoped_system_prompts=False),
    )
    native = AnthropicModel('claude-opus-5-5', provider=provider)
    foundry = AnthropicModel(
        'claude-opus-5-5',
        provider=AnthropicProvider(anthropic_client=AsyncAnthropicFoundry(api_key='x', base_url='https://example.com')),
    )
    assert model.profile.get('supports_inline_system_prompts') is True
    assert native.profile.get('supports_request_scoped_system_prompts') is True
    # Microsoft Foundry serves no `system` role, so it can't serve a request-scoped one either.
    assert foundry.profile.get('supports_request_scoped_system_prompts') is False

    await send(model, tool_loop_history())

    body = request_capture.body('/v1/messages')
    assert message_shape(body) == snapshot(
        [
            ('user', ['text']),
            ('assistant', ['tool_use']),
            ('user', ['tool_result']),
            ('assistant', ['tool_use']),
            ('user', ['tool_result']),
            ('system', ['text']),
        ]
    )
    messages = cast(list[dict[str, Any]], body['messages'])
    assert messages[-1] == {'role': 'system', 'content': [{'text': 'Reminder 2.', 'type': 'text'}]}
    assert _TURN_SCOPED_BETA not in request_capture.headers[0].get('anthropic-beta', '')


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_leading_request_scoped_prompt_is_not_hoisted(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """A request-scoped prompt opening the first request is not part of the standing system prompt.

    The top-level `system` parameter goes out with every request, so hoisting it there would make it
    permanent; it's sent as a `clear_at` entry after the user turn instead, with the beta header.
    """
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                SystemPromptPart('You are terse.'),
                SystemPromptPart('Answer in French this turn.', scope='request'),
                UserPromptPart('Hello'),
            ]
        )
    ]

    await send(anthropic_model('claude-opus-5-5', capture=True), messages)

    body = request_capture.body('/v1/messages')
    assert body['system'] == 'You are terse.'
    assert body['messages'] == snapshot(
        [
            {'role': 'user', 'content': [{'text': 'Hello', 'type': 'text'}]},
            {
                'role': 'system',
                'content': [{'text': 'Answer in French this turn.', 'type': 'text'}],
                'clear_at': 'next_user_message',
            },
        ]
    )
    assert _TURN_SCOPED_BETA in request_capture.headers[0].get('anthropic-beta', '')


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_bedrock_message_cache_point_goes_before_the_request_scoped_prompt(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """`bedrock_cache_messages` puts its cache point before the reminder, not after it.

    The Converse API has no `system` role in `messages`, so the reminder is `<system>`-tagged text that
    ends the last user turn and is gone from the next request; a cache point after it would write an
    entry that request can't read. With nothing ahead of the reminder in its message, there's nothing
    to cache up to.
    """
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    settings: BedrockModelSettings = {'bedrock_cache_messages': True}

    with converse_request_bodies(model) as bodies:
        await send(model, tool_loop_history(), settings)
        await send(model, REMINDER_ONLY, settings)

    tool_loop, only_reminder = bodies
    assert tool_loop['messages'][-1]['content'] == snapshot(
        [
            {'toolResult': {'toolUseId': 'call_2', 'content': [{'text': 'Oslo: 18C'}], 'status': 'success'}},
            {'cachePoint': {'type': 'default'}},
            {'text': '<system>Reminder 2.</system>'},
        ]
    )
    assert 'Reminder 1.' not in str(tool_loop['messages'])
    assert only_reminder['messages'] == snapshot(
        [{'role': 'user', 'content': [{'text': '<system>Reminder.</system>'}]}]
    )


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_openrouter_message_cache_control_goes_before_the_request_scoped_prompt(
    allow_model_requests: None, openrouter_api_key: str, request_capture: RequestCapture
):
    """`openrouter_cache_messages` marks the message before the reminder, not the reminder.

    OpenRouter takes no `system` entries mid-conversation, so the reminder is its own `<system>`-tagged
    user message at the end, and it's gone from the next request. With nothing ahead of the reminder,
    nothing is marked.
    """
    model = OpenRouterModel(
        'anthropic/claude-haiku-5.5',
        provider=OpenRouterProvider(api_key=openrouter_api_key, http_client=request_capture.client),
    )
    settings: OpenRouterModelSettings = {'openrouter_cache_messages': '5m'}

    await send(model, tool_loop_history(), settings)
    await send(model, REMINDER_ONLY, settings)

    tool_loop, only_reminder = request_capture.bodies('/chat/completions')
    assert cast(list[dict[str, Any]], tool_loop['messages'])[-2:] == snapshot(
        [
            {
                'role': 'tool',
                'tool_call_id': 'call_2',
                'content': [{'type': 'text', 'text': 'Oslo: 18C', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}],
            },
            {'role': 'user', 'content': '<system>Reminder 2.</system>'},
        ]
    )
    assert only_reminder['messages'] == snapshot([{'role': 'user', 'content': '<system>Reminder.</system>'}])


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_no_output_retry_leaves_out_the_reminder_nothing_clears(allow_model_requests: None):
    """On the wire: the reminder a `no_output` retry would leave visible is absent from that request and the next.

    The retry request holds only system-voice entries, so no user message would clear the first reminder
    there; the reminder sent with the retry is followed by a tool result, so it still goes out with
    `clear_at` for the API to clear. https://github.com/pydantic/pydantic-ai/issues/10935
    """
    usage = BetaUsage(input_tokens=10, output_tokens=5)
    mock_client = MockAnthropic.create_mock(
        [
            completion_message([BetaTextBlock(text='', type='text')], usage),
            completion_message(
                [BetaToolUseBlock(id='call_1', input={'city': 'Paris'}, name='get_weather', type='tool_use')], usage
            ),
            completion_message([BetaTextBlock(text='Done.', type='text')], usage),
        ]
    )
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = weather_agent(model)

    await agent.run(PROMPT)

    bodies = get_mock_chat_completion_kwargs(mock_client)
    assert [
        [(message['role'], message.get('clear_at'), message['content']) for message in body['messages'][1:]]
        for body in bodies
    ] == snapshot(
        [
            [('system', 'next_user_message', [{'text': 'Reminder: be terse. This is request 1.', 'type': 'text'}])],
            [
                ('system', None, [{'text': 'Please return text or call a tool.', 'type': 'text'}]),
                ('system', 'next_user_message', [{'text': 'Reminder: be terse. This is request 2.', 'type': 'text'}]),
            ],
            [
                ('system', None, [{'text': 'Please return text or call a tool.', 'type': 'text'}]),
                ('system', 'next_user_message', [{'text': 'Reminder: be terse. This is request 2.', 'type': 'text'}]),
                (
                    'assistant',
                    None,
                    [{'id': 'call_1', 'type': 'tool_use', 'name': 'get_weather', 'input': {'city': 'Paris'}}],
                ),
                (
                    'user',
                    None,
                    [
                        {
                            'tool_use_id': 'call_1',
                            'type': 'tool_result',
                            'content': [{'text': 'Paris: 18C, cloudy', 'type': 'text'}],
                            'is_error': False,
                        }
                    ],
                ),
                ('system', 'next_user_message', [{'text': 'Reminder: be terse. This is request 3.', 'type': 'text'}]),
            ],
        ]
    )
    # The first reminder was the first request's tail, after its breakpoint, so leaving it out moves nothing
    # cached; from the retry request on, the conversation only grows.
    assert_append_only(bodies[1:])


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_anthropic_unified_cache_skips_the_request_scoped_entry(allow_model_requests: None):
    """`cache=True` takes the same breakpoint path as the `anthropic_cache*` settings, so it honors request-scoped entries.

    It turns on automatic caching plus instruction and tool breakpoints. The server's automatic breakpoint skips a
    `clear_at` entry, so an explicit `CachePoint` on the block before one is the block automatic caching would mark:
    a `1h` breakpoint there with a `5m` automatic one is a 400, so the explicit one wins and the top-level one is
    left out, as it is for any other last block. A unit test: the claim is the request as built.
    """
    reply = completion_message([BetaTextBlock(text='Done.', type='text')], BetaUsage(input_tokens=10, output_tokens=5))
    mock_client = MockAnthropic.create_mock([reply, reply])
    model = AnthropicModel('claude-opus-5-5', provider=AnthropicProvider(anthropic_client=mock_client))
    agent = Agent(model, instructions='Be terse.', capabilities=[TurnReminder()], model_settings={'cache': True})

    await agent.run('Hello')
    await agent.run(['Hello', CachePoint(ttl='1h')])

    first, second = get_mock_chat_completion_kwargs(mock_client)
    for n, kwargs in enumerate((first, second), start=1):
        assert kwargs['messages'][-1] == {
            'role': 'system',
            'content': [{'text': f'Reminder: be terse. This is request {n}.', 'type': 'text'}],
            'clear_at': 'next_user_message',
        }
    assert first['cache_control'] == {'type': 'ephemeral', 'ttl': '5m'}
    assert first['system'] == snapshot(
        [{'type': 'text', 'text': 'Be terse.', 'cache_control': {'type': 'ephemeral', 'ttl': '5m'}}]
    )
    assert second['cache_control'] is OMIT
    # The instructions' breakpoint is raised to `1h`: a `1h` one can't come after a `5m` one.
    assert second['system'] == snapshot(
        [{'type': 'text', 'text': 'Be terse.', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}]
    )
    assert second['messages'][0]['content'] == snapshot(
        [{'text': 'Hello', 'type': 'text', 'cache_control': {'type': 'ephemeral', 'ttl': '1h'}}]
    )


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
def test_anthropic_bedrock_lookback_breakpoint_skips_the_request_scoped_entry():
    """On Bedrock, the previous request's end gets a breakpoint after a wide turn, but never on its `clear_at` entry.

    The previous request put its breakpoint on the block before its request-scoped entry (`cache_control` on one is a
    400), so that's the block the lookback breakpoint goes on, with the entry's blocks counted toward the lookback.
    Driven through the placement method on hand-built wire messages, as the existing lookback test in
    `test_anthropic.py` is: the property is about the previous request, which one request can't show, and the Bedrock
    client is a mock.
    """
    model = AnthropicModel(
        'claude-opus-5-5',
        provider=AnthropicProvider(
            anthropic_client=mock_anthropic_client(
                AsyncAnthropicBedrock, 'https://bedrock-runtime.us-east-1.amazonaws.com'
            )
        ),
    )
    messages: list[BetaMessageParam] = [
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Check the weather everywhere.'}]},
        {'role': 'system', 'content': [{'type': 'text', 'text': 'Reminder 1.'}], 'clear_at': 'next_user_message'},
        {
            'role': 'assistant',
            'content': [{'type': 'tool_use', 'id': f'call_{i}', 'name': 'get_weather', 'input': {}} for i in range(12)],
        },
        {
            'role': 'user',
            'content': [{'type': 'tool_result', 'tool_use_id': f'call_{i}', 'content': 'Sunny'} for i in range(12)],
        },
        {'role': 'system', 'content': [{'type': 'text', 'text': 'Reminder 2.'}], 'clear_at': 'next_user_message'},
    ]

    model._apply_message_cache_control(messages, '5m')  # pyright: ignore[reportPrivateUsage]

    assert [
        (index, block_index)
        for index, message in enumerate(messages)
        for block_index, block in enumerate(cast(list[dict[str, Any]], message['content']))
        if 'cache_control' in block
    ] == [(0, 0), (3, 11)]


def duplicate_text_history(lasting: ModelRequestPart | None = None) -> list[ModelMessage]:
    """A tool round whose lasting text is the same as the request-scoped reminder after it, as rendered.

    By default the lasting text is user text. On a model without mid-conversation system messages, user text can't
    carry a `<system>` tag, so `lasting` there is a lasting `SystemPromptPart` with the reminder's text.
    """
    return [
        ModelRequest(parts=[UserPromptPart('Weather in Paris?')]),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_1')]),
        ModelRequest(
            parts=[
                ToolReturnPart('get_weather', 'Paris: 18C', tool_call_id='call_1'),
                lasting or UserPromptPart('<system>Reminder.</system>'),
            ]
        ),
        ModelRequest(parts=[SystemPromptPart('Reminder.', scope='request')]),
    ]


@pytest.mark.skipif(not anthropic_available(), reason='anthropic not installed')
async def test_lasting_text_equal_to_a_reminder_keeps_the_breakpoint_anthropic(
    allow_model_requests: None, anthropic_model: AnthropicModelFactory, request_capture: RequestCapture
):
    """Only the reminder ending the request is skipped: lasting text with the same content gets the breakpoint.

    With nothing ahead of the reminder, there's nothing to cache up to.
    """
    model = anthropic_model('claude-sonnet-5', capture=True)
    settings: AnthropicModelSettings = {'anthropic_cache_messages': True}

    await send(model, duplicate_text_history(), settings)
    await send(model, REMINDER_ONLY, settings)

    duplicate, only_reminder = request_capture.bodies('/v1/messages')
    assert message_shape(duplicate) == snapshot(
        [('user', ['text']), ('assistant', ['tool_use']), ('user', ['tool_result', 'text']), ('user', ['text'])]
    )
    assert cache_breakpoints(duplicate) == snapshot((None, ['messages[2].content[1]']))
    assert cache_breakpoints(only_reminder) == snapshot((None, []))


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_lasting_text_equal_to_a_reminder_keeps_the_cache_point_bedrock(
    allow_model_requests: None, bedrock_provider: BedrockProvider
):
    """Only the reminder ending the request is skipped: lasting text with the same content stays ahead of the cache point."""
    model = BedrockConverseModel('us.anthropic.claude-haiku-4-5-20251001-v1:0', provider=bedrock_provider)
    settings: BedrockModelSettings = {'bedrock_cache_messages': True}

    with converse_request_bodies(model) as bodies:
        await send(model, duplicate_text_history(SystemPromptPart('Reminder.')), settings)

    assert bodies[0]['messages'][-1]['content'][-3:] == [
        {'text': '<system>Reminder.</system>'},
        {'cachePoint': {'type': 'default'}},
        {'text': '<system>Reminder.</system>'},
    ]


@pytest.mark.skipif(not openai_available(), reason='openai not installed')
async def test_lasting_text_equal_to_a_reminder_keeps_the_cache_control_openrouter(
    allow_model_requests: None, openrouter_api_key: str, request_capture: RequestCapture
):
    """Only the reminder message ending the request is skipped: a lasting one with the same content is marked."""
    model = OpenRouterModel(
        'anthropic/claude-haiku-5.5',
        provider=OpenRouterProvider(api_key=openrouter_api_key, http_client=request_capture.client),
    )
    settings: OpenRouterModelSettings = {'openrouter_cache_messages': '5m'}

    await send(model, duplicate_text_history(), settings)

    messages = cast(list[dict[str, Any]], request_capture.body('/chat/completions')['messages'])
    assert [index for index, message in enumerate(messages) if 'cache_control' in str(message)] == [len(messages) - 2]
    assert messages[-1] == {'role': 'user', 'content': '<system>Reminder.</system>'}


def ui_history() -> list[ModelMessage]:
    return [
        ModelRequest(parts=[SystemPromptPart('You are terse.'), UserPromptPart('Hello')]),
        ModelRequest(parts=[SystemPromptPart('Answer in French this turn.', scope='request')]),
        ModelResponse(parts=[TextPart('Bonjour')]),
    ]


@pytest.mark.skipif(not vercel_ai_available(), reason='vercel ai adapter not installed')
def test_vercel_ai_dump_leaves_out_request_scoped_prompts():
    """A request-scoped prompt is for the model's current request, so it's not sent to the frontend.

    Loaded back from the frontend, it would come back as a lasting system prompt.
    """
    dumped = VercelAIAdapter.dump_messages(ui_history())

    assert [message.model_dump(exclude_none=True) for message in dumped] == snapshot(
        [
            {'id': IsStr(), 'role': 'system', 'parts': [{'type': 'text', 'text': 'You are terse.', 'state': 'done'}]},
            {'id': IsStr(), 'role': 'user', 'parts': [{'type': 'text', 'text': 'Hello', 'state': 'done'}]},
            {
                'id': IsStr(),
                'role': 'assistant',
                'metadata': {'pydantic_ai': {'timestamp': IsStr()}},
                'parts': [{'type': 'text', 'text': 'Bonjour', 'state': 'done'}],
            },
        ]
    )


@pytest.mark.skipif(not ag_ui_available(), reason='ag-ui not installed')
def test_ag_ui_dump_leaves_out_request_scoped_prompts():
    """A request-scoped prompt is for the model's current request, so it's not sent to the frontend."""
    dumped = AGUIAdapter.dump_messages(ui_history())

    assert [message.model_dump(exclude_none=True) for message in dumped] == snapshot(
        [
            {'id': IsStr(), 'role': 'system', 'content': 'You are terse.'},
            {'id': IsStr(), 'role': 'user', 'content': 'Hello'},
            {'id': IsStr(), 'role': 'assistant', 'content': 'Bonjour'},
        ]
    )
