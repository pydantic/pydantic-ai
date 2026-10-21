from __future__ import annotations as _annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from decimal import Decimal
from functools import partial

import httpx2
import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRequest,
    ModelResponse,
    RunContext,
    TextPart,
    ThinkingPart,
    Tool,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import Capability, SelectModel, ToolSearch
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import ModelMessage, NativeToolSearchCallPart, NativeToolSearchReturnPart
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.output import NativeOutput
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.run import AgentRunResult, AgentRunResultEvent
from pydantic_ai.usage import RequestUsage

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsStr, try_import
from .conftest import RequestCapture, json_objects
from .mock_openai import completion_message

with try_import() as imports_successful:
    from openai.types.chat.chat_completion_message import ChatCompletionMessage
    from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
    from openai.types.chat.chat_completion_message_tool_call import Function

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.deepseek import DeepSeekProvider
    from pydantic_ai.providers.openai import OpenAIProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
]


@pytest.fixture
def freeze_deepseek_off_peak_pricing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make DeepSeek's time-of-day pricing deterministic for snapshot assertions."""
    timestamp = datetime(2026, 8, 13, tzinfo=UTC)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return timestamp if tz is not None else timestamp.replace(tzinfo=None)

    monkeypatch.setattr('pydantic_ai._utils.datetime', FrozenDatetime)


@pytest.mark.moves_cache_prefix(reason='dynamic tool disclosure after ToolSearch discovery')
async def test_deepseek_deferred_capability_with_thinking(allow_model_requests: None, deepseek_api_key: str):
    """Regression test for #5829: real-API check that deferred capabilities work on a DeepSeek thinking model.

    Loading a deferred capability used to inject a framework-synthesized `search_tools` assistant
    turn with tool calls but no thinking, which DeepSeek rejected with a 400. The reveal now goes out
    as `<system>`-tagged user text instead, so this recording guards the end-to-end deferred-capability
    run rather than the empty `reasoning_content` backfill; the mapping guard for that is
    `test_openai.py::test_field_mode_thinking_backfill_on_synthetic_tool_search_turn`.
    """
    model = OpenAIChatModel('deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key))

    def roll_dice() -> str:
        """Roll a six-sided die and return the result."""
        return '4'

    def get_player_name(ctx: RunContext[str]) -> str:
        """Get the player's name."""
        return ctx.deps

    agent = Agent(
        model,
        deps_type=str,
        instructions=(
            "You're a dice game, you should roll the die and see if the number you get back "
            "matches the user's guess. If so, tell them they're a winner. Use the player's name "
            'in the response.'
        ),
        capabilities=[Capability[str](id='DICE_ROLL', tools=[get_player_name, roll_dice], defer_loading=True)],
    )

    result = await agent.run('My guess is 4', deps='Anne')

    # The run completing at all is the core regression signal — it 400'd before the fix. The
    # structural checks make sure the recording exercised the deferred + thinking path rather than
    # the model answering directly (which would leave the bug untested).
    assert isinstance(result.output, str) and result.output
    messages = result.all_messages()
    assert any(
        isinstance(part, ToolCallPart) and part.tool_name == 'load_capability'
        for message in messages
        for part in message.parts
    ), 'expected the model to call `load_capability`; the deferred path was not exercised'
    assert any(isinstance(part, ThinkingPart) for message in messages for part in message.parts), (
        'expected a `ThinkingPart`; thinking was not exercised, so the reasoning_content round-trip is untested'
    )


_WEATHER_TURN: list[ModelMessage] = [
    ModelRequest(parts=[UserPromptPart('What is the weather in Paris and then London? One tool call per city.')]),
    ModelResponse(
        parts=[
            ThinkingPart('Check Paris first.', id='reasoning_content', provider_name='openai'),
            ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_1'),
        ],
        provider_name='openai',
    ),
    ModelRequest(parts=[ToolReturnPart('get_weather', 'sunny', tool_call_id='call_1')]),
    ModelResponse(
        parts=[ToolCallPart('get_weather', {'city': 'London'}, tool_call_id='call_2')], provider_name='openai'
    ),
    ModelRequest(parts=[ToolReturnPart('get_weather', 'rainy', tool_call_id='call_2')]),
]


@dataclass(frozen=True)
class AutoBackfillCase:
    id: str
    history: list[ModelMessage]
    prompt: str | None
    sent_reasoning: list[str | None]
    output: str


AUTO_BACKFILL_CASES = [
    AutoBackfillCase(
        id='current_turn',
        history=_WEATHER_TURN,
        prompt=None,
        sent_reasoning=['Check Paris first.', ''],
        output=snapshot("""\
Here's the weather for both cities:

- **Paris:** Sunny ☀️
- **London:** Rainy 🌧️

Let me know if you'd like the weather for any other cities.\
"""),
    ),
    AutoBackfillCase(
        id='earlier_turn',
        history=[
            *_WEATHER_TURN,
            ModelResponse(
                parts=[
                    ThinkingPart('Both done.', id='reasoning_content', provider_name='openai'),
                    TextPart('Paris is sunny and London is rainy.'),
                ],
                provider_name='openai',
            ),
        ],
        prompt='Which city should I visit? One word.',
        sent_reasoning=['Check Paris first.', None, 'Both done.'],
        output=snapshot('Paris'),
    ),
]


@pytest.mark.parametrize('case', AUTO_BACKFILL_CASES, ids=lambda case: case.id)
async def test_deepseek_auto_mode_thinking_backfill(
    case: AutoBackfillCase, allow_model_requests: None, deepseek_api_key: str, request_capture: RequestCapture
):
    """DeepSeek through a generic `OpenAIProvider` accepts what `'auto'` sends for a tool call that had no thinking.

    A generic provider leaves `openai_chat_send_back_thinking_parts` at `'auto'`. The London tool call
    carries no `ThinkingPart`, as when another model served that step. In the current user turn DeepSeek
    rejects it with a 400 unless it carries `reasoning_content`, so `'auto'` sends the field the Paris
    call already carries, empty. DeepSeek's thinking-mode guide asks for the field on every earlier turn
    too, but accepts that call without it once a newer user message follows, which is what `'auto'`
    sends there. The run completing is the API accepting the request; the request assertion pins it.
    """
    provider = OpenAIProvider(
        base_url='https://api.deepseek.com', api_key=deepseek_api_key, http_client=request_capture.client
    )
    agent = Agent(OpenAIChatModel('deepseek-v4-flash', provider=provider))

    @agent.tool_plain
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return 'sunny'  # pragma: no cover

    result = await agent.run(case.prompt, message_history=case.history)

    assert result.output == case.output
    messages = request_capture.body('/chat/completions')['messages']
    assert [
        message.get('reasoning_content') for message in json_objects(messages) if message['role'] == 'assistant'
    ] == case.sent_reasoning


DeepSeekModelFactory = Callable[[bool], 'OpenAIChatModel']
"""Builds a `DeepSeekProvider` model: `False` turns thinking off, `True` leaves the model's default."""


def _deepseek_model_factory(
    model_name: str, *, api_key: str, http_client: httpx2.AsyncClient, profile: OpenAIModelProfile | None = None
) -> DeepSeekModelFactory:
    def build(thinking: bool) -> OpenAIChatModel:
        return OpenAIChatModel(
            model_name,
            provider=DeepSeekProvider(api_key=api_key, http_client=http_client),
            profile=profile,
            settings=None if thinking else OpenAIChatModelSettings(thinking=False),
        )

    return build


_PARIS_PROMPT = 'What is the weather in Paris? Use the get_weather tool, then answer in one short sentence.'


def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return 'sunny'


async def _fallback(deepseek: DeepSeekModelFactory, *, thinking: bool = True) -> str:
    """A non-thinking primary serves the tool call, then fails over to DeepSeek."""
    primary_calls = 0

    def primary(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal primary_calls
        primary_calls += 1
        if primary_calls == 1:
            return ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_primary')])
        raise ModelHTTPError(status_code=503, model_name='primary')

    model = FallbackModel(FunctionModel(primary), deepseek(thinking))
    result = await Agent(model, tools=[get_weather]).run(_PARIS_PROMPT)
    return result.output


async def _model_switch(deepseek: DeepSeekModelFactory) -> None:
    """The run switches to a thinking model after a step served without thinking."""
    without_thinking, with_thinking = deepseek(False), deepseek(True)
    agent = Agent(
        tools=[get_weather],
        capabilities=[SelectModel(lambda ctx: without_thinking if ctx.run_step == 1 else with_thinking)],
    )
    await agent.run(_PARIS_PROMPT)


async def _resume(deepseek: DeepSeekModelFactory) -> None:
    """A run without thinking stops for tool approval, and the approved run resumes with thinking."""
    tools = [Tool(get_weather, requires_approval=True)]
    first = await Agent(deepseek(False), tools=tools, output_type=[str, DeferredToolRequests]).run(_PARIS_PROMPT)
    assert isinstance(first.output, DeferredToolRequests)
    approvals = DeferredToolResults(approvals={call.tool_call_id: True for call in first.output.approvals})
    await Agent(deepseek(True), tools=tools, output_type=[str, DeferredToolRequests]).run(
        message_history=first.all_messages(), deferred_tool_results=approvals
    )


async def _cross_provider(deepseek: DeepSeekModelFactory) -> None:
    """An Anthropic step that searched for the tool and called it, continued on DeepSeek."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(_PARIS_PROMPT)]),
        ModelResponse(
            parts=[
                NativeToolSearchCallPart(
                    args={'queries': ['weather']}, tool_call_id='srvtoolu_1', provider_name='anthropic'
                ),
                NativeToolSearchReturnPart(
                    content={'discovered_tools': [{'name': 'get_weather'}]},
                    tool_call_id='srvtoolu_1',
                    provider_name='anthropic',
                ),
                ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='toolu_2'),
            ],
            provider_name='anthropic',
        ),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'sunny', tool_call_id='toolu_2')]),
    ]
    agent = Agent(deepseek(True), tools=[Tool(get_weather, defer_loading=True)], capabilities=[ToolSearch()])
    await agent.run(message_history=history)


async def _earlier_turn(deepseek: DeepSeekModelFactory) -> str:
    """The tool call without thinking sits before a newer user message."""
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(_PARIS_PROMPT)]),
        ModelResponse(parts=[ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_primary')]),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'sunny', tool_call_id='call_primary')]),
    ]
    agent = Agent(deepseek(True), tools=[get_weather])
    result = await agent.run('Thanks! Reply with one word.', message_history=history)
    return result.output


@dataclass(frozen=True)
class FieldBackfillCase:
    id: str
    run: Callable[[DeepSeekModelFactory], Awaitable[object]]
    sent_reasoning: list[str | None]
    model_name: str = 'deepseek-v4-flash'
    profile: OpenAIModelProfile | None = None


FIELD_BACKFILL_CASES = [
    FieldBackfillCase(id='model_switch', run=_model_switch, sent_reasoning=['']),
    FieldBackfillCase(id='resume', run=_resume, sent_reasoning=['']),
    # The Anthropic native tool search goes out as a synthesized `search_tools` turn ahead of the call.
    FieldBackfillCase(id='cross_provider', run=_cross_provider, sent_reasoning=['', '']),
    FieldBackfillCase(id='thinking_off', run=partial(_fallback, thinking=False), sent_reasoning=[None]),
    # `DeepSeekProvider` turns thinking on by default only for the V4 models and `deepseek-reasoner`.
    FieldBackfillCase(id='default_thinking_off', run=_fallback, sent_reasoning=[None], model_name='deepseek-chat'),
    # Without the flag, as on the other `'field'`-mode providers, only a `ThinkingPart` in the history backfills.
    FieldBackfillCase(
        id='flag_off',
        run=_fallback,
        sent_reasoning=[None],
        profile=OpenAIModelProfile(openai_chat_thinking_requires_field_on_tool_calls=False),
    ),
]


@pytest.mark.vcr(ignore_hosts=['api.deepseek.com'])
@pytest.mark.parametrize('case', FIELD_BACKFILL_CASES, ids=lambda case: case.id)
async def test_deepseek_field_mode_thinking_backfill_without_thinking_part(
    allow_model_requests: None, case: FieldBackfillCase
):
    """A thinking request sends `reasoning_content` empty on every tool call that lacks it, before any step has thought.

    `DeepSeekProvider` uses `'field'` mode and sets `openai_chat_thinking_requires_field_on_tool_calls`. Each
    path reaches DeepSeek with thinking on and no `ThinkingPart` in the history, carrying a tool call from a
    step that didn't think, which DeepSeek 400s on without the field when it follows the last user message.
    The controls pin what stays unchanged: a request with thinking turned off, a model that doesn't think by
    default, and a profile without the flag. The `FallbackModel` path and a tool call before a newer user
    message are recorded by `test_deepseek_field_mode_thinking_backfill`, with DeepSeek accepting the field;
    these cases only pin which other paths send it, so a mock transport stands in for a recording per path,
    which would add no evidence about acceptance.
    """
    request_bodies: list[dict[str, JsonValue]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = TypeAdapter(dict[str, JsonValue]).validate_json(request.content)
        request_bodies.append(body)
        if any(message['role'] == 'tool' for message in json_objects(body['messages'])):
            message = ChatCompletionMessage(content='Sunny.', role='assistant')
        else:
            tool_call = ChatCompletionMessageFunctionToolCall(
                id='call_deepseek',
                function=Function(name='get_weather', arguments='{"city": "Paris"}'),
                type='function',
            )
            message = ChatCompletionMessage(content=None, role='assistant', tool_calls=[tool_call])
        return httpx2.Response(200, json=completion_message(message).model_dump(mode='json'))

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
        await case.run(
            _deepseek_model_factory(case.model_name, api_key='test', http_client=client, profile=case.profile)
        )

    assert [
        message.get('reasoning_content')
        for message in json_objects(request_bodies[-1]['messages'])
        if message['role'] == 'assistant'
    ] == case.sent_reasoning


@dataclass(frozen=True)
class LiveFieldBackfillCase:
    id: str
    run: Callable[[DeepSeekModelFactory], Awaitable[str]]
    output: str


LIVE_FIELD_BACKFILL_CASES = [
    LiveFieldBackfillCase(id='current_turn', run=_fallback, output=snapshot('The weather in Paris is sunny.')),
    # Earlier turns get the field too, so a turn goes out the same way in every later request.
    LiveFieldBackfillCase(id='earlier_turn', run=_earlier_turn, output=snapshot('Sunny.')),
]


@pytest.mark.parametrize('case', LIVE_FIELD_BACKFILL_CASES, ids=lambda case: case.id)
async def test_deepseek_field_mode_thinking_backfill(
    case: LiveFieldBackfillCase, allow_model_requests: None, deepseek_api_key: str, request_capture: RequestCapture
):
    """DeepSeek accepts the empty `reasoning_content` a thinking request sends on a tool call no step thought on.

    The history holds no `ThinkingPart`, and DeepSeek has thinking on. In `current_turn`, a non-thinking
    `FallbackModel` primary serves the tool call, then fails over to DeepSeek; without the field DeepSeek
    rejects that request with a 400 ("The reasoning_content in the thinking mode must be passed back to the
    API"). In `earlier_turn`, a newer user message follows the tool call, which DeepSeek also accepts without
    the field. The run completing is the API accepting the request; the request assertion pins it.
    """
    output = await case.run(
        _deepseek_model_factory('deepseek-v4-flash', api_key=deepseek_api_key, http_client=request_capture.client)
    )

    assert output == case.output
    messages = request_capture.body('/chat/completions')['messages']
    assert [
        message.get('reasoning_content') for message in json_objects(messages) if message['role'] == 'assistant'
    ] == ['']


async def test_deepseek_model_thinking_part(
    allow_model_requests: None, deepseek_api_key: str, freeze_deepseek_off_peak_pricing: None
):
    deepseek_model = OpenAIChatModel('deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)
    result = await agent.run('How do I cross the street?')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='How do I cross the street?', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(content=IsStr(), id='reasoning_content', provider_name='deepseek'),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=90,
                    output_tokens=166,
                    details={
                        'prompt_cache_hit_tokens': 0,
                        'prompt_cache_miss_tokens': 90,
                        'reasoning_tokens': 59,
                    },
                    output_reasoning_tokens=59,
                    cost=Decimal('0.00005908'),
                ),
                model_name='deepseek-v4-flash',
                timestamp=IsDatetime(),
                provider_name='deepseek',
                provider_url='https://api.deepseek.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='32beb152-2946-409a-8986-7f0e1e351ed2',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


async def test_deepseek_model_thinking_stream(
    allow_model_requests: None, deepseek_api_key: str, freeze_deepseek_off_peak_pricing: None
):
    deepseek_model = OpenAIChatModel('deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model=deepseek_model)

    result: AgentRunResult | None = None
    async with agent.run_stream_events(user_prompt='How do I cross the street?') as event_stream:
        async for event in event_stream:
            if isinstance(event, AgentRunResultEvent):
                result = event.result

    assert result is not None
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='How do I cross the street?',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[
                    ThinkingPart(
                        content=IsStr(),
                        id='reasoning_content',
                        provider_name='deepseek',
                    ),
                    TextPart(content=IsStr()),
                ],
                usage=RequestUsage(
                    input_tokens=90,
                    output_tokens=303,
                    details={'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 90, 'reasoning_tokens': 89},
                    output_reasoning_tokens=89,
                    cost=Decimal('0.00009744'),
                ),
                model_name='deepseek-v4-flash',
                timestamp=IsDatetime(),
                provider_name='deepseek',
                provider_url='https://api.deepseek.com',
                provider_details={
                    'finish_reason': 'stop',
                    'timestamp': IsDatetime(),
                },
                provider_response_id='fa23191e-e1bf-4fdc-9ba2-c33d951d7e32',
                finish_reason='stop',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


@dataclass(frozen=True)
class ToolChoiceCase:
    id: str
    model_name: str
    thinking_disabled: bool
    expected_tool_choice: str
    expected_reasoning_effort: str | None


TOOL_CHOICE_CASES = [
    ToolChoiceCase(
        id='thinking_on_downgrades',
        model_name='deepseek-v4-flash',
        thinking_disabled=False,
        expected_tool_choice='auto',
        expected_reasoning_effort=None,
    ),
    ToolChoiceCase(
        id='thinking_off_forces',
        model_name='deepseek-v4-flash',
        thinking_disabled=True,
        expected_tool_choice='required',
        expected_reasoning_effort='none',
    ),
    ToolChoiceCase(
        id='thinking_off_forces_pro',
        model_name='deepseek-v4-pro',
        thinking_disabled=True,
        expected_tool_choice='required',
        expected_reasoning_effort='none',
    ),
]


class Answer(BaseModel):
    text: str


@pytest.mark.parametrize('case', TOOL_CHOICE_CASES, ids=lambda c: c.id)
async def test_deepseek_tool_choice_follows_thinking(
    case: ToolChoiceCase,
    allow_model_requests: None,
    deepseek_api_key: str,
    request_capture: RequestCapture,
):
    """DeepSeek rejects a forced tool choice only while thinking is on, so the downgrade is per request.

    The wire assertion is what matters here: with thinking on, `tool_choice` must stay `'auto'` or the
    API answers `Thinking mode does not support this tool_choice`; with thinking off, forcing must
    survive, which is what makes tool-based structured output reliable on `deepseek-v4-pro`.
    """
    model = OpenAIChatModel(
        case.model_name, provider=DeepSeekProvider(api_key=deepseek_api_key, http_client=request_capture.client)
    )
    settings = OpenAIChatModelSettings(thinking=False) if case.thinking_disabled else None
    agent = Agent(model, output_type=Answer, model_settings=settings)

    result = await agent.run('Say hi.')

    assert isinstance(result.output, Answer)
    body = request_capture.body('/chat/completions')
    assert body['tool_choice'] == case.expected_tool_choice
    assert body.get('reasoning_effort') == case.expected_reasoning_effort


async def test_deepseek_chat_native_output_refused(allow_model_requests: None, deepseek_api_key: str):
    """Chat Completions has no `json_schema` tool choice, so `NativeOutput` must fail before any request."""
    model = OpenAIChatModel('deepseek-v4-flash', provider=DeepSeekProvider(api_key=deepseek_api_key))
    agent = Agent(model, output_type=NativeOutput(Answer))

    with pytest.raises(UserError, match=re.escape('Native structured output is not supported by this model.')):
        await agent.run('Say hi.')
