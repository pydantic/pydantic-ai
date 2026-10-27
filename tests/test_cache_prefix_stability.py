"""Prompt-cache prefix stability across the history round-trip seams: UI adapters and persisted JSON.

A UI-driven app (Vercel AI / AG-UI) holds conversation history in the protocol's own
message shape and re-submits it every turn. The server turns it back into Pydantic AI
messages with `load_messages` and sends those to the provider. If that round-trip
(`ModelMessages -> dump_messages -> load_messages`) changes the provider request, the
cacheable prefix moves and the provider's prompt cache silently misses — no error, just a
bigger bill and higher latency (the PR #4338 class of bug). An app that persists history with
`ModelMessagesTypeAdapter` and reloads it to continue the conversation crosses the same kind of seam.

Each test records a real conversation, round-trips the resulting history through an adapter,
then sends the same follow-up prompt twice — once with the original history, once with the
round-tripped one — and compares the two recorded request bodies. Note the comparison is over
the re-serialized JSON, not the raw wire bytes: cassetter stores each JSON body as a parsed
structure and `tests/cassette_utils.py` reads it back through `json.dumps`, so whitespace-only
differences are normalized away. What survives — and what these tests guard —
is the structural and field-value drift a lossy round-trip actually produces: dropped or reordered
blocks, a changed tool-arg representation, a missing thinking signature. Coverage targets the
PR #5873 wire-fidelity risks: tool-call arg serialization and thinking signatures.

On real provider data these round-trips are faithful, so the equality tests are regression guards.
The one real change is AG-UI < 0.1.11 dropping `ThinkingPart` (no reasoning carrier before 0.1.11);
that test asserts the request *changes*, documenting the protocol-version limitation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from uuid import UUID

import pytest
from cassetter import Cassette
from inline_snapshot import snapshot
from pydantic import AnyUrl, BaseModel, model_serializer

from pydantic_ai import Agent, ToolOutput
from pydantic_ai.messages import (
    BinaryContent,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryFeedbackPart,
    RetryInfo,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import StepStartUIPart, UIMessage

from .conftest import try_import

with try_import() as ag_ui_imports_successful:
    from pydantic_ai.ui.ag_ui import AGUIAdapter

with try_import() as ag_ui_reasoning_successful:
    # `ReasoningMessage`, the reasoning carrier, landed in ag-ui-protocol 0.1.11; older installs
    # (e.g. the `lowest-versions` CI job, pinned to 0.1.10) lack it, so that dump path is skipped.
    from ag_ui.core import ReasoningMessage  # noqa: F401  # pyright: ignore[reportUnusedImport]

with try_import() as ag_ui_multimodal_successful:
    # Typed multimodal input content, which carries `vendor_metadata`, landed in ag-ui-protocol 0.1.15.
    from ag_ui.core import ImageInputContent  # noqa: F401  # pyright: ignore[reportUnusedImport]

with try_import() as openai_imports_successful:
    from openai.types.chat import ChatCompletionMessage

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    from .models.mock_openai import (
        MockOpenAI,
        completion_message as openai_completion_message,
        get_mock_chat_completion_kwargs as get_mock_openai_kwargs,
    )

with try_import() as anthropic_imports_successful:
    from anthropic.types.beta import BetaTextBlock, BetaUsage

    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
    from pydantic_ai.providers.anthropic import AnthropicProvider

    from .models.test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

pytestmark = [pytest.mark.vcr]


def _post_bodies(vcr: Cassette) -> list[bytes | str | None]:
    """Bodies of the POST requests recorded in `vcr`, in order."""
    return [request.body for request in vcr.requests if request.method == 'POST']


def _vercel_roundtrip(history: list[ModelMessage]) -> list[ModelMessage]:
    return VercelAIAdapter.load_messages(VercelAIAdapter.dump_messages(history))


def _vercel_stream_roundtrip(history: list[ModelMessage]) -> list[ModelMessage]:
    """Round-trip through the shape an AI SDK client holds after streaming a run.

    The client accumulates every model response of a run into one assistant `UIMessage`,
    starting each response (a "step") with a `step-start` part, rather than the one message per
    response that `dump_messages` writes.
    """
    ui_messages: list[UIMessage] = []
    for ui_message in VercelAIAdapter.dump_messages(history):
        step = [StepStartUIPart(), *ui_message.parts]
        if ui_message.role == 'assistant' and ui_messages and ui_messages[-1].role == 'assistant':
            ui_messages[-1].parts = [*ui_messages[-1].parts, *step]
        elif ui_message.role == 'assistant':
            ui_messages.append(ui_message.model_copy(update={'parts': step}))
        else:
            ui_messages.append(ui_message)
    return VercelAIAdapter.load_messages(ui_messages)


def _ag_ui_roundtrip_latest(history: list[ModelMessage]) -> list[ModelMessage]:
    return AGUIAdapter.load_messages(AGUIAdapter.dump_messages(history, ag_ui_version='0.1.11'))


def _ag_ui_roundtrip_0_1_10(history: list[ModelMessage]) -> list[ModelMessage]:
    return AGUIAdapter.load_messages(AGUIAdapter.dump_messages(history, ag_ui_version='0.1.10'))


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_roundtrip, id='vercel'),
        pytest.param(
            _ag_ui_roundtrip_latest,
            id='ag_ui',
            marks=pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed'),
        ),
    ],
)
async def test_openai_tool_call_roundtrip_wire_stable(
    allow_model_requests: None,
    openai_api_key: str,
    vcr: Cassette,
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
):
    """A real OpenAI tool call re-sends with an identical request body after a UI round-trip.

    OpenAI carries `arguments` as a JSON string; a round-trip that reconstructed it differently
    would move the cacheable prefix under OpenAI's exact-prefix caching.
    """
    model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(api_key=openai_api_key))
    generator = Agent(model)

    @generator.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    history = (await generator.run('What is the weather in Paris? Use the tool.')).all_messages()

    probe = Agent(model)
    await probe.run('Reply with exactly: OK', message_history=history)
    await probe.run('Reply with exactly: OK', message_history=roundtrip(history))

    original_body, roundtripped_body = _post_bodies(vcr)[-2:]
    assert roundtripped_body == original_body


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_roundtrip, id='vercel'),
        pytest.param(_vercel_stream_roundtrip, id='vercel_stream'),
        pytest.param(
            _ag_ui_roundtrip_latest,
            id='ag_ui',
            marks=pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed'),
        ),
    ],
)
async def test_parallel_tool_calls_roundtrip_wire_stable(
    allow_model_requests: None,
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
):
    """Parallel tool calls re-send with an identical request after a UI round-trip.

    The calls must come back as one assistant message followed by all their results, as the model
    made them; one assistant message per call would move the cacheable prefix and misrepresent
    what the model did. The two requests are compared as built, not as recorded (a cassette
    replays its recording whatever body is sent), so this is a mocked test.
    """

    def parallel_calls(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1].parts[-1], ToolReturnPart):
            return ModelResponse(parts=[TextPart('Sunny in both.')])
        return ModelResponse(
            parts=[
                TextPart('Checking both cities.'),
                ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_paris'),
                ToolCallPart('get_weather', {'city': 'London'}, tool_call_id='call_london'),
            ]
        )

    generator = Agent(FunctionModel(parallel_calls))

    @generator.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny in {city}'

    history = (await generator.run('What is the weather in Paris and in London?')).all_messages()

    ok = completion_message([BetaTextBlock(text='OK', type='text')], BetaUsage(input_tokens=1, output_tokens=1))
    client = MockAnthropic.create_mock([ok, ok])
    probe = Agent(AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client)))
    await probe.run('Reply with exactly: OK', message_history=history)
    await probe.run('Reply with exactly: OK', message_history=roundtrip(history))

    original, roundtripped = (
        json.dumps(kwargs['messages'], sort_keys=True, default=str)
        for kwargs in get_mock_chat_completion_kwargs(client)
    )
    assert '"call_london"' in original and original.count('"tool_use"') == 2
    assert roundtripped == original


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_roundtrip, id='vercel'),
        pytest.param(
            _ag_ui_roundtrip_latest,
            id='ag_ui-0_1_11',
            marks=pytest.mark.skipif(
                not ag_ui_reasoning_successful(), reason='ag-ui-protocol < 0.1.11 (no ReasoningMessage)'
            ),
        ),
    ],
)
async def test_anthropic_thinking_roundtrip_wire_stable(
    allow_model_requests: None,
    anthropic_api_key: str,
    vcr: Cassette,
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
):
    """A real Anthropic thinking block (with its signature) re-sends with an identical request body after a round-trip."""
    settings = AnthropicModelSettings(
        anthropic_thinking={'type': 'adaptive', 'display': 'summarized'}, anthropic_effort='max'
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    generator = Agent(model, model_settings=settings)
    history = (await generator.run('Briefly: what is 17 * 23? Think first.')).all_messages()

    probe = Agent(model, model_settings=settings)
    await probe.run('Reply with exactly: OK', message_history=history)
    await probe.run('Reply with exactly: OK', message_history=roundtrip(history))

    original_body, roundtripped_body = _post_bodies(vcr)[-2:]
    assert roundtripped_body == original_body


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed')
@pytest.mark.moves_cache_prefix(
    reason='AG-UI < 0.1.11 has no reasoning carrier, so the round-trip deliberately drops the '
    'ThinkingPart and moves the prefix; this test asserts that documented limitation.'
)
async def test_anthropic_thinking_agui_0_1_10_drops_prefix(
    allow_model_requests: None,
    anthropic_api_key: str,
    vcr: Cassette,
):
    """AG-UI < 0.1.11 has no reasoning carrier, so dump drops the `ThinkingPart` and the re-sent
    prefix changes. Documents the protocol-version limitation (verified, not a defect to fix)."""
    settings = AnthropicModelSettings(
        anthropic_thinking={'type': 'adaptive', 'display': 'summarized'}, anthropic_effort='max'
    )
    model = AnthropicModel('claude-sonnet-5-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    generator = Agent(model, model_settings=settings)
    history = (await generator.run('Briefly: what is 17 * 23? Think first.')).all_messages()

    probe = Agent(model, model_settings=settings)
    await probe.run('Reply with exactly: OK', message_history=history)
    await probe.run('Reply with exactly: OK', message_history=_ag_ui_roundtrip_0_1_10(history))

    original_body, roundtripped_body = _post_bodies(vcr)[-2:]
    assert roundtripped_body != original_body


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_tool_return_order_wire_stable(allow_model_requests: None):
    """The same parallel tool exchange re-sends with an identical request body whatever order its results are stored in.

    `end_strategy='early'` stores the output tool's result ahead of the function tools' results, while a UI client
    rebuilds them from the stream (AG-UI, in the order the tools completed) or from per-call tool parts (Vercel AI,
    in call order). The prompt cache is keyed on the bytes sent, so results go out in call order either way.
    """

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart('get_weather', {'city': 'Paris'}, tool_call_id='call_weather'),
                ToolCallPart('final_result', {'response': 'sunny'}, tool_call_id='call_output'),
                ToolCallPart('get_time', {'city': 'Paris'}, tool_call_id='call_time'),
            ]
        )

    generator = Agent(FunctionModel(respond), output_type=ToolOutput(str), end_strategy='early')

    @generator.tool_plain
    def get_weather(city: str) -> str:
        return 'sunny'  # pragma: no cover

    @generator.tool_plain
    def get_time(city: str) -> str:
        return 'noon'  # pragma: no cover

    history = (await generator.run('Weather and time in Paris?')).all_messages()
    results = history[-1]
    assert isinstance(results, ModelRequest)
    by_id = {part.tool_call_id: part for part in results.parts if isinstance(part, ToolReturnPart)}
    assert list(by_id) == ['call_output', 'call_weather', 'call_time']
    in_call_order = [
        *history[:-1],
        replace(results, parts=[by_id[i] for i in ('call_weather', 'call_output', 'call_time')]),
    ]

    bodies: list[str] = []
    for messages in (history, in_call_order):
        client = MockOpenAI.create_mock(
            openai_completion_message(ChatCompletionMessage(content='OK', role='assistant'))
        )
        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
        await Agent(model).run('Reply with exactly: OK', message_history=messages)
        sent = get_mock_openai_kwargs(client)[0]['messages']
        assert [m['tool_call_id'] for m in sent if m['role'] == 'tool'] == ['call_weather', 'call_output', 'call_time']
        bodies.append(json.dumps(sent))

    original_body, reordered_body = bodies
    assert reordered_body == original_body


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_roundtrip, id='vercel'),
        pytest.param(
            _ag_ui_roundtrip_latest,
            id='ag_ui',
            marks=pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed'),
        ),
    ],
)
async def test_openai_spaced_tool_call_args_roundtrip_wire_stable(
    allow_model_requests: None,
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
):
    """Raw tool-call args with whitespace re-send with an identical request body after a UI round-trip.

    Many models (DeepSeek, Mistral, Cerebras, older GPT-4o snapshots) return arguments like `{"city": "Paris"}`,
    which the history keeps as that string. Vercel AI carries the call's input as an object, so the round-tripped
    copy holds a dict that serializes compactly. The prompt cache is keyed on the bytes sent, so both forms must
    render the same `arguments` string. This is mocked rather than recorded because VCR here doesn't match on the
    request body, and the real `gpt-4o` already returns compact arguments.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is the weather in Paris?')]),
        ModelResponse(parts=[ToolCallPart('get_weather', '{"city": "Paris", "days": 2}', tool_call_id='call_1')]),
        ModelRequest(parts=[ToolReturnPart('get_weather', 'sunny', tool_call_id='call_1')]),
        ModelResponse(parts=[TextPart('It is sunny in Paris.')]),
    ]

    bodies: list[str] = []
    for messages in (history, roundtrip(history)):
        client = MockOpenAI.create_mock(
            openai_completion_message(ChatCompletionMessage(content='OK', role='assistant'))
        )
        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
        await Agent(model).run('Reply with exactly: OK', message_history=messages)
        bodies.append(json.dumps(get_mock_openai_kwargs(client)[0]['messages']))

    original_body, roundtripped_body = bodies
    assert roundtripped_body == original_body
    assert '"arguments": "{\\"city\\":\\"Paris\\",\\"days\\":2}"' in original_body


@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_vercel_roundtrip, id='vercel'),
        pytest.param(_vercel_stream_roundtrip, id='vercel_stream'),
        pytest.param(
            _ag_ui_roundtrip_latest,
            id='ag_ui-0_1_11',
            marks=pytest.mark.skipif(
                not ag_ui_reasoning_successful(), reason='ag-ui-protocol < 0.1.11 (no ReasoningMessage)'
            ),
        ),
    ],
)
async def test_anthropic_foreign_thinking_roundtrip_wire_stable(
    allow_model_requests: None,
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
):
    """Another model's thinking re-sends to Anthropic with an identical request body after a UI round-trip.

    Anthropic can't replay a foreign `ThinkingPart` natively, so it sends it as text in an
    `<assistant_thinking>` tag. UI adapters deliberately don't round-trip `ModelResponse.model_name` or
    `provider_name`, so the tag must not be rendered from them, or the re-sent prefix misses the cache.
    """
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('What is 17 * 23?')]),
        ModelResponse(
            parts=[
                ThinkingPart('17 * 23 is 391.', signature='openai-signature', provider_name='openai'),
                TextPart('391'),
            ],
            model_name='gpt-5.2',
            provider_name='openai',
        ),
    ]

    bodies: list[str] = []
    for messages in (history, roundtrip(history)):
        client = MockAnthropic.create_mock(
            completion_message([BetaTextBlock(text='OK', type='text')], BetaUsage(input_tokens=1, output_tokens=1))
        )
        model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=client))
        await Agent(model).run('Reply with exactly: OK', message_history=messages)
        bodies.append(json.dumps(get_mock_chat_completion_kwargs(client)[0]['messages']))

    original_body, roundtripped_body = bodies
    assert '<assistant_thinking>\\n17 * 23 is 391.\\n</assistant_thinking>' in original_body
    assert roundtripped_body == original_body


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed')
@pytest.mark.parametrize(
    'prompts,ag_ui_version',
    [
        pytest.param([UserPromptPart('A1'), UserPromptPart('B1')], '0.1.11', id='text'),
        pytest.param([UserPromptPart('A1'), UserPromptPart('B1')], '0.1.10', id='text-0_1_10'),
        pytest.param(
            [
                UserPromptPart(['A1', BinaryContent(b'\x89PNG', media_type='image/png')]),
                UserPromptPart([ImageUrl('https://example.com/b.png', vendor_metadata={'detail': 'high'}), 'B1']),
                UserPromptPart('C1'),
            ],
            '0.1.15',
            id='media',
            marks=pytest.mark.skipif(
                not ag_ui_multimodal_successful(), reason='ag-ui-protocol < 0.1.15 (no typed multimodal content)'
            ),
        ),
        pytest.param(
            [
                UserPromptPart('A1'),
                RetryFeedbackPart('bad', retry=RetryInfo(cause='validation_error')),
                UserPromptPart('B1'),
            ],
            '0.1.11',
            id='retry',
        ),
    ],
)
async def test_openai_multiple_user_prompts_ag_ui_roundtrip_wire_stable(
    allow_model_requests: None, prompts: list[ModelRequestPart], ag_ui_version: str
):
    """A request with several `UserPromptPart`s re-sends with an identical request body after an AG-UI round-trip.

    Each prompt has to come back as its own `UserPromptPart`, rather than merged into one prompt that OpenAI
    renders as one user message instead of one per prompt. Validation feedback is sent as a user message of its own
    too, and reloads as the `RetryFeedbackPart` it was dumped from, which has to stay separate in the same way.
    """
    history: list[ModelMessage] = [ModelRequest(parts=prompts), ModelResponse(parts=[TextPart('OK')])]

    roundtripped = AGUIAdapter.load_messages(AGUIAdapter.dump_messages(history, ag_ui_version=ag_ui_version))
    assert len(roundtripped[0].parts) == len(prompts)

    bodies: list[str] = []
    for messages in (history, roundtripped):
        client = MockOpenAI.create_mock(
            openai_completion_message(ChatCompletionMessage(content='OK', role='assistant'))
        )
        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
        await Agent(model).run('Reply with exactly: OK', message_history=messages)
        bodies.append(json.dumps(get_mock_openai_kwargs(client)[0]['messages']))

    original_body, roundtripped_body = bodies
    assert roundtripped_body == original_body
    assert [m['role'] for m in json.loads(original_body)].count('user') == len(prompts) + 1


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    'prompts',
    [
        pytest.param([UserPromptPart('A1'), UserPromptPart('B1')], id='text'),
        pytest.param(
            [
                UserPromptPart(['A1', BinaryContent(b'\x89PNG', media_type='image/png')]),
                UserPromptPart([ImageUrl('https://example.com/b.png', vendor_metadata={'detail': 'high'}), 'B1']),
                UserPromptPart('C1'),
            ],
            id='media',
        ),
    ],
)
async def test_openai_multiple_user_prompts_roundtrip_wire_stable(
    allow_model_requests: None, prompts: list[UserPromptPart]
):
    """A request with several `UserPromptPart`s re-sends with an identical request body after a Vercel AI round-trip.

    All of a request's prompts go into one user `UIMessage`, so `load_messages` has to tell where each one begins
    to rebuild them, rather than merging them into one prompt that OpenAI renders as one user message instead of
    one per prompt.
    """
    history: list[ModelMessage] = [ModelRequest(parts=prompts), ModelResponse(parts=[TextPart('OK')])]

    roundtripped = _vercel_roundtrip(history)
    assert [type(part) for part in roundtripped[0].parts] == [UserPromptPart] * len(prompts)

    bodies: list[str] = []
    for messages in (history, roundtripped):
        client = MockOpenAI.create_mock(
            openai_completion_message(ChatCompletionMessage(content='OK', role='assistant'))
        )
        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
        await Agent(model).run('Reply with exactly: OK', message_history=messages)
        bodies.append(json.dumps(get_mock_openai_kwargs(client)[0]['messages']))

    original_body, roundtripped_body = bodies
    assert roundtripped_body == original_body
    assert [m['role'] for m in json.loads(original_body)].count('user') == len(prompts) + 1


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
async def test_openai_retry_feedback_after_user_prompt_vercel_roundtrip_wire_stable(allow_model_requests: None):
    """System-voice feedback authored after a user prompt keeps its place through a Vercel AI round-trip.

    Reloaded ahead of the prompt instead, it would open the first request, where it goes out as
    `<system>`-tagged user text rather than a system message, and the request would re-send different bytes.
    """
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart('how many?'),
                RetryFeedbackPart('the answer has to be a number', retry=RetryInfo(cause='model_retry')),
            ]
        ),
        ModelResponse(parts=[TextPart('OK')]),
    ]

    roundtripped = _vercel_roundtrip(history)
    assert [type(part) for part in roundtripped[0].parts] == [UserPromptPart, RetryFeedbackPart]

    bodies: list[str] = []
    for messages in (history, roundtripped):
        client = MockOpenAI.create_mock(
            openai_completion_message(ChatCompletionMessage(content='OK', role='assistant'))
        )
        model = OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=client))
        await Agent(model).run('Reply with exactly: OK', message_history=messages)
        bodies.append(json.dumps(get_mock_openai_kwargs(client)[0]['messages']))

    original_body, roundtripped_body = bodies
    assert roundtripped_body == original_body


def _json_roundtrip(history: list[ModelMessage]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(history))


@pytest.mark.skipif(not openai_imports_successful(), reason='openai not installed')
@pytest.mark.skipif(not anthropic_imports_successful(), reason='anthropic not installed')
@pytest.mark.parametrize(
    'roundtrip',
    [
        pytest.param(_json_roundtrip, id='json'),
        pytest.param(_vercel_roundtrip, id='vercel'),
        pytest.param(
            _ag_ui_roundtrip_latest,
            id='ag_ui',
            marks=pytest.mark.skipif(not ag_ui_imports_successful(), reason='ag-ui-protocol not installed'),
        ),
    ],
)
async def test_string_encoded_tool_returns_roundtrip_wire_stable(
    allow_model_requests: None,
    roundtrip: Callable[[list[ModelMessage]], list[ModelMessage]],
):
    """Tool returns whose JSON form is a string re-send with an identical request body after a round-trip.

    A persisted history reloads a `datetime`, `UUID`, `bytes`, str-returning serializer and the like as the plain
    `str` of its JSON form, so the model must see that unquoted string from the first run, or the restored request
    moves the cacheable prefix from that tool result onward. OpenAI renders through `model_response_str` and
    Anthropic through `content_items`, one text block per list item. This is mocked rather than recorded because a
    cassette replays its recording whatever body is sent.
    """

    class Citation(BaseModel):
        title: str

        @model_serializer
        def as_text(self) -> str:
            return f'<cite>{self.title}</cite>'

    returns: dict[str, object] = {
        'datetime': datetime(2026, 1, 1, 9, 30, tzinfo=UTC),
        'date': date(2026, 1, 1),
        'uuid': UUID('12345678-1234-5678-1234-567812345678'),
        'decimal': Decimal('1.50'),
        'path': Path('report.txt'),
        'bytes': b'abc',
        'url': AnyUrl('https://example.com/docs'),
        'serializer': Citation(title='Q3 report'),
        'list': [date(2026, 1, 1), UUID('12345678-1234-5678-1234-567812345678')],
    }

    def call_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if isinstance(messages[-1].parts[-1], ToolReturnPart):
            return ModelResponse(parts=[TextPart('Done.')])
        return ModelResponse(
            parts=[ToolCallPart('get_value', {'kind': kind}, tool_call_id=f'call_{kind}') for kind in returns]
        )

    generator = Agent(FunctionModel(call_tools))

    @generator.tool_plain
    def get_value(kind: str) -> object:
        return returns[kind]

    history = (await generator.run('Look up every value.')).all_messages()
    restored = roundtrip(history)

    openai_ok = openai_completion_message(ChatCompletionMessage(content='OK', role='assistant'))
    openai_client = MockOpenAI.create_mock([openai_ok, openai_ok])
    openai_probe = Agent(OpenAIChatModel('gpt-4o', provider=OpenAIProvider(openai_client=openai_client)))
    anthropic_ok = completion_message(
        [BetaTextBlock(text='OK', type='text')], BetaUsage(input_tokens=1, output_tokens=1)
    )
    anthropic_client = MockAnthropic.create_mock([anthropic_ok, anthropic_ok])
    anthropic_probe = Agent(
        AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(anthropic_client=anthropic_client))
    )
    for messages in (history, restored):
        await openai_probe.run('Reply with exactly: OK', message_history=messages)
        await anthropic_probe.run('Reply with exactly: OK', message_history=messages)

    openai_original, openai_restored = (kwargs['messages'] for kwargs in get_mock_openai_kwargs(openai_client))
    anthropic_original, anthropic_restored = (
        kwargs['messages'] for kwargs in get_mock_chat_completion_kwargs(anthropic_client)
    )
    assert openai_restored == openai_original
    assert anthropic_restored == anthropic_original
    assert [message['content'] for message in openai_original if message['role'] == 'tool'] == snapshot(
        [
            '2026-01-01T09:30:00Z',
            '2026-01-01',
            '12345678-1234-5678-1234-567812345678',
            '1.50',
            'report.txt',
            'YWJj',
            'https://example.com/docs',
            '<cite>Q3 report</cite>',
            '["2026-01-01","12345678-1234-5678-1234-567812345678"]',
        ]
    )
    assert [
        [block['text'] for block in block_param['content']]
        for message in anthropic_original
        for block_param in message['content']
        if block_param['type'] == 'tool_result'
    ][-1] == snapshot(['2026-01-01', '12345678-1234-5678-1234-567812345678'])
