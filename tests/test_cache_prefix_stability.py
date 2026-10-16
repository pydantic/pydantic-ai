"""Prompt-cache prefix stability across the UI-adapter round-trip seam.

A UI-driven app (Vercel AI / AG-UI) holds conversation history in the protocol's own
message shape and re-submits it every turn. The server turns it back into Pydantic AI
messages with `load_messages` and sends those to the provider. If that round-trip
(`ModelMessages -> dump_messages -> load_messages`) changes the provider request, the
cacheable prefix moves and the provider's prompt cache silently misses — no error, just a
bigger bill and higher latency (the PR #4338 class of bug).

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

import pytest
from cassetter import Cassette

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
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

with try_import() as openai_imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

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
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
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
    settings = AnthropicModelSettings(anthropic_thinking={'type': 'enabled', 'budget_tokens': 1024})
    model = AnthropicModel('claude-sonnet-4-5', provider=AnthropicProvider(api_key=anthropic_api_key))
    generator = Agent(model, model_settings=settings)
    history = (await generator.run('Briefly: what is 17 * 23? Think first.')).all_messages()

    probe = Agent(model, model_settings=settings)
    await probe.run('Reply with exactly: OK', message_history=history)
    await probe.run('Reply with exactly: OK', message_history=_ag_ui_roundtrip_0_1_10(history))

    original_body, roundtripped_body = _post_bodies(vcr)[-2:]
    assert roundtripped_body != original_body
