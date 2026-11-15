"""The framework records changes to the thinking effort in effect as `ModelSettingsDeltaPart`s.

A model with a mid-conversation effort channel renders them in place and keeps the request-level effort the
conversation opened with, so the prompt cache survives the change; every other model drops them and sends the
current effort as the request-level setting. These tests pin the recording and the projection; the adapters'
rendering is pinned in `tests/models/test_model_settings_delta.py`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    CompactionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSettingsDeltaEvent,
    ModelSettingsDeltaPart,
    RunContext,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.capabilities import Hooks
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.settings import ModelSettings, ThinkingLevel

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider


def _effort_parts(messages: list[ModelMessage]) -> list[list[tuple[ThinkingLevel | None, bool]]]:
    """Each request's `ModelSettingsDeltaPart`s, as `(effort, top_level)`."""
    return [
        [
            (part.settings.get('thinking'), part.top_level)
            for part in message.parts
            if isinstance(part, ModelSettingsDeltaPart)
        ]
        for message in messages
        if isinstance(message, ModelRequest)
    ]


async def test_a_conversation_records_each_change_once():
    """The first named level opens a window, a change between levels is a delta, and a bool opens a new window."""
    agent = Agent(TestModel())
    history: list[ModelMessage] = []
    for thinking in ('low', 'low', 'high', True, False, 'medium'):
        result = await agent.run('Hi', message_history=history, model_settings=ModelSettings(thinking=thinking))
        history = result.all_messages()
    assert _effort_parts(history) == snapshot(
        [[('low', True)], [], [('high', False)], [(None, True)], [], [('medium', True)]]
    )
    # The part goes at the end of the request, after the input it applies to.
    first = history[0]
    assert isinstance(first, ModelRequest)
    assert isinstance(first.parts[-1], ModelSettingsDeltaPart)


async def test_nothing_is_recorded_without_an_effort_level():
    agent = Agent(TestModel())
    result = await agent.run('Hi')
    result = await agent.run(
        'Again', message_history=result.all_messages(), model_settings=ModelSettings(thinking=True)
    )
    assert _effort_parts(result.all_messages()) == [[], []]


async def test_a_per_step_change_is_recorded_on_the_step_request():
    """A `model_settings` callable that raises the effort after a tool call records the change on the tool-result request."""

    def effort(ctx: RunContext[Any]) -> ModelSettings:
        return ModelSettings(thinking='high' if ctx.run_step > 1 else 'low')

    agent = Agent(TestModel(call_tools=['look_up']), model_settings=effort)

    @agent.tool_plain
    def look_up() -> str:
        return 'quiet'

    result = await agent.run('Look it up')
    assert _effort_parts(result.all_messages()) == [[('low', True)], [('high', False)]]


async def test_the_change_is_streamed_as_an_event():
    agent = Agent(TestModel())
    async with agent.run_stream_events('Hi', model_settings=ModelSettings(thinking='low')) as stream:
        events = [event async for event in stream]
    assert [event.part for event in events if isinstance(event, ModelSettingsDeltaEvent)] == [
        ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)
    ]


@pytest.mark.parametrize(
    ('history_parts', 'thinking', 'expected'),
    [
        pytest.param(
            [ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)],
            'high',
            [('low', True), ('high', False)],
            id='changed',
        ),
        pytest.param(
            [ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)],
            'low',
            [('low', True)],
            id='unchanged',
        ),
        pytest.param(
            [ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)],
            True,
            [('low', True), (None, True)],
            id='withdrawn',
        ),
    ],
)
async def test_an_unanswered_change_is_followed_by_the_one_sent(
    history_parts: list[ModelSettingsDeltaPart], thinking: ThinkingLevel, expected: list[tuple[str | None, bool]]
):
    """A request that no response answered yet keeps its part, and records what it ends up sending after it.

    Continuing a history that ends in such a request sends that request again. The part already recorded stays,
    as a UI client that rebuilt the history from the event stream keeps it too, and the change is streamed like any
    other. A model with a channel collapses the two into the last, as no response separates them.
    """
    history: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart('Hi'), *history_parts])]
    async with Agent(TestModel()).run_stream_events(
        message_history=history, model_settings=ModelSettings(thinking=thinking)
    ) as stream:
        events = [event async for event in stream]
    result = next(event.result for event in events if isinstance(event, AgentRunResultEvent))
    assert _effort_parts(result.all_messages()) == [expected]
    streamed = [
        (event.part.settings.get('thinking'), event.part.top_level)
        for event in events
        if isinstance(event, ModelSettingsDeltaEvent)
    ]
    assert streamed == expected[len(history_parts) :]


async def test_an_unanswered_change_back_to_the_effort_in_effect_is_streamed():
    """Going back to the window's effort before any response answered the change records the change back.

    Removing the part instead would leave it on a UI client that rebuilt the history from the event stream, and its
    next request would send the withdrawn effort.
    """
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart('Hi'), ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)]
        ),
        ModelResponse(parts=[TextPart('Hello')]),
        ModelRequest(parts=[UserPromptPart('Again'), ModelSettingsDeltaPart(settings={'thinking': 'high'})]),
    ]
    async with Agent(TestModel()).run_stream_events(
        message_history=history, model_settings=ModelSettings(thinking='low')
    ) as stream:
        events = [event async for event in stream]
    result = next(event.result for event in events if isinstance(event, AgentRunResultEvent))
    assert _effort_parts(result.all_messages()) == [[('low', True)], [('high', False), ('low', False)]]
    assert [event.part for event in events if isinstance(event, ModelSettingsDeltaEvent)] == [
        ModelSettingsDeltaPart(settings={'thinking': 'low'})
    ]


async def test_a_change_is_recorded_on_history_when_a_hook_rebuilt_the_tail():
    """A `before_model_request` hook that rebuilds the outgoing request's parts leaves history's request unchanged.

    The change goes on both, so `all_messages()` records the effort the request was sent with.
    """

    def rebuild_tail(ctx: RunContext[None], request_context: ModelRequestContext) -> ModelRequestContext:
        *earlier, tail = request_context.messages
        assert isinstance(tail, ModelRequest)
        request_context.messages = [*earlier, replace(tail, parts=[replace(part) for part in tail.parts])]
        return request_context

    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart('Hi'), ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)])
    ]
    agent = Agent(TestModel(), capabilities=[Hooks(before_model_request=rebuild_tail)])
    result = await agent.run(message_history=history, model_settings=ModelSettings(thinking='high'))
    assert _effort_parts(result.all_messages()) == [[('low', True), ('high', False)]]


async def test_a_compaction_opens_a_new_window():
    """The changes before a compaction are gone with it, so the next request that names a level opens a window."""
    history: list[ModelMessage] = [
        ModelRequest(
            parts=[UserPromptPart('Hi'), ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)]
        ),
        ModelResponse(parts=[CompactionPart(content='Summary.')]),
    ]
    result = await Agent(TestModel()).run(
        'Again', message_history=history, model_settings=ModelSettings(thinking='low')
    )
    assert _effort_parts(result.all_messages()) == [[('low', True)], [('low', True)]]


def test_otel_message_parts():
    from pydantic_ai.models.instrumented import InstrumentationSettings

    settings = InstrumentationSettings(include_content=False)
    assert ModelSettingsDeltaPart(settings={'thinking': 'high'}).otel_message_parts(settings) == [
        {'type': 'text', 'content': "Model settings changed: thinking='high'"}
    ]
    assert ModelSettingsDeltaPart(settings={'thinking': None}, top_level=True).otel_message_parts(settings) == [
        {'type': 'text', 'content': 'Model settings changed: thinking=None'}
    ]


async def test_a_model_without_a_channel_never_sees_the_parts():
    """`prepare_messages` drops them, so the adapter sends the current effort at request level as before."""
    seen: list[list[ModelMessage]] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        return ModelResponse(parts=[TextPart('ok')])

    agent = Agent(FunctionModel(respond))
    result = await agent.run('Hi', model_settings=ModelSettings(thinking='low'))
    await agent.run('Again', message_history=result.all_messages(), model_settings=ModelSettings(thinking='high'))
    assert all(_effort_parts(messages) == [[]] * len(_effort_parts(messages)) for messages in seen)


def _history(*requests: list[Any]) -> list[ModelMessage]:
    """Requests with the given parts, each answered by a text response except the last."""
    messages: list[ModelMessage] = []
    for index, parts in enumerate(requests):
        messages.append(ModelRequest(parts=[UserPromptPart(f'Turn {index}'), *parts]))
        if index < len(requests) - 1:
            messages.append(ModelResponse(parts=[TextPart('ok')]))
    return messages


@pytest.mark.skipif(not imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    ('history', 'expected'),
    [
        pytest.param(
            _history(
                [ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)],
                [ModelSettingsDeltaPart(settings={'thinking': 'high'})],
                [ModelSettingsDeltaPart(settings={'thinking': None}, top_level=True)],
                [ModelSettingsDeltaPart(settings={'thinking': 'medium'}, top_level=True)],
                [ModelSettingsDeltaPart(settings={'thinking': 'low'})],
            ),
            [[], [], [], [('medium', True)], [('low', False)]],
            id='a-new-window-supersedes-the-earlier-ones',
        ),
        pytest.param(
            [
                ModelRequest(
                    parts=[UserPromptPart('Hi'), ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)]
                ),
                ModelRequest(parts=[UserPromptPart('Again'), ModelSettingsDeltaPart(settings={'thinking': 'high'})]),
                ModelResponse(parts=[TextPart('ok')]),
                ModelRequest(parts=[UserPromptPart('More'), ModelSettingsDeltaPart(settings={'thinking': 'low'})]),
                ModelRequest(parts=[UserPromptPart('Still'), ModelSettingsDeltaPart(settings={'thinking': 'medium'})]),
            ],
            [[], [('high', True)], [], [('medium', False)]],
            id='changes-no-response-separates-collapse-into-the-last',
        ),
        pytest.param(
            _history(
                [ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)],
                [
                    ModelSettingsDeltaPart(settings={'thinking': 'high'}),
                    ModelSettingsDeltaPart(settings={'thinking': 'low'}),
                ],
            ),
            [[('low', True)], [('low', False)]],
            id='a-change-and-the-change-back-collapse-into-one',
        ),
    ],
)
def test_a_model_with_a_channel_keeps_the_current_window(
    history: list[ModelMessage], expected: list[list[tuple[str | None, bool]]]
):
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(api_key='test-key'))
    assert _effort_parts(model.prepare_messages(history)) == expected


@pytest.mark.skipif(not imports_successful(), reason='openai not installed')
@pytest.mark.parametrize(
    'model',
    [
        pytest.param(
            lambda: OpenAIResponsesModel('gpt-5.6-sol', provider=OpenAIProvider(api_key='x')), id='no-profile'
        ),
        pytest.param(lambda: OpenAIChatModel('gpt-6.1-sol', provider=OpenAIProvider(api_key='x')), id='no-renderer'),
    ],
)
def test_a_model_without_a_channel_drops_every_part(model: Any):
    history = _history(
        [ModelSettingsDeltaPart(settings={'thinking': 'low'}, top_level=True)],
        [ModelSettingsDeltaPart(settings={'thinking': 'high'})],
    )
    built: Model = model()
    assert _effort_parts(built.prepare_messages(history)) == [[], []]


@pytest.mark.skipif(not imports_successful(), reason='openai not installed')
def test_a_history_without_parts_is_left_alone():
    history = _history([], [])
    model = OpenAIResponsesModel('gpt-6.1-sol', provider=OpenAIProvider(api_key='test-key'))
    assert model.prepare_messages(history) == history
