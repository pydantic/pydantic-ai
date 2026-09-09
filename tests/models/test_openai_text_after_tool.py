from __future__ import annotations

import pytest

from pydantic_ai.messages import (
    ModelRequest,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from ..conftest import try_import

with try_import() as imports_successful:
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.profiles.openai import OpenAIModelProfile
    from pydantic_ai.providers.openai import OpenAIProvider

    from .mock_openai import MockOpenAI

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(not imports_successful(), reason='openai not installed')]


def chunk(delta: ChoiceDelta) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        id='response', object='chat.completion.chunk', created=1, model='test', choices=[Choice(index=0, delta=delta)]
    )


def tool_delta(index: int, arguments: str, *, start: bool = False) -> ChoiceDelta:
    return ChoiceDelta(
        tool_calls=[
            ChoiceDeltaToolCall(
                index=index,
                id=f'call-{index}' if start else None,
                type='function' if start else None,
                function=ChoiceDeltaToolCallFunction(name='lookup' if start else None, arguments=arguments),
            )
        ]
    )


@pytest.mark.parametrize('suffix', ['\n', 'More text.'])
@pytest.mark.parametrize('thinking', [False, True], ids=['text', 'closing-thinking-after-tool'])
@pytest.mark.parametrize('ignore_whitespace', [False, True])
async def test_text_after_tool_has_its_own_part(
    allow_model_requests: None, suffix: str, thinking: bool, ignore_whitespace: bool
):
    # Synthetic chunks make the captured text/tool/text ordering and late thinking delimiter deterministic.
    prefix = ['<think>', 'Checking'] if thinking else ['Checking', ' now.']
    deltas = [ChoiceDelta(content=text) for text in prefix]
    deltas.extend([tool_delta(0, '{', start=True), tool_delta(0, '"value":'), tool_delta(0, '1}')])
    if thinking:
        deltas.append(ChoiceDelta(content='</think>'))
    deltas.extend([ChoiceDelta(content=suffix), ChoiceDelta(content=' Continued.')])
    deltas.extend([tool_delta(1, '{', start=True), tool_delta(1, '"value":'), tool_delta(1, '2}')])
    deltas.extend([ChoiceDelta(content='Last'), ChoiceDelta(content=' text.')])
    model = OpenAIChatModel(
        'test',
        provider=OpenAIProvider(openai_client=MockOpenAI.create_mock_stream([chunk(delta) for delta in deltas])),
        profile=OpenAIModelProfile(
            thinking_tags=('<think>', '</think>'),
            ignore_streamed_leading_whitespace=ignore_whitespace,
        ),
    )
    parameters = ModelRequestParameters(
        function_tools=[
            ToolDefinition(
                name='lookup',
                parameters_json_schema={
                    'type': 'object',
                    'properties': {'value': {'type': 'integer'}},
                    'required': ['value'],
                },
            )
        ]
    )
    async with model.request_stream(
        [ModelRequest(parts=[UserPromptPart('Look up two values.')])], None, parameters
    ) as response:
        events = [event async for event in response]
        parts = response.get().parts

    assert [(event.index, type(event.part)) for event in events if isinstance(event, PartStartEvent)] == [
        (0, ThinkingPart if thinking else TextPart),
        (1, ToolCallPart),
        (2, TextPart),
        (3, ToolCallPart),
        (4, TextPart),
    ]
    ended: set[int] = set()
    for event in events:
        if isinstance(event, PartEndEvent):
            ended.add(event.index)
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            assert event.index not in ended

    expected_suffix = '' if ignore_whitespace and suffix == '\n' else suffix
    assert parts == [
        ThinkingPart(content='Checking', id='content', provider_name='openai')
        if thinking
        else TextPart('Checking now.'),
        ToolCallPart(tool_name='lookup', args='{"value":1}', tool_call_id='call-0'),
        TextPart(expected_suffix + ' Continued.'),
        ToolCallPart(tool_name='lookup', args='{"value":2}', tool_call_id='call-1'),
        TextPart('Last text.'),
    ]
    assert any(
        isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta) and event.index == 4
        for event in events
    )
