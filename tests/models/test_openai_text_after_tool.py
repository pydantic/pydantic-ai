from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.ui.vercel_ai.request_types import SubmitMessage
from pydantic_ai.ui.vercel_ai.response_types import (
    ReasoningDeltaChunk,
    TextDeltaChunk,
    TextEndChunk,
    TextStartChunk,
)

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


@pytest.mark.parametrize('suffix', ['\n', 'More text.'])
@pytest.mark.parametrize('thinking', [False, True])
@pytest.mark.parametrize('ignore_whitespace', [False, True])
async def test_text_after_tool_has_its_own_lifecycle(
    allow_model_requests: None, suffix: str, thinking: bool, ignore_whitespace: bool
):
    # Mock the intermittent text/tool/text ordering observed with an OpenAI-compatible provider.
    prefix = ['<think>', 'Checking', '</think>'] if thinking else []
    stream = [chunk(ChoiceDelta(content=text)) for text in [*prefix, 'Checking ', 'now.']]
    stream += [
        chunk(
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        id='call-1',
                        type='function',
                        function=ChoiceDeltaToolCallFunction(name='lookup', arguments='{'),
                    )
                ]
            )
        ),
        chunk(
            ChoiceDelta(
                tool_calls=[
                    ChoiceDeltaToolCall(
                        index=0,
                        function=ChoiceDeltaToolCallFunction(arguments='}'),
                    )
                ]
            )
        ),
    ]
    if thinking:
        stream.extend(chunk(ChoiceDelta(content=text)) for text in ['<think>', 'Considering', '</think>'])
    stream.extend(chunk(ChoiceDelta(content=text)) for text in [suffix, ' Done.'])
    client = MockOpenAI.create_mock_stream([stream, [chunk(ChoiceDelta(content='Finished.'))]])
    model = OpenAIChatModel(
        'test',
        provider=OpenAIProvider(openai_client=client),
        profile=OpenAIModelProfile(
            thinking_tags=('<think>', '</think>'),
            ignore_streamed_leading_whitespace=ignore_whitespace,
        ),
    )
    agent = Agent(model)
    calls: list[str] = []

    @agent.tool_plain
    def lookup() -> str:
        calls.append('lookup')
        return 'Found.'

    request = SubmitMessage.model_validate(
        {
            'id': 'chat',
            'messages': [{'id': 'user', 'role': 'user', 'parts': [{'type': 'text', 'text': 'Look up.'}]}],
        }
    )
    adapter = VercelAIAdapter(agent=agent, run_input=request, sdk_version=6)
    reasoning = ''
    active: set[str] = set()
    text_parts: dict[str, str] = {}
    async for event in adapter.transform_stream(adapter.run_stream_native()):
        if isinstance(event, ReasoningDeltaChunk):
            reasoning += event.delta
        elif isinstance(event, TextStartChunk):
            assert event.id not in text_parts
            active.add(event.id)
            text_parts[event.id] = ''
        elif isinstance(event, TextDeltaChunk):
            assert event.id in active
            text_parts[event.id] += event.delta
        elif isinstance(event, TextEndChunk):
            assert event.id in active
            active.remove(event.id)

    expected_suffix = '' if ignore_whitespace and suffix.isspace() else suffix
    assert list(text_parts.values()) == ['Checking now.', expected_suffix + ' Done.', 'Finished.']
    assert not active
    assert calls == ['lookup']
    assert reasoning == ('CheckingConsidering' if thinking else '')
