from __future__ import annotations

import pytest
from pydantic_ai import Agent, ModelMessage, ModelResponse, TextPart, ToolCallPart, ThinkingPart
from pydantic_ai.exceptions import ContentFilterError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# Import the mock client helper from sibling models folder
from .models.mock_openai import MockOpenAI

async def test_content_filter_on_tool_call():
    tool_executed = False

    async def mock_model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name='my_tool',
                    args={'x': 42},
                    tool_call_id='call_1',
                )
            ],
            model_name='test-model',
            finish_reason='content_filter',
        )

    model = FunctionModel(function=mock_model_func, model_name='test-model')
    agent = Agent(model)

    @agent.tool_plain
    def my_tool(x: int) -> str:
        nonlocal tool_executed
        tool_executed = True
        return f"Result: {x}"

    with pytest.raises(ContentFilterError, match="Content filter triggered."):
        await agent.run("Run tool")

    assert not tool_executed, "Tool should not have been executed when content filter is triggered!"


async def test_content_filter_on_thinking_only():
    async def mock_model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[ThinkingPart(content="I'm thinking about unsafe things...")],
            model_name='test-model',
            finish_reason='content_filter',
        )

    model = FunctionModel(function=mock_model_func, model_name='test-model')
    agent = Agent(model)

    with pytest.raises(ContentFilterError, match="Content filter triggered."):
        await agent.run("Run model")


async def test_content_filter_on_text_and_tool_call():
    tool_executed = False

    async def mock_model_func(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                TextPart(content="Partial content before block..."),
                ToolCallPart(
                    tool_name='my_tool',
                    args={'x': 42},
                    tool_call_id='call_1',
                )
            ],
            model_name='test-model',
            finish_reason='content_filter',
        )

    model = FunctionModel(function=mock_model_func, model_name='test-model')
    agent = Agent(model)

    @agent.tool_plain
    def my_tool(x: int) -> str:
        nonlocal tool_executed
        tool_executed = True
        return f"Result: {x}"

    with pytest.raises(ContentFilterError, match="Content filter triggered."):
        await agent.run("Run tool and text")

    assert not tool_executed, "Tool should not have been executed when content filter is triggered!"


async def test_content_filter_streaming(allow_model_requests: None):
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
    from openai.types.completion_usage import CompletionUsage

    def chunk(delta: list[ChoiceDelta], finish_reason: str | None = None) -> ChatCompletionChunk:
        return ChatCompletionChunk(
            id='123',
            choices=[
                ChunkChoice(index=index, delta=d, finish_reason=finish_reason)
                for index, d in enumerate(delta)
            ],
            created=1704067200,
            model='gpt-4o-123',
            object='chat.completion.chunk',
            usage=CompletionUsage(completion_tokens=1, prompt_tokens=2, total_tokens=3),
        )

    def text_chunk(text: str, finish_reason: str | None = None) -> ChatCompletionChunk:
        return chunk([ChoiceDelta(content=text, role='assistant')], finish_reason=finish_reason)

    stream = [
        text_chunk('Some safe output...'),
        text_chunk('Unsafe output starts...'),
        text_chunk('', finish_reason='content_filter'),
    ]
    mock_client = MockOpenAI.create_mock_stream(stream)
    m = OpenAIChatModel('gpt-5-mini', provider=OpenAIProvider(openai_client=mock_client))
    agent = Agent(m)

    async with agent.run_stream('Trigger filter in stream') as result:
        with pytest.raises(ContentFilterError, match="Content filter triggered."):
            async for text in result.stream_text():
                pass
            await result.get_output()
