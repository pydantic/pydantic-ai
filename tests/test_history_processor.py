import uuid
from collections.abc import AsyncIterator
from copy import deepcopy
from dataclasses import replace
from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
    capture_run_messages,
)
from pydantic_ai.capabilities import ProcessHistory, ReinjectSystemPrompt
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

from ._inline_snapshot import snapshot
from .conftest import IsDatetime, IsStr

pytestmark = [pytest.mark.anyio]


@pytest.fixture
def received_messages() -> list[ModelMessage]:
    return []


@pytest.fixture
def function_model(received_messages: list[ModelMessage]) -> FunctionModel:
    def capture_model_function(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Capture the messages that the provider actually receives
        received_messages.clear()
        received_messages.extend(messages)
        return ModelResponse(parts=[TextPart(content='Provider response')])

    async def capture_model_stream_function(messages: list[ModelMessage], _info: AgentInfo) -> AsyncIterator[str]:
        received_messages.clear()
        received_messages.extend(messages)
        yield 'hello'

    return FunctionModel(capture_model_function, stream_function=capture_model_stream_function)


async def test_history_processor_no_op(function_model: FunctionModel, received_messages: list[ModelMessage]):
    def no_op_history_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages

    agent = Agent(function_model, capabilities=[ProcessHistory(no_op_history_processor)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Previous question')]),
        ModelResponse(parts=[TextPart(content='Previous answer')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='New question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='New question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_run_replaces_message_history(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """Test that the history processor replaces the message history in the state."""

    def process_previous_answers(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Keep the last message (last question) and add a new system prompt
        return messages[-1:] + [ModelRequest(parts=[SystemPromptPart(content='Processed answer')])]

    agent = Agent(function_model, capabilities=[ProcessHistory(process_previous_answers)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
        ModelRequest(parts=[UserPromptPart(content='Question 2')]),
        ModelResponse(parts=[TextPart(content='Answer 2')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run('Question 3', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 3',
                        timestamp=IsDatetime(),
                    ),
                    SystemPromptPart(
                        content='Processed answer',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Question 3', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[SystemPromptPart(content='Processed answer', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()


async def test_history_processor_streaming_replaces_message_history(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """Test that the history processor replaces the message history in the state."""

    def process_previous_answers(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Keep the last message (last question) and add a new system prompt
        return messages[-1:] + [ModelRequest(parts=[SystemPromptPart(content='Processed answer')])]

    agent = Agent(function_model, capabilities=[ProcessHistory(process_previous_answers)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
        ModelRequest(parts=[UserPromptPart(content='Question 2')]),
        ModelResponse(parts=[TextPart(content='Answer 2')]),
    ]

    with capture_run_messages() as captured_messages:
        async with agent.run_stream('Question 3', message_history=message_history) as result:
            async for _ in result.stream_text():
                pass

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 3',
                        timestamp=IsDatetime(),
                    ),
                    SystemPromptPart(
                        content='Processed answer',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Question 3', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[SystemPromptPart(content='Processed answer', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello')],
                usage=RequestUsage(input_tokens=50, output_tokens=1),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()


async def test_history_processor_messages_sent_to_provider(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """Test what messages are actually sent to the provider after processing."""

    def capture_messages_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Filter out ModelResponse messages
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(function_model, capabilities=[ProcessHistory(capture_messages_processor)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Previous question')]),
        ModelResponse(parts=[TextPart(content='Previous answer')]),  # This should be filtered out
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Previous question',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='New question',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelRequest(
                parts=[UserPromptPart(content='New question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_multiple_history_processors(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """Test that multiple processors are applied in sequence."""

    def first_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Add a prefix to user prompts
        processed: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts: list[ModelRequestPart] = []
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):  # pragma: no branch
                        new_parts.append(UserPromptPart(content=f'[FIRST] {part.content}'))
                processed.append(ModelRequest(parts=new_parts))
            else:
                processed.append(msg)
        return processed

    def second_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Add another prefix to user prompts
        processed: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts: list[ModelRequestPart] = []
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):  # pragma: no branch
                        new_parts.append(UserPromptPart(content=f'[SECOND] {part.content}'))
                processed.append(ModelRequest(parts=new_parts))
            else:
                processed.append(msg)
        return processed

    agent = Agent(function_model, capabilities=[ProcessHistory(first_processor), ProcessHistory(second_processor)])

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Question')]),
        ModelResponse(parts=[TextPart(content='Answer')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)
    assert received_messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='[SECOND] [FIRST] Question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='[SECOND] [FIRST] New question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='[SECOND] [FIRST] Question',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Answer')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='[SECOND] [FIRST] New question',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=57, output_tokens=3),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_async_history_processor(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """Test that async processors work."""

    async def async_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(function_model, capabilities=[ProcessHistory(async_processor)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),  # Should be filtered out
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run('Question 2', message_history=message_history)
    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 1',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='Question 2',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 1',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 2',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_on_streamed_run(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """Test that history processors work on streamed runs."""

    async def async_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
    ]

    agent = Agent(function_model, capabilities=[ProcessHistory(async_processor)])
    with capture_run_messages() as captured_messages:
        async with agent.iter('Question 2', message_history=message_history) as run:
            async for node in run:
                if agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream.stream_response(debounce_by=None):
                            ...

    result = run.result
    assert result is not None
    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 1',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='Question 2',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 1',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 2',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello')],
                usage=RequestUsage(input_tokens=50, output_tokens=1),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_with_context(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """Test history processor that takes RunContext."""

    def context_processor(ctx: RunContext[str], messages: list[ModelMessage]) -> list[ModelMessage]:
        # Access deps from context
        prefix = ctx.deps
        processed: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts: list[ModelRequestPart] = []
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        new_parts.append(UserPromptPart(content=f'{prefix}: {part.content}'))
                    else:
                        new_parts.append(part)  # pragma: no cover
                processed.append(ModelRequest(parts=new_parts))
            else:
                processed.append(msg)  # pragma: no cover
        return processed

    agent = Agent(function_model, capabilities=[ProcessHistory(context_processor)], deps_type=str)
    with capture_run_messages() as captured_messages:
        result = await agent.run('test', deps='PREFIX')

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='PREFIX: test',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='PREFIX: test',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_with_context_async(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """Test async history processor that takes RunContext."""

    async def async_context_processor(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[-1:]  # Keep only the last message

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
        ModelRequest(parts=[UserPromptPart(content='Question 2')]),
        ModelResponse(parts=[TextPart(content='Answer 2')]),
    ]

    agent = Agent(function_model, capabilities=[ProcessHistory(async_context_processor)])
    with capture_run_messages() as captured_messages:
        result = await agent.run('Question 3', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 3',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Question 3',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_mixed_signatures(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """Test mixing processors with and without context."""

    def simple_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Filter out responses
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    def context_processor(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        # Add prefix based on deps
        prefix = getattr(ctx.deps, 'prefix', 'DEFAULT')
        processed: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts: list[ModelRequestPart] = []
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        new_parts.append(UserPromptPart(content=f'{prefix}: {part.content}'))
                    else:
                        new_parts.append(part)  # pragma: no cover
                processed.append(ModelRequest(parts=new_parts))
            else:
                processed.append(msg)  # pragma: no cover
        return processed

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),
    ]

    # Create deps with prefix attribute
    class Deps:
        prefix = 'TEST'

    agent = Agent(
        function_model,
        capabilities=[ProcessHistory(simple_processor), ProcessHistory(context_processor)],
        deps_type=Deps,
    )
    with capture_run_messages() as captured_messages:
        result = await agent.run('Question 2', message_history=message_history, deps=Deps())

    # Should have filtered responses and added prefix
    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='TEST: Question 1',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='TEST: Question 2',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='TEST: Question 1',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='TEST: Question 2',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=56, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_replace_messages(function_model: FunctionModel, received_messages: list[ModelMessage]):
    history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Original message')]),
        ModelResponse(parts=[TextPart(content='Original response')]),
        ModelRequest(parts=[UserPromptPart(content='Original followup')]),
    ]

    def return_new_history(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [
            ModelRequest(parts=[UserPromptPart(content='Modified message')]),
        ]

    agent = Agent(function_model, capabilities=[ProcessHistory(return_new_history)])

    with capture_run_messages() as captured_messages:
        result = await agent.run('foobar', message_history=history)

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Modified message',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Modified message',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_empty_history(function_model: FunctionModel, received_messages: list[ModelMessage]):
    def return_new_history(messages: list[ModelMessage]) -> list[ModelMessage]:
        return []

    agent = Agent(function_model, capabilities=[ProcessHistory(return_new_history)])

    with pytest.raises(UserError, match='Processed history cannot be empty.'):
        await agent.run('foobar')


async def test_history_processor_history_ending_in_response(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    def return_new_history(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [ModelResponse(parts=[TextPart(content='Provider response')])]

    agent = Agent(function_model, capabilities=[ProcessHistory(return_new_history)])

    with pytest.raises(UserError, match='Processed history must end with a `ModelRequest`.'):
        await agent.run('foobar')


async def test_callable_class_history_processor_no_op(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    class NoOpHistoryProcessor:
        def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
            return messages

    agent = Agent(function_model, capabilities=[ProcessHistory(NoOpHistoryProcessor())])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Previous question')]),
        ModelResponse(parts=[TextPart(content='Previous answer')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='New question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='New question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_callable_class_history_processor_with_ctx_no_op(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    class NoOpHistoryProcessorWithCtx:
        def __call__(self, _: RunContext, messages: list[ModelMessage]) -> list[ModelMessage]:
            return messages

    agent = Agent(function_model, capabilities=[ProcessHistory(NoOpHistoryProcessorWithCtx())])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Previous question')]),
        ModelResponse(parts=[TextPart(content='Previous answer')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='New question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='New question', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_new_messages_index_during_iter_with_pruning():
    """
    When a pruning history processor removes the initial user prompt during
    a multi-step tool calling run, new_messages() should still return all
    messages generated in this run.
    """

    def keep_last_2(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[-2:] if len(messages) > 2 else messages

    call_count = 0

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id='tool_call_1')],
            )
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(model=FunctionModel(model_function, model_name='test'), capabilities=[ProcessHistory(keep_last_2)])

    @agent.tool
    async def my_tool(ctx: RunContext) -> str:
        return 'tool executed'

    with capture_run_messages() as captured_messages:
        async with agent.iter('start') as run:
            async for _ in run:
                pass

    result = run.result
    assert result is not None

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=2),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content='tool executed',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=52, output_tokens=3),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()


async def test_new_messages_index_during_iter_with_pruning_and_history():
    """
    When running with prior message_history and a pruning history processor
    that progressively removes older messages during a multi-step tool calling
    run, new_messages() should return only the messages from the current run,
    excluding the pruned history.
    """

    def keep_last_2(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[-2:] if len(messages) > 2 else messages

    call_count = 0

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id='tool_call_1')],
            )
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(model=FunctionModel(model_function, model_name='test'), capabilities=[ProcessHistory(keep_last_2)])

    @agent.tool
    async def my_tool(ctx: RunContext) -> str:
        return 'tool executed'

    history = [
        ModelRequest(parts=[UserPromptPart(content='Old message 1')]),
        ModelResponse(parts=[TextPart(content='Old response 1')]),
    ]

    with capture_run_messages() as captured_messages:
        async with agent.iter('start', message_history=history) as run:
            async for _ in run:
                pass

    result = run.result
    assert result is not None

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id=IsStr())],
                usage=RequestUsage(input_tokens=51, output_tokens=5),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content='tool executed',
                        tool_call_id=IsStr(),
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=52, output_tokens=3),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()


async def test_history_processor_reorder_old_new(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """
    When a history processor reorders old and new messages, the old history
    message receives the current run_id, so new_messages() treats it as
    part of the current run and includes it in the result.
    """

    def swap_last_two(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[:-2] + messages[-2:][::-1]

    agent = Agent(function_model, capabilities=[ProcessHistory(swap_last_two)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Old question')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='New question', timestamp=IsDatetime()),
                    UserPromptPart(content='Old question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='New question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='Old question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == result.all_messages()


async def test_history_processor_injects_into_new_stream(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    When a history processor injects a new message tagged with the current
    run_id into the message list, new_messages() should include the injected
    message alongside the other messages from this run.
    """

    def inject_middle(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        return (
            messages[:-1]
            + [ModelRequest(parts=[UserPromptPart(content='Inserted')], run_id=ctx.run_id)]
            + messages[-1:]
        )

    agent = Agent(function_model, capabilities=[ProcessHistory(inject_middle)])

    message_history = [ModelRequest(parts=[UserPromptPart(content='Old')])]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Old', timestamp=IsDatetime()),
                    UserPromptPart(content='Inserted', timestamp=IsDatetime()),
                    UserPromptPart(content='New question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Old', timestamp=IsDatetime()),
                ]
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='Inserted', timestamp=IsDatetime()),
                ],
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='New question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    new_msgs = result.new_messages()
    assert new_msgs == result.all_messages()[1:]


async def test_history_processor_injects_without_run_id_before_current_run(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    When a history processor injects a message without a run_id before the
    current run, new_messages() should exclude the injected message and only
    return messages that belong to the current run.
    """

    def inject_middle_without_run_id(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[:-1] + [ModelRequest(parts=[UserPromptPart(content='Inserted')])] + messages[-1:]

    agent = Agent(function_model, capabilities=[ProcessHistory(inject_middle_without_run_id)])

    message_history = [ModelRequest(parts=[UserPromptPart(content='Old')])]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Old', timestamp=IsDatetime()),
                    UserPromptPart(content='Inserted', timestamp=IsDatetime()),
                    UserPromptPart(content='New question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Old', timestamp=IsDatetime()),
                ]
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='Inserted', timestamp=IsDatetime()),
                ]
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='New question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[2:]


async def test_history_processor_overrides_run_id_uses_response_as_new_messages(function_model: FunctionModel):
    """
    When a history processor overwrites the run_id on all messages,
    new_messages() should fall back to returning only the model response
    appended after processing.
    """

    def override_run_id(ctx: RunContext[Any], messages: list[ModelMessage]) -> list[ModelMessage]:
        override = f'{ctx.run_id}-override'
        for message in messages:
            message.run_id = override
        return messages

    agent = Agent(function_model, capabilities=[ProcessHistory(override_run_id)])

    message_history = [ModelRequest(parts=[UserPromptPart(content='Old')])]

    with capture_run_messages() as captured_messages:
        result = await agent.run('New question', message_history=message_history)

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(content='Old', timestamp=IsDatetime()),
                ],
                run_id=IsStr(regex='.+-override'),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(content='New question', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(regex='.+-override'),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=53, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == result.all_messages()[-1:]


async def test_history_processor_resuming_without_prompt(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    When running without a user prompt (resuming from history), new_messages()
    should exclude the request supplied via message_history even when that
    request gets the current run_id.
    """

    def prepend_summary(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [ModelRequest(parts=[SystemPromptPart(content='History summary')]), *messages]

    agent = Agent(function_model, capabilities=[ProcessHistory(prepend_summary)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Original prompt')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='History summary',
                        timestamp=IsDatetime(),
                    ),
                    UserPromptPart(
                        content='Original prompt',
                        timestamp=IsDatetime(),
                    ),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(
                        content='History summary',
                        timestamp=IsDatetime(),
                    )
                ]
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Original prompt',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-1:]


async def test_resuming_without_prompt_with_tool_calls_excludes_resumed_request():
    """
    When resuming without a user prompt and the model enters a tool-call loop,
    new_messages() should exclude the resumed history request even though it
    gets the current run_id.
    """

    call_count = 0

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id='tool_call_1')],
            )
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(model=FunctionModel(model_function, model_name='test'))

    @agent.tool
    async def my_tool(_ctx: RunContext) -> str:
        return 'tool executed'

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=[ModelRequest(parts=[UserPromptPart(content='Original prompt')])])

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Original prompt', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id='tool_call_1')],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name='my_tool',
                        content='tool executed',
                        tool_call_id='tool_call_1',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='done')],
                usage=RequestUsage(input_tokens=54, output_tokens=3),
                model_name='test',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == result.all_messages()[1:]


async def test_resuming_without_prompt_excludes_request_with_different_run_id(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    When running without a user prompt and the resumed request already has a
    run_id from a *previous* run, new_messages() should exclude it — only
    messages stamped with the current run_id should be returned.
    """
    previous_run_id = str(uuid.uuid4())

    agent = Agent(function_model)

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Earlier question')]),
        ModelResponse(parts=[TextPart(content='Earlier answer')]),
        ModelRequest(
            parts=[UserPromptPart(content='Previous run prompt')],
            run_id=previous_run_id,
        ),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=message_history)

    assert received_messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Earlier question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Earlier answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='Previous run prompt', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=previous_run_id,
                conversation_id=IsStr(),
            ),
        ]
    )

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Earlier question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Earlier answer')], timestamp=IsDatetime()),
            ModelRequest(
                parts=[UserPromptPart(content='Previous run prompt', timestamp=IsDatetime())],
                timestamp=IsDatetime(),
                run_id=previous_run_id,
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=55, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # The resumed request has a run_id from a different run: excluded from new_messages().
    assert result.new_messages() == result.all_messages()[-1:]
    assert result.new_messages()[0].run_id != previous_run_id


async def test_history_processor_deepcopy_resuming_without_prompt(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    When a history processor deep-copies messages (breaking object identity),
    new_messages() should still exclude the resumed request supplied via
    message_history.
    """

    def deepcopy_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return deepcopy(messages)

    agent = Agent(function_model, capabilities=[ProcessHistory(deepcopy_processor)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Original prompt')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=message_history)

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Original prompt',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == result.all_messages()[-1:]


async def test_history_processor_rebuild_resuming_without_prompt(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    When a history processor rebuilds `ModelRequest` instances with equivalent
    values, new_messages() should still exclude the resumed request supplied
    via message_history.
    """

    def rebuild_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        rebuilt_messages: list[ModelMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                rebuilt_messages.append(
                    ModelRequest(
                        parts=list(message.parts),
                        timestamp=message.timestamp,
                        instructions=message.instructions,
                        run_id=message.run_id,
                        metadata=message.metadata.copy() if message.metadata is not None else None,
                    )
                )
            else:
                rebuilt_messages.append(message)
        return rebuilt_messages

    agent = Agent(function_model, capabilities=[ProcessHistory(rebuild_processor)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Old question')]),
        ModelResponse(parts=[TextPart(content='Old answer')]),
        ModelRequest(parts=[UserPromptPart(content='Original prompt')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=message_history)

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Old question',
                        timestamp=IsDatetime(),
                    )
                ],
            ),
            ModelResponse(
                parts=[TextPart(content='Old answer')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Original prompt',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    assert result.new_messages() == result.all_messages()[-1:]


async def test_history_processor_replace_resumed_request_excludes_resumed_request(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    When a history processor replaces the resumed request with completely different content,
    new_messages() still excludes it. The full rewrite defeats both the object identity and value
    matches, so the pinned-position fallback is what keeps it excluded. This is consistent with the
    other (non-resumed) request the same processor replaces, which is likewise prior context: a
    processor's transient reshaping of prior context is not a new message to persist.
    """

    def replace_all_requests(messages: list[ModelMessage]) -> list[ModelMessage]:
        rebuilt: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                rebuilt.append(
                    ModelRequest(
                        parts=[UserPromptPart(content='Replaced content')],
                        timestamp=msg.timestamp,
                        run_id=msg.run_id,
                    )
                )
            else:
                rebuilt.append(msg)
        return rebuilt

    agent = Agent(function_model, capabilities=[ProcessHistory(replace_all_requests)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Old question')]),
        ModelResponse(parts=[TextPart(content='Old answer')]),
        ModelRequest(parts=[UserPromptPart(content='Original prompt')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=message_history)

    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Replaced content',
                        timestamp=IsDatetime(),
                    )
                ],
            ),
            ModelResponse(
                parts=[TextPart(content='Old answer')],
                timestamp=IsDatetime(),
            ),
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Replaced content',
                        timestamp=IsDatetime(),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )

    # The resumed request is excluded via the position fallback even though the processor rebuilt
    # it with the current run_id and different content; only the model response is new.
    assert result.new_messages() == result.all_messages()[-1:]


async def test_reinject_system_prompt_resuming_without_prompt_excludes_resumed_request(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    System-prompt reinjection (the UI adapters' default with `manage_system_prompt='server'`)
    rebuilds the first request via `replace(...)`, prepending a `SystemPromptPart`. When
    resuming without a new user prompt on the first turn, that first request *is* the resumed
    request, so the rewrite changes both its identity and its `parts`. It must still be treated
    as prior context and excluded from new_messages(). Regression test for the turn-1 leak in
    https://github.com/pydantic/pydantic-ai/issues/6025.
    """

    agent = Agent(
        function_model,
        system_prompt='Server prompt',
        capabilities=[ReinjectSystemPrompt(replace_existing=True)],
    )

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Original prompt')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=message_history)

    # The model received the resumed request with the server prompt reinjected into it.
    assert received_messages == snapshot(
        [
            ModelRequest(
                parts=[
                    SystemPromptPart(content='Server prompt', timestamp=IsDatetime()),
                    UserPromptPart(content='Original prompt', timestamp=IsDatetime()),
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
                conversation_id=IsStr(),
            )
        ]
    )
    assert captured_messages == result.all_messages()
    # The reinjected resumed request is excluded; only the model response is new.
    assert result.new_messages() == result.all_messages()[-1:]


async def test_history_processor_mutates_resumed_request_excludes_resumed_request(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    A history processor that rebuilds the trailing resumed request with any changed field
    (here `metadata`) must not leak it into new_messages(). Re-matching the request by value
    would fail on the changed field and fall back to run_id detection — which the framework
    stamps on the resumed request — leaking it. Tracking the boundary by position excludes it
    regardless of the rewrite. Covers the generalized leak described in
    https://github.com/pydantic/pydantic-ai/issues/6025.
    """

    def touch_request_metadata(messages: list[ModelMessage]) -> list[ModelMessage]:
        rebuilt: list[ModelMessage] = []
        for message in messages:
            if isinstance(message, ModelRequest):
                rebuilt.append(replace(message, metadata={'touched': True}))
            else:
                rebuilt.append(message)
        return rebuilt

    agent = Agent(function_model, capabilities=[ProcessHistory(touch_request_metadata)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Earlier question')]),
        ModelResponse(parts=[TextPart(content='Earlier answer')]),
        ModelRequest(parts=[UserPromptPart(content='Original prompt')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=message_history)

    assert captured_messages == result.all_messages()
    # The resumed request reached the model carrying the mutated metadata...
    assert result.all_messages()[-2].metadata == snapshot({'touched': True})
    # ...but it is still excluded from new_messages(): only the model response is new.
    assert result.new_messages() == result.all_messages()[-1:]


def _user_request_present(messages: list[ModelMessage]) -> bool:
    return any(isinstance(m, ModelRequest) and any(isinstance(p, UserPromptPart) for p in m.parts) for m in messages)


async def test_reinject_system_prompt_resumed_tool_loop_excludes_resumed_request():
    """
    System-prompt reinjection rebuilds the resumed request in place on *every* step of a
    multi-step resumed run (a tool-call loop). The resumed request must stay excluded from
    new_messages() across all steps, while every message produced this run is included.
    The pinned boundary is translated per step so the in-place rebuild never leaks it.
    """

    call_count = 0

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id='tool_call_1')])
        return ModelResponse(parts=[TextPart(content='done')])

    agent = Agent(
        model=FunctionModel(model_function, model_name='test'),
        system_prompt='Server prompt',
        capabilities=[ReinjectSystemPrompt(replace_existing=True)],
    )

    @agent.tool
    async def my_tool(_ctx: RunContext) -> str:
        return 'tool executed'

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=[ModelRequest(parts=[UserPromptPart(content='Original prompt')])])

    assert captured_messages == result.all_messages()
    # The reinjected resumed request stays excluded; everything produced this run is new.
    assert result.new_messages() == result.all_messages()[1:]
    assert not _user_request_present(result.new_messages())


async def test_history_processor_truncation_during_resumed_tool_loop_keeps_run_messages():
    """
    A "keep last N" history processor can drop the resumed request partway through a
    multi-step resumed run (here a tool-call loop). Once the resumed request is gone the
    pinned boundary must not exclude messages produced earlier in the same run: new_messages()
    should return every surviving message. Regression for the stale-pinned-index case raised
    in review of the position-based boundary fix.
    """

    call_count = 0

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id='tool_call_1')])
        return ModelResponse(parts=[TextPart(content='done')])

    def keep_last_two(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages[-2:]

    agent = Agent(
        model=FunctionModel(model_function, model_name='test'),
        capabilities=[ProcessHistory(keep_last_two)],
    )

    @agent.tool
    async def my_tool(_ctx: RunContext) -> str:
        return 'tool executed'

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=[ModelRequest(parts=[UserPromptPart(content='Original prompt')])])

    assert captured_messages == result.all_messages()
    # The resumed request was truncated away, so every surviving message is from this run —
    # in particular the first model response must not be dropped by a stale boundary.
    assert result.new_messages() == result.all_messages()
    assert not _user_request_present(result.new_messages())


async def test_history_processor_removes_message_after_resumed_request_excludes_resumed_request():
    """
    A history processor can remove a message positioned *after* the resumed request on a later step
    (here dropping the model's tool-call response once the tool result is in history). The pinned
    position can't follow a removal after the resumed request, but the resumed request object is
    left untouched, so identity matching still excludes it from new_messages(). Guards against the
    regression a purely position-based boundary would introduce — object matching and position
    matching each cover mutations the other misses.
    """

    call_count = 0

    def model_function(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return ModelResponse(parts=[ToolCallPart(tool_name='my_tool', args={}, tool_call_id='tool_call_1')])
        return ModelResponse(parts=[TextPart(content='done')])

    def drop_tool_call_response(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Only once the tool result is back: drop the model's tool-call response, which sits after
        # the resumed request — a count change after it that the pinned index cannot track.
        has_tool_return = any(
            isinstance(m, ModelRequest) and any(isinstance(p, ToolReturnPart) for p in m.parts) for m in messages
        )
        if not has_tool_return:
            return messages
        first_response = next((m for m in messages if isinstance(m, ModelResponse)), None)
        return [m for m in messages if m is not first_response]

    agent = Agent(
        model=FunctionModel(model_function, model_name='test'),
        capabilities=[ProcessHistory(drop_tool_call_response)],
    )

    @agent.tool
    async def my_tool(_ctx: RunContext) -> str:
        return 'tool result'

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=[ModelRequest(parts=[UserPromptPart(content='Original prompt')])])

    assert captured_messages == result.all_messages()
    # The tool-call response was dropped, but the resumed request stays excluded via identity
    # matching; only the messages after it are new.
    assert result.new_messages() == result.all_messages()[1:]
    assert not _user_request_present(result.new_messages())


async def test_history_processor_insert_and_replace_resumed_request_excludes_resumed_request(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """
    A processor can both insert a message ahead of the resumed request AND rebuild the
    resumed request itself in the same pass. The boundary is pinned by position *after*
    processing runs (when the resumed request is the trailing message), so neither the
    inserted message nor the rebuilt resumed request leaks into new_messages() — only the
    model response is new. Covers the combined insert+replace case raised in review of the
    position-based fix for https://github.com/pydantic/pydantic-ai/issues/6025.
    """

    def insert_and_replace(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Rebuild every request (changed `metadata` defeats value re-matching) and prepend a
        # fresh request at the front, shifting the resumed request off its original position.
        rebuilt: list[ModelMessage] = [
            replace(message, metadata={'touched': True}) if isinstance(message, ModelRequest) else message
            for message in messages
        ]
        rebuilt.insert(0, ModelRequest(parts=[SystemPromptPart(content='Injected context')]))
        return rebuilt

    agent = Agent(function_model, capabilities=[ProcessHistory(insert_and_replace)])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Earlier question')]),
        ModelResponse(parts=[TextPart(content='Earlier answer')]),
        ModelRequest(parts=[UserPromptPart(content='Original prompt')]),
    ]

    with capture_run_messages() as captured_messages:
        result = await agent.run(message_history=message_history)

    assert captured_messages == result.all_messages()
    # Both the inserted request and the rebuilt resumed request are prior context; only the
    # model response is new, and no user request leaks in.
    assert result.new_messages() == result.all_messages()[-1:]
    assert not _user_request_present(result.new_messages())


def test_takes_ctx_returns_false_for_untyped_processor():
    """takes_run_context returns False when the processor's first param has no type annotation."""
    from pydantic_ai._utils import takes_run_context

    def untyped_processor(messages) -> list[ModelMessage]:  # pyright: ignore[reportUnknownParameterType,reportMissingParameterType]
        return messages  # pyright: ignore[reportUnknownVariableType] # pragma: no cover

    # When first param has no type annotation, takes_run_context returns False
    assert takes_run_context(untyped_processor) is False  # pyright: ignore[reportUnknownArgumentType]
