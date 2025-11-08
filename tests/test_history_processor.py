from collections.abc import AsyncIterator
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    Agent,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
    capture_run_messages,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RequestUsage

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

    agent = Agent(function_model, history_processors=[no_op_history_processor])

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
                run_id=IsStr(),
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
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

    agent = Agent(function_model, history_processors=[process_previous_answers])

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
                ]
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Question 3', timestamp=IsDatetime())],
                run_id=IsStr(),
            ),
            ModelRequest(parts=[SystemPromptPart(content='Processed answer', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_streaming_replaces_message_history(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """Test that the history processor replaces the message history in the state."""

    def process_previous_answers(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Keep the last message (last question) and add a new system prompt
        return messages[-1:] + [ModelRequest(parts=[SystemPromptPart(content='Processed answer')])]

    agent = Agent(function_model, history_processors=[process_previous_answers])

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
                ]
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='Question 3', timestamp=IsDatetime())],
                run_id=IsStr(),
            ),
            ModelRequest(parts=[SystemPromptPart(content='Processed answer', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='hello')],
                usage=RequestUsage(input_tokens=50, output_tokens=1),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_messages_sent_to_provider(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    """Test what messages are actually sent to the provider after processing."""

    def capture_messages_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Filter out ModelResponse messages
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(function_model, history_processors=[capture_messages_processor])

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
                ]
            )
        ]
    )
    assert captured_messages == result.all_messages()
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelRequest(
                parts=[UserPromptPart(content='New question', timestamp=IsDatetime())],
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
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

    agent = Agent(function_model, history_processors=[first_processor, second_processor])

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
            ModelRequest(parts=[UserPromptPart(content='[SECOND] [FIRST] New question', timestamp=IsDatetime())]),
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
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=57, output_tokens=3),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_async_history_processor(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """Test that async processors work."""

    async def async_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(function_model, history_processors=[async_processor])

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
                ]
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
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

    agent = Agent(function_model, history_processors=[async_processor])
    with capture_run_messages() as captured_messages:
        async with agent.iter('Question 2', message_history=message_history) as run:
            async for node in run:
                if agent.is_model_request_node(node):
                    async with node.stream(run.ctx) as stream:
                        async for _ in stream.stream_responses(debounce_by=None):
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
                ]
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='hello')],
                usage=RequestUsage(input_tokens=50, output_tokens=1),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
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

    agent = Agent(function_model, history_processors=[context_processor], deps_type=str)
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
                ]
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
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
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

    agent = Agent(function_model, history_processors=[async_context_processor])
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
                run_id=IsStr(),
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
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

    agent = Agent(function_model, history_processors=[simple_processor, context_processor], deps_type=Deps)
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
                ]
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
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=56, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
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

    agent = Agent(function_model, history_processors=[return_new_history])

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
                ]
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
                ]
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=52, output_tokens=2),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]


async def test_history_processor_empty_history(function_model: FunctionModel, received_messages: list[ModelMessage]):
    def return_new_history(messages: list[ModelMessage]) -> list[ModelMessage]:
        return []

    agent = Agent(function_model, history_processors=[return_new_history])

    with pytest.raises(UserError, match='Processed history cannot be empty.'):
        await agent.run('foobar')


async def test_history_processor_history_ending_in_response(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    def return_new_history(messages: list[ModelMessage]) -> list[ModelMessage]:
        return [ModelResponse(parts=[TextPart(content='Provider response')])]

    agent = Agent(function_model, history_processors=[return_new_history])

    with pytest.raises(UserError, match='Processed history must end with a `ModelRequest`.'):
        await agent.run('foobar')


async def test_callable_class_history_processor_no_op(
    function_model: FunctionModel, received_messages: list[ModelMessage]
):
    class NoOpHistoryProcessor:
        def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
            return messages

    agent = Agent(function_model, history_processors=[NoOpHistoryProcessor()])

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
                run_id=IsStr(),
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
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

    agent = Agent(function_model, history_processors=[NoOpHistoryProcessorWithCtx()])

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
                run_id=IsStr(),
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
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=RequestUsage(input_tokens=54, output_tokens=4),
                model_name='function:capture_model_function:capture_model_stream_function',
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
        ]
    )
    assert result.new_messages() == result.all_messages()[-2:]
