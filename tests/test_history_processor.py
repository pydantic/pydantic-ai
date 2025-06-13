import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelRequestPart, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import Usage

from .conftest import IsDatetime

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

    return FunctionModel(capture_model_function)


async def test_history_processor_no_op(function_model: FunctionModel, received_messages: list[ModelMessage]):
    def no_op_history_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        return messages

    agent = Agent(function_model, history_processors=[no_op_history_processor])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Previous question')]),
        ModelResponse(parts=[TextPart(content='Previous answer')]),
    ]

    result = await agent.run('New question', message_history=message_history)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(parts=[UserPromptPart(content='New question', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=Usage(requests=1, request_tokens=54, response_tokens=4, total_tokens=58),
                model_name='function:capture_model_function:',
                timestamp=IsDatetime(),
            ),
        ]
    )
    assert received_messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(parts=[UserPromptPart(content='New question', timestamp=IsDatetime())]),
        ]
    )


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

    result = await agent.run('New question', message_history=message_history)
    assert result.all_messages() == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelResponse(parts=[TextPart(content='Previous answer')], timestamp=IsDatetime()),
            ModelRequest(parts=[UserPromptPart(content='New question', timestamp=IsDatetime())]),
            ModelResponse(
                parts=[TextPart(content='Provider response')],
                usage=Usage(requests=1, request_tokens=54, response_tokens=2, total_tokens=56),
                model_name='function:capture_model_function:',
                timestamp=IsDatetime(),
            ),
        ]
    )
    assert received_messages == snapshot(
        [
            ModelRequest(parts=[UserPromptPart(content='Previous question', timestamp=IsDatetime())]),
            ModelRequest(parts=[UserPromptPart(content='New question', timestamp=IsDatetime())]),
        ]
    )


async def test_multiple_history_processors(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """Test that multiple processors are applied in sequence."""

    def first_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Add a prefix to user prompts
        processed: list[ModelMessage] = []
        for msg in messages:
            if isinstance(msg, ModelRequest):
                new_parts: list[ModelRequestPart] = []
                for part in msg.parts:
                    if isinstance(part, UserPromptPart):
                        new_parts.append(UserPromptPart(content=f'[FIRST] {part.content}'))
                    else:
                        new_parts.append(part)
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
                    if isinstance(part, UserPromptPart):
                        new_parts.append(UserPromptPart(content=f'[SECOND] {part.content}'))
                    else:
                        new_parts.append(part)
                processed.append(ModelRequest(parts=new_parts))
            else:
                processed.append(msg)
        return processed

    agent = Agent(function_model, history_processors=[first_processor, second_processor])

    message_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content='Question')]),
    ]

    await agent.run('New question', message_history=message_history)

    # Check that both processors were applied in sequence
    assert len(received_messages) == 2
    first_part = received_messages[0].parts[0]
    assert isinstance(first_part, UserPromptPart)
    assert first_part.content == '[SECOND] [FIRST] Question'
    second_part = received_messages[1].parts[0]
    assert isinstance(second_part, UserPromptPart)
    assert second_part.content == '[SECOND] [FIRST] New question'


async def test_async_history_processor(function_model: FunctionModel, received_messages: list[ModelMessage]):
    """Test that async processors work."""
    import asyncio

    async def async_processor(messages: list[ModelMessage]) -> list[ModelMessage]:
        # Simulate async work
        await asyncio.sleep(0.001)
        # Filter out responses
        return [msg for msg in messages if isinstance(msg, ModelRequest)]

    agent = Agent(function_model, history_processors=[async_processor])

    message_history = [
        ModelRequest(parts=[UserPromptPart(content='Question 1')]),
        ModelResponse(parts=[TextPart(content='Answer 1')]),  # Should be filtered out
    ]

    await agent.run('Question 2', message_history=message_history)

    # Should only have the two requests, response filtered out
    assert len(received_messages) == 2
    assert all(isinstance(msg, ModelRequest) for msg in received_messages)
    first_part = received_messages[0].parts[0]
    assert isinstance(first_part, UserPromptPart)
    assert first_part.content == 'Question 1'
    second_part = received_messages[1].parts[0]
    assert isinstance(second_part, UserPromptPart)
    assert second_part.content == 'Question 2'
