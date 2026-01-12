from dataclasses import dataclass
from datetime import timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior

from ..conftest import IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    from mcp import CreateMessageResult
    from mcp.types import TextContent

    from pydantic_ai.models.mcp_sampling import MCPSamplingModel

pytestmark = pytest.mark.skipif(not imports_successful(), reason='mcp package not installed')


@dataclass
class FakeSession:
    create_message: Any


def fake_session(create_message: Any) -> Any:
    return FakeSession(create_message)


def test_mcp_sampling_model():
    model = MCPSamplingModel(fake_session(AsyncMock()))
    assert model.model_name == 'mcp-sampling'
    assert model.system == 'MCP'


def test_assistant_text():
    result = CreateMessageResult(
        role='assistant', content=TextContent(type='text', text='text content'), model='test-model'
    )
    create_message = AsyncMock(return_value=result)
    agent = Agent(model=MCPSamplingModel(fake_session(create_message)))

    result = agent.run_sync('Hello')
    assert result.output == snapshot('text content')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content='Hello',
                        timestamp=IsNow(tz=timezone.utc),
                    )
                ],
                timestamp=IsDatetime(),
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='text content')],
                model_name='test-model',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_user_text():
    result = CreateMessageResult(role='user', content=TextContent(type='text', text='text content'), model='test-model')
    create_message = AsyncMock(return_value=result)
    agent = Agent(model=MCPSamplingModel(fake_session(create_message)))

    expected_match = 'Unexpected result from MCP sampling, expected "assistant" role, got user.'
    with pytest.raises(UnexpectedModelBehavior, match=expected_match):
        agent.run_sync('Hello')


def test_assistant_text_history():
    result = CreateMessageResult(
        role='assistant', content=TextContent(type='text', text='text content'), model='test-model'
    )
    create_message = AsyncMock(return_value=result)
    agent = Agent(model=MCPSamplingModel(fake_session(create_message)), instructions='testing')

    result = agent.run_sync('1')
    result = agent.run_sync('2', message_history=result.all_messages())

    assert result.output == snapshot('text content')
    assert result.all_messages() == snapshot(
        [
            ModelRequest(
                parts=[UserPromptPart(content='1', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                instructions='testing',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='text content')],
                model_name='test-model',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='2', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                instructions='testing',
                run_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='text content')],
                model_name='test-model',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
            ),
        ]
    )


def test_assistant_text_history_complex():
    history = [
        ModelRequest(
            parts=[
                UserPromptPart(content='1'),
                UserPromptPart(content=['a string', BinaryContent(data=b'data', media_type='image/jpeg')]),
                SystemPromptPart(content='system content'),
            ],
            timestamp=IsDatetime(),
        ),
        ModelResponse(
            parts=[TextPart(content='text content')],
            model_name='test-model',
        ),
    ]

    result = CreateMessageResult(
        role='assistant', content=TextContent(type='text', text='text content'), model='test-model'
    )
    create_message = AsyncMock(return_value=result)
    agent = Agent(model=MCPSamplingModel(fake_session(create_message)))
    result = agent.run_sync('1', message_history=history)
    assert result.output == snapshot('text content')


def test_mcp_sampling_history_with_tool_return():
    result = CreateMessageResult(
        role='assistant', content=TextContent(type='text', text='text content'), model='test-model'
    )
    create_message = AsyncMock(return_value=result)
    agent = Agent(model=MCPSamplingModel(fake_session(create_message)))

    history = [
        ModelRequest(
            parts=[
                ToolReturnPart(tool_name='test_tool', content={'value': 1}, tool_call_id='tool-call-1'),
            ],
            timestamp=IsDatetime(),
        )
    ]

    agent.run_sync('Hello', message_history=history)

    sampling_messages = create_message.call_args.args[0]
    assert [message.model_dump(by_alias=True) for message in sampling_messages] == snapshot(
        [
            {
                'role': 'user',
                'content': {
                    'type': 'text',
                    'text': 'Tool `test_tool` (id=tool-call-1) returned:\n{"value": 1}',
                    'annotations': None,
                    '_meta': None,
                },
            },
            {
                'role': 'user',
                'content': {'type': 'text', 'text': 'Hello', 'annotations': None, '_meta': None},
            },
        ]
    )
