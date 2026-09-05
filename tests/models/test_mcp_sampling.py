from dataclasses import dataclass
from datetime import timezone
from typing import Any
from unittest.mock import AsyncMock

import pytest

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
from pydantic_ai.messages import RetryPromptPart, ToolCallPart

from .._inline_snapshot import snapshot
from ..conftest import IsDatetime, IsNow, IsStr, try_import

with try_import() as imports_successful:
    # `mcp.types` serves either SDK generation: v2 keeps it as an exact re-export of `mcp_types`.
    from mcp.types import CreateMessageResult, TextContent

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
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='text content')],
                model_name='test-model',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
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
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='text content')],
                model_name='test-model',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelRequest(
                parts=[UserPromptPart(content='2', timestamp=IsNow(tz=timezone.utc))],
                timestamp=IsDatetime(),
                instructions='testing',
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
            ModelResponse(
                parts=[TextPart(content='text content')],
                model_name='test-model',
                timestamp=IsNow(tz=timezone.utc),
                run_id=IsStr(),
                conversation_id=IsStr(),
            ),
        ]
    )


def test_standing_system_prompt_history():
    history = [
        ModelRequest(parts=[SystemPromptPart(content='standing system content'), UserPromptPart(content='1')]),
        ModelResponse(parts=[TextPart(content='text content')], model_name='test-model'),
    ]

    result = CreateMessageResult(
        role='assistant', content=TextContent(type='text', text='text content'), model='test-model'
    )
    create_message = AsyncMock(return_value=result)
    agent = Agent(model=MCPSamplingModel(fake_session(create_message)))
    agent.run_sync('2', message_history=history)

    sampling_messages = create_message.call_args.args[0]
    assert create_message.call_args.kwargs['system_prompt'] == 'standing system content'
    assert all(
        not isinstance(message.content, TextContent) or message.content.text != 'standing system content'
        for message in sampling_messages
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
    sampling_messages = create_message.call_args.args[0]
    assert create_message.call_args.kwargs['system_prompt'] == ''
    assert any(
        isinstance(message.content, TextContent) and message.content.text == '<system>system content</system>'
        for message in sampling_messages
    )


def test_tool_history_rendered_as_text():
    """A history containing a tool call/return is rendered as text instead of raising.

    The MCP sampling protocol only carries text/image/audio content, so a conversation that used
    tools on another model can't round-trip natively — but it should map to readable text rather
    than crash with `UnexpectedModelBehavior` (or silently drop the tool result).
    """
    result = CreateMessageResult(
        role='assistant', content=TextContent(type='text', text='text content'), model='test-model'
    )
    create_message = AsyncMock(return_value=result)
    agent = Agent(model=MCPSamplingModel(fake_session(create_message)))

    tool_call_id = 'pyd_ai_test_12345'
    history = [
        ModelRequest(parts=[UserPromptPart(content='what time is it')]),
        ModelResponse(
            parts=[ToolCallPart(tool_name='get_time', args={'tz': 'UTC'}, tool_call_id=tool_call_id)],
            model_name='test-model',
        ),
        ModelRequest(parts=[ToolReturnPart(tool_name='get_time', content='12:00', tool_call_id=tool_call_id)]),
    ]

    result = agent.run_sync('thanks', message_history=history)
    assert result.output == snapshot('text content')

    sampling_messages = create_message.call_args.args[0]
    assert [(m.role, m.content.text) for m in sampling_messages] == snapshot(
        [
            ('user', 'what time is it'),
            ('assistant', '[Tool pyd_ai_test_12345: get_time({"tz":"UTC"})]'),
            ('user', '[Tool pyd_ai_test_12345: get_time returned: 12:00]'),
            ('user', 'thanks'),
        ]
    )


def test_retry_prompt_history_rendered_as_text():
    """A `RetryPromptPart` from a prior turn is rendered as text instead of being dropped."""
    result = CreateMessageResult(
        role='assistant', content=TextContent(type='text', text='text content'), model='test-model'
    )
    create_message = AsyncMock(return_value=result)
    agent = Agent(model=MCPSamplingModel(fake_session(create_message)))

    tool_call_id = 'pyd_ai_test_67890'
    history = [
        ModelRequest(parts=[UserPromptPart(content='what time is it')]),
        ModelResponse(
            parts=[ToolCallPart(tool_name='get_time', args={}, tool_call_id=tool_call_id)],
            model_name='test-model',
        ),
        ModelRequest(
            parts=[RetryPromptPart(content='wrong arguments', tool_name='get_time', tool_call_id=tool_call_id)]
        ),
    ]

    result = agent.run_sync('try again', message_history=history)
    assert result.output == snapshot('text content')

    sampling_messages = create_message.call_args.args[0]
    assert [(m.role, m.content.text) for m in sampling_messages] == snapshot(
        [
            ('user', 'what time is it'),
            ('assistant', '[Tool pyd_ai_test_67890: get_time({})]'),
            ('user', '[Tool pyd_ai_test_67890: get_time error: wrong arguments\n\nFix the errors and try again.]'),
            ('user', 'try again'),
        ]
    )
