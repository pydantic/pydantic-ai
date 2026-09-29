from dataclasses import dataclass
from datetime import timezone
from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

from pydantic_ai import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    FilePart,
    LoadCapabilityCallPart,
    LoadCapabilityReturnPart,
    RetryPromptPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
)

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


@pytest.mark.parametrize('outcome', ['success', 'failed'])
def test_tool_history(outcome: Literal['success', 'failed']):
    """Inspect the actual sampling payload: mock responses cannot detect lost history."""
    history = [
        ModelRequest(parts=[UserPromptPart('Look up both cities')]),
        ModelResponse(
            parts=[
                ThinkingPart('Hidden'),
                TextPart('Checking.'),
                ToolCallPart('weather', '{"city":"London"}', tool_call_id='one'),
                ToolCallPart('weather', {'city': 'Paris'}, tool_call_id='two'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart('weather', {'temperature': 20}, tool_call_id='one', outcome=outcome),
                RetryPromptPart('Unavailable', tool_name='weather', tool_call_id='two'),
            ]
        ),
    ]
    create_message = AsyncMock(
        return_value=CreateMessageResult(role='assistant', content=TextContent(type='text', text='Done'), model='test')
    )
    result = Agent(MCPSamplingModel(fake_session(create_message))).run_sync('Thanks', message_history=history)
    assert result.output == 'Done'
    payload = [msg.model_dump(by_alias=True, exclude_none=True) for msg in create_message.call_args.args[0]]
    expected = [
        {'role': 'user', 'content': {'type': 'text', 'text': 'Look up both cities'}},
        {
            'role': 'assistant',
            'content': [
                {'type': 'text', 'text': 'Checking.'},
                {'type': 'tool_use', 'id': 'one', 'name': 'weather', 'input': {'city': 'London'}},
                {'type': 'tool_use', 'id': 'two', 'name': 'weather', 'input': {'city': 'Paris'}},
            ],
        },
        {
            'role': 'user',
            'content': [
                {
                    'type': 'tool_result',
                    'toolUseId': 'one',
                    'content': [{'type': 'text', 'text': '{"temperature":20}'}],
                    'isError': outcome == 'failed',
                },
                {
                    'type': 'tool_result',
                    'toolUseId': 'two',
                    'content': [{'type': 'text', 'text': 'Unavailable\n\nFix the errors and try again.'}],
                    'isError': True,
                },
            ],
        },
        {'role': 'user', 'content': {'type': 'text', 'text': 'Thanks'}},
    ]
    assert payload == expected


def test_output_retry_history():
    create_message = AsyncMock(
        return_value=CreateMessageResult(role='assistant', content=TextContent(type='text', text='Done'), model='test')
    )
    agent = Agent(MCPSamplingModel(fake_session(create_message)))
    agent.run_sync(
        message_history=[
            ModelRequest(parts=[UserPromptPart('Hello')]),
            ModelResponse(parts=[TextPart('One'), ThinkingPart('Hidden'), TextPart('Two')]),
            ModelRequest(parts=[RetryPromptPart('Try again')]),
        ]
    )
    assert [msg.model_dump(by_alias=True, exclude_none=True) for msg in create_message.call_args.args[0]] == snapshot(
        [
            {'role': 'user', 'content': {'type': 'text', 'text': 'Hello'}},
            {'role': 'assistant', 'content': {'type': 'text', 'text': 'OneTwo'}},
            {
                'role': 'user',
                'content': {'type': 'text', 'text': 'Validation feedback:\nTry again\n\nFix the errors and try again.'},
            },
        ]
    )


def test_framework_tool_history():
    """Locally executed framework tools share the ordinary function-tool wire format."""
    history = [
        ModelResponse(
            parts=[
                ToolSearchCallPart(args={'queries': ['weather']}, tool_call_id='search'),
                LoadCapabilityCallPart(args={'id': 'forecast'}, tool_call_id='load'),
            ]
        ),
        ModelRequest(
            parts=[
                ToolSearchReturnPart(content={'discovered_tools': [{'name': 'weather'}]}, tool_call_id='search'),
                LoadCapabilityReturnPart(content={'instructions': 'Use Celsius'}, tool_call_id='load'),
            ]
        ),
    ]
    create_message = AsyncMock(
        return_value=CreateMessageResult(role='assistant', content=TextContent(type='text', text='Done'), model='test')
    )
    Agent(MCPSamplingModel(fake_session(create_message))).run_sync('Continue', message_history=history)
    assert [msg.model_dump(by_alias=True, exclude_none=True) for msg in create_message.call_args.args[0]] == snapshot(
        [
            {
                'role': 'assistant',
                'content': [
                    {'type': 'tool_use', 'id': 'search', 'name': 'search_tools', 'input': {'queries': ['weather']}},
                    {'type': 'tool_use', 'id': 'load', 'name': 'load_capability', 'input': {'id': 'forecast'}},
                ],
            },
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'toolUseId': 'search',
                        'content': [{'type': 'text', 'text': '{"discovered_tools":[{"name":"weather"}]}'}],
                        'isError': False,
                    },
                    {
                        'type': 'tool_result',
                        'toolUseId': 'load',
                        'content': [{'type': 'text', 'text': '{"instructions":"Use Celsius"}'}],
                        'isError': False,
                    },
                ],
            },
            {'role': 'user', 'content': {'type': 'text', 'text': 'Continue'}},
        ]
    )


@pytest.mark.parametrize('file_in_result', [True, False])
def test_unsupported_tool_history(file_in_result: bool):
    file = BinaryContent(data=b'image', media_type='image/png')
    call = ToolCallPart('screenshot', {}, tool_call_id='one')
    response = ModelResponse(parts=[call] if file_in_result else [call, FilePart(file)])
    history = [
        ModelRequest(parts=[UserPromptPart('Show me')]),
        response,
        ModelRequest(parts=[ToolReturnPart('screenshot', file if file_in_result else 'Done', tool_call_id='one')]),
    ]
    agent = Agent(MCPSamplingModel(fake_session(AsyncMock())))
    if file_in_result:
        with pytest.raises(NotImplementedError, match='Multimodal tool results'):
            agent.run_sync('Continue', message_history=history)
    else:
        with pytest.raises(UnexpectedModelBehavior, match='Unexpected part type: FilePart'):
            agent.run_sync('Continue', message_history=history)
