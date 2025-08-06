import uuid
from typing import Any, cast

import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart as PydanticAITextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import Usage

from .conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from fasta2a.broker import StreamEvent
    from fasta2a.client import A2AClient
    from fasta2a.schema import DataPart, FilePart, Message, TextPart
    from fasta2a.storage import InMemoryStorage


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='fasta2a not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]


def return_string(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    assert info.output_tools is not None
    args_json = '{"response": ["foo", "bar"]}'
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])


model = FunctionModel(return_string)


# Define a test Pydantic model
class UserProfile(BaseModel):
    name: str
    age: int
    email: str


def return_pydantic_model(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    assert info.output_tools is not None
    args_json = '{"name": "John Doe", "age": 30, "email": "john@example.com"}'
    return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])


pydantic_model = FunctionModel(return_pydantic_model)


async def test_a2a_pydantic_model_output():
    """Test that Pydantic model outputs have correct metadata including JSON schema."""
    agent = Agent(model=pydantic_model, output_type=UserProfile)
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Get user profile', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'

            task_id = result['id']

            # Wait for completion
            await anyio.sleep(0.1)
            task = await a2a_client.get_task(task_id)

            assert 'result' in task
            result = task['result']
            assert result['status']['state'] == 'completed'

            # Check artifacts
            assert 'artifacts' in result
            assert len(result['artifacts']) == 1
            artifact = result['artifacts'][0]

            # Verify the data
            assert artifact['parts'][0]['kind'] == 'data'
            assert artifact['parts'][0]['data'] == {
                'result': {'name': 'John Doe', 'age': 30, 'email': 'john@example.com'}
            }

            metadata = artifact['parts'][0].get('metadata')
            assert metadata is not None

            assert metadata['json_schema'] == snapshot(
                {
                    'properties': {
                        'name': {'title': 'Name', 'type': 'string'},
                        'age': {'title': 'Age', 'type': 'integer'},
                        'email': {'title': 'Email', 'type': 'string'},
                    },
                    'required': ['name', 'age', 'email'],
                    'title': 'UserProfile',
                    'type': 'object',
                }
            )

            assert result.get('history') == snapshot(
                [
                    {
                        'role': 'user',
                        'parts': [{'kind': 'text', 'text': 'Get user profile'}],
                        'kind': 'message',
                        'message_id': IsStr(),
                        'context_id': IsStr(),
                        'task_id': IsStr(),
                    }
                ]
            )


async def test_a2a_runtime_error_without_lifespan():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    transport = httpx.ASGITransport(app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        a2a_client = A2AClient(http_client=http_client)

        message = Message(
            role='user',
            parts=[TextPart(text='Hello, world!', kind='text')],
            kind='message',
            message_id=str(uuid.uuid4()),
        )

        with pytest.raises(RuntimeError, match='TaskManager was not properly initialized.'):
            await a2a_client.send_message(message=message)


async def test_a2a_simple():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'kind': 'data',
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_a2a_file_message_with_file():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[
                    FilePart(
                        kind='file',
                        file={'uri': 'https://example.com/file.txt', 'mime_type': 'text/plain'},
                    )
                ],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [
                                {
                                    'kind': 'file',
                                    'file': {'mime_type': 'text/plain', 'uri': 'https://example.com/file.txt'},
                                }
                            ],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [
                                    {
                                        'kind': 'file',
                                        'file': {'mime_type': 'text/plain', 'uri': 'https://example.com/file.txt'},
                                    }
                                ],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'kind': 'data',
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_a2a_file_message_with_file_content():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[
                    FilePart(file={'bytes': 'foo', 'mime_type': 'text/plain'}, kind='file'),
                ],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'file', 'file': {'bytes': 'foo', 'mime_type': 'text/plain'}}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'file', 'file': {'bytes': 'foo', 'mime_type': 'text/plain'}}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'kind': 'data',
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_a2a_file_message_with_data():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[DataPart(kind='data', data={'foo': 'bar'})],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'data', 'data': {'foo': 'bar'}}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                }
            )

            task_id = result['id']

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'failed':
                    break
                await anyio.sleep(0.1)
            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'failed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'data', 'data': {'foo': 'bar'}}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )


async def test_a2a_error_handling():
    """Test that errors during task execution properly update task state."""

    def raise_error(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('Test error during agent execution')

    error_model = FunctionModel(raise_error)
    agent = Agent(model=error_model, output_type=str)
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'

            task_id = result['id']

            # Wait for task to fail
            await anyio.sleep(0.1)
            task = await a2a_client.get_task(task_id)

            assert 'result' in task
            assert task['result']['status']['state'] == 'failed'


async def test_a2a_multiple_tasks_same_context():
    """Test that multiple tasks can share the same context_id with accumulated history."""

    messages_received: list[list[ModelMessage]] = []

    def track_messages(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Store a copy of the messages received by the model
        messages_received.append(messages.copy())
        # Return the standard response
        assert info.output_tools is not None
        args_json = '{"response": ["foo", "bar"]}'
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args_json)])

    tracking_model = FunctionModel(track_messages)
    agent = Agent(model=tracking_model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            # First message - should create a new context
            message1 = Message(
                role='user',
                parts=[TextPart(text='First message', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response1 = await a2a_client.send_message(message=message1)
            assert 'error' not in response1
            assert 'result' in response1
            result1 = response1['result']
            assert result1['kind'] == 'task'

            task1_id = result1['id']
            context_id = result1['context_id']

            # Wait for first task to complete
            await anyio.sleep(0.1)
            task1 = await a2a_client.get_task(task1_id)
            assert 'result' in task1
            assert task1['result']['status']['state'] == 'completed'

            # Verify the model received at least one message
            assert len(messages_received) == 1
            first_run_history = messages_received[0]
            assert first_run_history == snapshot(
                [ModelRequest(parts=[UserPromptPart(content='First message', timestamp=IsDatetime())])]
            )

            # Second message - reuse the same context_id
            message2 = Message(
                role='user',
                parts=[TextPart(text='Second message', kind='text')],
                kind='message',
                context_id=context_id,
                message_id=str(uuid.uuid4()),
            )
            response2 = await a2a_client.send_message(message=message2)
            assert 'error' not in response2
            assert 'result' in response2
            result2 = response2['result']
            assert result2['kind'] == 'task'

            task2_id = result2['id']
            # Verify we got a new task ID but same context ID
            assert task2_id != task1_id
            assert result2['context_id'] == context_id

            # Wait for second task to complete
            while task2 := await a2a_client.get_task(task2_id):  # pragma: no branch
                if 'result' in task2 and task2['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)

            # Verify the model received the full history on the second call
            assert len(messages_received) == 2
            second_run_history = messages_received[1]
            assert second_run_history[0] == first_run_history[0]

            assert second_run_history == snapshot(
                [
                    ModelRequest(parts=[UserPromptPart(content='First message', timestamp=IsDatetime())]),
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name='final_result', args='{"response": ["foo", "bar"]}', tool_call_id=IsStr()
                            )
                        ],
                        usage=Usage(requests=1, request_tokens=52, response_tokens=7, total_tokens=59),
                        model_name='function:track_messages:',
                        timestamp=IsDatetime(),
                    ),
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name='final_result',
                                content='Final result processed.',
                                tool_call_id=IsStr(),
                                timestamp=IsDatetime(),
                            )
                        ]
                    ),
                    ModelRequest(parts=[UserPromptPart(content='Second message', timestamp=IsDatetime())]),
                ]
            )


async def test_a2a_thinking_response():
    """Test that ModelResponse messages with ThinkingPart are properly handled."""

    def return_thinking_response(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        assert info.output_tools is not None
        # Create a response with thinking part and text part
        return ModelResponse(
            parts=[
                ThinkingPart(content='Let me think about this...', id='thinking_1'),
                PydanticAITextPart(content="Here's my response"),
            ]
        )

    thinking_model = FunctionModel(return_thinking_response)
    agent = Agent(model=thinking_model, output_type=str)
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'

            task_id = result['id']

            # Wait for completion
            await anyio.sleep(0.1)
            task = await a2a_client.get_task(task_id)

            assert 'result' in task
            assert task['result'] == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        },
                        {
                            'role': 'agent',
                            'parts': [
                                {
                                    'metadata': {'type': 'thinking', 'thinking_id': 'thinking_1', 'signature': None},
                                    'kind': 'text',
                                    'text': 'Let me think about this...',
                                },
                                {'kind': 'text', 'text': "Here's my response"},
                            ],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        },
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [{'kind': 'text', 'text': "Here's my response"}],
                        }
                    ],
                }
            )


async def test_a2a_multiple_messages():
    agent = Agent(model=model, output_type=tuple[str, str])
    storage = InMemoryStorage()
    app = agent.to_a2a(storage=storage)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )

            # NOTE: We include the agent history before we start working on the task.
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']
            task = storage.tasks[task_id]
            assert 'history' in task
            task['history'].append(
                Message(
                    role='agent',
                    parts=[TextPart(text='Whats up?', kind='text')],
                    kind='message',
                    message_id=str(uuid.uuid4()),
                )
            )

            response = await a2a_client.get_task(task_id)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'kind': 'text', 'text': 'Whats up?'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                            },
                        ],
                    },
                }
            )

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    break
                await anyio.sleep(0.1)

            assert task == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': None,
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'kind': 'text', 'text': 'Whats up?'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                            },
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [
                                    {
                                        'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                        'kind': 'data',
                                        'data': {'result': ['foo', 'bar']},
                                    }
                                ],
                            }
                        ],
                    },
                }
            )


async def test_a2a_multiple_send_task_messages():
    agent = Agent(model=model, output_type=tuple[str, str])
    storage = InMemoryStorage()
    app = agent.to_a2a(storage=storage)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )
            assert 'result' in response
            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']
            context_id = result['context_id']

            await anyio.sleep(0.1)
            response = await a2a_client.get_task(task_id)
            assert response.get('result') == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [
                                {
                                    'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                    'kind': 'data',
                                    'data': {'result': ['foo', 'bar']},
                                }
                            ],
                        }
                    ],
                }
            )

            message = Message(
                role='user',
                parts=[TextPart(text='Did you get my first message?', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
                context_id=context_id,
            )
            response = await a2a_client.send_message(message=message)
            assert response == snapshot(
                {
                    'jsonrpc': '2.0',
                    'id': IsStr(),
                    'result': {
                        'id': IsStr(),
                        'context_id': IsStr(),
                        'kind': 'task',
                        'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                        'history': [
                            {
                                'role': 'user',
                                'parts': [{'kind': 'text', 'text': 'Did you get my first message?'}],
                                'kind': 'message',
                                'message_id': IsStr(),
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                    },
                }
            )

            await anyio.sleep(0.1)
            response = await a2a_client.get_task(task_id)
            assert response.get('result') == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'completed', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'text', 'text': 'Hello, world!'}],
                            'kind': 'message',
                            'message_id': IsStr(),
                            'context_id': IsStr(),
                            'task_id': IsStr(),
                        }
                    ],
                    'artifacts': [
                        {
                            'artifact_id': IsStr(),
                            'name': 'result',
                            'parts': [
                                {
                                    'metadata': {'json_schema': {'items': {}, 'type': 'array'}},
                                    'kind': 'data',
                                    'data': {'result': ['foo', 'bar']},
                                }
                            ],
                        }
                    ],
                }
            )


async def test_streaming_emits_incremental_messages(mocker: Any) -> None:
    """Verify that enable_streaming=True produces incremental messages during agent execution."""
    from fasta2a.broker import InMemoryBroker

    # Create a model that produces multiple text parts to simulate streaming
    def return_multiple_text_parts(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                PydanticAITextPart(content='First part of response'),
                PydanticAITextPart(content='Second part of response'),
                PydanticAITextPart(content='Final part'),
            ]
        )

    streaming_model = FunctionModel(return_multiple_text_parts)

    # Create agent with streaming enabled
    agent = Agent(model=streaming_model, output_type=str)
    storage = InMemoryStorage()
    broker = InMemoryBroker()

    # Spy on the broker's send_stream_event method to capture calls
    mock_send: Any = mocker.spy(broker, 'send_stream_event')

    app = agent.to_a2a(enable_streaming=True, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response

            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task completion
            await anyio.sleep(0.1)
            final_response = await a2a_client.get_task(task_id)
            assert 'result' in final_response
            final_result = final_response['result']
            assert final_result['status']['state'] == 'completed'

            # Verify streaming events were captured
            assert mock_send.call_count > 0

            # Extract events from mock calls
            captured_events: list[StreamEvent] = [call.args[1] for call in mock_send.call_args_list]

            # Check that we got different types of events
            event_kinds = [event.get('kind') for event in captured_events if event.get('kind')]

            # Look for agent messages
            message_events: list[Message] = []
            for event in captured_events:
                if event.get('kind') == 'message' and event.get('role') == 'agent':
                    message_events.append(cast(Message, event))

            # Should have status updates at minimum
            assert 'status-update' in event_kinds

            # Verify we got at least one agent message during streaming
            assert len(message_events) > 0, f'Expected agent messages during streaming, got events: {event_kinds}'

            # Verify the agent message contains the expected content
            agent_message = message_events[0]
            assert agent_message['role'] == 'agent'
            assert agent_message['kind'] == 'message'
            assert 'message_id' in agent_message
            assert 'parts' in agent_message
            parts = agent_message['parts']
            assert len(parts) == 3  # Should have 3 text parts
            first_part = parts[0]
            assert first_part.get('kind') == 'text'
            assert first_part.get('text') == 'First part of response'


async def test_streaming_disabled_sends_only_final_results(mocker: Any) -> None:
    """Verify enable_streaming=False sends only status updates and final results, no incremental messages."""
    from fasta2a.broker import InMemoryBroker

    # Create a model that produces multiple text parts - same as streaming test for comparison
    def return_multiple_text_parts(_: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                PydanticAITextPart(content='First part of response'),
                PydanticAITextPart(content='Second part of response'),
                PydanticAITextPart(content='Final part'),
            ]
        )

    streaming_model = FunctionModel(return_multiple_text_parts)

    # Create agent with streaming DISABLED (default behavior)
    agent = Agent(model=streaming_model, output_type=str)
    storage = InMemoryStorage()
    broker = InMemoryBroker()

    # Spy on the broker's send_stream_event method to capture calls
    mock_send: Any = mocker.spy(broker, 'send_stream_event')

    # Note: enable_streaming defaults to False, but being explicit for clarity
    app = agent.to_a2a(enable_streaming=False, storage=storage, broker=broker)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(
                role='user',
                parts=[TextPart(text='Hello, world!', kind='text')],
                kind='message',
                message_id=str(uuid.uuid4()),
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response

            result = response['result']
            assert result['kind'] == 'task'
            task_id = result['id']

            # Wait for task completion
            await anyio.sleep(0.1)
            final_response = await a2a_client.get_task(task_id)
            assert 'result' in final_response
            final_result = final_response['result']
            assert final_result['status']['state'] == 'completed'

            # Verify streaming events were captured
            assert mock_send.call_count > 0

            # Extract events from mock calls
            captured_events: list[StreamEvent] = [call.args[1] for call in mock_send.call_args_list]

            # Analyze event types
            event_kinds = [event.get('kind') for event in captured_events if event.get('kind')]

            # Look for any agent messages (should be NONE when streaming disabled)
            agent_messages: list[Message] = []
            for event in captured_events:
                if event.get('kind') == 'message' and event.get('role') == 'agent':
                    agent_messages.append(cast(Message, event))

            # Verify expected behavior: only status updates, NO incremental agent messages
            assert 'status-update' in event_kinds, 'Should have status updates'
            assert len(agent_messages) == 0, (
                f'Should have NO agent messages when streaming disabled, but got: {agent_messages}'
            )

            # Verify clean event stream: only status updates (working -> completed)
            status_events = [event for event in captured_events if event.get('kind') == 'status-update']
            assert len(status_events) >= 2, 'Should have at least working and completed status updates'

            # Verify final result is complete and correct
            assert 'artifacts' in final_result
            artifacts = final_result['artifacts']
            assert len(artifacts) == 1
            artifact = cast(dict[str, Any], artifacts[0])
            assert artifact['name'] == 'result'
            assert len(artifact['parts']) == 1
            artifact_part = cast(dict[str, Any], artifact['parts'][0])
            assert artifact_part['kind'] == 'text'
            # Final result should contain all text parts concatenated
            assert 'First part of response' in artifact_part['text']
