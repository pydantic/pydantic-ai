import uuid

import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import (
    Agent,
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
from pydantic_ai.usage import RequestUsage

from .conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
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
            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    result = task['result']
                    break
                await anyio.sleep(0.1)

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
            max_attempts = 50  # 5 seconds total
            for _ in range(max_attempts):
                task = await a2a_client.get_task(task_id)
                if 'result' in task and task['result']['status']['state'] == 'failed':
                    break
                await anyio.sleep(0.1)
            else:  # pragma: no cover
                raise AssertionError(f'Task did not fail within {max_attempts * 0.1} seconds')

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

            while task1 := await a2a_client.get_task(task1_id):  # pragma: no branch
                if 'result' in task1 and task1['result']['status']['state'] == 'completed':
                    result1 = task1['result']
                    break
                await anyio.sleep(0.1)

            # Verify the model received at least one message
            assert len(messages_received) == 1
            first_run_history = messages_received[0]
            assert first_run_history == snapshot(
                [
                    ModelRequest(
                        parts=[UserPromptPart(content='First message', timestamp=IsDatetime())],
                        run_id=IsStr(),
                    )
                ]
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
                    ModelRequest(
                        parts=[UserPromptPart(content='First message', timestamp=IsDatetime())],
                        run_id=IsStr(),
                    ),
                    ModelResponse(
                        parts=[
                            ToolCallPart(
                                tool_name='final_result', args='{"response": ["foo", "bar"]}', tool_call_id=IsStr()
                            )
                        ],
                        usage=RequestUsage(input_tokens=52, output_tokens=7),
                        model_name='function:track_messages:',
                        timestamp=IsDatetime(),
                        run_id=IsStr(),
                    ),
                    ModelRequest(
                        parts=[
                            ToolReturnPart(
                                tool_name='final_result',
                                content='Final result processed.',
                                tool_call_id=IsStr(),
                                timestamp=IsDatetime(),
                            ),
                            UserPromptPart(content='Second message', timestamp=IsDatetime()),
                        ],
                        run_id=IsStr(),
                    ),
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
            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    result = task['result']
                    break
                await anyio.sleep(0.1)

            assert result == snapshot(
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

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    result = task['result']
                    break
                await anyio.sleep(0.1)

            assert result == snapshot(
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

            while task := await a2a_client.get_task(task_id):  # pragma: no branch
                if 'result' in task and task['result']['status']['state'] == 'completed':
                    result = task['result']
                    break
                await anyio.sleep(0.1)  # pragma: lax no cover

            assert result == snapshot(
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
