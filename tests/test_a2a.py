import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .conftest import IsDatetime, IsStr, try_import

with try_import() as imports_successful:
    from fasta2a.client import A2AClient
    from fasta2a.schema import DataPart, FilePart, Message, TextPart, is_task
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

            message = Message(role='user', parts=[TextPart(text='Get user profile', kind='text')], kind='message')
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert is_task(result)

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
            assert artifact['parts'][0]['data'] == {'name': 'John Doe', 'age': 30, 'email': 'john@example.com'}

            # Verify metadata
            assert 'metadata' in artifact
            metadata = artifact['metadata']
            assert metadata['type'] == 'UserProfile'

            # Verify JSON schema is present and correct
            assert 'json_schema' in metadata
            json_schema = metadata['json_schema']
            assert json_schema['type'] == 'object'
            assert 'properties' in json_schema
            assert set(json_schema['properties'].keys()) == {'name', 'age', 'email'}
            assert json_schema['properties']['name']['type'] == 'string'
            assert json_schema['properties']['age']['type'] == 'integer'
            assert json_schema['properties']['email']['type'] == 'string'
            assert json_schema['required'] == ['name', 'age', 'email']

            # Check the message history also has the data
            assert 'history' in result
            assert len(result['history']) == 2
            agent_message = result['history'][1]
            assert agent_message['role'] == 'agent'
            assert agent_message['parts'][0]['kind'] == 'data'
            assert agent_message['parts'][0]['data'] == {'name': 'John Doe', 'age': 30, 'email': 'john@example.com'}


async def test_a2a_runtime_error_without_lifespan():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    transport = httpx.ASGITransport(app)
    async with httpx.AsyncClient(transport=transport) as http_client:
        a2a_client = A2AClient(http_client=http_client)

        message = Message(role='user', parts=[TextPart(text='Hello, world!', kind='text')], kind='message')

        with pytest.raises(RuntimeError, match='TaskManager was not properly initialized.'):
            await a2a_client.send_message(message=message)


async def test_a2a_simple():
    agent = Agent(model=model, output_type=tuple[str, str])
    app = agent.to_a2a()

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(role='user', parts=[TextPart(text='Hello, world!', kind='text')], kind='message')
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert is_task(result)
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
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
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
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
                                'metadata': {
                                    'type': 'tuple',
                                    'json_schema': {'items': {}, 'type': 'array'},
                                },
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
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert is_task(result)
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
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
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
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
                                'metadata': {
                                    'type': 'tuple',
                                    'json_schema': {'items': {}, 'type': 'array'},
                                },
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
                    FilePart(file={'data': 'foo', 'mime_type': 'text/plain'}, kind='file'),
                ],
                kind='message',
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert is_task(result)
            assert result == snapshot(
                {
                    'id': IsStr(),
                    'context_id': IsStr(),
                    'kind': 'task',
                    'status': {'state': 'submitted', 'timestamp': IsDatetime(iso_string=True)},
                    'history': [
                        {
                            'role': 'user',
                            'parts': [{'kind': 'file', 'file': {'mime_type': 'text/plain', 'data': 'foo'}}],
                            'kind': 'message',
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
                                'parts': [{'kind': 'file', 'file': {'mime_type': 'text/plain', 'data': 'foo'}}],
                                'kind': 'message',
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {
                                'role': 'agent',
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
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
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
                                'metadata': {
                                    'type': 'tuple',
                                    'json_schema': {'items': {}, 'type': 'array'},
                                },
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
            )
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert is_task(result)
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

            message = Message(role='user', parts=[TextPart(text='Hello, world!', kind='text')], kind='message')
            response = await a2a_client.send_message(message=message)
            assert 'error' not in response
            assert 'result' in response
            result = response['result']
            assert is_task(result)

            task_id = result['id']

            # Wait for task to fail
            await anyio.sleep(0.1)
            task = await a2a_client.get_task(task_id)

            assert 'result' in task
            assert task['result']['status']['state'] == 'failed'


async def test_a2a_multiple_messages():
    agent = Agent(model=model, output_type=tuple[str, str])
    storage = InMemoryStorage()
    app = agent.to_a2a(storage=storage)

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app)
        async with httpx.AsyncClient(transport=transport) as http_client:
            a2a_client = A2AClient(http_client=http_client)

            message = Message(role='user', parts=[TextPart(text='Hello, world!', kind='text')], kind='message')
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
            assert is_task(result)
            task_id = result['id']
            task = storage.tasks[task_id]
            assert 'history' in task
            task['history'].append(
                Message(role='agent', parts=[TextPart(text='Whats up?', kind='text')], kind='message')
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
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {'role': 'agent', 'parts': [{'kind': 'text', 'text': 'Whats up?'}], 'kind': 'message'},
                        ],
                    },
                }
            )

            await anyio.sleep(0.1)
            task = await a2a_client.get_task(task_id)
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
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            },
                            {'role': 'agent', 'parts': [{'kind': 'text', 'text': 'Whats up?'}], 'kind': 'message'},
                            {
                                'role': 'agent',
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
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
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
                                'metadata': {
                                    'type': 'tuple',
                                    'json_schema': {'items': {}, 'type': 'array'},
                                },
                            }
                        ],
                    },
                }
            )
