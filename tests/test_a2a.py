import anyio
import httpx
import pytest
from asgi_lifespan import LifespanManager
from inline_snapshot import snapshot

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
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
                                'metadata': {
                                    'type': 'tuple',
                                    'json_schema': {'items': {}, 'type': 'array'},
                                    'class_name': 'tuple',
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
                        file={
                            'uri': 'https://example.com/file.txt',
                            'mime_type': 'text/plain',
                        },
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
                                    'file': {
                                        'uri': 'https://example.com/file.txt',
                                        'mime_type': 'text/plain',
                                    },
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
                                        'file': {
                                            'uri': 'https://example.com/file.txt',
                                            'mime_type': 'text/plain',
                                        },
                                    }
                                ],
                                'kind': 'message',
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
                                'metadata': {
                                    'type': 'tuple',
                                    'json_schema': {'items': {}, 'type': 'array'},
                                    'class_name': 'tuple',
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
                    FilePart(kind='file', file={'data': 'foo', 'mime_type': 'text/plain'}),
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
                            'parts': [{'kind': 'file', 'file': {'data': 'foo', 'mime_type': 'text/plain'}}],
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
                                'parts': [{'kind': 'file', 'file': {'data': 'foo', 'mime_type': 'text/plain'}}],
                                'kind': 'message',
                                'context_id': IsStr(),
                                'task_id': IsStr(),
                            }
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
                                'metadata': {
                                    'type': 'tuple',
                                    'json_schema': {'items': {}, 'type': 'array'},
                                    'class_name': 'tuple',
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

            message = Message(role='user', parts=[DataPart(kind='data', data={'foo': 'bar'})], kind='message')
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
                if 'result' in task and task['result']['status']['state'] in ('failed', 'completed'):
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

            # NOTE: We include the agent history before we start working on the task.
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
                        ],
                        'artifacts': [
                            {
                                'artifact_id': IsStr(),
                                'name': 'result',
                                'parts': [{'kind': 'data', 'data': ['foo', 'bar']}],
                                'metadata': {
                                    'type': 'tuple',
                                    'json_schema': {'items': {}, 'type': 'array'},
                                    'class_name': 'tuple',
                                },
                            }
                        ],
                    },
                }
            )
