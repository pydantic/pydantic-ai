"""Resuming batched agent runs from stored message histories, against an in-process fake of OpenAI's Batch API.

Unlike the in-memory batches of the test models, these batches live outside any model instance, as a provider's do,
so a history can be stored as JSON and resumed in what amounts to a new process: a fresh model, client and
`BatchModelRequests`, even with a different API key.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pytest

from pydantic_ai import Agent, ModelMessagesTypeAdapter, PendingModelResponse
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, SuspendedResponseExpired
from pydantic_ai.messages import ModelResponse

from ..batch_utils import FAST_BATCH_SETTINGS, FAST_POLLING, UNRUN_MESSAGES
from ..conftest import try_import

with try_import() as imports_successful:
    import httpx2
    from openai import AsyncOpenAI

    from pydantic_ai.models.openai import OpenAIResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.usefixtures('allow_model_requests'),
]


def _response_body(text: str) -> dict[str, Any]:
    return {
        'id': f'resp_{text}',
        'object': 'response',
        'created_at': 1_767_225_600,
        'model': 'gpt-4.1-nano-2025-04-14',
        'status': 'completed',
        'output': [
            {
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'status': 'completed',
                'content': [{'type': 'output_text', 'text': text, 'annotations': []}],
            }
        ],
        'parallel_tool_calls': True,
        'tool_choice': 'auto',
        'tools': [],
        'usage': {
            'input_tokens': 10,
            'input_tokens_details': {'cached_tokens': 0},
            'output_tokens': 2,
            'output_tokens_details': {'reasoning_tokens': 0},
            'total_tokens': 12,
        },
    }


@dataclass
class _BatchAPI:
    """The Files and Batches endpoints, holding batches until the test ends them."""

    files: dict[str, str] = field(default_factory=dict[str, str])
    batches: dict[str, dict[str, Any]] = field(default_factory=dict[str, dict[str, Any]])
    calls: list[tuple[str, str, str]] = field(default_factory=list[tuple[str, str, str]])
    """Each request's method, path and API key."""
    forgotten: set[str] = field(default_factory=set[str])
    """Batches past their retention, which the API no longer has."""
    denied_keys: set[str] = field(default_factory=set[str])
    """API keys of another project, which can't see this one's batches or files."""

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        path = request.url.path.removeprefix('/v1')
        api_key = request.headers['authorization'].removeprefix('Bearer ')
        self.calls.append((request.method, path, api_key))
        if api_key in self.denied_keys:
            return httpx2.Response(404, json={'error': {'message': 'Not found', 'type': 'invalid_request_error'}})
        if request.method == 'POST' and path == '/files':
            file_id = f'file-{len(self.files)}'
            # The JSONL file is the last part of the multipart upload.
            self.files[file_id] = request.read().decode().split('\r\n\r\n')[-1].rsplit('\r\n--', 1)[0]
            return httpx2.Response(
                200,
                json={
                    'id': file_id,
                    'object': 'file',
                    'bytes': 1,
                    'created_at': 0,
                    'filename': 'batch.jsonl',
                    'purpose': 'batch',
                },
            )
        if request.method == 'POST' and path == '/batches':
            batch_id = f'batch_{len(self.batches)}'
            self.batches[batch_id] = {
                'id': batch_id,
                'object': 'batch',
                'endpoint': '/v1/responses',
                'completion_window': '24h',
                'input_file_id': json.loads(request.read())['input_file_id'],
                'created_at': 1_767_225_600,
                'status': 'in_progress',
            }
            return httpx2.Response(200, json=self.batches[batch_id])
        if request.method == 'DELETE':
            # Kept, so a test can still read the requests a batch held.
            return httpx2.Response(200, json={'id': path.split('/')[2], 'object': 'file', 'deleted': True})
        if path.startswith('/files/'):
            return httpx2.Response(200, content=self.files[path.split('/')[2]])
        batch_id = path.split('/')[2]
        if batch_id in self.forgotten:
            return httpx2.Response(404, json={'error': {'message': 'No batch found', 'type': 'invalid_request_error'}})
        return httpx2.Response(200, json=self.batches[batch_id])

    def custom_ids(self, batch_id: str) -> list[str]:
        lines = self.files[self.batches[batch_id]['input_file_id']].splitlines()
        return [json.loads(line)['custom_id'] for line in lines]

    def complete(self, batch_id: str, text: str) -> None:
        """End the batch with a response with `text` to each of its requests."""
        output_id = f'file-output-{batch_id}'
        self.files[output_id] = '\n'.join(
            json.dumps({'custom_id': custom_id, 'response': {'status_code': 200, 'body': _response_body(text)}})
            for custom_id in self.custom_ids(batch_id)
        )
        self.batches[batch_id] |= {'status': 'completed', 'output_file_id': output_id}

    def fail(self, batch_id: str, status_code: int, error: dict[str, Any]) -> None:
        """End the batch with every request failing with an HTTP error."""
        error_id = f'file-errors-{batch_id}'
        self.files[error_id] = '\n'.join(
            json.dumps({'custom_id': custom_id, 'response': {'status_code': status_code, 'body': error}})
            for custom_id in self.custom_ids(batch_id)
        )
        self.batches[batch_id] |= {'status': 'completed', 'error_file_id': error_id}

    def cancel_partway(self, batch_id: str, text: str) -> None:
        """End the batch as cancelled after it ran its first request, with `text` as its response."""
        output_id = f'file-output-{batch_id}'
        custom_id = self.custom_ids(batch_id)[0]
        self.files[output_id] = json.dumps(
            {'custom_id': custom_id, 'response': {'status_code': 200, 'body': _response_body(text)}}
        )
        self.batches[batch_id] |= {'status': 'cancelled', 'output_file_id': output_id}

    def expire(self, batch_id: str) -> None:
        """End the batch after its completion window, without running its requests."""
        self.batches[batch_id] |= {'status': 'expired'}


def _agent(
    api: _BatchAPI,
    api_key: str = 'submitting-key',
    batching: BatchModelRequests | None = None,
    *,
    in_header: bool = False,
) -> Agent[None, str | PendingModelResponse]:
    """A new agent with a new client, model and capability, as a new process would build.

    With `in_header`, the client's API key is a placeholder that its `Authorization` default header overrides.
    """
    http_client = httpx2.AsyncClient(transport=httpx2.MockTransport(api.handle))
    if in_header:
        client = AsyncOpenAI(
            api_key='placeholder', default_headers={'Authorization': f'Bearer {api_key}'}, http_client=http_client
        )
    else:
        client = AsyncOpenAI(api_key=api_key, http_client=http_client)
    model = OpenAIResponsesModel('gpt-4.1-nano', provider=OpenAIProvider(openai_client=client))
    return Agent(
        model,
        output_type=[str, PendingModelResponse],
        capabilities=[batching or BatchModelRequests(**FAST_BATCH_SETTINGS)],
    )


async def _resume(
    api: _BatchAPI,
    stored: bytes,
    api_key: str = 'submitting-key',
    batching: BatchModelRequests | None = None,
    *,
    in_header: bool = False,
) -> tuple[str | PendingModelResponse, bytes]:
    result = await _agent(api, api_key, batching, in_header=in_header).run(
        message_history=ModelMessagesTypeAdapter.validate_json(stored)
    )
    return result.output, result.all_messages_json()


async def test_a_stored_history_resumes_in_a_new_process_with_a_rotated_api_key() -> None:
    """A history stored as JSON resumes with a new model, client and capability, whose API key isn't the one that submitted."""
    api = _BatchAPI()
    submitted = await _agent(api).run('Hello')
    assert isinstance(submitted.output, PendingModelResponse)
    stored = submitted.all_messages_json()

    output, stored = await _resume(api, stored, api_key='rotated-key')
    assert isinstance(output, PendingModelResponse)

    api.complete('batch_0', 'Hi!')
    output, stored = await _resume(api, stored, api_key='rotated-key')

    assert output == 'Hi!'
    assert list(api.batches) == ['batch_0']
    assert [(method, path, key) for method, path, key in api.calls if not path.startswith('/files')] == [
        ('POST', '/batches', 'submitting-key'),
        ('GET', '/batches/batch_0', 'rotated-key'),
        ('GET', '/batches/batch_0', 'rotated-key'),
    ]
    messages = ModelMessagesTypeAdapter.validate_json(stored)
    assert [message.state for message in messages if isinstance(message, ModelResponse)] == ['complete']
    response = messages[-1]
    assert isinstance(response, ModelResponse)
    assert response.provider_details is not None
    assert response.provider_details['batch'] == {'id': 'batch_0', 'custom_id': api.custom_ids('batch_0')[0]}


async def test_a_resumed_history_whose_batch_expired_submits_the_request_again() -> None:
    """Reading the same expired result again would fail every time, so the resumed run submits the request again."""
    api = _BatchAPI()
    stored = (await _agent(api).run('Hello')).all_messages_json()

    api.expire('batch_0')
    output, stored = await _resume(api, stored)
    assert isinstance(output, PendingModelResponse)
    assert list(api.batches) == ['batch_0', 'batch_1']

    api.complete('batch_1', 'Hi!')
    output, stored = await _resume(api, stored)
    assert output == 'Hi!'
    response = ModelMessagesTypeAdapter.validate_json(stored)[-1]
    assert isinstance(response, ModelResponse) and response.provider_details is not None
    assert response.provider_details['batch'] == {
        'id': 'batch_1',
        'custom_id': api.custom_ids('batch_1')[0],
        'resubmissions': 1,
    }
    # The request was submitted again as it was first sent.
    first, second = (json.loads(api.files[api.batches[batch]['input_file_id']]) for batch in ('batch_0', 'batch_1'))
    assert first == second


@pytest.mark.parametrize('status_code', [429, 500, 503])
async def test_a_transient_error_is_submitted_again_until_resubmissions_run_out(status_code: int) -> None:
    api = _BatchAPI()
    stored = (
        await _agent(api, batching=BatchModelRequests(**FAST_BATCH_SETTINGS, max_resubmissions=1)).run('Hello')
    ).all_messages_json()
    error = {'error': {'message': 'Try again later', 'type': 'server_error'}}

    api.fail('batch_0', status_code, error)
    output, stored = await _resume(api, stored, batching=BatchModelRequests(**FAST_BATCH_SETTINGS, max_resubmissions=1))
    assert isinstance(output, PendingModelResponse)

    # The resubmission count is stored in the history, so a new process doesn't start counting again.
    api.fail('batch_1', status_code, error)
    with pytest.raises(ModelHTTPError) as exc_info:
        await _resume(api, stored, batching=BatchModelRequests(**FAST_BATCH_SETTINGS, max_resubmissions=1))
    assert (exc_info.value.status_code, exc_info.value.body) == (status_code, error['error'])
    assert list(api.batches) == ['batch_0', 'batch_1']


async def test_an_invalid_request_is_not_submitted_again() -> None:
    api = _BatchAPI()
    stored = (await _agent(api).run('Hello')).all_messages_json()
    error = {'error': {'message': 'Invalid input', 'type': 'invalid_request_error'}}

    api.fail('batch_0', 400, error)
    with pytest.raises(ModelHTTPError) as exc_info:
        await _resume(api, stored)
    assert exc_info.value.status_code == 400
    assert list(api.batches) == ['batch_0']


async def test_a_batch_past_its_retention_is_expired() -> None:
    """A batch the provider no longer has raises `SuspendedResponseExpired`, as an expired background response does."""
    api = _BatchAPI()
    stored = (await _agent(api).run('Hello')).all_messages_json()

    api.forgotten.add('batch_0')
    with pytest.raises(SuspendedResponseExpired, match="Batch 'batch_0' could not be checked because 'openai'"):
        await _resume(api, stored)
    assert list(api.batches) == ['batch_0']


async def test_resuming_many_histories_checks_each_batch_once() -> None:
    """Histories resumed one after another in one process share a check of their batch per `poll_interval`."""
    api = _BatchAPI()
    submitting = _agent(api, batching=BatchModelRequests(**FAST_POLLING, max_requests=3, max_wait=None))
    results = await asyncio.gather(*(submitting.run(f'Question {i}') for i in range(6)))
    histories = [result.all_messages_json() for result in results]
    assert list(api.batches) == ['batch_0', 'batch_1']

    hourly = BatchModelRequests(poll_interval=timedelta(hours=1), max_poll_interval=timedelta(hours=1))
    collecting = _agent(api, batching=hourly)
    api.calls.clear()
    for stored in histories:
        output = (await collecting.run(message_history=ModelMessagesTypeAdapter.validate_json(stored))).output
        assert isinstance(output, PendingModelResponse)
    assert [path for _, path, _ in api.calls] == ['/batches/batch_0', '/batches/batch_1']

    # Once the batch has ended, its results are downloaded once and shared too.
    api.complete('batch_0', 'Hi!')
    api.complete('batch_1', 'Hi!')
    api.calls.clear()
    collecting = _agent(api)
    outputs = [
        (await collecting.run(message_history=ModelMessagesTypeAdapter.validate_json(stored))).output
        for stored in histories
    ]
    assert outputs == ['Hi!'] * 6
    assert [(method, path) for method, path, _ in api.calls if method == 'DELETE'] == [
        ('DELETE', '/files/file-0'),
        ('DELETE', '/files/file-1'),
    ]
    assert [path for method, path, _ in api.calls if method == 'GET'] == [
        '/batches/batch_0',
        '/files/file-output-batch_0/content',
        '/batches/batch_1',
        '/files/file-output-batch_1/content',
    ]


async def _two_histories(api: _BatchAPI) -> tuple[bytes, bytes]:
    submitting = _agent(api, batching=BatchModelRequests(**FAST_POLLING, max_requests=2, max_wait=None))
    first, second = await asyncio.gather(submitting.run('Question 1'), submitting.run('Question 2'))
    return first.all_messages_json(), second.all_messages_json()


@pytest.mark.parametrize('in_header', [False, True], ids=['api-key', 'authorization-header'])
async def test_a_credential_the_provider_denies_doesnt_get_cached_results(in_header: bool) -> None:
    """Results fetched with one credential aren't handed to a request checking the batch with another."""
    api = _BatchAPI(denied_keys={'other-project-key'})
    first, second = await _two_histories(api)
    api.complete('batch_0', 'Hi!')
    shared = BatchModelRequests(**FAST_BATCH_SETTINGS)

    assert (await _resume(api, first, batching=shared, in_header=in_header))[0] == 'Hi!'
    with pytest.raises(SuspendedResponseExpired):
        await _resume(api, second, api_key='other-project-key', batching=shared, in_header=in_header)
    assert ('GET', '/batches/batch_0', 'other-project-key') in api.calls


@pytest.mark.parametrize('in_header', [False, True], ids=['api-key', 'authorization-header'])
async def test_a_credential_the_provider_denies_doesnt_get_a_cached_status(in_header: bool) -> None:
    api = _BatchAPI(denied_keys={'other-project-key'})
    first, second = await _two_histories(api)
    shared = BatchModelRequests(poll_interval=timedelta(hours=1), max_poll_interval=timedelta(hours=1))

    assert isinstance((await _resume(api, first, batching=shared, in_header=in_header))[0], PendingModelResponse)
    with pytest.raises(SuspendedResponseExpired):
        await _resume(api, second, api_key='other-project-key', batching=shared, in_header=in_header)
    # Another history checked with the first credential still shares its status.
    assert isinstance((await _resume(api, second, batching=shared, in_header=in_header))[0], PendingModelResponse)
    assert [key for _, path, key in api.calls if path == '/batches/batch_0'] == [
        'submitting-key',
        'other-project-key',
    ]


@pytest.mark.parametrize('in_header', [False, True], ids=['api-key', 'authorization-header'])
async def test_a_credential_the_provider_denies_doesnt_join_a_concurrent_check(in_header: bool) -> None:
    api = _BatchAPI(denied_keys={'other-project-key'})
    first, second = await _two_histories(api)
    api.complete('batch_0', 'Hi!')
    shared = BatchModelRequests(**FAST_BATCH_SETTINGS)

    allowed, denied = await asyncio.gather(
        _resume(api, first, batching=shared, in_header=in_header),
        _resume(api, second, api_key='other-project-key', batching=shared, in_header=in_header),
        return_exceptions=True,
    )
    assert isinstance(allowed, tuple) and allowed[0] == 'Hi!'
    assert isinstance(denied, SuspendedResponseExpired)


async def test_a_batch_cancelled_partway_fails_the_requests_it_didnt_run() -> None:
    """A batch someone cancelled delivers what it ran; the rest raise, saying so, and aren't submitted again."""
    api = _BatchAPI()
    first, second = await _two_histories(api)
    api.cancel_partway('batch_0', 'Hi!')
    ran, unran = sorted((first, second), key=lambda stored: api.custom_ids('batch_0')[0] not in stored.decode())

    assert (await _resume(api, ran))[0] == 'Hi!'
    with pytest.raises(ModelAPIError, match=UNRUN_MESSAGES['canceled']):
        await _resume(api, unran)
    assert list(api.batches) == ['batch_0']
