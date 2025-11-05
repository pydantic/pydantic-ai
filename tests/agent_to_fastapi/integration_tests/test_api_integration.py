import contextlib
import time

import pytest
from aioresponses import aioresponses

from ...conftest import try_import

with try_import() as imports_successful:
    from fastapi import FastAPI
    from fastapi.routing import APIRoute
    from httpx import ASGITransport, AsyncClient

    from pydantic_ai.fastapi.agent_router import create_agent_router
    from pydantic_ai.fastapi.registry import AgentRegistry


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='OpenAI client not installed or FastAPI not installed'),
    pytest.mark.anyio,
]


@pytest.mark.asyncio
async def test_models_list_and_get(
    registry_with_openai_clients: AgentRegistry,
) -> None:
    """Verify model listing and retrieval endpoints behave as expected when real Agents are registered."""
    registry = registry_with_openai_clients

    router = create_agent_router(agent_registry=registry)

    app = FastAPI()
    app.include_router(router)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url='http://testserver') as client:
        resp = await client.get('/v1/models')
        assert resp.status_code == 200

        body = resp.json()
        assert 'data' in body

        ids = [m['id'] for m in body['data']]
        assert set(ids) == {
            'test-model',
            'test-model-only-completions',
            'test-model-only-responses',
        }

        resp2 = await client.get('/v1/models/test-model')
        assert resp2.status_code == 200

        model_body = resp2.json()
        assert model_body['id'] == 'test-model'

        resp3 = await client.get('/v1/models/nonexistent-model')
        assert resp3.status_code == 404

        detail = resp3.json().get('detail')
        assert isinstance(detail, dict)
        assert 'error' in detail
        assert detail['error']['type'] == 'not_found_error'


@pytest.mark.asyncio
async def test_routers_disabled(
    registry_with_openai_clients: AgentRegistry,
) -> None:
    """Verify whether disabling apis actually effectively not adds APIRoutes to the app."""
    registry = registry_with_openai_clients

    router = create_agent_router(agent_registry=registry, disable_completions_api=True, disable_responses_api=True)

    app = FastAPI()
    app.include_router(router)

    transport = ASGITransport(app=app)

    api_routes: list[APIRoute] = list(filter(lambda x: isinstance(x, APIRoute), app.routes))  # type: ignore
    assert {item.path for item in api_routes} == {'/v1/models', '/v1/models/{model_id}'}

    async with AsyncClient(transport=transport, base_url='http://testserver') as client:
        payload = {
            'model': 'test-model',
            'messages': [{'role': 'user', 'content': 'hello'}],
        }

        response = await client.post('/v1/chat/completions', json=payload)
        assert response.is_error
        assert response.status_code == 404

        response = await client.post('/v1/responses', json=payload)
        assert response.is_error
        assert response.status_code == 404


@pytest.mark.asyncio
async def test_route_not_implemented(registry_with_openai_clients: AgentRegistry) -> None:
    """Isolated test to assert registry raises KeyError for models that only implement the other route."""
    registry = registry_with_openai_clients

    with pytest.raises(KeyError) as excinfo:
        registry.get_completions_agent('test-model-only-responses')
    assert excinfo.value.args == (
        'Completions agent with %s has not been registered.',
        'test-model-only-responses',
    )

    with pytest.raises(KeyError) as excinfo:
        registry.get_responses_agent('test-model-only-completions')
    assert excinfo.value.args == (
        'Responses agent with %s has not been registered.',
        'test-model-only-completions',
    )


@pytest.mark.asyncio
async def test_chat_completions_e2e_with_mocked_openai(
    registry_with_openai_clients: AgentRegistry,
    allow_model_requests: None,
) -> None:
    """End-to-end-ish test for /v1/chat/completions:
    - Registers a real pydantic-ai Agent backed by AsyncOpenAI pointed at a fake base URL.
    - Uses aioresponses to intercept the outbound HTTP POST to the OpenAI chat completions endpoint.
    - Asserts that the final FastAPI response is the expected OpenAI-style chat completion.
    """
    fake_openai_base = 'https://api.openai.test/v1'
    registry = registry_with_openai_clients

    router = create_agent_router(agent_registry=registry)

    app = FastAPI()
    app.include_router(router)
    transport = ASGITransport(app=app)

    fake_openai_resp = {
        'id': 'chatcmpl-test',
        'object': 'chat.completion',
        'created': int(time.time()),
        'model': 'test-model',
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'hello from mocked openai'},
            },
        ],
        'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
    }

    async with AsyncClient(transport=transport, base_url='http://testserver') as client:
        # Intercept outbound AsyncOpenAI request and return our canned response
        with aioresponses() as mocked:
            mocked.post(  # type: ignore
                fake_openai_base + '/chat/completions',
                payload=fake_openai_resp,
                status=200,
            )

            payload = {
                'model': 'test-model',
                'messages': [{'role': 'user', 'content': 'hello'}],
            }

            r2 = await client.post('/v1/chat/completions', json=payload)
            assert r2.status_code == 200

            body = r2.json()
            assert body['model'] == 'test-model'
            assert body['choices'][0]['message']['content'] == 'hello from mocked openai'

        # Intercept outbound AsyncOpenAI request and return our canned response
        with aioresponses() as mocked:
            mocked.post(  # type: ignore
                fake_openai_base + '/chat/completions',
                payload=fake_openai_resp,
                status=200,
            )
            payload_missing = {
                'model': 'test-model-only-responses',
                'messages': [{'role': 'user', 'content': 'hello'}],
            }

            r_missing = await client.post('/v1/chat/completions', json=payload_missing)
            assert r_missing.status_code == 404


@pytest.mark.asyncio
async def test_responses_e2e_with_mocked_openai(
    registry_with_openai_clients: AgentRegistry,
    allow_model_requests: None,
) -> None:
    """End-to-end-ish test for /v1/responses:
    - Registers a real pydantic-ai Agent backed by AsyncOpenAI pointed at a fake base URL.
    - Uses aioresponses to intercept the outbound HTTP POST to the OpenAI Responses endpoint.
    - Asserts that the final FastAPI response contains the expected output text.
    """
    fake_openai_base = 'https://api.openai.test/v1'
    registry = registry_with_openai_clients

    router = create_agent_router(agent_registry=registry)

    # Disable response_model on the /v1/responses route so tests can return simple dicts if needed
    for route in list(getattr(router, 'routes', [])):
        if getattr(route, 'path', None) == '/v1/responses':
            with contextlib.suppress(Exception):
                route.response_model = None

    app = FastAPI()
    app.include_router(router)
    transport = ASGITransport(app=app)

    fake_openai_resp = {
        'id': 'resp-test',
        'object': 'response',
        'model': 'test-model',
        'created_at': int(time.time()),
        'output': [
            {
                'id': 'msg-1',
                'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'response from mocked openai'}],
            },
        ],
    }

    async with AsyncClient(transport=transport, base_url='http://testserver') as client:
        with aioresponses() as mocked:
            mocked.post(fake_openai_base + '/responses', payload=fake_openai_resp, status=200)  # type: ignore

            payload_missing = {'model': 'test-model-only-completions', 'input': 'Say hi'}
            r_missing = await client.post('/v1/responses', json=payload_missing)
            assert r_missing.status_code == 404

            payload = {'model': 'test-model', 'input': 'Say hi'}
            r2 = await client.post('/v1/responses', json=payload)
            assert r2.status_code == 200

            body = r2.json()
            assert body['model'] == 'test-model'

            assert any(
                isinstance(part, dict) and part.get('text') == 'response from mocked openai'  # type: ignore
                for out in body.get('output', [])
                for part in out.get('content', [])
            ), 'Expected to find output text from mocked openai'
