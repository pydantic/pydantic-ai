"""Saved provider connections resolve references instead of persisting key copies."""

import json

import httpx
import pytest
from pydantic import SecretStr
from pydantic_ai.exceptions import UserError

from pydantic_clai2 import api_keys, openrouter, vllm
from pydantic_clai2.credential_store import load_codex_credentials


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.mark.parametrize('provider', ['vllm', 'openrouter'])
async def test_reference_lifecycle(provider: str) -> None:
    reference = api_keys.KeyReference(name='SHARED')
    api_keys.save_key(name='SHARED', value='old-secret')
    if provider == 'vllm':
        vllm.save_connection(vllm.Connection(url='http://localhost:8000', token=reference))
    else:
        openrouter.save_connection(openrouter.Connection(token=reference))
    raw = load_codex_credentials(account=provider)
    assert raw is not None
    assert json.loads(raw)['token'] == {'name': 'SHARED'}
    assert 'old-secret' not in raw

    for secret in ('old-secret', 'rotated-secret'):
        api_keys.save_key(name='SHARED', value=secret)

        def respond(request: httpx.Request, *, expected: str = secret) -> httpx.Response:
            assert request.headers['authorization'] == f'Bearer {expected}'
            return httpx.Response(200, json={'data': [{'id': 'test'}]})

        transport = httpx.MockTransport(respond)
        if provider == 'vllm':
            connection = vllm.Connection.model_validate_json(raw)
            assert await vllm.discover(connection, transport=transport) == ['test']
            assert vllm.model('vllm:test').client.api_key == secret
        else:
            router_connection = openrouter.Connection.model_validate_json(raw)
            assert await openrouter.discover(router_connection, transport=transport) == ['test']
            assert openrouter.model('openrouter:my/test').client.api_key == secret

    api_keys.delete_key(name='SHARED')
    with pytest.raises(UserError, match='SHARED is missing'):
        if provider == 'vllm':
            vllm.model('vllm:test')
        else:
            openrouter.model('openrouter:my/test')
    with pytest.raises(UserError, match='SHARED is missing'):
        if provider == 'vllm':
            await vllm.discover(vllm.Connection.model_validate_json(raw))
        else:
            await openrouter.discover(openrouter.Connection.model_validate_json(raw))

    api_keys.save_key(name='SHARED', value='restored')
    assert api_keys.resolve_key(token=reference) == 'restored'
    assert api_keys.resolve_key(token=SecretStr('legacy-inline')) == 'legacy-inline'
