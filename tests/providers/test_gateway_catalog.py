from __future__ import annotations as _annotations

import warnings
from dataclasses import dataclass
from typing import Any, get_args

import pytest
from typing_inspection.introspection import get_literal_values

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models import KnownModelName
from pydantic_ai.providers.gateway import ModelProvider as GatewayModelProvider

from ..conftest import try_import

with try_import() as imports_successful:
    import anthropic as anthropic
    import boto3 as boto3
    import google.genai as google_genai  # noqa: F401  # type: ignore[reportUnusedImport]
    import groq as groq
    import openai as openai

if not imports_successful():
    pytest.skip('gateway model checks require provider packages to be installed', allow_module_level=True)

pytestmark = pytest.mark.anyio


@pytest.fixture(scope='module')
def gateway_live_api_key(pytestconfig: pytest.Config, gateway_api_key: str | None) -> str:
    return _require_gateway_live_api_key(
        run_gateway_live=pytestconfig.getoption('--run-gateway-live'),
        gateway_api_key=gateway_api_key,
    )


def _require_gateway_live_api_key(*, run_gateway_live: bool, gateway_api_key: str | None) -> str:
    if not run_gateway_live:
        pytest.skip('gateway catalog smoke tests require --run-gateway-live')
    if not gateway_api_key:
        pytest.skip('gateway catalog smoke tests require `PYDANTIC_AI_GATEWAY_API_KEY` or `PAIG_API_KEY`')
    return gateway_api_key


def _gateway_known_model_names() -> list[str]:
    return sorted(
        name
        for name in get_literal_values(KnownModelName.__value__, unpack_type_aliases='eager')
        if name.startswith('gateway/')
    )


def _gateway_supported_providers() -> set[str]:
    return {f'gateway/{provider}' for provider in get_args(GatewayModelProvider)}


def _is_retryable_gateway_failure(error: ModelAPIError) -> bool:
    if isinstance(error, ModelHTTPError):
        return error.status_code == 429 or error.status_code >= 500
    return error.message == 'Connection error.'


async def _run_gateway_smoke_test(agent: Any) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        result = await agent.run('Reply with exactly OK.')
    output = result.output
    assert isinstance(output, str)
    return output


async def _smoke_test_model_name(model_name: str) -> str:
    agent = Agent(model_name, model_settings={'max_tokens': 256}, retries=3)
    output = await _run_gateway_smoke_test(agent)
    assert output.strip()
    return output


def test_gateway_known_model_names_only_use_supported_providers() -> None:
    known_gateway_providers = {model_name.split(':', maxsplit=1)[0] for model_name in _gateway_known_model_names()}
    assert known_gateway_providers <= _gateway_supported_providers()


def test_require_gateway_live_api_key_requires_flag() -> None:
    with pytest.raises(pytest.skip.Exception, match='--run-gateway-live'):
        _require_gateway_live_api_key(run_gateway_live=False, gateway_api_key='testing')


def test_require_gateway_live_api_key_requires_key() -> None:
    with pytest.raises(pytest.skip.Exception, match='PYDANTIC_AI_GATEWAY_API_KEY'):
        _require_gateway_live_api_key(run_gateway_live=True, gateway_api_key=None)


def test_require_gateway_live_api_key_returns_key() -> None:
    assert _require_gateway_live_api_key(run_gateway_live=True, gateway_api_key='testing') == 'testing'


@pytest.mark.parametrize(
    ('error', 'expected'),
    [
        pytest.param(ModelHTTPError(status_code=429, model_name='test', body=None), True, id='http-429'),
        pytest.param(ModelHTTPError(status_code=500, model_name='test', body=None), True, id='http-500'),
        pytest.param(ModelHTTPError(status_code=400, model_name='test', body=None), False, id='http-400'),
        pytest.param(ModelAPIError(model_name='test', message='Connection error.'), True, id='connection-error'),
        pytest.param(ModelAPIError(model_name='test', message='bad request'), False, id='api-error'),
    ],
)
def test_is_retryable_gateway_failure(error: ModelAPIError, expected: bool) -> None:
    assert _is_retryable_gateway_failure(error) is expected


@dataclass
class _FakeRunResult:
    output: Any


class _FakeAgent:
    def __init__(self, output: Any):
        self.output = output

    async def run(self, prompt: str) -> _FakeRunResult:
        assert prompt == 'Reply with exactly OK.'
        warnings.warn('deprecated', DeprecationWarning)
        return _FakeRunResult(self.output)


async def test_run_gateway_smoke_test_ignores_deprecation_warning() -> None:
    assert await _run_gateway_smoke_test(_FakeAgent('OK')) == 'OK'


async def test_smoke_test_model_name_builds_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class FakeGatewayAgent:
        def __init__(self, model_name: str, *, model_settings: dict[str, int], retries: int):
            captured['model_name'] = model_name
            captured['model_settings'] = model_settings
            captured['retries'] = retries

    async def fake_run_gateway_smoke_test(agent: Any) -> str:
        captured['agent'] = agent
        return 'OK'

    monkeypatch.setattr('tests.providers.test_gateway_catalog.Agent', FakeGatewayAgent)
    monkeypatch.setattr('tests.providers.test_gateway_catalog._run_gateway_smoke_test', fake_run_gateway_smoke_test)

    assert await _smoke_test_model_name('gateway/openai:gpt-5') == 'OK'
    assert captured == {
        'model_name': 'gateway/openai:gpt-5',
        'model_settings': {'max_tokens': 256},
        'retries': 3,
        'agent': captured['agent'],
    }


@pytest.mark.parametrize('model_name', _gateway_known_model_names(), ids=str)
async def test_gateway_known_model_name_smoke_test(
    model_name: str, allow_model_requests: None, gateway_live_api_key: str
) -> None:
    await _smoke_test_model_name(model_name)
