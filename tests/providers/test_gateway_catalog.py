from __future__ import annotations as _annotations

import os
from typing import get_args

import pytest
from typing_inspection.introspection import get_literal_values

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models import KnownModelName
from pydantic_ai.providers.gateway import ModelProvider as GatewayModelProvider

from ..conftest import try_import
from ..models.test_model_names import UNSUPPORTED_GATEWAY_MODEL_NAMES

with try_import() as imports_successful:
    import anthropic as anthropic
    import boto3 as boto3
    import google.genai as google_genai  # noqa: F401  # type: ignore[reportUnusedImport]
    import groq as groq
    import openai as openai

if not imports_successful():
    pytest.skip('gateway model checks require provider packages to be installed', allow_module_level=True)

pytestmark = [pytest.mark.anyio, pytest.mark.filterwarnings('ignore::DeprecationWarning')]


@pytest.fixture(scope='module')
def gateway_live_api_key(pytestconfig: pytest.Config, gateway_api_key: str | None) -> str:
    if not pytestconfig.getoption('--run-gateway-live'):
        pytest.skip('gateway catalog smoke tests require --run-gateway-live')
    if not gateway_api_key:
        message = 'gateway catalog smoke tests require `PYDANTIC_AI_GATEWAY_API_KEY` or `PAIG_API_KEY`'
        if os.getenv('CI'):
            pytest.fail(message)
        pytest.skip(message)
    return gateway_api_key


def _gateway_known_model_names() -> list[str]:
    return sorted(
        name
        for name in get_literal_values(KnownModelName.__value__, unpack_type_aliases='eager')
        if name.startswith('gateway/')
    )


def _gateway_supported_providers() -> set[str]:
    return {f'gateway/{provider}' for provider in get_args(GatewayModelProvider)}


async def _run_gateway_smoke_test(model_name: str) -> None:
    agent = Agent(model_name, model_settings={'max_tokens': 256}, tool_retries=3, output_retries=3)
    result = await agent.run('Reply with exactly OK.')
    assert result.output.strip()

    async with agent.run_stream('Reply with exactly OK.') as streamed_result:
        chunks = [chunk async for chunk in streamed_result.stream_text(debounce_by=None)]
        streamed_output = await streamed_result.get_output()
    assert chunks
    assert streamed_output.strip()
    assert chunks[-1].strip() == streamed_output.strip()


def test_gateway_known_model_names_only_use_supported_providers() -> None:
    known_gateway_providers = {model_name.split(':', maxsplit=1)[0] for model_name in _gateway_known_model_names()}
    assert known_gateway_providers <= _gateway_supported_providers()


@pytest.mark.parametrize('model_name', _gateway_known_model_names(), ids=str)
async def test_gateway_known_model_name_smoke_test(
    model_name: str, allow_model_requests: None, gateway_live_api_key: str
) -> None:
    await _run_gateway_smoke_test(model_name)


@pytest.mark.parametrize('model_name', sorted(UNSUPPORTED_GATEWAY_MODEL_NAMES), ids=str)
@pytest.mark.xfail(
    strict=True,
    raises=(ModelAPIError, ModelHTTPError),
    reason='Excluded gateway models should still fail; an XPASS means the literal can likely be restored.',
)
async def test_unsupported_gateway_known_model_name_smoke_test(
    model_name: str, allow_model_requests: None, gateway_live_api_key: str
) -> None:
    await _run_gateway_smoke_test(model_name)
