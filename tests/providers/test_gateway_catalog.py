from __future__ import annotations as _annotations

from typing import get_args

import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError
from pydantic_ai.models import infer_model, known_model_names
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

pytestmark = [pytest.mark.filterwarnings('ignore::DeprecationWarning')]


def _gateway_known_model_names() -> list[str]:
    return sorted(name for name in known_model_names() if name.startswith('gateway/'))


def _gateway_supported_providers() -> set[str]:
    return {f'gateway/{provider}' for provider in get_args(GatewayModelProvider)}


async def _run_gateway_smoke_test(model_name: str) -> None:
    model = infer_model(model_name)
    # Reasoning models (OpenAI o-series, DeepSeek R1, etc.) always spend tokens on reasoning before
    # emitting any output, so a tiny budget can be exhausted before a visible reply is produced
    # (`finish_reason='length'` with no parts). Give them more headroom while keeping the cheap
    # models cheap.
    max_tokens = 4096 if model.profile.get('thinking_always_enabled', False) else 256
    agent = Agent(model, model_settings={'max_tokens': max_tokens}, retries={'tools': 3, 'output': 3})
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
