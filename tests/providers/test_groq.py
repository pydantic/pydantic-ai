from __future__ import annotations as _annotations

import json
import re
from typing import Any

import httpx
import pytest
from pydantic import BaseModel
from pytest_mock import MockerFixture

from pydantic_ai import Agent, NativeOutput, PromptedOutput
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.groq import groq_model_profile
from pydantic_ai.profiles.meta import meta_model_profile
from pydantic_ai.profiles.mistral import mistral_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import openai_model_profile
from pydantic_ai.profiles.qwen import qwen_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    from groq import AsyncGroq

    from pydantic_ai.models.groq import GroqModel
    from pydantic_ai.providers.groq import GroqProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='groq not installed')


def test_groq_provider():
    provider = GroqProvider(api_key='api-key')
    assert provider.name == 'groq'
    assert provider.base_url == 'https://api.groq.com'
    assert isinstance(provider.client, AsyncGroq)
    assert provider.client.api_key == 'api-key'


def test_groq_provider_need_api_key(env: TestEnv) -> None:
    env.remove('GROQ_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GROQ_API_KEY` environment variable or pass it via `GroqProvider(api_key=...)`'
            ' to use the Groq provider.'
        ),
    ):
        GroqProvider()


def test_groq_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = GroqProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_groq_provider_pass_groq_client() -> None:
    groq_client = AsyncGroq(api_key='api-key')
    provider = GroqProvider(groq_client=groq_client)
    assert provider.client == groq_client


def test_groq_provider_with_env_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test with environment variable for base_url
    monkeypatch.setenv('GROQ_BASE_URL', 'https://custom.groq.com/v1')
    provider = GroqProvider(api_key='api-key')
    assert provider.base_url == 'https://custom.groq.com/v1/'


def test_groq_provider_model_profile(mocker: MockerFixture):
    groq_client = AsyncGroq(api_key='api-key')
    provider = GroqProvider(groq_client=groq_client)

    ns = 'pydantic_ai.providers.groq'
    meta_model_profile_mock = mocker.patch(f'{ns}.meta_model_profile', wraps=meta_model_profile)
    deepseek_model_profile_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_model_profile_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    mistral_model_profile_mock = mocker.patch(f'{ns}.mistral_model_profile', wraps=mistral_model_profile)
    qwen_model_profile_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    moonshotai_model_profile_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)
    groq_model_profile_mock = mocker.patch(f'{ns}.groq_model_profile', wraps=groq_model_profile)
    openai_model_profile_mock = mocker.patch(f'{ns}.openai_model_profile', wraps=openai_model_profile)

    meta_profile = provider.model_profile('meta-llama/Llama-Guard-4-12B')
    meta_model_profile_mock.assert_called_with('llama-guard-4-12b')
    assert meta_profile is not None
    assert meta_profile.get('supports_json_object_output', False) is True
    assert meta_profile.get('supports_json_schema_output', False) is False
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('meta-llama/llama-4-maverick-17b-128e-instruct')
    meta_model_profile_mock.assert_called_with('llama-4-maverick-17b-128e-instruct')
    assert meta_profile is not None
    assert meta_profile.get('supports_json_object_output', False) is True
    assert meta_profile.get('supports_json_schema_output', False) is True
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('meta-llama/llama-4-scout-17b-16e-instruct')
    meta_model_profile_mock.assert_called_with('llama-4-scout-17b-16e-instruct')
    assert meta_profile is not None
    assert meta_profile.get('supports_json_object_output', False) is True
    assert meta_profile.get('supports_json_schema_output', False) is True
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    meta_profile = provider.model_profile('llama-3.3-70b-versatile')
    meta_model_profile_mock.assert_called_with('llama-3.3-70b-versatile')
    assert meta_profile is not None
    assert meta_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    google_profile = provider.model_profile('gemma2-9b-it')
    google_model_profile_mock.assert_called_with('gemma2-9b-it')
    assert google_profile is not None
    assert google_profile.get('json_schema_transformer', None) == GoogleJsonSchemaTransformer

    deepseek_profile = provider.model_profile('deepseek-r1-distill-llama-70b')
    deepseek_model_profile_mock.assert_called_with('deepseek-r1-distill-llama-70b')
    assert deepseek_profile is not None
    assert deepseek_profile.get('ignore_streamed_leading_whitespace', False) is True

    mistral_profile = provider.model_profile('mistral-saba-24b')
    mistral_model_profile_mock.assert_called_with('mistral-saba-24b')
    assert mistral_profile is not None
    assert mistral_profile.get('supports_inline_system_prompts', False) is True

    qwen_profile = provider.model_profile('qwen-qwq-32b')
    qwen_model_profile_mock.assert_called_with('qwen-qwq-32b')
    assert qwen_profile is not None
    assert qwen_profile.get('json_schema_transformer', None) == InlineDefsJsonSchemaTransformer

    qwen_profile = provider.model_profile('compound-beta')
    groq_model_profile_mock.assert_called_with('compound-beta')
    assert qwen_profile is not None
    assert isinstance(qwen_profile, dict)
    assert qwen_profile.get('groq_always_has_web_search_builtin_tool', False) is True

    moonshotai_profile = provider.model_profile('moonshotai/kimi-k2-instruct')
    moonshotai_model_profile_mock.assert_called_with('kimi-k2-instruct')
    assert moonshotai_profile is not None
    assert moonshotai_profile.get('supports_json_object_output', False) is True
    assert moonshotai_profile.get('supports_json_schema_output', False) is True
    assert moonshotai_profile.get('ignore_streamed_leading_whitespace', False) is True

    openai_profile = provider.model_profile('openai/gpt-oss-20b')
    openai_model_profile_mock.assert_called_with('gpt-oss-20b')
    assert openai_profile is not None
    assert openai_profile.get('supports_json_object_output', False) is True
    assert openai_profile.get('supports_json_schema_output', False) is True

    openai_profile = provider.model_profile('openai/gpt-oss-120b')
    openai_model_profile_mock.assert_called_with('gpt-oss-120b')
    assert openai_profile is not None
    assert openai_profile.get('supports_json_object_output', False) is True
    assert openai_profile.get('supports_json_schema_output', False) is True

    unknown_profile = provider.model_profile('unknown-model')
    assert unknown_profile is not None
    assert unknown_profile.get('supports_inline_system_prompts', False) is True


def test_groq_provider_model_profile_reasoning_flags():
    """`GroqProvider.model_profile` overlays Groq's reasoning flags onto every family profile.

    Unit (not VCR): the overlay resolves profile flags read at request-build time, and the cassette
    matcher isn't sensitive to which flags resolved — so a regression that flips a reasoning model
    back to non-reasoning (or vice versa) would still match an existing recording. A direct profile
    assertion is what catches it.
    """
    provider = GroqProvider(groq_client=AsyncGroq(api_key='api-key'))

    # qwen3 (current): reasons, and can truly disable via `reasoning_effort='none'`.
    qwen3_profile = provider.model_profile('qwen/qwen3-32b')
    assert qwen3_profile is not None
    assert qwen3_profile.get('supports_thinking', False) is True
    assert qwen3_profile.get('thinking_always_enabled', False) is False
    assert qwen3_profile.get('groq_supports_reasoning_disable', False) is True

    # Legacy reasoning models route through generic family profiles that don't flag reasoning; the
    # overlay restores it. They reason always-on (no true disable).
    for model_name in ('qwen-qwq-32b', 'llama-4-maverick'):
        legacy_profile = provider.model_profile(model_name)
        assert legacy_profile is not None, model_name
        assert legacy_profile.get('supports_thinking', False) is True, model_name
        assert legacy_profile.get('thinking_always_enabled', False) is True, model_name
        assert legacy_profile.get('groq_supports_reasoning_disable', False) is False, model_name

    # Non-reasoning model: the overlay leaves it untouched.
    plain_profile = provider.model_profile('llama-3.1-8b-instant')
    assert plain_profile is not None
    assert plain_profile.get('supports_thinking', False) is False
    assert plain_profile.get('thinking_always_enabled', False) is False


class CityLocation(BaseModel):
    city: str
    country: str


def test_groq_model_profile_json_flags_and_reasoning_list():
    """`supports_json_object_output` resolves provider-wide; JSON schema and thinking resolve per model."""
    provider = GroqProvider(api_key='api-key')

    # JSON mode is provider-wide.
    for model_name in ('llama-3.1-8b-instant', 'meta-llama/Llama-Guard-4-12B', 'minimaxai/minimax-m2.7'):
        profile = provider.model_profile(model_name)
        assert profile is not None, model_name
        assert profile.get('supports_json_object_output', False) is True, model_name

    # JSON-schema structured output only on the models that accept it.
    for model_name in ('meta-llama/Llama-Guard-4-12B', 'minimaxai/minimax-m2.7'):
        profile = provider.model_profile(model_name)
        assert profile is not None, model_name
        assert profile.get('supports_json_schema_output', False) is False, model_name

    for model_name in ('qwen/qwen3.8-32b', 'qwen/qwen3.6-32b', 'meta-models/muse-spark-open'):
        profile = provider.model_profile(model_name)
        assert profile is not None, model_name
        assert profile.get('supports_json_schema_output', False) is True, model_name

    # Muse, MiniMax M2.7 and the qwen3.8 line all reason on Groq.
    for model_name in ('meta-models/muse-spark-open', 'minimaxai/minimax-m2.7', 'qwen/qwen3.8-32b'):
        profile = provider.model_profile(model_name)
        assert profile is not None, model_name
        assert profile.get('supports_thinking', False) is True, model_name


def test_groq_native_output_with_function_tools_raises_user_error(allow_model_requests: None):
    """`NativeOutput` with function tools raises before any request is sent, with the `ToolOutput` remedy."""
    provider = GroqProvider(api_key='api-key')
    model = GroqModel('meta-models/muse-spark-open', provider=provider)
    agent = Agent(model, output_type=NativeOutput(CityLocation))

    @agent.tool_plain
    def get_city() -> str:
        return 'Mexico City'

    # The tool itself is fine; only native output combined with function tools is rejected.
    assert get_city() == 'Mexico City'

    with pytest.raises(UserError, match=re.escape('does not support `NativeOutput` and function tools')):
        agent.run_sync('What is the city?')


async def test_groq_structured_output_streaming_and_prompted_request_bodies(allow_model_requests: None):
    """Streaming native output sends a `json_schema` response format; prompted output sends `json_object`."""
    captured: list[dict[str, Any]] = []

    def stream_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        chunk = {
            'id': '0',
            'object': 'chat.completion.chunk',
            'created': 0,
            'model': 'meta-models/muse-spark-open',
            'choices': [
                {
                    'index': 0,
                    'delta': {
                        'role': 'assistant',
                        'content': '{"city": "Mexico City", "country": "Mexico"}',
                    },
                    'finish_reason': None,
                }
            ],
        }
        stop_chunk: dict[str, Any] = {
            'id': '0',
            'object': 'chat.completion.chunk',
            'created': 0,
            'model': 'meta-models/muse-spark-open',
            'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
        }
        body = ''.join(f'data: {json.dumps(c)}\n\n' for c in (chunk, stop_chunk)) + 'data: [DONE]\n\n'
        return httpx.Response(200, content=body, headers={'content-type': 'text/event-stream'})

    async with httpx.AsyncClient(transport=httpx.MockTransport(stream_handler)) as http_client:
        provider = GroqProvider(api_key='test-key', base_url='http://localhost', http_client=http_client)
        agent = Agent(
            GroqModel('meta-models/muse-spark-open', provider=provider), output_type=NativeOutput(CityLocation)
        )
        async with agent.run_stream('What is the city?') as result:
            assert await result.get_output() == CityLocation(city='Mexico City', country='Mexico')

    assert len(captured) == 1
    response_format = captured[0]['response_format']
    assert response_format['type'] == 'json_schema'
    assert set(response_format['json_schema']['schema']['properties']) == {'city', 'country'}

    captured.clear()

    def completion_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                'id': '0',
                'object': 'chat.completion',
                'created': 0,
                'model': 'meta-models/muse-spark-open',
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': '{"city": "Mexico City", "country": "Mexico"}',
                            'tool_calls': None,
                        },
                        'finish_reason': 'stop',
                    }
                ],
                'usage': {
                    'prompt_tokens': 1,
                    'completion_tokens': 1,
                    'total_tokens': 2,
                    'prompt_time': 0.001,
                    'completion_time': 0.001,
                    'queue_time': None,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(completion_handler)) as http_client:
        provider = GroqProvider(api_key='test-key', base_url='http://localhost', http_client=http_client)
        agent = Agent(
            GroqModel('meta-models/muse-spark-open', provider=provider), output_type=PromptedOutput(CityLocation)
        )
        result = await agent.run('What is the city?')
        assert result.output == CityLocation(city='Mexico City', country='Mexico')

    assert len(captured) == 1
    assert captured[0]['response_format'] == {'type': 'json_object'}
