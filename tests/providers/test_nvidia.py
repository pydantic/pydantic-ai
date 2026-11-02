from __future__ import annotations

import json
import re

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.nvidia import NvidiaProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_nvidia_provider(env: TestEnv):
    """Test basic NVIDIA provider initialization."""
    env.remove('NVIDIA_BASE_URL')
    provider = NvidiaProvider(api_key='api-key')
    assert provider.name == 'nvidia'
    assert provider.base_url == 'https://integrate.api.nvidia.com/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_nvidia_provider_need_api_key(env: TestEnv) -> None:
    """Test that NVIDIA provider requires an API key."""
    env.remove('NVIDIA_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `NVIDIA_API_KEY` environment variable or pass it via `NvidiaProvider(api_key=...)`'
            ' to use the NVIDIA provider.'
        ),
    ):
        NvidiaProvider()


def test_nvidia_pass_openai_client(env: TestEnv) -> None:
    """Test passing a custom OpenAI client to NVIDIA provider."""
    env.set('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
    openai_client = openai.AsyncOpenAI(api_key='api-key', base_url='http://localhost:8000/v1')
    provider = NvidiaProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert provider.base_url == 'http://localhost:8000/v1/'


def test_nvidia_provider_custom_base_url(env: TestEnv) -> None:
    env.set('NVIDIA_BASE_URL', 'https://integrate.api.nvidia.com/v1')
    provider = NvidiaProvider(api_key='api-key', base_url='http://localhost:8000/v1')
    assert provider.base_url == 'http://localhost:8000/v1'
    assert str(provider.client.base_url) == 'http://localhost:8000/v1/'


@pytest.mark.parametrize('base_url', ['http://localhost:8000/v1', ''])
def test_nvidia_provider_env_base_url(base_url: str, env: TestEnv) -> None:
    """Test model-string endpoint resolution without a live API request."""
    env.set('NVIDIA_BASE_URL', base_url)
    env.set('NVIDIA_API_KEY', 'api-key')
    agent = Agent('nvidia:nvidia/nemotron-3.5-super-vl-120b-a12b')
    assert isinstance(agent.model, OpenAIChatModel)
    expected_url = base_url or 'https://integrate.api.nvidia.com/v1'
    assert agent.model.base_url == f'{expected_url}/'
    assert agent.model.client.api_key == 'api-key'


def test_nvidia_provider_openai_client_and_base_url_are_exclusive() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `base_url`'):
        # No overload accepts both arguments; the call checks the runtime guard behind that.
        NvidiaProvider(openai_client=openai_client, base_url='http://localhost:8000/v1')  # type: ignore[call-overload]


def test_nvidia_model_profile():
    provider = NvidiaProvider(api_key='api-key')
    model = OpenAIChatModel('nvidia/nemotron-3.5-super-vl-120b-a12b', provider=provider)
    assert isinstance(model.profile, dict)
    assert model.profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


# Ids from the NVIDIA API catalog (https://integrate.api.nvidia.com/v1/models); Qwen has none listed there today.
# Each expectation is a field the publisher's family profile sets and the OpenAI-compatible fallback doesn't.
PUBLISHER_PROFILES: dict[str, dict[str, object]] = {
    'meta/llama-3.2-90b-vision-instruct': {'json_schema_transformer': InlineDefsJsonSchemaTransformer},
    'deepseek-ai/deepseek-v4.1-flash': {'supports_thinking': False, 'ignore_streamed_leading_whitespace': False},
    'qwen/qwen3-next-80b-a3b-instruct': {
        'json_schema_transformer': InlineDefsJsonSchemaTransformer,
        'ignore_streamed_leading_whitespace': True,
    },
    'google/gemma-4-31b-it': {'json_schema_transformer': GoogleJsonSchemaTransformer},
    'openai/gpt-oss-20b': {'supports_forced_tool_choice': False},
    'mistralai/mistral-large-4-1t-a52b': {'tool_call_id_length': 9},
    'moonshotai/kimi-k2.6': {'supports_thinking': True},
    'z-ai/glm-5.3-flash': {'supports_thinking': True, 'thinking_always_enabled': True},
    'nvidia/nemotron-3.5-super-vl-120b-a12b': {
        'json_schema_transformer': OpenAIJsonSchemaTransformer,
        'supports_thinking': None,
    },
}


@pytest.mark.parametrize(('model_name', 'expected'), PUBLISHER_PROFILES.items(), ids=list(PUBLISHER_PROFILES))
def test_nvidia_model_profile_by_publisher(model_name: str, expected: dict[str, object]):
    # Unit (not VCR): this pins which family profile is resolved from the `<publisher>/` prefix at
    # model-build time, which a recorded round-trip doesn't assert.
    profile = NvidiaProvider.model_profile(model_name)
    assert profile is not None
    assert {key: profile.get(key) for key in expected} == expected


_COMPLETION: dict[str, object] = {
    'id': 'chatcmpl-1',
    'object': 'chat.completion',
    'created': 0,
    'model': 'nvidia/nemotron-3.5-super-vl-120b-a12b',
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '4'}, 'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
}


async def test_nvidia_request(allow_model_requests: None):
    """Mocked transport, not VCR: there's no NVIDIA API key at hand to record with."""
    requests: list[tuple[str, dict[str, object]]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append((str(request.url), json.loads(request.content)))
        return httpx2.Response(200, json=_COMPLETION)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        provider = NvidiaProvider(api_key='api-key', http_client=http_client)
        agent = Agent(OpenAIChatModel('nvidia/nemotron-3.5-super-vl-120b-a12b', provider=provider))
        result = await agent.run('What is 2+2?')
    assert result.output == '4'
    [(url, body)] = requests
    assert url == 'https://integrate.api.nvidia.com/v1/chat/completions'
    assert body['model'] == 'nvidia/nemotron-3.5-super-vl-120b-a12b'
