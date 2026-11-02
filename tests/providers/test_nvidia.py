from __future__ import annotations

import re

import pytest
from inline_snapshot import snapshot
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer

from ..conftest import RequestCapture, TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.embeddings import Embedder, infer_embedding_model
    from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
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


@pytest.mark.parametrize('env_base_url', [None, 'https://integrate.api.nvidia.com/v1'])
def test_nvidia_provider_need_api_key(env_base_url: str | None, env: TestEnv) -> None:
    """Test that the hosted NVIDIA API requires an API key, including when `NVIDIA_BASE_URL` names it."""
    env.remove('NVIDIA_API_KEY')
    if env_base_url is None:
        env.remove('NVIDIA_BASE_URL')
    else:
        env.set('NVIDIA_BASE_URL', env_base_url)
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `NVIDIA_API_KEY` environment variable or pass it via `NvidiaProvider(api_key=...)`'
            ' to use the NVIDIA provider.'
        ),
    ):
        NvidiaProvider()


@pytest.mark.parametrize('base_url_source', ['argument', 'env'])
def test_nvidia_provider_self_hosted_without_api_key(base_url_source: str, env: TestEnv) -> None:
    """A self-hosted NIM deployment can accept anonymous requests, so a custom base URL needs no key."""
    env.remove('NVIDIA_API_KEY')
    if base_url_source == 'env':
        env.set('NVIDIA_BASE_URL', 'http://localhost:8000/v1')
        provider = NvidiaProvider()
    else:
        env.remove('NVIDIA_BASE_URL')
        provider = NvidiaProvider(base_url='http://localhost:8000/v1')
    assert provider.base_url == 'http://localhost:8000/v1'
    assert provider.client.api_key == 'api-key-not-set'


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


# Ids from the NVIDIA API catalog (https://integrate.api.nvidia.com/v1/models), except Qwen, which has none listed
# there today. Each expectation is a field only that publisher's family profile sets; the `nvidia/` row pins the
# OpenAI-compatible fallback that NVIDIA's own models get.
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


@pytest.mark.vcr
async def test_nvidia_chat(allow_model_requests: None, nvidia_api_key: str, request_capture: RequestCapture):
    provider = NvidiaProvider(api_key=nvidia_api_key, http_client=request_capture.client)
    agent = Agent(OpenAIChatModel('nvidia/nemotron-3.5-super-vl-120b-a12b', provider=provider))
    result = await agent.run('What is the capital of France? Answer in one word.')
    assert result.output == snapshot()
    assert request_capture.paths == ['/v1/chat/completions']


@pytest.mark.vcr
@pytest.mark.parametrize(
    'model_name',
    ['deepseek-ai/deepseek-v4.1-flash', 'moonshotai/kimi-k2.6', 'z-ai/glm-5.3-flash'],
    ids=['deepseek', 'moonshotai', 'zai'],
)
async def test_nvidia_reasoning_round_trip(
    model_name: str, allow_model_requests: None, nvidia_api_key: str, request_capture: RequestCapture
):
    """Pins where NIM returns reasoning and what a tool-call turn sends back with the default profile fields.

    The direct DeepSeek, Moonshot AI and Z.AI providers set `openai_chat_thinking_field='reasoning_content'`;
    `NvidiaProvider` leaves it to the default parsing of `reasoning` and `reasoning_content`.
    """
    provider = NvidiaProvider(api_key=nvidia_api_key, http_client=request_capture.client)
    agent = Agent(OpenAIChatModel(model_name, provider=provider))

    @agent.tool_plain
    def get_temperature(city: str) -> str:
        return f'21 degrees Celsius in {city}'

    result = await agent.run('What is the temperature in Paris? Use the tool.')
    assert result.all_messages() == snapshot()
    _, tool_result_request = request_capture.bodies('/chat/completions')
    assert tool_result_request['messages'] == snapshot()


class CityLocation(BaseModel):
    city: str
    country: str


@pytest.mark.vcr
async def test_nvidia_nemotron_forced_tool_choice(
    allow_model_requests: None, nvidia_api_key: str, request_capture: RequestCapture
):
    """Output tools send `tool_choice: 'required'`, which Nemotron answered as text on other hosts (#6280)."""
    provider = NvidiaProvider(api_key=nvidia_api_key, http_client=request_capture.client)
    agent = Agent(
        OpenAIChatModel('nvidia/nemotron-3.5-super-vl-120b-a12b', provider=provider), output_type=CityLocation
    )
    result = await agent.run('The largest city in Japan.')
    assert result.output == snapshot()
    assert request_capture.body('/chat/completions')['tool_choice'] == 'required'


@pytest.mark.vcr
async def test_nvidia_embeddings(
    allow_model_requests: None, nvidia_api_key: str, request_capture: RequestCapture, env: TestEnv
):
    """`nvidia:` embedding models route through `OpenAIEmbeddingModel` to NIM's `/v1/embeddings`."""
    env.set('NVIDIA_API_KEY', nvidia_api_key)
    env.remove('NVIDIA_BASE_URL')
    inferred = infer_embedding_model('nvidia:nvidia/nemotron-3.5-embed-1b')
    assert isinstance(inferred, OpenAIEmbeddingModel)
    assert inferred.system == 'nvidia'

    provider = NvidiaProvider(api_key=nvidia_api_key, http_client=request_capture.client)
    embedder = Embedder(OpenAIEmbeddingModel('nvidia/nemotron-3.5-embed-1b', provider=provider))
    result = await embedder.embed_query('Hello, world!')
    assert result == snapshot()
    assert request_capture.paths == ['/v1/embeddings']
