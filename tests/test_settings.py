import importlib
import json
from typing import Any

import pytest
from vcr.cassette import Cassette

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings, merge_model_settings

from ._inline_snapshot import snapshot
from .conftest import try_import

with try_import() as google_available:
    from pydantic_ai.models.google import GoogleModel, GoogleModelSettings
    from pydantic_ai.providers.google import GoogleProvider

with try_import() as bedrock_available:
    from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings

pytestmark = [pytest.mark.anyio, pytest.mark.vcr]

# Provider-specific keys for common model settings in the request body
MAX_TOKENS_KEYS: dict[str, str] = {
    'openai': 'max_completion_tokens',
    'anthropic': 'max_tokens',
    'google-gla': 'maxOutputTokens',
    'bedrock': 'maxTokens',
    'groq': 'max_tokens',
    'mistral': 'max_tokens',
    'cohere': 'max_tokens',
}

TOP_P_KEYS: dict[str, str] = {
    'openai': 'top_p',
    'anthropic': 'top_p',
    'google-gla': 'topP',
    'bedrock': 'topP',
    'groq': 'top_p',
    'mistral': 'top_p',
    'cohere': 'p',
}


def _get_request_body(vcr: Cassette | None) -> dict[str, Any]:  # pragma: lax no cover
    assert vcr is not None
    assert vcr.requests, 'No requests recorded'  # pyright: ignore[reportUnknownMemberType]
    # Iterate from the end to find the model API request (JSON body), skipping auth requests like STS
    for request in reversed(vcr.requests):  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType,reportUnknownArgumentType]
        body = request.body  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        if not isinstance(body, str):
            continue
        try:
            result: dict[str, Any] = json.loads(body)
            return result
        except (json.JSONDecodeError, ValueError):
            continue
    raise AssertionError('No JSON request body found in cassette')


@pytest.fixture(params=['openai_', 'anthropic_', 'bedrock_', 'groq_', 'gemini_', 'mistral_', 'cohere_'])
def settings(request: pytest.FixtureRequest) -> tuple[type[ModelSettings], str]:
    prefix_cls_name = request.param.replace('_', '')
    try:
        module = importlib.import_module(f'pydantic_ai.models.{prefix_cls_name}')
    except ImportError:  # pragma: lax no cover
        pytest.skip(f'{prefix_cls_name} is not installed')
    capitalized_prefix = prefix_cls_name.capitalize().replace('Openai', 'OpenAI')
    cls = getattr(module, capitalized_prefix + 'ModelSettings')
    return cls, request.param


def test_specific_prefix_settings(settings: tuple[type[ModelSettings], str]):
    settings_cls, prefix = settings
    global_settings = set(ModelSettings.__annotations__.keys())
    specific_settings = set(settings_cls.__annotations__.keys()) - global_settings
    assert all(setting.startswith(prefix) for setting in specific_settings), (
        f'{prefix} is not a prefix for {specific_settings}'
    )


@pytest.mark.parametrize(
    'model', ['openai', 'anthropic', 'bedrock', 'mistral', 'groq', 'cohere', 'google'], indirect=True
)
async def test_stop_settings(allow_model_requests: None, model: Model) -> None:
    agent = Agent(model=model, model_settings=ModelSettings(stop_sequences=['Paris']))
    result = await agent.run(
        'What is the capital of France? Give me an answer that contains the word "Paris", but is not the first word.'
    )

    # NOTE: Bedrock has a slightly different behavior. It will include the stop sequence in the response.
    if model.system == 'bedrock':
        assert result.output.endswith('Paris')
    else:
        assert 'Paris' not in result.output


@pytest.mark.parametrize(
    'model', ['openai', 'anthropic', 'bedrock', 'mistral', 'groq', 'cohere', 'google'], indirect=True
)
async def test_max_tokens_settings(allow_model_requests: None, model: Model, vcr: Cassette | None) -> None:
    agent = Agent(model=model, model_settings=ModelSettings(max_tokens=500))
    result = await agent.run('What is the capital of France?')

    request_body = _get_request_body(vcr)
    key = MAX_TOKENS_KEYS[model.system]
    body_str = json.dumps(request_body)
    assert f'"{key}": 500' in body_str

    assert 'Paris' in result.output or 'paris' in result.output.lower()


@pytest.mark.filterwarnings('ignore:Sampling parameters.*not supported when reasoning is enabled')
@pytest.mark.parametrize(
    'model', ['openai', 'anthropic', 'bedrock', 'mistral', 'groq', 'cohere', 'google'], indirect=True
)
async def test_top_p_settings(allow_model_requests: None, model: Model, vcr: Cassette | None) -> None:
    agent = Agent(model=model, model_settings=ModelSettings(top_p=0.5))
    result = await agent.run('What is the capital of France?')

    request_body = _get_request_body(vcr)
    key = TOP_P_KEYS[model.system]
    body_str = json.dumps(request_body)
    # Reasoning models (e.g. o3-mini) strip sampling params from the request
    if key in body_str:
        assert f'"{key}": 0.5' in body_str or f'"{key}":0.5' in body_str

    assert 'Paris' in result.output or 'paris' in result.output.lower()


# --- Google provider-specific settings ---


@pytest.fixture()
def google_provider(gemini_api_key: str):
    return GoogleProvider(api_key=gemini_api_key)


@pytest.mark.skipif(not google_available(), reason='google-genai not installed')
async def test_google_model_thinking_config(allow_model_requests: None, google_provider: Any, vcr: Cassette | None):
    model = GoogleModel('gemini-3-flash-preview', provider=google_provider)
    settings = GoogleModelSettings(google_thinking_config={'include_thoughts': False})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot('The capital of France is **Paris**.')

    request_body = _get_request_body(vcr)
    assert request_body['generationConfig']['thinkingConfig'] == {'include_thoughts': False}


# --- Bedrock provider-specific settings ---


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_bedrock_model_performance_config(
    allow_model_requests: None, bedrock_provider: Any, vcr: Cassette | None
):
    model = BedrockConverseModel('us.amazon.nova-pro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_performance_configuration={'latency': 'optimized'})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. It is one of the most visited cities in the world and is known for its rich history, culture, and iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral. Paris is also a major center for finance, diplomacy, commerce, fashion, science, and arts.'
    )

    request_body = _get_request_body(vcr)
    assert request_body['performanceConfig'] == {'latency': 'optimized'}


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_bedrock_model_guardrail_config(allow_model_requests: None, bedrock_provider: Any, vcr: Cassette | None):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_guardrail_config={
            'guardrailIdentifier': 'xbgw7g293v7o',
            'guardrailVersion': 'DRAFT',
            'trace': 'enabled',
        }
    )
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the capital city but also a major cultural, economic, and political center of the country. It is well-known for its historical landmarks, such as the Eiffel Tower, Louvre Museum, Notre-Dame Cathedral, and Champs-Élysées, among many other attractions.'
    )

    request_body = _get_request_body(vcr)
    assert request_body['guardrailConfig'] == {
        'guardrailIdentifier': 'xbgw7g293v7o',
        'guardrailVersion': 'DRAFT',
        'trace': 'enabled',
    }


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_bedrock_model_request_metadata(allow_model_requests: None, bedrock_provider: Any, vcr: Cassette | None):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_request_metadata={'test': 'test'})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        "The capital of France is Paris. Paris is not only the political center of the country but also one of the world's most influential cities in terms of culture, art, fashion, and cuisine. It is located in the northern central part of France and is known for iconic landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
    )

    request_body = _get_request_body(vcr)
    assert request_body['requestMetadata'] == {'test': 'test'}


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_bedrock_model_service_tier(allow_model_requests: None, bedrock_provider: Any, vcr: Cassette | None):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_service_tier={'type': 'default'})
    agent = Agent(model=model, system_prompt='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Known for its rich history, iconic landmarks such as the Eiffel Tower, Louvre Museum, and Notre-Dame Cathedral, as well as its significant influence in art, culture, fashion, and gastronomy, Paris is a major global city and a central hub of French political, economic, and cultural life.'
    )

    request_body = _get_request_body(vcr)
    assert request_body['serviceTier'] == {'type': 'default'}


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_bedrock_model_prompt_variables(allow_model_requests: None, bedrock_provider: Any, vcr: Cassette | None):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(bedrock_prompt_variables={'leo': {'text': 'aaaa'}})
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        'The capital of France is Paris. Paris is not only the political center of France but also one of the most significant cultural, historical, and economic cities in the world. It is known for its landmarks such as the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral, among many others.'
    )

    request_body = _get_request_body(vcr)
    assert request_body['promptVariables'] == {'leo': {'text': 'aaaa'}}


@pytest.mark.skipif(not bedrock_available(), reason='bedrock not installed')
async def test_bedrock_model_additional_response_fields(
    allow_model_requests: None, bedrock_provider: Any, vcr: Cassette | None
):
    model = BedrockConverseModel('us.amazon.nova-micro-v1:0', provider=bedrock_provider)
    model_settings = BedrockModelSettings(
        bedrock_additional_model_response_fields_paths=['/amazon-bedrock-invocationMetrics'],
    )
    agent = Agent(model=model, instructions='You are a helpful chatbot.', model_settings=model_settings)
    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        "The capital of France is Paris. Paris is not only the political and administrative center of the country but also a major cultural, historical, and economic hub. It's well-known for its landmarks such as the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral, and the Champs-Élysées."
    )

    request_body = _get_request_body(vcr)
    assert request_body['additionalModelResponseFieldPaths'] == ['/amazon-bedrock-invocationMetrics']


class TestMergeModelSettingsThinking:
    """merge_model_settings with unified thinking fields."""

    def test_merge_thinking_bool_override(self):
        base: ModelSettings = {'thinking': True}
        overrides: ModelSettings = {'thinking': False}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('thinking') is False

    def test_merge_effort_override(self):
        base: ModelSettings = {'thinking': 'low'}
        overrides: ModelSettings = {'thinking': 'high'}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('thinking') == 'high'

    def test_merge_preserves_non_thinking_settings(self):
        base: ModelSettings = {'max_tokens': 1000, 'temperature': 0.5}
        overrides: ModelSettings = {'thinking': True}
        result = merge_model_settings(base, overrides)
        assert result is not None
        assert result.get('max_tokens') == 1000
        assert result.get('temperature') == 0.5
        assert result.get('thinking') is True

    def test_merge_with_none_returns_base(self):
        base: ModelSettings = {'thinking': True}
        result = merge_model_settings(base, None)
        assert result == base

    def test_merge_with_none_base_returns_overrides(self):
        overrides: ModelSettings = {'thinking': True}
        result = merge_model_settings(None, overrides)
        assert result == overrides

    def test_merge_with_both_none(self):
        result = merge_model_settings(None, None)
        assert result is None
