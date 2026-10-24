import json
import re
from typing import Any

import httpx2
import pytest

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.settings import ModelSettings

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
    from pydantic_ai.providers.moonshotai import MoonshotAIProvider

pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_moonshotai_provider(env: TestEnv):
    """Test basic MoonshotAI provider initialization."""
    env.remove('MOONSHOTAI_BASE_URL')
    provider = MoonshotAIProvider(api_key='api-key')
    assert provider.name == 'moonshotai'
    assert provider.base_url == 'https://api.moonshot.ai/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_moonshotai_provider_need_api_key(env: TestEnv) -> None:
    """Test that MoonshotAI provider requires an API key."""
    env.remove('MOONSHOTAI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `MOONSHOTAI_API_KEY` environment variable or pass it via `MoonshotAIProvider(api_key=...)`'
            ' to use the MoonshotAI provider.'
        ),
    ):
        MoonshotAIProvider()


def test_moonshotai_pass_openai_client(env: TestEnv) -> None:
    """Test passing a custom OpenAI client to MoonshotAI provider."""
    env.set('MOONSHOTAI_BASE_URL', 'https://api.moonshot.ai/v1')
    openai_client = openai.AsyncOpenAI(api_key='api-key', base_url='https://api.moonshot.cn/v1')
    provider = MoonshotAIProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert provider.base_url == 'https://api.moonshot.cn/v1/'


def test_moonshotai_provider_custom_base_url(env: TestEnv) -> None:
    env.set('MOONSHOTAI_BASE_URL', 'https://api.moonshot.ai/v1')
    provider = MoonshotAIProvider(api_key='api-key', base_url='https://api.moonshot.cn/v1')
    assert provider.base_url == 'https://api.moonshot.cn/v1'
    assert str(provider.client.base_url) == 'https://api.moonshot.cn/v1/'


@pytest.mark.parametrize('base_url', ['https://api.moonshot.cn/v1', ''])
def test_moonshotai_provider_env_base_url(base_url: str, env: TestEnv) -> None:
    """Test model-string endpoint resolution without a live API request."""
    env.set('MOONSHOTAI_BASE_URL', base_url)
    env.set('MOONSHOTAI_API_KEY', 'api-key')
    agent = Agent('moonshotai:kimi-k3')
    assert isinstance(agent.model, OpenAIChatModel)
    expected_url = base_url or 'https://api.moonshot.ai/v1'
    assert agent.model.base_url == f'{expected_url}/'
    assert agent.model.client.api_key == 'api-key'


def test_moonshotai_provider_openai_client_and_base_url_are_exclusive() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    with pytest.raises(UserError, match='Cannot provide both `openai_client` and `base_url`'):
        MoonshotAIProvider(openai_client=openai_client, base_url='https://api.moonshot.cn/v1')  # type: ignore[call-overload]


def test_moonshotai_provider_creates_http_client() -> None:
    """Test MoonshotAI provider creates its own HTTP client."""
    provider = MoonshotAIProvider(api_key='api-key')
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_moonshotai_model_profile():
    provider = MoonshotAIProvider(api_key='api-key')
    model = OpenAIChatModel('kimi-k2-0711-preview', provider=provider)
    assert isinstance(model.profile, dict)
    assert model.profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
    assert model.profile.get('supports_forced_tool_choice', True) is False
    assert model.profile.get('supports_json_object_output', False) is True


def test_moonshotai_model_profile_thinking():
    # Unit (not VCR): these pin the profile flags resolved from the model id at model-build time, which
    # the cassette doesn't exercise — it records the wire round-trip, not the resolved profile, and our
    # matchers aren't body-sensitive, so a regression flipping a model's reasoning flags could still
    # replay an existing recording green. A direct profile assertion is what catches it.
    provider = MoonshotAIProvider(api_key='api-key')

    # Kimi K2.5 and K2.6 reason by default and turn it off with `thinking={'type': 'disabled'}`.
    for model_name in ('kimi-k2.5', 'kimi-k2.6'):
        profile = provider.model_profile(model_name)
        assert profile is not None
        assert profile.get('supports_thinking') is True
        assert profile.get('thinking_always_enabled') is False
        assert profile.get('thinking_enabled_by_default') is True
        assert profile.get('openai_chat_thinking_false_body') == {'thinking': {'type': 'disabled'}}
        assert profile.get('openai_chat_thinking_field') == 'reasoning_content'
        assert profile.get('openai_chat_send_back_thinking_parts') == 'field'
        assert profile.get('openai_chat_supports_max_reasoning_effort') is (model_name != 'kimi-k2.5')

    # Kimi K3 and K2.7 Code always reason, since Moonshot rejects `reasoning_effort='none'` and they have no off switch.
    for model_name in (
        'kimi-k2.7-code',
        'kimi-k2.7-code-highspeed',
        'kimi-k3',
        'kimi-k3.1-pro',
        'kimi-k3.1-pro-highspeed',
    ):
        profile = provider.model_profile(model_name)
        assert profile is not None
        assert profile.get('supports_thinking') is True
        assert profile.get('thinking_always_enabled') is True
        assert profile.get('openai_chat_thinking_field') == 'reasoning_content'
        assert profile.get('openai_chat_send_back_thinking_parts') == 'field'
        assert profile.get('openai_chat_supports_max_reasoning_effort') is True

    # Kimi K3 and K3.1 Pro take `reasoning_effort` `low`, `high` or `max`, so `thinking=False` sends `low`.
    for model_name in ('kimi-k3', 'kimi-k3.1-pro', 'kimi-k3.1-pro-highspeed'):
        profile = provider.model_profile(model_name)
        assert profile is not None
        assert profile.get('openai_chat_thinking_false_body') == {'reasoning_effort': 'low'}

    # Instruct/base models don't reason, so thinking stays off.
    for non_reasoning_model in ('moonshot-v1-8k', 'moonshot-v1-auto', 'kimi-k2-0711-preview', 'kimi-latest'):
        profile = provider.model_profile(non_reasoning_model)
        assert profile is not None
        assert profile.get('supports_thinking', False) is False
        assert profile.get('thinking_always_enabled', False) is False
        assert 'openai_chat_thinking_false_body' not in profile
        assert profile.get('openai_chat_thinking_field') == 'reasoning_content'


_COMPLETION = {
    'id': 'chatcmpl-1',
    'object': 'chat.completion',
    'created': 0,
    'model': 'kimi',
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': '4'}, 'finish_reason': 'stop'}],
    'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
}


async def _request_body(model_name: str, model_settings: ModelSettings) -> dict[str, Any]:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        bodies.append(json.loads(request.content))
        return httpx2.Response(200, json=_COMPLETION)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as http_client:
        provider = MoonshotAIProvider(api_key='api-key', http_client=http_client)
        agent = Agent(OpenAIChatModel(model_name, provider=provider), model_settings=model_settings)
        await agent.run('What is 2+2?')
    [body] = bodies
    return body


@pytest.mark.parametrize(
    ('model_name', 'model_settings', 'expected_fields'),
    [
        # Kimi K2.5 and K2.6 turn thinking off with a `thinking` body object; `reasoning_effort='none'` is rejected
        # (https://github.com/pydantic/pydantic-ai/pull/6208).
        pytest.param('kimi-k2.6', ModelSettings(thinking=False), {'thinking': {'type': 'disabled'}}, id='k2.6-false'),
        pytest.param('kimi-k2.5', ModelSettings(thinking=False), {'thinking': {'type': 'disabled'}}, id='k2.5-false'),
        # A `thinking` object the caller sets in `extra_body` wins.
        pytest.param(
            'kimi-k2.6',
            ModelSettings(thinking=False, extra_body={'thinking': {'type': 'enabled'}}),
            {'thinking': {'type': 'enabled'}},
            id='k2.6-false-extra-body-wins',
        ),
        # Kimi K3 always reasons, at efforts `low`, `high` and `max`
        # (https://platform.kimi.ai/docs/guide/use-reasoning-effort), so `thinking=False` sends `low`.
        pytest.param('kimi-k3', ModelSettings(thinking=False), {'reasoning_effort': 'low'}, id='k3-false'),
        # Kimi K2.7 Code always reasons and has no effort knob, so `thinking=False` sends nothing.
        pytest.param('kimi-k2.7-code', ModelSettings(thinking=False), {}, id='k2.7-code-false'),
        # Other levels are sent as on `main`.
        pytest.param('kimi-k3', ModelSettings(thinking=True), {'reasoning_effort': 'medium'}, id='k3-true'),
        pytest.param('kimi-k3', ModelSettings(thinking='max'), {'reasoning_effort': 'max'}, id='k3-max'),
        pytest.param('kimi-k2.6', ModelSettings(thinking='xhigh'), {'reasoning_effort': 'xhigh'}, id='k2.6-xhigh'),
        # An explicit `openai_reasoning_effort` wins over `thinking=False`, and adds no `thinking` object.
        pytest.param(
            'kimi-k2.6',
            OpenAIChatModelSettings(thinking=False, openai_reasoning_effort='low'),
            {'reasoning_effort': 'low'},
            id='k2.6-false-explicit-effort',
        ),
    ],
)
async def test_moonshotai_thinking_request_fields(
    allow_model_requests: None, model_name: str, model_settings: ModelSettings, expected_fields: dict[str, Any]
):
    """Mocked transport, not VCR: there's no Moonshot API key at hand to record with. The wire shapes come from
    Kimi's docs and the live checks in #6208."""
    body = await _request_body(model_name, model_settings)
    assert {key: body[key] for key in ('thinking', 'reasoning_effort') if key in body} == expected_fields
