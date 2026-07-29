import re

import httpx
import pytest
from pytest_mock import MockerFixture

from pydantic_ai.agent import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.amazon import amazon_model_profile
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.harmony import harmony_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile

from .._inline_snapshot import snapshot
from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.heroku import HerokuProvider

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='openai not installed'),
    pytest.mark.vcr,
    pytest.mark.anyio,
]


def test_heroku_provider():
    provider = HerokuProvider(api_key='api-key')
    assert provider.name == 'heroku'
    assert provider.base_url == 'https://us.inference.heroku.com/v1/'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


@pytest.mark.parametrize(
    'base_url',
    [
        'https://us.inference.heroku.com',
        'https://us.inference.heroku.com/',
        'https://us.inference.heroku.com/v1',
        'https://us.inference.heroku.com/v1/',
    ],
)
def test_heroku_provider_normalizes_base_url(base_url: str):
    provider = HerokuProvider(api_key='api-key', base_url=base_url)
    assert provider.base_url == 'https://us.inference.heroku.com/v1/'


def test_heroku_provider_need_api_key(env: TestEnv) -> None:
    env.remove('HEROKU_INFERENCE_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `HEROKU_INFERENCE_KEY` environment variable or pass it via `HerokuProvider(api_key=...)`'
            ' to use the Heroku provider.'
        ),
    ):
        HerokuProvider()


def test_heroku_provider_pass_http_client() -> None:
    http_client = httpx.AsyncClient()
    provider = HerokuProvider(http_client=http_client, api_key='api-key')
    assert provider.client._client == http_client  # type: ignore[reportPrivateUsage]


def test_heroku_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = HerokuProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_heroku_model_profile():
    provider = HerokuProvider(api_key='api-key')
    model = OpenAIChatModel('claude-3-7-sonnet', provider=provider)
    assert isinstance(model.profile, dict)
    assert model.profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer


def test_heroku_model_profile_routes_thinking_capable_families():
    """Heroku serves reasoning-capable models under bare names; their family profiles must be applied.

    Before this routing, `model_profile` returned a bare `OpenAIModelProfile` for every model, so
    `supports_thinking` defaulted to `False`/unset and unified `thinking` settings were silently
    dropped for Claude/DeepSeek reasoning models Heroku hosts.
    """
    provider = HerokuProvider(api_key='api-key')

    # Anthropic-family models served by Heroku gain thinking support via anthropic_model_profile.
    for model_name in ('claude-3-7-sonnet', 'claude-4-5-sonnet', 'claude-opus-4-5'):
        profile = provider.model_profile(model_name)
        assert profile is not None
        assert profile.get('supports_thinking') is True, model_name
        # OpenAI-compatible base is preserved.
        assert profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer, model_name

    # DeepSeek-R1 reasoning models also gain thinking support.
    deepseek_profile = provider.model_profile('deepseek-r1')
    assert deepseek_profile is not None
    assert deepseek_profile.get('supports_thinking') is True

    # Unknown / unmapped models fall back to the OpenAI-compatible base unchanged.
    fallback = provider.model_profile('some-unknown-model')
    assert fallback is not None
    assert fallback.get('json_schema_transformer') == OpenAIJsonSchemaTransformer
    assert fallback.get('supports_thinking') is None


@pytest.mark.parametrize('model_name', ['glm-4-7', 'glm-4-7-flash'])
def test_heroku_glm_routes_to_zai_profile(model_name: str):
    """GLM is Z.AI (Zhipu), not Moonshot, and Heroku hyphenates the minor version.

    Two things had to line up: the `'glm'` prefix was mapped to `moonshotai_model_profile`, and even
    the right profile wouldn't have matched, because `zai_model_profile` keys off Z.AI's dotted ids
    (`glm-4.7`) while Heroku serves `glm-4-7`. GLM-4.7 and GLM-4.7-Flash both support thinking per
    the Z.AI docs, and `Model.prepare_request` silently discards `ModelSettings(thinking=...)` when
    the profile doesn't advertise support.
    """
    provider = HerokuProvider(api_key='api-key')

    profile = provider.model_profile(model_name)
    assert profile is not None
    assert profile.get('supports_thinking') is True
    # `zai_supports_reasoning_effort` is GLM-5.2+ only, so it must resolve (not be absent) and be False.
    assert profile.get('zai_supports_reasoning_effort') is False
    # `ignore_streamed_leading_whitespace` is a Kimi trait that came along with the wrong routing.
    assert profile.get('ignore_streamed_leading_whitespace') is None
    assert profile.get('json_schema_transformer') == OpenAIJsonSchemaTransformer


@pytest.mark.parametrize(
    'model_name',
    [
        # The GLM minor-version restoration must not touch any other family Heroku serves.
        'claude-4-5-sonnet',
        'claude-opus-4-6',
        'nova-2-lite',
        'qwen3-235b',
        'gpt-oss-120b',
        'deepseek-v3-2',
    ],
)
def test_heroku_non_glm_names_are_passed_through(model_name: str, mocker: MockerFixture):
    """Only `glm-*` names get the dot restored; everything else reaches its profile verbatim."""
    ns = 'pydantic_ai.providers.heroku'
    mocks = {
        'claude': mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile),
        'nova': mocker.patch(f'{ns}.amazon_model_profile', wraps=amazon_model_profile),
        'qwen': mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile),
        'gpt-oss': mocker.patch(f'{ns}.harmony_model_profile', wraps=harmony_model_profile),
        'deepseek': mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile),
    }

    HerokuProvider(api_key='api-key').model_profile(model_name)

    called = [m for m in mocks.values() if m.call_args is not None]
    assert len(called) == 1
    called[0].assert_called_with(model_name)


async def test_heroku_model_provider_claude_3_7_sonnet(allow_model_requests: None, heroku_inference_key: str):
    provider = HerokuProvider(api_key=heroku_inference_key)
    m = OpenAIChatModel('claude-3-7-sonnet', provider=provider)
    agent = Agent(m)

    result = await agent.run('What is the capital of France?')
    assert result.output == snapshot(
        "The capital of France is Paris. It's not only the political capital but also a major cultural and economic hub in Europe, known for landmarks like the Eiffel Tower, the Louvre Museum, and Notre-Dame Cathedral."
    )
