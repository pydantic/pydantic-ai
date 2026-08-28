"""Tests for `SynthoraiProvider`.

Synthorai is a gateway, so the interesting behaviour is which model profile it resolves for
a given id. Its ids carry no vendor prefix - they are flat names such as `claude-opus-5` -
so the family is matched on a leading substring rather than split on '/' the way
vendor-prefixed providers do, and the prefix table is what these tests pin.
"""

from __future__ import annotations as _annotations

import re

import pytest
from pytest_mock import MockerFixture

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.anthropic import anthropic_model_profile
from pydantic_ai.profiles.deepseek import deepseek_model_profile
from pydantic_ai.profiles.google import google_model_profile
from pydantic_ai.profiles.moonshotai import moonshotai_model_profile
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.profiles.qwen import qwen_model_profile
from pydantic_ai.profiles.zai import zai_model_profile

from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.synthorai import SynthoraiProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_synthorai_provider():
    provider = SynthoraiProvider(api_key='api-key')
    assert provider.name == 'synthorai'
    assert provider.base_url == 'https://synthorai.io/v1'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'api-key'


def test_synthorai_provider_need_api_key(env: TestEnv) -> None:
    env.remove('SYNTHORAI_API_KEY')
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `SYNTHORAI_API_KEY` environment variable or pass it via '
            '`SynthoraiProvider(api_key=...)` to use the Synthorai provider.'
        ),
    ):
        SynthoraiProvider()


def test_synthorai_pass_openai_client() -> None:
    openai_client = openai.AsyncOpenAI(api_key='api-key')
    provider = SynthoraiProvider(openai_client=openai_client)
    assert provider.client == openai_client


def test_synthorai_provider_model_profile(mocker: MockerFixture):
    provider = SynthoraiProvider(api_key='api-key')

    ns = 'pydantic_ai.providers.synthorai'
    anthropic_mock = mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile)
    deepseek_mock = mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile)
    google_mock = mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile)
    moonshotai_mock = mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile)
    qwen_mock = mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile)
    zai_mock = mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile)

    # The whole id is passed through, lowercased - there is no vendor segment to strip.
    profile = provider.model_profile('claude-opus-5')
    anthropic_mock.assert_called_with('claude-opus-5')
    assert profile is not None

    profile = provider.model_profile('deepseek-v4-pro')
    deepseek_mock.assert_called_with('deepseek-v4-pro')
    assert profile is not None

    profile = provider.model_profile('gemini-3.5-flash')
    google_mock.assert_called_with('gemini-3.5-flash')
    assert profile is not None

    profile = provider.model_profile('kimi-k3')
    moonshotai_mock.assert_called_with('kimi-k3')
    assert profile is not None

    profile = provider.model_profile('qwen3.8-max')
    qwen_mock.assert_called_with('qwen3.8-max')
    assert profile is not None

    profile = provider.model_profile('glm-5.2')
    zai_mock.assert_called_with('glm-5.2')
    assert profile is not None


def test_synthorai_provider_model_profile_is_case_insensitive(mocker: MockerFixture):
    provider = SynthoraiProvider(api_key='api-key')
    anthropic_mock = mocker.patch(
        'pydantic_ai.providers.synthorai.anthropic_model_profile', wraps=anthropic_model_profile
    )
    provider.model_profile('Claude-Opus-5')
    anthropic_mock.assert_called_with('claude-opus-5')


def test_synthorai_provider_unmapped_family_falls_back(mocker: MockerFixture):
    """Families the catalog serves that have no profile in this repo.

    minimax, hunyuan and the Seed models resolve to the OpenAI-compatible base rather than
    being mapped to an approximate profile, which would claim capabilities nobody checked.
    """
    provider = SynthoraiProvider(api_key='api-key')
    ns = 'pydantic_ai.providers.synthorai'
    mocks = [
        mocker.patch(f'{ns}.anthropic_model_profile', wraps=anthropic_model_profile),
        mocker.patch(f'{ns}.deepseek_model_profile', wraps=deepseek_model_profile),
        mocker.patch(f'{ns}.google_model_profile', wraps=google_model_profile),
        mocker.patch(f'{ns}.moonshotai_model_profile', wraps=moonshotai_model_profile),
        mocker.patch(f'{ns}.qwen_model_profile', wraps=qwen_model_profile),
        mocker.patch(f'{ns}.zai_model_profile', wraps=zai_model_profile),
    ]

    for model_name in ('minimax-m2.5', 'hunyuan-3', 'some-future-model'):
        profile = provider.model_profile(model_name)
        assert profile is not None
        assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer

    for m in mocks:
        m.assert_not_called()


def test_synthorai_provider_gpt_prefix_uses_openai_shape():
    """`gpt-` has no dedicated profile function here; it takes the OpenAI-compatible base."""
    provider = SynthoraiProvider(api_key='api-key')
    profile = provider.model_profile('gpt-5.6-sol')
    assert profile is not None
    assert profile.get('json_schema_transformer', None) == OpenAIJsonSchemaTransformer
