"""Tests for `GitHubCopilotProvider`.

Copilot is an OpenAI-compatible gateway, so the provider's whole job is credentials, base URL,
client headers, and picking a family profile from a bare model id. None of that makes a request, so
these are unit tests rather than VCR ones; the wire behavior lives in `tests/models/test_github_copilot.py`.
"""

from __future__ import annotations as _annotations

import re

import httpx2
import pytest

from pydantic_ai.exceptions import UserError
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer, google_model_profile
from pydantic_ai.profiles.openai import SAMPLING_PARAMS, OpenAIJsonSchemaTransformer

from .._inline_snapshot import snapshot
from ..conftest import TestEnv, try_import

with try_import() as imports_successful:
    import openai

    from pydantic_ai.providers.github_copilot import GitHubCopilotProvider


pytestmark = pytest.mark.skipif(not imports_successful(), reason='openai not installed')


def test_github_copilot_provider():
    provider = GitHubCopilotProvider(api_key='gho_test_token')
    assert provider.name == 'github-copilot'
    assert provider.base_url == 'https://api.githubcopilot.com'
    assert isinstance(provider.client, openai.AsyncOpenAI)
    assert provider.client.api_key == 'gho_test_token'


_COPILOT_HEADER_NAMES = frozenset(
    {'editor-version', 'copilot-integration-id', 'editor-plugin-version', 'openai-intent', 'x-github-api-version'}
)


def test_github_copilot_provider_client_headers():
    """The Copilot client headers ride on the client, so every request carries them.

    None is documented as required and none could be made required against `api.githubcopilot.com`;
    they are parity with other Copilot clients for the enterprise hosts we can't probe. The
    `User-Agent` is not among them — `test_github_copilot_sends_model_id_verbatim` asserts on the
    wire that it stays `pydantic-ai/<version>`, which Copilot accepts.
    """
    provider = GitHubCopilotProvider(api_key='gho_test_token')
    headers = provider.client.default_headers
    assert {key: headers[key] for key in headers if key in _COPILOT_HEADER_NAMES} == snapshot(
        {
            'editor-version': 'vscode/1.95.0',
            'copilot-integration-id': 'vscode-chat',
            'editor-plugin-version': 'copilot-chat/0.26.7',
            'openai-intent': 'conversation-panel',
            'x-github-api-version': '2025-04-01',
        }
    )


_API_KEY_ENV_VARS = ('GITHUB_COPILOT_API_KEY', 'GITHUB_COPILOT_API_TOKEN', 'COPILOT_GITHUB_TOKEN')


@pytest.mark.parametrize('env_var', _API_KEY_ENV_VARS)
def test_github_copilot_provider_api_key_from_env(env: TestEnv, env_var: str):
    """`GITHUB_COPILOT_API_TOKEN` and `COPILOT_GITHUB_TOKEN` are GitHub's own names for a Copilot bearer."""
    for name in _API_KEY_ENV_VARS:
        if name != env_var:
            env.remove(name)
    env.set(env_var, 'gho_from_env')

    assert GitHubCopilotProvider().client.api_key == 'gho_from_env'


def test_github_copilot_provider_ignores_general_github_tokens(env: TestEnv):
    """A token meant for the GitHub API must never be sent to Copilot, so those names aren't read."""
    for name in _API_KEY_ENV_VARS:
        env.remove(name)
    for name in ('GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_API_KEY'):
        env.set(name, 'gho_wrong_token')

    with pytest.raises(UserError, match=re.escape('Set the `GITHUB_COPILOT_API_KEY` environment variable')):
        GitHubCopilotProvider()


def test_github_copilot_provider_need_api_key(env: TestEnv):
    for name in _API_KEY_ENV_VARS:
        env.remove(name)
    with pytest.raises(
        UserError,
        match=re.escape(
            'Set the `GITHUB_COPILOT_API_KEY` environment variable or pass it via'
            ' `GitHubCopilotProvider(api_key=...)` to use the GitHub Copilot provider.'
        ),
    ):
        GitHubCopilotProvider()


_BASE_URL_ENV_VARS = ('GITHUB_COPILOT_BASE_URL', 'COPILOT_API_URL', 'GITHUB_COPILOT_API_BASE')


@pytest.mark.parametrize('env_var', _BASE_URL_ENV_VARS)
def test_github_copilot_provider_base_url_from_env(env: TestEnv, env_var: str):
    """Enterprise and proxy hosts are reached with a custom base URL, not a separate provider."""
    for name in _BASE_URL_ENV_VARS:
        if name != env_var:
            env.remove(name)
    env.set(env_var, 'https://copilot.example.com')

    assert GitHubCopilotProvider(api_key='gho_test_token').base_url == 'https://copilot.example.com'


def test_github_copilot_provider_base_url_argument():
    provider = GitHubCopilotProvider(api_key='gho_test_token', base_url='https://copilot.example.com/api')
    assert provider.base_url == 'https://copilot.example.com/api/'


def test_github_copilot_provider_pass_http_client():
    http_client = httpx2.AsyncClient()
    provider = GitHubCopilotProvider(api_key='gho_test_token', http_client=http_client)
    assert provider.client._client is http_client  # pyright: ignore[reportPrivateUsage]


def test_github_copilot_provider_pass_openai_client():
    """A caller-supplied client is used as-is, headers included; we don't inject the Copilot set."""
    openai_client = openai.AsyncOpenAI(api_key='gho_test_token', base_url='https://api.githubcopilot.com')
    provider = GitHubCopilotProvider(openai_client=openai_client)
    assert provider.client is openai_client
    assert 'editor-version' not in provider.client.default_headers


def test_github_copilot_provider_claude_profile():
    """Claude ids get the Anthropic family profile, plus the Copilot overlay on top."""
    profile = GitHubCopilotProvider.model_profile('claude-haiku-4.5')
    assert profile is not None
    assert profile.get('supports_thinking') is True
    assert profile.get('supports_json_schema_output') is True
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer
    assert profile.get('github_copilot_supports_reasoning_effort') is False
    assert profile.get('openai_chat_supports_max_completion_tokens') is True


def test_github_copilot_profile_dot_to_hyphen_is_anthropic_only():
    """The regression guard for the id-normalization rule.

    `anthropic_model_profile` matches hyphenated ids, so Copilot's dotted `claude-haiku-4.5` needs
    the rewrite to resolve at all — `supports_json_schema_output` is `False` without it. But
    `grok_model_profile` and `moonshotai_model_profile` match *dotted* ids, so applying the same
    rewrite globally would blank those two. The asserts below pin both directions.
    """
    dotted = GitHubCopilotProvider.model_profile('claude-haiku-4.5')
    hyphenated = GitHubCopilotProvider.model_profile('claude-haiku-4-5')
    assert dotted == hyphenated

    grok = GitHubCopilotProvider.model_profile('grok-4.5')
    assert grok is not None
    assert grok.get('grok_reasoning_efforts') == snapshot(frozenset({'low', 'medium', 'high'}))

    kimi = GitHubCopilotProvider.model_profile('kimi-k2.5')
    assert kimi is not None
    assert kimi.get('supports_thinking') is True


def test_github_copilot_provider_gpt_profile():
    profile = GitHubCopilotProvider.model_profile('gpt-5.4')
    assert profile is not None
    assert profile.get('supports_thinking') is True
    assert profile.get('github_copilot_supports_reasoning_effort') is True
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer


def test_github_copilot_provider_gemini_profile():
    """Copilot speaks OpenAI tools and `response_format`, not Gemini `generateContent`."""
    family = google_model_profile('gemini-3.0-pro')
    assert family is not None and family.get('json_schema_transformer') is GoogleJsonSchemaTransformer

    profile = GitHubCopilotProvider.model_profile('gemini-3.0-pro')
    assert profile is not None
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer


def test_github_copilot_provider_sampling_restriction_follows_the_family():
    """Sampling restriction comes from the family profile, never from a hardcoded id list.

    Copilot accepts `temperature` on ids that public catalogs list as sampling-restricted, so the
    only trustworthy source is the model family's own `anthropic_disallows_sampling_settings`.
    """
    restricted = GitHubCopilotProvider.model_profile('claude-opus-4.8')
    assert restricted is not None
    assert restricted.get('openai_unsupported_model_settings') == SAMPLING_PARAMS

    unrestricted = GitHubCopilotProvider.model_profile('claude-haiku-4.5')
    assert unrestricted is not None
    assert unrestricted.get('openai_unsupported_model_settings') is None


def test_github_copilot_provider_unknown_model_profile():
    """An id from no known family gets the OpenAI-compatible fallback and no claimed capabilities."""
    profile = GitHubCopilotProvider.model_profile('some-future-copilot-model')
    assert profile is not None
    assert profile.get('json_schema_transformer') is OpenAIJsonSchemaTransformer
    assert profile.get('supports_thinking') is None


def test_github_copilot_provider_strips_copilot_namespace_for_profile_lookup():
    """Other Copilot clients namespace ids as `copilot/<id>`; a user who passes one still gets a profile.

    Only the lookup strips it — `GitHubCopilotModel` sends the id exactly as given.
    """
    assert GitHubCopilotProvider.model_profile('copilot/claude-haiku-4.5') == GitHubCopilotProvider.model_profile(
        'claude-haiku-4.5'
    )
