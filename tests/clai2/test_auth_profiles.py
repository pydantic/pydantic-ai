"""Auth profiles (`PROVIDER@PROFILE:MODEL`, `/login NAME@PROFILE`) and fallback chains (`chain:NAME`)."""

import io
import json
import time
from pathlib import Path

import pytest
from pydantic import SecretStr
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, override_allow_model_requests
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.models.openai_codex import OpenAICodexModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.github_copilot import GitHubCopilotCredentials
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexOAuthFlow
from pydantic_clai2 import chat
from pydantic_clai2._app import _ModelResolver  # pyright: ignore[reportPrivateUsage]
from pydantic_clai2.auth import CodexAuth, CodexCredentials, login_command
from pydantic_clai2.cli.command_context import CommandContext
from pydantic_clai2.config import Settings
from pydantic_clai2.config.api_keys import KeyReference, key_users, save_key, save_key_connection
from pydantic_clai2.config.credential_store import (
    credentials_path,
    load_codex_credentials,
    profile_accounts,
    save_codex_credentials,
)
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models import github_copilot, key_profiles, openrouter, vllm
from pydantic_clai2.models.chains import settings_model
from pydantic_clai2.models.model_catalog import check_installed
from pydantic_clai2.models.model_options import model_options, validate_model_options
from pydantic_clai2.models.model_settings import ModelSettingsForm
from pydantic_clai2.models.profiles import ModelRef, base_model, parse_model, provider_of, with_profile
from pydantic_clai2.plugins import ModelProvider, PluginLogin
from pydantic_clai2.ui.menus.model_menu import ModelSettingsSource, model_settings_summary
from tests.clai2.test_app_edges import inputs
from tests.clai2.test_auth import fake_browser, never_pasted

CREDENTIALS = OpenAICodexCredentials(access_token='work-access', refresh_token='work-refresh', account_id='work')


def resolver(tmp_path: Path, providers: dict[str, ModelProvider] | None = None) -> _ModelResolver:
    return _ModelResolver(
        console=Console(file=io.StringIO()),
        plugins=lambda: providers or {},
        store=SettingsStore(tmp_path / 'config.db'),
    )


def test_model_names_split_into_provider_profile_and_name() -> None:
    assert parse_model('openai-codex@work:gpt-6') == ModelRef(provider='openai-codex', profile='work', name='gpt-6')
    assert parse_model('openai:gpt-5').account == 'openai'
    assert parse_model('openai@side_2:gpt-5').account == 'openai@side_2'
    # Only the prefix names a profile; Vertex model IDs keep their own `@`.
    assert parse_model('google-vertex:claude@20240620') == ModelRef(
        provider='google-vertex', profile=None, name='claude@20240620'
    )
    assert parse_model('test') == ModelRef(provider='', profile=None, name='test')
    assert provider_of('openai-codex@work:gpt-6') == 'openai-codex'
    assert base_model('openai-codex@work:gpt-6') == 'openai-codex:gpt-6'
    assert base_model('test') == 'test'
    assert with_profile('claude-code:opus', 'work') == 'claude-code@work:opus'
    assert with_profile('claude-code:opus', None) == 'claude-code:opus'


@pytest.mark.parametrize('model', ['openai@:gpt-5', 'openai@Work:gpt-5', 'openai@a.b:gpt-5', 'openai@-x:gpt-5'])
def test_malformed_profiles_are_rejected(model: str) -> None:
    with pytest.raises(ValueError, match=r'Profile .* must be 1 to 32 lowercase'):
        parse_model(model)


async def test_the_default_profile_names_the_account_without_a_profile() -> None:
    assert parse_model('openai-codex@default:gpt-6') == ModelRef(
        provider='openai-codex', profile='default', name='gpt-6'
    )

    async def default_only() -> str:
        return 'Signed in to the default account.'

    async def profile_only(profile: str) -> str:  # pragma: no cover -- @default is not a profile
        raise AssertionError(profile)

    plugins = {'claude': PluginLogin(name='claude', handler=default_only, profile_handler=profile_only)}
    codex = CodexAuth(Console(file=io.StringIO()))
    assert await login_command(['claude@default'], codex=codex, plugins=plugins) == 'Signed in to the default account.'


async def test_codex_profiles_sign_in_and_run_on_separate_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        return CREDENTIALS

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('webbrowser.open', fake_browser)
    output = io.StringIO()
    auth = CodexAuth(Console(file=output), read_line=never_pasted)
    message = await login_command(['codex@work'], codex=auth)
    assert 'Codex connected as openai-codex@work; use openai-codex@work:MODEL.' in message
    assert 'ChatGPT/Codex for profile work' in output.getvalue()
    assert load_codex_credentials() is None  # the default account is untouched
    assert await CodexCredentials(account='openai-codex@work').load() == CREDENTIALS
    with pytest.raises(UserError, match='Run /login openai-codex@home'):
        await CodexCredentials(account='openai-codex@home').load()

    default, work = auth.model('openai-codex:gpt-6'), auth.model('openai-codex@work:gpt-6')
    assert default.model_name == work.model_name == 'gpt-6'
    assert default.provider is auth.provider
    assert work.provider is not default.provider
    assert auth.model('openai-codex@work:gpt-5').provider is work.provider
    for args in (['openai-codex', 'extra'], ['github-copilot@work']):
        with pytest.raises(ValueError, match=r'Usage: /login openai-codex\[@PROFILE\]'):
            await auth.login(args)


async def test_a_plugin_login_signs_in_profiles_when_it_offers_them(tmp_path: Path) -> None:
    profiles: list[str] = []

    async def default() -> str:
        return 'Signed in.'

    async def for_profile(profile: str) -> str:
        profiles.append(profile)
        return f'Signed in as {profile}.'

    store = SettingsStore(tmp_path / 'config.db')
    codex = CodexAuth(Console(file=io.StringIO()))
    plugins = {
        'claude': PluginLogin(
            name='claude', handler=default, profile_handler=for_profile, models=('claude-code:opus',)
        ),
        'single': PluginLogin(name='single', handler=default),
    }
    assert await login_command(['claude@work'], codex=codex, plugins=plugins, store=store) == 'Signed in as work.'
    assert profiles == ['work']
    assert store.models() == ['claude-code@work:opus']
    # Without a profile, the plugin's own sign-in runs and saves the plain model name.
    assert await login_command(['claude'], codex=codex, plugins=plugins, store=store) == 'Signed in.'
    assert profiles == ['work']
    assert store.models() == ['claude-code:opus', 'claude-code@work:opus']
    with pytest.raises(ValueError, match='The single sign-in does not support profiles'):
        await login_command(['single@work'], codex=codex, plugins=plugins, store=store)
    with pytest.raises(ValueError, match='Profile'):
        await login_command(['claude@Bad'], codex=codex, plugins=plugins)


@pytest.mark.parametrize('answer', ['typed', 'saved', 'cancel', 'empty'])
async def test_a_core_provider_profile_saves_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    save_key(name='OPENAI_WORK', value='sk-saved')
    replies: dict[str, str | KeyReference | None] = {
        'typed': '  sk-typed  ',
        'saved': KeyReference(name='OPENAI_WORK'),
        'cancel': None,
        'empty': ' ',
    }
    labels: list[str] = []

    async def prompt_api_key(*, prompt: object, label: str) -> str | KeyReference | None:
        labels.append(label)
        return replies[answer]

    monkeypatch.setattr(key_profiles, 'prompt_api_key', prompt_api_key)
    codex = CodexAuth(Console(file=io.StringIO()))
    if answer == 'empty':
        with pytest.raises(ValueError, match='An API key is required for openai@work'):
            await login_command(['openai@work'], codex=codex)
        return
    message = await login_command(['openai@work'], codex=codex)
    assert labels == ['openai API key for openai@work: ']
    if answer == 'cancel':
        assert message == 'Sign-in cancelled.'
        assert load_codex_credentials(account='openai@work') is None
        return
    assert message == 'openai@work connected, saved in the OS credential store. Use openai@work:MODEL.'
    assert 'openai@work' in profile_accounts()
    model = key_profiles.model('openai@work:gpt-5')
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == 'gpt-5'
    assert model.client.api_key == ('sk-typed' if answer == 'typed' else 'sk-saved')
    assert key_users(name='OPENAI_WORK') == ([] if answer == 'typed' else ['openai@work'])


async def test_a_key_profile_for_a_provider_missing_other_settings_is_not_saved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def prompt_api_key(*, prompt: object, label: str) -> str:
        return 'sk-typed'

    monkeypatch.setattr(key_profiles, 'prompt_api_key', prompt_api_key)
    monkeypatch.delenv('OLLAMA_BASE_URL', raising=False)
    with pytest.raises(UserError, match=r'ollama@work was not saved: Set the `OLLAMA_BASE_URL`'):
        await login_command(['ollama@work'], codex=CodexAuth(Console(file=io.StringIO())))
    assert load_codex_credentials(account='ollama@work') is None
    # With the server configured, the same sign-in is saved and runs.
    monkeypatch.setenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
    assert (await login_command(['ollama@work'], codex=CodexAuth(Console(file=io.StringIO())))).startswith(
        'ollama@work connected'
    )
    assert key_profiles.model('ollama@work:llama3').model_name == 'llama3'


def test_a_gateway_profile_keeps_the_gateway_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('PYDANTIC_AI_GATEWAY_API_KEY', raising=False)
    save_codex_credentials(account='gateway/openai@work', value=json.dumps({'token': 'pylf_v1_us_work'}))
    model = key_profiles.model('gateway/openai@work:gpt-5')
    assert isinstance(model, OpenAIResponsesModel)
    assert model.client.api_key == 'pylf_v1_us_work'
    assert 'api.openai.com' not in str(model.client.base_url)
    assert 'gateway' in str(model.client.base_url)


async def test_core_providers_need_a_profile_and_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    codex = CodexAuth(Console(file=io.StringIO()))
    with pytest.raises(ValueError, match=r'^Usage: /login'):
        await login_command(['openai'], codex=codex)  # the default profile keeps core's environment variables
    for name in ('snowflake@work', 'grok@work'):  # no API key, or no such provider
        with pytest.raises(ValueError, match=r'^Usage: /login'):
            await login_command([name], codex=codex)
    assert key_profiles.supports('openrouter', profile=None)
    with pytest.raises(UserError, match='snowflake has no profiles; only providers that sign in with an API key'):
        key_profiles.model('snowflake@work:x')
    with pytest.raises(UserError, match='snowflake has no profiles'):
        key_profiles.build_provider('snowflake', key='sk-x')
    with pytest.raises(UserError, match=r'openai@work is not connected. Run /login openai@work.'):
        key_profiles.model('openai@work:gpt-5')
    # An empty or missing key would let the provider fall back to the environment's key.
    for stored in ('not json', '{}', '{"token": ""}'):
        save_codex_credentials(account='openai@work', value=stored)
        with pytest.raises(UserError, match='Stored openai@work credentials are invalid'):
            key_profiles.model('openai@work:gpt-5')
    with pytest.raises(UserError, match=r'no longer exists\. Select a saved key again through /login openai@work'):
        save_key_connection(account='openai@work', token=KeyReference(name='GONE'), value='{}')


@pytest.mark.parametrize('provider', ['openrouter', 'vllm'])
@pytest.mark.parametrize('cancel', [False, True])
async def test_connection_profiles_reuse_the_connection_prompt(
    monkeypatch: pytest.MonkeyPatch, provider: str, *, cancel: bool
) -> None:
    async def openrouter_prompt() -> openrouter.Connection | None:
        return None if cancel else openrouter.Connection(token=SecretStr('or-key'))

    async def vllm_prompt() -> vllm.Connection | None:
        return None if cancel else vllm.Connection(url='http://lab:8000/v1', token=SecretStr('lab-key'))

    monkeypatch.setattr(openrouter, 'prompt_connection', openrouter_prompt)
    monkeypatch.setattr(vllm, 'prompt_connection', vllm_prompt)
    message = await login_command([f'{provider}@lab'], codex=CodexAuth(Console(file=io.StringIO())))
    if cancel:
        assert message == 'Sign-in cancelled.'
        return
    assert message.startswith(f'{provider}@lab connected')
    assert load_codex_credentials(account=provider) is None
    built = openrouter.model('openrouter@lab:openai/x') if provider == 'openrouter' else vllm.model('vllm@lab:openai/x')
    assert built.model_name == 'openai/x'
    assert built.client.api_key == ('or-key' if provider == 'openrouter' else 'lab-key')


@pytest.mark.parametrize('provider', ['openrouter', 'vllm'])
def test_connection_profile_errors_name_the_profile_login(provider: str) -> None:
    build = openrouter.model if provider == 'openrouter' else vllm.model
    with pytest.raises(UserError, match=f'Connect first through /login {provider}@lab'):
        build(f'{provider}@lab:x')
    save_codex_credentials(account=f'{provider}@lab', value='not json')
    with pytest.raises(UserError, match=f'Reconfigure through /login {provider}@lab'):
        build(f'{provider}@lab:x')


def test_copilot_profiles_use_their_own_login() -> None:
    with pytest.raises(UserError, match='Run /login github-copilot@work'):
        github_copilot.token('github-copilot@work')
    connection = github_copilot.Connection(
        credentials=GitHubCopilotCredentials(access_token='work-token', token_type='bearer', scope=''),
        issued_at=time.time(),
    )
    save_codex_credentials(account='github-copilot@work', value=connection.model_dump_json())
    model = github_copilot.model('github-copilot@work:gpt-5')
    assert model.model_name == 'gpt-5'
    assert model.client.api_key == 'work-token'


async def test_copilot_profile_login_saves_to_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    class Flow:
        def __init__(self, **kwargs: object) -> None:
            pass

        async def start(self) -> object:
            return type('Authorization', (), {'verification_uri': 'https://github.com/login/device', 'user_code': 'C'})

        async def wait_for_authorization(self) -> GitHubCopilotCredentials:
            return GitHubCopilotCredentials(access_token='work-token', token_type='bearer', scope='')

    monkeypatch.setattr(github_copilot, 'GitHubCopilotOAuthFlow', Flow)
    message = await github_copilot.login(console=Console(file=io.StringIO()), account='github-copilot@work')
    assert message.startswith('GitHub login saved as github-copilot@work.')
    assert github_copilot.token('github-copilot@work') == 'work-token'
    assert load_codex_credentials(account='github-copilot') is None


def fails(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    raise ModelHTTPError(status_code=429, model_name='limited', body='usage limit reached')


async def test_a_chain_falls_back_from_a_limited_account(tmp_path: Path) -> None:
    def resolve_profile(name: str, profile: str) -> Model:
        return TestModel(custom_output_text=f'{profile} answered with {name}')

    providers = {
        'limited': ModelProvider(prefix='limited', resolve=lambda name: FunctionModel(fails)),
        'pooled': ModelProvider(prefix='pooled', resolve=echo, resolve_profile=resolve_profile),
    }
    models = resolver(tmp_path, providers)
    assert models.store is not None
    models.store.save_chain(name='pool', models=['limited:a', 'pooled@spare:b'])
    model = await models.resolve('chain:pool')
    assert isinstance(model, FallbackModel)
    result = await Agent(model).run('hello')
    assert result.output == 'spare answered with b'


def echo(name: str) -> Model:
    return TestModel(custom_output_text=name)


async def test_a_chain_does_not_hide_a_profile_that_is_not_signed_in(tmp_path: Path) -> None:
    models = resolver(tmp_path, {'pooled': ModelProvider(prefix='pooled', resolve=echo)})
    assert models.store is not None
    models.store.save_chain(name='pool', models=['openai-codex@nobody:gpt-5', 'pooled:spare'])
    model = await models.resolve('chain:pool')
    # Loading the credentials fails before any HTTP request is made.
    with override_allow_model_requests(True):
        with pytest.raises(UserError, match=r'Codex is not connected\. Run /login openai-codex@nobody\.'):
            await Agent(model).run('hello')


async def test_resolver_reports_profile_problems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    models = resolver(tmp_path, {'single': ModelProvider(prefix='single', resolve=echo)})
    with pytest.raises(UserError, match=r'No chain named missing\. Create one with /model chains\.'):
        await models.resolve('chain:missing')
    with pytest.raises(UserError, match='No chain named any'):
        await _ModelResolver(console=Console(file=io.StringIO())).resolve('chain:any')
    with pytest.raises(UserError, match="openai@BAD:x: Profile 'BAD'"):
        await models.resolve('openai@BAD:x')
    with pytest.raises(UserError, match=r'single does not support profiles\. Use single:x\.'):
        await models.resolve('single@work:x')
    model = await models.resolve('single:x')
    assert isinstance(model, TestModel) and model.custom_output_text == 'x'
    assert await models.resolve('openai:gpt-5') == 'openai:gpt-5'  # the default profile stays core's

    def keyed(name: str) -> Model:
        return TestModel(custom_output_text=f'key profile {name}')

    monkeypatch.setattr(key_profiles, 'model', keyed)
    model = await models.resolve('openai@work:gpt-5')
    assert isinstance(model, TestModel) and model.custom_output_text == 'key profile openai@work:gpt-5'


def test_chains_and_profiles_take_their_model_settings(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_chain(name='pool', models=['openai-codex@work:gpt-6-astra', 'anthropic:claude-sonnet-4-5'])
    assert settings_model(store, 'chain:pool') == 'openai-codex@work:gpt-6-astra'
    assert settings_model(store, 'chain:missing') == 'chain:missing'
    assert settings_model(store, 'openai:gpt-5') == 'openai:gpt-5'
    # A profile offers the same controls as its provider's default account.
    assert model_options(model='openai-codex@work:gpt-6-astra') == model_options(model='openai-codex:gpt-6-astra')
    assert 'anthropic_thinking_mode' in model_options(model='anthropic@work:claude-sonnet-4-5')
    with pytest.raises(ValueError, match='requires thinking for xhigh'):
        validate_model_options(
            model='anthropic@work:claude-opus-5',
            form=ModelSettingsForm(anthropic_thinking_mode='disabled', anthropic_effort='xhigh'),
        )
    check_installed('openai@work:gpt-5')

    context = CommandContext(
        settings=Settings(model='chain:pool'),
        store=store,
        clear_history=lambda: None,
        apply_setting=lambda key, settings: None,
        settings_model=lambda model: settings_model(store, model),
    )
    # The chain runs with its first model's family defaults, under its own saved overrides.
    assert (context.model_settings('chain:pool') or {}).get('openai_reasoning_effort') == 'medium'
    source = ModelSettingsSource(store, 'chain:pool', settings_as=context.settings_model('chain:pool'))
    rows = {row.key: row for row in source.rows()}
    assert rows['service_tier'].label == 'Service Tier / Fast Mode'
    assert source.current(rows['openai_reasoning_effort']) == 'medium'
    summary = model_settings_summary(store=store, model='chain:pool', settings_as=context.settings_model('chain:pool'))
    assert 'Reasoning' in summary


async def test_shell_runs_a_chain_of_codex_accounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = SettingsStore(tmp_path / 'config.db')
    store.save_chain(name='pool', models=['openai-codex:test', 'openai-codex@work:test'])
    inputs(
        monkeypatch,
        [
            '/chain pool openai-codex:test',  # chains are made in the /model picker now
            '/model chain:pool',
            '/fast',
            'hello',
            '/exit',
        ],
    )
    resolved: list[str] = []

    def model(self: CodexAuth, name: str) -> OpenAICodexModel | TestModel:
        resolved.append(name)
        return TestModel(custom_output_text=f'{name} answered')

    monkeypatch.setattr(CodexAuth, 'model', model)
    output = io.StringIO()
    await chat(
        Agent(TestModel()),
        deps=None,
        settings=Settings(model='test', session_namer=False),
        store=store,
        console=Console(file=output, width=200),
    )
    text = output.getvalue()
    assert 'Usage: /model chains. Create, edit, rename, and delete fallback chains in its picker.' in text
    assert 'Fast mode on for chain:pool' in text
    assert 'openai-codex:test answered' in text
    assert resolved == ['openai-codex:test', 'openai-codex@work:test']
    assert store.model_settings('chain:pool') == {'service_tier': 'priority'}


def test_key_users_cover_profile_connections() -> None:
    save_key(name='SHARED', value='sk-shared')
    value = json.dumps({'token': {'name': 'SHARED'}})
    save_key_connection(account='anthropic@work', token=KeyReference(name='SHARED'), value=value)
    save_codex_credentials(account='openai-codex@work', value=CREDENTIALS_JSON)
    save_key_connection(account='gateway/openai@work', token=KeyReference(name='SHARED'), value=value)
    # A gateway profile is one file beside the others, so it is found like them.
    assert credentials_path(account='gateway/openai@work').parent == credentials_path(account='openai').parent
    assert profile_accounts() == ['anthropic@work', 'gateway/openai@work', 'openai-codex@work']
    assert key_users(name='SHARED') == ['anthropic@work', 'gateway/openai@work']


CREDENTIALS_JSON = json.dumps({'access_token': 'a', 'refresh_token': 'r', 'account_id': 'x'})


def test_plugins_cannot_take_the_chain_prefix() -> None:
    with pytest.raises(ValueError, match='CLAI already runs'):
        ModelProvider(prefix='chain', resolve=echo)
