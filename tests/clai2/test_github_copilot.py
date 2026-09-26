"""Copilot login orchestration, credential isolation, and model selection."""

import io
import time
from dataclasses import replace
from pathlib import Path

import anyio
import httpx2
import keyring
import pytest
from cassetter import use_cassette
from keyring.errors import NoKeyringError
from menu_script import Script, make_context, pick, typed
from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.github_copilot import (
    GitHubCopilotCredentials,
    GitHubCopilotDeviceAuthorization,
    GitHubCopilotOAuthFlow,
)
from rich.console import Console
from termflow.tui import MenuItem  # pyright: ignore[reportMissingTypeStubs]
from termflow.tui.menu import MenuResult  # pyright: ignore[reportMissingTypeStubs]
from test_app_edges import inputs

from pydantic_clai2 import chat, github_copilot
from pydantic_clai2.credential_store import credentials_path, load_codex_credentials, save_codex_credentials
from pydantic_clai2.model_catalog import github_copilot_models
from pydantic_clai2.model_menu import ModelMenu, open_add_model_menu
from pydantic_clai2.settings_store import SettingsStore

CREDENTIALS = GitHubCopilotCredentials(access_token='fake-access', token_type='bearer', scope='')


@pytest.fixture(autouse=True)
def isolate_copilot(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        'GITHUB_COPILOT_CLIENT_ID',
        'GITHUB_COPILOT_API_KEY',
        'GITHUB_COPILOT_API_TOKEN',
        'COPILOT_GITHUB_TOKEN',
        'GITHUB_COPILOT_BASE_URL',
        'COPILOT_API_URL',
        'GITHUB_COPILOT_API_BASE',
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def device_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    async def start(self: GitHubCopilotOAuthFlow) -> GitHubCopilotDeviceAuthorization:
        return GitHubCopilotDeviceAuthorization(
            device_code='private-device-code',
            user_code='ABCD-EFGH',
            verification_uri='https://github.com/login/device',
            expires_in=900,
        )

    async def authorize(self: GitHubCopilotOAuthFlow) -> GitHubCopilotCredentials:
        return CREDENTIALS

    monkeypatch.setattr(GitHubCopilotOAuthFlow, 'start', start)
    monkeypatch.setattr(GitHubCopilotOAuthFlow, 'wait_for_authorization', authorize)


@pytest.mark.parametrize('fallback', [False, True])
async def test_login_and_model(device_flow: None, monkeypatch: pytest.MonkeyPatch, *, fallback: bool) -> None:
    if fallback:

        def no_keyring(service: str, account: str) -> None:
            raise NoKeyringError

        monkeypatch.setattr(keyring, 'get_password', no_keyring)
    else:
        save_codex_credentials(value='existing-codex-login')
    output = io.StringIO()
    result = await github_copilot.login(console=Console(file=output))
    assert 'GitHub login saved' in result
    assert ('plaintext' in result) is fallback
    assert 'ABCD-EFGH' in output.getvalue()
    assert 'https://github.com/login/device' in output.getvalue()
    assert 'fake-access' not in output.getvalue() + result
    assert 'private-device-code' not in output.getvalue()
    assert github_copilot.token() == CREDENTIALS.access_token
    model = github_copilot.model('github-copilot:claude-haiku-4.5')
    assert model.model_name == 'claude-haiku-4.5'
    assert model.system == 'github-copilot'
    assert model.client.api_key == CREDENTIALS.access_token
    assert model.client.default_headers['copilot-integration-id'] == 'vscode-chat'
    async with model:
        pass
    if fallback:
        assert credentials_path(account='github-copilot').stat().st_mode & 0o777 == 0o600
    else:
        assert load_codex_credentials() == 'existing-codex-login'


@pytest.mark.parametrize('client_id', [None, '', '   ', ' custom-client '])
async def test_client_id_default_and_override(
    device_flow: None, monkeypatch: pytest.MonkeyPatch, client_id: str | None
) -> None:
    if client_id is not None:
        monkeypatch.setenv('GITHUB_COPILOT_CLIENT_ID', client_id)
    configured: list[tuple[str, str]] = []

    def flow(*, client_id: str, scope: str) -> GitHubCopilotOAuthFlow:
        configured.append((client_id, scope))
        return GitHubCopilotOAuthFlow(client_id=client_id, scope=scope)

    monkeypatch.setattr(github_copilot, 'GitHubCopilotOAuthFlow', flow)
    assert 'GitHub login saved' in await github_copilot.login(console=Console(file=io.StringIO()))
    assert configured == [('custom-client' if client_id == ' custom-client ' else 'Iv1.b507a08c87ecfe98', 'read:user')]


@pytest.mark.vcr
async def test_default_client_starts_device_flow(monkeypatch: pytest.MonkeyPatch) -> None:
    async def stop_before_approval(self: GitHubCopilotOAuthFlow) -> GitHubCopilotCredentials:
        raise UserError('Stop before approving the device.')

    monkeypatch.setattr(GitHubCopilotOAuthFlow, 'wait_for_authorization', stop_before_approval)
    output = io.StringIO()
    with (
        use_cassette(
            Path(__file__).parent / 'cassettes/github_copilot_device_start.yaml',
            record_mode='none',
            match_on=['method', 'uri', 'body'],
            body_scrub_patterns=['device_code', 'user_code'],
        ),
        pytest.raises(UserError, match='Stop before approving the device'),
    ):
        await github_copilot.login(console=Console(file=output))
    assert 'https://github.com/login/device' in output.getvalue()
    assert 'Enter code: [FILTERED]' in output.getvalue()
    assert load_codex_credentials(account='github-copilot') is None


async def test_failed_login_preserves_credentials(device_flow: None, monkeypatch: pytest.MonkeyPatch) -> None:
    async def denied(self: GitHubCopilotOAuthFlow) -> GitHubCopilotCredentials:
        raise UserError('access_denied')

    monkeypatch.setattr(GitHubCopilotOAuthFlow, 'wait_for_authorization', denied)
    save_codex_credentials(account='github-copilot', value='previous')
    with pytest.raises(UserError, match='access_denied'):
        await github_copilot.login(console=Console(file=io.StringIO()))
    assert load_codex_credentials(account='github-copilot') == 'previous'


async def test_cancel_stops_polling(device_flow: None, monkeypatch: pytest.MonkeyPatch) -> None:
    started, stopped = anyio.Event(), anyio.Event()

    async def waiting(self: GitHubCopilotOAuthFlow) -> GitHubCopilotCredentials:
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            stopped.set()
        raise AssertionError('unreachable')

    monkeypatch.setattr(GitHubCopilotOAuthFlow, 'wait_for_authorization', waiting)
    save_codex_credentials(account='github-copilot', value='previous')

    async def login() -> None:
        await github_copilot.login(console=Console(file=io.StringIO()))

    with anyio.fail_after(10):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(login)
            await started.wait()
            tasks.cancel_scope.cancel()
    assert stopped.is_set()
    assert load_codex_credentials(account='github-copilot') == 'previous'


@pytest.mark.parametrize('variable', ['GITHUB_COPILOT_API_KEY', 'GITHUB_COPILOT_API_TOKEN', 'COPILOT_GITHUB_TOKEN'])
def test_environment_token(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setenv('GH_TOKEN', 'not-for-copilot')
    monkeypatch.setenv('GITHUB_TOKEN', 'not-for-copilot')
    with pytest.raises(UserError, match='/login github-copilot'):
        github_copilot.token()
    monkeypatch.setenv(variable, 'environment-token')
    assert github_copilot.token() == 'environment-token'
    connection = github_copilot.Connection(credentials=CREDENTIALS, issued_at=time.time())
    save_codex_credentials(account='github-copilot', value=connection.model_dump_json())
    assert github_copilot.token() == 'fake-access'


@pytest.mark.parametrize('stored', ['not-json', '{"credentials": {}}', 'expired', 'valid'])
def test_saved_credentials(stored: str) -> None:
    if stored in ('expired', 'valid'):
        connection = github_copilot.Connection(
            credentials=replace(CREDENTIALS, expires_in=3600, refresh_token='fake-refresh'),
            issued_at=0 if stored == 'expired' else time.time(),
        )
        value = connection.model_dump_json()
    else:
        value = stored
    save_codex_credentials(account='github-copilot', value=value)
    if stored == 'valid':
        assert github_copilot.token() == 'fake-access'
    else:
        with pytest.raises(UserError, match='/login github-copilot'):
            github_copilot.token()


@pytest.mark.parametrize(
    'selected', [pick('github-copilot:claude-haiku-4.5'), MenuResult(cancelled=True), MenuResult(), pick(1)]
)
async def test_connect(
    device_flow: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected: MenuResult
) -> None:
    async def discover() -> list[str]:
        assert github_copilot.token() == 'fake-access'
        return ['claude-haiku-4.5']

    monkeypatch.setattr(github_copilot, 'discover', discover)
    context, _ = make_context(tmp_path)
    previous = context.settings.model
    for _ in range(2):
        script = Script(lists=[pick('github-copilot'), selected, MenuResult(cancelled=True)], choices=[], texts=[])
        result = await open_add_model_menu(context, runners=script.runners)
        if selected.item and selected.item.value == 'github-copilot:claude-haiku-4.5':
            assert context.settings.model == 'github-copilot:claude-haiku-4.5'
            assert result == 'Saved model. Applied.'
        else:
            assert context.settings.model == previous
            assert result == 'No changes.'


async def test_provider_menu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context, _ = make_context(tmp_path)
    monkeypatch.setenv('GITHUB_COPILOT_API_KEY', 'test-token')

    async def discover() -> list[str]:
        return ['claude-haiku-4.5']

    monkeypatch.setattr(github_copilot, 'discover', discover)
    name = 'github-copilot:claude-haiku-4.5'
    discovered = await github_copilot_models()
    context.set_setting(['model', name])
    menu = ModelMenu(context, discovered=discovered).for_provider('github-copilot')
    assert [item.value for item in menu.items()] == [name]
    assert menu.items()[0].label.endswith('(current)')
    assert 'provider  github-copilot' in menu.details(menu.items()[0])
    script = Script(
        lists=[
            pick('github-copilot'),
            menu.settings_marker(object(), MenuItem(name, value=name)),
            pick('max_tokens'),
            MenuResult(cancelled=True),
            pick(name),
        ],
        choices=[],
        texts=[typed('42')],
    )
    result = await open_add_model_menu(context, runners=script.runners)
    assert f'Saved max_tokens for {name}' in result
    assert 'Saved model. Applied.' in result
    assert context.settings.model == name
    assert context.store.model_settings(name) == {'max_tokens': 42}


async def test_login_command(device_flow: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs(monkeypatch, ['/login github-copilot', '/login unknown', '/exit'])
    output = io.StringIO()
    await chat(Agent(TestModel()), deps=None, console=Console(file=output), store=SettingsStore(tmp_path / 'config.db'))
    assert github_copilot.token() == 'fake-access'
    assert 'Usage: /login [openai-codex|github-copilot]' in output.getvalue()
    assert 'fake-access' not in output.getvalue()


@pytest.mark.vcr
async def test_recorded_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('GITHUB_COPILOT_API_KEY', 'test-token')
    with use_cassette(Path(__file__).parent / 'cassettes/github_copilot_models.yaml', record_mode='none'):
        names = await github_copilot.discover()
    assert names == sorted(set(names))
    assert {'claude-haiku-4.5', 'gpt-5.4', 'gemini-3.8-flash'} <= set(names)
    assert not any(name.startswith('grok-') for name in names)
    assert 'trajectory-compaction' not in names


@pytest.mark.parametrize('status', [401, 403, 302, 500])
async def test_discovery_http_failure(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    monkeypatch.setenv('GITHUB_COPILOT_API_KEY', 'test-token')
    monkeypatch.setenv('GITHUB_COPILOT_BASE_URL', 'https://copilot.example.test/proxy')
    requests: list[httpx2.Request] = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        assert str(request.url) == 'https://copilot.example.test/proxy/models'
        assert request.headers['authorization'] == 'Bearer test-token'
        assert request.headers['copilot-integration-id'] == 'vscode-chat'
        requests.append(request)
        return httpx2.Response(status, headers={'Location': 'https://untrusted.example/models'}, text='private error')

    with pytest.raises(UserError, match='Copilot model discovery failed') as caught:
        await github_copilot.discover(transport=httpx2.MockTransport(respond))
    assert len(requests) == 1
    assert 'private error' not in str(caught.value)


@pytest.mark.parametrize('body', ['{"data": []}', '{"data": [{"id": "responses-only"}]}', '{"bad": true}'])
async def test_unusable_discovery(monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    monkeypatch.setenv('GITHUB_COPILOT_API_KEY', 'test-token')
    with pytest.raises(UserError, match='no Chat Completions models|invalid model list'):
        await github_copilot.discover(transport=httpx2.MockTransport(lambda request: httpx2.Response(200, text=body)))
