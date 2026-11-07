"""Keys & accounts: `/keys`, `/login`, and `/accounts` as declared menus, asking through `Dialogs`, in the page."""

import asyncio
import io
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

import httpx2
import pytest
from keyring.errors import KeyringError
from pydantic import JsonValue, TypeAdapter
from rich.console import Console
from starlette.testclient import WebSocketTestSession

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.providers.github_copilot import GitHubCopilotCredentials, GitHubCopilotOAuthFlow
from pydantic_ai.providers.openai_codex import OpenAICodexCredentials, OpenAICodexOAuthFlow
from pydantic_clai2.auth import CodexAuth, login_command
from pydantic_clai2.builtin_plugins.web.protocol import (
    PROTOCOL,
    DialogClosed,
    DialogOpened,
    DialogShown,
    SettingsChanged,
)
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.credential_store import has_credentials, load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.models import github_copilot, key_profiles
from pydantic_clai2.models.accounts import Account, accounts, remember
from pydantic_clai2.plugins import PluginLogin
from pydantic_clai2.plugins.menus import Action, Dialogs
from pydantic_clai2.ui.menus import key_menu
from pydantic_clai2.ui.menus.accounts_menu import ADD, AccountsSource, account_key, accounts_settings_menu
from pydantic_clai2.ui.menus.key_menu import KeysSource, keys_menu
from tests.clai2.dialog_script import Back, Confirm, Finish, Pick, ScriptedDialogs, Type
from tests.clai2.test_auth import CREDENTIALS, fake_browser, never_called_back, never_pasted
from tests.clai2.test_github_copilot import device_flow as device_flow
from tests.clai2.test_web import TOKEN, MakeWeb, Web, make_web as make_web, receive_until

SOCKET = f'/ws?token={TOKEN}&protocol={PROTOCOL}'
ORIGIN = {'origin': 'http://testserver'}
CODEX = json.dumps({'access_token': 'a', 'refresh_token': 'r', 'account_id': 'x'})


def fields(web_: Web, command: str) -> dict[str, dict[str, JsonValue]]:
    response = web_.get(f'/api/commands/{command}/fields')
    assert response.status_code == 200, response.text
    rows = TypeAdapter(list[dict[str, JsonValue]]).validate_python(response.json()['fields'])
    return {str(row['key']): row for row in rows}


def answered(web_: Web, socket: WebSocketTestSession, path: str, *answers: str | None) -> httpx2.Response:
    """Run a menu action, answering each dialog it opens in turn, and return what the action answered."""
    with ThreadPoolExecutor(1) as pool:
        action = pool.submit(web_.post, path)
        for answer in answers:
            dialog = receive_until(socket, DialogOpened)[1].dialog
            assert web_.post(f'/api/dialogs/{dialog.id}', {'value': answer}).status_code == 204
        return action.result(timeout=10)


def terminal(web_: Web) -> str:
    file = web_.shell.console.file
    assert isinstance(file, io.StringIO)
    return file.getvalue()


def test_bare_keys_login_and_accounts_open_the_page_menu(make_web: MakeWeb) -> None:
    web_ = make_web(Agent(TestModel()))
    for command in ('keys', 'login', 'accounts'):
        assert web_.post('/api/input', {'text': f'/{command}'}).json() == {'menu': command}
    assert fields(web_, 'login').keys() == fields(web_, 'accounts').keys() == {ADD}


def test_the_keys_menu_adds_renames_and_deletes_without_sending_secrets(make_web: MakeWeb) -> None:
    api_keys.save_key(name='USED', value='sk-used-secret')
    save_codex_credentials(account='vllm', value='{"token":{"name":"USED"}}')
    web_ = make_web(Agent(TestModel()))
    rows = fields(web_, 'keys')
    assert list(rows) == ['USED', ADD]
    assert (rows['USED']['value'], rows['USED']['secret'], rows['USED']['action']) == ('(hidden)', True, True)
    assert rows['USED']['description'] == 'Used by vllm.'
    assert rows[ADD]['value'] == '' and rows[ADD]['action'] is True
    assert 'sk-used-secret' not in web_.get('/api/commands/keys/fields').text
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        added = answered(web_, socket, '/api/commands/keys/fields/add/action', 'new_key', 'sk-new-secret').json()
        assert added['messages'][0].startswith('Saved NEW_KEY')
        # The menu is built again, so the new key's row has its action.
        shown = [(row['key'], row['action']) for row in added['menu']['fields']]
        assert shown == [('NEW_KEY', True), ('USED', True), (ADD, True)]
        assert api_keys.load_keys()['NEW_KEY'].get_secret_value() == 'sk-new-secret'
        assert 'sk-new-secret' not in terminal(web_) and 'Saved NEW_KEY' in terminal(web_)
        receive_until(socket, SettingsChanged)
        # A key a connection uses cannot be renamed; the page says why.
        refused = answered(web_, socket, '/api/commands/keys/fields/USED/action', 'rename', 'other')
        assert refused.status_code == 400
        assert refused.json()['detail'] == 'Key is used by vllm. Reconfigure those connections before renaming.'
        deleted = answered(web_, socket, '/api/commands/keys/fields/NEW_KEY/action', 'delete', 'yes').json()
        assert deleted['messages'][0].startswith('Deleted NEW_KEY')
        assert answered(web_, socket, '/api/commands/keys/fields/USED/action', None).json()['messages'] == []
    # A value typed into a key's row replaces it; nothing removes a key but its delete.
    assert web_.post('/api/commands/keys/fields/USED', {'value': 'sk-replaced'}).status_code == 200
    assert api_keys.load_keys()['USED'].get_secret_value() == 'sk-replaced'
    assert 'Delete it from its menu' in web_.post('/api/commands/keys/fields/USED', {'value': ''}).json()['detail']
    assert 'Add an API key' in web_.post('/api/commands/keys/fields/add', {'value': 'x'}).json()['detail']


async def test_key_operations_ask_through_dialogs(monkeypatch: pytest.MonkeyPatch) -> None:
    api_keys.save_key(name='KEY', value='secret')
    menu = keys_menu()
    assert list(menu.actions) == ['KEY', ADD]
    replaced = await menu.actions['KEY'](ScriptedDialogs([Pick(value='replace'), Type(text='new-secret')]))
    assert replaced == ['Saved KEY in the OS keyring.']
    assert await menu.actions['KEY'](ScriptedDialogs([Pick(value='replace'), Back()])) == []
    renamed = await menu.actions['KEY'](ScriptedDialogs([Pick(value='rename'), Type(text='other')]))
    assert renamed == ['Renamed KEY to OTHER.']
    assert await key_menu.rename_key('OTHER', ScriptedDialogs([Back()])) == ''
    assert await key_menu.add_key(ScriptedDialogs([Back()])) == ''
    names = key_menu._name_problem  # pyright: ignore[reportPrivateUsage]
    assert names('bad-name') is not None and names('fine') is None
    source = KeysSource()
    add = source.rows()[-1]
    assert (add.key, source.current(add)) == (ADD, '')
    # A credential store that fails says so, without the error's text, which could hold a secret.

    def broken(*, name: str, value: str) -> str:
        raise KeyringError('secret-in-error')

    monkeypatch.setattr(api_keys, 'save_key', broken)
    with pytest.raises(ValueError, match='Check your credential backend'):
        await keys_menu().actions['OTHER'](ScriptedDialogs([Pick(value='replace'), Type(text='x')]))
    # A connection that cannot be read is named instead of where the key is used.
    save_codex_credentials(account='vllm', value='not json')
    assert 'Reconfigure the invalid vllm connection' in source.rows()[0].description


def store_at(tmp_path: Path) -> SettingsStore:
    return SettingsStore(tmp_path / 'config.db')


async def test_the_accounts_menu_signs_in_renames_reorders_and_signs_out(tmp_path: Path) -> None:
    store = store_at(tmp_path)
    save_codex_credentials(account='openai-codex', value=CODEX)
    save_codex_credentials(account='openai-codex@work', value=CODEX)
    save_codex_credentials(account='gateway/openai@team', value='{}')
    claude = PluginLogin(name='claude', handler=lambda: asyncio.sleep(0, 'unused'), models=('claude-code:opus',))
    remember(store, login='claude', profile='work', plugin=claude)
    signed: list[tuple[str, Dialogs]] = []
    forgotten: list[str] = []

    async def sign_in(target: str, dialogs: Dialogs) -> str:
        signed.append((target, dialogs))
        if target == 'openai-codex@broken':
            raise UserError('Codex login timed out.')
        return f'Signed in as {target}.'

    def menu_for() -> Callable[[str], Action]:
        menu = accounts_settings_menu(
            store, sign_in=sign_in, plugins=lambda: {'claude': claude}, forget=forgotten.append
        )
        return lambda key: menu.actions[key]

    source = AccountsSource(store)
    rows = source.rows()
    assert [(row.key, row.label, source.current(row)) for row in rows] == [
        ('claude@work', 'claude-code · work', 'signed in'),
        ('gateway:openai@team', 'gateway/openai · team', 'signed in'),
        ('openai-codex', 'openai-codex · default', 'signed in'),
        ('openai-codex@work', 'openai-codex · work', 'signed in'),
        (ADD, 'Add an account', ''),
    ]
    assert rows[0].description == 'Use claude-code@work:MODEL. Its plugin keeps the sign-in.'
    assert rows[2].description == (
        'Use openai-codex@default:MODEL. openai-codex@*:MODEL tries 2 accounts; this one is number 1.'
    )
    assert all(row.default == source.current(row) for row in rows), 'nothing to reset'
    assert source.problem(rows[0], 'x') is not None
    for edit in (lambda: source.apply(rows[0], 'x'), lambda: source.reset(rows[0])):
        with pytest.raises(ValueError, match='Choose the account'):
            edit()
    action = menu_for()
    dialogs = ScriptedDialogs([Pick(value='later')])
    assert await action('openai-codex')(dialogs) == ['openai-codex@*:MODEL now tries openai-codex later.']
    assert [option.value for option in dialogs.asked[0].options] == ['sign-in', 'rename', 'later', 'sign-out']
    assert [item.login for item in accounts(store) if item.provider == 'openai-codex'] == [
        'openai-codex@work',
        'openai-codex',
    ]
    dialogs = ScriptedDialogs([Pick(value='earlier')])
    assert await menu_for()('openai-codex')(dialogs) == ['openai-codex@*:MODEL now tries openai-codex earlier.']
    assert [option.value for option in dialogs.asked[0].options] == ['sign-in', 'rename', 'earlier', 'sign-out']
    plugin = ScriptedDialogs([Back()])
    assert await action('claude@work')(plugin) == []
    assert plugin.asked[0].options[-1].label == 'Remove from the list'
    sign = ScriptedDialogs([Pick(value='sign-in')])
    assert await action('openai-codex@work')(sign) == ['Signed in as openai-codex@work.']
    assert signed[-1] == ('openai-codex@work', sign), 'the sign-in asks through the same dialogs'
    renamed = await action('openai-codex')(ScriptedDialogs([Pick(value='rename'), Type(text='Main')]))
    assert renamed == ['openai-codex is shown as Main.']
    assert await action('openai-codex')(ScriptedDialogs([Pick(value='rename'), Back()])) == []
    assert await action('openai-codex')(ScriptedDialogs([Pick(value='sign-out'), Confirm(yes=False)])) == []
    left = await action('openai-codex')(ScriptedDialogs([Pick(value='sign-out'), Confirm(yes=True)]))
    assert left == ['Signed out of openai-codex.'] and forgotten == ['openai-codex']
    with pytest.raises(LookupError, match='no longer an account'):
        await action('openai-codex')(ScriptedDialogs([]))
    # Adding one picks a provider and a free name, then signs in to it.
    # No default account any more: it is the one to sign in to, with no name to type.
    assert await action(ADD)(ScriptedDialogs([Pick(value='openai-codex')])) == ['Signed in as openai-codex.']
    save_codex_credentials(account='openai-codex', value=CODEX)
    with pytest.raises(ValueError, match='Codex login timed out'):
        await action(ADD)(ScriptedDialogs([Pick(value='openai-codex'), Type(text='broken')]))
    assert await action(ADD)(ScriptedDialogs([Back()])) == []


def test_account_keys_avoid_the_route_separator() -> None:
    item = Account(provider='gateway/openai', profile='team', label=None, plugin_login=None, signed_in=True)
    assert account_key(item) == 'gateway:openai@team'


async def test_codex_signs_in_behind_the_waiting_screen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        return CREDENTIALS

    def no_browser(url: str) -> bool:
        raise AssertionError('the page shows the link instead')  # pragma: no cover

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    monkeypatch.setattr('pydantic_clai2.auth.open_browser', no_browser)
    output = io.StringIO()
    auth = CodexAuth(Console(file=output), read_line=never_pasted)
    dialogs = ScriptedDialogs([Finish()])
    assert (await auth.login(['openai-codex@work'], dialogs=dialogs)).startswith('Codex connected as openai-codex@work')
    assert dialogs.asked[0].title == 'Signing in to ChatGPT/Codex for profile work'
    assert 'https://auth.openai.com/oauth/authorize' in dialogs.asked[0].shown[0]
    assert output.getvalue() == '', 'nothing is printed in the terminal'
    assert has_credentials(account='openai-codex@work')

    # A plaintext fallback file says so.
    plaintext = tmp_path / 'credentials.json'
    plaintext.write_text('{}')

    def plaintext_path(*, account: str) -> Path:
        return plaintext

    monkeypatch.setattr('pydantic_clai2.auth.credentials_path', plaintext_path)
    fallback = await auth.login([], dialogs=ScriptedDialogs([Finish()]))
    assert fallback.startswith('Codex connected. No OS keyring is available')


async def test_codex_sign_in_cancelled_failed_or_timed_out(monkeypatch: pytest.MonkeyPatch) -> None:
    async def port_in_use(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        raise OSError(48, 'Address already in use')

    monkeypatch.setattr('pydantic_clai2.auth.open_browser', fake_browser)
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', never_called_back)
    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted)
    assert await auth.login([], dialogs=ScriptedDialogs([Back()])) == 'ChatGPT/Codex sign-in cancelled.'
    timed_out = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted, login_timeout=0)
    with pytest.raises(UserError, match='timed out'):
        await timed_out.login([], dialogs=ScriptedDialogs([Finish()]))
    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', port_in_use)
    with pytest.raises(UserError, match='in the terminal to paste the URL instead'):
        await auth.login([], dialogs=ScriptedDialogs([Finish()]))
    assert load_codex_credentials() is None


async def test_copilot_shows_its_code_on_the_waiting_screen(device_flow: None, monkeypatch: pytest.MonkeyPatch) -> None:
    output = io.StringIO()
    dialogs = ScriptedDialogs([Finish()])
    message = await github_copilot.login(console=Console(file=output), dialogs=dialogs)
    assert message.startswith('GitHub login saved.') and output.getvalue() == ''
    assert dialogs.asked[0].shown == ['Open https://github.com/login/device and enter code ABCD-EFGH']
    assert has_credentials(account='github-copilot')

    async def never_approved(self: GitHubCopilotOAuthFlow) -> GitHubCopilotCredentials:
        await asyncio.Event().wait()
        raise AssertionError('unreachable')  # pragma: no cover

    monkeypatch.setattr(GitHubCopilotOAuthFlow, 'wait_for_authorization', never_approved)
    cancelled = await github_copilot.login(
        console=Console(file=output), account='github-copilot@x', dialogs=ScriptedDialogs([Back()])
    )
    assert cancelled == 'GitHub Copilot sign-in cancelled.'
    assert not has_credentials(account='github-copilot@x')


async def test_login_with_dialogs_saves_api_keys_in_keys_and_sends_the_rest_to_the_terminal() -> None:
    auth = CodexAuth(Console(file=io.StringIO()), read_line=never_pasted)
    dialogs = ScriptedDialogs([Type(text='sk-typed')])
    message = await login_command(['openai@work'], codex=auth, dialogs=dialogs)
    assert message.startswith('openai@work connected')
    assert api_keys.load_keys()['OPENAI_WORK'].get_secret_value() == 'sk-typed', 'kept in /keys'
    assert json.loads(load_codex_credentials(account='openai@work') or '') == {'token': {'name': 'OPENAI_WORK'}}
    picked = await login_command(['openai@side'], codex=auth, dialogs=ScriptedDialogs([Pick(value='OPENAI_WORK')]))
    assert picked.startswith('openai@side connected')
    cancelled = await key_profiles.login(
        provider='openai', profile='gone', console=auth.console, dialogs=ScriptedDialogs([Back()])
    )
    assert cancelled == 'Sign-in cancelled.'
    with pytest.raises(ValueError, match=r'Connect openrouter@x in the terminal: /login openrouter@x\.'):
        await login_command(['openrouter@x'], codex=auth, dialogs=ScriptedDialogs([]))
    plugins = {'claude': PluginLogin(name='claude', handler=lambda: asyncio.sleep(0, 'unused'))}
    with pytest.raises(ValueError, match=r'Sign in to claude@work in the terminal: /login claude@work\.'):
        await login_command(['claude@work'], codex=auth, plugins=plugins, dialogs=ScriptedDialogs([]))


def test_adding_an_account_from_the_page_signs_in_behind_its_waiting_screen(
    make_web: MakeWeb, monkeypatch: pytest.MonkeyPatch
) -> None:
    called_back = asyncio.Event()

    async def exchange(self: OpenAICodexOAuthFlow) -> OpenAICodexCredentials:
        await called_back.wait()
        return CREDENTIALS

    monkeypatch.setattr(OpenAICodexOAuthFlow, 'exchange_code_from_callback', exchange)
    web_ = make_web(Agent(TestModel()))
    with web_.client.websocket_connect(SOCKET, headers=ORIGIN) as socket:
        socket.receive_text()
        with ThreadPoolExecutor(1) as pool:
            action = pool.submit(web_.post, '/api/commands/accounts/fields/add/action')
            picker = receive_until(socket, DialogOpened)[1].dialog
            assert web_.post(f'/api/dialogs/{picker.id}', {'value': 'openai-codex'}).status_code == 204
            waiting = receive_until(socket, DialogOpened)[1].dialog
            assert (waiting.kind, waiting.title) == ('wait', 'Signing in to ChatGPT/Codex')
            shown = receive_until(socket, DialogShown)[1].text
            assert any(urlsplit(word).hostname == 'auth.openai.com' for word in shown.split())
            # The browser comes back to this machine: the screen closes by itself.
            web_.portal.call(called_back.set)
            assert receive_until(socket, DialogClosed)[1].id == waiting.id
            done = action.result(timeout=10).json()
        assert done['messages'][0].startswith('Codex connected.')
        assert [row['key'] for row in done['menu']['fields']] == ['openai-codex', ADD]
        assert 'Codex connected.' in terminal(web_)
        # Signing out from the page drops the login.
        left = answered(web_, socket, '/api/commands/accounts/fields/openai-codex/action', 'sign-out', 'yes').json()
        assert left['messages'] == ['Signed out of openai-codex.']
        assert not has_credentials(account='openai-codex')
