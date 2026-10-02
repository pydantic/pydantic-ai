"""The built-in `linear` plugin: declared disabled, set up in its settings menu, key named in `/keys`.

The menu tests drive the real termflow widgets with scripted keys, the way `test_api_keys` drives the key picker.
"""

import io
import itertools
import json
from pathlib import Path
from typing import Protocol, TypeGuard

import pytest
from fastmcp import Client
from fastmcp.client.auth import OAuth
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.linear import Linear
from pydantic_clai2 import DEFAULT_PLUGINS
from pydantic_clai2.builtin_plugins import linear
from pydantic_clai2.builtin_plugins.linear import ACCOUNT, KEY_NAME, TOKEN_ACCOUNT, choose_key, reference
from pydantic_clai2.commands import Commands
from pydantic_clai2.config import api_keys
from pydantic_clai2.config.api_keys import delete_key, key_users, load_keys, rename_key, save_key
from pydantic_clai2.config.credential_store import load_codex_credentials, save_codex_credentials
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.mcp import TokenStore, http_client
from pydantic_clai2.plugins import SessionStart
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.ui.menus import field_menu
from tests.clai2.conftest import stored_accounts

Vault = dict[tuple[str, str], str]


class Press(Protocol):
    def __call__(self, *keys: str) -> None: ...


@pytest.fixture(autouse=True)
def press(monkeypatch: pytest.MonkeyPatch) -> Press:
    """Script the keys every menu reads; once they run out, Esc closes whatever is open."""

    def script(*keys: str) -> None:
        pressed = itertools.chain(keys, itertools.repeat('escape'))
        monkeypatch.setattr(field_menu, 'menu_key', pressed.__next__)
        monkeypatch.setattr(api_keys, 'menu_key', pressed.__next__)

    script()
    return script


class Prompt:
    def __init__(self, *values: str | BaseException) -> None:
        self.values = iter(values)
        self.labels: list[str] = []

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        self.labels.append(label)
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        return value


def make(tmp_path: Path) -> tuple[PluginLoader[None], Commands, io.StringIO]:
    store = SettingsStore(tmp_path / 'settings.db')
    commands = Commands()
    output = io.StringIO()
    loader = PluginLoader[None](
        store=store,
        console=Console(file=output, width=400),
        commands=commands,
        session_start=lambda: SessionStart(agent=Agent(TestModel()), settings=store.load()),
        builtin=[plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'linear'],
    )
    return loader, commands, output


async def declare(loader: PluginLoader[None], settings: dict[str, JsonValue]) -> None:
    await loader.remove('linear')
    await loader.command(['add', 'linear', 'pydantic_clai2.builtin_plugins.linear', json.dumps(settings)])


def saved(tmp_path: Path) -> dict[str, JsonValue]:
    [declaration] = [plugin for plugin in SettingsStore(tmp_path / 'settings.db').plugins() if plugin.id == 'linear']
    return declaration.settings


def is_linear(capability: object) -> TypeGuard[Linear[None]]:
    return isinstance(capability, Linear)


def only(loader: PluginLoader[None]) -> Linear[None]:
    """The capability the next run gets; under OAuth, built per run."""
    [capability] = loader.capabilities()
    if callable(capability):
        capability = capability(RunContext(deps=None, model=TestModel(), usage=RunUsage()))
    assert is_linear(capability)
    return capability


def token(loader: PluginLoader[None]) -> str | None:
    """What the next run connects with."""
    capability = only(loader)
    assert callable(capability.auth)
    return capability.auth(RunContext(deps=None, model=TestModel(), usage=RunUsage()))


async def run(commands: Commands, text: str) -> str:
    result = commands.execute(text)
    return result if isinstance(result, str) else await result


def test_declared_as_a_disabled_built_in_not_the_raw_capability() -> None:
    [declaration] = [plugin for plugin in DEFAULT_PLUGINS if plugin.id == 'linear']
    assert declaration.factory == 'pydantic_clai2.builtin_plugins.linear'
    assert not declaration.enabled
    assert all(not plugin.factory.startswith('pydantic_ai_harness.linear') for plugin in DEFAULT_PLUGINS)


async def test_each_run_resolves_the_named_key(tmp_path: Path, vault: Vault) -> None:
    loader, _, output = make(tmp_path)
    await loader.load_all()
    assert loader.capabilities() == [], 'disabled until the user enables it'
    save_key(name=KEY_NAME, value='lin_1')
    assert await loader.command(['enable', 'linear']) == 'Enabled linear.\nLinear settings unchanged.'
    assert output.getvalue() == ''
    capability = only(loader)
    assert capability.read_only and capability.include_instructions
    assert token(loader) == 'lin_1'

    save_key(name=KEY_NAME, value='lin_2')
    assert token(loader) == 'lin_2', 'replacing the key in /keys reaches the next run without a reload'
    delete_key(name=KEY_NAME)
    with pytest.raises(
        UserError, match=f'{KEY_NAME} is missing. Restore it in /keys or reconfigure through /plugins configure linear'
    ):
        token(loader)
    await loader.close('exit')


async def test_menu_saves_each_edit_and_reloads(tmp_path: Path, vault: Vault, press: Press) -> None:
    save_key(name=KEY_NAME, value='lin')
    loader, _, _ = make(tmp_path)
    await loader.command(['enable', 'linear'])
    # Rows: Sign-in, API key, Access, Server instructions.
    press(
        *('down', 'down', 'enter', 'down', 'enter'),  # Access -> Read and write
        *('down', 'enter', 'escape'),  # Server instructions, Esc keeps it
        *('enter', 'down', 'enter'),  # Server instructions -> Leave out
        'R',  # ... and back to the default
    )
    message = await loader.command(['configure', 'linear'])
    assert message.splitlines() == [
        'Linear Access: Read and write.',
        'Linear Server instructions: Leave out.',
        'Linear Server instructions: Include (default).',
    ]
    assert saved(tmp_path) == {'auth': 'api_key', 'read_only': False, 'include_instructions': True}, 'saved as edited'
    capability = only(loader)
    assert not capability.read_only and capability.include_instructions, 'reloaded with the new settings'
    assert token(loader) == 'lin'


async def test_switching_to_oauth_hides_the_key_row(tmp_path: Path, vault: Vault, press: Press) -> None:
    loader, _, _ = make(tmp_path)
    press('enter', 'down', 'enter', 'down', 'enter', 'down', 'enter')  # Sign-in -> OAuth; then row 2 is Access
    assert (await loader.command(['enable', 'linear'])).splitlines() == [
        'Enabled linear.',
        'Linear Sign-in: Browser sign-in (OAuth).',
        'Linear Access: Read and write.',
    ]
    assert saved(tmp_path) == {'auth': 'oauth', 'read_only': False, 'include_instructions': True}
    client = only(loader).client
    assert isinstance(client, Client) and isinstance(client.transport, StreamableHttpTransport)
    assert client.transport.url == 'https://mcp.linear.app/mcp'


async def test_key_row_picks_a_saved_key_and_resets(tmp_path: Path, vault: Vault, press: Press) -> None:
    save_key(name='GITHUB_TOKEN', value='gh')
    save_key(name='SHARED_LINEAR', value='lin_shared')
    loader, _, _ = make(tmp_path)
    press('down', 'enter', 'down', 'enter')  # API key -> the searchable /keys picker -> SHARED_LINEAR
    assert (
        await loader.command(['enable', 'linear'])
    ) == 'Enabled linear.\nLinear uses SHARED_LINEAR from /keys from the next run.'
    assert token(loader) == 'lin_shared'
    raw = load_codex_credentials(account=ACCOUNT)
    assert raw is not None and json.loads(raw) == {'token': {'name': 'SHARED_LINEAR'}}
    assert 'lin_' not in json.dumps(saved(tmp_path)) and saved(tmp_path) == {}
    assert key_users(name='SHARED_LINEAR') == ['linear']
    with pytest.raises(ValueError, match='used by linear'):
        rename_key(name='SHARED_LINEAR', new_name='OTHER')

    press('down', 'R')
    assert await loader.command(['configure', 'linear']) == f'Linear uses {KEY_NAME} from /keys from the next run.'
    assert reference().name == KEY_NAME


async def test_missing_key_is_reported_at_load_and_fixed_in_the_menu(
    tmp_path: Path, vault: Vault, monkeypatch: pytest.MonkeyPatch, press: Press
) -> None:
    loader, _, output = make(tmp_path)
    await loader.enable('linear')
    assert f'Linear: Saved API key {KEY_NAME} is missing' in output.getvalue()
    with pytest.raises(UserError, match='is missing'):
        token(loader)

    prompt = Prompt('lin_typed')
    monkeypatch.setattr(linear, 'PromptSession', lambda: prompt)
    press('down', 'enter')  # No saved keys, so the masked prompt opens directly.
    message = await loader.command(['configure', 'linear'])
    assert message.endswith(f'Linear uses {KEY_NAME} from /keys from the next run.')
    assert prompt.labels == [f'Linear API key (saved in /keys as {KEY_NAME}): ']
    assert token(loader) == 'lin_typed'

    save_codex_credentials(account=ACCOUNT, value='{"token": "inline-secret"}')
    prompt = Prompt('  ')
    press('down', 'enter', 'down', 'enter')  # The picker now lists LINEAR_API_KEY; choose to enter another.
    assert await loader.command(['configure', 'linear']) == 'A Linear API key is required.'
    with pytest.raises(UserError, match='saved Linear key choice is invalid'):
        reference()


@pytest.mark.parametrize('settings', [{'api_key': 'lin_secret'}, {'oauth': True}, {'auth': 'token'}])
async def test_settings_carry_no_credential(tmp_path: Path, settings: dict[str, JsonValue]) -> None:
    loader, _, _ = make(tmp_path)
    with pytest.raises(PluginError, match='validation error'):
        await declare(loader, settings)
    assert loader.capabilities() == []


@pytest.mark.parametrize(
    ('existing', 'answer', 'stored', 'result'),
    [
        (False, None, 'lin_new', f'Saved {KEY_NAME} in the OS keyring. Linear uses {KEY_NAME}'),
        (True, 'y', 'lin_new', f'Saved {KEY_NAME} in the OS keyring. Linear uses {KEY_NAME}'),
        (True, 'n', 'lin_old', 'Linear key unchanged.'),
        (True, EOFError(), 'lin_old', 'Linear key unchanged.'),
    ],
)
async def test_enter_a_new_key(
    vault: Vault,
    monkeypatch: pytest.MonkeyPatch,
    existing: bool,
    answer: str | BaseException | None,
    stored: str,
    result: str,
) -> None:
    if existing:
        save_key(name=KEY_NAME, value='lin_old')
        monkeypatch.setattr(api_keys, 'menu_key', iter(['down', 'enter']).__next__)
    prompt = Prompt(' lin_new ', *([] if answer is None else [answer]))
    assert (await choose_key(prompt)).startswith(result)
    assert load_keys()[KEY_NAME].get_secret_value() == stored
    raw = load_codex_credentials(account=ACCOUNT) or ''
    assert 'lin_' not in raw, 'the credential-store entry holds a name, never the key'
    if existing:
        assert prompt.labels[-1] == f'Replace {KEY_NAME} in /keys for every plugin using it? [y/N]: '


async def test_key_saved_elsewhere_while_typing_is_not_replaced_unasked(vault: Vault) -> None:
    class Racing(Prompt):
        """Another CLAI saves the key while this one is typing a new one."""

        async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
            if is_password:
                save_key(name=KEY_NAME, value='lin_other')
            return await super().prompt_async(label, is_password=is_password)

    prompt = Racing('lin_new', 'n')
    assert await choose_key(prompt) == 'Linear key unchanged.'
    assert prompt.labels[-1] == f'Replace {KEY_NAME} in /keys for every plugin using it? [y/N]: '
    assert load_keys()[KEY_NAME].get_secret_value() == 'lin_other'
    assert load_codex_credentials(account=ACCOUNT) is None


async def test_key_deleted_while_choosing(vault: Vault, monkeypatch: pytest.MonkeyPatch) -> None:
    async def gone(*, prompt: object, label: str) -> api_keys.KeyReference:
        return api_keys.KeyReference(name='GONE')

    monkeypatch.setattr(linear, 'prompt_api_key', gone)
    with pytest.raises(UserError, match='Select a saved key again through /plugins configure linear'):
        await choose_key(Prompt())
    assert load_codex_credentials(account=ACCOUNT) is None


async def test_cancel_empty_and_invalid_choice(vault: Vault) -> None:
    assert await choose_key(Prompt(EOFError())) == 'Linear key unchanged.'
    with pytest.raises(ValueError, match='A Linear API key is required'):
        await choose_key(Prompt('  '))
    assert reference().name == KEY_NAME
    save_codex_credentials(account=ACCOUNT, value='{"token": "inline-secret"}')
    with pytest.raises(UserError, match='saved Linear key choice is invalid'):
        reference()


@pytest.mark.parametrize(
    ('read_only', 'url'), [(True, 'https://mcp.linear.app/mcp/readonly'), (False, 'https://mcp.linear.app/mcp')]
)
async def test_oauth_signs_in_with_keyring_tokens(tmp_path: Path, vault: Vault, read_only: bool, url: str) -> None:
    loader, commands, _ = make(tmp_path)
    await declare(loader, {'auth': 'oauth', 'read_only': read_only})
    capability = only(loader)
    assert only(loader).client is not capability.client, 'each run connects afresh, so a logout drops cached tokens'
    assert not capability.read_only, 'the URL carries read_only, not tool annotations'
    client = capability.client
    assert isinstance(client, Client)
    transport = client.transport
    assert isinstance(transport, StreamableHttpTransport)
    assert transport.url == url
    assert transport.httpx_client_factory is http_client, 'redirects stay off, as for /mcp servers'
    assert isinstance(transport.auth, OAuth)

    await TokenStore(TOKEN_ACCOUNT).put('x', {'access_token': 'a'}, collection='mcp-oauth-token')
    assert TokenStore(TOKEN_ACCOUNT).signed_in()
    with pytest.raises(ValueError, match='Usage: /linear logout'):
        await run(commands, '/linear')
    assert (await run(commands, '/linear logout')).startswith('Signed out of Linear.')
    assert stored_accounts() == set()

    await loader.disable('linear')
    assert 'linear' not in {command.name for command in commands}
