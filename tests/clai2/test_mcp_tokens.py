"""OAuth tokens for MCP servers survive a restart through the keyring, and `/mcp auth` manages them."""

import json
from pathlib import Path

import keyring
import pytest
from fastmcp.client.auth import OAuth
from fastmcp.client.auth.oauth import TokenStorageAdapter
from keyring.errors import KeyringLocked, PasswordDeleteError
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from menu_script import Script, pick, typed
from pydantic import AnyUrl, HttpUrl

from pydantic_clai2.mcp import HTTPServer, MCPCommand, MCPServers, MCPStore, SSEServer, StdioServer, TokenStore, oauth

URL = 'https://mcp.example.com/mcp'
Vault = dict[tuple[str, str], str]


@pytest.fixture
def vault(monkeypatch: pytest.MonkeyPatch) -> Vault:
    entries: Vault = {}

    def get(service: str, account: str) -> str | None:
        return entries.get((service, account))

    def set_value(service: str, account: str, value: str) -> None:
        entries[service, account] = value

    def delete(service: str, account: str) -> None:
        if (service, account) not in entries:
            raise PasswordDeleteError('Not found')
        del entries[service, account]

    monkeypatch.setattr(keyring, 'get_password', get)
    monkeypatch.setattr(keyring, 'set_password', set_value)
    monkeypatch.setattr(keyring, 'delete_password', delete)
    return entries


def token() -> OAuthToken:
    return OAuthToken(access_token='access', token_type='Bearer', refresh_token='refresh', expires_in=3600)


async def test_tokens_survive_a_restart_in_the_keyring(vault: Vault) -> None:
    first = TokenStorageAdapter(TokenStore('logfire'), server_url=URL)
    await first.set_tokens(token())
    client = OAuthClientInformationFull(client_id='cid', redirect_uris=[AnyUrl('http://127.0.0.1/cb')])
    await first.set_client_info(client)
    assert list(vault) == [('pydantic-clai2', 'mcp-logfire')]
    assert 'refresh' in vault['pydantic-clai2', 'mcp-logfire']

    restarted = TokenStorageAdapter(TokenStore('logfire'), server_url=URL)
    assert await restarted.get_tokens() == token()
    assert (await restarted.get_token_expiry() or 0) > 0
    stored = await restarted.get_client_info()
    assert stored is not None and stored.client_id == 'cid'
    assert TokenStore('logfire').signed_in() and not TokenStore('other').signed_in()
    moved = TokenStorageAdapter(TokenStore('logfire'), server_url='https://moved.example.com')
    assert await moved.get_tokens() is None, 'tokens are tied to the URL they were issued for'

    await restarted.clear()
    assert vault == {} and not TokenStore('logfire').signed_in()


async def test_key_value_protocol_edges(vault: Vault) -> None:
    store = TokenStore('x')
    await store.put('a', {'v': 1}, ttl=60)
    await store.put('gone', {'v': 2}, collection='c', ttl=-1)
    value, left = await store.ttl('a')
    assert value == {'v': 1} and left is not None and 0 < left <= 60
    assert await store.get('gone', collection='c') is None, 'expired entries are hidden'
    await store.put_many(['b', 'c'], [{'v': 3}, {'v': 4}])
    assert await store.get_many(['b', 'c', 'nope']) == [{'v': 3}, {'v': 4}, None]
    assert await store.ttl('b') == ({'v': 3}, None)
    assert await store.delete('b') and not await store.delete('b')
    assert await store.delete_many(['a', 'c']) == 2
    assert vault == {}, 'the credential goes once nothing is stored'


async def test_unreadable_bundle_means_signing_in_again(vault: Vault) -> None:
    vault['pydantic-clai2', 'mcp-x'] = 'not json'
    assert await TokenStore('x').get('a') is None
    vault['pydantic-clai2', 'mcp-x'] = 'clai-chunks-v1:broken'
    assert not TokenStore('x').signed_in()


def test_oauth_uses_the_keyring_store() -> None:
    assert oauth('x', HTTPServer(type='http', url=HttpUrl(URL))) is None
    assert isinstance(oauth('x', SSEServer(type='sse', url=HttpUrl(URL), auth='oauth')), OAuth)


def make(tmp_path: Path, script: Script | None = None) -> tuple[MCPCommand, MCPStore]:
    store = MCPStore(tmp_path / 'config', workspace=tmp_path)
    command = MCPCommand(servers=MCPServers(store))
    if script is not None:
        command.runners = script.runners
    return command, store


async def test_auth_command(tmp_path: Path, vault: Vault) -> None:
    command, store = make(tmp_path)
    dead = 'http://127.0.0.1:9/mcp'
    store.put('local', StdioServer(type='stdio', command='python'))
    store.put('plain', HTTPServer(type='http', url=HttpUrl(URL)))
    store.put('dead', HTTPServer(type='http', url=HttpUrl(dead), auth='oauth', timeout=5))
    for name in ('local', 'plain'):
        with pytest.raises(ValueError, match=f'{name} does not use OAuth'):
            await command(['auth', name])
    with pytest.raises(ValueError, match='Usage: /mcp auth NAME'):
        await command(['auth', 'dead', 'now'])
    assert 'oauth' not in await command(['status', 'plain'])
    assert 'oauth    not signed in' in await command(['status', 'dead'])

    await TokenStorageAdapter(TokenStore('dead'), server_url=dead).set_tokens(token())
    assert 'oauth    signed in (/mcp auth dead [logout])' in await command(['status', 'dead'])
    assert (await command(['auth', 'dead', 'logout'])).startswith('Signed out of dead.')
    assert vault == {}

    await TokenStore('dead').put('k', {'v': 1})
    assert (await command(['auth', 'dead'])).startswith('Could not start dead'), 'signing in reconnects'
    assert vault == {}, 'old tokens are dropped before signing in again'
    assert tuple(command.complete(['auth', 'dead', ''])) == ('logout',)

    await TokenStore('dead').put('k', {'v': 1})
    await command(['remove', 'dead'])
    assert vault == {}, 'removing a server signs it out'


async def test_rename_signs_out_the_old_name(tmp_path: Path, vault: Vault) -> None:
    command, store = make(tmp_path, Script(lists=[pick('name'), pick('save')], choices=[], texts=[typed('renamed')]))
    store.put('docs', HTTPServer(type='http', url=HttpUrl(URL), auth='oauth'))
    await TokenStore('docs').put('k', {'v': 1})
    assert (await command(['edit', 'docs'])).startswith('Updated renamed.')
    assert vault == {} and json.loads(store.path.read_text())['servers']['renamed']['auth'] == 'oauth'


async def test_locked_keyring_does_not_break_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def locked(service: str, account: str) -> str | None:
        raise KeyringLocked('locked')

    monkeypatch.setattr(keyring, 'get_password', locked)
    command, store = make(tmp_path)
    store.put('docs', HTTPServer(type='http', url=HttpUrl(URL), auth='oauth'))
    assert TokenStore('docs').signed_in() is None
    assert 'oauth    unknown; the keyring could not be read' in await command(['status', 'docs'])
