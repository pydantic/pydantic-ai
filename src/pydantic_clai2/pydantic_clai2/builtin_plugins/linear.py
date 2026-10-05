"""Use Linear, with its key kept in /keys.

The built-in `linear` plugin: harness `Linear`, set up in a settings menu, with its key named in `/keys`.

`/plugins configure linear` (also opened on `/plugins add`, `/plugins enable`, and Space or `c` in `/plugins`) edits
every setting harness `Linear` takes from a user: how to sign in, which `/keys` entry to use, read-only access,
and whether to pass the server's instructions. Edits are saved as they are made. Plugin settings are plaintext
SQLite, so they carry no credential: the menu's key row picks a saved key or saves a new one under
`LINEAR_API_KEY`, and only that name is stored. Each run resolves the name, so replacing the key in `/keys`
reaches Linear on the next run and a deleted key fails the run instead of connecting without it.
"""

from collections.abc import Sequence
from functools import partial
from typing import Generic, Literal

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.linear import Linear
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import (
    KeyExistsError,
    KeyReference,
    SavedKey,
    SecretPrompt,
    forget_connection,
    load_keys,
    prompt_api_key,
    save_key,
    save_key_connection,
)
from pydantic_clai2.config.credential_store import load_codex_credentials
from pydantic_clai2.mcp import HTTPServer, TokenStore, http_client, oauth
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.ui.menus.field_menu import FieldMenu, FieldRow, run_flow_async, shown
from pydantic_clai2.ui.rendering import theme

KEY_NAME = 'LINEAR_API_KEY'
"""The `/keys` label a new Linear key is saved under, and the one used until the menu picks another."""
ACCOUNT = 'linear'
"""Credential-store account holding the chosen key's name. `api_keys.key_users` checks it before a rename."""
TOKEN_ACCOUNT = 'linear_plugin'
"""`TokenStore` name for OAuth tokens. `/mcp` server names cannot contain `_`, so no `/mcp` server shares it."""
# The endpoints harness `Linear` connects to; with `client`, the plugin picks the URL itself.
_URL = 'https://mcp.linear.app/mcp'
_READ_ONLY_URL = 'https://mcp.linear.app/mcp/readonly'
_RECONFIGURE = '/plugins configure linear'


class LinearSettings(BaseModel):
    """The JSON a `linear` declaration may carry. Credentials live in `/keys` or the keyring, never here.

    These are the harness `Linear` fields a user chooses. Linear's hosted server has one URL (plus its read-only
    variant) and takes the workspace from the signed-in account, so there is no base URL or workspace field.
    """

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    auth: Literal['api_key', 'oauth'] = Field(
        default='api_key', description='An API key from /keys, or browser sign-in with tokens kept in the keyring.'
    )
    read_only: bool = Field(default=True, description="Connect to Linear's read-only endpoint.")
    include_instructions: bool = Field(default=True, description="Pass the server's own instructions to the agent.")


class _Connection(BaseModel):
    """The saved choice: a key name, in the shape `api_keys.key_users` reads."""

    token: KeyReference


class LinearPlugin(Plugin[LinearSettings, DepsT]):
    """`Linear` and its settings menu; a missing key is reported on load and fails each run until one is chosen."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        settings = self.settings
        if settings.auth == 'oauth':

            def connect(_: RunContext[DepsT]) -> Linear[DepsT]:
                # A new client per run: FastMCP keeps tokens in memory once connected, so `/linear logout` would
                # otherwise leave this session signed in.
                client = _oauth_client(settings.read_only)
                return Linear[DepsT](client=client, include_instructions=settings.include_instructions)

            return (connect,)

        def token(ctx: RunContext[DepsT]) -> str:
            return saved_key()(ctx)

        return (
            Linear[DepsT](auth=token, read_only=settings.read_only, include_instructions=settings.include_instructions),
        )

    def get_commands(self) -> Sequence[Command]:
        if self.settings.auth != 'oauth':
            return ()
        return (
            Command(
                name='linear',
                description='Sign out of Linear (/linear logout).',
                handler=_logout,
                complete=lambda _: ('logout',),
            ),
        )

    async def configure(self) -> str:
        return await configure(self.host)

    async def on_session_start(self, event: SessionStart) -> None:
        if self.settings.auth == 'oauth':
            return
        try:
            await to_thread.run_sync(lambda: saved_key()(None))
        except UserError as exc:
            self.host.console.print(f'Linear: {exc}', style=theme.color(theme.WARNING), markup=False)


async def _logout(args: list[str]) -> str:
    if args != ['logout']:
        raise ValueError('Usage: /linear logout')
    await to_thread.run_sync(TokenStore(TOKEN_ACCOUNT).forget)
    return 'Signed out of Linear. The next run opens the browser to sign in again.'


async def configure(host: PluginHost[DepsT]) -> str:
    """The settings menu: each edit is saved at once, and the key row opens the `/keys` picker."""
    prompt: PromptSession[str] = PromptSession()

    async def pick_key() -> list[str]:
        try:
            return [await choose_key(prompt)]
        except (UserError, ValueError) as exc:
            return [str(exc)]

    menu = FieldMenu(_Settings(host))
    messages = await run_flow_async(menu, submenus={'api_key': pick_key})
    return '\n'.join(messages) or 'Linear settings unchanged.'


class _Settings(Generic[DepsT]):
    """The rows of the Linear settings menu; a `field_menu.FieldSource`."""

    title = 'Linear settings'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        self._host = host

    def rows(self) -> list[FieldRow]:
        saved = self._saved()
        rows = [
            FieldRow(
                key='auth',
                label='Sign-in',
                description=LinearSettings.model_fields['auth'].description or '',
                default='api_key',
                choices=('api_key', 'oauth'),
                choice_labels={'api_key': 'API key from /keys', 'oauth': 'Browser sign-in (OAuth)'},
                allow_custom=False,
            )
        ]
        if saved.auth == 'api_key':
            rows.append(
                FieldRow(
                    key='api_key',
                    label='API key',
                    description=(
                        'Enter picks a saved /keys entry or enters a new key, masked. Only the name is stored; '
                        f'R goes back to {KEY_NAME}. Several plugins can share one entry.'
                    ),
                    default=KEY_NAME,
                    note=self._key_note(),
                )
            )
        rows += [
            FieldRow(
                key='read_only',
                label='Access',
                description=LinearSettings.model_fields['read_only'].description or '',
                default='true',
                choices=('true', 'false'),
                choice_labels={'true': 'Read-only', 'false': 'Read and write'},
                allow_custom=False,
            ),
            FieldRow(
                key='include_instructions',
                label='Server instructions',
                description=LinearSettings.model_fields['include_instructions'].description or '',
                default='true',
                choices=('true', 'false'),
                choice_labels={'true': 'Include', 'false': 'Leave out'},
                allow_custom=False,
            ),
        ]
        return rows

    def current(self, row: FieldRow) -> str:
        if row.key == 'api_key':
            try:
                return reference().name
            except UserError:
                return '(invalid choice)'
        return shown(self._saved().model_dump()[row.key])

    def problem(self, row: FieldRow, text: str) -> str | None:  # pragma: no cover -- no row opens a text editor.
        """Nothing here is typed: every row is a fixed choice or the `/keys` picker."""
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        self._host.save_settings(self._validated(row, raw))
        return f'Linear {row.label}: {row.display(raw)}.'

    def reset(self, row: FieldRow) -> str:
        if row.key == 'api_key':
            forget_connection(account=ACCOUNT)
            return f'Linear uses {KEY_NAME} from /keys from the next run.'
        self._host.save_settings(self._validated(row, row.default))
        return f'Linear {row.label}: {row.display(row.default)} (default).'

    def _saved(self) -> LinearSettings:
        # Settings that fail validation fail the load, so the menu of a loaded plugin never sees them.
        return self._host.settings(LinearSettings)

    def _validated(self, row: FieldRow, text: str) -> LinearSettings:
        settings: dict[str, JsonValue] = self._saved().model_dump()
        settings[row.key] = text == 'true' if isinstance(settings[row.key], bool) else text
        return LinearSettings.model_validate(settings)

    def _key_note(self) -> str:
        try:
            name = reference().name
        except UserError as exc:
            return str(exc)
        return '' if name in load_keys() else f'{name} is missing from /keys'


def reference() -> KeyReference:
    """The key Linear uses: the one chosen in the settings menu, else `LINEAR_API_KEY`."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return KeyReference(name=KEY_NAME)
    try:
        return _Connection.model_validate_json(raw).token
    except ValidationError:
        raise UserError(f'The saved Linear key choice is invalid. Choose again with {_RECONFIGURE}.') from None


def saved_key() -> SavedKey:
    """The chosen key, read again on each call so a new choice in the menu reaches the next run."""
    return SavedKey(name=reference().name, setup=f'Restore it in /keys or reconfigure through {_RECONFIGURE}.')


async def choose_key(prompt: SecretPrompt) -> str:
    """Pick a `/keys` entry or enter a new key masked; save the name, and the value only in `/keys`."""
    choice = await prompt_api_key(prompt=prompt, label=f'Linear API key (saved in /keys as {KEY_NAME}): ')
    if choice is None:
        return 'Linear key unchanged.'
    saved = ''
    if isinstance(choice, KeyReference):
        key = choice
    else:
        value = choice.strip()
        if not value:
            raise ValueError('A Linear API key is required.')
        # Checked and saved under one lock, so a key another process just saved is never replaced unasked.
        try:
            saved = await to_thread.run_sync(partial(save_key, name=KEY_NAME, value=value, replace=False))
        except KeyExistsError:
            try:
                answer = await prompt.prompt_async(f'Replace {KEY_NAME} in /keys for every plugin using it? [y/N]: ')
            except (EOFError, KeyboardInterrupt):
                answer = ''
            if answer.strip().lower() != 'y':
                return 'Linear key unchanged.'
            saved = await to_thread.run_sync(partial(save_key, name=KEY_NAME, value=value))
        saved += ' '
        key = KeyReference(name=KEY_NAME)
    value_json = _Connection(token=key).model_dump_json()
    await to_thread.run_sync(partial(save_key_connection, account=ACCOUNT, token=key, value=value_json))
    return f'{saved}Linear uses {key.name} from /keys from the next run.'


def _oauth_client(read_only: bool) -> Client[StreamableHttpTransport]:
    """Connect the way `/mcp` connects an OAuth server: keyring tokens, no redirects, time for a browser sign-in.

    Plain `Linear(auth='oauth')` keeps tokens in memory and allows the 5-second default for `initialize`, which
    a browser sign-in does not fit in. The URL carries `read_only` here: `Linear`'s own `read_only` with a
    `client` filters on tool annotations instead, which would be a second, different boundary.
    """
    server = HTTPServer.model_validate({'type': 'http', 'url': _READ_ONLY_URL if read_only else _URL, 'auth': 'oauth'})
    transport = StreamableHttpTransport(
        url=str(server.url), auth=oauth(TOKEN_ACCOUNT, server), httpx_client_factory=http_client
    )
    return Client(transport, init_timeout=server.init_timeout())
