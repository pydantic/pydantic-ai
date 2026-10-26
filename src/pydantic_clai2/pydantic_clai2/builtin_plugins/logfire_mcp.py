"""Logfire: let the agent query your Logfire traces, logs, and metrics through Logfire's MCP server.

The built-in `logfire_mcp` plugin, shown as Logfire: harness `LogfireMCP`, a settings menu, keys kept in `/keys`, and
`/logfire login|logout`. Its id stays `logfire_mcp` because older builds store the `observability` plugin's settings
under `logfire`, so that id cannot be reused; `/logfire_mcp` stays as the command's old name.

Plugin settings are plaintext SQLite, so they hold only the name of a `/keys` entry plus `LogfireMCP`'s
non-secret options, all edited in its declared settings menu: `/plugins configure logfire_mcp` in the terminal, or the
web UI's plugins panel. Which Logfire to connect to is picked as in `observability` setup (see `logfire_destination`),
starting from the one last set up in either.

The browser sign-in runs only from `/logfire login`; see `pydantic_clai2.plugins.sign_in`.
"""

import os
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from urllib.parse import urlsplit

import anyio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import KeyringError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.logfire_mcp import LOGFIRE_US_MCP_URL, LogfireMCP
from pydantic_clai2.builtin_plugins.logfire_destination import (
    REGIONS,
    choose_destination,
    parse_destination,
    remember,
    remembered,
)
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, SavedKey, load_keys
from pydantic_clai2.logfire_oauth import DeviceAuth, LogfireSignIn, forget, status
from pydantic_clai2.mcp import http_client
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, SettingsMenu
from pydantic_clai2.plugins.sign_in import SUBCOMMANDS, run_subcommand, warn_if_signed_out
from pydantic_clai2.runtime._session import WorkspacelessCapability
from pydantic_clai2.ui.menus.field_menu import first_error
from pydantic_clai2.ui.rendering import theme

KEY_NAME = 'LOGFIRE_API_KEY'
"""The conventional `/keys` label, matching the variable `LogfireMCP` reads, so other tools can share one key."""
SETUP = 'Run /plugins configure logfire_mcp to choose or enter a Logfire API key.'
URL = 'url'
"""The row that opens the Logfire picker shared with `observability` setup."""


class LogfireMCPSettings(BaseModel):
    """The JSON a `logfire_mcp` declaration may carry: `LogfireMCP`'s non-secret options and a key's name."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    key: KeyReference | None = Field(default=None, description='The saved API key in /keys to connect with.')
    url: str = Field(
        default=LOGFIRE_US_MCP_URL,
        description='Which Logfire to connect to: a host, the URL you open Logfire at, or its MCP URL; saved as the '
        'MCP URL.',
    )
    oauth: bool = Field(default=True, description='Sign in, or sign up, through the browser when there is no API key.')
    read_only: bool = Field(default=True, description='Offer only the tools the server marks read-only.')
    include_instructions: bool = Field(default=True, description="Forward the server's instructions to the agent.")

    @field_validator('url')
    @classmethod
    def _mcp_url(cls, url: str) -> str:
        parts = urlsplit(url)
        if '://' not in url or not parts.path.strip('/'):
            # A host, or the URL you open Logfire at, names the Logfire whose MCP server is at `/mcp`.
            return parse_destination(url).mcp_url
        # An MCP URL is kept exactly as given, as earlier builds kept it: browser sign-ins are stored under it.
        # A query or fragment would make the URL differ from the resource Logfire signs in for.
        extras = parts.username or parts.password or parts.query or parts.fragment
        if parts.scheme != 'https' or not parts.hostname or extras:
            raise ValueError('Use an https:// URL without credentials, a query, or a fragment.')
        return url


class LogfireMCPPlugin(Plugin[LogfireMCPSettings]):
    """The settings menu and `/logfire` are there on load; the connection is chosen at session start, off the loop."""

    def __init__(self, host: PluginHost[None], settings: LogfireMCPSettings) -> None:
        super().__init__(host, settings)
        self.capability: LogfireMCP[None] | None = None
        """Built by `on_session_start`; runs that start earlier get no Logfire tools."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (WorkspacelessCapability[None](self._for_run),)

    def get_commands(self) -> Sequence[Command]:
        async def command(args: list[str]) -> str:
            return await _command(args, settings=self.settings)

        return (
            Command(
                name='logfire',
                description='Sign in to Logfire for its MCP tools through the browser, sign out, or show the sign-in '
                '(/logfire login|logout|status).',
                handler=command,
                complete=lambda args: SUBCOMMANDS if len(args) <= 1 else (),
            ),
            Command(
                name='logfire_mcp',
                description='The old name of /logfire.',
                handler=command,
                complete=lambda args: SUBCOMMANDS if len(args) <= 1 else (),
            ),
        )

    def settings_menu(self) -> SettingsMenu:
        source = LogfireMCPSource(self.host)
        return SettingsMenu(
            source=source, actions={'key': partial(_pick_key, source=source), URL: partial(_pick_url, source=source)}
        )

    async def configure(self) -> str:
        # Only the terminal menu has a moment it opens, so only it switches to the Logfire last set up; see `_seed`.
        source = LogfireMCPSource(self.host)
        seeded = await anyio.to_thread.run_sync(_seed, source) if self.host.console.is_terminal else None
        shown = await super().configure()
        if seeded is None:
            return shown
        return seeded if shown == f'{source.title}: nothing changed.' else f'{seeded}\n{shown}'

    async def on_session_start(self, event: SessionStart) -> None:
        # A worker thread: `/keys` takes a lock another CLAI process can hold, and the keyring can block.
        self.capability, missing = await anyio.to_thread.run_sync(
            partial(_capability, settings=self.settings), abandon_on_cancel=True
        )
        if self.capability.client is not None:
            await warn_if_signed_out(_sign_in(self.settings), self.host.console)
        if missing is not None:
            # Loading anyway keeps the settings menu available; each run fails closed until the key is saved.
            self.host.console.print(
                f'Logfire has no credential: {missing} is not in /keys. {SETUP}',
                style=theme.color(theme.WARNING),
                markup=False,
            )

    async def _for_run(self, ctx: RunContext[None]) -> LogfireMCP[None] | None:
        capability = self.capability
        if capability is not None and capability.client is not None:
            # Browser sign-in: no tools until `/logfire login`, which applies to the next prompt.
            signed_in = await anyio.to_thread.run_sync(_sign_in(self.settings).signed_in, abandon_on_cancel=True)
            return capability if signed_in else None
        if capability is None or not isinstance(capability.auth, SavedKey):
            return capability
        # Resolved here, not by `LogfireMCP`: a run would not set up its per-run toolset nested in this factory's.
        token = await anyio.to_thread.run_sync(capability.auth, ctx, abandon_on_cancel=True)
        return replace(capability, auth=token)


def _sign_in(settings: LogfireMCPSettings) -> LogfireSignIn:
    return LogfireSignIn(resource=settings.url, read_only=settings.read_only)


def _capability(*, settings: LogfireMCPSettings) -> tuple[LogfireMCP[None], str | None]:
    """The first of: chosen key, `LOGFIRE_API_KEY` env, `/keys` `LOGFIRE_API_KEY`, then browser sign-in.

    Blocking; returns the name of the `/keys` entry the capability needs when it is missing.
    """

    def build(
        *, auth: SavedKey | None = None, client: Client[StreamableHttpTransport] | None = None
    ) -> LogfireMCP[None]:
        return LogfireMCP[None](
            auth=auth,
            client=client,
            url=LOGFIRE_US_MCP_URL if client else settings.url,
            read_only=settings.read_only,
            include_instructions=settings.include_instructions,
        )

    if settings.key is None and os.environ.get(KEY_NAME):
        return build(), None
    saved = load_keys()
    if settings.key is None and KEY_NAME not in saved and settings.oauth:
        return build(client=_oauth_client(settings=settings)), None
    # Resolved on every run, so saving the key in /keys connects without a reload.
    name = settings.key.name if settings.key is not None else KEY_NAME
    return build(auth=SavedKey(name=name, setup=SETUP)), None if name in saved else name


def _oauth_client(*, settings: LogfireMCPSettings) -> Client[StreamableHttpTransport]:
    """The stored device-flow sign-in, refreshed as needed; `/logfire login` makes one."""
    auth = DeviceAuth(resource=settings.url, read_only=settings.read_only)
    transport = StreamableHttpTransport(settings.url, auth=auth, httpx_client_factory=http_client)
    return Client(transport)


async def _command(args: list[str], *, settings: LogfireMCPSettings) -> str:
    """`login` signs in (or up) now, `status` says whether runs can use it, `logout` forgets every sign-in."""
    if args == ['logout']:
        try:
            forgotten = await anyio.to_thread.run_sync(forget, abandon_on_cancel=True)
        except (KeyringError, OSError) as exc:
            raise ValueError(
                f'Could not delete the saved Logfire browser sign-in ({type(exc).__name__}); run /logfire logout to retry.'
            ) from None
        if forgotten:
            return 'Forgot the Logfire browser sign-in. Keys in /keys are kept.'
        return 'There was no Logfire browser sign-in to forget.'
    message = await run_subcommand(_sign_in(settings), args)
    if message is None:
        raise ValueError(
            'Usage: /logfire [login | logout | status] (settings and keys: /plugins configure logfire_mcp)'
        )
    return message


_KEY = FieldRow(
    key='key',
    label='API key',
    # One short line each: the preview panel cuts long lines off.
    description='\n'.join(
        [
            'The API key Logfire connects with.',
            'Enter: pick a saved key in /keys, or paste',
            f'  a new one to save as {KEY_NAME}.',
            f'R: no chosen key: use {KEY_NAME} from the',
            '  environment or /keys, then browser sign-in.',
            'Only the key name is kept here, so any plugin',
            'naming the same key shares it.',
        ]
    ),
    default='(none)',
)
_ROWS = (
    _KEY,
    FieldRow(
        key=URL,
        label='Which Logfire',
        description='\n'.join(
            [
                'The Logfire your data lives in.',
                'Enter: choose Logfire US or EU, or type another',
                '  (self-hosted or staging) by its host, URL,',
                '  or MCP URL.',
                'R: back to Logfire US.',
                'A Logfire picked with Enter is where',
                '  observability setup starts, if it has none.',
            ]
        ),
        default=LOGFIRE_US_MCP_URL,
        choice_labels={region.mcp_url: name for name, region in REGIONS.items()},
    ),
    FieldRow(
        key='read_only',
        label='Tools',
        description="Read-only keeps the agent from changing Logfire resources. The key's scopes still apply.",
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'read-only', 'false': 'read and write'},
        allow_custom=False,
    ),
    FieldRow(
        key='include_instructions',
        label='Server instructions',
        description="Whether the server's instructions, query guidance, and the current UTC time reach the agent.",
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'forwarded', 'false': 'left out'},
        allow_custom=False,
    ),
    FieldRow(
        key='oauth',
        label='Browser sign-in',
        description=(
            'Sign in, or sign up, through the browser when no API key is chosen, set, or saved. /logfire login '
            'shows a link and a code that also work from another device. Tokens stay in the OS keyring.'
        ),
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'when there is no key', 'false': 'off'},
        allow_custom=False,
    ),
)
_FLAGS = ('read_only', 'include_instructions', 'oauth')


class LogfireMCPSource:
    """The settings menu's rows, read from and saved straight to the plugin's settings."""

    title = 'Logfire'

    def __init__(self, host: PluginHost[None]) -> None:
        """Every edit goes through `host.save_settings`."""
        self._host = host

    @property
    def settings(self) -> LogfireMCPSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(LogfireMCPSettings)

    def rows(self) -> list[FieldRow]:
        """Every option, with the key marked when it is gone from `/keys`."""
        settings = self.settings
        signed = status(resource=settings.url, read_only=settings.read_only)
        notes = {
            'key': _key_note(settings.key),
            'oauth': signed if signed == 'signed in' else f'{signed}: /logfire login signs in',
        }
        return [replace(row, note=notes.get(row.key, '')) for row in _ROWS]

    def current(self, row: FieldRow) -> str:
        """The value as the user would type it."""
        settings = self.settings
        if row.key == 'key':
            return settings.key.name if settings.key else _KEY.default
        value: object = getattr(settings, row.key)
        return str(value).lower() if isinstance(value, bool) else str(value)

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Validate against the whole settings model, as saving would."""
        try:
            self._updated(row, text)
        except ValidationError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save immediately; the loader loads the plugin again when the menu closes."""
        self.save(self._updated(row, raw))
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        """Restore one option's default."""
        data = self.settings.model_dump(mode='json')
        del data[row.key]
        self.save(LogfireMCPSettings.model_validate(data))
        return f'Reset {row.label}.'

    def save(self, settings: LogfireMCPSettings) -> None:
        """Persist to the plugin's declaration."""
        self._host.save_settings(settings)

    def _updated(self, row: FieldRow, raw: str) -> LogfireMCPSettings:
        data = self.settings.model_dump(mode='json')
        value: JsonValue = raw
        if row.key in _FLAGS and raw in ('true', 'false'):
            value = raw == 'true'
        data[row.key] = value
        return LogfireMCPSettings.model_validate(data)


def _key_note(key: KeyReference | None) -> str:
    if key is not None:
        return 'missing from /keys' if key.name not in load_keys() else ''
    if os.environ.get(KEY_NAME):
        return f'{KEY_NAME} from the environment'
    return f'{KEY_NAME} from /keys' if KEY_NAME in load_keys() else 'browser sign-in, if on'


async def _pick_key(dialogs: Dialogs, /, *, source: LogfireMCPSource) -> list[str]:
    """The API key row: a key saved in `/keys`, or a new one saved there as `LOGFIRE_API_KEY`."""
    key = await choose_key(name=KEY_NAME, label=f'Logfire API key (saved in /keys as {KEY_NAME})', dialogs=dialogs)
    if key is None:
        return []
    await anyio.to_thread.run_sync(source.save, source.settings.model_copy(update={'key': key}))
    return [f'Logfire uses the saved key {key.name}. Manage it in /keys.']


async def _pick_url(dialogs: Dialogs, /, *, source: LogfireMCPSource) -> list[str]:
    """The Which Logfire row: the picker `observability` setup shares, starting from the Logfire last set up."""
    settings = source.settings
    # Until this plugin has any saved setting, start from the Logfire last set up in either plugin.
    last = None if settings.model_fields_set else await anyio.to_thread.run_sync(remembered)
    try:
        current = parse_destination(settings.url)
    except ValueError:
        current = None  # An MCP URL at another path, as earlier builds took, has no row in the picker.
    destination = await choose_destination(dialogs, current=last or current, then=_THEN)
    if destination is None:
        return []
    if destination != current:
        # Picking the Logfire already saved keeps its URL as it is: browser sign-ins are stored under it.
        await anyio.to_thread.run_sync(source.save, settings.model_copy(update={URL: destination.mcp_url}))
    await anyio.to_thread.run_sync(remember, destination)
    return [f'Logfire tools connect to {destination.label} at {source.settings.url}.']


_THEN = "The agent's Logfire tools connect to the\nchosen one, with the API key or browser\nsign-in set in this menu."


def _seed(source: LogfireMCPSource) -> str | None:
    """Before the terminal menu opens, connect to the Logfire last set up, if this plugin has no saved setting yet.

    Only a hosted region is switched to: any other Logfire is only highlighted in the picker, so the key this
    plugin already has never reaches a server the user did not confirm for it. Blocking: reads a file, saves.
    """
    settings = source.settings
    last = None if settings.model_fields_set else remembered()
    if last is None or last.region is None or last.mcp_url == settings.url:
        return None
    source.save(settings.model_copy(update={URL: last.mcp_url}))
    return f'Logfire tools connect to {last.label}, the Logfire you last set up. Change it under Which Logfire.'
