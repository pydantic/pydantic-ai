"""Query your Logfire data from the agent through the Logfire MCP server.

The built-in `logfire_mcp` plugin: harness `LogfireMCP`, a settings menu, and keys kept in `/keys`.

Plugin settings are plaintext SQLite, so they hold only the name of a `/keys` entry plus `LogfireMCP`'s
non-secret options, all edited in the menu that `/plugins configure logfire_mcp` opens.
"""

import asyncio
import os
import threading
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from urllib.parse import urlsplit

import anyio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from keyring.errors import KeyringError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.logfire_mcp import LOGFIRE_EU_MCP_URL, LOGFIRE_US_MCP_URL, LogfireMCP
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyExistsError, KeyReference, SavedKey, load_keys, prompt_api_key, save_key
from pydantic_clai2.logfire_oauth import SIGN_IN_TIMEOUT, Announce, DeviceAuth, SignInError, forget, status
from pydantic_clai2.mcp import http_client
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker, worker_stopping
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style
from pydantic_clai2.ui.rendering.tool_output import terminal_text

KEY_NAME = 'LOGFIRE_API_KEY'
"""The conventional `/keys` label, matching the variable `LogfireMCP` reads, so other tools can share one key."""
SETUP = 'Run /plugins configure logfire_mcp to choose or enter a Logfire API key.'
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""


class LogfireMCPSettings(BaseModel):
    """The JSON a `logfire_mcp` declaration may carry: `LogfireMCP`'s non-secret options and a key's name."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    key: KeyReference | None = Field(default=None, description='The saved API key in /keys to connect with.')
    url: str = Field(default=LOGFIRE_US_MCP_URL, description='Hosted US, hosted EU, or self-hosted MCP endpoint.')
    oauth: bool = Field(default=True, description='Sign in, or sign up, through the browser when there is no API key.')
    read_only: bool = Field(default=True, description='Offer only the tools the server marks read-only.')
    include_instructions: bool = Field(default=True, description="Forward the server's instructions to the agent.")

    @field_validator('url')
    @classmethod
    def _https(cls, url: str) -> str:
        parts = urlsplit(url)
        # A query or fragment would make the URL differ from the resource Logfire signs in for.
        extras = parts.username or parts.password or parts.query or parts.fragment
        if parts.scheme != 'https' or not parts.hostname or extras:
            raise ValueError('Use an https:// URL without credentials, a query, or a fragment.')
        return url


class LogfireMCPPlugin(Plugin[LogfireMCPSettings]):
    """The settings menu and `/logfire_mcp` are there on load; the connection is chosen at session start, off the loop."""

    def __init__(self, host: PluginHost[None], settings: LogfireMCPSettings) -> None:
        super().__init__(host, settings)
        self.capability: LogfireMCP[None] | None = None
        """Built by `on_session_start`; runs that start earlier get no Logfire MCP tools."""

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (self._for_run,)

    def get_commands(self) -> Sequence[Command]:
        async def command(args: list[str]) -> str:
            return await _command(args, settings=self.settings, announce=self._announce)

        return (
            Command(
                name='logfire_mcp',
                description='Sign in to Logfire through the browser, or forget that sign-in (/logfire_mcp login|logout).',
                handler=command,
                complete=lambda _: ('login', 'logout'),
            ),
        )

    async def configure(self) -> str:
        return await _configure(LogfireMCPSource(self.host))

    async def on_session_start(self, event: SessionStart) -> None:
        # A worker thread: `/keys` takes a lock another CLAI process can hold, and the keyring can block.
        self.capability, missing = await anyio.to_thread.run_sync(
            partial(_capability, settings=self.settings, announce=self._announce), abandon_on_cancel=True
        )
        if missing is not None:
            # Loading anyway keeps the settings menu available; each run fails closed until the key is saved.
            self.host.console.print(
                f'Logfire MCP has no credential: {missing} is not in /keys. {SETUP}',
                style=theme.color(theme.WARNING),
                markup=False,
            )

    async def _for_run(self, ctx: RunContext[None]) -> LogfireMCP[None] | None:
        capability = self.capability
        if capability is None or not callable(capability.auth):
            return capability
        # Resolved here, not by `LogfireMCP`: a run would not set up its per-run toolset nested in this factory's.
        token = await anyio.to_thread.run_sync(capability.auth, ctx, abandon_on_cancel=True)
        return replace(capability, auth=token)

    def _announce(self, line: str) -> None:
        # Links, codes, and notices can carry text from a self-hosted server, so terminal controls are made inert.
        self.host.console.print(terminal_text(line), markup=False, highlight=False)


def _capability(*, settings: LogfireMCPSettings, announce: Announce) -> tuple[LogfireMCP[None], str | None]:
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
        return build(client=_oauth_client(settings=settings, announce=announce)), None
    # Resolved on every run, so saving the key in /keys connects without a reload.
    name = settings.key.name if settings.key is not None else KEY_NAME
    return build(auth=SavedKey(name=name, setup=SETUP)), None if name in saved else name


def _oauth_client(*, settings: LogfireMCPSettings, announce: Announce) -> Client[StreamableHttpTransport]:
    """Device-flow sign-in, started by the first connection that has no usable token."""
    auth = DeviceAuth(resource=settings.url, read_only=settings.read_only, announce=announce)
    transport = StreamableHttpTransport(settings.url, auth=auth, httpx_client_factory=http_client)
    return Client(transport, init_timeout=SIGN_IN_TIMEOUT)


async def _command(args: list[str], *, settings: LogfireMCPSettings, announce: Announce) -> str:
    """`/logfire_mcp login` signs in (or up) now; `/logfire_mcp logout` forgets every sign-in."""
    if args == ['login']:
        try:
            await DeviceAuth(resource=settings.url, read_only=settings.read_only, announce=announce).sign_in()
        except SignInError as exc:
            raise ValueError(str(exc)) from None
        return 'Logfire runs use this sign-in when no API key is chosen, set, or saved.'
    if args == ['logout']:
        try:
            forgotten = await anyio.to_thread.run_sync(forget, abandon_on_cancel=True)
        except (KeyringError, OSError) as exc:
            raise ValueError(
                f'Could not delete the saved Logfire browser sign-in ({type(exc).__name__}); run /logfire_mcp logout to retry.'
            ) from None
        if forgotten:
            return 'Forgot the Logfire browser sign-in. Keys in /keys are kept.'
        return 'There was no Logfire browser sign-in to forget.'
    raise ValueError('Usage: /logfire_mcp login|logout (settings and keys: /plugins configure logfire_mcp)')


_KEY = FieldRow(
    key='key',
    label='API key',
    description=(
        f'The saved key in /keys that Logfire connects with. Enter picks a saved key or saves a new one as {KEY_NAME}; '
        f'plugin settings keep only its name. Unset uses {KEY_NAME} from the environment or /keys, then browser '
        'sign-in. Any plugin naming the same key shares it.'
    ),
    default='(none)',
)
_ROWS = (
    _KEY,
    FieldRow(
        key='url',
        label='Destination',
        description='The Logfire region your data lives in, or the MCP URL of a self-hosted Logfire.',
        default=LOGFIRE_US_MCP_URL,
        choices=(LOGFIRE_US_MCP_URL, LOGFIRE_EU_MCP_URL),
        choice_labels={LOGFIRE_US_MCP_URL: 'Logfire US', LOGFIRE_EU_MCP_URL: 'Logfire EU'},
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
            'Sign in, or sign up, through the browser when no API key is chosen, set, or saved. The first run, or '
            '/logfire_mcp login, shows a link and a code that also work from another device. Tokens stay in the '
            'OS keyring.'
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

    title = 'Logfire MCP'

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
            'oauth': signed if signed == 'signed in' else f'{signed}: signs in on the next run',
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


async def _configure(source: LogfireMCPSource) -> str:
    loop = asyncio.get_running_loop()

    def pick_key() -> list[str]:
        # The key picker is async, so the menu's thread hands it back to the event loop. Its widgets
        # watch their own stop signal, so cancelling this worker must cancel the picker explicitly, then
        # wait until its own workers have released the terminal before this one does.
        finished = threading.Event()

        async def start() -> asyncio.Task[KeyReference | str | None]:
            task = asyncio.create_task(_choose_key())
            task.add_done_callback(lambda _: finished.set())
            return task

        picking = asyncio.run_coroutine_threadsafe(start(), loop).result()
        while not (finished.is_set() or worker_stopping()):
            finished.wait(timeout=0.05)
        if not finished.is_set():
            loop.call_soon_threadsafe(picking.cancel)
            finished.wait()
            return []
        choice = picking.result()
        if choice is None:
            return []
        key = choice if isinstance(choice, KeyReference) else None
        source.save(source.settings.model_copy(update={'key': key}))
        if key is None:
            return [f'Logfire uses {KEY_NAME} from the environment or /keys, then browser sign-in.']
        return [f'Logfire uses the saved key {key.name}. Manage it in /keys.']

    # Built in the worker: its rows read `/keys` (a cross-process lock) and the keyring.
    messages = await run_worker(lambda: run_flow(FieldMenu(source), RUNNERS, submenus={'key': pick_key}))
    return '\n'.join(messages) or 'Logfire MCP settings unchanged.'


class _MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked termflow input, matching the settings menu."""

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        builder = (
            TextInputBuilder(label)
            .style(markdown_style())
            .prompt('Key: ')
            .placeholder('Paste a Logfire API key; it is saved in /keys')
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: RUNNERS.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            raise EOFError  # `prompt_api_key` reads this as cancellation.
        return result.value


async def _choose_key() -> KeyReference | None | str:
    """A saved key, `''` for no key, or `None` when cancelled; a typed value is saved as `LOGFIRE_API_KEY`."""
    label = f'Logfire API key (saved in /keys as {KEY_NAME})'
    choice = await prompt_api_key(prompt=_MaskedPrompt(), label=label, optional=True)
    if choice is None or isinstance(choice, KeyReference) or choice == '':
        return choice
    value = choice.strip()
    if not value:
        return None
    # Added only if absent, atomically, so a key another CLAI process just saved is never replaced unasked.
    try:
        await anyio.to_thread.run_sync(
            partial(save_key, name=KEY_NAME, value=value, replace=False), abandon_on_cancel=True
        )
    except KeyExistsError:
        if not await run_worker(_confirm_replace):
            return None
        await anyio.to_thread.run_sync(partial(save_key, name=KEY_NAME, value=value), abandon_on_cancel=True)
    return KeyReference(name=KEY_NAME)


def _confirm_replace() -> bool:
    menu = (
        MenuBuilder(f'{KEY_NAME} is already in /keys')
        .style(markdown_style())
        .items(
            [
                MenuItem('Keep the saved key', value=False),
                MenuItem(f'Replace {KEY_NAME} for every plugin and connection that uses it', value=True),
            ]
        )
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    pick = RUNNERS.run_choice(menu)
    return not pick.cancelled and pick.item is not None and pick.item.value is True
