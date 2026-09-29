"""The built-in `day_ai` plugin: harness `DayAI`, with a token from `/keys` or a browser sign-in.

Settings hold `DayAI`'s non-secret options and at most the name of a `/keys` entry, never a token; the menu that
`/plugins configure day_ai` opens edits them. The browser sign-in works the way `/mcp` does for an OAuth server:
FastMCP's flow, with tokens kept in the keyring under `mcp-day_ai`. `/mcp` server names cannot contain underscores,
so that credential never belongs to one of your servers. The environment is not read.
"""

import asyncio
from dataclasses import replace
from typing import Generic, Literal

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.day_ai import DayAI

from . import theme
from .api_keys import KeyReference, SavedKey, load_keys
from .field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from .mcp import TokenStore, http_client, sign_in
from .menu_worker import run_worker
from .plugin_keys import pick_key_from_menu
from .plugins import DepsT, PluginHost, SessionStart

DAY_AI_MCP_URL = 'https://day.ai/api/mcp'
"""The hosted MCP endpoint harness `DayAI` connects to; Day AI has no other."""

KEY_NAME = 'DAY_AI_ACCESS_TOKEN'
"""The conventional `/keys` label, the variable harness `DayAI` documents. Only a label; not read from the environment."""

TOKEN_ACCOUNT = 'day_ai'
"""The `/mcp` token store name, so the keyring credential is `mcp-day_ai`."""

SETUP = 'Run /plugins configure day_ai to choose a token in /keys or browser sign-in.'
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""

Auth = KeyReference | Literal['oauth']


class DayAISettings(BaseModel):
    """The JSON a `day_ai` declaration may carry: `DayAI`'s non-secret options and the name of its token."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    auth: Auth | None = Field(
        default=None,
        description=f"A saved API key in /keys, or 'oauth' for browser sign-in. Unset uses {KEY_NAME} when saved, "
        'else a stored browser sign-in.',
    )
    include_instructions: bool = Field(default=True, description="Forward the server's instructions to the agent.")


def resolve_auth(settings: DayAISettings) -> Auth | None:
    """What an unset `auth` means now: the conventional key, a stored sign-in, or nothing chosen yet."""
    if settings.auth is not None:
        return settings.auth
    if KEY_NAME in load_keys():
        return KeyReference(name=KEY_NAME)
    return 'oauth' if TokenStore(TOKEN_ACCOUNT).signed_in() else None


def activate(host: PluginHost[DepsT]) -> None:
    """Add `DayAI` with a `/keys` token resolved on every run, or sign in through the browser before loading."""
    settings = host.settings(DayAISettings)
    auth = resolve_auth(settings)

    @host.configure
    async def configure() -> str:
        return await _configure(DayAISource(host))

    if auth is None:
        # Nothing to connect with; loading anyway keeps the menu available, and adds no broken capability.
        host.console.print(f'Day AI is not connected. {SETUP}', style=theme.color(theme.WARNING), markup=False)
        return
    if isinstance(auth, KeyReference):
        host.add(
            DayAI[DepsT](auth=SavedKey(name=auth.name, setup=SETUP), include_instructions=settings.include_instructions)
        )
        if auth.name not in load_keys():
            # Each run fails closed until the key is saved.
            host.console.print(
                f'Day AI has no token: {auth.name} is not in /keys. {SETUP}',
                style=theme.color(theme.WARNING),
                markup=False,
            )
        return
    host.add(DayAI[DepsT](client=_transport(), include_instructions=settings.include_instructions))

    @host.on('session_start')
    async def sign_in(_: SessionStart) -> None:
        if await to_thread.run_sync(TokenStore(TOKEN_ACCOUNT).signed_in):
            return
        if not host.console.is_terminal:
            raise UserError(f'Save {KEY_NAME} in /keys, or sign in to Day AI from an interactive CLAI session first.')
        host.console.print('Opening your browser to sign in to Day AI.', style=theme.color(theme.MUTED))
        # A throwaway connection runs the sign-in now, so a failure fails the load rather than the next prompt.
        async with Client(_transport()):
            pass


_AUTOMATIC = 'automatic'
_KEY = 'key'
_AUTH = FieldRow(
    key='auth',
    label='Sign-in',
    description=(
        'How Day AI connects. A key lives in /keys and plugin settings keep only its name, so any plugin '
        f'naming the same key shares it. Automatic uses {KEY_NAME} when it is saved, else a stored browser sign-in.'
    ),
    default=_AUTOMATIC,
    choices=(_AUTOMATIC, _KEY, 'oauth'),
    choice_labels={
        _AUTOMATIC: 'automatic',
        _KEY: 'choose or enter a key in /keys...',
        'oauth': 'browser sign-in',
    },
    allow_custom=False,
)
_INSTRUCTIONS = FieldRow(
    key='include_instructions',
    label='Server instructions',
    description="Whether the Day AI server's own instructions reach the agent.",
    default='true',
    choices=('true', 'false'),
    choice_labels={'true': 'forwarded', 'false': 'left out'},
    allow_custom=False,
)


class DayAISource(Generic[DepsT]):
    """The settings menu's rows, read from and saved straight to the plugin's settings."""

    title = 'Day AI'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Every edit goes through `host.save_settings`."""
        self._host = host

    @property
    def settings(self) -> DayAISettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(DayAISettings)

    def rows(self) -> list[FieldRow]:
        """Every option, with the sign-in annotated by what it resolves to."""
        return [replace(_AUTH, note=self._auth_note()), _INSTRUCTIONS]

    def _auth_note(self) -> str:
        auth = self.settings.auth
        if isinstance(auth, KeyReference):
            return '' if auth.name in load_keys() else 'missing from /keys'
        if auth == 'oauth':
            return ''
        resolved = resolve_auth(self.settings)
        if resolved is None:
            return 'not connected'
        return f'uses {KEY_NAME}' if isinstance(resolved, KeyReference) else 'uses the stored browser sign-in'

    def current(self, row: FieldRow) -> str:
        """The value as the menu shows it: a key's name, `oauth`, or `automatic`."""
        settings = self.settings
        if row.key == 'auth':
            auth = settings.auth
            return auth.name if isinstance(auth, KeyReference) else (auth or _AUTOMATIC)
        return str(settings.include_instructions).lower()

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
        self.save(DayAISettings.model_validate(data))
        return f'Reset {row.label}.'

    def save(self, settings: DayAISettings) -> None:
        """Persist to the plugin's declaration."""
        self._host.save_settings(settings)

    def _updated(self, row: FieldRow, raw: str) -> DayAISettings:
        data = self.settings.model_dump(mode='json')
        value: JsonValue = raw
        if row.key == 'auth':
            value = None if raw == _AUTOMATIC else raw if raw == 'oauth' else {'name': raw}
        elif raw in ('true', 'false'):
            value = raw == 'true'
        data[row.key] = value
        return DayAISettings.model_validate(data)


async def _configure(source: DayAISource[DepsT]) -> str:
    loop = asyncio.get_running_loop()
    menu = FieldMenu(source)

    def pick_auth() -> list[str]:
        pick = RUNNERS.run_choice(menu.build_choices(_AUTH))
        if pick.cancelled or pick.item is None:
            return []
        if pick.item.value != _KEY:
            return [source.apply(_AUTH, str(pick.item.value))]
        reference = pick_key_from_menu(
            loop,
            name=KEY_NAME,
            label=f'Day AI access token (saved in /keys as {KEY_NAME})',
            placeholder='Paste a Day AI access token; it is saved in /keys',
            runners=RUNNERS,
        )
        if reference is None:
            return []
        source.save(source.settings.model_copy(update={'auth': reference}))
        return [f'Day AI uses the saved key {reference.name}. Manage it in /keys.']

    messages = await run_worker(lambda: run_flow(menu, RUNNERS, submenus={'auth': pick_auth}))
    return '\n'.join(messages) or 'Day AI settings unchanged.'


def _transport() -> StreamableHttpTransport:
    return StreamableHttpTransport(DAY_AI_MCP_URL, auth=sign_in(TOKEN_ACCOUNT), httpx_client_factory=http_client)
