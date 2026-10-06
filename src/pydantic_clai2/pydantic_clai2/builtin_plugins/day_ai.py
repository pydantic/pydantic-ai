"""Use Day AI, signed in in the browser or with a token from /keys.

The built-in `day_ai` plugin: harness `DayAI`, with a browser sign-in by default or a token from `/keys`.

Settings hold `DayAI`'s non-secret options and at most the name of a `/keys` entry, never a token; the menu that
`/plugins configure day_ai` opens edits them. The browser sign-in works the way `/mcp` does for an OAuth server:
FastMCP's flow, with tokens kept in the keyring under `mcp-day_ai`. `/mcp` server names cannot contain underscores,
so that credential never belongs to one of your servers. The environment is not read.

The sign-in runs only from that menu, where Esc cancels it. Loading never opens the browser: the session waits for
every plugin to load, so a sign-in nobody finishes would leave CLAI unable to run anything. Until it is done, runs
leave Day AI out, and the first run after it uses it without a reload.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import Generic, Literal

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai_harness.day_ai import DayAI
from pydantic_clai2.config.api_keys import KeyReference, SavedKey, load_keys
from pydantic_clai2.mcp import OAUTH_TIMEOUT, TokenStore, http_client, sign_in
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import choose_key, on_loop, wait_for_sign_in
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.rendering import theme

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
    auth: Auth = Field(
        default='oauth', description="'oauth' (the default) for browser sign-in, or a saved API key in /keys."
    )
    include_instructions: bool = Field(default=True, description="Forward the server's instructions to the agent.")

    @field_validator('auth', mode='before')
    @classmethod
    def _automatic_is_browser(cls, value: object) -> object:
        """Earlier builds saved `null` for an automatic choice; it now means the browser sign-in default."""
        return 'oauth' if value is None else value


class DayAIPlugin(Plugin[DayAISettings, DepsT]):
    """`DayAI` with a browser sign-in made in the settings menu, or a `/keys` token resolved on every run."""

    def __init__(self, host: PluginHost[DepsT], settings: DayAISettings) -> None:
        super().__init__(host, settings)
        self.tokens = TokenStore(TOKEN_ACCOUNT)

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        auth, include_instructions = self.settings.auth, self.settings.include_instructions
        if isinstance(auth, KeyReference):
            return (
                DayAI[DepsT](auth=SavedKey(name=auth.name, setup=SETUP), include_instructions=include_instructions),
            )
        browser = DayAI[DepsT](client=_transport(), include_instructions=include_instructions)

        async def signed_in(_: RunContext[DepsT]) -> DayAI[DepsT] | None:
            # Checked per run, so a sign-in finished in the menu applies without a reload, and a run never
            # connects without one: that would open the browser mid-turn.
            return browser if await to_thread.run_sync(self.tokens.signed_in) else None

        return (signed_in,)

    async def configure(self) -> str:
        return await _configure(DayAISource(self.host))

    async def on_session_start(self, event: SessionStart) -> None:
        console = self.host.console
        auth = self.settings.auth
        if isinstance(auth, KeyReference):
            if auth.name not in load_keys():
                # Each run fails closed until the key is saved.
                console.print(
                    f'Day AI has no token: {auth.name} is not in /keys. {SETUP}',
                    style=theme.color(theme.WARNING),
                    markup=False,
                )
            return
        if not await to_thread.run_sync(self.tokens.signed_in):
            console.print(f'Day AI is not signed in. {SETUP}', style=theme.color(theme.WARNING), markup=False)


_KEY = 'key'
_AUTH = FieldRow(
    key='auth',
    label='Sign-in',
    description=(
        'How Day AI connects. Browser sign-in, the default, keeps its tokens in the keyring. A key lives in /keys '
        'and plugin settings keep only its name, so any plugin naming the same key shares it.'
    ),
    default='oauth',
    choices=('oauth', _KEY),
    choice_labels={'oauth': 'browser sign-in', _KEY: 'choose or enter a key in /keys...'},
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
        """Every option, with a chosen key marked when it is gone from `/keys`."""
        auth = self.settings.auth
        missing = isinstance(auth, KeyReference) and auth.name not in load_keys()
        return [replace(_AUTH, note='missing from /keys') if missing else _AUTH, _INSTRUCTIONS]

    def current(self, row: FieldRow) -> str:
        """The value as the menu shows it: a key's name or `oauth`."""
        settings = self.settings
        if row.key == 'auth':
            auth = settings.auth
            return auth.name if isinstance(auth, KeyReference) else auth
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
            value = raw if raw == 'oauth' else {'name': raw}
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
        if pick.item.value == 'oauth':
            return [source.apply(_AUTH, 'oauth'), *sign_in_now()]
        label = f'Day AI access token (saved in /keys as {KEY_NAME})'
        reference = on_loop(lambda: choose_key(name=KEY_NAME, label=label, runners=RUNNERS), loop)
        if reference is None:
            return []
        source.save(source.settings.model_copy(update={'auth': reference}))
        return [f'Day AI uses the saved key {reference.name}. Manage it in /keys.']

    def sign_in_now() -> list[str]:
        if TokenStore(TOKEN_ACCOUNT).signed_in():
            return []
        try:
            signed_in = on_loop(
                lambda: wait_for_sign_in(_sign_in(), service='Day AI', note=_WAITING, runners=RUNNERS), loop
            )
        except Exception as exc:  # noqa: BLE001 -- FastMCP fails in many ways; each leaves Day AI signed out.
            return [f'Could not sign in to Day AI: {exc}. {SETUP}']
        if not signed_in:
            return [f'Day AI sign-in cancelled. {SETUP}']
        return ['Signed in to Day AI. Tokens are kept in the OS credential store and renew themselves.']

    messages = await run_worker(lambda: run_flow(menu, RUNNERS, submenus={'auth': pick_auth}))
    return '\n'.join(messages) or 'Day AI settings unchanged.'


_WAITING = 'Finish in the browser window that opened. Esc cancels; CLAI keeps working without Day AI.'


async def _sign_in() -> None:
    """Connect once so FastMCP signs in through the browser and stores the tokens."""
    async with Client(_transport(), init_timeout=OAUTH_TIMEOUT):
        pass


def _transport() -> StreamableHttpTransport:
    return StreamableHttpTransport(DAY_AI_MCP_URL, auth=sign_in(TOKEN_ACCOUNT), httpx_client_factory=http_client)
