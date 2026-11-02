"""Use Day AI, signed in in the browser or with a token from /keys.

The built-in `day_ai` plugin: harness `DayAI`, with a browser sign-in by default or a token from `/keys`.

Settings hold `DayAI`'s non-secret options and at most the name of a `/keys` entry, never a token; the menu that
`/plugins configure day_ai` opens, also shown in the web UI's plugins panel, edits them. The browser sign-in works the
way `/mcp` does for an OAuth server: FastMCP's flow, with tokens kept in the keyring under `mcp-day_ai`. `/mcp` server
names cannot contain underscores, so that credential never belongs to one of your servers. The environment is not read.

The sign-in runs only from that menu or `/day_ai login`, never while loading or in a prompt; see
`pydantic_clai2.plugins.sign_in`.
"""

from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import Generic, Literal

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.day_ai import DayAI
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, SavedKey, load_keys
from pydantic_clai2.mcp import OAuthSignIn
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.plugins.sign_in import sign_in_command, sign_in_now, warn_if_signed_out
from pydantic_clai2.runtime.session import WorkspacelessCapability
from pydantic_clai2.ui.menus.field_menu import first_error
from pydantic_clai2.ui.rendering import theme

DAY_AI_MCP_URL = 'https://day.ai/api/mcp'
"""The hosted MCP endpoint harness `DayAI` connects to; Day AI has no other."""

KEY_NAME = 'DAY_AI_ACCESS_TOKEN'
"""The conventional `/keys` label, the variable harness `DayAI` documents. Only a label; not read from the environment."""

TOKEN_ACCOUNT = 'day_ai'
"""The `/mcp` token store name, so the keyring credential is `mcp-day_ai`."""

SETUP = 'Run /plugins configure day_ai to choose a token in /keys or browser sign-in.'
SIGN_IN = OAuthSignIn(name=TOKEN_ACCOUNT, service='Day AI', setup='/day_ai login', url=DAY_AI_MCP_URL)
"""The browser sign-in; its tokens are the keyring credential `mcp-day_ai`."""

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
    """`DayAI` with a browser sign-in made from the menu or `/day_ai login`, or a `/keys` token resolved per run."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        auth = self.settings.auth
        if isinstance(auth, KeyReference):
            return (
                DayAI[DepsT](
                    defer_loading=True,
                    auth=SavedKey(name=auth.name, setup=SETUP),
                    include_instructions=self.settings.include_instructions,
                ),
            )
        return (WorkspacelessCapability[DepsT](self.for_run),)

    def get_commands(self) -> Sequence[Command]:
        return (sign_in_command('day_ai', SIGN_IN),)

    async def for_run(self, ctx: RunContext[DepsT]) -> DayAI[DepsT] | None:
        # Checked per run, so signing in from the menu or `/day_ai login` applies to the next prompt.
        if not await to_thread.run_sync(SIGN_IN.signed_in, abandon_on_cancel=True):
            return None
        return DayAI[DepsT](
            defer_loading=True, client=SIGN_IN.client(), include_instructions=self.settings.include_instructions
        )

    def settings_menu(self) -> SettingsMenu:
        source = DayAISource(self.host)
        return SettingsMenu(source=source, actions={'auth': partial(_choose_auth, source)})

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
        await warn_if_signed_out(SIGN_IN, console)


_KEY = 'key'
_AUTH = FieldRow(
    key='auth',
    label='Sign-in',
    description=(
        'How Day AI connects. Browser sign-in, the default, keeps its tokens in the keyring; choosing it signs in '
        'now if needed, and /day_ai login signs in again later. A key lives in /keys and plugin settings keep only '
        'its name, so any plugin naming the same key shares it.'
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


async def _choose_auth(source: DayAISource[DepsT], dialogs: Dialogs) -> list[str]:
    """The Sign-in row: browser sign-in, signing in now if needed, or a key picked or entered in `/keys`."""
    picked = await dialogs.choose(
        title=f'Choose {_AUTH.label}',
        options=[Option(value=choice, label=_AUTH.display(choice)) for choice in _AUTH.choices],
        current=await to_thread.run_sync(source.current, _AUTH),
    )
    if picked is None:
        return []
    if picked == 'oauth':
        saved = await to_thread.run_sync(source.apply, _AUTH, 'oauth')
        if await to_thread.run_sync(SIGN_IN.signed_in, abandon_on_cancel=True):
            return [saved]
        return [saved, await sign_in_now(SIGN_IN, dialogs)]
    label = f'Day AI access token (saved in /keys as {KEY_NAME})'
    reference = await choose_key(name=KEY_NAME, label=label, dialogs=dialogs)
    if reference is None:
        return []
    await to_thread.run_sync(lambda: source.save(source.settings.model_copy(update={'auth': reference})))
    return [f'Day AI uses the saved key {reference.name}. Manage it in /keys.']
