"""Use Notion, signed in with a key from /keys or in the browser.

The built-in `notion` plugin: harness `Notion`, connected with a named key from `/keys` or a browser sign-in.

Plugin settings are plaintext SQLite, so they hold only `Notion`'s non-secret options, all edited in the menu
`/plugins configure notion` opens. Its key row picks or enters a key in the named keystore and saves only the
name, in CLAI's credential store; each run resolves it again, so replacing the key in `/keys` reaches every
plugin that shares it, and a deleted key fails the run rather than connecting.

Without a chosen key, Notion uses a browser sign-in kept in the keyring under `mcp-plugin_notion`. The sign-in runs
only from that menu or `/notion login`, never while loading or in a prompt; see `pydantic_clai2.plugins.sign_in`.
"""

import asyncio
from collections.abc import Sequence
from functools import partial
from typing import Generic, Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.notion import Notion
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, resolve_key, save_key_connection
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials
from pydantic_clai2.mcp import OAuthSignIn
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import on_loop
from pydantic_clai2.plugins.sign_in import (
    SUBCOMMANDS,
    run_subcommand,
    sign_in_now,
    sign_out_now,
    warn_if_signed_out,
)
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from pydantic_clai2.ui.menus.key_picker import pick_key
from pydantic_clai2.ui.menus.menu_worker import run_worker
from pydantic_clai2.ui.rendering import theme

NOTION_MCP_URL = 'https://mcp.notion.com/mcp'
KEY_NAME = 'NOTION_API_KEY'
"""The `/keys` label a newly entered token is saved under, so other Notion consumers can share it."""
ACCOUNT = 'notion'
"""The credential account holding the selected key's name, never its value."""
TOKEN_ACCOUNT = 'plugin_notion'
"""The browser sign-in's token store. `/mcp` server names cannot contain `_`, so it never collides with one."""
SETUP = 'Choose one with /plugins configure notion.'
SIGN_IN = OAuthSignIn(name=TOKEN_ACCOUNT, service='Notion', setup='/notion login', url=NOTION_MCP_URL)
"""The browser sign-in; its tokens are the keyring credential `mcp-plugin_notion`."""
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""


class NotionSettings(BaseModel):
    """The JSON a `notion` declaration may carry. Secrets are not accepted here; they live in `/keys`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    auth: Literal['key', 'oauth'] | None = Field(
        default=None,
        description='`key` uses only the chosen key; `oauth` uses only the browser sign-in; unset uses the chosen '
        'key when there is one, else the browser sign-in.',
    )
    read_only: bool = Field(default=False, description='Keep only the tools the server marks as read-only.')
    include_instructions: bool = Field(default=True, description="Forward the server's instructions to the agent.")


class _Selection(BaseModel):
    token: KeyReference


def selected_key() -> KeyReference | None:
    """The key chosen in the settings menu, or `None`. Only the name is stored."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return None
    try:
        return _Selection.model_validate_json(raw).token
    except ValidationError:
        raise UserError(f'The saved Notion key selection is invalid. {SETUP}') from None


def select_key(reference: KeyReference) -> None:
    """Remember `reference` as Notion's key, under the `/keys` lock so it cannot race a rename."""
    save_key_connection(account=ACCOUNT, token=reference, value=_Selection(token=reference).model_dump_json())


class NotionPlugin(Plugin[NotionSettings, DepsT]):
    """Choose the connection per run, so a new key, a sign-in, or a logout applies without a reload."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (self._connect,)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='notion',
                description='Sign in to Notion in the browser, sign out, or show how it connects '
                '(/notion login|logout|status).',
                handler=self._command,
                complete=lambda args: SUBCOMMANDS if len(args) <= 1 else (),
            ),
        )

    async def configure(self) -> str:
        return await _configure(NotionSource(self.host))

    async def on_session_start(self, event: SessionStart) -> None:
        # Here rather than on load, so the keyring read does not block the shell's loop.
        auth = self.settings.auth
        if auth != 'oauth':
            try:
                reference = await anyio.to_thread.run_sync(selected_key, abandon_on_cancel=True)
            except UserError:
                return  # Each run reports the invalid selection.
            if reference is not None:
                return
            if auth == 'key':
                self.host.console.print(
                    f'Notion has no key selected, so runs fail. {SETUP}',
                    style=theme.color(theme.WARNING),
                    markup=False,
                )
                return
        await warn_if_signed_out(SIGN_IN, self.host.console)

    async def _connect(self, _: RunContext[DepsT]) -> Notion[DepsT] | None:
        settings = self.settings
        reference = (
            None if settings.auth == 'oauth' else await anyio.to_thread.run_sync(selected_key, abandon_on_cancel=True)
        )
        if reference is not None:
            token = await anyio.to_thread.run_sync(partial(resolve_key, token=reference), abandon_on_cancel=True)
            return Notion[DepsT](
                auth=token, read_only=settings.read_only, include_instructions=settings.include_instructions
            )
        if settings.auth == 'key':
            raise UserError(f'No Notion key is selected. {SETUP}')
        # Checked per run, so signing in from the menu or `/notion login` applies to the next prompt.
        if not await anyio.to_thread.run_sync(SIGN_IN.signed_in, abandon_on_cancel=True):
            return None
        return Notion[DepsT](
            client=SIGN_IN.client(), read_only=settings.read_only, include_instructions=settings.include_instructions
        )

    async def _command(self, args: list[str]) -> str:
        if args == ['logout']:
            await anyio.to_thread.run_sync(partial(delete_credentials, account=ACCOUNT), abandon_on_cancel=True)
            return f'{await sign_out_now(SIGN_IN)} Cleared the selected key; the key itself stays in /keys.'
        if args in ([], ['status']) and self.settings.auth != 'oauth':
            reference = await anyio.to_thread.run_sync(selected_key, abandon_on_cancel=True)
            if reference is not None:
                return f'Notion connects with {reference.name} from /keys.'
            if self.settings.auth == 'key':
                return f'No Notion key is selected. {SETUP}'
        message = await run_subcommand(SIGN_IN, args)
        if message is None:
            raise ValueError(
                'Usage: /notion [login | logout | status] (settings and the key: /plugins configure notion)'
            )
        return message


_KEY = FieldRow(
    key='key',
    label='Key',
    description=(
        f'The saved API key in /keys that Notion connects with. Enter picks a saved key or saves a new one there '
        f'as {KEY_NAME}; only its name is kept. Any plugin naming the same key shares it. R clears the choice, '
        'so Notion uses the browser sign-in (/notion login) instead.'
    ),
    default='(none)',
)
_AUTH = FieldRow(
    key='auth',
    label='Sign-in',
    description='Automatic uses the chosen key, else the browser sign-in. Key only never uses the browser '
    'sign-in, for headless and remote machines. Browser always uses it, and choosing it signs in now if needed; '
    '/notion login signs in again later.',
    default='auto',
    choices=('auto', 'key', 'oauth'),
    choice_labels={'auto': 'automatic', 'key': 'key only', 'oauth': 'browser'},
    allow_custom=False,
)
_ROWS = (
    _KEY,
    _AUTH,
    FieldRow(
        key='read_only',
        label='Tools',
        description='Read-only keeps only the tools the Notion server marks as read-only, so the agent cannot '
        'change pages.',
        default='false',
        choices=('false', 'true'),
        choice_labels={'true': 'read-only', 'false': 'read and write'},
        allow_custom=False,
    ),
    FieldRow(
        key='include_instructions',
        label='Server instructions',
        description="Whether the Notion server's own instructions reach the agent.",
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'forwarded', 'false': 'left out'},
        allow_custom=False,
    ),
)


class NotionSource(Generic[DepsT]):
    """The settings menu's rows, read from and saved straight to the plugin's settings and `/keys` choice."""

    title = 'Notion'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Every option edit goes through `host.save_settings`."""
        self._host = host

    @property
    def settings(self) -> NotionSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(NotionSettings)

    def rows(self) -> tuple[FieldRow, ...]:
        """Every option, in display order."""
        return _ROWS

    def current(self, row: FieldRow) -> str:
        """The value as the user would type it."""
        if row.key == 'key':
            try:
                reference = selected_key()
            except UserError:
                return '(invalid; choose again)'
            return reference.name if reference else row.default
        value: object = getattr(self.settings, row.key)
        if value is None:
            return 'auto'
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
        self._host.save_settings(self._updated(row, raw))
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        """Restore one option's default, or forget the chosen key (the key itself stays in `/keys`)."""
        if row.key == 'key':
            delete_credentials(account=ACCOUNT)
            return 'Notion no longer uses a saved key. The key itself stays in /keys.'
        data = self.settings.model_dump(mode='json')
        del data[row.key]
        self._host.save_settings(NotionSettings.model_validate(data))
        return f'Reset {row.label}.'

    def _updated(self, row: FieldRow, raw: str) -> NotionSettings:
        data = self.settings.model_dump(mode='json')
        value: JsonValue = raw
        if row.key == 'auth':
            value = None if raw == 'auto' else raw
        elif raw in ('true', 'false'):
            value = raw == 'true'
        data[row.key] = value
        return NotionSettings.model_validate(data)


async def _configure(source: NotionSource[DepsT]) -> str:
    loop = asyncio.get_running_loop()

    def choose() -> list[str]:
        label = f'Notion OAuth access token (saved in /keys as {KEY_NAME})'
        reference = pick_key(loop, name=KEY_NAME, label=label, runners=RUNNERS)
        if reference is None:
            return []
        select_key(reference)
        return [f'Notion uses the saved key {reference.name}. Manage it in /keys.']

    menu = FieldMenu(source)

    def pick_auth() -> list[str]:
        pick = RUNNERS.run_choice(menu.build_choices(_AUTH))
        if pick.cancelled or pick.item is None:
            return []
        saved = menu.apply(_AUTH, str(pick.item.value))
        if pick.item.value != 'oauth' or SIGN_IN.signed_in():
            return [saved]
        message = on_loop(lambda: sign_in_now(SIGN_IN, RUNNERS), loop)
        return [saved] if message is None else [saved, message]

    messages = await run_worker(lambda: run_flow(menu, RUNNERS, submenus={'key': choose, 'auth': pick_auth}))
    return '\n'.join(messages) or 'Notion settings unchanged.'
