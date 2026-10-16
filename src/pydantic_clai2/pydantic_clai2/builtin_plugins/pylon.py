"""Look up Pylon support issues, accounts, and contacts.

The built-in `pylon` plugin: Pylon's support issues, accounts, and contacts through harness `Pylon`.

The settings menu is declared by `PylonPlugin.settings_menu`, so it opens in the terminal (on enable, `c` in
`/plugins`, `/plugins configure pylon`, and `/pylon`) and in the web UI's plugins panel. Each edit is saved to the
plugin's settings at once and applies from the next run, because the capability is rebuilt per run from the
current settings.

The token is never kept in plugin settings, which are plaintext SQLite. By default the plugin connects with a
named key from `/keys`, chosen through the shared key picker; only the key's name is saved. The key is resolved
on every run, so replacing it in `/keys` takes effect on the next run, and a deleted key fails the run instead
of connecting without it.

With browser sign-in, CLAI signs in the way `/mcp` OAuth servers do, keeping the tokens in the keyring under
`mcp-plugin_pylon`. The sign-in runs only from the settings menu or `/pylon login`, never while loading or in a
prompt; see `pydantic_clai2.plugins.sign_in`. Until then, runs get no Pylon tools.
"""

import json
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.pylon import Pylon
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, forget_connection, resolve_key, save_key_connection
from pydantic_clai2.config.credential_store import load_codex_credentials
from pydantic_clai2.mcp import OAuthSignIn
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.plugins.sign_in import SUBCOMMANDS, run_subcommand, sign_in_now, status, warn_if_signed_out
from pydantic_clai2.runtime._session import WorkspacelessCapability
from pydantic_clai2.ui.menus import dialogs as menu_dialogs
from pydantic_clai2.ui.menus.field_menu import first_error

PYLON_MCP_URL = 'https://mcp.usepylon.com'
"""Harness `Pylon`'s endpoint, repeated here because the browser sign-in's client owns its URL."""

KEY_NAME = 'PYLON_ACCESS_TOKEN'
"""The `/keys` label for a new token: harness `Pylon`'s documented variable name, used as a label only."""

ACCOUNT = 'pylon'
"""The credential account holding the key reference, beside the `vllm` and `openrouter` connections."""

TOKEN_ACCOUNT = 'plugin_pylon'
"""Browser tokens are stored as `mcp-plugin_pylon`; `/mcp` server names cannot contain `_`, so none shares them."""

SIGN_IN = OAuthSignIn(name=TOKEN_ACCOUNT, service='Pylon', setup='/pylon login', url=PYLON_MCP_URL)
"""The browser sign-in; its tokens are the keyring credential `mcp-plugin_pylon`."""

NOT_CHOSEN = '(not chosen)'
_WORDS = ('key', *SUBCOMMANDS)
_HELP = (
    'Usage: /pylon (settings menu), /pylon key (choose the /keys entry), /pylon login or logout (browser sign-in), '
    'or /pylon status'
)
_KEY_LABEL = f'{KEY_NAME} (a Pylon OAuth access token; Pylon API keys are not accepted)'


class PylonSettings(BaseModel):
    """The JSON a `pylon` declaration may carry. Nothing here is secret."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    auth: Literal['key', 'browser'] = Field(
        default='key', description='Connect with a named key from /keys, or sign in through the browser.'
    )
    read_only: bool = Field(default=False, description="Keep only the tools Pylon's server labels read-only.")
    include_instructions: bool = Field(default=True, description="Pass Pylon's own server instructions to the agent.")


class _Saved(BaseModel):
    token: KeyReference


def saved_key() -> KeyReference | None:
    """The key Pylon connects with, or `None` before one is chosen."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return None
    try:
        return _Saved.model_validate_json(raw).token
    except ValidationError:
        raise UserError('The saved Pylon key reference is invalid. Choose a key again with /pylon key.') from None


class PylonConfig:
    """The settings menu's rows, backed by plugin settings plus the `/keys` reference."""

    title = 'Pylon settings'

    def __init__(self, settings: PylonSettings, save: Callable[[PylonSettings], None]) -> None:
        """`save` persists new settings."""
        self.settings = settings
        self._save = save

    def rows(self) -> Sequence[FieldRow]:
        """Sign-in first, the key only when it is used, then the capability's switches."""
        rows = [_AUTH]
        if self.settings.auth == 'key':
            rows.append(
                FieldRow(
                    key='key',
                    label='Key (/keys)',
                    description=(
                        'The /keys entry Pylon connects with. Enter picks a saved key or saves a new one as '
                        f'{KEY_NAME}; R forgets the choice. Only the name is saved; the secret stays in /keys.'
                    ),
                    default=NOT_CHOSEN,
                    note='name only',
                )
            )
        rows += [
            FieldRow(
                key='read_only',
                label='Read-only tools',
                description="Keep only the tools Pylon's server labels read-only.",
                default='false',
                choices=('true', 'false'),
                allow_custom=False,
            ),
            FieldRow(
                key='include_instructions',
                label='Server instructions',
                description="Pass Pylon's own server instructions to the agent.",
                default='true',
                choices=('true', 'false'),
                allow_custom=False,
            ),
        ]
        return rows

    def current(self, row: FieldRow) -> str:
        """Settings as typed; the key row shows the referenced name, never a value."""
        if row.key == 'key':
            try:
                reference = saved_key()
            except UserError:
                return '(invalid; choose again)'
            return NOT_CHOSEN if reference is None else reference.name
        return _text(getattr(self.settings, row.key))

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Only choice rows are editable here, so any typed text is checked against the model."""
        try:
            self._validated(row, text)
        except ValidationError as exc:
            return first_error(exc)
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save the new value to plugin settings now; it applies from the next run."""
        return self._store(self._validated(row, raw), row)

    def reset(self, row: FieldRow) -> str:
        """Restore a setting's default, or forget the key reference."""
        if row.key == 'key':
            forget_connection(account=ACCOUNT)
            return 'Pylon no longer uses a /keys entry; runs get no Pylon tools until you choose one.'
        return self._store(self.settings.model_copy(update={row.key: PylonSettings.model_fields[row.key].default}), row)

    def _validated(self, row: FieldRow, raw: str) -> PylonSettings:
        value: JsonValue = raw == 'true' if raw in ('true', 'false') and row.key != 'auth' else raw
        return PylonSettings.model_validate({**self.settings.model_dump(), row.key: value})

    def _store(self, settings: PylonSettings, row: FieldRow) -> str:
        self._save(settings)
        self.settings = settings
        shown = row.display(_text(getattr(settings, row.key)))
        return f'Pylon {row.label.lower()}: {shown}. Applies from the next run.'


_AUTH = FieldRow(
    key='auth',
    label='Sign-in',
    description='How Pylon authenticates. Pylon only accepts OAuth access tokens, not REST API keys. Choosing '
    'browser sign-in signs in now if needed; /pylon login signs in again later.',
    default='key',
    choices=('key', 'browser'),
    choice_labels={'key': 'Named key from /keys', 'browser': 'Browser sign-in (OAuth)'},
    allow_custom=False,
)


def _text(value: object) -> str:
    return json.dumps(value) if isinstance(value, bool) else str(value)


class PylonPlugin(Plugin[PylonSettings, DepsT]):
    """A per-run `Pylon` built from the current settings, its settings menu, and `/pylon`."""

    def __init__(self, host: PluginHost[DepsT], settings: PylonSettings) -> None:
        super().__init__(host, settings)
        self.config = PylonConfig(settings, host.save_settings)

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (WorkspacelessCapability[DepsT](self._for_run),)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='pylon',
                description='Pylon settings: sign-in, the /keys entry, and tool options '
                '(/pylon key, /pylon login|logout, /pylon status).',
                handler=self._command,
                complete=lambda args: (
                    [word for word in _WORDS if word.startswith(args[0] if args else '')] if len(args) <= 1 else []
                ),
                during_turn=True,
            ),
        )

    def settings_menu(self) -> SettingsMenu:
        return SettingsMenu(source=self.config, actions={'auth': self._pick_auth, 'key': pick_key})

    async def on_session_start(self, event: SessionStart) -> None:
        if self.config.settings.auth == 'browser':
            await warn_if_signed_out(SIGN_IN, self.host.console)

    async def _for_run(self, _: RunContext[DepsT]) -> Pylon[DepsT] | None:
        # The token is resolved here, not passed as a callable `auth`: Pylon would answer with a
        # `DynamicToolset`, and pydantic-ai 2.49 drops a dynamic toolset returned from a `CapabilityFunc`.
        settings = self.config.settings
        if settings.auth == 'browser':
            # Checked per run, so signing in from the menu or `/pylon login` applies to the next prompt.
            if not await anyio.to_thread.run_sync(SIGN_IN.signed_in, abandon_on_cancel=True):
                return None
            client, token = SIGN_IN.client(), None
        else:
            reference = saved_key()
            if reference is None:
                return None
            client, token = None, resolve_key(token=reference)
        return Pylon[DepsT](
            auth=token,
            client=client,
            read_only=settings.read_only,
            include_instructions=settings.include_instructions,
        )

    async def _pick_auth(self, dialogs: Dialogs) -> list[str]:
        """Pick how Pylon signs in, and sign in through the browser now when that is picked and needed."""
        picked = await dialogs.choose(
            title=f'Choose {_AUTH.label}',
            options=[Option(value=choice, label=_AUTH.display(choice)) for choice in _AUTH.choices],
            current=self.config.settings.auth,
        )
        if picked is None:
            return []
        messages = [await anyio.to_thread.run_sync(self.config.apply, _AUTH, picked)]
        if picked == 'browser' and not await anyio.to_thread.run_sync(SIGN_IN.signed_in, abandon_on_cancel=True):
            messages.append(await sign_in_now(SIGN_IN, dialogs))
        return messages

    async def _command(self, args: list[str]) -> str:
        if not args:
            return await self.configure()
        if args == ['key']:
            # The module's `RUNNERS` is read here, not imported, so tests can script the terminal.
            messages = await pick_key(menu_dialogs.TerminalDialogs(menu_dialogs.RUNNERS))
            return '\n'.join(messages) or 'Pylon key unchanged.'
        if args == ['status']:
            return await anyio.to_thread.run_sync(_status, self.config, abandon_on_cancel=True)
        message = await run_subcommand(SIGN_IN, args)
        if message is None:
            raise ValueError(_HELP)
        return message


def _status(config: PylonConfig) -> str:
    settings = config.settings
    options = f'read-only tools {_text(settings.read_only)}, server instructions {_text(settings.include_instructions)}'
    if settings.auth == 'browser':
        return f'{status(SIGN_IN)} Pylon uses the browser sign-in; {options}.'
    reference = saved_key()
    if reference is None:
        return f'Pylon has no key yet, so runs get no Pylon tools. Choose one with /pylon key; {options}.'
    return f'Pylon connects with {reference.name} from /keys; {options}.'


async def pick_key(dialogs: Dialogs) -> list[str]:
    """Pick a saved key or enter a new masked one through `dialogs`; only the key's name is saved for Pylon."""
    reference = await choose_key(name=KEY_NAME, label=_KEY_LABEL, dialogs=dialogs)
    if reference is None:
        return []
    saved = json.dumps({'token': reference.model_dump()})
    try:
        await anyio.to_thread.run_sync(
            partial(save_key_connection, account=ACCOUNT, token=reference, value=saved), abandon_on_cancel=True
        )
    except UserError as exc:  # The picked key was deleted meanwhile.
        return [str(exc)]
    return [f'Pylon connects with {reference.name} from /keys.']
