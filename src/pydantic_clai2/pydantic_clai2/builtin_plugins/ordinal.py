"""Use Ordinal, with its token kept in /keys.

The built-in `ordinal` plugin: harness `Ordinal`, set up in a settings menu, with no secret in plugin settings.

`/plugins configure ordinal` (also opened by `/plugins enable ordinal`, `/plugins add`, and `c` in `/plugins`, and
shown in the web UI's plugins panel) edits the non-secret options, saved to plugin settings as each one changes,
and picks the token from `/keys`.
Plugin settings are plaintext SQLite, so the token never goes there: only the chosen key's name is kept, in the
credential store, and it is resolved on every run. Replacing the key in `/keys` applies to the next run, deleting
it fails the run closed, and `/keys` refuses to rename it while Ordinal uses it.

Harness `Ordinal` has one endpoint and reaches every workspace the token's user belongs to, so there is no base URL
or workspace to choose. Its non-secret options are `include_instructions` and, here, how to sign in: `browser`
(the default), `key`, or `environment`, each used alone and failing closed when it is missing.

A browser sign-in keeps its OAuth tokens in the OS keyring, as `/mcp` servers' do. Harness `Ordinal(auth='oauth')`
would keep them in memory, so every launch would sign in again. The sign-in runs only from the settings menu or
`/ordinal login`, never while loading or in a prompt; see `pydantic_clai2.plugins.sign_in`.
"""

import os
import sys
from collections.abc import Sequence
from dataclasses import replace
from functools import partial
from typing import Generic, Literal

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.ordinal import Ordinal
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, load_keys, resolve_key, save_key_connection
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials
from pydantic_clai2.mcp import OAuthSignIn
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.plugins.sign_in import (
    SUBCOMMANDS,
    run_subcommand,
    sign_in_now,
    status as sign_in_status,
    warn_if_signed_out,
)
from pydantic_clai2.runtime._session import WorkspacelessCapability
from pydantic_clai2.ui.menus.field_menu import first_error
from pydantic_clai2.ui.rendering import theme

URL = 'https://app.tryordinal.com/mcp'
"""Harness `Ordinal`'s endpoint, which this plugin needs to build its own signed-in client."""
KEY_NAME = 'ORDINAL_ACCESS_TOKEN'
"""The variable harness `Ordinal` reads, and the `/keys` label a token typed into the menu is saved under."""
KEY_ACCOUNT = 'ordinal'
"""The credential account holding the chosen `/keys` entry's name, never the token."""
TOKENS = 'plugin_ordinal'
"""The keyring entry for browser tokens (`mcp-plugin_ordinal`). `/mcp` server names cannot contain `_`."""
SETUP = 'Run /plugins configure ordinal to choose how it signs in.'
USAGE = 'Usage: /ordinal [login | logout | status]'
SIGN_IN = OAuthSignIn(name=TOKENS, service='Ordinal', setup='/ordinal login', url=URL)
"""The browser sign-in; its tokens are the keyring credential `mcp-plugin_ordinal`."""
KEY_LABEL = f'Ordinal access token (saved in /keys as {KEY_NAME})'

SignIn = Literal['browser', 'key', 'environment']


class OrdinalSettings(BaseModel):
    """The plugin's settings. Nothing here is secret; an unknown field such as a pasted token fails to load."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    sign_in: SignIn = 'browser'
    """Which credential runs use; see the module docstring."""
    include_instructions: bool = True
    """Pass Ordinal's own server instructions to the model."""

    @field_validator('sign_in', mode='before')
    @classmethod
    def _automatic_is_browser(cls, value: object) -> object:
        """Earlier builds offered `auto`; it now means the browser sign-in default."""
        return 'browser' if value == 'auto' else value


class KeyChoice(BaseModel):
    """The `/keys` entry Ordinal uses, by name. `key_users` reads this so `/keys` cannot rename it away."""

    token: KeyReference


def saved_key() -> KeyReference | None:
    """The `/keys` entry chosen for Ordinal, or `None` when there is none."""
    raw = load_codex_credentials(account=KEY_ACCOUNT)
    if raw is None:
        return None
    try:
        return KeyChoice.model_validate_json(raw).token
    except ValidationError:
        raise UserError(f'The saved Ordinal key choice is invalid. {SETUP}') from None


def source(method: SignIn) -> KeyReference | Literal['environment', 'browser', 'none']:
    """The credential a run with `method` uses; `'none'` only when `key` has nothing chosen.

    Runs, `/ordinal`, the session-start notice, and the no-terminal check all decide here, so they cannot disagree.
    """
    if method == 'key':
        return saved_key() or 'none'
    return method


def missing(method: SignIn) -> bool:
    """Whether `method` names a key or variable that is not there. A browser sign-in is never missing at load."""
    match source(method):
        case 'environment':
            return not os.environ.get(KEY_NAME)
        case 'none':
            return True
        case _:
            return False


class OrdinalAuth(Generic[DepsT]):
    """Hands each run an `Ordinal` for the current credential, so a `/keys` change or a sign-in applies without a reload."""

    def __init__(self, settings: OrdinalSettings) -> None:
        """Changed settings reload the plugin, so they are fixed for this instance."""
        self.settings = settings

    async def __call__(self, ctx: RunContext[DepsT]) -> Ordinal[DepsT] | None:
        """The capability for this run; `None` without a browser sign-in, and a missing chosen key raises."""
        instructions = self.settings.include_instructions
        match await to_thread.run_sync(source, self.settings.sign_in):
            case KeyReference() as reference:
                token = await to_thread.run_sync(partial(resolve_key, token=reference))
                return Ordinal[DepsT](auth=token, include_instructions=instructions)
            case 'environment':
                # Harness reads the variable when it connects and fails closed when it is unset.
                return Ordinal[DepsT](include_instructions=instructions)
            case 'browser':
                # Each run builds its own client from the stored tokens, so a sign-out reaches the next run.
                if not await to_thread.run_sync(SIGN_IN.signed_in, abandon_on_cancel=True):
                    return None
                return Ordinal[DepsT](client=SIGN_IN.client(), include_instructions=instructions)
            case 'none':  # pragma: no branch -- the cases cover every `source` result.
                raise UserError(f'Ordinal has no /keys entry. {SETUP}')


class OrdinalPlugin(Plugin[OrdinalSettings, DepsT]):
    """`Ordinal` and its settings menu; without a terminal, refuses to load when its chosen key is missing."""

    def __init__(self, host: PluginHost[DepsT], settings: OrdinalSettings) -> None:
        super().__init__(host, settings)
        if not sys.stdin.isatty() and missing(settings.sign_in):
            raise UserError(f'Ordinal has no credential for sign-in `{settings.sign_in}` and no terminal. {SETUP}')
        self.auth = OrdinalAuth[DepsT](settings)

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (WorkspacelessCapability[DepsT](self.auth),)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='ordinal',
                description='Show how Ordinal signs in, or sign in or out in the browser (/ordinal login|logout).',
                handler=self._command,
                complete=lambda args: SUBCOMMANDS if len(args) <= 1 else (),
            ),
        )

    def settings_menu(self) -> SettingsMenu:
        source = OrdinalSource(self.host)
        return SettingsMenu(source=source, actions={METHOD.key: source.pick_method, KEY.key: source.pick_key})

    async def on_session_start(self, event: SessionStart) -> None:
        console = self.host.console
        try:
            method = await to_thread.run_sync(source, self.settings.sign_in, abandon_on_cancel=True)
        except UserError as exc:
            # Loading anyway keeps the menu available to choose again; runs fail closed meanwhile.
            console.print(str(exc), style=theme.color(theme.WARNING), markup=False)
            return
        if method == 'browser':
            await warn_if_signed_out(SIGN_IN, console)

    async def _command(self, args: list[str]) -> str:
        if args in ([], ['status']):
            return await to_thread.run_sync(status, self.settings, abandon_on_cancel=True)
        message = await run_subcommand(SIGN_IN, args)
        return USAGE if message is None else message


def status(settings: OrdinalSettings) -> str:
    """One line on which credential the next run uses, and whether that run will fail for want of it."""
    match source(settings.sign_in):
        case KeyReference(name=name) if name in load_keys():
            return f'Ordinal uses {name} from /keys.'
        case KeyReference(name=name):
            return f'Ordinal uses {name}, which is missing from /keys, so runs fail. {SETUP}'
        case 'none':
            return f'Ordinal has no /keys entry, so runs fail. {SETUP}'
        case 'environment' if os.environ.get(KEY_NAME):
            return f'Ordinal uses `{KEY_NAME}` from the environment.'
        case 'environment':
            return f'Ordinal uses `{KEY_NAME}`, which is not set, so runs fail. {SETUP}'
        case 'browser':  # pragma: no branch -- the cases cover every `source` result.
            return sign_in_status(SIGN_IN)


METHOD = FieldRow(
    key='sign_in',
    label='Sign-in',
    description='Which credential runs use: the browser sign-in by default, a /keys entry, or the environment.',
    default='browser',
    choices=('browser', 'key', 'environment'),
    choice_labels={
        'browser': 'Browser sign-in',
        'key': 'Saved key from /keys',
        'environment': f'{KEY_NAME} environment variable',
    },
    allow_custom=False,
)
KEY = FieldRow(
    key='key',
    label='/keys entry',
    description=(
        f'The Ordinal access token, chosen from /keys or typed masked and saved there as {KEY_NAME}. Only its name '
        'is kept; other plugins can use the same key. R stops using it.'
    ),
    default='(none)',
)
INVALID = '(invalid; choose again)'
INSTRUCTIONS = FieldRow(
    key='include_instructions',
    label='Server instructions',
    description="Pass Ordinal's own server instructions to the model.",
    default='true',
    choices=('true', 'false'),
    choice_labels={'true': 'Included', 'false': 'Left out'},
    allow_custom=False,
)


class OrdinalSource(Generic[DepsT]):
    """The settings menu's rows. Options save to plugin settings; the key's name saves to the credential store."""

    title = 'Ordinal'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Every option edit goes through `host.save_settings`."""
        self._host = host

    @property
    def settings(self) -> OrdinalSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(OrdinalSettings)

    def rows(self) -> list[FieldRow]:
        """Every option, with the key marked when it is gone from `/keys`."""
        name = self._key_name()
        gone = name not in (KEY.default, INVALID) and name not in load_keys()
        return [METHOD, replace(KEY, note='missing from /keys') if gone else KEY, INSTRUCTIONS]

    def current(self, row: FieldRow) -> str:
        """The value as the user would pick it."""
        if row.key == KEY.key:
            return self._key_name()
        value = getattr(self.settings, row.key)
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
        """Restore an option's default, or stop using the `/keys` entry."""
        if row.key == KEY.key:
            delete_credentials(account=KEY_ACCOUNT)
            return 'Ordinal uses no /keys entry.'
        data = self.settings.model_dump()
        del data[row.key]
        self.save(OrdinalSettings.model_validate(data))
        return f'Reset {row.label}.'

    def save(self, settings: OrdinalSettings) -> None:
        """Persist to the plugin's declaration."""
        self._host.save_settings(settings)

    async def pick_method(self, dialogs: Dialogs) -> list[str]:
        """The Sign-in row: choosing the browser signs in now, behind a waiting screen, unless a sign-in is stored."""
        picked = await dialogs.choose(
            title=f'Choose {METHOD.label}',
            options=[Option(value=choice, label=METHOD.display(choice)) for choice in METHOD.choices],
            current=await to_thread.run_sync(self.current, METHOD),
        )
        if picked is None:
            return []
        saved = await to_thread.run_sync(self.apply, METHOD, picked)
        if picked != 'browser' or await to_thread.run_sync(SIGN_IN.signed_in):
            return [saved]
        return [saved, await sign_in_now(SIGN_IN, dialogs)]

    async def pick_key(self, dialogs: Dialogs) -> list[str]:
        """The `/keys` entry row: pick a saved key or save a new one in `/keys`, then switch Sign-in to the key."""
        reference = await choose_key(name=KEY_NAME, label=KEY_LABEL, dialogs=dialogs)
        if reference is None:
            return []
        await to_thread.run_sync(self._choose, reference)
        messages = [f'Ordinal uses {reference.name} from /keys. Manage it there.']
        # A chosen key is used only with the key sign-in, so choosing one switches to it.
        if await to_thread.run_sync(self._switch_to_key):
            messages.append(f'Saved {METHOD.label}: {METHOD.display("key")}.')
        return messages

    @staticmethod
    def _choose(reference: KeyReference) -> None:
        """Keep the key's name, never its value, in the credential store."""
        save_key_connection(account=KEY_ACCOUNT, token=reference, value=KeyChoice(token=reference).model_dump_json())

    def _switch_to_key(self) -> bool:
        """Use the chosen key for runs; whether that changed the setting."""
        if self.settings.sign_in == 'key':
            return False
        self.save(self.settings.model_copy(update={'sign_in': 'key'}))
        return True

    def _key_name(self) -> str:
        try:
            reference = saved_key()
        except UserError:
            return INVALID
        return KEY.default if reference is None else reference.name

    def _updated(self, row: FieldRow, raw: str) -> OrdinalSettings:
        value: object = raw == 'true' if row.key == INSTRUCTIONS.key and raw in ('true', 'false') else raw
        return OrdinalSettings.model_validate({**self.settings.model_dump(), row.key: value})
