"""The built-in `ordinal` plugin: harness `Ordinal`, set up in a settings menu, with no secret in plugin settings.

`/plugins configure ordinal` (also opened by `/plugins enable ordinal`, `/plugins add`, and `C` in `/plugins`)
edits the non-secret options, saved to plugin settings as each one changes, and picks the token from `/keys`.
Plugin settings are plaintext SQLite, so the token never goes there: only the chosen key's name is kept, in the
credential store, and it is resolved on every run. Replacing the key in `/keys` applies to the next run, deleting
it fails the run closed, and `/keys` refuses to rename it while Ordinal uses it.

Harness `Ordinal` has one endpoint and reaches every workspace the token's user belongs to, so there is no base URL
or workspace to choose. Its non-secret options are `include_instructions` and, here, how to sign in:

- `auto`: a chosen `/keys` entry, else `ORDINAL_ACCESS_TOKEN`, else a browser sign-in.
- `key`, `environment`, or `browser`: only that one, failing closed when it is missing.

A browser sign-in keeps its OAuth tokens in the OS keyring, as `/mcp` servers' do. Harness `Ordinal(auth='oauth')`
would keep them in memory, so every launch would sign in again.
"""

import asyncio
import os
import sys
from dataclasses import replace
from functools import partial
from typing import Generic, Literal

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, ValidationError
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.ordinal import Ordinal

from ._rendering import markdown_style
from .api_keys import (
    KeyExistsError,
    KeyReference,
    load_keys,
    prompt_api_key,
    resolve_key,
    save_key,
    save_key_connection,
)
from .commands import Command
from .credential_store import delete_credentials, load_codex_credentials
from .field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from .mcp import OAUTH_TIMEOUT, TokenStore, http_client, sign_in
from .menu_worker import menu_key, run_worker
from .plugins import DepsT, PluginHost

URL = 'https://app.tryordinal.com/mcp'
"""Harness `Ordinal`'s endpoint, which this plugin needs to build its own signed-in client."""
KEY_NAME = 'ORDINAL_ACCESS_TOKEN'
"""The variable harness `Ordinal` reads, and the `/keys` label a token typed into the menu is saved under."""
KEY_ACCOUNT = 'ordinal'
"""The credential account holding the chosen `/keys` entry's name, never the token."""
TOKENS = 'plugin_ordinal'
"""The keyring entry for browser tokens (`mcp-plugin_ordinal`). `/mcp` server names cannot contain `_`."""
SETUP = 'Run /plugins configure ordinal to choose how it signs in.'
USAGE = 'Usage: /ordinal [logout]'
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""

SignIn = Literal['auto', 'key', 'environment', 'browser']


class OrdinalSettings(BaseModel):
    """The plugin's settings. Nothing here is secret; an unknown field such as a pasted token fails to load."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    sign_in: SignIn = 'auto'
    """Which credential runs use; see the module docstring."""
    include_instructions: bool = True
    """Pass Ordinal's own server instructions to the model."""


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

    Runs, `/ordinal`, and the no-terminal check all decide here, so they cannot disagree.
    """
    if method in ('auto', 'key'):
        reference = saved_key()
        if reference is not None:
            return reference
        if method == 'key':
            return 'none'
    if method == 'environment' or (method == 'auto' and os.environ.get(KEY_NAME)):
        return 'environment'
    return 'browser'


def ready(method: SignIn, tokens: TokenStore) -> bool:
    """Whether a run can authenticate with `method` without asking anyone."""
    match source(method):
        case KeyReference():
            return True
        case 'environment':
            return bool(os.environ.get(KEY_NAME))
        case 'browser':
            return bool(tokens.signed_in())
        case 'none':  # pragma: no branch -- the cases cover every `source` result.
            return False


class OrdinalAuth(Generic[DepsT]):
    """Hands each run an `Ordinal` for the current credential, so a `/keys` change applies without a reload.

    Once connected, FastMCP's `OAuth` keeps the access token in memory, so clearing the keyring alone would leave
    this session signed in. `logout` therefore replaces the browser `Ordinal`, client and sign-in handler included.
    """

    def __init__(self, settings: OrdinalSettings, tokens: TokenStore) -> None:
        """Changed settings reload the plugin, so they are fixed for this instance."""
        self.settings = settings
        self.tokens = tokens
        self.browser = self._browser()

    async def __call__(self, ctx: RunContext[DepsT]) -> Ordinal[DepsT]:
        """The capability for this run; a missing chosen credential raises instead of connecting."""
        instructions = self.settings.include_instructions
        match await to_thread.run_sync(source, self.settings.sign_in):
            case KeyReference() as reference:
                token = await to_thread.run_sync(partial(resolve_key, token=reference))
                return Ordinal[DepsT](auth=token, include_instructions=instructions)
            case 'environment':
                # Harness reads the variable when it connects and fails closed when it is unset.
                return Ordinal[DepsT](include_instructions=instructions)
            case 'browser':
                return self.browser
            case 'none':  # pragma: no branch -- the cases cover every `source` result.
                raise UserError(f'Ordinal has no /keys entry. {SETUP}')

    def logout(self) -> None:
        """Forget the saved browser tokens and the in-memory ones, so the next browser run signs in again."""
        self.tokens.forget()
        self.browser = self._browser()

    def _browser(self) -> Ordinal[DepsT]:
        transport = StreamableHttpTransport(url=URL, auth=sign_in(self.tokens.name), httpx_client_factory=http_client)
        # A bare transport gets `MCPToolset`'s 5 second handshake timeout, which would end a browser sign-in early.
        client = Client(transport, init_timeout=OAUTH_TIMEOUT)
        return Ordinal[DepsT](client=client, include_instructions=self.settings.include_instructions)


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Ordinal` and its settings menu, or refuse to load when no one could sign in."""
    settings = host.settings(OrdinalSettings)
    tokens = TokenStore(TOKENS)
    if not sys.stdin.isatty() and not ready(settings.sign_in, tokens):
        raise UserError(f'Ordinal has no credential for sign-in `{settings.sign_in}` and no terminal. {SETUP}')
    auth = OrdinalAuth[DepsT](settings, tokens)
    host.add(auth)
    host.configure(partial(configure, OrdinalSource(host)))

    async def command(args: list[str]) -> str:
        match args:
            case []:
                return await to_thread.run_sync(status, settings, tokens)
            case ['logout']:
                await to_thread.run_sync(auth.logout)
                return 'Signed out of the Ordinal browser session; a /keys entry or the environment is unaffected.'
            case _:
                return USAGE

    host.commands.register(
        Command(
            name='ordinal',
            description='Show how Ordinal signs in, or end its browser session (/ordinal logout).',
            handler=command,
            complete=lambda args: ['logout'] if len(args) <= 1 else [],
        )
    )


def status(settings: OrdinalSettings, tokens: TokenStore) -> str:
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
            return {
                True: 'Ordinal: signed in through the browser. /ordinal logout signs out.',
                False: 'Ordinal: not signed in; the browser opens on first use.',
                None: 'Ordinal: sign-in unknown; the keyring could not be read.',
            }[tokens.signed_in()]


SIGN_IN = FieldRow(
    key='sign_in',
    label='Sign-in',
    description='Which credential runs use. Automatic tries a /keys entry, then the environment, then the browser.',
    default='auto',
    choices=('auto', 'key', 'environment', 'browser'),
    choice_labels={
        'auto': 'Automatic',
        'key': 'Saved key from /keys',
        'environment': f'{KEY_NAME} environment variable',
        'browser': 'Browser sign-in',
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
        missing = name not in (KEY.default, INVALID) and name not in load_keys()
        return [SIGN_IN, replace(KEY, note='missing from /keys') if missing else KEY, INSTRUCTIONS]

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

    def _key_name(self) -> str:
        try:
            reference = saved_key()
        except UserError:
            return INVALID
        return KEY.default if reference is None else reference.name

    def _updated(self, row: FieldRow, raw: str) -> OrdinalSettings:
        value: object = raw == 'true' if row.key == INSTRUCTIONS.key and raw in ('true', 'false') else raw
        return OrdinalSettings.model_validate({**self.settings.model_dump(), row.key: value})


async def configure(source: OrdinalSource[DepsT]) -> str:
    """The settings menu: list, edit, back, until Esc; the key row opens the `/keys` picker."""
    loop = asyncio.get_running_loop()

    def pick_key() -> list[str]:
        # The key picker is async, so the menu's thread hands it back to the event loop.
        reference = asyncio.run_coroutine_threadsafe(choose_key(), loop).result()
        if reference is None:
            return []
        choice = KeyChoice(token=reference).model_dump_json()
        save_key_connection(account=KEY_ACCOUNT, token=reference, value=choice)
        messages = [f'Ordinal uses {reference.name} from /keys. Manage it there.']
        if source.settings.sign_in in ('environment', 'browser'):
            source.save(source.settings.model_copy(update={'sign_in': 'key'}))
            messages.append(f'Saved {SIGN_IN.label}: {SIGN_IN.display("key")}.')
        return messages

    menu = FieldMenu(source)
    messages = await run_worker(lambda: run_flow(menu, RUNNERS, submenus={KEY.key: pick_key}))
    return '\n'.join(messages) or 'Ordinal settings unchanged.'


class MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked Termflow input, matching the settings menu."""

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        """Read the token; Esc raises `EOFError`, which `prompt_api_key` reads as cancellation."""
        builder = (
            TextInputBuilder(label)
            .style(markdown_style())
            .prompt('Token: ')
            .placeholder('Paste an Ordinal access token; it is saved in /keys')
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: RUNNERS.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            raise EOFError
        return result.value


async def choose_key() -> KeyReference | None:
    """Pick a `/keys` entry, or save a masked new token as `ORDINAL_ACCESS_TOKEN`; `None` means cancelled."""
    choice = await prompt_api_key(prompt=MaskedPrompt(), label=f'Ordinal access token (saved in /keys as {KEY_NAME})')
    if choice is None or isinstance(choice, KeyReference):
        return choice
    value = choice.strip()
    if not value:
        return None
    # Checked and saved under one lock, so a key another process just saved is never replaced unasked.
    try:
        await to_thread.run_sync(partial(save_key, name=KEY_NAME, value=value, replace=False))
    except KeyExistsError:
        if not await run_worker(confirm_replace):
            return None
        await to_thread.run_sync(partial(save_key, name=KEY_NAME, value=value))
    return KeyReference(name=KEY_NAME)


def confirm_replace() -> bool:
    """Ask before a typed token replaces a `/keys` entry other plugins may share."""
    menu = (
        MenuBuilder(f'{KEY_NAME} is already in /keys')
        .style(markdown_style())
        .items(
            [
                MenuItem('Keep the saved token', value=False),
                MenuItem(f'Replace {KEY_NAME} for every plugin and connection that uses it', value=True),
            ]
        )
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    pick = RUNNERS.run_choice(menu)
    return not pick.cancelled and pick.item is not None and pick.item.value is True
