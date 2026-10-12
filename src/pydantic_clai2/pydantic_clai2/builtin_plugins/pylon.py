"""Look up Pylon support issues, accounts, and contacts.

The built-in `pylon` plugin: Pylon's support issues, accounts, and contacts through harness `Pylon`.

The settings menu is `PylonPlugin.configure`, so turning the plugin on opens it, as do `c` in
`/plugins`, `/plugins configure pylon`, and `/pylon`. Each edit is saved to the plugin's settings at once and
applies from the next run, because the capability is rebuilt per run from the current settings.

The token is never kept in plugin settings, which are plaintext SQLite. By default the plugin connects with a
named key from `/keys`, chosen through the shared key picker; only the key's name is saved. The key is resolved
on every run, so replacing it in `/keys` takes effect on the next run, and a deleted key fails the run instead
of connecting without it.

With browser sign-in, CLAI signs in the way `/mcp` OAuth servers do, keeping the tokens in the keyring.
Harness `Pylon(auth='oauth')` would keep them in memory and give the browser the 5-second connect timeout,
so CLAI builds that client itself.
"""

import json
from collections.abc import Callable, Sequence
from functools import partial
from typing import Literal

import anyio
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.pylon import Pylon
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import (
    KeyReference,
    forget_connection,
    load_keys,
    prompt_api_key,
    resolve_key,
    save_key,
    save_key_connection,
)
from pydantic_clai2.config.credential_store import load_codex_credentials
from pydantic_clai2.mcp import OAUTH_TIMEOUT, http_client, sign_in
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from pydantic_clai2.ui.menus.menu_worker import run_worker

PYLON_MCP_URL = 'https://mcp.usepylon.com'
"""Harness `Pylon`'s endpoint, repeated here because a custom client owns its URL."""

KEY_NAME = 'PYLON_ACCESS_TOKEN'
"""The `/keys` label for a new token: harness `Pylon`'s documented variable name, used as a label only."""

ACCOUNT = 'pylon'
"""The credential account holding the key reference, beside the `vllm` and `openrouter` connections."""

TOKEN_ACCOUNT = 'plugin_pylon'
"""Browser tokens are stored as `mcp-plugin_pylon`; `/mcp` server names cannot contain `_`, so none shares them."""

NOT_CHOSEN = '(not chosen)'
_HELP = 'Usage: /pylon (settings menu), /pylon key (choose the /keys entry), or /pylon status'


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


class _ChooseKey(Exception):
    """Release the menu worker so the async key picker can own the terminal."""


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
        """`save` persists new settings; every applied edit is also recorded in `notes`."""
        self.settings = settings
        self._save = save
        self.notes: list[str] = []

    def rows(self) -> Sequence[FieldRow]:
        """Sign-in first, the key only when it is used, then the capability's switches."""
        rows = [
            FieldRow(
                key='auth',
                label='Sign-in',
                description='How Pylon authenticates. Pylon only accepts OAuth access tokens, not REST API keys.',
                default='key',
                choices=('key', 'browser'),
                choice_labels={'key': 'Named key from /keys', 'browser': 'Browser sign-in (OAuth)'},
                allow_custom=False,
            )
        ]
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
            return self._note('Pylon no longer uses a /keys entry; runs get no Pylon tools until you choose one.')
        return self._store(self.settings.model_copy(update={row.key: PylonSettings.model_fields[row.key].default}), row)

    def _validated(self, row: FieldRow, raw: str) -> PylonSettings:
        value: JsonValue = raw == 'true' if raw in ('true', 'false') and row.key != 'auth' else raw
        return PylonSettings.model_validate({**self.settings.model_dump(), row.key: value})

    def _store(self, settings: PylonSettings, row: FieldRow) -> str:
        self._save(settings)
        self.settings = settings
        shown = row.display(_text(getattr(settings, row.key)))
        return self._note(f'Pylon {row.label.lower()}: {shown}. Applies from the next run.')

    def _note(self, message: str) -> str:
        self.notes.append(message)
        return message


def _text(value: object) -> str:
    return json.dumps(value) if isinstance(value, bool) else str(value)


def _release_for_key() -> list[str]:
    raise _ChooseKey


async def configure(config: PylonConfig, runners: Runners | None = None) -> str:
    """Open the settings menu until Esc, stepping out of the menu worker whenever the key picker is needed."""
    notes = len(config.notes)
    while True:
        try:
            await run_worker(
                lambda: run_flow(FieldMenu(config), runners or TERMINAL, submenus={'key': _release_for_key})
            )
        except _ChooseKey:
            try:
                config.notes.append(await choose_key())
            except (ValueError, UserError) as exc:
                config.notes.append(str(exc))
            continue
        return '\n'.join(config.notes[notes:]) or 'Pylon settings unchanged.'


class PylonPlugin(Plugin[PylonSettings, DepsT]):
    """A per-run `Pylon` built from the current settings, its settings menu, and `/pylon`."""

    def __init__(self, host: PluginHost[DepsT], settings: PylonSettings) -> None:
        super().__init__(host, settings)
        self.config = PylonConfig(settings, host.save_settings)

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        return (self._for_run,)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='pylon',
                description='Pylon settings: sign-in, the /keys entry, and tool options (/pylon key, /pylon status).',
                handler=self._command,
                complete=lambda args: (
                    [word for word in ('key', 'status') if word.startswith(args[0] if args else '')]
                    if len(args) <= 1
                    else []
                ),
                during_turn=True,
            ),
        )

    async def configure(self) -> str:
        return await configure(self.config)

    def _for_run(self, _: RunContext[DepsT]) -> Pylon[DepsT] | None:
        # The token is resolved here, not passed as a callable `auth`: Pylon would answer with a
        # `DynamicToolset`, and pydantic-ai 2.49 drops a dynamic toolset returned from a `CapabilityFunc`.
        settings = self.config.settings
        if settings.auth == 'browser':
            client, token = _browser_client(), None
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

    async def _command(self, args: list[str]) -> str:
        if not args:
            return await self.configure()
        if args == ['key']:
            return await choose_key()
        if args == ['status']:
            return await anyio.to_thread.run_sync(_status, self.config, abandon_on_cancel=True)
        raise ValueError(_HELP)


def _status(config: PylonConfig) -> str:
    settings = config.settings
    options = f'read-only tools {_text(settings.read_only)}, server instructions {_text(settings.include_instructions)}'
    if settings.auth == 'browser':
        return f'Pylon signs in through the browser; {options}.'
    reference = saved_key()
    if reference is None:
        return f'Pylon has no key yet, so runs get no Pylon tools. Choose one with /pylon key; {options}.'
    return f'Pylon connects with {reference.name} from /keys; {options}.'


async def choose_key() -> str:
    """Pick a saved key or enter a new masked one; only the key's name is saved for Pylon."""
    prompt: PromptSession[str] = PromptSession()
    label = f'{KEY_NAME} (a Pylon OAuth access token; Pylon API keys are not accepted): '
    token = await prompt_api_key(prompt=prompt, label=label)
    if token is None:
        return 'Pylon key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            raise ValueError('A Pylon access token is required.')
        if KEY_NAME in await anyio.to_thread.run_sync(load_keys, abandon_on_cancel=True):
            try:
                answer = await prompt.prompt_async(
                    f'Replace {KEY_NAME} in /keys for every connection using it? [y/N]: '
                )
            except (EOFError, KeyboardInterrupt):
                return 'Pylon key unchanged.'
            if answer.strip().lower() != 'y':
                return 'Pylon key unchanged.'
        await anyio.to_thread.run_sync(partial(save_key, name=KEY_NAME, value=token), abandon_on_cancel=True)
        token = KeyReference(name=KEY_NAME)
    saved = json.dumps({'token': token.model_dump()})
    await anyio.to_thread.run_sync(
        partial(save_key_connection, account=ACCOUNT, token=token, value=saved), abandon_on_cancel=True
    )
    return f'Pylon connects with {token.name} from /keys.'


def _browser_client() -> Client[StreamableHttpTransport]:
    """A Pylon connection that signs in on first use and allows the browser as long as `/mcp` does."""
    transport = StreamableHttpTransport(
        url=PYLON_MCP_URL, auth=sign_in(TOKEN_ACCOUNT), httpx_client_factory=http_client
    )
    return Client(transport, init_timeout=OAUTH_TIMEOUT)
