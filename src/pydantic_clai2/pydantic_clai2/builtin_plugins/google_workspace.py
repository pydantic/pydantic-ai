"""Use Gmail, Calendar, and Drive through Google's hosted Workspace MCP servers.

The built-in `google_workspace` plugin: Google's hosted Workspace MCP servers, through harness `GoogleWorkspace`.
"""

from collections.abc import Sequence
from functools import partial
from typing import Generic, get_args

import anyio
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from termflow.tui import MenuBuilder, MenuItem

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.google_workspace import GoogleWorkspace, GoogleWorkspaceService
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import (
    KeyReference,
    load_keys,
    prompt_api_key,
    resolve_key,
    save_key,
    save_key_connection,
)
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, run_flow
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

TOKEN_LABEL = 'GOOGLE_ACCESS_TOKEN'
"""The `/keys` name used until `/google_workspace` picks another; a label, not an environment variable."""

ACCOUNT = 'google-workspace'
"""Credential-store account holding the chosen key's name, never its value."""

SERVICES: tuple[GoogleWorkspaceService, ...] = get_args(GoogleWorkspaceService)


class GoogleWorkspaceSettings(BaseModel):
    """The JSON a `google_workspace` declaration may carry. Plain SQLite, so the token never goes here."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    services: list[GoogleWorkspaceService] = Field(
        default_factory=lambda: list[GoogleWorkspaceService](['gmail', 'calendar', 'drive']),
        min_length=1,
        description='Workspace products to connect; the token must carry their scopes.',
    )
    read_only: bool = Field(
        default=True, description='Keep only the tools Google marks as read-only; CLAI runs tools without approval.'
    )
    include_instructions: bool = Field(
        default=True, description="Pass the Google servers' own instructions to the agent."
    )


class Connection(BaseModel):
    """Which saved `/keys` entry the plugin authenticates with."""

    token: KeyReference


def load_connection() -> Connection:
    """The chosen key reference, defaulting to the conventional label so an existing key needs no setup."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return Connection(token=KeyReference(name=TOKEN_LABEL))
    try:
        return Connection.model_validate_json(raw)
    except ValidationError:
        raise UserError('The saved Google Workspace key choice is invalid. Run /google_workspace again.') from None


def missing(name: str) -> str:
    """Explain how to supply the token, naming the key the plugin is looking for."""
    return (
        f'Google Workspace needs a Google OAuth access token in /keys as {name}. '
        'Run /google_workspace to choose a saved key or enter one.'
    )


def access_token() -> str:
    """Resolve at use time: replacing the key in /keys reaches the next turn, and a deleted key fails closed."""
    reference = load_connection().token
    if reference.name not in load_keys():
        raise UserError(missing(reference.name))
    return resolve_key(token=reference)


async def choose_key() -> str:
    """Pick a saved key or enter one; only the key's name is remembered outside /keys."""
    prompt: PromptSession[str] = PromptSession()
    token = await prompt_api_key(prompt=prompt, label=f'Google OAuth access token (saved in /keys as {TOKEN_LABEL}): ')
    if token is None:
        return 'Google Workspace key unchanged.'
    if not isinstance(token, KeyReference):
        await anyio.to_thread.run_sync(partial(save_key, name=TOKEN_LABEL, value=token), abandon_on_cancel=True)
        token = KeyReference(name=TOKEN_LABEL)
    connection = Connection(token=token)
    await anyio.to_thread.run_sync(
        partial(save_key_connection, account=ACCOUNT, token=token, value=connection.model_dump_json()),
        abandon_on_cancel=True,
    )
    return f'Google Workspace uses the saved key {token.name} from the next turn.'


class _PickKey(Exception):
    """Leave the menu worker so the key picker can prompt on the event loop, then reopen the menu."""


_BOOLEAN = ('true', 'false')


class SettingsSource(Generic[DepsT]):
    """The `/google_workspace` rows. Every edit is saved to the plugin declaration as it is made."""

    title = 'Google Workspace'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Read and write through `host`, so edits reach the next run without reloading."""
        self._settings = partial(host.settings, GoogleWorkspaceSettings)
        self._save_settings = host.save_settings
        self.log: list[str] = []
        """What changed, in order; kept here because the key picker restarts the field menu."""

    def rows(self) -> list[FieldRow]:
        """The token's key name first, then the capability's non-secret options."""
        defaults = GoogleWorkspaceSettings()
        return [
            FieldRow(
                key='token',
                label='Access token key',
                default=TOKEN_LABEL,
                description='Which /keys entry holds the Google OAuth access token. Enter lists saved key names or '
                'asks for a new token without echoing it; only the name is stored here. r goes back to '
                f'{TOKEN_LABEL}.',
                note='name in /keys',
            ),
            FieldRow(
                key='services',
                label='Products',
                default=', '.join(defaults.services),
                description='Workspace products to connect. Enter opens a searchable checklist; the token must '
                'carry the scopes each product needs.',
            ),
            FieldRow(
                key='read_only',
                label='Read-only tools',
                default='true',
                choices=_BOOLEAN,
                allow_custom=False,
                description='Keep only the tools Google marks as read-only. CLAI runs tools without asking, so '
                'false lets the agent send, change, and delete.',
            ),
            FieldRow(
                key='include_instructions',
                label='Server instructions',
                default='true',
                choices=_BOOLEAN,
                allow_custom=False,
                description="Pass the Google servers' own instructions to the agent.",
            ),
        ]

    def current(self, row: FieldRow) -> str:
        """The value as the menu shows it; the token row shows only a key name."""
        if row.key == 'token':
            try:
                return load_connection().token.name
            except UserError:
                return '(invalid; Enter to choose again)'
        value = getattr(self._settings(), row.key)
        if isinstance(value, bool):
            return 'true' if value else 'false'
        return ', '.join(value)

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Only the true/false rows take typed values, and their choices are fixed."""
        return None if text in _BOOLEAN else 'Choose true or false.'

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save one true/false option."""
        return self._save(f'Saved {row.label}: {raw}.', **{row.key: raw == 'true'})

    def reset(self, row: FieldRow) -> str:
        """Return one option to its default; for the token row, forget the chosen key name."""
        if row.key == 'token':
            delete_credentials(account=ACCOUNT)
            message = f'Google Workspace uses the saved key {TOKEN_LABEL} again.'
            self.log.append(message)
            return message
        default = getattr(GoogleWorkspaceSettings(), row.key)
        return self._save(f'Reset {row.label}.', **{row.key: default})

    def edit_services(self, runners: Runners) -> list[str]:
        """Toggle products until Esc; each toggle is saved, and the last product cannot be removed."""
        cursor = 0
        while True:
            chosen = self._settings().services
            menu = (
                MenuBuilder('Google Workspace products')
                .style(markdown_style())
                .items([MenuItem(f'[{"x" if name in chosen else " "}] {name}', value=name) for name in SERVICES])
                .searchable()
                .initial_index(cursor)
                .footer_hint('type to filter - Enter toggle - Esc back')
                .key_source(menu_key)
                .build()
            )
            result = runners.run_choice(menu)
            service = result.item.value if result.item is not None else None
            if result.cancelled or service not in SERVICES:
                return []
            cursor = SERVICES.index(service)
            if service in chosen and len(chosen) == 1:
                self.log.append('Google Workspace needs at least one product.')
                continue
            services = [name for name in SERVICES if (name in chosen) != (name == service)]
            self._save(f'Products: {", ".join(services)}.', services=services)

    def pick_key(self) -> list[str]:
        """Hand the key choice to the event loop; see `_PickKey`."""
        raise _PickKey

    def _save(self, message: str, **changes: object) -> str:
        current = self._settings().model_dump()
        self._save_settings(GoogleWorkspaceSettings.model_validate({**current, **changes}))
        self.log.append(message)
        return message


async def configure(host: PluginHost[DepsT], args: list[str], *, runners: Runners = TERMINAL) -> str:
    """Open the settings menu; Esc or Save & close leaves it with every edit already saved."""
    if args:
        raise ValueError('Usage: /google_workspace (opens the settings menu)')
    source = SettingsSource(host)

    def flow() -> list[str]:
        submenus = {'services': partial(source.edit_services, runners), 'token': source.pick_key}
        return run_flow(FieldMenu(source, searchable=False), runners, submenus=submenus)

    while True:
        try:
            await run_worker(flow)
        except _PickKey:
            source.log.append(await choose_key())
            continue
        return '\n'.join(source.log) or 'No changes.'


class GoogleWorkspacePlugin(Plugin[GoogleWorkspaceSettings, DepsT]):
    """Load without a token so `/google_workspace` is available to supply one; every run needs it."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        host = self.host

        def token(ctx: RunContext[DepsT]) -> str:
            # `GoogleWorkspace` drops the tools for a run whose token is empty; failing says why instead.
            return access_token()

        def workspace(ctx: RunContext[DepsT]) -> GoogleWorkspace[DepsT]:
            # Built per run so `/google_workspace` edits apply to the next turn without a reload.
            settings = host.settings(GoogleWorkspaceSettings)
            return GoogleWorkspace[DepsT](
                services=settings.services,
                auth=token,
                read_only=settings.read_only,
                include_instructions=settings.include_instructions,
            )

        return (workspace,)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='google_workspace',
                description='Google Workspace settings: products, read-only tools, and the /keys token',
                handler=partial(configure, self.host),
                during_turn=True,
            ),
        )

    async def configure(self) -> str:
        return await configure(self.host, [])

    async def on_session_start(self, event: SessionStart) -> None:
        try:
            access_token()
        except UserError as exc:
            self.host.console.print(str(exc), style=theme.color(theme.WARNING), markup=False)
