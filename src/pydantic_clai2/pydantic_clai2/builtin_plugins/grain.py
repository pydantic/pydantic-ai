"""Use Grain meeting recordings, with its token kept in /keys.

The built-in `grain` plugin: harness's `Grain` capability, with no secret in plugin settings.

`/grain` (or `c` in `/plugins`, and turning the plugin on) opens a menu for the token source and the non-secret settings,
also shown in the web UI's plugins panel; each change is saved at once and applies to the next prompt. The token comes from, in order: the `GRAIN_ACCESS_TOKEN` environment variable; a named key from
`/keys` (only the key's name is saved, and it is resolved on every run, so replacing the key in `/keys` applies and
deleting it fails closed); or a browser sign-in whose tokens go to the OS keyring the way `/mcp` OAuth servers keep
theirs. The browser sign-in runs only from `/grain login` or choosing it in the menu, never while loading or in a
prompt; see `pydantic_clai2.plugins.sign_in`. Until it is done, Grain adds no tools.
"""

import os
from collections.abc import Hashable, Sequence
from functools import partial

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, ValidationError

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.grain import Grain
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, resolve_key, save_key_connection
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials
from pydantic_clai2.mcp import OAuthSignIn
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.plugins.sign_in import SUBCOMMANDS, run_subcommand, sign_in_now, warn_if_signed_out
from pydantic_clai2.runtime.session import WorkspacelessCapability
from pydantic_clai2.ui.menus import dialogs as terminal
from pydantic_clai2.ui.rendering import theme

GRAIN_MCP_URL = 'https://api.grain.com/_/mcp'
"""Grain's hosted MCP endpoint, the one `Grain` connects to when it is given a token rather than a client."""

TOKEN_ACCOUNT = 'plugin_grain'
"""The `TokenStore` name. `/mcp` server names cannot contain `_`, so no `/mcp` server shares these tokens."""

KEY_NAME = 'GRAIN_ACCESS_TOKEN'
"""The environment variable harness's `Grain` reads, and the `/keys` label a token typed into `/grain key` gets."""

KEY_ACCOUNT = 'grain'
"""The credential-store account holding the name of the chosen `/keys` entry, never the token."""


class KeyChoice(BaseModel):
    """The `/keys` entry Grain uses, by name. `key_users` reads this to stop renaming a key still in use."""

    token: KeyReference


def saved_key() -> KeyReference | None:
    """The `/keys` entry chosen with `/grain key`, or `None` when there is none."""
    raw = load_codex_credentials(account=KEY_ACCOUNT)
    if raw is None:
        return None
    try:
        return KeyChoice.model_validate_json(raw).token
    except ValidationError:
        raise UserError('The saved Grain key choice is invalid. Choose a key again with /grain key.') from None


class GrainSettings(BaseModel):
    """Plugin settings, edited in the `/grain` menu. They are plaintext, so they hold no token.

    Grain's MCP endpoint is fixed and has no workspace or base URL option, so these are all the knobs `Grain` has.
    """

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    read_only: bool = True
    """Offer only the tools Grain marks read-only; `false` also lets the agent create clips and tag meetings."""
    include_instructions: bool = True
    """Pass Grain's own server instructions to the agent."""


SIGN_IN = OAuthSignIn(
    name=TOKEN_ACCOUNT,
    service='Grain',
    setup='/grain login',
    url=GRAIN_MCP_URL,
    # Grain's client registration rejects a `127.0.0.1` redirect URI with `invalid_redirect_uri`
    # and accepts `localhost` (checked 2026-09-25).
    callback_host='localhost',
)
"""The browser sign-in; its tokens are the keyring credential `mcp-plugin_grain`."""


class GrainConnection:
    """What `/grain` changes mid-session: the settings, the chosen key, and the `Grain` built from them."""

    def __init__(self, host: PluginHost[None]) -> None:
        """Read the saved settings and key choice."""
        self.host = host
        self.settings = host.settings(GrainSettings)
        self.key = saved_key()
        self._built: tuple[Hashable, Grain[None]] | None = None

    @property
    def source(self) -> OAuthSignIn | KeyReference | None:
        """Where this run's token comes from; `None` means the environment variable."""
        if os.environ.get(KEY_NAME):
            return None
        return self.key or SIGN_IN

    def save(self, settings: GrainSettings) -> None:
        """Keep new settings for the next prompt and save them to the plugin's declaration."""
        self.settings = settings
        self.host.save_settings(settings)

    def mark_configured(self) -> None:
        """Save the defaults once, which marks the plugin configured and ends the startup hint."""
        if not self.settings.model_fields_set:
            self.save(GrainSettings.model_validate(self.settings.model_dump()))

    async def capability(self, _ctx: RunContext[None]) -> Grain[None] | None:
        """The `Grain` for this run, or nothing while the browser sign-in is not done.

        A token source's `Grain` is rebuilt only when the source or a setting changed. The browser sign-in's is
        new each run, since FastMCP keeps tokens in memory once connected and `/grain logout` must reach the next run.
        """
        source = self.source
        read_only, instructions = self.settings.read_only, self.settings.include_instructions
        if isinstance(source, OAuthSignIn):
            if not await to_thread.run_sync(source.signed_in, abandon_on_cancel=True):
                return None
            return Grain[None](
                defer_loading=True, client=source.client(), read_only=read_only, include_instructions=instructions
            )
        identity = (source.name if isinstance(source, KeyReference) else None, self.settings)
        if self._built is None or self._built[0] != identity:
            if source is None:
                built = Grain[None](defer_loading=True, read_only=read_only, include_instructions=instructions)
            else:
                built = Grain[None](
                    defer_loading=True,
                    auth=partial(_resolve, source),
                    read_only=read_only,
                    include_instructions=instructions,
                )
            self._built = (identity, built)
        return self._built[1]


class GrainPlugin(Plugin[GrainSettings]):
    """`Grain`, authenticated by `GRAIN_ACCESS_TOKEN`, a named `/keys` entry, or a browser sign-in."""

    def __init__(self, host: PluginHost[None], settings: GrainSettings) -> None:
        super().__init__(host, settings)
        self.connection = GrainConnection(host)

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (WorkspacelessCapability[None](self.connection.capability),)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='grain',
                description='Configure Grain (/grain), or /grain status | key | login | logout.',
                handler=partial(grain_command, plugin=self),
                complete=lambda args: ('key', *SUBCOMMANDS) if len(args) <= 1 else (),
                during_turn=True,
            ),
        )

    def settings_menu(self) -> SettingsMenu:
        return SettingsMenu(
            source=GrainForm(self.connection),
            actions={TOKEN_ROW: partial(choose_token, connection=self.connection)},
            searchable=False,
        )

    async def configure(self) -> str:
        message = await super().configure()
        if self.host.console.is_terminal:
            # Closing the terminal menu counts as configuring it; the web UI's panel has no close to hook.
            await to_thread.run_sync(self.connection.mark_configured)
        return message

    async def on_session_start(self, event: SessionStart) -> None:
        connection = self.connection
        if connection.source is not SIGN_IN:
            return
        if not connection.settings.model_fields_set:
            self.host.console.print(
                'Grain uses its defaults (read-only, browser sign-in). /grain picks a /keys token and changes settings.',
                style=theme.color(theme.INFO),
                markup=False,
            )
        await warn_if_signed_out(SIGN_IN, self.host.console)


def _resolve(reference: KeyReference, _ctx: RunContext[None]) -> str:
    # Per run, so a key replaced in /keys applies and a deleted one fails closed.
    return resolve_key(token=reference)


async def grain_command(args: list[str], *, plugin: GrainPlugin) -> str:
    """Open the settings menu, report the token source, choose a `/keys` token, sign in, or sign out."""
    if not args:
        return await plugin.configure()
    if args == ['key']:
        return '\n'.join(await choose_token(terminal.TerminalDialogs(terminal.RUNNERS), connection=plugin.connection))
    source = plugin.connection.source
    if args == ['status']:
        if source is None:
            return f'Grain uses the {KEY_NAME} environment variable.'
        if isinstance(source, KeyReference):
            return f'Grain uses the /keys entry {source.name}.'
    elif args == ['logout']:
        if source is None:
            return f'Grain uses {KEY_NAME}, which /grain logout cannot revoke. Unset it, then /plugins reload grain.'
        if isinstance(source, KeyReference):
            return f'Grain uses the /keys entry {source.name}. Choose "No API key" in /grain key to stop using it.'
    message = await run_subcommand(SIGN_IN, args)
    if message is None:
        raise ValueError('Usage: /grain [status | key | login | logout]')
    return message


_KEY = 'key'
_NO_KEY = 'none'
_UNCHANGED = 'Grain key unchanged.'


async def choose_token(dialogs: Dialogs, *, connection: GrainConnection) -> list[str]:
    """Pick a `/keys` entry or type a token, saved to `/keys` as `GRAIN_ACCESS_TOKEN`; only the name is kept here.

    Choosing no key means the browser sign-in, which starts now unless one is stored.
    """
    source = await dialogs.choose(
        title='Grain token',
        options=[
            Option(
                value=_KEY,
                label='A /keys entry',
                description=f'Pick a saved key, or type a token (masked), saved in /keys as {KEY_NAME}.',
            ),
            Option(value=_NO_KEY, label='No API key', description='Sign in to Grain in the browser instead.'),
        ],
        current=_NO_KEY if connection.key is None else _KEY,
        description=f'Only the key name is saved. {KEY_NAME} in the environment overrides this while it is set.',
    )
    if source is None:
        return [_UNCHANGED]
    if source == _NO_KEY:
        return await _use_sign_in(dialogs, connection=connection)
    token = await choose_key(name=KEY_NAME, label='Grain access token', dialogs=dialogs)
    if token is None:
        return [_UNCHANGED]
    choice = KeyChoice(token=token).model_dump_json()
    await to_thread.run_sync(partial(save_key_connection, account=KEY_ACCOUNT, token=token, value=choice))
    connection.key = token
    await to_thread.run_sync(connection.mark_configured)
    return [f'Grain uses the /keys entry {token.name} from the next prompt.']


async def _use_sign_in(dialogs: Dialogs, *, connection: GrainConnection) -> list[str]:
    """Forget the chosen `/keys` entry, then sign in through the browser unless a sign-in is stored."""
    await to_thread.run_sync(partial(delete_credentials, account=KEY_ACCOUNT))
    connection.key = None
    await to_thread.run_sync(connection.mark_configured)
    chosen = 'Grain uses no /keys entry, so it uses the browser sign-in.'
    if await to_thread.run_sync(SIGN_IN.signed_in, abandon_on_cancel=True):
        return [chosen]
    return [chosen, await sign_in_now(SIGN_IN, dialogs)]


TOKEN_ROW = 'token'
_BOOLEANS = ('true', 'false')
_ROWS = (
    FieldRow(
        key=TOKEN_ROW,
        label='Token',
        description=(
            f'Enter picks a /keys entry, types a new token (masked, saved in /keys as {KEY_NAME}), or chooses '
            f'"No API key" for the browser sign-in. Only the key name is saved. {KEY_NAME} in the environment '
            'overrides this while it is set.'
        ),
        default='browser sign-in',
    ),
    FieldRow(
        key='read_only',
        label='Tools',
        description='Read-only offers only the tools Grain marks read-only. All tools also lets the agent create '
        'clips and tag meetings.',
        default='true',
        choices=_BOOLEANS,
        choice_labels={'true': 'read-only', 'false': 'all tools'},
        allow_custom=False,
    ),
    FieldRow(
        key='include_instructions',
        label='Server instructions',
        description="Pass Grain's own instructions for its tools to the agent.",
        default='true',
        choices=_BOOLEANS,
        choice_labels={'true': 'included', 'false': 'left out'},
        allow_custom=False,
    ),
)


class GrainForm:
    """The `/grain` menu's rows; every edit is saved to the plugin settings at once."""

    title = 'Grain'

    def __init__(self, connection: GrainConnection) -> None:
        """Edits go to `connection`, so they apply to the next prompt without a reload."""
        self.connection = connection

    def rows(self) -> Sequence[FieldRow]:
        """The token source, then each `GrainSettings` field."""
        return _ROWS

    def current(self, row: FieldRow) -> str:
        """The token source by name, or a setting as `true`/`false`."""
        if row.key != TOKEN_ROW:
            return str(getattr(self.connection.settings, row.key)).lower()
        source = self.connection.source
        if source is None:
            return f'{KEY_NAME} (environment)'
        return f'/keys: {source.name}' if isinstance(source, KeyReference) else row.default

    def problem(self, row: FieldRow, text: str) -> str | None:
        """Every editable row is a fixed choice, so nothing typed needs checking."""
        return None

    def apply(self, row: FieldRow, raw: str) -> str:
        """Save one setting."""
        self.connection.save(self.connection.settings.model_copy(update={row.key: raw == 'true'}))
        return f'Grain {row.label.lower()}: {row.display(raw)}. Saved; applies to the next prompt.'

    def reset(self, row: FieldRow) -> str:
        """Put a setting back to its default; the token row has no default to go back to."""
        if row.key == TOKEN_ROW:
            return 'Choose "No API key" to go back to the browser sign-in.'
        return self.apply(row, row.default)
