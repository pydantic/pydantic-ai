"""Use Grain meeting recordings, with its token kept in /keys.

The built-in `grain` plugin: harness's `Grain` capability, with no secret in plugin settings.

`/grain` (or `c` in `/plugins`, and turning the plugin on) opens a menu for the token source and the non-secret settings; each change is saved at once and applies
to the next prompt. The token comes from, in order: the `GRAIN_ACCESS_TOKEN` environment variable; a named key from
`/keys` (only the key's name is saved, and it is resolved on every run, so replacing the key in `/keys` applies and
deleting it fails closed); or a browser sign-in whose tokens go to the OS keyring the way `/mcp` OAuth servers keep
theirs. The browser sign-in runs only from `/grain login` or choosing it in the menu, never while loading or in a
prompt; see `pydantic_clai2.plugins.sign_in`. Until it is done, Grain adds no tools.
"""

import os
from collections.abc import Hashable, Sequence
from functools import partial

from anyio import to_thread
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, ValidationError

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.grain import Grain
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, prompt_api_key, resolve_key, save_key, save_key_connection
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials
from pydantic_clai2.mcp import OAuthSignIn
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.sign_in import SUBCOMMANDS, run_subcommand, sign_in_now, warn_if_signed_out
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, run_flow
from pydantic_clai2.ui.menus.menu_worker import run_worker
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
            return Grain[None](client=source.client(), read_only=read_only, include_instructions=instructions)
        identity = (source.name if isinstance(source, KeyReference) else None, self.settings)
        if self._built is None or self._built[0] != identity:
            if source is None:
                built = Grain[None](read_only=read_only, include_instructions=instructions)
            else:
                built = Grain[None](
                    auth=partial(_resolve, source), read_only=read_only, include_instructions=instructions
                )
            self._built = (identity, built)
        return self._built[1]


class GrainPlugin(Plugin[GrainSettings]):
    """`Grain`, authenticated by `GRAIN_ACCESS_TOKEN`, a named `/keys` entry, or a browser sign-in."""

    def __init__(self, host: PluginHost[None], settings: GrainSettings) -> None:
        super().__init__(host, settings)
        self.connection = GrainConnection(host)

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (self.connection.capability,)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='grain',
                description='Configure Grain (/grain), or /grain status | key | login | logout.',
                handler=partial(grain_command, connection=self.connection),
                complete=lambda args: ('key', *SUBCOMMANDS) if len(args) <= 1 else (),
                during_turn=True,
            ),
        )

    async def configure(self) -> str:
        return await configure(self.connection)

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


async def grain_command(args: list[str], *, connection: GrainConnection) -> str:
    """Open the settings menu, report the token source, choose a `/keys` token, sign in, or sign out."""
    if not args:
        return await configure(connection)
    if args == ['key']:
        return await choose_key(connection)
    source = connection.source
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


async def choose_key(connection: GrainConnection, *, runners: Runners | None = None) -> str:
    """Pick a `/keys` entry or type a token, saved to `/keys` as `GRAIN_ACCESS_TOKEN`; only the name is kept here.

    Choosing none means the browser sign-in, which starts now unless one is stored.
    """
    prompt: PromptSession[str] = PromptSession()
    label = f'Grain access token (saved in /keys as {KEY_NAME}; Enter for none): '
    token = await prompt_api_key(prompt=prompt, label=label, optional=True)
    if token is None:
        return 'Grain key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            await to_thread.run_sync(partial(delete_credentials, account=KEY_ACCOUNT))
            connection.key = None
            chosen = 'Grain uses no /keys entry, so it uses the browser sign-in.'
            if await to_thread.run_sync(SIGN_IN.signed_in, abandon_on_cancel=True):
                return chosen
            return f'{chosen}\n{await sign_in_now(SIGN_IN, runners)}'
        await to_thread.run_sync(partial(save_key, name=KEY_NAME, value=token))
        token = KeyReference(name=KEY_NAME)
    choice = KeyChoice(token=token).model_dump_json()
    await to_thread.run_sync(partial(save_key_connection, account=KEY_ACCOUNT, token=token, value=choice))
    connection.key = token
    return f'Grain uses the /keys entry {token.name} from the next prompt.'


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


class _PickToken(Exception):
    """Leave the synchronous field list so the token row can run the async `/keys` picker."""


class GrainForm:
    """The `/grain` menu's rows; every edit is saved to the plugin settings at once."""

    title = 'Grain'

    def __init__(self, connection: GrainConnection) -> None:
        """Edits go to `connection`, so they apply to the next prompt without a reload."""
        self.connection = connection
        self.messages: list[str] = []

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
        message = f'Grain {row.label.lower()}: {row.display(raw)}. Saved; applies to the next prompt.'
        self.messages.append(message)
        return message

    def reset(self, row: FieldRow) -> str:
        """Put a setting back to its default; the token row has no default to go back to."""
        if row.key == TOKEN_ROW:
            return 'Choose "No API key" to go back to the browser sign-in.'
        return self.apply(row, row.default)


def _pick_token() -> list[str]:
    raise _PickToken


async def configure(connection: GrainConnection, runners: Runners | None = None) -> str:
    """Show the `/grain` menu until Esc; the token row leaves it for the `/keys` picker and comes back."""
    runners = runners or TERMINAL
    form = GrainForm(connection)
    menu = FieldMenu(form, searchable=False)
    while True:
        try:
            await run_worker(lambda: run_flow(menu, runners, submenus={TOKEN_ROW: _pick_token}))
        except _PickToken:
            form.messages.append(await choose_key(connection, runners=runners))
            continue
        if not connection.settings.model_fields_set:
            # Saving the defaults once marks the plugin configured, which ends the startup hint.
            connection.save(GrainSettings.model_validate(connection.settings.model_dump()))
        return '\n'.join(form.messages) or 'Grain settings unchanged.'
