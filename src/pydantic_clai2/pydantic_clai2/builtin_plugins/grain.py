"""Use Grain meeting recordings, with its token kept in /keys.

The built-in `grain` plugin: harness's `Grain` capability, with no secret in plugin settings.

`/grain` (or `c` in `/plugins`, and turning the plugin on) opens a menu for the token source and the non-secret settings; each change is saved at once and applies
to the next prompt. The token comes from, in order: the `GRAIN_ACCESS_TOKEN` environment variable; a named key from
`/keys` (only the key's name is saved, and it is resolved on every run, so replacing the key in `/keys` applies and
deleting it fails closed); or a browser sign-in whose tokens go to the OS keyring the way `/mcp` OAuth servers keep
theirs. A sign-in needs someone at the terminal: in headless mode (`clai2 -p`) with no saved sign-in, connecting
fails with a message saying how to sign in.
"""

import os
from collections.abc import Hashable, Sequence
from functools import partial

from anyio import to_thread
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from prompt_toolkit import PromptSession
from pydantic import BaseModel, ConfigDict, ValidationError

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.grain import Grain
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, prompt_api_key, resolve_key, save_key, save_key_connection
from pydantic_clai2.config.credential_store import delete_credentials, load_codex_credentials
from pydantic_clai2.mcp import OAUTH_TIMEOUT, SignIn, http_client
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart
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


class GrainSignIn(SignIn):
    """Say where the sign-in happens before FastMCP opens the browser, and refuse when no one can sign in."""

    def __init__(self, host: PluginHost[None]) -> None:
        """Tokens persist under `TOKEN_ACCOUNT`."""
        # Grain's client registration rejects a `127.0.0.1` redirect URI with `invalid_redirect_uri`
        # and accepts `localhost` (checked 2026-09-25).
        super().__init__(TOKEN_ACCOUNT, callback_host='localhost')
        self._host = host

    async def redirect_handler(self, authorization_url: str) -> None:
        """Print the URL too, for a browser that does not open (for example over SSH)."""
        try:
            async with self._host.full_screen():
                self._host.console.print(
                    f'Signing in to Grain in your browser. If it does not open, visit:\n{authorization_url}',
                    style=theme.color(theme.INFO),
                    markup=False,
                )
        except RuntimeError as exc:
            # Headless mode binds a screen that refuses interaction, since no one is there to sign in.
            raise UserError(
                f'Grain needs a browser sign-in. Run clai2 interactively once to sign in, or set {KEY_NAME}.'
            ) from exc
        await super().redirect_handler(authorization_url)

    async def forget(self) -> None:
        """Sign out now: drop the saved sign-in and the tokens this session holds in memory."""
        await to_thread.run_sync(self.tokens.forget)
        self.context.clear_tokens()


class GrainConnection:
    """What `/grain` changes mid-session: the settings, the chosen key, and the `Grain` built from them."""

    def __init__(self, host: PluginHost[None]) -> None:
        """Read the saved settings and key choice; the browser sign-in client is ready but not connected."""
        self.host = host
        self.settings = host.settings(GrainSettings)
        self.key = saved_key()
        self.sign_in = GrainSignIn(host)
        transport = StreamableHttpTransport(GRAIN_MCP_URL, auth=self.sign_in, httpx_client_factory=http_client)
        # The default 5 second handshake timeout would end a browser sign-in before the user finishes it.
        self._client = Client(transport, init_timeout=OAUTH_TIMEOUT)
        self._built: tuple[Hashable, Grain[None]] | None = None

    @property
    def source(self) -> GrainSignIn | KeyReference | None:
        """Where this run's token comes from; `None` means the environment variable."""
        if os.environ.get(KEY_NAME):
            return None
        return self.key or self.sign_in

    def save(self, settings: GrainSettings) -> None:
        """Keep new settings for the next prompt and save them to the plugin's declaration."""
        self.settings = settings
        self.host.save_settings(settings)

    def capability(self, _ctx: RunContext[None]) -> Grain[None]:
        """The `Grain` for this run, rebuilt only when the token source or a setting changed."""
        source = self.source
        identity = (source.name if isinstance(source, KeyReference) else source is None, self.settings)
        if self._built is None or self._built[0] != identity:
            read_only, instructions = self.settings.read_only, self.settings.include_instructions
            if source is None:
                built = Grain[None](read_only=read_only, include_instructions=instructions)
            elif isinstance(source, KeyReference):
                built = Grain[None](
                    auth=partial(_resolve, source), read_only=read_only, include_instructions=instructions
                )
            else:
                built = Grain[None](client=self._client, read_only=read_only, include_instructions=instructions)
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
                description='Configure Grain: token source and settings (/grain), or /grain status | key | logout.',
                handler=partial(grain_command, connection=self.connection),
                complete=lambda args: ('key', 'logout', 'status') if len(args) <= 1 else (),
                during_turn=True,
            ),
        )

    async def configure(self) -> str:
        return await configure(self.connection)

    async def on_session_start(self, event: SessionStart) -> None:
        connection = self.connection
        if not connection.settings.model_fields_set and connection.source is connection.sign_in:
            self.host.console.print(
                'Grain uses its defaults (read-only, browser sign-in). /grain picks a /keys token and changes settings.',
                style=theme.color(theme.INFO),
                markup=False,
            )


def _resolve(reference: KeyReference, _ctx: RunContext[None]) -> str:
    # Per run, so a key replaced in /keys applies and a deleted one fails closed.
    return resolve_key(token=reference)


async def grain_command(args: list[str], *, connection: GrainConnection) -> str:
    """Open the settings menu, report the token source, choose a `/keys` token, or sign out."""
    if not args:
        return await configure(connection)
    if args == ['key']:
        return await choose_key(connection)
    if args == ['status']:
        return await status(connection)
    if args != ['logout']:
        raise ValueError('Usage: /grain [status | key | logout]')
    source = connection.source
    if source is None:
        return f'Grain uses {KEY_NAME}, which /grain logout cannot revoke. Unset it, then /plugins reload grain.'
    if isinstance(source, KeyReference):
        return f'Grain uses the /keys entry {source.name}. Choose "No API key" in /grain key to stop using it.'
    await source.forget()
    return 'Signed out of Grain. The next prompt that uses Grain opens the browser to sign in.'


async def status(connection: GrainConnection) -> str:
    """Say where the token comes from, and whether a browser sign-in is saved."""
    source = connection.source
    if source is None:
        return f'Grain uses the {KEY_NAME} environment variable.'
    if isinstance(source, KeyReference):
        return f'Grain uses the /keys entry {source.name}.'
    signed_in = await to_thread.run_sync(source.tokens.signed_in)
    return {
        True: 'Signed in to Grain; the tokens are in the OS keyring. /grain logout signs out.',
        False: 'Not signed in to Grain; the next prompt opens the browser to sign in.',
        None: 'Unknown: the keyring could not be read.',
    }[signed_in]


async def choose_key(connection: GrainConnection) -> str:
    """Pick a `/keys` entry or type a token, saved to `/keys` as `GRAIN_ACCESS_TOKEN`; only the name is kept here."""
    prompt: PromptSession[str] = PromptSession()
    label = f'Grain access token (saved in /keys as {KEY_NAME}; Enter for none): '
    token = await prompt_api_key(prompt=prompt, label=label, optional=True)
    if token is None:
        return 'Grain key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            await to_thread.run_sync(partial(delete_credentials, account=KEY_ACCOUNT))
            connection.key = None
            return 'Grain uses no /keys entry; the next prompt signs in through the browser if needed.'
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
            form.messages.append(await choose_key(connection))
            continue
        if not connection.settings.model_fields_set:
            # Saving the defaults once marks the plugin configured, which ends the startup hint.
            connection.save(GrainSettings.model_validate(connection.settings.model_dump()))
        return '\n'.join(form.messages) or 'Grain settings unchanged.'
