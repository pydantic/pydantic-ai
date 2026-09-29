"""The built-in `slack` plugin: harness `Slack`, set up in the settings menu that `/plugins configure slack` opens.

It connects as the user in one of two ways. A browser sign-in through the user's own CLAI Slack app (`slack_app`)
keeps rotating tokens in the credential store. A user token from `/keys` is referenced by name only, so `/keys`
refuses to rename it while Slack uses it. The plugin's settings hold only non-secret options and the app's public
Client ID.
"""

import asyncio
import os
import webbrowser
from dataclasses import replace
from typing import Generic, Literal

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from termflow.tui import TextInputBuilder

from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.slack import Slack

from . import slack_app, theme
from ._rendering import markdown_style
from .api_keys import KeyReference, load_keys, resolve_key, save_key_connection
from .credential_store import delete_credentials, load_codex_credentials
from .field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from .menu_worker import menu_key, run_worker
from .pkce import PKCESignIn
from .plugin_keys import browser_sign_in, choose_key, on_loop
from .plugins import DepsT, PluginHost, SessionStart, TurnStart

TOKEN_NAME = 'SLACK_USER_TOKEN'
"""The `/keys` label a newly entered token is saved under: harness `Slack`'s documented name, not an exported variable."""
SETUP = 'Run /plugins configure slack to choose its user token from /keys.'
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""

_ACCOUNT = 'slack'


class SlackSettings(BaseModel):
    """The JSON a `slack` declaration may carry. Settings are stored in plaintext, so the token is not one of them."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    auth: Literal['key', 'browser'] = Field(
        default='key', description='Connect with a user token from /keys, or sign in through your Slack app.'
    )
    client_id: str | None = Field(
        default=None,
        pattern=slack_app.CLIENT_ID_PATTERN,
        description="Your CLAI Slack app's Client ID, for browser sign-in. It is public, not a secret.",
    )
    read_only: bool = Field(
        default=True,
        description='Keep only the tools Slack marks read-only, so the agent cannot post or edit as you.',
    )
    include_instructions: bool = Field(default=True, description="Forward the Slack server's instructions.")


class SlackConnection(BaseModel):
    """What the menu saves in the credential store: the name of a `/keys` entry, never its value."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    token: KeyReference


class _NotSetUp(UserError):
    """Slack was never set up: no key chosen, or no app for browser sign-in. Turns skip Slack without a warning."""


def activate(host: PluginHost[DepsT]) -> None:
    """Add `Slack` with a token resolved from `/keys` before each turn; without one, the turn has no Slack tools."""
    settings = host.settings(SlackSettings)
    token: str | None = None

    def current_token(ctx: RunContext[DepsT]) -> str | None:
        return token

    async def refresh() -> None:
        nonlocal token
        try:
            if settings.auth == 'browser':
                token = await signed_in_token(settings)
            else:
                # `/keys` takes a cross-process lock that can wait up to 20 seconds; keep it off the event loop.
                token = await to_thread.run_sync(connected_token)
        except _NotSetUp:
            token = None  # Not set up yet: `/plugins` and the settings menu say so, so turns stay quiet.
        except UserError as exc:
            token = None
            host.console.print(f'Slack tools are off. {exc}', style=theme.color(theme.WARNING), markup=False)

    @host.on('session_start')
    async def start(event: SessionStart) -> None:
        await refresh()

    @host.on('turn_start')
    async def turn(event: TurnStart) -> None:
        await refresh()

    @host.configure
    async def configure() -> str:
        message = await configure_menu(SlackSource(host))
        # A token change needs no reload; the loader reloads for settings changes, which refreshes again.
        await refresh()
        return message

    host.add(
        Slack[DepsT](
            auth=current_token, read_only=settings.read_only, include_instructions=settings.include_instructions
        )
    )


async def signed_in_token(settings: SlackSettings) -> str:
    """The browser sign-in's access token, renewed when close to expiry."""
    if settings.client_id is None:
        raise _NotSetUp('Set up your Slack app: run /plugins configure slack.')
    return await slack_app.session(settings.client_id, read_only=settings.read_only).token()


def connected_token() -> str:
    """The chosen key's current value, so replacing it in `/keys` reaches the next turn and deleting it fails closed."""
    connection = saved_connection()
    if connection is None:
        if os.environ.get(TOKEN_NAME):
            raise UserError(f'CLAI does not read {TOKEN_NAME} from the environment. {SETUP}')
        raise _NotSetUp(SETUP)
    return resolve_key(token=connection.token)


def saved_connection() -> SlackConnection | None:
    """The saved key reference, or `None` when no key was chosen."""
    raw = load_codex_credentials(account=_ACCOUNT)
    if raw is None:
        return None
    try:
        return SlackConnection.model_validate_json(raw)
    except ValidationError:
        raise UserError(f'The saved Slack connection is invalid. {SETUP}') from None


def save_connection(reference: KeyReference) -> None:
    """Persist the key's name; `save_key_connection` checks it still exists under the `/keys` lock."""
    connection = SlackConnection(token=reference)
    save_key_connection(account=_ACCOUNT, token=reference, value=connection.model_dump_json())


def user_token_only(value: str) -> None:
    """Slack's MCP server acts as a user; it rejects bot tokens, so catch one before saving it."""
    if value.startswith('xoxb-'):
        raise ValueError("That is a bot token (xoxb-). Slack's MCP server accepts only user tokens (xoxp-).")


_AUTH = FieldRow(
    key='auth',
    label='Sign-in',
    description=(
        'How Slack connects as you: a user token (xoxp-) from /keys, or a browser sign-in through your own CLAI '
        'Slack app, which renews itself. Both act as you in the workspace they belong to.'
    ),
    default='key',
    choices=('key', 'browser'),
    choice_labels={'key': 'User token from /keys', 'browser': 'Browser sign-in (your Slack app)'},
    allow_custom=False,
)
_APP = FieldRow(
    key='client_id',
    label='Slack app',
    description=(
        "Enter opens Slack's create-app page filled in for CLAI. Pick the workspace, click Create, then paste the "
        "Client ID from the app's Basic Information page; the browser sign-in follows. The Client ID is not a "
        'secret, and no client secret is needed. R forgets the app and signs out.'
    ),
    default='(not set up)',
)
_SIGNED_IN = FieldRow(
    key='account',
    label='Browser sign-in',
    description='Enter signs in through the browser again, for example as another user. R signs out.',
    default='signed out',
)
_TOKEN = FieldRow(
    key='token',
    label='User token',
    description=(
        'The /keys entry Slack connects with. Enter picks a saved key or saves a new user token (xoxp-) as '
        f'{TOKEN_NAME}; only the name is saved for Slack. The token decides the workspace and user, so pick '
        "another key to use another workspace. Slack's MCP server rejects bot tokens (xoxb-). R forgets the choice."
    ),
    default='(not chosen)',
)
_ROWS = (
    FieldRow(
        key='read_only',
        label='Tools',
        description='Read-only keeps only the tools Slack marks read-only, so the agent cannot post or edit as you.',
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'read-only', 'false': 'read and write'},
        allow_custom=False,
    ),
    FieldRow(
        key='include_instructions',
        label='Server instructions',
        description="Whether the Slack server's own instructions reach the agent.",
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'forwarded', 'false': 'left out'},
        allow_custom=False,
    ),
)


class SlackSource(Generic[DepsT]):
    """The settings menu's rows: credentials from the credential store, the rest from plugin settings."""

    title = 'Slack'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Option edits go through `host.save_settings`; credentials go to the credential store."""
        self._host = host

    @property
    def settings(self) -> SlackSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(SlackSettings)

    def session(self) -> PKCESignIn | None:
        """The browser sign-in for the configured app, if one is set up."""
        settings = self.settings
        if settings.client_id is None:
            return None
        return slack_app.session(settings.client_id, read_only=settings.read_only)

    def rows(self) -> list[FieldRow]:
        """Sign-in first, then only the credential rows it uses, each marked when it needs attention."""
        if self.settings.auth == 'browser':
            credentials = [replace(_APP, note=self._app_note()), _SIGNED_IN]
        else:
            credentials = [replace(_TOKEN, note=self._token_note())]
        return [_AUTH, *credentials, *_ROWS]

    def current(self, row: FieldRow) -> str:
        """The value as the user would type it."""
        if row.key == 'client_id':
            return self.settings.client_id or row.default
        if row.key == 'account':
            session = self.session()
            return 'signed in' if session is not None and session.signed_in() else row.default
        if row.key == 'token':
            try:
                connection = saved_connection()
            except UserError:
                return '(invalid)'
            return connection.token.name if connection else row.default
        value: object = getattr(self.settings, row.key)
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
        if row.key == 'read_only' and raw == 'false' and self.settings.auth == 'browser':
            return f'Saved {row.label}. Sign in again (Browser sign-in row) so Slack grants the write scopes.'
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        """Restore an option's default, forget the chosen key, or sign out; the last two turn Slack's tools off."""
        if row.key == 'token':
            delete_credentials(account=_ACCOUNT)
            return 'Slack no longer uses a key from /keys; its tools are off until you choose one.'
        if row.key in ('client_id', 'account') and (session := self.session()):
            session.sign_out()
        if row.key == 'account':  # Not a setting: signing out is all there is to reset.
            return 'Signed out of Slack; its tools are off until you sign in again.'
        data = self.settings.model_dump()
        del data[row.key]
        self._host.save_settings(SlackSettings.model_validate(data))
        return f'Reset {row.label}.'

    def _updated(self, row: FieldRow, raw: str) -> SlackSettings:
        data: dict[str, object] = self.settings.model_dump()
        data[row.key] = {'true': True, 'false': False}.get(raw, raw)
        return SlackSettings.model_validate(data)

    def _app_note(self) -> str:
        if self.settings.client_id is None:
            return 'Enter to set up'
        session = self.session()
        return '' if session is not None and session.signed_in() else 'sign in: Enter on Browser sign-in'

    def _token_note(self) -> str:
        try:
            connection = saved_connection()
        except UserError:
            return 'invalid; choose again'
        if connection is None:
            return 'choose a key'
        try:
            keys = load_keys()
        except UserError:  # The menu must still open, so the user can pick another sign-in.
            return '/keys is unreadable'
        return '' if connection.token.name in keys else 'missing from /keys'


async def configure_menu(source: SlackSource[DepsT]) -> str:
    """The settings menu: Enter edits a row, Save & close or Esc closes, and every change is saved as it is made."""
    loop = asyncio.get_running_loop()

    def pick_token() -> list[str]:
        name = TOKEN_NAME
        label = f'Slack user token (xoxp-); a new one is saved in /keys as {name}'
        try:
            reference = on_loop(
                lambda: choose_key(name=name, label=label, runners=RUNNERS, check=user_token_only), loop
            )
            if reference is None:
                return []
            save_connection(reference)
        except (ValueError, UserError) as exc:
            return [str(exc)]
        return [f'Slack uses the saved key {reference.name}. Manage it in /keys.']

    def sign_in() -> list[str]:
        session = source.session()
        if session is None:
            return ['Set up the Slack app first (Slack app row).']
        try:
            if not on_loop(lambda: browser_sign_in(session, RUNNERS), loop):
                return ['Slack sign-in cancelled.']
        except UserError as exc:
            return [str(exc)]
        return ['Signed in to Slack. Tokens are kept in the OS credential store and renew themselves.']

    def set_up_app() -> list[str]:
        title = "Paste your CLAI Slack app's Client ID (Basic Information > App Credentials)"
        if source.settings.client_id is None:  # With an app set up, Enter only changes the ID; R starts over.
            url = slack_app.create_app_url()
            try:
                opened = webbrowser.open(url)
            except webbrowser.Error:
                opened = False
            if not opened:
                title = f'Open this URL, pick a workspace, and click Create:\n{url}\n\n{title}'
        builder = (
            TextInputBuilder(title)
            .style(markdown_style())
            .prompt('Client ID: ')
            .placeholder('Looks like 1234567890.1234567890')
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        answer = RUNNERS.run_text(builder.build())
        if answer.cancelled or not isinstance(answer.value, str) or not answer.value.strip():
            return []
        if problem := source.problem(_APP, answer.value.strip()):
            return [problem]
        return [source.apply(_APP, answer.value.strip()), *sign_in()]

    menu = FieldMenu(source)
    submenus = {'token': pick_token, 'client_id': set_up_app, 'account': sign_in}
    messages = await run_worker(lambda: run_flow(menu, RUNNERS, submenus=submenus))
    return '\n'.join(messages) or 'Slack settings unchanged.'
