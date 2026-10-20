"""Work with GitHub through GitHub's hosted MCP tools.

The built-in `github` plugin: GitHub's hosted MCP tools, through harness `GitHub`.

The token comes from the GitHub CLI's browser sign-in, or from `/keys`. The plugin's settings hold only
which one (and a key's name) plus the non-secret `GitHub` options, all edited in the settings menu that
`GitHubPlugin.settings_menu` declares, so it opens with `/plugins configure github` and in the web UI's
plugins panel.
"""

import re
import threading
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Generic, Literal
from urllib.parse import urlsplit

import anyio
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.github import GITHUB_MCP_URL, GitHub
from pydantic_clai2.config.api_keys import KeyReference, SavedKey, load_keys
from pydantic_clai2.gh_cli import GhLogin, GhToken, gh_host, gh_token, start_login
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.ui.browser import open_browser
from pydantic_clai2.ui.menus.field_menu import first_error
from pydantic_clai2.ui.rendering import theme

KEY_NAME = 'GITHUB_TOKEN'
"""The conventional `/keys` label, shared by every plugin that uses a GitHub token. Not read from the environment."""
SETUP = 'Run /plugins configure github to sign in or choose a token.'
ENTERPRISE = 'enterprise'
"""The host choice that asks for a GitHub Enterprise Cloud URL."""
OPEN_BROWSER: Callable[[str], bool] = open_browser
"""Opens GitHub's device page during `gh` sign-in; tests swap it out."""
_GROUP = re.compile(r'[a-z][a-z0-9_]*')


class GitHubSettings(BaseModel):
    """The JSON a `github` declaration may carry: `GitHub`'s non-secret options and the name of its token."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)
    login: Literal['gh', 'key'] = Field(
        default='gh', description="Use the GitHub CLI's browser sign-in, or the saved key named by `token`."
    )
    token: KeyReference = Field(
        default_factory=lambda: KeyReference(name=KEY_NAME), description='The saved API key in /keys to connect with.'
    )
    url: str = Field(default=GITHUB_MCP_URL, description="GitHub's MCP endpoint, or a ghe.com one.")
    read_only: bool = Field(default=True, description="Offer only GitHub's read tools.")
    toolsets: list[str] | None = Field(default=None, description="GitHub's tool groups; `None` keeps its defaults.")
    include_instructions: bool = Field(default=True, description="Forward the server's instructions to the agent.")

    @field_validator('url')
    @classmethod
    def _https(cls, url: str) -> str:
        parts = urlsplit(url)
        if parts.scheme != 'https' or not parts.hostname:
            raise ValueError('Use an https:// URL.')
        return url

    @field_validator('toolsets')
    @classmethod
    def _groups(cls, groups: list[str] | None) -> list[str] | None:
        if groups is not None and not groups:
            raise ValueError('Name at least one tool group, or use the server defaults.')
        if groups is not None and not all(_GROUP.fullmatch(group) for group in groups):
            raise ValueError('Tool groups are lowercase names such as repos, separated by commas.')
        return groups


class GitHubPlugin(Plugin[GitHubSettings, DepsT]):
    """`GitHub` with a token resolved from `gh` or `/keys` on every run, and a settings menu."""

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        settings = self.settings
        auth = (
            GhToken(hostname=gh_host(settings.url), setup=SETUP)
            if settings.login == 'gh'
            else SavedKey(name=settings.token.name, setup=SETUP)
        )
        return (
            GitHub[DepsT](
                auth=auth,
                url=settings.url,
                read_only=settings.read_only,
                toolsets=settings.toolsets,
                include_instructions=settings.include_instructions,
            ),
        )

    def settings_menu(self) -> SettingsMenu:
        source = GitHubSource(self.host)
        return SettingsMenu(
            source=source,
            actions={
                'login': lambda dialogs: _pick_login(source, dialogs),
                'url': lambda dialogs: _pick_host(source, dialogs),
            },
        )

    async def on_session_start(self, event: SessionStart) -> None:
        # Loading anyway keeps the settings menu available; each run fails closed until there is a token.
        # A worker thread, because `gh` and the `/keys` lock can take a while.
        problem = await anyio.to_thread.run_sync(_token_problem, self.settings, abandon_on_cancel=True)
        if problem is not None:
            self.host.console.print(
                f'GitHub has no token: {problem} {SETUP}', style=theme.color(theme.WARNING), markup=False
            )


def _token_problem(settings: GitHubSettings) -> str | None:
    """Why no token is available now, or `None` when there is one."""
    if settings.login == 'key':
        return None if settings.token.name in load_keys() else f'{settings.token.name} is not in /keys.'
    hostname = gh_host(settings.url)
    try:
        token = gh_token(hostname)
    except UserError as exc:
        return str(exc)
    return None if token else f'the GitHub CLI is not signed in to {hostname}.'


def enterprise_url(text: str) -> str:
    """The MCP endpoint for a ghe.com host such as `octocorp.ghe.com`; any other URL is kept as typed."""
    text = text.strip()
    parts = urlsplit(text if '://' in text else f'https://{text}')
    host = parts.hostname or ''
    if host.endswith('.ghe.com') and not host.startswith('copilot-api.'):
        return f'https://copilot-api.{host}/mcp'
    if host.startswith('copilot-api.') and parts.path in ('', '/'):
        return f'https://{host}/mcp'
    return parts.geturl()


_LOGIN = FieldRow(
    key='login',
    label='Sign-in',
    description=(
        'Sign in through the GitHub CLI in your browser (gh keeps the token in your OS keyring), or use a '
        'token saved in /keys. Plugin settings never hold the token itself.'
    ),
    default='gh',
    choices=('gh', 'key'),
    choice_labels={'gh': 'GitHub CLI (browser)', 'key': 'Token saved in /keys'},
    allow_custom=False,
)
_HOST = FieldRow(
    key='url',
    label='GitHub host',
    description='github.com, or GitHub Enterprise Cloud with data residency (ghe.com). Enterprise Server has no hosted MCP.',
    default=GITHUB_MCP_URL,
    choices=(GITHUB_MCP_URL, ENTERPRISE),
    choice_labels={GITHUB_MCP_URL: 'github.com', ENTERPRISE: 'GitHub Enterprise Cloud (ghe.com)'},
    allow_custom=False,
)
_ENTERPRISE_URL = FieldRow(
    key='enterprise_url',
    label='Enterprise URL',
    description='Your ghe.com host, such as octocorp.ghe.com, or the full MCP URL.',
    default=GITHUB_MCP_URL,
)
_ROWS = (
    _LOGIN,
    _HOST,
    FieldRow(
        key='read_only',
        label='Tools',
        description='Read-only keeps the agent from changing repositories, issues, or pull requests.',
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'read-only', 'false': 'read and write'},
        allow_custom=False,
    ),
    FieldRow(
        key='toolsets',
        label='Tool groups',
        description=(
            'Comma-separated GitHub tool groups, such as repos, issues, pull_requests, actions, code_security, '
            'discussions, gists, notifications, orgs, projects, or users.'
        ),
        default='default',
        choices=('default', 'all', 'repos,issues,pull_requests', 'context,repos,issues,pull_requests,users'),
        choice_labels={'default': 'server defaults', 'all': 'every group'},
    ),
    FieldRow(
        key='include_instructions',
        label='Server instructions',
        description="Whether the GitHub server's own instructions reach the agent.",
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'forwarded', 'false': 'left out'},
        allow_custom=False,
    ),
)
_FIELDS = {'url': 'url', 'enterprise_url': 'url'}


class GitHubSource(Generic[DepsT]):
    """The settings menu's rows, read from and saved straight to the plugin's settings."""

    title = 'GitHub'

    def __init__(self, host: PluginHost[DepsT]) -> None:
        """Every edit goes through `host.save_settings`."""
        self._host = host

    @property
    def settings(self) -> GitHubSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(GitHubSettings)

    def rows(self) -> list[FieldRow]:
        """Every option, with sign-in marked when no token is available."""
        problem = _token_problem(self.settings)
        return [replace(_LOGIN, note='no token') if problem else _LOGIN, *_ROWS[1:]]

    def current(self, row: FieldRow) -> str:
        """The value as the user would type it."""
        settings = self.settings
        if row.key == 'login' and settings.login == 'key':
            return f'{settings.token.name} in /keys'
        if row.key == 'toolsets':
            return ','.join(settings.toolsets) if settings.toolsets else 'default'
        value: object = getattr(settings, _FIELDS.get(row.key, row.key))
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
        """Restore one option's default."""
        data = self.settings.model_dump(mode='json')
        del data[_FIELDS.get(row.key, row.key)]
        self.save(GitHubSettings.model_validate(data))
        return f'Reset {row.label}.'

    def save(self, settings: GitHubSettings) -> None:
        """Persist to the plugin's declaration."""
        self._host.save_settings(settings)

    def _updated(self, row: FieldRow, raw: str) -> GitHubSettings:
        data = self.settings.model_dump(mode='json')
        value: JsonValue = raw
        if row.key in ('read_only', 'include_instructions'):
            value = raw == 'true' if raw in ('true', 'false') else raw
        elif row.key == 'toolsets':
            value = None if raw == 'default' else [group.strip() for group in raw.split(',')]
        elif row.key == 'enterprise_url':
            value = enterprise_url(raw)
        data[_FIELDS.get(row.key, row.key)] = value
        return GitHubSettings.model_validate(data)


def _options(row: FieldRow) -> list[Option]:
    return [Option(value=choice, label=row.display(choice)) for choice in row.choices]


async def _pick_login(source: GitHubSource[DepsT], dialogs: Dialogs) -> list[str]:
    """Pick where the token comes from, then sign in through `gh` or pick a key in `/keys`."""
    picked = await dialogs.choose(
        title=f'Choose {_LOGIN.label}', options=_options(_LOGIN), current=source.settings.login
    )
    if picked is None:
        return []
    return await (_pick_token(source, dialogs) if picked == 'key' else _sign_in_with_gh(source, dialogs))


async def _pick_token(source: GitHubSource[DepsT], dialogs: Dialogs) -> list[str]:
    """Pick a saved key or save a new masked one; only its name goes into the settings."""
    name = source.settings.token.name
    reference = await choose_key(name=name, label=f'GitHub token (saved in /keys as {name})', dialogs=dialogs)
    if reference is None:
        return []
    settings = source.settings.model_copy(update={'login': 'key', 'token': reference})
    await anyio.to_thread.run_sync(source.save, settings)
    return [f'GitHub uses the saved key {reference.name}. Manage it in /keys.']


async def _sign_in_with_gh(source: GitHubSource[DepsT], dialogs: Dialogs) -> list[str]:
    """Use `gh`'s login for the configured host, signing in through the browser when it has none."""
    hostname = gh_host(source.settings.url)
    messages: list[str] = []
    try:
        if await anyio.to_thread.run_sync(gh_token, hostname, abandon_on_cancel=True) is None:
            if not await dialogs.wait(
                title='Signing in to GitHub',
                description=(
                    'Approve the one-time code in your browser; gh copied it to your clipboard.\n\n'
                    'This closes by itself once gh finishes. Esc cancels and stops gh.'
                ),
                task=lambda show: _gh_login(hostname, show, messages),
            ):
                return []
            if await anyio.to_thread.run_sync(gh_token, hostname, abandon_on_cancel=True) is None:
                return messages
    except UserError as exc:
        return [*messages, str(exc)]
    await anyio.to_thread.run_sync(source.save, source.settings.model_copy(update={'login': 'gh'}))
    return [*messages, f'GitHub uses your GitHub CLI login for {hostname}.']


async def _gh_login(hostname: str, show: Callable[[str], None], messages: list[str]) -> None:
    """Run `gh auth login --web`, showing its code and opening the page; add how it ended to `messages`.

    Cancelling stops `gh`, whether it is still starting or waiting for the browser.
    """
    stop = threading.Event()
    started: list[GhLogin] = []

    def start() -> GhLogin | str | None:
        login = start_login(hostname, stopping=stop.is_set)
        if isinstance(login, GhLogin):
            started.append(login)
            if stop.is_set():  # Cancelled just as the code arrived: `finally` below may have missed it.
                login.cancel()
        return login

    try:
        login = await anyio.to_thread.run_sync(start, abandon_on_cancel=True)
        if isinstance(login, str):
            messages.append(login)
        if not isinstance(login, GhLogin):
            return  # A failure, or `None` once stopped, which nothing awaits by then.
        show(f'Enter {login.code} at {login.url}')
        await anyio.to_thread.run_sync(_open, login.url, abandon_on_cancel=True)
        messages.append(await anyio.to_thread.run_sync(login.finish, abandon_on_cancel=True))
    finally:
        stop.set()
        for login in started:
            if login.process.poll() is None:
                login.cancel()


def _open(url: str) -> None:
    try:
        OPEN_BROWSER(url)
    except webbrowser.Error:
        pass  # The URL is on screen.


async def _pick_host(source: GitHubSource[DepsT], dialogs: Dialogs) -> list[str]:
    """github.com, or a GitHub Enterprise Cloud host typed and checked like any other edit."""
    url = source.settings.url
    picked = await dialogs.choose(
        title=f'Choose {_HOST.label}',
        options=_options(_HOST),
        current=GITHUB_MCP_URL if url == GITHUB_MCP_URL else ENTERPRISE,
    )
    if picked is None:
        return []
    if picked != ENTERPRISE:
        return [await anyio.to_thread.run_sync(source.apply, _HOST, GITHUB_MCP_URL)]
    typed = await dialogs.text(
        title=f'New value for {_ENTERPRISE_URL.label}',
        placeholder=f'current: {url}',
        check=lambda text: source.problem(_ENTERPRISE_URL, text),
        description=_ENTERPRISE_URL.description,
    )
    if not typed:
        return []
    return [await anyio.to_thread.run_sync(source.apply, _ENTERPRISE_URL, typed)]
