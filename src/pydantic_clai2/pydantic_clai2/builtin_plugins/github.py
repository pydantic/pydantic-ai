"""Work with GitHub through GitHub's hosted MCP tools.

The built-in `github` plugin: GitHub's hosted MCP tools, through harness `GitHub`.

The token comes from the GitHub CLI's browser sign-in, or from `/keys`. The plugin's settings hold only
which one (and a key's name) plus the non-secret `GitHub` options, all edited in the settings menu that
`/plugins configure github` opens.
"""

import asyncio
import concurrent.futures
import re
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Generic, Literal
from urllib.parse import urlsplit

import anyio
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.menu import Menu, MenuResult

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.github import GITHUB_MCP_URL, GitHub
from pydantic_clai2.config.api_keys import KeyExistsError, KeyReference, SavedKey, load_keys, prompt_api_key, save_key
from pydantic_clai2.gh_cli import GhLogin, GhToken, gh_host, gh_token, start_login
from pydantic_clai2.plugins import DepsT, Plugin, PluginHost, SessionStart
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker, worker_stopping
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

KEY_NAME = 'GITHUB_TOKEN'
"""The conventional `/keys` label, shared by every plugin that uses a GitHub token. Not read from the environment."""
SETUP = 'Run /plugins configure github to sign in or choose a token.'
ENTERPRISE = 'enterprise'
"""The host choice that asks for a GitHub Enterprise Cloud URL."""
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""
OPEN_BROWSER: Callable[[str], bool] = webbrowser.open
"""Opens GitHub's device page during `gh` sign-in; tests swap it out."""
FINISHED = 'gh-finished'
"""The key the sign-in screen receives once `gh auth login` exits."""
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

    async def configure(self) -> str:
        return await _configure(GitHubSource(self.host))

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


async def _configure(source: GitHubSource[DepsT]) -> str:
    loop = asyncio.get_running_loop()

    def pick_token() -> list[str]:
        # The key picker is async, so the menu's thread hands it back to the event loop. Its widgets
        # watch their own stop signal, so cancelling this worker must cancel the picker explicitly.
        name = source.settings.token.name
        label = f'GitHub token (saved in /keys as {name})'
        picking = asyncio.run_coroutine_threadsafe(prompt_api_key(prompt=_MaskedPrompt(), label=label), loop)
        while not (picking.done() or worker_stopping()):
            concurrent.futures.wait([picking], timeout=0.05)
        if not picking.done():
            picking.cancel()
            return []
        # Saving happens here, after the cancellable picker, so a cancelled menu saves nothing.
        reference = _saved(name, picking.result())
        if reference is None:
            return []
        source.save(source.settings.model_copy(update={'login': 'key', 'token': reference}))
        return [f'GitHub uses the saved key {reference.name}. Manage it in /keys.']

    def pick_login(menu: FieldMenu) -> list[str]:
        pick = RUNNERS.run_choice(menu.build_choices(_LOGIN))
        if pick.cancelled or pick.item is None:
            return []
        return pick_token() if pick.item.value == 'key' else _sign_in_with_gh(source)

    def flow() -> list[str]:
        menu = FieldMenu(source)

        def refreshed(submenu: Callable[[], list[str]]) -> Callable[[], list[str]]:
            def run() -> list[str]:
                messages = submenu()
                menu.rows = list(source.rows())
                return messages

            return run

        submenus = {'login': refreshed(lambda: pick_login(menu)), 'url': refreshed(lambda: _pick_host(menu, source))}
        return run_flow(menu, RUNNERS, submenus=submenus)

    messages = await run_worker(flow)
    return '\n'.join(messages) or 'GitHub settings unchanged.'


def _sign_in_with_gh(source: GitHubSource[DepsT]) -> list[str]:
    """Use `gh`'s login for the configured host, signing in through the browser when it has none."""
    hostname = gh_host(source.settings.url)
    messages: list[str] = []
    try:
        if gh_token(hostname) is None:
            login = start_login(hostname, stopping=worker_stopping)
            if login is None:
                return []
            if isinstance(login, str):
                return [login]
            if not _wait_for_browser(login):
                login.cancel()
                return []
            messages.append(login.finish())
            if gh_token(hostname) is None:
                return messages
    except UserError as exc:
        return [*messages, str(exc)]
    source.save(source.settings.model_copy(update={'login': 'gh'}))
    return [*messages, f'GitHub uses your GitHub CLI login for {hostname}.']


def _wait_for_browser(login: GhLogin) -> bool:
    """Show the one-time code until `gh` finishes; Enter opens the page again, Esc gives up."""
    _open(login.url)
    menu = _browser_menu(login)
    while True:
        pick = RUNNERS.run_choice(menu)
        if pick.cancelled or pick.item is None:
            return False
        if pick.item.value == FINISHED:
            return True
        _open(login.url)


def _open(url: str) -> None:
    try:
        OPEN_BROWSER(url)
    except webbrowser.Error:
        pass  # The URL is on screen.


def _browser_menu(login: GhLogin) -> Menu:
    def finished_key() -> str:  # pragma: no cover -- read by the terminal menu
        return FINISHED if login.process.poll() is not None else menu_key()

    return (
        MenuBuilder(f'Enter {login.code} at {login.url}')
        .style(markdown_style())
        .items([MenuItem(f'Open {login.url} again', value='open')])
        .on_key(FINISHED, lambda menu, item: MenuResult(item=MenuItem('finished', value=FINISHED)))
        .footer_hint('Approve the code in your browser; gh copied it to your clipboard - Esc cancel')
        .key_source(finished_key)
        .build()
    )


def _pick_host(menu: FieldMenu, source: GitHubSource[DepsT]) -> list[str]:
    pick = RUNNERS.run_choice(menu.build_choices(_HOST))
    if pick.cancelled or pick.item is None:
        return []
    if pick.item.value != ENTERPRISE:
        return [source.apply(_HOST, GITHUB_MCP_URL)]
    typed = RUNNERS.run_text(menu.build_editor(_ENTERPRISE_URL))
    if typed.cancelled or not isinstance(typed.value, str) or not typed.value.strip():
        return []
    return [source.apply(_ENTERPRISE_URL, typed.value.strip())]


class _MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked termflow input, matching the settings menu."""

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        builder = (
            TextInputBuilder(label)
            .style(markdown_style())
            .prompt('Token: ')
            .placeholder('Paste a GitHub token; it is saved in /keys')
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: RUNNERS.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            raise EOFError  # `prompt_api_key` reads this as cancellation.
        return result.value


def _saved(name: str, choice: str | KeyReference | None) -> KeyReference | None:
    """A picked key as is, or a typed token saved under `name`; `None` means cancelled."""
    if choice is None or isinstance(choice, KeyReference):
        return choice
    value = choice.strip()
    if not value:
        return None
    try:
        save_key(name=name, value=value, replace=False)
    except KeyExistsError:
        if not _confirm_replace(name):
            return None
        save_key(name=name, value=value)
    return KeyReference(name=name)


def _confirm_replace(name: str) -> bool:
    menu = (
        MenuBuilder(f'{name} is already in /keys')
        .style(markdown_style())
        .items(
            [
                MenuItem('Keep the saved token', value=False),
                MenuItem(f'Replace {name} for every plugin and connection that uses it', value=True),
            ]
        )
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    pick = RUNNERS.run_choice(menu)
    return not pick.cancelled and pick.item is not None and pick.item.value is True
