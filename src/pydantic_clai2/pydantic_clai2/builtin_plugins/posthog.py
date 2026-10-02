"""Use PostHog through PostHog's hosted MCP server.

The built-in `posthog` plugin: PostHog's hosted MCP server through harness `PostHog`.

The personal API key is never kept in plugin settings, which are plaintext SQLite. It lives in `/keys`, and the
plugin saves only the key's name, in the credential store beside the `vllm` and `openrouter` connections, so `/keys`
refuses to rename a key PostHog uses. The name is resolved on every request, so replacing the key in `/keys` reaches
the next turn, and a deleted key fails the run instead of connecting without it.

The non-secret options live in the plugin's settings and are edited in the menu that `/plugins configure posthog`
opens. The plugin always builds its own FastMCP client: harness `PostHog`'s `auth` connects only to the US endpoint
and `auth='oauth'` keeps browser tokens in memory with a 5-second connect timeout, so the region, the project and
organization pins, and keyring-backed sign-in all need a client of CLAI's own.
"""

import asyncio
import re
from collections.abc import AsyncGenerator, Generator, Sequence
from dataclasses import replace
from functools import partial
from typing import Literal
from urllib.parse import urlencode, urlsplit

import anyio
import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator
from termflow.tui import MenuBuilder, MenuItem, TextInputBuilder
from termflow.tui.menu import Menu

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai_harness.posthog import PostHog
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import (
    KeyExistsError,
    KeyReference,
    load_keys,
    prompt_api_key,
    resolve_key,
    save_key,
    save_key_connection,
)
from pydantic_clai2.config.credential_store import load_codex_credentials
from pydantic_clai2.mcp import OAUTH_TIMEOUT, TokenStore, http_client, sign_in
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart
from pydantic_clai2.ui.menus.field_menu import (
    SAVE_AND_CLOSE_DETAILS,
    TERMINAL,
    FieldMenu,
    FieldRow,
    Runners,
    first_error,
    is_save_and_close,
    picked,
    run_flow,
    save_and_close_item,
)
from pydantic_clai2.ui.menus.menu_worker import menu_key, run_worker
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._rendering import markdown_style

KEY_NAME = 'POSTHOG_PERSONAL_API_KEY'
"""The `/keys` label for a new key: harness `PostHog`'s documented variable name, used as a label only."""

ACCOUNT = 'posthog'
"""The credential account holding the key reference, never the key."""

TOKENS = 'posthog_plugin'
"""Browser tokens are the `mcp-posthog_plugin` credential. `/mcp` server names cannot contain `_`, so none shares it."""

US_URL = 'https://mcp.posthog.com/mcp'
"""PostHog's default endpoint, the one harness `PostHog` uses."""

EU_URL = 'https://mcp-eu.posthog.com/mcp'
"""PostHog's EU endpoint, which keeps browser sign-in on the EU instance."""

SETUP = 'Run /plugins configure posthog to choose or enter a key.'
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""

FEATURE_GROUPS = (
    *('actions', 'alerts', 'annotations', 'batch_exports', 'business_knowledge', 'canvas', 'cohorts'),
    *('conversations', 'core', 'customer_analytics', 'dashboards', 'data_catalog', 'data_schema'),
    *('data_warehouse', 'debug', 'docs', 'early_access_features', 'endpoints', 'engineering_analytics'),
    *('error_tracking', 'events', 'experiments', 'feedback', 'field_notes', 'flags', 'health_issues'),
    *('hog_function_templates', 'hog_functions', 'insights', 'integrations', 'links', 'llm_analytics', 'logs'),
    *('managed_migrations', 'marketing_analytics', 'messaging', 'mcp_analytics', 'mcp_store', 'metrics'),
    *('notebooks', 'persons', 'platform_features', 'product_analytics', 'reminders', 'replay', 'replay_vision'),
    *('reverse_proxy', 'review_hog', 'signals', 'skills', 'sql', 'stamphog', 'streamlit_apps', 'subscriptions'),
    *('surveys', 'tasks', 'tracing', 'user_interviews', 'visual_review', 'warehouse_sources', 'web_analytics'),
    *('workflows', 'workspace'),
)
"""PostHog's documented feature groups; see the MCP server's README, "Feature Filtering"."""

_GROUP = re.compile(r'[a-z][a-z0-9_]*')
_ID = re.compile(r'[A-Za-z0-9-]+')
_LOOPBACK = ('localhost', '127.0.0.1', '::1')
_SIGN_IN_STATES: dict[bool | None, str] = {
    True: 'signed in through the browser',
    False: 'not signed in; the first prompt that uses it opens the browser',
    None: 'in an unknown sign-in state: the keyring cannot be read',
}


class PostHogSettings(BaseModel):
    """The JSON a `posthog` declaration may carry. Nothing here is secret."""

    # A rejected field may be a pasted key, so errors never echo the input.
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    auth: Literal['key', 'browser'] = Field(
        default='key', description='Connect with a named key from /keys, or sign in through the browser.'
    )
    url: str = Field(default=US_URL, description="PostHog's MCP endpoint: US, EU, or one you run yourself.")
    read_only: bool = Field(default=True, description='Ask PostHog to serve only the tools it marks read-only.')
    features: list[str] | None = Field(default=None, description='Feature groups to offer; `None` offers every group.')
    mode: Literal['auto', 'cli', 'tools'] = Field(
        default='auto', description="One `posthog` tool (cli), one tool per operation (tools), or the server's pick."
    )
    project_id: str | None = Field(default=None, description='Pin every request to this PostHog project.')
    organization_id: str | None = Field(default=None, description='Pin every request to this organization.')
    include_instructions: bool = Field(default=True, description="Forward the server's instructions to the agent.")

    @field_validator('url')
    @classmethod
    def _endpoint(cls, url: str) -> str:
        parts = urlsplit(url)
        local = parts.scheme == 'http' and parts.hostname in _LOOPBACK
        # `?` and `#` are rejected even when empty: the feature filter is appended as the query string.
        if not (parts.scheme == 'https' or local) or not parts.hostname or '?' in url or '#' in url:
            raise ValueError('Use an https:// URL (http:// only for localhost) with no query string.')
        if parts.username is not None or parts.password is not None:
            # Settings are plaintext, so a password in the URL would be stored in the clear.
            raise ValueError('Leave credentials out of the URL; keep keys in /keys.')
        return url

    @field_validator('features')
    @classmethod
    def _groups(cls, groups: list[str] | None) -> list[str] | None:
        # PostHog reads an empty list as every group, which includes write tools.
        if groups is not None and not groups:
            raise ValueError('Choose at least one feature group, or every group.')
        if groups is not None and not all(_GROUP.fullmatch(group) for group in groups):
            raise ValueError('Feature groups are lowercase names such as flags or insights.')
        return groups

    @field_validator('project_id', 'organization_id')
    @classmethod
    def _identifier(cls, value: str | None) -> str | None:
        if value is not None and not _ID.fullmatch(value):
            raise ValueError('Use the ID as PostHog shows it: letters, digits, and dashes.')
        return value


class _Saved(BaseModel):
    token: KeyReference


def saved_key() -> KeyReference | None:
    """The key PostHog connects with, or `None` before one is chosen."""
    raw = load_codex_credentials(account=ACCOUNT)
    if raw is None:
        return None
    try:
        return _Saved.model_validate_json(raw).token
    except ValidationError:
        raise UserError(f'The saved PostHog key reference is invalid. {SETUP}') from None


class SavedKeyAuth(httpx.Auth):
    """Send the chosen `/keys` entry as the bearer token, looked up again for every request."""

    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        """Fail closed when no key is chosen or the chosen one is gone."""
        request.headers['Authorization'] = _bearer()
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        """The same lookup off the event loop: the keyring and the `/keys` lock can block."""
        request.headers['Authorization'] = await anyio.to_thread.run_sync(_bearer, abandon_on_cancel=True)
        yield request


def _bearer() -> str:
    reference = saved_key()
    if reference is None:
        raise UserError(f'PostHog has no key. {SETUP}')
    return f'Bearer {resolve_key(token=reference)}'


class PostHogPlugin(Plugin[PostHogSettings]):
    """`PostHog` from the saved settings, with a settings menu and `/posthog`."""

    def __init__(self, host: PluginHost[None], settings: PostHogSettings) -> None:
        super().__init__(host, settings)
        self.capability = PostHog[None](client=client(settings), include_instructions=settings.include_instructions)
        self.tokens = TokenStore(TOKENS)

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (self.capability,)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='posthog',
                description='Show how PostHog connects, or forget its browser sign-in (/posthog logout).',
                handler=self._command,
                complete=lambda args: ['logout'] if len(args) <= 1 else [],
            ),
        )

    async def configure(self) -> str:
        if not self.host.console.is_terminal:
            return f'Configure PostHog from a terminal: {SETUP}'
        return await _configure(PostHogSource(self.host))

    async def on_session_start(self, event: SessionStart) -> None:
        usable = _usable_key() if self.settings.auth == 'key' else None
        if isinstance(usable, str):
            # Loading anyway keeps the settings menu available; each run fails closed until a key is chosen.
            self.host.console.print(usable, style=theme.color(theme.WARNING), markup=False)

    async def _command(self, args: list[str]) -> str:
        if args == ['logout']:
            await anyio.to_thread.run_sync(self.tokens.forget, abandon_on_cancel=True)
            # The live sign-in still holds the tokens it loaded, so later runs need a fresh one.
            self.capability.client = client(self.host.settings(PostHogSettings))
            return 'Signed out of PostHog. The next prompt that uses browser sign-in opens the browser again.'
        if args:
            raise ValueError('Usage: /posthog [logout]; change settings with /plugins configure posthog')
        return await anyio.to_thread.run_sync(
            _status, self.host.settings(PostHogSettings), self.tokens, abandon_on_cancel=True
        )


def client(settings: PostHogSettings) -> Client[StreamableHttpTransport]:
    """The connection `settings` describe; PostHog reads the region from the key, the rest from the request."""
    url = settings.url
    if settings.features is not None:
        url += '?' + urlencode({'features': ','.join(settings.features)})
    headers: dict[str, str] = {}
    if settings.read_only:
        headers['x-posthog-read-only'] = 'true'
    if settings.mode != 'auto':
        headers['x-posthog-mcp-mode'] = settings.mode
    if settings.project_id is not None:
        headers['x-posthog-project-id'] = settings.project_id
    if settings.organization_id is not None:
        headers['x-posthog-organization-id'] = settings.organization_id
    browser = settings.auth == 'browser'
    transport = StreamableHttpTransport(
        url,
        headers=headers,
        auth=sign_in(TOKENS) if browser else SavedKeyAuth(),
        httpx_client_factory=http_client,
    )
    return Client(transport, init_timeout=OAUTH_TIMEOUT if browser else None)


def _usable_key() -> KeyReference | str:
    """The chosen key if `/keys` still has it, otherwise what is wrong and how to fix it."""
    try:
        reference = saved_key()
    except UserError as exc:
        # Reported, not raised, so the plugin loads and its menu can replace the reference.
        return str(exc)
    if reference is None:
        return f'PostHog has no key yet, so each run fails until one is chosen. {SETUP}'
    if reference.name not in load_keys():
        return f'PostHog uses {reference.name}, which is missing from /keys. {SETUP}'
    return reference


def _status(settings: PostHogSettings, tokens: TokenStore) -> str:
    access = 'read-only' if settings.read_only else 'read-write'
    if settings.auth == 'browser':
        return f'PostHog ({access}, {settings.url}) is {_SIGN_IN_STATES[tokens.signed_in()]}.'
    usable = _usable_key()
    if isinstance(usable, str):
        return usable
    return f'PostHog ({access}, {settings.url}) connects with {usable.name} from /keys.'


_KEY = FieldRow(
    key='key',
    label='API key',
    description=(
        'The /keys entry PostHog connects with: a personal API key made with the "MCP Server" preset. Enter picks a '
        'saved key or saves a new one in /keys as POSTHOG_PERSONAL_API_KEY; only its name is remembered. Plugins '
        'and connections that pick the same key share it. Ignored with browser sign-in.'
    ),
    default='(none)',
)
_FEATURES = FieldRow(
    key='features',
    label='Feature groups',
    description=(
        'The PostHog feature groups the server offers. Enter opens a searchable list; Enter toggles a group and '
        'saves it. Leaving none selected offers every group.'
    ),
    default='every group',
)
_ROWS = (
    _KEY,
    FieldRow(
        key='auth',
        label='Sign-in',
        description='A personal API key from /keys, or a browser sign-in whose tokens are kept in the keyring.',
        default='key',
        choices=('key', 'browser'),
        choice_labels={'key': 'API key from /keys', 'browser': 'browser sign-in'},
        allow_custom=False,
    ),
    FieldRow(
        key='url',
        label='Region',
        description=(
            'US or EU cloud. PostHog routes a key to its own region either way, but browser sign-in stays on the '
            'region you pick. Type a URL for a PostHog MCP server you run yourself.'
        ),
        default=US_URL,
        choices=(US_URL, EU_URL),
        choice_labels={US_URL: 'US cloud (mcp.posthog.com)', EU_URL: 'EU cloud (mcp-eu.posthog.com)'},
    ),
    FieldRow(
        key='read_only',
        label='Tools',
        description='Read-only asks PostHog to drop every tool that makes changes, such as editing feature flags.',
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'read-only', 'false': 'read and write'},
        allow_custom=False,
    ),
    _FEATURES,
    FieldRow(
        key='mode',
        label='Server mode',
        description=(
            'cli wraps every PostHog tool in one `posthog` tool the agent drives with commands; tools registers each '
            'one separately. auto lets the server choose, which is cli for CLAI.'
        ),
        default='auto',
        choices=('auto', 'cli', 'tools'),
        choice_labels={'auto': 'server default', 'cli': 'one posthog tool', 'tools': 'one tool per operation'},
        allow_custom=False,
    ),
    FieldRow(
        key='project_id',
        label='Project ID',
        description='Pin requests to one project, from its PostHog URL. Empty lets the agent switch projects.',
        default='(not set)',
    ),
    FieldRow(
        key='organization_id',
        label='Organization ID',
        description='Pin requests to one organization. Empty lets the agent switch organizations.',
        default='(not set)',
    ),
    FieldRow(
        key='include_instructions',
        label='Server instructions',
        description="Whether the PostHog server's own instructions reach the agent.",
        default='true',
        choices=('true', 'false'),
        choice_labels={'true': 'forwarded', 'false': 'left out'},
        allow_custom=False,
    ),
)


class PostHogSource:
    """The settings menu's rows, read from and saved straight to the plugin's settings."""

    title = 'PostHog'

    def __init__(self, host: PluginHost[None]) -> None:
        """Every edit goes through `host.save_settings`; the key reference goes to the credential store."""
        self._host = host

    @property
    def settings(self) -> PostHogSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(PostHogSettings)

    def rows(self) -> list[FieldRow]:
        """Every option, with the key marked when it needs attention."""
        missing = self.settings.auth == 'key' and isinstance(_usable_key(), str)
        key = replace(_KEY, note='needs a key' if missing else '')
        return [key, *_ROWS[1:]]

    def current(self, row: FieldRow) -> str:
        """The value as the user would type it."""
        if row.key == 'key':
            try:
                reference = saved_key()
            except UserError:
                return '(invalid)'
            return '(none)' if reference is None else reference.name
        if row.key == 'features':
            return ','.join(self.settings.features) if self.settings.features else 'every group'
        value: object = getattr(self.settings, row.key)
        if value is None:
            return '(not set)'
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
        if row.key == 'key':
            return 'The API key has no default: choose another one, or delete keys in /keys.'
        data = self.settings.model_dump(mode='json')
        del data[row.key]
        self.save(PostHogSettings.model_validate(data))
        return f'Reset {row.label}.'

    def save(self, settings: PostHogSettings) -> None:
        """Persist to the plugin's declaration."""
        self._host.save_settings(settings)

    def _updated(self, row: FieldRow, raw: str) -> PostHogSettings:
        data = self.settings.model_dump(mode='json')
        value: JsonValue = raw
        if row.key in ('read_only', 'include_instructions') and raw in ('true', 'false'):
            value = raw == 'true'
        data[row.key] = value
        return PostHogSettings.model_validate(data)


async def _configure(source: PostHogSource) -> str:
    loop = asyncio.get_running_loop()

    def pick_key() -> list[str]:
        # The key picker is async, so the menu's thread hands it back to the event loop.
        try:
            return [asyncio.run_coroutine_threadsafe(choose_key(), loop).result()]
        except (ValueError, UserError) as exc:
            return [str(exc)]

    menu = FieldMenu(source)
    submenus = {'key': pick_key, 'features': lambda: _pick_features(source)}
    messages = await run_worker(lambda: run_flow(menu, RUNNERS, submenus=submenus))
    return '\n'.join(messages) or 'PostHog settings unchanged.'


EVERY_GROUP = 'every'
"""The feature list's first row: offer every group."""


def feature_menu(selected: list[str] | None, cursor: int) -> Menu:
    """Build the searchable feature-group toggle list; `cursor` keeps the place between toggles."""
    chosen = set(selected or ())
    groups = _listed(selected)
    items = [MenuItem(f'{"[x]" if selected is None else "[ ]"} every group (no filter)', value=EVERY_GROUP)]
    items += [MenuItem(f'{"[x]" if group in chosen else "[ ]"} {group}', value=group) for group in groups]
    items.append(save_and_close_item())
    return (
        MenuBuilder('PostHog feature groups')
        .style(markdown_style())
        .items(items)
        .searchable()
        .initial_index(cursor)
        .preview(lambda item: SAVE_AND_CLOSE_DETAILS if is_save_and_close(item) else _FEATURES.description)
        .footer_hint('type to filter - Enter toggle (saved) - Esc done')
        .key_source(menu_key)
        .build()
    )


def _pick_features(source: PostHogSource) -> list[str]:
    cursor = 0
    changed = False
    while True:
        selected = source.settings.features
        item = picked(RUNNERS.run_choice(feature_menu(selected, cursor)))
        if item is None or not isinstance(item.value, str):
            return [f'Saved {_FEATURES.label}.'] if changed else []
        group = item.value
        if group == EVERY_GROUP:
            features = None
        else:
            remaining = [name for name in selected or () if name != group]
            features = remaining if group in (selected or ()) else [*(selected or ()), group]
        source.save(source.settings.model_copy(update={'features': features or None}))
        changed = True
        if group == EVERY_GROUP:
            return [f'Saved {_FEATURES.label}.']
        listed = _listed(features)
        cursor = 1 + listed.index(group) if group in listed else 0


def _listed(selected: list[str] | None) -> list[str]:
    """The documented groups plus any saved ones PostHog added since, so they can be unchecked."""
    return sorted({*FEATURE_GROUPS, *(selected or ())})


class _MaskedPrompt:
    """`prompt_api_key`'s value prompt as a masked termflow input, matching the settings menu."""

    async def prompt_async(self, label: str, /, *, is_password: bool = False) -> str:
        builder = (
            TextInputBuilder(label)
            .style(markdown_style())
            .prompt('Key: ')
            .placeholder('Paste a PostHog personal API key; it is saved in /keys')
            .footer_hint('Enter save - Esc cancel')
            .key_source(menu_key)
        )
        builder.mask()
        widget = builder.build()
        result = await run_worker(lambda: RUNNERS.run_text(widget))
        if result.cancelled or not isinstance(result.value, str):
            raise EOFError  # `prompt_api_key` reads this as cancellation.
        return result.value


async def choose_key() -> str:
    """Pick a saved key or enter a new masked one; only the key's name is saved for PostHog."""
    label = f'PostHog personal API key (new keys are saved in /keys as {KEY_NAME})'
    token = await prompt_api_key(prompt=_MaskedPrompt(), label=label)
    if token is None:
        return 'PostHog key unchanged.'
    if isinstance(token, str):
        if not token.strip():
            return 'PostHog key unchanged.'
        try:
            # Checked and written under one /keys lock, so a key saved meanwhile elsewhere is not overwritten.
            await anyio.to_thread.run_sync(
                partial(save_key, name=KEY_NAME, value=token, replace=False), abandon_on_cancel=True
            )
        except KeyExistsError:
            if not await run_worker(_confirm_replace):
                return 'PostHog key unchanged.'
            await anyio.to_thread.run_sync(partial(save_key, name=KEY_NAME, value=token), abandon_on_cancel=True)
        token = KeyReference(name=KEY_NAME)
    saved = _Saved(token=token).model_dump_json()
    await anyio.to_thread.run_sync(
        partial(save_key_connection, account=ACCOUNT, token=token, value=saved), abandon_on_cancel=True
    )
    return f'PostHog connects with {token.name} from /keys.'


def _confirm_replace() -> bool:
    menu = (
        MenuBuilder(f'{KEY_NAME} is already in /keys')
        .style(markdown_style())
        .items(
            [
                MenuItem('Keep the saved key', value=False),
                MenuItem(f'Replace {KEY_NAME} for every plugin and connection that uses it', value=True),
            ]
        )
        .footer_hint('Enter select - Esc keep')
        .key_source(menu_key)
        .build()
    )
    pick = RUNNERS.run_choice(menu)
    return not pick.cancelled and pick.item is not None and pick.item.value is True
