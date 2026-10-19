"""Use PostHog through PostHog's hosted MCP server.

The built-in `posthog` plugin: PostHog's hosted MCP server through harness `PostHog`.

The personal API key is never kept in plugin settings, which are plaintext SQLite. It lives in `/keys`, and the
plugin saves only the key's name, in the credential store beside the `vllm` and `openrouter` connections, so `/keys`
refuses to rename a key PostHog uses. The name is resolved on every request, so replacing the key in `/keys` reaches
the next turn, and a deleted key fails the run instead of connecting without it.

The non-secret options live in the plugin's settings and are edited in the menu that `/plugins configure posthog`
opens, or the web UI's plugins panel. The plugin always builds its own FastMCP client: harness `PostHog`'s `auth` connects only to the US endpoint
and `auth='oauth'` keeps browser tokens in memory with a 5-second connect timeout, so the region, the project and
organization pins, and keyring-backed sign-in all need a client of CLAI's own.

The browser sign-in runs only from the settings menu or `/posthog login`, never while loading or in a prompt; see
`pydantic_clai2.plugins.sign_in`.
"""

import re
from collections.abc import AsyncGenerator, Generator, Sequence
from dataclasses import replace
from typing import Literal, TypeVar
from urllib.parse import urlencode, urlsplit

import anyio
import httpx
import httpx2
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError, field_validator

from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.posthog import PostHog
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, load_keys, resolve_key, save_key_connection
from pydantic_clai2.config.credential_store import load_codex_credentials
from pydantic_clai2.mcp import OAuthSignIn, http_client
from pydantic_clai2.plugins import Plugin, PluginHost, SessionStart
from pydantic_clai2.plugins.keys import choose_key
from pydantic_clai2.plugins.menus import Dialogs, FieldRow, Option, SettingsMenu
from pydantic_clai2.plugins.sign_in import (
    SUBCOMMANDS,
    run_subcommand,
    sign_in_now,
    status as sign_in_status,
    warn_if_signed_out,
)
from pydantic_clai2.runtime._session import WorkspacelessCapability
from pydantic_clai2.ui.menus.field_menu import first_error
from pydantic_clai2.ui.rendering import theme

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
KEY_LABEL = f'PostHog personal API key (new keys are saved in /keys as {KEY_NAME})'

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

_RequestT = TypeVar('_RequestT', httpx.Request, httpx2.Request)
_GROUP = re.compile(r'[a-z][a-z0-9_]*')
_ID = re.compile(r'[A-Za-z0-9-]+')
_LOOPBACK = ('localhost', '127.0.0.1', '::1')


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


# The connection's client is legacy `httpx` under FastMCP 3 and `httpx2` under FastMCP 4, and each rejects
# another family's `Auth` as an invalid `auth`. The flows below serve either, so be both; the inherited
# `sync_auth_flow` the two declare for their own `Request` is what Pyright objects to.
class SavedKeyAuth(httpx.Auth, httpx2.Auth):  # pyright: ignore[reportIncompatibleMethodOverride]
    """Send the chosen `/keys` entry as the bearer token, looked up again for every request."""

    def auth_flow(self, request: _RequestT) -> Generator[_RequestT, object, None]:
        """Fail closed when no key is chosen or the chosen one is gone."""
        request.headers['Authorization'] = _bearer()
        yield request

    async def async_auth_flow(self, request: _RequestT) -> AsyncGenerator[_RequestT, object]:
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

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        if self.settings.auth == 'browser':
            return (WorkspacelessCapability[None](self.for_run),)
        return (PostHog[None](client=client(self.settings), include_instructions=self.settings.include_instructions),)

    def get_commands(self) -> Sequence[Command]:
        return (
            Command(
                name='posthog',
                description='Show how PostHog connects, or sign in or out in the browser (/posthog login|logout).',
                handler=self._command,
                complete=lambda args: SUBCOMMANDS if len(args) <= 1 else (),
            ),
        )

    async def for_run(self, ctx: RunContext[None]) -> PostHog[None] | None:
        """The browser sign-in's capability, checked per run so signing in applies to the next prompt."""
        method = browser(self.settings)
        if not await anyio.to_thread.run_sync(method.signed_in, abandon_on_cancel=True):
            return None
        return PostHog[None](client=method.client(), include_instructions=self.settings.include_instructions)

    def settings_menu(self) -> SettingsMenu:
        source = PostHogSource(self.host)
        return SettingsMenu(
            source=source,
            actions={_KEY.key: source.pick_key, _AUTH.key: source.pick_auth, _FEATURES.key: source.pick_features},
        )

    async def on_session_start(self, event: SessionStart) -> None:
        if self.settings.auth == 'browser':
            await warn_if_signed_out(browser(self.settings), self.host.console)
            return
        usable = _usable_key()
        if isinstance(usable, str):
            # Loading anyway keeps the settings menu available; each run fails closed until a key is chosen.
            self.host.console.print(usable, style=theme.color(theme.WARNING), markup=False)

    async def _command(self, args: list[str]) -> str:
        settings = self.host.settings(PostHogSettings)
        if args in ([], ['status']):
            return await anyio.to_thread.run_sync(_status, settings, abandon_on_cancel=True)
        message = await run_subcommand(browser(settings), args)
        if message is None:
            raise ValueError(
                'Usage: /posthog [login | logout | status]; change settings with /plugins configure posthog'
            )
        return message


def browser(settings: PostHogSettings) -> OAuthSignIn:
    """The browser sign-in for the endpoint `settings` describe; its tokens are the `mcp-posthog_plugin` credential."""
    return OAuthSignIn(
        name=TOKENS, service='PostHog', setup='/posthog login', url=_url(settings), headers=_headers(settings)
    )


def client(settings: PostHogSettings) -> Client[StreamableHttpTransport]:
    """The key connection `settings` describe; PostHog reads the region from the key, the rest from the request."""
    transport = StreamableHttpTransport(
        _url(settings), headers=_headers(settings), auth=SavedKeyAuth(), httpx_client_factory=http_client
    )
    return Client(transport)


def _url(settings: PostHogSettings) -> str:
    if settings.features is None:
        return settings.url
    return settings.url + '?' + urlencode({'features': ','.join(settings.features)})


def _headers(settings: PostHogSettings) -> dict[str, str]:
    headers: dict[str, str] = {}
    if settings.read_only:
        headers['x-posthog-read-only'] = 'true'
    if settings.mode != 'auto':
        headers['x-posthog-mcp-mode'] = settings.mode
    if settings.project_id is not None:
        headers['x-posthog-project-id'] = settings.project_id
    if settings.organization_id is not None:
        headers['x-posthog-organization-id'] = settings.organization_id
    return headers


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


def _status(settings: PostHogSettings) -> str:
    access = 'read-only' if settings.read_only else 'read-write'
    if settings.auth == 'browser':
        return f'PostHog ({access}, {settings.url}) uses browser sign-in. {sign_in_status(browser(settings))}'
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
_AUTH = FieldRow(
    key='auth',
    label='Sign-in',
    description='A personal API key from /keys, or a browser sign-in whose tokens are kept in the keyring.',
    default='key',
    choices=('key', 'browser'),
    choice_labels={'key': 'API key from /keys', 'browser': 'browser sign-in'},
    allow_custom=False,
)
_ROWS = (
    _KEY,
    _AUTH,
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

    async def pick_key(self, dialogs: Dialogs) -> list[str]:
        """The API key row: pick a saved key or save a new one in `/keys`; only its name is remembered."""
        try:
            reference = await choose_key(name=KEY_NAME, label=KEY_LABEL, dialogs=dialogs)
            if reference is None:
                return ['PostHog key unchanged.']
            await anyio.to_thread.run_sync(_choose, reference, abandon_on_cancel=True)
        except (ValueError, UserError) as exc:
            return [str(exc)]
        return [f'PostHog connects with {reference.name} from /keys.']

    async def pick_auth(self, dialogs: Dialogs) -> list[str]:
        """The Sign-in row: choosing the browser signs in now, behind a waiting screen, unless a sign-in is stored."""
        picked = await dialogs.choose(
            title=f'Choose {_AUTH.label}',
            options=[Option(value=choice, label=_AUTH.display(choice)) for choice in _AUTH.choices],
            current=self.settings.auth,
            description=_AUTH.description,
        )
        if picked is None:
            return []
        saved = await anyio.to_thread.run_sync(self.apply, _AUTH, picked)
        method = browser(self.settings)
        if picked != 'browser' or await anyio.to_thread.run_sync(method.signed_in, abandon_on_cancel=True):
            return [saved]
        return [saved, await sign_in_now(method, dialogs)]

    async def pick_features(self, dialogs: Dialogs) -> list[str]:
        """The Feature groups row: each pick toggles a group and saves it, until Done, every group, or backing out."""
        changed = False
        while True:
            selected = self.settings.features
            group = await dialogs.choose(
                title='PostHog feature groups', options=feature_options(selected), description=_FEATURES_HELP
            )
            if group is None or group == DONE:
                return [f'Saved {_FEATURES.label}.'] if changed else []
            update = {'features': _toggled(selected, group)}
            await anyio.to_thread.run_sync(self.save, self.settings.model_copy(update=update))
            changed = True
            if group == EVERY_GROUP:
                return [f'Saved {_FEATURES.label}.']

    def _updated(self, row: FieldRow, raw: str) -> PostHogSettings:
        data = self.settings.model_dump(mode='json')
        value: JsonValue = raw
        if row.key in ('read_only', 'include_instructions') and raw in ('true', 'false'):
            value = raw == 'true'
        data[row.key] = value
        return PostHogSettings.model_validate(data)


DONE = ':done'
"""The feature list's last row: close it. Group names never start with `:`."""

EVERY_GROUP = 'every'
"""The feature list's first row: offer every group."""

_FEATURES_HELP = 'Each pick toggles a group and saves it. Leaving none selected offers every group.'


def feature_options(selected: list[str] | None) -> list[Option]:
    """The feature-group toggle list, `[x]` marking what is offered, then Done."""
    chosen = set(selected or ())
    every = Option(
        value=EVERY_GROUP,
        label=f'{"[x]" if selected is None else "[ ]"} every group (no filter)',
        description='Offer every group, and close the list.',
    )
    groups = [
        Option(value=group, label=f'{"[x]" if group in chosen else "[ ]"} {group}') for group in _listed(selected)
    ]
    done = Option(value=DONE, label='Done', description='Close the list. Each toggle was saved as you made it.')
    return [every, *groups, done]


def _toggled(selected: list[str] | None, group: str) -> list[str] | None:
    """`selected` with `group` switched; unchecking the last group offers every group, never an empty list."""
    if group == EVERY_GROUP:
        return None
    remaining = [name for name in selected or () if name != group]
    return (remaining if group in (selected or ()) else [*(selected or ()), group]) or None


def _listed(selected: list[str] | None) -> list[str]:
    """The documented groups plus any saved ones PostHog added since, so they can be unchecked."""
    return sorted({*FEATURE_GROUPS, *(selected or ())})


def _choose(reference: KeyReference) -> None:
    """Keep the key's name, never its value, in the credential store."""
    save_key_connection(account=ACCOUNT, token=reference, value=_Saved(token=reference).model_dump_json())
