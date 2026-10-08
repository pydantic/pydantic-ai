"""Send traces of your agent runs to Logfire.

The default-enabled `observability` plugin: Logfire instrumentation owned by the plugin, not the process.

With `ui_events` on, the same instance also records CLAI's UI interactions (see `pydantic_clai2.ui.telemetry`).
With `token` naming a `/keys` entry, everything goes to that key's Logfire project, such as one a team shares.

`configure` opens the settings menu (turning the plugin on, `c` in `/plugins`, or `/plugins configure
observability`). Each edit is saved at once, and the loader loads the plugin again when the menu closes, so the
next run uses it. Its first row runs the project setup in `logfire_setup`.
"""

import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Annotated, ClassVar, Literal, Self

import logfire
from anyio import CancelScope, to_thread
from opentelemetry.instrumentation.httpx import HTTPX2ClientInstrumentor, HTTPXClientInstrumentor
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, ValidationError

from pydantic_ai.capabilities import AgentCapability, Instrumentation
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_clai2.builtin_plugins.logfire_session import SessionTracing, git_email
from pydantic_clai2.builtin_plugins.logfire_setup import Setup, https_origin, run_setup
from pydantic_clai2.config.api_keys import KeyReference, load_keys
from pydantic_clai2.plugins import Plugin, PluginHost, PluginLoadFailed, SessionEnd, SessionStart, TurnEnd
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow_async
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.tool_output import terminal_text


class LogfireAccount(BaseModel):
    """The Logfire account that signed in during project setup, and the `/keys` entry that setup saved."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    email: str
    token: KeyReference


class LogfireSettings(BaseModel):
    """Non-secret telemetry options; a token stays in `LOGFIRE_TOKEN`, Logfire's credential file, or `/keys`."""

    model_config = ConfigDict(extra='forbid', frozen=True, strict=True, hide_input_in_errors=True)
    service_name: str = Field(default='pydantic-clai2', min_length=1)
    send_to_logfire: Literal[False, 'if-token-present'] = 'if-token-present'
    include_content: bool = True
    include_binary_content: bool = True
    user_tag: Literal['logfire-account', 'git-email', False] = Field(
        default='logfire-account',
        description='Tag session roots with the email of the Logfire account that signed in during project setup, '
        'or with git config user.email. Never added to child spans or logs.',
    )
    account: LogfireAccount | None = Field(
        default=None,
        description='Saved by project setup. Its email tags session roots only while `token` still names the '
        '/keys entry that setup saved, so pointing `token` elsewhere from any CLAI build drops the tag.',
    )
    token: KeyReference | None = Field(
        default=None,
        description='A /keys entry holding the Logfire write token to send with, instead of LOGFIRE_TOKEN or the '
        'credentials file; its project receives the telemetry.',
    )
    base_url: Annotated[str, AfterValidator(https_origin)] | None = Field(
        default=None,
        description='The Logfire to send to, as the setup menu saves it. Unset, the SDK uses LOGFIRE_BASE_URL, '
        'else the region the token names.',
    )
    httpx: bool = Field(
        default=False,
        description='Also trace HTTP requests made with httpx or httpx2. With message content included, capture headers and request and response bodies.',
    )
    ui_events: bool = Field(
        default=True,
        description='Also record UI interactions: menus, commands, settings, plugins, keys, and prompt actions. '
        'With message content included, submitted prompts carry their text.',
    )


class LogfirePlugin(Plugin[LogfireSettings]):
    """Core instrumentation, without changing the supplied agent or global OTel providers."""

    _active_httpx: ClassVar[list['LogfirePlugin']] = []
    """Live opt-in instances, ordered so HTTPX instrumentation can move to a remaining instance on unload."""

    def __init__(self, host: PluginHost[None], settings: LogfireSettings) -> None:
        super().__init__(host, settings)
        self._unsubscribe: Callable[[], None] | None = None
        self._httpx_instrumentors: list[HTTPXClientInstrumentor | HTTPX2ClientInstrumentor] = []
        token, send_to_logfire = _destination(settings, host)
        private_dir = logfire_dir()
        propagator = get_global_textmap()
        try:
            self.instance = logfire.configure(
                local=True,
                send_to_logfire=send_to_logfire,
                token=token,
                service_name=settings.service_name,
                console=False,
                # CLAI passes attributes, never f-strings. Inspecting the caller's source fails once that file
                # changes on disk mid-session, and the warning it prints to stderr tears through the live display.
                inspect_arguments=False,
                config_dir=private_dir,
                data_dir=private_dir,
                # UI events name settings and keys, such as `sessions.naming` or `OPENAI_API_KEY`, that look like secrets.
                scrubbing=logfire.ScrubbingOptions(callback=telemetry.keep_names) if settings.ui_events else None,
                advanced=logfire.AdvancedOptions(base_url=settings.base_url) if settings.base_url else None,
            )
        finally:
            # Even local SDK configuration replaces the process-wide propagator.
            set_global_textmap(propagator)
        try:
            self.instrumentation = Instrumentation(
                settings=InstrumentationSettings(
                    tracer_provider=self.instance.config.get_tracer_provider(),
                    meter_provider=self.instance.config.get_meter_provider(),
                    include_content=settings.include_content,
                    include_binary_content=settings.include_binary_content,
                )
            )
        except BaseException:
            _shutdown(self.instance)
            raise
        self._session_tracing = SessionTracing(instance=self.instance, session_id=lambda: self.host.session_id)
        # UI records and plugin errors are CLAI's own, so they share the session root's scope.
        self._clai2 = logfire.Logfire(config=self.instance.config, otel_scope=telemetry.SCOPE)

    @classmethod
    def from_host(cls, host: PluginHost[None]) -> Self:
        """Tag the identity settings so older builds sharing the database can ignore them."""
        requires = {'user_tag': ['logfire-user-tag'], 'account': ['logfire-user-tag'], 'httpx': ['logfire-httpx']}
        return cls(host, host.settings(LogfireSettings, requires=requires))

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        return (self._session_tracing, self.instrumentation)

    async def configure(self) -> str:
        """The settings menu; its project row runs the setup that signs in and picks where traces go."""

        async def set_up() -> list[str]:
            return [await _configure(self.host, SETUP(self.host))]

        messages = await run_flow_async(FieldMenu(LogfireSource(self.host)), RUNNERS, submenus={PROJECT: set_up})
        return '\n'.join(messages) or 'Logfire settings unchanged.'

    # The UI lifecycle goes only to this plugin's own instance: every enabled copy of the plugin hears these events.
    async def on_session_start(self, event: SessionStart) -> None:
        self._session_tracing.start(await _user_email(self.settings))
        if self.settings.httpx:
            if not self._active_httpx:
                self._instrument_httpx()
            self._active_httpx.append(self)
        if self.settings.ui_events:
            self._unsubscribe = telemetry.subscribe(
                self._clai2, root=self._session_tracing.root, include_content=self.settings.include_content
            )
            model = event.settings.model or 'agent default'
            with telemetry.parent_span(self._session_tracing.root()):
                self._clai2.log('info', 'session started', attributes={'model': model})

    def _instrument_httpx(self) -> None:
        instrumentors = (HTTPXClientInstrumentor(), HTTPX2ClientInstrumentor())
        available = [
            instrumentor for instrumentor in instrumentors if not instrumentor.is_instrumented_by_opentelemetry
        ]
        self.instance.instrument_httpx(capture_all=self.settings.include_content)
        self._httpx_instrumentors = [
            instrumentor for instrumentor in available if instrumentor.is_instrumented_by_opentelemetry
        ]

    async def on_plugin_load_failed(self, event: PluginLoadFailed) -> None:
        with telemetry.parent_span(self._session_tracing.root()):
            self._clai2.log(
                'error', 'Plugin {plugin!r} failed to load', attributes={'plugin': event.plugin}, exc_info=event.error
            )

    async def on_turn_end(self, event: TurnEnd) -> None:
        if self.settings.ui_events:
            with telemetry.parent_span(self._session_tracing.root()):
                self._clai2.log('info', 'turn {outcome}', attributes={'outcome': event.outcome})

    async def on_session_end(self, event: SessionEnd) -> None:
        # HTTPX instrumentors are global: if another plugin instance remains, point them at its live provider.
        if self in self._active_httpx:
            self._active_httpx.remove(self)
        if self._httpx_instrumentors:
            for instrumentor in self._httpx_instrumentors:
                instrumentor.uninstrument()
            self._httpx_instrumentors.clear()
            if self._active_httpx:
                self._active_httpx[-1]._instrument_httpx()
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        self._session_tracing.end(event.reason)
        with CancelScope(shield=True):
            finished = await to_thread.run_sync(_shutdown, self.instance)
            if not finished:
                self.host.console.print(
                    'Logfire shutdown timed out; some telemetry may not have been sent.',
                    style=theme.color(theme.WARNING),
                )


async def _user_email(settings: LogfireSettings) -> str | None:
    """The email `user_tag` names; Git is queried only when chosen."""
    if settings.user_tag == 'git-email':
        return await git_email()
    account = settings.account
    if settings.user_tag == 'logfire-account' and account is not None and account.token == settings.token:
        return account.email
    return None


def logfire_dir() -> Path:
    """CLAI's private Logfire SDK directory: configuration and credentials are read only from here."""
    config_home = Path(os.getenv('XDG_CONFIG_HOME', '')).expanduser()
    if not config_home.is_absolute():
        config_home = Path.home() / '.config'
    return config_home / 'pydantic-clai2' / 'logfire'


CREDENTIALS_FILE = 'logfire_credentials.json'
"""The file the SDK writes on `logfire auth`/`projects use` and reads from `data_dir`."""
RUNNERS: Runners = TERMINAL
"""How the settings menu's widgets are shown; tests swap in scripted ones."""
PROJECT = 'project'
"""The row that runs the project setup instead of editing a value."""


def credentials() -> str:
    """Where the SDK would find a write token when no `/keys` entry is chosen, as a short note."""
    if os.getenv('LOGFIRE_TOKEN'):
        return 'LOGFIRE_TOKEN is set'
    if (logfire_dir() / CREDENTIALS_FILE).is_file():
        return 'credentials file found'
    return 'no LOGFIRE_TOKEN or credentials file'


_BOOLEAN = ('true', 'false')
_INCLUDED = {'true': 'included', 'false': 'left out'}
_PROJECT_ROW = FieldRow(
    key=PROJECT,
    label='Logfire project',
    description=(
        'Enter signs in to Logfire (US, EU, or self-hosted), picks a project, and saves its write token in /keys; '
        'only the key name and the account email are kept here. R goes back to LOGFIRE_TOKEN or the credentials '
        'file in ~/.config/pydantic-clai2/logfire/ (or under $XDG_CONFIG_HOME).'
    ),
    default='LOGFIRE_TOKEN or credentials file',
)
_ROWS = (
    _PROJECT_ROW,
    FieldRow(
        key='send_to_logfire',
        label='Send to Logfire',
        description='Export traces when a token is found. Without one nothing is sent. Tokens never live here.',
        default='if-token-present',
        choices=('if-token-present', 'false'),
        choice_labels={'if-token-present': 'when a token is found', 'false': 'never'},
        allow_custom=False,
    ),
    FieldRow(
        key='service_name',
        label='Service name',
        description='The OpenTelemetry service.name that CLAI traces are filed under in Logfire.',
        default='pydantic-clai2',
    ),
    FieldRow(
        key='include_content',
        label='Message content',
        description='Record prompts, responses, and tool arguments and results in spans.',
        default='true',
        choices=_BOOLEAN,
        choice_labels=_INCLUDED,
        allow_custom=False,
    ),
    FieldRow(
        key='include_binary_content',
        label='Binary content',
        description='Record images, audio, and other file data in spans. Needs message content included.',
        default='true',
        choices=_BOOLEAN,
        choice_labels=_INCLUDED,
        allow_custom=False,
    ),
    FieldRow(
        key='user_tag',
        label='User tag',
        description=LogfireSettings.model_fields['user_tag'].description or '',
        default='logfire-account',
        choices=('logfire-account', 'git-email', 'false'),
        choice_labels={'logfire-account': 'Logfire sign-in email', 'git-email': 'Git email', 'false': 'off'},
        allow_custom=False,
    ),
    FieldRow(
        key='httpx',
        label='HTTP requests',
        description=LogfireSettings.model_fields['httpx'].description or '',
        default='false',
        choices=_BOOLEAN,
        choice_labels={'true': 'recorded', 'false': 'off'},
        allow_custom=False,
    ),
    FieldRow(
        key='ui_events',
        label='UI events',
        description=LogfireSettings.model_fields['ui_events'].description or '',
        default='true',
        choices=_BOOLEAN,
        choice_labels={'true': 'recorded', 'false': 'off'},
        allow_custom=False,
    ),
)


class LogfireSource:
    """The settings menu's rows, read from and saved straight to the plugin's settings; a `FieldSource`."""

    title = 'Observability (Logfire)'

    def __init__(self, host: PluginHost[None]) -> None:
        """Every edit goes through `host.save_settings`."""
        self._host = host

    @property
    def settings(self) -> LogfireSettings:
        """The saved settings, including edits made earlier in this menu."""
        return self._host.settings(LogfireSettings)

    def rows(self) -> Sequence[FieldRow]:
        """Every option; the project row notes where a token would come from without one chosen."""
        note = '' if self.settings.token else credentials()
        return [replace(_PROJECT_ROW, note=note), *_ROWS[1:]]

    def current(self, row: FieldRow) -> str:
        """The value as the user would type it; the project row names the `/keys` entry and the server."""
        settings = self.settings
        if row.key == PROJECT:
            if settings.token is None:
                return row.default
            return settings.token.name + (f' at {settings.base_url}' if settings.base_url else '')
        value: object = getattr(settings, row.key)
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
        return f'Saved {row.label}.'

    def reset(self, row: FieldRow) -> str:
        """Restore one option's default; the project row forgets the chosen key, server, and sign-in email."""
        data = self.settings.model_dump(mode='json')
        for key in ('token', 'base_url', 'account') if row.key == PROJECT else (row.key,):
            data.pop(key, None)
        self._host.save_settings(LogfireSettings.model_validate(data))
        return f'Reset {row.label}.'

    def _updated(self, row: FieldRow, raw: str) -> LogfireSettings:
        value: JsonValue = raw == 'true' if row.choices and raw in _BOOLEAN else raw
        return LogfireSettings.model_validate({**self.settings.model_dump(mode='json'), row.key: value})


def _announce(host: PluginHost[None]) -> Setup:
    def announce(line: str) -> None:
        # Links can come from a self-hosted server, so terminal controls in them are made inert.
        host.console.print(terminal_text(line), markup=False, highlight=False)

    return Setup(announce=announce)


SETUP: Callable[[PluginHost[None]], Setup] = _announce
"""How setup talks to the terminal, the browser, and Logfire; tests swap in scripted ones."""


async def _configure(host: PluginHost[None], setup: Setup) -> str:
    """The setup menu; saving new settings makes the loader load the plugin again, now sending to the project."""
    config = host.settings(LogfireSettings)
    chosen = await run_setup(setup, current=config.base_url, owned=config.token)
    if chosen is None:
        return 'Logfire setup cancelled; settings unchanged.'
    # Setting up a project means sending to it, even if sending had been turned off.
    email = chosen.account_email
    update = {
        'token': chosen.token,
        'base_url': chosen.base_url,
        'account': LogfireAccount(email=email, token=chosen.token) if email else None,
        'send_to_logfire': 'if-token-present',
    }
    host.save_settings(config.model_copy(update=update))
    kept = (
        'only that name and the email you signed in with'
        if email
        else 'only that name; Logfire did not share your email'
    )
    return (
        f'Logfire traces now go to {chosen.project.label}. Its write token is saved in /keys as '
        f'{chosen.token.name}; plugin settings keep {kept}.'
    )


def _destination(
    config: LogfireSettings, host: PluginHost[None]
) -> tuple[str | None, Literal[False, 'if-token-present']]:
    """The token to send with, and whether to send: a chosen key missing from `/keys` keeps telemetry local.

    Falling back to `LOGFIRE_TOKEN` instead would send to a project the user did not choose.
    """
    if config.token is None or config.send_to_logfire is False:
        return None, config.send_to_logfire
    keys = load_keys()
    if config.token.name not in keys:
        host.console.print(
            f'Logfire is not sending telemetry: {config.token.name} is not in /keys. Save it there, then run '
            '/plugins reload observability.',
            style=theme.color(theme.WARNING),
            markup=False,
        )
        return None, False
    return keys[config.token.name].get_secret_value(), config.send_to_logfire


def _shutdown(instance: logfire.Logfire) -> bool:
    # SDK shutdown with flush=True can return on a flush timeout before stopping providers.
    try:
        flushed = instance.force_flush(timeout_millis=3000)
    finally:
        stopped = instance.shutdown(timeout_millis=3000, flush=False)
    return flushed and stopped
