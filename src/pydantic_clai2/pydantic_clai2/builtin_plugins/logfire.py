"""Send traces of your agent runs to Logfire.

The default-enabled `observability` plugin: Logfire instrumentation owned by the plugin, not the process.

With `ui_events` on, the same instance also records CLAI's UI interactions (see `pydantic_clai2.ui.telemetry`).
With `token` naming a `/keys` entry, everything goes to that key's Logfire project, such as one a team shares.

`configure` opens the settings menu (turning the plugin on, `c` in `/plugins`, or `/plugins configure
observability`). Each edit is saved at once, and the loader loads the plugin again when the menu closes, so the
next run uses it. Its first row runs the project setup in `logfire_setup`.
"""

import asyncio
import os
import sys
import uuid
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import logfire
from anyio import CancelScope, to_thread
from opentelemetry.propagate import get_global_textmap, set_global_textmap
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, JsonValue, ValidationError
from rich.console import RenderableType
from termflow.tui import MenuItem, MenuResult

from pydantic_ai import AgentStreamEvent, FunctionToolResultEvent
from pydantic_ai.capabilities import AgentCapability, Instrumentation
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_ai_harness.ask_user import AskUserRequest, Question, QuestionOption
from pydantic_ai_harness.logfire import AgentControl
from pydantic_ai_harness.policy import PolicyDecision, PolicyRule, decision_attributes
from pydantic_clai2 import policy_state
from pydantic_clai2.builtin_plugins.ask_user_menu import TerminalAnswerer
from pydantic_clai2.builtin_plugins.fleet import Build, Change, Consent, Fleet, FleetControl, Snapshot
from pydantic_clai2.builtin_plugins.fleet_ui import (
    BLOCKED_PREFIX,
    CatalogRow,
    blocked_panel,
    catalog_menu,
    notice_panel,
    preview_panel,
    why_text,
)
from pydantic_clai2.builtin_plugins.logfire_session import SessionTracing, git_email, repo_attributes
from pydantic_clai2.builtin_plugins.logfire_setup import Setup, https_origin, run_setup
from pydantic_clai2.commands import Command
from pydantic_clai2.config.api_keys import KeyReference, load_keys
from pydantic_clai2.plugins import (
    Plugin,
    PluginHost,
    PluginLoadFailed,
    SessionEnd,
    SessionStart,
    TurnEnd,
    TurnStart,
)
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.field_menu import TERMINAL, FieldMenu, FieldRow, Runners, first_error, run_flow_async
from pydantic_clai2.ui.menus.menu_worker import run_worker
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
    ui_events: bool = Field(
        default=True,
        description='Also record UI interactions: menus, commands, settings, plugins, keys, and prompt actions. '
        'With message content included, submitted prompts carry their text.',
    )
    # Hackathon: Logfire as the fleet's control plane.
    agent_control: bool = Field(
        default=True,
        description='Read the company config (`agent__<agent_control_name>`) and catalog '
        '(`catalog__<agent_control_name>`) from Logfire managed variables. Needs an API key that can read variables.',
    )
    agent_control_name: str = Field(default='clai2', min_length=1)
    gateway: bool = Field(
        default=False,
        description='Run `gateway/...` models through the Pydantic AI Gateway with the `api_key` setup saved; '
        'set by setup when your role allows the gateway.',
    )
    project: str | None = Field(
        default=None, description='`organization/project` that setup picked, for links to Logfire.'
    )
    api_key: KeyReference | None = Field(
        default=None,
        description='A /keys entry holding a Logfire API key with `project:read_variables`. Unset, '
        'LOGFIRE_CLAI2_API_KEY or LOGFIRE_API_KEY is used.',
    )
    team: str | None = Field(
        default=None, description='Your team, sent with every span and used for targeting. Unset, CLAI2_TEAM is used.'
    )
    fleet_env_allow: list[str] = Field(
        default_factory=list[str],
        description='Globs over environment variable names that config pushed from Logfire may send, '
        'such as `GITHUB_TOKEN`.',
    )
    allowed_catalog_plugins: list[str] = Field(
        default_factory=list[str],
        description='`module:Class` capability factories the Logfire catalog may enable as plugins.',
    )


class LogfirePlugin(Plugin[LogfireSettings]):
    """Core instrumentation, without changing the supplied agent or global OTel providers."""

    def __init__(self, host: PluginHost[None], settings: LogfireSettings) -> None:
        super().__init__(host, settings)
        self._unsubscribe: Callable[[], None] | None = None
        token, send_to_logfire = _destination(settings, host)
        api_key = _api_key(settings) if settings.agent_control else None
        private_dir = logfire_dir()
        propagator = get_global_textmap()
        try:
            self.instance = logfire.configure(
                local=True,
                send_to_logfire=send_to_logfire,
                token=token,
                service_name=settings.service_name,
                console=False,
                config_dir=private_dir,
                data_dir=private_dir,
                # UI events name settings and keys, such as `sessions.naming` or `OPENAI_API_KEY`, that look like secrets.
                scrubbing=logfire.ScrubbingOptions(callback=telemetry.keep_names),
                advanced=logfire.AdvancedOptions(base_url=settings.base_url) if settings.base_url else None,
                **_variables_options(api_key),
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
        self._session_tracing = SessionTracing(
            instance=self.instance,
            session_id=lambda: self.host.session_id,
            team=settings.team or os.getenv('CLAI2_TEAM'),
            include_repo=settings.include_content,
        )
        self.fleet: Fleet | None = None
        self._pending = 0
        self._asked: set[tuple[str, str]] = set()
        self._announced: set[str] = set()
        """Notices already printed by the idle watcher, which the next turn start must not repeat."""
        self._reported_failures: set[tuple[str, str]] = set()
        self._watcher: asyncio.Task[None] | None = None
        install_id = self._install_id = _install_id()
        if api_key:
            tracing = self._session_tracing
            self.fleet = Fleet(
                instance=self.instance,
                name=settings.agent_control_name,
                state_file=logfire_dir() / 'fleet_state.json',
                allowed_plugins=tuple(settings.allowed_catalog_plugins),
                env_allow=tuple(settings.fleet_env_allow),
                attributes=tracing.identity,
                targeting_key=lambda: tracing.email or install_id,
                user=lambda: tracing.email or 'local',
                scope=lambda: (tracing.team, repo_attributes(Path.cwd()).get('clai2.repo_slug')),
            )
        # UI records and plugin errors are CLAI's own, so they share the session root's scope.
        self._clai2 = logfire.Logfire(config=self.instance.config, otel_scope=telemetry.SCOPE)

    @classmethod
    def from_host(cls, host: PluginHost[None]) -> Self:
        """Tag the identity settings so older builds sharing the database can ignore them."""
        requires = {
            'user_tag': ['logfire-user-tag'],
            'account': ['logfire-user-tag'],
            **dict.fromkeys(
                (
                    'agent_control',
                    'agent_control_name',
                    'api_key',
                    'team',
                    'allowed_catalog_plugins',
                    'fleet_env_allow',
                    'project',
                    'gateway',
                ),
                ['fleet-control'],
            ),
        }
        return cls(host, host.settings(LogfireSettings, requires=requires))

    def get_capabilities(self) -> Sequence[AgentCapability[None]]:
        if self.fleet is None:
            return (self._session_tracing, self.instrumentation)
        tracing = self._session_tracing
        control = AgentControl[None](
            self.fleet.agent_variable,
            targeting_key=lambda _: tracing.email or self._install_id,
            attributes=lambda _: tracing.identity(),
            client_features=('catalog', 'policy', 'applies_to'),
            applies=self.fleet.applies_here,
        )
        fleet_control = FleetControl(fleet=self.fleet, approver=self._approve, blocked_message=self._blocked_message)
        return (self._session_tracing, self.instrumentation, control, fleet_control)

    def _blocked_message(self, rule: PolicyRule) -> str:
        """What the model reads for a denied call; the user sees it as a panel with a Learn more link."""
        what = rule.description.rstrip('.') if rule.description else 'this is not allowed'
        return (
            f'{BLOCKED_PREFIX}: {what} (policy {rule.name}). Ask your admin to change it. '
            'Do not retry it or work around it; suggest a safe alternative to the user instead.'
        )

    async def _approve(self, decision: PolicyDecision) -> bool:
        """Put an `ask` rule's call to the user with the `ask_user` picker; no terminal means no."""
        if not sys.stdin.isatty():
            return False
        question = Question(
            header='Policy',
            question=(
                f'Your organization asks before this: {decision.description or decision.rule} '
                f'(policy {decision.rule}). Run `{decision.subject}`?'
            )[:400],
            options=(
                QuestionOption(label='Allow', description='Run it this once.'),
                QuestionOption(label='Deny', description='Skip it and tell the agent.'),
            ),
        )
        answerer = TerminalAnswerer(full_screen=self.host.full_screen, console=self.host.console)
        response = await answerer(AskUserRequest(questions=(question,)))
        return not response.cancelled and bool(response.answers) and response.answers[0].selected == ('Allow',)

    def _record_outside_run(self, decision: PolicyDecision) -> None:
        """A decision made outside an agent run (the MCP allowlist), on the session root with who made it."""
        attributes = {**decision_attributes(decision, prefix='clai2.policy'), **self._session_tracing.identity()}
        tracer = self.instance.config.get_tracer_provider().get_tracer(telemetry.SCOPE)
        with telemetry.parent_span(self._session_tracing.root()):
            with tracer.start_as_current_span('policy decision', attributes=attributes):
                pass

    def get_commands(self) -> Sequence[Command]:
        if self.fleet is None:
            return ()
        fleet = self.fleet

        async def catalog(args: list[str]) -> str:
            if len(args) == 2 and args[0] in ('enable', 'disable'):
                return fleet.set_opt(args[1], args[0] == 'enable')
            if len(args) == 2 and args[0] == 'why':
                row = next((row for row in fleet.rows(fleet.snapshot()) if row.name == args[1]), None)
                return why_text(row, link=self._link()) if row else f'Nothing called {args[1]} from your organization.'
            if args or not sys.stdin.isatty():
                return fleet.listing()
            return await self._catalog_picker()

        return (
            Command(
                name='catalog',
                description='What your organization provides, and optional add-ons you can turn on',
                handler=catalog,
                complete=lambda _: ('enable', 'disable', 'why'),
            ),
        )

    async def _catalog_picker(self) -> str:
        """Enter toggles an optional add-on and reopens the picker on the same row; Esc closes."""
        assert self.fleet is not None
        fleet, index = self.fleet, 0
        changed: list[str] = []
        while True:
            snapshot = fleet.snapshot()
            rows = fleet.rows(snapshot)
            previewed: list[bool] = []

            def preview(menu: object, item: MenuItem) -> MenuResult:
                previewed.append(True)
                return MenuResult(item=item)

            menu = catalog_menu(rows, snapshot=snapshot, link=self._link(), index=index, hotkeys={'p': preview})
            result = await run_worker(lambda: RUNNERS.run_choice(menu))
            if result.cancelled or result.item is None or not isinstance(result.item.value, CatalogRow):
                return '\n'.join(changed) or 'Catalog unchanged.'
            row = result.item.value
            index = rows.index(row)
            if previewed:
                self.host.console.print(preview_panel(row))
                continue
            if row.elsewhere:
                continue
            if row.declined:
                changed.append(fleet.forget_consent(row.key))
            elif row.toggleable:
                changed.append(fleet.set_opt(row.key, not row.on))

    def get_status_segments(self) -> Sequence[Callable[[], str]]:
        return (self._status,) if self.fleet is not None else ()

    def _status(self) -> str:
        """`◆ Logfire config v10`, plus how many pushed items await the user's OK."""
        assert self.fleet is not None
        snapshot = self.fleet.latest
        versions = snapshot.versions() if snapshot is not None else ''
        if not versions:
            return ''
        pending = f' · {self._pending} awaiting your OK' if self._pending else ''
        return f'Logfire {versions}{pending}'

    def _source(self, snapshot: Snapshot) -> str:
        """Who the config is from: its `display_name`, else the Logfire project, else just Logfire."""
        return snapshot.config.display_name or self.settings.project or 'Logfire'

    def _link(self, anchor: str = '') -> str | None:
        """This agent's configuration page in Logfire (Behavior, where policy lives too), when setup recorded the project."""
        if not self.settings.project or self.fleet is None:
            return None
        base = (self.settings.base_url or 'https://logfire-us.pydantic.dev').rstrip('/')
        return f'{base}/{self.settings.project}/agents/{self.settings.agent_control_name}/configure/edit{anchor}'

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        """A policy block gets a panel for the user; the model already has the plain message."""
        if isinstance(event, FunctionToolResultEvent) and isinstance(event.part, ToolReturnPart):
            content = event.part.content
            if isinstance(content, str) and content.startswith(BLOCKED_PREFIX):
                return blocked_panel(content, link=self._link('#policy'))
        return None

    async def on_turn_start(self, event: TurnStart) -> None:
        await self._ask_consent()
        self._announce_changes()

    async def _ask_consent(self, pending: Sequence[Consent] | None = None) -> bool:
        """Ask about each pushed MCP server or plugin whose target or env changed; headless never asks.

        Returns whether anything was decided. At a turn start every pending item is asked; from the idle
        watcher only ones not yet asked this session.
        """
        if self.fleet is None or not sys.stdin.isatty():
            return False
        if pending is None:
            try:
                pending = self.fleet.build().pending
            except Exception:  # noqa: BLE001 -- the turn-start pass reports a broken config
                return False
        decided = False
        for consent in pending:
            self._asked.add((consent.item.key, consent.fingerprint))
            why = consent.item.provenance.describe()
            question = Question(
                header='Logfire',
                question=(consent.question() + (f' ({why})' if why else ''))[:400],
                options=(
                    QuestionOption(label='Allow', description='Turn it on; asked again if its target or env changes.'),
                    QuestionOption(label='Deny', description='Keep it off.'),
                ),
            )
            answerer = TerminalAnswerer(full_screen=self.host.full_screen, console=self.host.console)
            response = await answerer(AskUserRequest(questions=(question,)))
            if response.cancelled:
                continue  # Asked again next turn.
            allow = bool(response.answers) and response.answers[0].selected == ('Allow',)
            self.fleet.decide(consent, allow=allow)
            decided = True
        return decided

    def _announce_changes(self) -> None:
        """At turn start: show what Logfire pushed since the user last looked, and mark it seen."""
        if self.fleet is None:
            return
        try:
            build = self.fleet.prepare()
            changes = self.fleet.changes(build)
        except Exception as error:  # noqa: BLE001 -- a control-plane hiccup must not block the prompt
            self.host.console.print(f'Logfire fleet config unavailable: {error}', style=theme.color(theme.WARNING))
            return
        self._show(build, changes)
        self._announced.clear()

    def _show(self, build: Build, changes: Sequence[Change]) -> None:
        """Print each change and load failure once; the status row keeps the latest change."""
        fresh = [change for change in changes if change.describe() not in self._announced]
        self._announced.update(change.describe() for change in fresh)
        if fresh:
            self.host.console.print(
                notice_panel(fresh, snapshot=build.snapshot, source=self._source(build.snapshot), link=self._link())
            )
        self._pending = len(build.pending)
        for item, error in build.failed:
            # A broken item fails on every build; say so once per version of it, not on every prompt.
            if (item.key, error) in self._reported_failures:
                continue
            self._reported_failures.add((item.key, error))
            noun = {'mcp_server': 'MCP server'}.get(item.kind, item.kind)
            self.host.console.print(
                f"Couldn't load {noun} {item.name} from Logfire: {error}",
                style=theme.color(theme.WARNING),
                markup=False,
            )

    async def _watch(self) -> None:
        """While the prompt is idle, show a push within seconds instead of at the next prompt.

        Reads the provider's cached values (no span) every few seconds and builds only when they changed. It
        never marks anything seen: the next turn start does, so a notice shown here is not repeated there.
        """
        assert self.fleet is not None
        last = self.fleet.fingerprint()
        while True:
            await asyncio.sleep(WATCH_INTERVAL)
            try:
                current = self.fleet.fingerprint()
                if current == last:
                    continue
                last = current
                build = self.fleet.build()
                self._show(build, self.fleet.changes(build, mark_seen=False))
                unasked = [c for c in build.pending if (c.item.key, c.fingerprint) not in self._asked]
                if unasked and await self._ask_consent(unasked):
                    build = self.fleet.build()
                    self._show(build, self.fleet.changes(build, mark_seen=False))
            except Exception:  # noqa: BLE001 -- a watcher hiccup is retried on the next tick
                continue

    async def configure(self) -> str:
        """The settings menu; its project row runs the setup that signs in and picks where traces go."""

        async def set_up() -> list[str]:
            return [await _configure(self.host, SETUP(self.host))]

        messages = await run_flow_async(FieldMenu(LogfireSource(self.host)), RUNNERS, submenus={PROJECT: set_up})
        return '\n'.join(messages) or 'Logfire settings unchanged.'

    # The UI lifecycle goes only to this plugin's own instance: every enabled copy of the plugin hears these events.
    async def on_session_start(self, event: SessionStart) -> None:
        self._session_tracing.start(await _user_email(self.settings))
        if self.fleet is not None:
            fleet = self.fleet
            policy_state.install(
                policy_state.PolicySource(
                    policy=fleet.current_policy,
                    record=self._record_outside_run,
                )
            )
        self._announce_changes()
        if self.fleet is not None and self._watcher is None:
            self._watcher = asyncio.get_running_loop().create_task(self._watch())
        if self.settings.ui_events:
            self._unsubscribe = telemetry.subscribe(
                self._clai2,
                root=self._session_tracing.root,
                include_content=self.settings.include_content,
                identity=self._session_tracing.ui_identity,
            )
            model = event.settings.model or 'agent default'
            with telemetry.parent_span(self._session_tracing.root()):
                self._clai2.log('info', 'session started', attributes={'model': model})

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
        policy_state.install(None)
        if self._watcher is not None:
            self._watcher.cancel()
            self._watcher = None
        # Stop receiving UI events before the instance shuts down.
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


def _variables_options(api_key: str | None) -> dict[str, Any]:
    """`logfire.configure` arguments that read managed variables; none at all without a key, as before."""
    return {'api_key': api_key, 'variables': logfire.VariablesOptions()} if api_key else {}


WATCH_INTERVAL = 2.0 if os.getenv('CLAI2_FLEET_DEMO') else 5.0
"""Seconds between idle checks for a push from Logfire; `CLAI2_FLEET_DEMO=1` makes it snappier on stage."""


def _install_id() -> str:
    """A stable id for this install: the targeting key without an email, so rollouts don't flip per turn."""
    path = logfire_dir() / 'install_id'
    try:
        return path.read_text(encoding='utf-8').strip() or _new_install_id(path)
    except OSError:
        return _new_install_id(path)


def _new_install_id(path: Path) -> str:
    value = uuid.uuid4().hex
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8')
    except OSError:
        pass  # Unwritable config: this process still gets a stable id.
    return value


def _api_key(settings: LogfireSettings) -> str | None:
    """The API key that reads the fleet config: the chosen `/keys` entry, else the environment."""
    if settings.api_key is not None:
        key = load_keys().get(settings.api_key.name)
        if key is not None:
            return key.get_secret_value()
    return os.getenv('LOGFIRE_CLAI2_API_KEY') or os.getenv('LOGFIRE_API_KEY') or None


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
        key='team',
        label='Team',
        description='Your team, sent with every span and used to target team config from Logfire. Unset, '
        'CLAI2_TEAM is used.',
        default='none',
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
        if value is None:
            return row.default
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
        if row.key == 'team' and raw.strip().lower() in ('', 'none'):
            value = None
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
    chosen = await run_setup(
        setup, current=config.base_url, owned=config.token, owned_variables=config.api_key, team=config.team
    )
    if chosen is None:
        return 'Logfire setup cancelled; settings unchanged.'
    # Setting up a project means sending to it, even if sending had been turned off.
    email = chosen.account_email
    update = {
        'token': chosen.token,
        'base_url': chosen.base_url,
        'account': LogfireAccount(email=email, token=chosen.token) if email else None,
        'send_to_logfire': 'if-token-present',
        'api_key': chosen.variables_key or config.api_key,
        'gateway': chosen.gateway,
        'project': f'{chosen.project.organization_name}/{chosen.project.project_name}',
        'team': chosen.team,
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
