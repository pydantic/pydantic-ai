"""Interactive terminal shell around a capability-independent session."""

import asyncio
import math
import os
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager, nullcontext
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import TYPE_CHECKING, Generic, TypeVar

from anyio import CancelScope, Lock, create_memory_object_stream, create_task_group, to_thread
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import History
from pydantic import JsonValue, ValidationError
from rich.console import Console

from pydantic_ai import Agent, AgentRunResult, AgentStreamEvent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from pydantic_clai2 import warm_imports
from pydantic_clai2.cli.command_context import CommandContext, CommandProvider
from pydantic_clai2.cli.effort import effort_command, effort_completions
from pydantic_clai2.cli.self_update import Relaunch, Updates
from pydantic_clai2.cli.shell_passthrough import HELP as SHELL_HELP, run_shell_command, shell_command
from pydantic_clai2.commands import (
    Command,
    Commands,
    config_command,
    config_completions,
    expand_bare_command,
    is_command_input,
    is_silent,
    set_completions,
)
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.project_settings import ProjectSettings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.customization import customization_guide
from pydantic_clai2.errors import error_message
from pydantic_clai2.models import login_names
from pydantic_clai2.models.chains import settings_model as chain_settings_model
from pydantic_clai2.models.profiles import ALL, DEFAULT, ModelRef, base_model, parse_model, provider_of
from pydantic_clai2.plugins import (
    ModelProvider,
    PluginLogin,
    Renderer,
    SessionEndReason,
    SessionStart,
    TurnEnd,
    TurnStart,
    bare_screen,
)
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.runtime._session import (
    ModelDefaults,
    Session,
    SessionModels,
    StockAgent,
    current_session_id,
    local_workspace,
    resolve_model_name,
)
from pydantic_clai2.runtime.capability_guard import CapabilitySetupError
from pydantic_clai2.runtime.forks import Forks
from pydantic_clai2.runtime.imported_sessions import IMPORT_SOURCES, ImportSource
from pydantic_clai2.runtime.reloading import reload_clai
from pydantic_clai2.runtime.session_settings import SessionSettings
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.runtime.speculation import Speculation
from pydantic_clai2.runtime.tasks import Tasks, task_row
from pydantic_clai2.runtime.worktrees import Worktree
from pydantic_clai2.ui.menus.key_menu import keys_command
from pydantic_clai2.ui.menus.model_picker import MODEL_SUBCOMMANDS, model_command, model_completions
from pydantic_clai2.ui.menus.plugin_menu import open_plugins_menu
from pydantic_clai2.ui.menus.rewind import rewind
from pydantic_clai2.ui.menus.set_menu import set_command
from pydantic_clai2.ui.menus.spinner_picker import spinner_command, spinner_completions
from pydantic_clai2.ui.menus.task_menu import open_tasks
from pydantic_clai2.ui.menus.theme_picker import theme_command
from pydantic_clai2.ui.prompt._completion_adapter import COMPLETION_STYLE, PromptCompleter
from pydantic_clai2.ui.prompt.image_input import ImageInput
from pydantic_clai2.ui.prompt.input_history import input_history
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt, PromptRewind, PromptWakeup
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.prompt.screen import Screen
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering._branding import print_banner
from pydantic_clai2.ui.rendering._rendering import StreamRenderer
from pydantic_clai2.ui.rendering.history import render_history
from pydantic_clai2.ui.rendering.spinners import Spinner, Spinners
from pydantic_clai2.ui.rendering.status import Status, StatusLine
from pydantic_clai2.ui.rendering.tool_output import terminal_text
from pydantic_clai2.ui.rendering.usage_report import cost_line, session_usage

if TYPE_CHECKING:
    from pydantic_clai2.auth import CodexAuth
    from pydantic_clai2.models.accounts import Account

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
_PLUGIN_ACTIONS = ('list', 'add', 'enable', 'disable', 'remove', 'reload', 'configure')
_PLUGINS_OFF = 'Plugins are off for this session; saved plugin settings are unchanged.'


DEFAULT_PLUGINS: tuple[PluginSettings, ...] = (
    PluginSettings(
        id='coder',
        factory='pydantic_clai2.builtin_plugins.coder',
        settings={
            'unrestricted_filesystem': True,
            'repo_context': False,
            'sub_agents': False,
            'agent_folders': ['agents'],
        },
    ),
    PluginSettings(id='ask_user', factory='pydantic_clai2.builtin_plugins.ask_user_menu'),
    PluginSettings(id='repo_context', factory='pydantic_clai2.builtin_plugins.repo_context'),
    PluginSettings(id='compaction', factory='pydantic_clai2.builtin_plugins.compaction', settings={}),
    PluginSettings(id='persistence', factory='pydantic_clai2.runtime.sessions'),
    PluginSettings(id='observability', factory='pydantic_clai2.builtin_plugins.logfire'),
    PluginSettings(id='notifications', factory='pydantic_clai2.builtin_plugins.notifications'),
    PluginSettings(id='mcp', factory='pydantic_clai2.mcp'),
    PluginSettings(id='github', factory='pydantic_clai2.builtin_plugins.github', enabled=False),
    PluginSettings(id='pylon', factory='pydantic_clai2.builtin_plugins.pylon', enabled=False),
    PluginSettings(id='google_workspace', factory='pydantic_clai2.builtin_plugins.google_workspace', enabled=False),
    PluginSettings(id='day_ai', factory='pydantic_clai2.builtin_plugins.day_ai', enabled=False),
    PluginSettings(id='ordinal', factory='pydantic_clai2.builtin_plugins.ordinal', enabled=False),
    PluginSettings(id='notion', factory='pydantic_clai2.builtin_plugins.notion', enabled=False),
    PluginSettings(id='slack', factory='pydantic_clai2.builtin_plugins.slack', enabled=False),
    PluginSettings(id='logfire_mcp', factory='pydantic_clai2.builtin_plugins.logfire_mcp', enabled=False),
    PluginSettings(id='posthog', factory='pydantic_clai2.builtin_plugins.posthog', enabled=False),
    PluginSettings(id='grain', factory='pydantic_clai2.builtin_plugins.grain', enabled=False),
    PluginSettings(id='linear', factory='pydantic_clai2.builtin_plugins.linear', enabled=False),
    PluginSettings(id='herdr', factory='pydantic_clai2.builtin_plugins.herdr', enabled=False),
)
"""Built-in declarations, each integrated with the shell. `remove` restores their defaults.

Other harness capabilities are not listed here: a user adds one on purpose with `/plugins add` or a plugin module.

`coder` leaves out its own `RepoContext` because `repo_context` binds one, so instruction files load once.
`compaction` runs alongside `coder`, whose `ClearToolResults` only empties old tool results; see `plugins.compatibility`.
"""

STOCK_PLUGINS: tuple[PluginSettings, ...] = tuple(
    plugin.model_copy(update={'settings': {**plugin.settings, 'sub_agents': True}}) if plugin.id == 'coder' else plugin
    for plugin in DEFAULT_PLUGINS
)
"""CLI-owned agents bind plugins at construction, so Coder can delegate safely.

`DEFAULT_PLUGINS` keeps its run-level delegation opt-out for caller-supplied agents.
"""


def create_agent(model: str | None = None) -> Agent[None, str]:
    """Build the base CLAI agent. The coding tools come from the built-in `coder` plugin, not from here."""
    return Agent(model, deps_type=type(None), capabilities=[customization_guide()])


def create_stock_agent(model: Model | str | None = None) -> StockAgent[None, str]:
    """Build the CLI-owned template; caller-supplied agents are never reconstructed."""
    return StockAgent(model, deps_type=type(None), output_type=str, capabilities=[customization_guide()])


@asynccontextmanager
async def open_stock_agent(
    *,
    workspace: str | Path,
    model: Model | str | None = None,
    capabilities: Sequence[AgentCapability[None]] = (),
    plugin_settings: Mapping[str, Mapping[str, JsonValue]] | None = None,
) -> AsyncGenerator[Agent[None, str]]:
    """Open CLAI's stock coding agent for runs from code, without the terminal.

    The agent has the `coder`, `repo_context`, and `compaction` built-ins as the CLI configures them,
    then `capabilities`, all bound at construction so delegated tasks carry them too. It works in
    `workspace` on this machine unless one of `capabilities` supplies a workspace, such as a sandbox.
    Commands get this process's environment minus LLM provider API keys. Model names, the agent's own
    and any a run passes, resolve as in the CLI and get CLAI's per-model defaults. Without `model`,
    each run must pass one.

    `plugin_settings` maps a built-in's id to settings merged over its stock ones, such as
    `{'coder': {'sub_agents': False}}`. Nothing saved for the `clai2` CLI applies: no saved or drop-in
    plugins, no project `.clai/settings.json`, no saved model settings or `chain:` fallback chains.
    Raises `UserError` when `plugin_settings` names another plugin, and `PluginSettingsError` when a
    built-in rejects its merged settings.

    The agent and its plugins close when the context exits.
    """
    from pydantic_clai2.models.model_settings import default_model_settings

    stock = {plugin.id: plugin for plugin in STOCK_PLUGINS if plugin.id in ('coder', 'repo_context', 'compaction')}
    overrides = plugin_settings or {}
    if unknown := sorted(overrides.keys() - stock.keys()):
        raise UserError(
            f'`plugin_settings` configures only {", ".join(map(repr, stock))}, not {", ".join(map(repr, unknown))}.'
        )
    declarations = [
        plugin.model_copy(update={'settings': {**plugin.settings, **overrides.get(plugin.id, {})}})
        for plugin in stock.values()
    ]
    # Unlike `create_stock_agent`, no guide to customizing the terminal app, whose plugins never load here.
    template = StockAgent(
        model if isinstance(model, Model) else None, deps_type=type(None), output_type=str, capabilities=[]
    )
    reason: SessionEndReason = 'error'
    # A private settings store keeps the user's saved and drop-in plugins out; plugin output goes nowhere.
    with TemporaryDirectory(prefix='clai2-') as config, open(os.devnull, 'w', encoding='utf-8') as sink:
        console = Console(file=sink, force_terminal=False)
        store = SettingsStore(Path(config) / 'config.db')
        loader = PluginLoader[None](
            store=store,
            console=console,
            commands=Commands(),
            session_start=lambda: SessionStart(
                agent=template, settings=Settings(model=model if isinstance(model, str) else None)
            ),
            builtin=declarations,
        )
        models = _ModelResolver(console=console, store=store, plugins=loader.model_providers)
        try:
            for declaration in declarations:
                await loader.load(declaration.id)
            bound: list[AgentCapability[None]] = [*loader.run_capabilities(), *capabilities]
            if (local := local_workspace(bound, workspace)) is not None:
                bound.append(local)
            agent = template.with_plugins(
                [
                    *bound,
                    SessionModels[None](lambda _, name: resolve_model_name(models.resolve, name)),
                    ModelDefaults[None](lambda name: default_model_settings(model=name, saved={})),
                ],
                model=model if isinstance(model, str) else None,
            )
            async with agent:
                yield agent
            reason = 'exit'
        finally:
            with CancelScope(shield=True):
                await loader.close(reason)


async def chat(
    agent: AbstractAgent[DepsT, OutputT],
    *,
    deps: DepsT,
    plugins: Sequence[AgentCapability[DepsT]] = (),
    usage_limits: UsageLimits | None = None,
    console: Console | None = None,
    settings: Settings | None = None,
    store: SettingsStore | None = None,
    builtin_plugins: Sequence[PluginSettings] = (),
    project: ProjectSettings | None = None,
    resume: str | None = None,
    resume_from: ImportSource | None = None,
    load_plugins: bool = True,
    worktree: Worktree | None = None,
) -> None:
    """Start an asyncio terminal conversation with a caller-supplied agent.

    Esc cancels the current turn; Ctrl-C also clears idle input. Ctrl-D and `/exit` quit.
    Failed and cancelled turns retain their captured history. Resume never replays tools.
    `resume_from` imports `resume` from Claude Code or Codex instead; an empty `resume` browses its sessions.
    `project` is the parsed `.clai/settings.json`; layer its overrides into `settings` yourself.
    `load_plugins=False` loads no built-in, project, saved, or drop-in plugin and turns `/plugins` off for this
    session only; saved plugin preferences are untouched.
    `worktree` is the checkout `--worktree` opened; its path and branch are shown under the launch banner.
    """
    console = console or Console()
    rebuild_stock = agent.with_plugins if isinstance(agent, StockAgent) else None
    transcript = TranscriptBuffer()
    with theme.use(lambda: settings.theme if settings is not None else 'default'), transcript.capture(console):
        project = project or ProjectSettings()
        _print_welcome(project, console, worktree=worktree)
        use_defaults = builtin_plugins is DEFAULT_PLUGINS
        use_stock_defaults = builtin_plugins is STOCK_PLUGINS
        shell = create_shell(
            agent,
            deps=deps,
            plugins=plugins,
            usage_limits=usage_limits,
            console=console,
            settings=settings,
            store=store,
            builtin_plugins=builtin_plugins,
            project=project,
            transcript=transcript,
            load_plugins=load_plugins,
        )
    fresh = False
    warming: Thread | None = None
    async with agent:
        while True:
            reason: SessionEndReason = 'error'
            with theme.use(lambda: shell.context.settings.theme, output=console.file if console.is_terminal else None):
                try:
                    async with create_task_group() as workers:
                        workers.start_soon(shell.sessions.namer.run)
                        try:
                            with (
                                transcript.capture(console),
                                shell.defer_identity() if resume is not None else nullcontext(),
                            ):
                                await shell.loader.load_all(fresh=fresh)
                                _report_project_plugins(shell.loader, console)
                                if resume is not None:
                                    source = [resume_from] if resume_from else []
                                    console.print(
                                        await shell.sessions.command([*source, resume] if resume else source),
                                        markup=False,
                                    )
                                    resume = None
                            warming = warming or warm_imports.start()
                            reason = await shell.run()
                        finally:
                            workers.cancel_scope.cancel()
                except BaseExceptionGroup as exc:
                    if len(exc.exceptions) == 1:
                        raise exc.exceptions[0] from None
                    raise
                finally:
                    with transcript.capture(console):
                        await shell.loader.close(reason)
            if not shell.reload_requested:
                if (executable := shell.updates.relaunch) is not None:
                    summary = shell.session.summary
                    raise Relaunch(executable=executable, session_id=summary.id if summary.revision else None)
                return
            shell.reload_requested = False
            if warming is not None:  # pragma: no branch -- a reload follows a run, which started warming
                warming.join()
            try:
                shell = reload_clai(
                    lambda shell=shell: create_shell(
                        rebuild_stock(()) if rebuild_stock is not None else agent,
                        deps=deps,
                        plugins=plugins,
                        usage_limits=shell.session.usage_limits,
                        console=console,
                        settings=shell.context.settings,
                        store=SettingsStore(shell.context.store.path),
                        builtin_plugins=(
                            STOCK_PLUGINS
                            if use_stock_defaults
                            else DEFAULT_PLUGINS
                            if use_defaults
                            else builtin_plugins
                        ),
                        project=project,
                        message_history=shell.session.messages,
                        summary=shell.session.summary,
                        transcript=shell.transcript,
                        load_plugins=load_plugins,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- development edits must not discard the conversation.
                with transcript.capture(console):
                    console.print(
                        f'Reload failed: {type(exc).__name__}: {exc}', style=theme.color(theme.ERROR), markup=False
                    )
                    if isinstance(exc, ImportError) and exc.name and exc.name.startswith('pydantic_ai_harness'):
                        console.print(
                            'Harness is not refreshed by /reload. Restart CLAI2 with the same launch options and '
                            '--resume to continue this session. Keep the worktree if asked to remove it.',
                            style=theme.color(theme.INFO),
                            markup=False,
                        )
                fresh = False
            else:
                with transcript.capture(console):
                    console.print('CLAI2 reloaded. Conversation preserved.', style=theme.color(theme.INFO))
                fresh = True


def _parse(name: str) -> ModelRef:
    """Split a model name, reporting a malformed profile as a `UserError` that names the model."""
    try:
        return parse_model(name)
    except ValueError as exc:
        raise UserError(f'{name}: {exc}') from None


@dataclass(kw_only=True)
class _ModelResolver:
    """Load provider integrations on demand, retaining Codex authentication per conversation."""

    console: Console
    plugins: Callable[[], Mapping[str, ModelProvider]] = lambda: {}
    """Model prefixes registered by loaded plugins, read per resolution so enabling one applies at once."""
    logins: Callable[[], Mapping[str, PluginLogin]] = lambda: {}
    """Sign-ins registered by loaded plugins, read per `/login` like `plugins`."""
    store: SettingsStore | None = None
    """Where a plugin sign-in saves its models."""
    pool_accounts: Callable[[], bool] = lambda: False
    """The `accounts.pool` setting, read per resolution so `/set` applies to the next run."""
    _auth: 'CodexAuth | None' = None

    def codex_auth(self) -> 'CodexAuth':
        if self._auth is None:
            from pydantic_clai2.auth import CodexAuth

            self._auth = CodexAuth(self.console)
        return self._auth

    async def login(self, args: list[str]) -> str:
        from pydantic_clai2.auth import login_command

        return await login_command(args, codex=self.codex_auth(), plugins=self.logins(), store=self.store)

    async def resolve(self, name: str) -> Model | str:
        """Build `PROVIDER[@PROFILE]:NAME` or `chain:NAME`; a core model without a profile stays a string.

        While `accounts.pool` is on, a name without a profile runs on every signed-in account, as `@*`
        does, once its provider has two or more; `@default` runs the default account alone.
        """
        from pydantic_clai2.models.chains import chain_name

        if (chain := chain_name(name)) is not None:
            return await self._chain(chain)
        ref = _parse(name)
        if ref.profile == ALL:
            return await self._all_accounts(ref.provider, ref.name)
        if ref.profile == DEFAULT:
            return await self._one_account(base_model(name))
        if ref.profile is None and ref.provider and self.pool_accounts():
            members = await self._pool(ref.provider)
            if len(members) > 1:
                return await self._fall_back(members, ref.name)
        return await self._one_account(name)

    async def _one_account(self, name: str) -> Model | str:
        """Build a model on the one account it names, the default one when it names none."""
        ref = _parse(name)
        if ref.provider in ('openrouter', 'vllm', 'github-copilot'):
            from pydantic_clai2.models import github_copilot, openrouter, vllm

            build = {'openrouter': openrouter.model, 'vllm': vllm.model, 'github-copilot': github_copilot.model}
            return await to_thread.run_sync(build[ref.provider], name, abandon_on_cancel=True)
        if ref.provider == 'openai-codex':
            return self.codex_auth().model(name)
        if (provider := self.plugins().get(ref.provider)) is not None:
            if ref.profile is None:
                return await to_thread.run_sync(provider.resolve, ref.name, abandon_on_cancel=True)
            if provider.resolve_profile is None:
                raise UserError(f'{ref.provider} does not support profiles. Use {ref.provider}:{ref.name}.')
            resolve = partial(provider.resolve_profile, ref.name, ref.profile)
            return await to_thread.run_sync(resolve, abandon_on_cancel=True)
        if ref.profile is not None:
            from pydantic_clai2.models import key_profiles

            return await to_thread.run_sync(key_profiles.model, name, abandon_on_cancel=True)
        return name

    async def _chain(self, chain: str) -> Model:
        """Resolve every model of a saved chain and fall back through them in order."""
        from pydantic_ai.models.fallback import FallbackModel

        models = self.store.chains().get(chain) if self.store is not None else None
        if not models:
            raise UserError(f'No chain named {chain}. Create one with /model chains.')
        first, *rest = [await self.resolve(model) for model in models]
        return FallbackModel(first, *rest)

    async def _all_accounts(self, provider: str, name: str) -> Model | str:
        """`PROVIDER@*:NAME`: the model on every signed-in account, in `/accounts` order, falling back in turn."""
        members = await self._pool(provider)
        if not members:
            raise UserError(f'No {provider} account is signed in. Add one with /accounts.')
        return await self._fall_back(members, name)

    async def _pool(self, provider: str) -> 'list[Account]':
        """The provider's signed-in accounts, in `/accounts` order; none without a settings store."""
        from pydantic_clai2.models.accounts import pool

        store = self.store
        return await to_thread.run_sync(lambda: pool(store, provider), abandon_on_cancel=True) if store else []

    async def _fall_back(self, members: 'list[Account]', name: str) -> Model | str:
        """NAME on each account in turn, the next one taking over when a request fails."""
        from pydantic_ai.models.fallback import FallbackModel

        first, *rest = [await self._one_account(member.model(name)) for member in members]
        return FallbackModel(first, *rest) if rest else first

    async def accounts(self, args: list[str]) -> str:
        """`/accounts`: list, add, rename, reorder, and sign out of accounts."""
        from pydantic_clai2.models.usage import usage_fetcher
        from pydantic_clai2.ui.menus.accounts_menu import open_accounts_menu

        if args:
            raise ValueError('Usage: /accounts (opens the menu)')
        if self.store is None:  # pragma: no cover -- the shell always has a store.
            raise ValueError('Accounts need a settings database.')
        auth = self.codex_auth()
        return await open_accounts_menu(
            self.store,
            login=self.login,
            plugins=self.logins,
            forget=auth.forget,
            usage=lambda item: usage_fetcher(item, codex=auth.account_provider, plugins=self.logins()),
        )


def create_shell(
    agent: AbstractAgent[DepsT, OutputT],
    *,
    deps: DepsT,
    plugins: Sequence[AgentCapability[DepsT]],
    usage_limits: UsageLimits | None,
    console: Console,
    settings: Settings | None,
    store: SettingsStore | None,
    builtin_plugins: Sequence[PluginSettings],
    project: ProjectSettings,
    message_history: Sequence[ModelMessage] = (),
    summary: ConversationSummary | None = None,
    transcript: TranscriptBuffer | None = None,
    headless: bool = False,
    load_plugins: bool = True,
) -> '_Shell[DepsT, OutputT]':
    """Build shared session services, without attaching terminal input in headless mode."""
    settings = (
        Settings.model_validate(settings.model_dump(exclude_unset=True))
        if settings is not None
        else Settings(model=None)
    )
    store = store or SettingsStore()
    transcript = transcript if transcript is not None else TranscriptBuffer()
    conversations = SqliteConversationStore(database=store.path.with_name('sessions.db'))
    session = Session(
        agent,
        deps=deps,
        plugins=plugins,
        usage_limits=usage_limits,
        message_history=message_history,
        conversations=conversations,
    )
    if summary is not None:
        session.summary = summary
    session.model = settings.model
    session.model_chosen = 'model' in settings.model_fields_set
    session.tool_retries = settings.tool_retries
    models = _ModelResolver(console=console, store=store)
    session.resolve_model = models.resolve
    if session.model is None and agent.model is None:
        console.print('Add a model with /model add.', style=theme.color(theme.INFO))

    session_settings = SessionSettings(session=session, console=console, settings=settings)

    context = CommandContext(
        settings=settings, store=store, clear_history=session.clear, apply_setting=session_settings, project=project
    )
    models.pool_accounts = lambda: context.settings.pool_accounts

    def fast(args: list[str]) -> str:
        if args not in ([], ['on'], ['off']):
            raise ValueError('Usage: /fast [on|off]')
        model = session.model or _model_label(agent)
        current = context.model_settings(model) or {}
        saved = store.model_settings(model)
        custom = saved.get('custom_params')
        if isinstance(custom, dict) and any(key.partition('.')[0] == 'service_tier' for key in custom):
            raise ValueError('Custom service_tier overrides fast mode. Remove it with /model settings first.')
        enabled = args == ['on'] or (not args and current.get('service_tier') != 'priority')
        tier = 'priority' if enabled else 'default'
        store.save_model_settings(model, {**saved, 'service_tier': tier})
        return (
            f'Fast mode {"on" if enabled else "off"} for {model} (service_tier={tier}). Applies on the next prompt.'
            + (' Uses more ChatGPT credits; availability depends on your model and account.' if enabled else '')
        )

    sessions = Sessions(session=session, store=conversations, context=context)
    commands = Commands()
    commands.register(
        Command(
            name='effort',
            description='View or set reasoning effort: /effort [VALUE|reset]',
            handler=lambda args: effort_command(context, args, model=session.model or _model_label(agent)),
            complete=lambda args: effort_completions(context, args, model=session.model or _model_label(agent)),
        )
    )
    commands.register(
        Command(
            name='fast',
            description='Toggle Codex priority processing: /fast [on|off] (uses more ChatGPT credits)',
            handler=fast,
            complete=lambda args: ('on', 'off') if len(args) <= 1 else (),
            available=lambda: (
                provider_of(context.settings_model(session.model or _model_label(agent))) == 'openai-codex'
            ),
        )
    )
    commands.register(
        Command(
            name='resume',
            description='Browse or restore a saved session; claude or codex imports theirs',
            handler=sessions.command,
            complete=lambda args: IMPORT_SOURCES if len(args) <= 1 else (),
            during_turn=True,
        )
    )
    commands.register(Command(name='keys', description='Manage saved API keys', handler=keys_command, during_turn=True))
    commands.register(
        Command(
            name='login',
            description=(
                'Sign in: openai-codex, github-copilot, or one a plugin adds; NAME@PROFILE adds another account'
            ),
            handler=models.login,
            complete=lambda args: login_names(models.logins()) if len(args) <= 1 else (),
            during_turn=True,
        )
    )
    commands.register(
        Command(
            name='accounts',
            description='Add, rename, reorder, and sign out of accounts; MODEL@* tries them all in order',
            handler=models.accounts,
            during_turn=True,
        )
    )
    set_ = Command(
        name='set',
        description='Change settings; no arguments opens the menu',
        handler=lambda args: set_command(context, args),
        complete=lambda args: set_completions(args, plugin_models=context.plugin_models()),
        during_turn=True,
    )
    commands.register(set_)
    commands.register(replace(set_, name='settings', description='Alias of /set'))
    commands.register(
        Command(
            name='theme',
            description='Select a Termflow palette; no arguments opens the picker',
            handler=lambda args: theme_command(context, args),
            complete=lambda args: theme.names() if len(args) <= 1 else (),
            during_turn=True,
        )
    )
    commands.register(
        Command(
            name='model',
            description=(
                'Select any model or fallback chain, or open the picker; also /model add [NAME], '
                '/model settings [NAME], and /model chains'
            ),
            handler=lambda args: model_command(context, args),
            complete=lambda args: model_completions(context, args),
            during_turn=True,
            during_turn_subcommands=MODEL_SUBCOMMANDS,
        )
    )
    # Deprecated spellings of `/model add`, `/model settings`, and `/model chains`, kept working for existing habits.
    commands.register(
        Command(
            name='add_model',
            description='Deprecated: use /model add',
            handler=lambda args: model_command(context, ['add', *args]),
            complete=lambda args: model_completions(context, ['add', *args]),
            during_turn=True,
        )
    )
    commands.register(
        Command(
            name='model_settings',
            description='Deprecated: use /model settings',
            handler=lambda args: model_command(context, ['settings', *args]),
            complete=lambda args: model_completions(context, ['settings', *args]),
            during_turn=True,
        )
    )
    commands.register(
        Command(
            name='chain',
            description='Alias of /model chains: fallback chains live in the /model picker',
            handler=lambda args: model_command(context, ['chains', *args]),
            during_turn=True,
        )
    )
    commands.register(
        Command(name='help', description='Show commands', handler=lambda args: f'{commands.help(args)}\n{SHELL_HELP}')
    )

    def reset_screen() -> None:
        console.clear()
        # Forget the old output too, or a resize or the exit printout would show it again.
        transcript.clear()
        _print_welcome(project, console)

    def clear(_: list[str]) -> str:
        session.clear()
        reset_screen()
        return ''

    clear_ = Command(
        name='clear',
        description='Start a new session on a clear screen; the previous session stays saved',
        handler=clear,
    )
    commands.register(clear_)
    commands.register(replace(clear_, name='new', description='Alias of /clear'))
    commands.register(
        Command(
            name='usage',
            description='Show tokens and cost per turn',
            handler=lambda _: sessions.usage(console=console),
        )
    )
    commands.register(
        Command(
            name='cost',
            description='Show retained history cost and tokens',
            handler=lambda _: cost_line(session_usage(session.messages)),
        )
    )
    commands.register(Command(name='exit', description='Quit CLAI', handler=lambda _: 'Goodbye.'))
    updates = Updates(channel=lambda: context.settings.update_channel)
    commands.register(
        Command(
            name='update',
            description='Install the newest CLAI from the updates.channel setting (stable or main)',
            handler=updates.command,
        )
    )
    commands.register(
        Command(
            name='config',
            description='show|get|set|reset settings',
            handler=lambda args: config_command(store, args),
            complete=config_completions,
        )
    )
    screen = Screen()
    status = Status()
    loader: PluginLoader[DepsT] = PluginLoader(
        store=store,
        console=console,
        commands=commands,
        session_start=lambda: SessionStart(agent=agent, settings=context.settings),
        builtin=tuple(PluginSettings.model_validate(plugin.model_dump()) for plugin in builtin_plugins),
        full_screen=screen.full,
        project=tuple(PluginSettings.model_validate(plugin.model_dump()) for plugin in project.plugins),
        conversation=session,
        session_id=lambda: shell.session_id,
        status=status,
        enabled=load_plugins,
    )
    models.plugins = loader.model_providers
    models.logins = loader.logins
    context.plugin_models = loader.model_names
    context.settings_model = lambda model: loader.settings_model(chain_settings_model(store, model))
    spinners = Spinners(selected=lambda: context.settings.spinner, registered=loader.spinners)
    commands.register(
        Command(
            name='spinner',
            description='Select the working animation; no arguments opens the picker',
            handler=lambda args: spinner_command(context, spinners, args),
            complete=lambda args: spinner_completions(spinners, args),
            during_turn=True,
        )
    )
    commands.register(
        Command(
            name='plugins',
            description='Manage plugins; no arguments opens the menu',
            handler=lambda args: (
                _PLUGINS_OFF if not loader.enabled else loader.command(args) if args else open_plugins_menu(loader)
            ),
            complete=lambda args: _PLUGIN_ACTIONS if len(args) <= 1 else (entry.name for entry in loader.entries()),
            during_turn=True,
            args_during_turn=True,
        )
    )
    for plugin in plugins:
        if isinstance(plugin, CommandProvider):
            commands.register_many(plugin.get_commands(context))
    images = ImageInput()
    history = input_history(store.path.with_name('input-history'))
    prompt = None
    if not headless and not console.is_terminal:
        prompt = PromptSession[str](
            history=history,
            completer=PromptCompleter(commands),
            complete_while_typing=True,
            style=COMPLETION_STYLE,
            reserve_space_for_menu=6,
            bottom_toolbar=lambda: FormattedText(
                [
                    (theme.color(style), text)
                    for style, text in ([(theme.MUTED, images.notice)] if images.notice else status.toolbar())
                ]
            ),
        )
    shell = _Shell(
        agent=agent,
        session=session,
        plugins=tuple(plugins),
        loader=loader,
        commands=commands,
        console=console,
        context=context,
        status=status,
        prompt=prompt,
        history=history,
        transcript=transcript,
        images=images,
        interrupts=Interrupts(),
        screen=screen,
        sessions=sessions,
        speculation=Speculation(context=context, console=console),
        session_settings=session_settings,
        spinners=spinners,
        updates=updates,
    )
    if prompt is not None:
        prompt.key_bindings = images.bindings()
    commands.register(
        Command(name='reload', description='Reload CLAI2 code without restarting', handler=shell.request_reload)
    )
    commands.register(
        Command(
            name='fork',
            description='Run a copy of this conversation in the background: /fork [@model] PROMPT',
            handler=shell.forks.fork_command,
            complete=shell.forks.complete,
            raw=True,
        )
    )
    commands.register(
        Command(
            name='tasks',
            description='Inspect and control delegated tasks',
            handler=shell.tasks_command,
            during_turn=True,
        )
    )
    commands.register(Command(name='forks', description='Show background forks', handler=shell.forks.status_command))

    async def show_resumed(messages: Sequence[ModelMessage]) -> None:
        # The live panel swaps to the restored conversation; startup output above it stays.
        if shell.editor is not None:
            reset_screen()
        renderer = _stream_renderer(console, settings=context.settings, renderers=loader.renderers(), smooth=False)
        await render_history(messages, console=console, renderer=renderer)

    sessions.on_resume = show_resumed
    # Mutate retained state only after the rebuild has succeeded, so reload failures can roll back.
    TranscriptBuffer.rebind(transcript)
    return shell


@dataclass(kw_only=True)
class _Shell(Generic[DepsT, OutputT]):
    """The prompt loop; one turn is one `TurnStart`, one agent run, one `TurnEnd`."""

    agent: AbstractAgent[DepsT, OutputT]
    session: Session[DepsT, OutputT]
    plugins: tuple[AgentCapability[DepsT], ...]
    loader: PluginLoader[DepsT]
    commands: Commands
    console: Console
    context: CommandContext
    status: Status
    prompt: PromptSession[str] | None
    history: History
    interrupts: Interrupts
    screen: Screen
    sessions: Sessions[DepsT, OutputT]
    speculation: Speculation
    session_settings: SessionSettings[DepsT, OutputT]
    spinners: Spinners
    updates: Updates
    transcript: TranscriptBuffer = field(default_factory=TranscriptBuffer)
    images: ImageInput = field(default_factory=ImageInput)
    reload_requested: bool = False
    editor: LivePrompt | None = None
    forks: Forks[DepsT, OutputT] = field(init=False)
    tasks: Tasks = field(init=False)
    _mid_turn_commands: MemoryObjectSendStream[str] | None = field(default=None, init=False, repr=False)
    _steering: MemoryObjectSendStream[str] | None = field(default=None, init=False, repr=False)
    _identity_pending: bool = field(default=False, init=False)

    @property
    def session_id(self) -> str | None:
        """The active saved ID, unavailable while startup is choosing a conversation to resume."""
        return None if self._identity_pending else current_session_id() or self.session.summary.id

    @contextmanager
    def defer_identity(self) -> Generator[None]:
        """Keep startup telemetry unassigned until the requested conversation is selected."""
        self._identity_pending = True
        try:
            yield
        finally:
            self._identity_pending = False

    def __post_init__(self) -> None:
        self.tasks = Tasks(
            console=self.console,
            conversation_id=lambda: self.session.summary.id,
            directory=None
            if str(self.sessions.store.database) == ':memory:'
            else Path(str(self.sessions.store.database) + '.tasks'),
            step_store=self.session.step_store,
        )
        self.status.subagent = self.tasks.focused
        self.forks = Forks(
            console=self.console,
            history=lambda: self.session.messages,
            spawn=self.fork_session,
            fire=self.loader.fire,
            models=self.context.store.models,
        )
        self.session.on_setup_error = self.capability_failed

    def run_plugins(self) -> tuple[AgentCapability[DepsT], ...]:
        """Capabilities bound to the next run: supplied, plugin-registered (guarded), then speculation.

        Speculation sees the others unguarded, so its sandbox mount stays within their `FileSystem`.
        """
        granted = (*self.plugins, *self.loader.capabilities())
        bound = (*self.plugins, *self.loader.run_capabilities())
        if self.session.delegations is not None:
            granted = (*granted, self.tasks.presentation)
            bound = (*bound, self.tasks.presentation)
        return (*bound, *self.speculation.capabilities(granted))

    def capability_failed(self, error: CapabilitySetupError) -> None:
        """Report that a failing settings-built capability is left out of later turns."""
        self.console.print(self.loader.suspend(error), style=theme.color(theme.WARNING), markup=False)

    def fork_session(self, model: str | None, history: Sequence[ModelMessage]) -> Session[DepsT, OutputT]:
        """A separately saved session configured like the foreground one, seeded with `history`."""
        child = Session(
            self.agent,
            deps=self.session.deps,
            plugins=self.run_plugins(),
            message_history=history,
            usage_limits=self.session.usage_limits,
            conversations=self.session.conversations,
            workspace=Path(self.session.workspace),
        )
        child.model = model or self.session.model
        child.model_chosen = model is not None or self.session.model_chosen
        child.tool_retries = self.session.tool_retries
        child.resolve_model = self.session.resolve_model
        child.model_settings = self.context.live_model_overrides(child.model or _model_label(self.agent))
        child.model_defaults = self.context.model_defaults(child.model or _model_label(self.agent))
        child.on_setup_error = self.capability_failed
        return child

    async def tasks_command(self, args: list[str]) -> str:
        if not args:
            return await open_tasks(self.tasks)
        if len(args) != 2 or args[0] not in ('stop', 'background', 'resume'):
            raise ValueError('Usage: /tasks [stop|background|resume ID]')
        record = self.tasks.resolve(args[1])
        if args[0] == 'stop':
            await self.tasks.owner.cancel(record.id)
            await self.tasks.owner.save(record)
            return f'Stopping task {record.id[:8]} and its descendants.'
        if args[0] == 'background':
            self.tasks.owner.background(record.id)
            return f'Task {record.id[:8]} moved to background.'
        await self.tasks.owner.allow_resume(record.id)
        if self.editor is None:
            return f'Task {record.id} may now be resumed with delegate_task(resume={record.id!r}).'
        self.editor.submit(
            f'Resume task {record.id} using delegate_task with agent_name={record.agent_name!r}, '
            f'resume={record.id!r}. Continue from its saved history and report the result.'
        )
        return f'Resume requested for task {record.id[:8]}.'

    def request_reload(self, args: list[str]) -> str:
        if args:
            raise ValueError('Usage: /reload')
        self.reload_requested = True
        return 'Reloading CLAI2...'

    async def run(self) -> SessionEndReason:
        try:
            async with self.tasks.owner.opened():
                if isinstance(self.agent, StockAgent):
                    self.session.delegations = self.tasks.owner
                return await self._run()
        finally:
            self.session.delegations = None
            await self.forks.close()

    async def _run(self) -> SessionEndReason:
        if self.console.is_terminal:
            self.editor = LivePrompt(
                console=self.console,
                commands=self.commands,
                history=self.history,
                images=self.images,
                interrupts=self.interrupts,
                toolbar=self.status.toolbar,
                steer=self.steer,
                run_now=self.run_now,
                transcript=self.transcript,
                chords={'ctrl-x ctrl-s': self.speculation.toggle, 'ctrl-b': self.tasks.promote},
                pinned=self.speculation.row,
                spinner=self.spinners.active,
                panel=lambda glyph: (*self.tasks.rows(glyph), *self.forks.rows(glyph)),
            )
            self.screen.editor = self.editor.suspended
            try:
                async with self.editor.opened():
                    self.tasks.wake = self.editor.wake
                    if self.tasks.owner.reports(conversation_id=self.session.summary.id):
                        self.editor.wake()
                    return await self._read_loop()
            finally:
                self.tasks.wake = None
                self.screen.editor = None
                self.editor = None
        return await self._read_loop()

    def steer(self, text: str) -> bool:
        """Enqueue steering in core and hand transcript feedback to the active turn."""
        try:
            resolved, images = self.images.resolve(text)
        except ValueError as exc:
            self.images.notice = str(exc)
            return True
        if not self.session.steer(resolved, images=images):
            return False
        self.images.notice = ''
        if self._steering is not None:
            self._steering.send_nowait(text)
        else:
            # Direct session users have no active stream renderer to serialize against.
            self.console.print(f'> {terminal_text(text)}', markup=False, highlight=False)
            self.console.print()
        return True

    def _released(self) -> AbstractAsyncContextManager[None]:
        """Hand the terminal to a command or shell, restoring the editor afterwards."""
        return (self.editor.suspended if self.editor is not None else bare_screen)()

    def run_now(self, text: str) -> bool:
        """Open a bare `during_turn` command's menu over a streaming turn instead of queueing it.

        Key handlers call this from an input reader callback, which runs on the event loop but
        outside any task, so it hands the command to the turn's task rather than spawning one.
        """
        if self._mid_turn_commands is None or not self.commands.runs_during_turn(text):
            return False
        self._mid_turn_commands.send_nowait(text)
        return True

    async def _serve_mid_turn(self, commands: MemoryObjectReceiveStream[str]) -> None:
        async with commands, create_task_group() as menus:
            async for text in commands:
                menus.start_soon(self._run_mid_turn, text)

    def plugins_busy(self, text: str) -> bool:
        """Keep plugin-owned transports alive until all children using them settle."""
        if text.split(maxsplit=1)[0] != '/plugins':
            return False
        if not any(record.status == 'running' for record in self.tasks.owner.records.values()):
            return False
        self.console.print(
            'Plugin changes wait for delegated tasks. Stop them in /tasks or wait for completion.', markup=False
        )
        return True

    async def _run_mid_turn(self, text: str) -> None:
        async with self.screen.overlay():
            self.console.print(f'> {terminal_text(text)}', markup=False, highlight=False)
            self.console.print()
            if self.plugins_busy(text):
                return
            # A running conversation cannot be replaced, so the turn's footer counters stay.
            await _execute_command(self.commands, text, console=self.console, status=None)
            self._show_status_segments()

    def _show_status_segments(self) -> None:
        """Paint the status segments of the plugins loaded now."""
        self.status.status_segments = (*self.loader.status_segments(), self.updates.segment)

    async def _rewind(self) -> None:
        assert self.editor is not None
        async with self.forks.busy(), self._released():
            try:
                notice = await rewind(self.session, self.editor)
            except Exception as exc:  # noqa: BLE001 -- a failed save must not end the shell.
                self.console.print(f'Rewind failed: {exc}', style=theme.color(theme.WARNING), markup=False)
            else:
                self.console.print(notice, markup=False)

    async def _read_loop(self) -> SessionEndReason:
        while True:
            self.images.retain(
                [self.editor.buffer.text, *self.editor.queued_messages] if self.editor is not None else []
            )
            try:
                model = self.session.model or _model_label(self.agent)
                if model != self.status.model:
                    self.status.context_window = None
                    self.status.context_alert = False
                self.status.model = model
                self.status.workspace = self.session.workspace
                self._show_status_segments()
                if self.editor is not None:
                    text = await self.editor.read()
                else:
                    assert self.prompt is not None
                    text = expand_bare_command((await self.prompt.prompt_async('> ')).strip())
            except PromptRewind:
                await self._rewind()
                continue
            except PromptWakeup:
                if not self.tasks.owner.reports(conversation_id=self.session.summary.id):
                    continue
                async with self.forks.busy():
                    if await self._turn(None):
                        return 'exit'
                continue
            except KeyboardInterrupt:
                if self.interrupts.press():
                    return 'exit'
                self.console.print(
                    'Input cleared. Press Ctrl-C again within 2 seconds to exit.', style=theme.color(theme.MUTED)
                )
                continue
            except EOFError:
                return 'eof'
            self.images.notice = ''
            if not text:
                continue
            if self.editor is not None:
                self.console.print(f'> {terminal_text(text)}', markup=False, highlight=False)
            self.console.print()
            if await self._dispatch_input(text):
                return 'exit'

    async def _dispatch_input(self, text: str) -> bool:
        if (command := shell_command(text)) is not None:
            async with self.forks.busy(), self._released():
                context = await run_shell_command(command, console=self.console, interrupts=self.interrupts)
                await self.session.commit_messages(
                    [*self.session.messages, ModelRequest(parts=[UserPromptPart(context)])]
                )
            return self.interrupts.exit_requested
        if is_command_input(text):
            return await self._command(text)
        if self.session.model is None and self.agent.model is None:
            self.images.retry_text = text
            self.console.print('Choose a model first: /set model <Tab>', style=theme.color(theme.WARNING))
            return False
        try:
            async with self.forks.busy():
                return await self._turn(text)
        finally:
            if self.editor is not None:
                await self.editor.output.drain()

    async def _command(self, text: str) -> bool:
        if self.plugins_busy(text):
            return False
        async with self.forks.busy(), self._released():
            await self.interrupts.run(_execute_command(self.commands, text, console=self.console, status=self.status))
        return (
            text == '/exit' or self.interrupts.exit_requested or self.reload_requested or self.updates.restart_required
        )

    async def _turn(self, text: str | None) -> bool:
        automated = text is None
        try:
            text, images = self.images.resolve(text) if text is not None else ('', [])
        except ValueError as exc:
            self.console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
            return False
        start = TurnStart(text=text)
        ended: TurnEnd | None = None

        async def run_turn() -> None:
            nonlocal ended
            ended = await self.run_turn(start, images=images, automated=automated)

        previous_tasks = {(record.id, record.generation) for record in self.tasks.records()}
        completed = await self.interrupts.run(run_turn())
        if not completed:
            for record in self.tasks.records():
                if (
                    (record.id, record.generation) not in previous_tasks
                    and not record.background
                    and record.outcome == 'cancelled'
                ):
                    await self.tasks.owner.cancel(record.id)
        self.sessions.namer.submit(self.session.summary.id)
        if self.editor is not None:
            await self.editor.output.drain()
        _report_interrupt(completed, self.console)
        if not completed and (cancelled := self.forks.cancel_running()):
            self.console.print(
                f'Cancelled {cancelled} running fork(s) with the turn.',
                style=theme.color(theme.MUTED),
            )
            self.console.print()
        await self.interrupts.run(self.loader.fire(ended or TurnEnd(text=start.text, outcome='cancelled')))
        return self.interrupts.exit_requested

    async def run_turn(
        self, start: TurnStart, *, images: Sequence[BinaryContent] = (), headless: bool = False, automated: bool = False
    ) -> TurnEnd:
        """Apply turn hooks and settings, then run with optional terminal rendering."""
        try:
            await self.loader.fire(start)
        except PluginError as exc:
            self.console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
            self.console.print()
            return TurnEnd(text=start.text, outcome='failed', error=exc)
        if start.cancelled:
            self.console.print(
                f'Turn cancelled by a plugin: {start.cancel_reason or "no reason given"}',
                style=theme.color(theme.WARNING),
            )
            self.console.print()
            return TurnEnd(text=start.text, outcome='cancelled')
        self.session.plugins = self.run_plugins()
        model = self.session.model or _model_label(self.agent)
        try:
            self.session.model_settings = self.context.live_model_overrides(model)
            self.session.model_defaults = self.context.model_defaults(model)
        except ValidationError as exc:
            self.console.print(
                f'Invalid saved model settings for {model}. Fix or reset them with /model settings {model}.',
                style=theme.ERROR,
                markup=False,
            )
            for error in exc.errors(include_input=False, include_url=False):
                location = '.'.join(str(part) for part in error['loc'])
                self.console.print(f'{location}: {error["msg"]}', style=theme.ERROR, markup=False)
            self.console.print()
            return TurnEnd(text=start.text, outcome='failed', error=exc)
        if headless:
            try:
                result = await self.session.prompt(None if automated else start.text)
            except Exception as exc:  # noqa: BLE001 -- report a failed headless turn to the CLI.
                return TurnEnd(text=start.text, outcome='failed', error=exc)
            return TurnEnd(text=start.text, outcome='completed', result=result)
        # Menus open mid-turn only after this turn has captured its settings and plugins; session
        # changes they save apply once it ends, as does the teardown of plugins they unload, while
        # this model's saved settings reach its next request. A menu still open when it ends delays
        # the next prompt.
        ended = TurnEnd(text=start.text, outcome='cancelled')
        with self.session_settings.turn(), self.speculation.turn():
            send, receive = create_memory_object_stream[str](math.inf)
            steering_send, steering_receive = create_memory_object_stream[str](math.inf)
            async with self.loader.turn(), create_task_group() as mid_turn:
                mid_turn.start_soon(self._serve_mid_turn, receive)
                self._mid_turn_commands = send
                self._steering = steering_send
                try:
                    ended = await _run_prompt(
                        self.session,
                        None if automated else start.text,
                        images=images,
                        console=self.console,
                        settings=self.context.settings,
                        status=self.status,
                        renderers=self.loader.renderers(),
                        screen=self.screen,
                        spinner=self.spinners.active,
                        steering=(steering_send, steering_receive),
                        tasks=self.tasks if self.session.delegations is not None else None,
                    )
                finally:
                    self._mid_turn_commands = None
                    self._steering = None
                    steering_send.close()
                    steering_receive.close()
                    send.close()
        return ended


def _print_welcome(project: ProjectSettings, console: Console, *, worktree: Worktree | None = None) -> None:
    """The banner and hints a fresh launch shows, which `/clear` returns to without the launch's worktree notice."""
    console.print()
    print_banner(console)
    if worktree is not None:
        # Soft wrap keeps the path copyable: hard wrapping would break it with newlines.
        console.print(worktree.notice, style=theme.color(theme.MUTED), markup=False, highlight=False, soft_wrap=True)
    console.print(
        '/new starts a session; /resume restores one; /exit quits. Esc or Ctrl-C interrupts a turn.',
        style=theme.color(theme.MUTED),
    )
    _report_project(project, console)


def _report_project(project: ProjectSettings, console: Console) -> None:
    if project.path is None:
        return
    console.print(f'Project settings: {project.path}', style=theme.color(theme.MUTED))
    if project.unknown:
        console.print(f'Ignoring unknown settings: {", ".join(project.unknown)}', style=theme.color(theme.WARNING))


def _report_project_plugins(loader: PluginLoader[DepsT], console: Console) -> None:
    waiting = [entry.name for entry in loader.entries() if entry.project and entry.loaded is None]
    if waiting:
        console.print(
            f'Project plugins not loaded; approve one with /plugins enable NAME: {", ".join(waiting)}',
            style=theme.color(theme.INFO),
        )


def _report_interrupt(completed: bool, console: Console) -> None:
    if not completed:
        console.print('Turn cancelled. Use /exit to quit.', style=theme.color(theme.MUTED), highlight=False)
        console.print()


async def _execute_command(commands: Commands, text: str, *, console: Console, status: Status | None) -> None:
    try:
        result = await commands.execute_async(text)
        # The echoed command already ends in a blank line; a menu closed without changes adds nothing.
        if not is_silent(result):
            console.print(result, markup=False)
            console.print()
    except Exception as exc:  # noqa: BLE001 -- command failures must not exit the interactive shell.
        console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
        console.print()
    if status is not None:
        _reset_status(text, status)


def _reset_status(command: str, status: Status) -> None:
    if command.split(maxsplit=1)[0] in ('/new', '/clear', '/resume'):
        status.context_tokens = None
        status.context_window = None
        status.context_alert = False
        status.output_tokens = None
        status.cost = None
        status.streamed_chars = 0


def _model_label(agent: AbstractAgent[DepsT, OutputT]) -> str:
    model = agent.model
    if isinstance(model, str):  # pragma: no cover -- concrete Agent resolves string models before chat.
        return model
    return model.model_name if model else 'agent default'


def _stream_renderer(
    console: Console, *, settings: Settings, renderers: Sequence[Renderer], smooth: bool = True
) -> StreamRenderer:
    """The renderer for a turn's events, configured by the display settings."""
    return StreamRenderer(
        console,
        stop_loading=lambda: None,
        show_thinking=settings.thinking,
        smooth_seconds=settings.smooth_seconds,
        show_tool_output=settings.tool_output,
        shell_lines=settings.shell_lines,
        grep_lines=settings.grep_lines,
        tool_arg_chars=settings.tool_arg_chars,
        tool_calls=settings.tool_calls,
        renderers=renderers,
        smooth=smooth,
    )


async def _prompt_with_steering(
    session: Session[DepsT, OutputT],
    text: str | None,
    images: Sequence[BinaryContent],
    steering: tuple[MemoryObjectSendStream[str], MemoryObjectReceiveStream[str]] | None,
    echo: Callable[[str], Awaitable[None]],
) -> AgentRunResult[OutputT]:
    """Own feedback alongside the prompt, draining accepted messages at normal EOF."""
    if steering is None:
        return await session.prompt(text, images=images)
    send, receive = steering

    async def consume() -> None:
        async with receive:
            async for message in receive:
                await echo(message)

    error: Exception | None = None
    result: AgentRunResult[OutputT] | None = None
    async with create_task_group() as feedback:
        feedback.start_soon(consume)
        try:
            result = await session.prompt(text, images=images)
        except Exception as exc:  # noqa: BLE001 -- preserve the prompt error outside the task group.
            error = exc
        finally:
            send.close()
    if error is not None:
        raise error
    assert result is not None
    return result


async def _run_prompt(
    session: Session[DepsT, OutputT],
    text: str | None,
    *,
    console: Console,
    settings: Settings,
    status: Status,
    renderers: Sequence[Renderer],
    screen: Screen,
    spinner: Callable[[], Spinner],
    images: Sequence[BinaryContent] = (),
    tasks: Tasks | None = None,
    steering: tuple[MemoryObjectSendStream[str], MemoryObjectReceiveStream[str]] | None = None,
) -> TurnEnd:
    renderer = _stream_renderer(
        console, settings=settings, renderers=[*renderers, task_row] if tasks is not None else renderers
    )
    status.streamed_chars = 0
    status.output_tokens = None
    status.activity = 'waiting'

    render_lock = Lock()

    async def observe(event: AgentStreamEvent) -> None:
        async with render_lock:
            status.observe(event)
            await renderer.on_stream_event(event)

    async def echo_steering(text: str) -> None:
        async with render_lock:
            await renderer.echo_prompt(text)

    if tasks is not None:
        tasks.sink = observe

    def context_usage(tokens: int) -> None:
        status.context_tokens = tokens

    def context_window(window: int) -> None:
        # The `compaction` gauge, when loaded, already measured this request, honouring its override.
        if status.context_window is None:
            status.context_window = window

    session.on_context_usage = context_usage
    session.on_context_window = context_window
    session.on_stream_event = observe
    status_line = StatusLine(console, status, enabled=screen.editor is None, spinner=spinner)

    @asynccontextmanager
    async def take_screen() -> AsyncGenerator[None]:
        async with render_lock:
            await renderer.finish()
        async with status_line.paused():
            yield

    try:
        with screen.bound(take_screen):
            async with status_line:
                result = await _prompt_with_steering(session, text, images, steering, echo_steering)
                await renderer.finish()
        status.output_tokens = result.usage.output_tokens
        for message in reversed(result.all_messages()):  # pragma: no branch -- successful runs contain a response.
            if isinstance(message, ModelResponse):
                status.context_tokens = message.usage.total_tokens or None
                break
        if not renderer.rendered_text or not isinstance(result.output, str):
            console.print(str(result.output), markup=False)
            console.print()
        return TurnEnd(text=text or '', outcome='completed', result=result)
    except asyncio.CancelledError:
        await renderer.abort()
        raise
    except Exception as exc:  # noqa: BLE001 -- interactive boundary reports plugin/provider failures.
        await renderer.finish()
        console.print(f'{type(exc).__name__}: {error_message(exc)}', style=theme.color(theme.ERROR), markup=False)
        console.print(
            'Turn failed. Retained history may include partial progress. External tool side effects may already have occurred.',
            style=theme.color(theme.MUTED),
        )
        console.print()
        return TurnEnd(text=text or '', outcome='failed', error=exc)
    finally:
        if tasks is not None:
            tasks.sink = None
        status.activity = 'ready'
        status.cost = session_usage(session.messages).total.cost
        session.on_context_usage = None
        session.on_context_window = None
        await renderer.finish()
