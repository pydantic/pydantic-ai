"""The terminal app: `chat`, its stock agent, and `open_stock_agent` for runs from code."""

import os
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from typing import TypeVar

from anyio import CancelScope, create_task_group
from pydantic import JsonValue
from rich.console import Console

from pydantic_ai import Agent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits
from pydantic_clai2 import warm_imports
from pydantic_clai2.cli.self_update import WebHandoff
from pydantic_clai2.commands import (
    Commands,
)
from pydantic_clai2.config import PluginSettings, Settings
from pydantic_clai2.config.builtins import DEFAULT_PLUGINS, STOCK_PLUGINS
from pydantic_clai2.config.project_settings import ProjectSettings, load_project_settings
from pydantic_clai2.config.settings_store import SettingsStore
from pydantic_clai2.customization import customization_guide
from pydantic_clai2.models.resolver import ModelResolver
from pydantic_clai2.plugins import (
    SessionEndReason,
    SessionStart,
)
from pydantic_clai2.plugins.loader import SessionPlugins
from pydantic_clai2.runtime.imported_sessions import ImportSource
from pydantic_clai2.runtime.launch import launch_session_id
from pydantic_clai2.runtime.reloading import reload_clai
from pydantic_clai2.runtime.session import (
    ModelDefaults,
    SessionModels,
    StockAgent,
    local_workspace,
    resolve_model_name,
)
from pydantic_clai2.runtime.session_runtime import SessionRuntime, create_shell
from pydantic_clai2.runtime.worktrees import Worktree
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.welcome import print_welcome
from pydantic_clai2.ui.stderr_relay import relay_stderr
from pydantic_clai2.ui.telemetry_log import telemetry_log

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


def create_agent(model: str | None = None) -> Agent[None, str]:
    """Build the base CLAI agent. The coding tools come from the built-in `coder` plugin, not from here."""
    return Agent(model, deps_type=type(None), capabilities=[customization_guide()])


def create_stock_agent(model: Model | str | None = None) -> StockAgent[None, str]:
    """Build the CLI-owned template; caller-supplied agents are never reconstructed.

    It is named `clai2`, which its run spans carry and which Logfire Agent Control addresses it by.
    """
    return StockAgent(model, deps_type=type(None), output_type=str, capabilities=[customization_guide()], name='clai2')


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
    `{'coder': {'sub_agents': False}}`. Unlike in the CLI, `coder`'s `task_control` is off: pass
    `{'coder': {'task_control': True}}` to give the model the task tools and background delegation.
    Nothing saved for the `clai2` CLI applies: no saved or drop-in
    plugins, no project `.clai/settings.json`, no saved model settings or `chain:` fallback chains.
    Raises `UserError` when `plugin_settings` names another plugin, and `PluginSettingsError` when a
    built-in rejects its merged settings.

    The agent and its plugins close when the context exits.
    """
    from pydantic_clai2.models.model_settings import default_model_settings

    stock = {plugin.id: plugin for plugin in STOCK_PLUGINS if plugin.id in ('coder', 'repo_context', 'compaction')}
    # Task control is opt-in for an application's agent, unlike in the CLI.
    defaults: Mapping[str, Mapping[str, JsonValue]] = {'coder': {'task_control': False}}
    overrides = plugin_settings or {}
    if unknown := sorted(overrides.keys() - stock.keys()):
        raise UserError(
            f'`plugin_settings` configures only {", ".join(map(repr, stock))}, not {", ".join(map(repr, unknown))}.'
        )
    declarations = [
        plugin.model_copy(
            update={'settings': {**plugin.settings, **defaults.get(plugin.id, {}), **overrides.get(plugin.id, {})}}
        )
        for plugin in stock.values()
    ]
    # Unlike `create_stock_agent`, no guide to customizing the terminal app, whose plugins never load here.
    template = StockAgent(
        model if isinstance(model, Model) else None, deps_type=type(None), output_type=str, capabilities=[]
    )
    reason: SessionEndReason = 'error'
    # A private settings store keeps the user's saved and drop-in plugins out; plugin output goes nowhere.
    root = Path(workspace).resolve()
    with TemporaryDirectory(prefix='clai2-') as config, open(os.devnull, 'w', encoding='utf-8') as sink:
        console = Console(file=sink, force_terminal=False)
        store = SettingsStore(Path(config) / 'config.db')
        models = ModelResolver(console=console, store=store)
        model_defaults = ModelDefaults[None](lambda name: default_model_settings(model=name, saved={}))
        loader = SessionPlugins[None](
            store=store,
            console=console,
            commands=Commands(workspace=lambda: root),
            session_start=lambda: SessionStart(
                agent=template, settings=Settings(model=model if isinstance(model, str) else None)
            ),
            builtin=declarations,
            workspace=lambda: root,
            resolve_model=models.resolve,
            model_settings=model_defaults.defaults,
        )
        models.plugins = loader.model_providers
        try:
            for declaration in declarations:
                await loader.load(declaration.id)
            bound: list[AgentCapability[None]] = [*loader.run_capabilities(), *capabilities]
            if (local := local_workspace(bound, root)) is not None:
                bound.append(local)
            agent = template.with_plugins(
                [
                    *bound,
                    SessionModels[None](lambda _, name: resolve_model_name(models.resolve, name)),
                    model_defaults,
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
    session_id: str | None = None,
    fork_session: bool = False,
    worktree: Worktree | None = None,
    web: WebHandoff | None = None,
    settings_for: Callable[[ProjectSettings], Settings] | None = None,
) -> None:
    """Start an asyncio terminal conversation with a caller-supplied agent.

    Esc cancels the current turn; Ctrl-C also clears idle input. Ctrl-D and `/exit` quit.
    Failed and cancelled turns retain their captured history. Resume never replays tools.
    `resume_from` imports `resume` from Claude Code or Codex instead; an empty `resume` browses its sessions.
    `project` is the parsed `.clai/settings.json`; layer its overrides into `settings` yourself.
    `load_plugins=False` loads no built-in, project, saved, or drop-in plugin and turns `/plugins` off for this
    session only; saved plugin preferences are untouched.
    `resume` restores a saved conversation (`''` opens the browser); `fork_session` then continues in a copy.
    `session_id` names the new conversation, or the copy. These follow the CLI's rules: a bad combination,
    or a `session_id` that is not a UUID, raises `ValueError` before anything is drawn.
    `worktree` is the checkout `--worktree` opened; its path and branch are shown under the launch banner.
    `web` is the web UI CLAI served before `/update` restarted it: the `web` plugin serves it again, with no
    browser opening, so its open pages reconnect.
    `settings_for` gives the settings of a session the web UI opens in another directory, beside that one's
    project file; without it, such a session starts with `settings`. The terminal shows one session at a
    time: `/resume` of one the web UI has open shows that one, and the other stays live.
    """
    # Before anything is drawn, as the CLI checks these before it starts.
    session_id = launch_session_id(resume=resume, session_id=session_id, fork=fork_session)
    console = console or Console()
    rebuild_stock = agent.with_plugins if isinstance(agent, StockAgent) else None
    transcript = TranscriptBuffer()
    with theme.use(lambda: settings.theme if settings is not None else 'default'), transcript.capture(console):
        project = project or ProjectSettings()
        print_welcome(project, console, worktree=worktree)
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
    served: WebHandoff | None = None
    with (
        telemetry_log(shell.context.store.path.with_name('telemetry.log'), console=console),
        relay_stderr() if console.is_terminal else nullcontext(),
    ):
        async with agent:
            while True:
                reason: SessionEndReason = 'error'
                # The terminal shows this session, live again after a failed reload.
                shell.join()
                shell.process.live.focused = shell.live_session.id
                terminal = shell.terminal
                assert terminal is not None, 'a shell built for the terminal is shown in it'
                shell.process.live.build = _opener(
                    shell,
                    agent=lambda: rebuild_stock(()) if rebuild_stock is not None else agent,
                    deps=deps,
                    plugins=plugins,
                    settings=lambda project: (
                        settings_for(project) if settings_for is not None else shell.context.settings
                    ),
                    load_plugins=load_plugins,
                    console=terminal.session_console,
                )
                with theme.use(
                    lambda terminal=terminal: (terminal.shown or shell).context.settings.theme,
                    output=console.file if console.is_terminal else None,
                ):
                    try:
                        # Sessions the web UI opens run beside this one, from the start: plugins may open some.
                        async with create_task_group() as workers, shell.process.live.serving():
                            workers.start_soon(shell.process.namer.run)
                            try:
                                with transcript.capture(console):
                                    # Apply launch options first, so plugins start with the conversation they
                                    # name. The browser waits for plugins, whose models may name its sessions,
                                    # and until it picks one, telemetry has no session to assign.
                                    browse = resume == ''
                                    restored = bool(resume)
                                    notice = ''
                                    if not browse and (resume is not None or session_id is not None):
                                        notice = await shell.sessions.start(
                                            resume=resume,
                                            resume_from=resume_from,
                                            session_id=session_id,
                                            fork=fork_session,
                                        )
                                    with shell.defer_identity() if browse else nullcontext():
                                        await shell.loader.load_all(fresh=fresh)
                                        _report_project_plugins(shell.loader, console)
                                        await _serve_again(shell.loader, web, console=console)
                                        web = None
                                        if restored:
                                            # Now that observability has subscribed and renderers have loaded.
                                            shell.session.record_resumed()
                                            await shell.sessions.show_resumed()
                                        if browse:
                                            notice = await shell.sessions.start(
                                                resume='',
                                                resume_from=resume_from,
                                                session_id=session_id,
                                                fork=fork_session,
                                            )
                                        if notice:
                                            console.print(notice, markup=False)
                                    # Applied once: a failed reload resumes the session shown without them.
                                    resume = session_id = resume_from = None
                                    fork_session = False
                                warming = warming or warm_imports.start()
                                reason = await shell.run()
                            finally:
                                workers.cancel_scope.cancel()
                    except BaseExceptionGroup as exc:
                        if len(exc.exceptions) == 1:
                            raise exc.exceptions[0] from None
                        raise
                    finally:
                        # The session shown last restarts CLAI: `/reload` and `/update` work in it.
                        last = terminal.ended_in or shell
                        served = terminal.served
                        if not terminal.home_closed:
                            with transcript.capture(console):
                                # It ends for good when another session's reload or exit restarts or ends CLAI.
                                await shell.close(reason if last is shell else 'exit')
                if not last.reload_requested:
                    if (relaunch := last.relaunch(web=served)) is not None:
                        raise relaunch
                    return
                last.reload_requested = False
                if warming is not None:  # pragma: no branch -- a reload follows a run, which started warming
                    warming.join()
                try:
                    shell = reload_clai(
                        lambda shell=last: create_shell(
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
                            project=shell.context.project,
                            message_history=shell.session.messages,
                            summary=shell.session.summary,
                            launched=shell.sessions.launched,
                            transcript=transcript,
                            load_plugins=load_plugins,
                            workspace=Path(shell.session.workspace),
                        )
                    )
                except Exception as exc:  # noqa: BLE001 -- development edits must not discard the conversation.
                    with transcript.capture(console):
                        _report_reload_failure(exc, console=console)
                    fresh = False
                    # When the session shown asked for the reload, the one CLAI started with continues its conversation.
                    resume = None if last is shell else last.session.conversation_id
                else:
                    with transcript.capture(console):
                        console.print('CLAI2 reloaded. Conversation preserved.', style=theme.color(theme.INFO))
                    fresh = True


def _report_reload_failure(exc: Exception, *, console: Console) -> None:
    console.print(f'Reload failed: {type(exc).__name__}: {exc}', style=theme.color(theme.ERROR), markup=False)
    if isinstance(exc, ImportError) and exc.name and exc.name.startswith('pydantic_ai_harness'):
        console.print(
            'Harness is not refreshed by /reload. Restart CLAI2 with the same launch options and '
            '--resume to continue this session. Keep the worktree if asked to remove it.',
            style=theme.color(theme.INFO),
            markup=False,
        )


async def _serve_again(loader: SessionPlugins[DepsT], web: WebHandoff | None, *, console: Console) -> None:
    """Serve `web` again, the web UI CLAI served before `/update` restarted it; nothing without one."""
    if web is not None:
        from pydantic_clai2.cli.web import serve_again

        console.print(await serve_again(loader, web), markup=False)


def _opener(
    shell: SessionRuntime[DepsT, OutputT],
    *,
    agent: Callable[[], AbstractAgent[DepsT, OutputT]],
    deps: DepsT,
    plugins: Sequence[AgentCapability[DepsT]],
    settings: Callable[[ProjectSettings], Settings],
    load_plugins: bool,
    console: Callable[[], Console],
) -> Callable[[Path | None], SessionRuntime[DepsT, OutputT]]:
    """Builds a session the web UI opens beside `shell`, in a directory, by default the one CLAI started in."""
    launched_in, launch_project = Path(shell.session.workspace), shell.context.project

    def build(directory: Path | None) -> SessionRuntime[DepsT, OutputT]:
        project = launch_project if directory in (None, launched_in) else load_project_settings(directory)
        return create_shell(
            agent(),
            deps=deps,
            plugins=plugins,
            usage_limits=shell.session.usage_limits,
            console=console(),
            settings=settings(project),
            store=None,
            builtin_plugins=(),
            project=project,
            headless=True,
            load_plugins=load_plugins,
            workspace=directory or launched_in,
            process=shell.process,
            signals=False,
        )

    return build


def _report_project_plugins(loader: SessionPlugins[DepsT], console: Console) -> None:
    waiting = [entry.name for entry in loader.entries() if entry.project and entry.loaded is None]
    if waiting:
        console.print(
            f'Project plugins not loaded; approve one with /plugins enable NAME: {", ".join(waiting)}',
            style=theme.color(theme.INFO),
        )
