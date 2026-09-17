"""Interactive terminal shell around a capability-independent session."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from pydantic_ai import Agent, AgentStreamEvent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits
from rich.console import Console

from . import openrouter, theme, vllm
from ._branding import print_banner
from ._completion_adapter import COMPLETION_STYLE, PromptCompleter
from ._rendering import StreamRenderer
from ._session import Session
from .auth import CodexAuth
from .command_context import CommandContext, CommandProvider
from .commands import Command, Commands, config_command, config_completions, set_completions
from .config import PluginSettings, Settings
from .customization import customization_guide
from .input_history import input_history
from .interrupts import Interrupts
from .model_menu import open_model_menu
from .plugin_loader import PluginError, PluginLoader
from .plugin_menu import open_plugins_menu
from .plugins import Renderer, SessionEndReason, SessionStart, TurnEnd, TurnStart
from .project_settings import ProjectSettings
from .set_menu import open_settings_menu
from .settings_store import SettingsStore
from .status import Status, StatusLine
from .usage_report import cost_line, session_usage, usage_command

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
_PLUGIN_ACTIONS = ('list', 'add', 'enable', 'disable', 'remove', 'reload')


DEFAULT_PLUGINS: tuple[PluginSettings, ...] = (
    PluginSettings(
        id='coder',
        factory='pydantic_ai_harness.coder:Coder',
        settings={'unrestricted_filesystem': True, 'repo_context': False},
    ),
    PluginSettings(id='repo_context', factory='pydantic_clai2.repo_context'),
    PluginSettings(id='compaction', factory='pydantic_clai2.compaction', settings={}),
)
"""Plugins CLAI ships enabled. `/plugins disable NAME` turns one off; `remove` restores this.

`coder` leaves out its own `RepoContext` because `repo_context` binds one, so instruction files load once.
"""


def create_agent(model: str | None = None) -> Agent[None, str]:
    """Build the base CLAI agent. The coding tools come from the built-in `coder` plugin, not from here."""
    return Agent(model, deps_type=type(None), capabilities=[customization_guide()])


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
) -> None:
    """Start an asyncio terminal conversation with a caller-supplied agent.

    Ctrl-C cancels the current turn or clears input; Ctrl-D and `/exit` quit.
    Failed and cancelled turns are not added to the retained history.
    `project` is the parsed `.clai/settings.json`; layer its overrides into `settings` yourself.
    """
    console = console or Console()
    console.print()
    print_banner(console)
    console.print('/new clears history; /exit quits. Ctrl-C interrupts a turn.', style=theme.MUTED)
    settings = settings or Settings(model=None)
    store = store or SettingsStore()
    project = project or ProjectSettings()
    _report_project(project, console)
    session = Session(agent, deps=deps, plugins=plugins, usage_limits=usage_limits)
    session.model = settings.model
    auth = CodexAuth(console)

    async def resolve_model(name: str) -> Model | str:
        if name.startswith('openrouter:'):
            return await asyncio.to_thread(openrouter.model, name)
        if name.startswith('vllm:'):
            return await asyncio.to_thread(vllm.model, name)
        return auth.model(name) if name.startswith('openai-codex:') else name

    session.resolve_model = resolve_model
    if session.model is None and agent.model is None:
        console.print('Choose a model with /set model <Tab>.', style=theme.INFO)

    def apply_setting(key: str, updated: Settings) -> None:
        if key == 'model':
            session.model = updated.model
        elif key == 'run.request_limit':
            session.usage_limits = replace(session.usage_limits or UsageLimits(), request_limit=updated.request_limit)

    context = CommandContext(
        settings=settings, store=store, clear_history=session.clear, apply_setting=apply_setting, project=project
    )

    commands = Commands()
    commands.register(
        Command(
            name='login',
            description='Connect your ChatGPT/Codex subscription',
            handler=auth.login,
            complete=lambda _: ('openai-codex',),
        )
    )
    commands.register(
        Command(
            name='set',
            description='Change settings; no arguments opens the menu',
            handler=lambda args: context.set_setting(args) if args else open_settings_menu(context),
            complete=set_completions,
        )
    )
    commands.register(
        Command(
            name='model',
            description='Pick a model or edit its settings; no arguments opens the menu',
            handler=lambda args: context.set_setting(['model', *args]) if args else open_model_menu(context),
            complete=lambda args: set_completions(['model', *args]) if len(args) <= 1 else (),
        )
    )
    commands.register(Command(name='help', description='Show commands', handler=commands.help))
    commands.register(
        Command(
            name='new',
            description='Clear conversation history',
            handler=lambda _: session.clear() or 'Conversation cleared.',
        )
    )
    commands.register(
        Command(
            name='usage',
            description='Show tokens and cost per turn',
            handler=lambda _: usage_command(session.messages, console=console),
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
    commands.register(
        Command(
            name='config',
            description='show|get|set|reset settings',
            handler=lambda args: config_command(store, args),
            complete=config_completions,
        )
    )
    status = Status()
    loader: PluginLoader[DepsT] = PluginLoader(
        store=store,
        console=console,
        commands=commands,
        session_start=lambda: SessionStart(agent=agent, settings=context.settings),
        builtin=builtin_plugins,
        project=project.plugins,
        conversation=session,
        status=status,
    )
    commands.register(
        Command(
            name='plugins',
            description='Manage plugins; no arguments opens the menu',
            handler=lambda args: loader.command(args) if args else open_plugins_menu(loader),
            complete=lambda args: _PLUGIN_ACTIONS if len(args) <= 1 else (entry.name for entry in loader.entries()),
        )
    )
    for plugin in plugins:
        if isinstance(plugin, CommandProvider):
            commands.register_many(plugin.get_commands(context))
    prompt = PromptSession[str](
        history=input_history(store.path.with_name('input-history')),
        completer=PromptCompleter(commands),
        complete_while_typing=True,
        style=COMPLETION_STYLE,
        reserve_space_for_menu=6,
        bottom_toolbar=lambda: FormattedText(status.toolbar()),
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
        interrupts=Interrupts(),
    )
    reason: SessionEndReason = 'error'
    try:
        async with agent:
            await loader.load_all()
            _report_project_plugins(loader, console)
            reason = await shell.run()
    finally:
        await loader.close(reason)


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
    prompt: PromptSession[str]
    interrupts: Interrupts

    async def run(self) -> SessionEndReason:
        while True:
            try:
                self.status.model = self.session.model or _model_label(self.agent)
                text = (await self.prompt.prompt_async('> ')).strip()
            except KeyboardInterrupt:
                if self.interrupts.press():
                    return 'exit'
                self.console.print('Input cleared. Press Ctrl-C again within 2 seconds to exit.', style=theme.MUTED)
                continue
            except EOFError:
                return 'eof'
            if not text:
                continue
            self.console.print()
            if text.startswith('/'):
                await self.interrupts.run(
                    _execute_command(self.commands, text, console=self.console, status=self.status)
                )
                if text == '/exit' or self.interrupts.exit_requested:
                    return 'exit'
                continue
            if self.session.model is None and self.agent.model is None:
                self.console.print('Choose a model first: /set model <Tab>', style=theme.WARNING)
                continue
            if await self._turn(text):
                return 'exit'

    async def _turn(self, text: str) -> bool:
        start = TurnStart(text=text)
        try:
            await self.loader.fire(start)
        except PluginError as exc:
            self.console.print(str(exc), style=theme.ERROR, markup=False)
            self.console.print()
            await self.loader.fire(TurnEnd(text=start.text, outcome='failed', error=exc))
            return False
        if start.cancelled:
            self.console.print(
                f'Turn cancelled by a plugin: {start.cancel_reason or "no reason given"}', style=theme.WARNING
            )
            self.console.print()
            await self.loader.fire(TurnEnd(text=start.text, outcome='cancelled'))
            return False
        self.session.plugins = (*self.plugins, *self.loader.capabilities())
        self.session.model_settings = self.context.model_settings(self.session.model or _model_label(self.agent))
        ended: TurnEnd | None = None

        async def run_prompt() -> None:
            nonlocal ended
            ended = await _run_prompt(
                self.session,
                start.text,
                console=self.console,
                settings=self.context.settings,
                status=self.status,
                renderers=self.loader.renderers(),
            )

        completed = await self.interrupts.run(run_prompt())
        _report_interrupt(completed, self.console)
        await self.loader.fire(ended or TurnEnd(text=start.text, outcome='cancelled'))
        return self.interrupts.exit_requested


def _report_project(project: ProjectSettings, console: Console) -> None:
    if project.path is None:
        return
    console.print(f'Project settings: {project.path}', style=theme.MUTED)
    if project.unknown:
        console.print(f'Ignoring unknown settings: {", ".join(project.unknown)}', style=theme.WARNING)


def _report_project_plugins(loader: PluginLoader[DepsT], console: Console) -> None:
    waiting = [entry.name for entry in loader.entries() if entry.project and entry.host is None]
    if waiting:
        console.print(
            f'Project plugins not loaded; approve one with /plugins enable NAME: {", ".join(waiting)}',
            style=theme.INFO,
        )


def _report_interrupt(completed: bool, console: Console) -> None:
    if not completed:
        console.print('Turn cancelled. Press Ctrl-C again within 2 seconds to exit.', style=theme.MUTED)
        console.print()


async def _execute_command(commands: Commands, text: str, *, console: Console, status: Status) -> None:
    try:
        console.print(await commands.execute_async(text), markup=False)
    except Exception as exc:  # noqa: BLE001 -- command failures must not exit the interactive shell.
        console.print(str(exc), style=theme.ERROR, markup=False)
    console.print()
    _reset_status(text, status)


def _reset_status(command: str, status: Status) -> None:
    if command.split(maxsplit=1)[0] == '/new':
        status.context_tokens = None
        status.context_alert = False
        status.output_tokens = None
        status.cost = None
        status.streamed_chars = 0


def _model_label(agent: AbstractAgent[DepsT, OutputT]) -> str:
    model = agent.model
    if isinstance(model, str):  # pragma: no cover -- concrete Agent resolves string models before chat.
        return model
    return model.model_name if model else 'agent default'


async def _run_prompt(
    session: Session[DepsT, OutputT],
    text: str,
    *,
    console: Console,
    settings: Settings,
    status: Status,
    renderers: Sequence[Renderer[AgentStreamEvent]] = (),
) -> TurnEnd:
    renderer = StreamRenderer(
        console,
        stop_loading=lambda: None,
        show_thinking=settings.thinking,
        smooth_seconds=settings.smooth_seconds,
        shell_lines=settings.shell_lines,
        grep_lines=settings.grep_lines,
        renderers=renderers,
    )
    status.streamed_chars = 0
    status.output_tokens = None
    status.activity = 'waiting'

    async def observe(event: AgentStreamEvent) -> None:
        status.observe(event)
        await renderer.on_stream_event(event)

    def context_usage(tokens: int) -> None:
        status.context_tokens = tokens

    session.on_context_usage = context_usage
    session.on_stream_event = observe
    try:
        async with StatusLine(console, status):
            result = await session.prompt(text)
            await renderer.finish()
        status.output_tokens = result.usage.output_tokens
        for message in reversed(result.all_messages()):  # pragma: no branch -- successful runs contain a response.
            if isinstance(message, ModelResponse):
                status.context_tokens = message.usage.total_tokens or None
                break
        if not renderer.rendered_text or not isinstance(result.output, str):
            console.print(str(result.output), markup=False)
            console.print()
        return TurnEnd(text=text, outcome='completed', result=result)
    except asyncio.CancelledError:
        await renderer.abort()
        raise
    except Exception as exc:  # noqa: BLE001 -- interactive boundary reports plugin/provider failures.
        await renderer.finish()
        console.print(f'{type(exc).__name__}: {exc}', style=theme.ERROR, markup=False)
        console.print('Turn not saved. External tool side effects may already have occurred.', style=theme.MUTED)
        console.print()
        return TurnEnd(text=text, outcome='failed', error=exc)
    finally:
        status.activity = 'ready'
        status.cost = session_usage(session.messages).total.cost
        session.on_context_usage = None
        await renderer.finish()
