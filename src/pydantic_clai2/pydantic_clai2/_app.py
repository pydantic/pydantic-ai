"""Interactive terminal shell around a capability-independent session."""

import asyncio
from collections.abc import AsyncGenerator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Generic, TypeVar

from anyio import create_task_group
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import History
from pydantic import ValidationError
from pydantic_ai import Agent, AgentStreamEvent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore
from rich.console import Console

from . import openrouter, theme, vllm
from ._branding import print_banner
from ._completion_adapter import COMPLETION_STYLE, PromptCompleter
from ._rendering import StreamRenderer
from ._session import Session
from .auth import CodexAuth
from .capability_catalog import HARNESS_PLUGINS
from .command_context import CommandContext, CommandProvider
from .commands import Command, Commands, config_command, config_completions, is_command_input, set_completions
from .config import PluginSettings, Settings
from .customization import customization_guide
from .image_input import ImageInput
from .input_history import input_history
from .interrupts import Interrupts
from .key_menu import keys_command
from .live_prompt import LivePrompt
from .model_menu import model_settings_command, open_add_model_menu
from .model_picker import model_command, model_completions
from .plugin_loader import PluginError, PluginLoader
from .plugin_menu import open_plugins_menu
from .plugins import Renderer, SessionEndReason, SessionStart, TurnEnd, TurnStart, bare_screen
from .project_settings import ProjectSettings
from .prompt_transcript import TranscriptBuffer
from .reloading import reload_clai
from .screen import Screen
from .sessions import Sessions
from .set_menu import set_command
from .settings_store import SettingsStore
from .status import Status, StatusLine
from .theme_picker import theme_command
from .tool_output import terminal_text
from .usage_report import cost_line, session_usage

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
_PLUGIN_ACTIONS = ('list', 'add', 'enable', 'disable', 'remove', 'reload')


DEFAULT_PLUGINS: tuple[PluginSettings, ...] = (
    PluginSettings(
        id='coder',
        factory='pydantic_ai_harness.coder:Coder',
        settings={'unrestricted_filesystem': True, 'repo_context': False},
    ),
    PluginSettings(id='ask_user', factory='pydantic_clai2.ask_user_menu:activate'),
    PluginSettings(id='repo_context', factory='pydantic_clai2.repo_context'),
    PluginSettings(id='compaction', factory='pydantic_clai2.compaction', settings={}),
    PluginSettings(id='persistence', factory='pydantic_clai2.sessions'),
    PluginSettings(id='logfire', factory='pydantic_clai2.logfire'),
    PluginSettings(id='notifications', factory='pydantic_clai2.notifications'),
    PluginSettings(id='mcp', factory='pydantic_clai2.mcp'),
    *HARNESS_PLUGINS,
)
"""Built-in declarations, including opt-in harness capabilities. `remove` restores their defaults.

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
    resume: str | None = None,
) -> None:
    """Start an asyncio terminal conversation with a caller-supplied agent.

    Esc cancels the current turn; Ctrl-C also clears idle input. Ctrl-D and `/exit` quit.
    Failed and cancelled turns retain their captured history. Resume never replays tools.
    `project` is the parsed `.clai/settings.json`; layer its overrides into `settings` yourself.
    """
    console = console or Console()
    transcript = TranscriptBuffer()
    with theme.use(lambda: settings.theme if settings is not None else 'default'), transcript.capture(console):
        console.print()
        print_banner(console)
        console.print(
            '/new starts a session; /resume restores one; /exit quits. Esc or Ctrl-C interrupts a turn.',
            style=theme.color(theme.MUTED),
        )
        project = project or ProjectSettings()
        _report_project(project, console)
        use_defaults = builtin_plugins is DEFAULT_PLUGINS
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
        )
    fresh = False
    async with agent:
        while True:
            reason: SessionEndReason = 'error'
            with theme.use(lambda: shell.context.settings.theme, output=console.file if console.is_terminal else None):
                try:
                    async with create_task_group() as workers:
                        workers.start_soon(shell.sessions.namer.run)
                        try:
                            with transcript.capture(console):
                                await shell.loader.load_all(fresh=fresh)
                                _report_project_plugins(shell.loader, console)
                                if resume is not None:
                                    console.print(
                                        await shell.sessions.command([resume] if resume else []), markup=False
                                    )
                                    resume = None
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
                return
            shell.reload_requested = False
            try:
                shell = reload_clai(
                    lambda shell=shell: create_shell(
                        agent,
                        deps=deps,
                        plugins=plugins,
                        usage_limits=shell.session.usage_limits,
                        console=console,
                        settings=shell.context.settings,
                        store=SettingsStore(shell.context.store.path),
                        builtin_plugins=DEFAULT_PLUGINS if use_defaults else builtin_plugins,
                        project=project,
                        message_history=shell.session.messages,
                        summary=shell.session.summary,
                        transcript=shell.transcript,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- development edits must not discard the conversation.
                with transcript.capture(console):
                    console.print(
                        f'Reload failed: {type(exc).__name__}: {exc}', style=theme.color(theme.ERROR), markup=False
                    )
                fresh = False
            else:
                with transcript.capture(console):
                    console.print('CLAI2 reloaded. Conversation preserved.', style=theme.color(theme.INFO))
                fresh = True


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
) -> '_Shell[DepsT, OutputT]':
    """Build shared session services, without attaching terminal input in headless mode."""
    settings = Settings.model_validate(settings.model_dump()) if settings is not None else Settings(model=None)
    store = store or SettingsStore()
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
    session.tool_retries = settings.tool_retries
    auth = CodexAuth(console)

    async def resolve_model(name: str) -> Model | str:
        if name.startswith('openrouter:'):
            return await asyncio.to_thread(openrouter.model, name)
        if name.startswith('vllm:'):
            return await asyncio.to_thread(vllm.model, name)
        return auth.model(name) if name.startswith('openai-codex:') else name

    session.resolve_model = resolve_model
    if session.model is None and agent.model is None:
        console.print('Add a model with /add_model.', style=theme.color(theme.INFO))

    previous_theme = settings.theme

    def apply_setting(key: str, updated: Settings) -> None:
        nonlocal previous_theme
        if key == 'model':
            session.model = updated.model
        elif key == 'run.tool_retries':
            session.tool_retries = updated.tool_retries
        elif key == 'run.request_limit':
            session.usage_limits = replace(session.usage_limits or UsageLimits(), request_limit=updated.request_limit)
        elif key == 'display.theme' and console.is_terminal and previous_theme != updated.theme:
            theme.apply(updated.theme, output=console.file)
        previous_theme = updated.theme

    context = CommandContext(
        settings=settings, store=store, clear_history=session.clear, apply_setting=apply_setting, project=project
    )

    sessions = Sessions(session=session, store=conversations, context=context)
    commands = Commands()
    commands.register(Command(name='resume', description='Browse or restore a saved session', handler=sessions.command))
    commands.register(Command(name='keys', description='Manage saved API keys', handler=keys_command))
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
            handler=lambda args: set_command(context, args),
            complete=set_completions,
        )
    )
    commands.register(
        Command(
            name='theme',
            description='Select a Termflow palette; no arguments opens the picker',
            handler=lambda args: theme_command(context, args),
            complete=lambda args: theme.names() if len(args) <= 1 else (),
        )
    )
    commands.register(
        Command(
            name='model',
            description='Select an added model; no arguments opens the picker',
            handler=lambda args: model_command(context, args),
            complete=lambda args: model_completions(context, args),
        )
    )
    commands.register(
        Command(
            name='add_model',
            description='Add and use a model, or browse providers and model settings',
            handler=lambda args: context.set_setting(['model', *args]) if args else open_add_model_menu(context),
            complete=lambda args: set_completions(['model', *args]) if len(args) <= 1 else (),
        )
    )
    commands.register(
        Command(
            name='model_settings',
            description='Choose an added model to configure, or edit a named model',
            handler=lambda args: model_settings_command(context, args),
            complete=lambda args: model_completions(context, args),
        )
    )
    commands.register(Command(name='help', description='Show commands', handler=commands.help))
    commands.register(
        Command(
            name='new',
            description='Start a new session; preserve the previous session',
            handler=lambda _: session.clear() or 'New session started. Previous session remains saved.',
        )
    )
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
        transcript=transcript if transcript is not None else TranscriptBuffer(),
        images=images,
        interrupts=Interrupts(),
        screen=screen,
        sessions=sessions,
    )
    if prompt is not None:
        prompt.key_bindings = images.bindings()
    commands.register(
        Command(name='reload', description='Reload CLAI2 code without restarting', handler=shell.request_reload)
    )
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
    transcript: TranscriptBuffer = field(default_factory=TranscriptBuffer)
    images: ImageInput = field(default_factory=ImageInput)
    reload_requested: bool = False
    editor: LivePrompt | None = None

    def request_reload(self, args: list[str]) -> str:
        if args:
            raise ValueError('Usage: /reload')
        self.reload_requested = True
        return 'Reloading CLAI2...'

    async def run(self) -> SessionEndReason:
        if self.console.is_terminal:
            self.editor = LivePrompt(
                console=self.console,
                commands=self.commands,
                history=self.history,
                images=self.images,
                interrupts=self.interrupts,
                toolbar=self.status.toolbar,
                steer=self.steer,
                transcript=self.transcript,
            )
            self.screen.editor = self.editor.suspended
            try:
                async with self.editor.opened():
                    return await self._read_loop()
            finally:
                self.screen.editor = None
                self.editor = None
        return await self._read_loop()

    def steer(self, text: str) -> bool:
        """Resolve attachments and route input without printing over streamed output."""
        try:
            resolved, images = self.images.resolve(text)
        except ValueError as exc:
            self.images.notice = str(exc)
            return True
        if not self.session.steer(resolved, images=images):
            return False
        self.images.notice = f'Steering sent: {text}'
        return True

    async def _read_loop(self) -> SessionEndReason:
        while True:
            self.images.retain(
                [self.editor.buffer.text, *self.editor.queued_messages] if self.editor is not None else []
            )
            try:
                self.status.model = self.session.model or _model_label(self.agent)
                self.status.status_segments = tuple(self.loader.status_segments())
                if self.editor is not None:
                    text = await self.editor.read()
                else:
                    assert self.prompt is not None
                    text = (await self.prompt.prompt_async('> ')).strip()
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
            if is_command_input(text):
                async with (self.editor.suspended if self.editor is not None else bare_screen)():
                    await self.interrupts.run(
                        _execute_command(self.commands, text, console=self.console, status=self.status)
                    )
                if text == '/exit' or self.interrupts.exit_requested or self.reload_requested:
                    return 'exit'
                continue
            if self.session.model is None and self.agent.model is None:
                self.images.retry_text = text
                self.console.print('Choose a model first: /set model <Tab>', style=theme.color(theme.WARNING))
                continue
            try:
                if await self._turn(text):
                    return 'exit'
            finally:
                if self.editor is not None:
                    await self.editor.output.drain()

    async def _turn(self, text: str) -> bool:
        try:
            text, images = self.images.resolve(text)
        except ValueError as exc:
            self.console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
            return False
        start = TurnStart(text=text)
        ended: TurnEnd | None = None

        async def run_turn() -> None:
            nonlocal ended
            ended = await self.run_turn(start, images=images)

        completed = await self.interrupts.run(run_turn())
        self.sessions.namer.submit(self.session.summary.id)
        if self.editor is not None:
            await self.editor.output.drain()
        _report_interrupt(completed, self.console)
        await self.interrupts.run(self.loader.fire(ended or TurnEnd(text=start.text, outcome='cancelled')))
        return self.interrupts.exit_requested

    async def run_turn(
        self, start: TurnStart, *, images: Sequence[BinaryContent] = (), headless: bool = False
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
        self.session.plugins = (*self.plugins, *self.loader.capabilities())
        model = self.session.model or _model_label(self.agent)
        try:
            self.session.model_settings = self.context.model_settings(model)
        except ValidationError as exc:
            self.console.print(
                f'Invalid saved model settings for {model}. Fix or reset them with /model_settings {model}.',
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
                result = await self.session.prompt(start.text)
            except Exception as exc:  # noqa: BLE001 -- report a failed headless turn to the CLI.
                return TurnEnd(text=start.text, outcome='failed', error=exc)
            return TurnEnd(text=start.text, outcome='completed', result=result)
        return await _run_prompt(
            self.session,
            start.text,
            images=images,
            console=self.console,
            settings=self.context.settings,
            status=self.status,
            renderers=self.loader.renderers(),
            screen=self.screen,
        )


def _report_project(project: ProjectSettings, console: Console) -> None:
    if project.path is None:
        return
    console.print(f'Project settings: {project.path}', style=theme.color(theme.MUTED))
    if project.unknown:
        console.print(f'Ignoring unknown settings: {", ".join(project.unknown)}', style=theme.color(theme.WARNING))


def _report_project_plugins(loader: PluginLoader[DepsT], console: Console) -> None:
    waiting = [entry.name for entry in loader.entries() if entry.project and entry.host is None]
    if waiting:
        console.print(
            f'Project plugins not loaded; approve one with /plugins enable NAME: {", ".join(waiting)}',
            style=theme.color(theme.INFO),
        )


def _report_interrupt(completed: bool, console: Console) -> None:
    if not completed:
        console.print('Turn cancelled. Use /exit to quit.', style=theme.color(theme.MUTED), highlight=False)
        console.print()


async def _execute_command(commands: Commands, text: str, *, console: Console, status: Status) -> None:
    try:
        console.print(await commands.execute_async(text), markup=False)
    except Exception as exc:  # noqa: BLE001 -- command failures must not exit the interactive shell.
        console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
    console.print()
    _reset_status(text, status)


def _reset_status(command: str, status: Status) -> None:
    if command.split(maxsplit=1)[0] in ('/new', '/resume'):
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
    renderers: Sequence[Renderer[AgentStreamEvent]],
    screen: Screen,
    images: Sequence[BinaryContent] = (),
) -> TurnEnd:
    renderer = StreamRenderer(
        console,
        stop_loading=lambda: None,
        show_thinking=settings.thinking,
        smooth_seconds=settings.smooth_seconds,
        show_tool_output=settings.tool_output,
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
    status_line = StatusLine(console, status, enabled=screen.editor is None)

    @asynccontextmanager
    async def take_screen() -> AsyncGenerator[None]:
        await renderer.finish()
        async with status_line.paused():
            yield

    try:
        with screen.bound(take_screen):
            async with status_line:
                result = await session.prompt(text, images=images)
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
        console.print(f'{type(exc).__name__}: {exc}', style=theme.color(theme.ERROR), markup=False)
        console.print(
            'Turn failed. Retained history may include partial progress. External tool side effects may already have occurred.',
            style=theme.color(theme.MUTED),
        )
        console.print()
        return TurnEnd(text=text, outcome='failed', error=exc)
    finally:
        status.activity = 'ready'
        status.cost = session_usage(session.messages).total.cost
        session.on_context_usage = None
        await renderer.finish()
