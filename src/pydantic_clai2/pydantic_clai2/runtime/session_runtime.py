"""One live CLAI session: its conversation, plugins, commands, input, and the turn it runs."""

import asyncio
import math
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Generic, TypeVar

from anyio import Lock, create_memory_object_stream, create_task_group
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import History
from pydantic import ValidationError
from rich.console import Console

from pydantic_ai import AgentRunResult, AgentStreamEvent
from pydantic_ai._context_size import context_tokens
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary
from pydantic_clai2.cli.command_context import CommandContext, CommandProvider
from pydantic_clai2.cli.effort import effort_command, effort_completions
from pydantic_clai2.cli.self_update import Relaunch, Updates, WebHandoff
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
from pydantic_clai2.errors import error_message
from pydantic_clai2.models import login_names
from pydantic_clai2.models.chains import settings_model as chain_settings_model
from pydantic_clai2.models.profiles import provider_of
from pydantic_clai2.models.resolver import ModelResolver
from pydantic_clai2.plugins import (
    CommandEnd,
    ConversationChanged,
    Renderer,
    SessionEndReason,
    SessionStart,
    TaskEvent,
    TurnEnd,
    TurnStart,
    UpdateProgress,
    bare_screen,
)
from pydantic_clai2.plugins.loader import PluginError, PluginLoader
from pydantic_clai2.runtime._session import (
    Session,
    StockAgent,
    agent_capabilities,
    current_session_id,
)
from pydantic_clai2.runtime.capability_guard import CapabilitySetupError
from pydantic_clai2.runtime.forks import Forks
from pydantic_clai2.runtime.imported_sessions import IMPORT_SOURCES
from pydantic_clai2.runtime.process import ClaiProcess
from pydantic_clai2.runtime.session_settings import SessionSettings
from pydantic_clai2.runtime.sessions import Sessions
from pydantic_clai2.runtime.speculation import Speculation
from pydantic_clai2.runtime.tasks import Tasks, resume_prompt, task_row
from pydantic_clai2.ui import telemetry
from pydantic_clai2.ui.menus.key_menu import keys_command
from pydantic_clai2.ui.menus.model_picker import MODEL_SUBCOMMANDS, model_command, model_completions
from pydantic_clai2.ui.menus.plugin_menu import open_plugins_menu
from pydantic_clai2.ui.menus.rewind import rewind
from pydantic_clai2.ui.menus.set_menu import set_command
from pydantic_clai2.ui.menus.spinner_picker import spinner_command, spinner_completions
from pydantic_clai2.ui.menus.system_prompt_menu import system_prompt_command
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
from pydantic_clai2.ui.rendering._rendering import StreamRenderer
from pydantic_clai2.ui.rendering.history import render_history
from pydantic_clai2.ui.rendering.spinners import Spinner, Spinners
from pydantic_clai2.ui.rendering.status import Status, StatusLine
from pydantic_clai2.ui.rendering.tool_output import terminal_text
from pydantic_clai2.ui.rendering.usage_report import cost_line, session_usage
from pydantic_clai2.ui.rendering.welcome import print_welcome

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
_PLUGIN_ACTIONS = ('list', 'add', 'enable', 'disable', 'remove', 'reload', 'configure')
_PLUGINS_OFF = 'Plugins are off for this session; saved plugin settings are unchanged.'
_PLUGINS_WAIT = 'Plugin changes wait for delegated tasks. Stop them in /tasks or wait for completion.'


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
    launched: str | None = None,
    transcript: TranscriptBuffer | None = None,
    headless: bool = False,
    load_plugins: bool = True,
    workspace: Path | None = None,
    process: ClaiProcess | None = None,
    signals: bool = True,
) -> 'SessionRuntime[DepsT, OutputT]':
    """Build one session's runtime, without attaching terminal input in headless mode.

    `workspace` is the directory the session works in, by default the current one. `process` holds
    what the session shares with the others in this CLAI; without one, it gets its own over `store`.
    `signals` lets Ctrl-C cancel this session's turns; only the session the terminal shows takes it.
    """
    settings = (
        Settings.model_validate(settings.model_dump(exclude_unset=True))
        if settings is not None
        else Settings(model=None)
    )
    if process is not None and store is not None:
        raise TypeError('create_shell takes `store` or `process`, not both: the process holds its store.')
    process = process or ClaiProcess.open(store)
    store, conversations = process.store, process.conversations
    transcript = transcript if transcript is not None else TranscriptBuffer()
    session = Session(
        agent,
        deps=deps,
        plugins=plugins,
        usage_limits=usage_limits,
        message_history=message_history,
        conversations=conversations,
        summary=summary,
        workspace=workspace,
    )
    session.model = settings.model
    session.model_chosen = 'model' in settings.model_fields_set
    session.tool_retries = settings.tool_retries
    session.instructions = settings.instructions
    models = ModelResolver(console=console, store=store, codex_providers=process.codex_providers)
    session.resolve_model = models.resolve
    if session.model is None and agent.model is None:
        console.print('Add a model with /model add.', style=theme.color(theme.INFO))

    session_settings = SessionSettings(session=session, console=console, settings=settings)

    context = CommandContext(settings=settings, store=store, apply_setting=session_settings, project=project)
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

    sessions = Sessions(session=session, store=conversations, context=context, launched=launched, namer=process.namer)
    commands = Commands(workspace=lambda: Path(session.workspace))
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
            args_during_turn=True,
        )
    )
    commands.register(
        Command(
            name='system_prompt',
            description='View the system prompt and edit your own instructions, sent after the built-in ones',
            handler=lambda args: system_prompt_command(context, args, history=lambda: session.messages),
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
        print_welcome(project, console)

    async def clear(_: list[str]) -> str:
        await session.clear()
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
        session_start=lambda: SessionStart(agent=agent, settings=context.settings, conversation_chosen=sessions.chosen),
        builtin=tuple(PluginSettings.model_validate(plugin.model_dump()) for plugin in builtin_plugins),
        full_screen=screen.full,
        project=tuple(PluginSettings.model_validate(plugin.model_dump()) for plugin in project.plugins),
        conversation=session,
        session_id=lambda: shell.session_id,
        status=status,
        enabled=load_plugins,
        submit_prompt=lambda text: shell.submit_prompt(text),
        send_input=lambda text: shell.send_input(text),
        interrupt=lambda: shell.interrupts.cancel(exit_on_repeat=False),
        sessions=sessions,
        setting=context.setting,
        tasks=lambda: shell.tasks if shell.session.delegations is not None else None,
        workspace=lambda: Path(session.workspace),
    )
    session.on_change = _ConversationFooter(status=status, fire=loader.fire, shown=session.conversation_id).changed
    updates.report = lambda: loader.fire if loader.handles(UpdateProgress) else None
    # Only a page has no terminal to watch uv in; elsewhere uv keeps drawing its progress there.
    updates.capture = lambda: web_serving(loader) is not None
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
    shell = SessionRuntime(
        process=process,
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
        interrupts=Interrupts(signals=signals),
        screen=screen,
        sessions=sessions,
        speculation=Speculation(context=context, console=console, repl_states=session.repl_states),
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
    commands.register(
        Command(
            name='monty',
            description="Run your own Python in Monty, this session's run_code sandbox: /monty CODE",
            handler=shell.monty_command,
            raw=True,
            live=True,
        )
    )

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


def web_serving(loader: PluginLoader[DepsT]) -> WebHandoff | None:
    """The port and token of the web UI the `web` plugin serves now; `None` while it serves none."""
    from pydantic_clai2.cli.web import serving

    return serving(loader)


@dataclass(kw_only=True)
class _ConversationFooter:
    """Tells plugins about a conversation change, starting the footer afresh when the ID changed."""

    status: Status
    fire: Callable[[ConversationChanged], Awaitable[None]]
    shown: str

    async def changed(self, event: ConversationChanged) -> None:
        # The footer belongs to one conversation: any switch, by command or plugin, starts it afresh.
        if event.conversation_id != self.shown:
            self.shown = event.conversation_id
            self.status.clear_conversation()
        await self.fire(event)


@dataclass(kw_only=True)
class SessionRuntime(Generic[DepsT, OutputT]):
    """One live session and its prompt loop; one turn is one `TurnStart`, one agent run, one `TurnEnd`.

    It reads input from the terminal's editor when its console is one, and otherwise from `send_input`.
    """

    process: ClaiProcess
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
    submitted: list[str] | None = None
    """Prompts the running `/command` submitted with `host.submit_prompt`; `None` while no command runs."""
    reload_requested: bool = False
    editor: LivePrompt | None = None
    forks: Forks[DepsT, OutputT] = field(init=False)
    tasks: Tasks = field(init=False)
    _mid_turn_commands: MemoryObjectSendStream[str] | None = field(default=None, init=False, repr=False)
    _steering: MemoryObjectSendStream[str] | None = field(default=None, init=False, repr=False)
    _identity_pending: bool = field(default=False, init=False)
    _inbox: asyncio.Queue[str] = field(default_factory=asyncio.Queue[str], init=False, repr=False)
    """Input `send_input` entered while no editor reads keys, as in a shell driven only by a plugin."""

    def relaunch(self, *, web: WebHandoff | None) -> Relaunch | None:
        """Restart as the build `/update` installed, resuming this conversation and serving `web` again; else `None`."""
        if (executable := self.updates.relaunch) is None:
            return None
        summary = self.session.summary
        return Relaunch(
            executable=executable,
            session_id=summary.id if summary.revision else None,
            # A conversation named with `--session-id` keeps its ID until its first prompt saves it.
            new_session_id=summary.id if not summary.revision and self.sessions.chosen else None,
            web=web,
        )

    @property
    def session_id(self) -> str | None:
        """The active saved ID, unavailable while startup is choosing a conversation to resume."""
        return None if self._identity_pending else current_session_id() or self.session.summary.id

    @contextmanager
    def defer_identity(self) -> Generator[None]:
        """Keep startup telemetry unassigned until the requested conversation is selected, then bind roots to it."""
        self._identity_pending = True
        try:
            yield
        finally:
            self._identity_pending = False
        telemetry.conversation_selected()

    def __post_init__(self) -> None:
        self.tasks = Tasks(
            console=self.console,
            conversation_id=lambda: self.session.summary.id,
            router=self.process.tasks,
            fire=self.loader.fire,
            listening=lambda: self.loader.handles(TaskEvent),
            send_input=self.send_input,
        )
        self.status.subagent = self.tasks.focused
        self.forks = Forks(
            console=self.console,
            history=lambda: self.session.messages,
            spawn=self.fork_session,
            fire=self.loader.fire,
            models=self.context.store.models,
        )
        self.sessions.quiet = self.forks.quiet
        self.session.on_setup_error = self.capability_failed

    def run_plugins(self) -> tuple[AgentCapability[DepsT], ...]:
        """Capabilities bound to the next run: supplied, plugin-registered (guarded), then speculation.

        Speculation sees the others unguarded, and the agent's own capabilities too, so its sandbox
        mount stays within every `FileSystem` and `ToolGuardrail` the run has.
        """
        granted = (*self.plugins, *self.loader.capabilities())
        bound = (*self.plugins, *self.loader.run_capabilities())
        if self.session.delegations is not None:
            granted = (*granted, self.tasks.presentation)
            bound = (*bound, self.tasks.presentation)
        own = agent_capabilities(self.agent)
        # An agent that hides its capabilities may guard tools a mount would bypass, so it gets none.
        return (*bound, *self.speculation.capabilities(None if own is None else (*own, *granted)))

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
        child.instructions = self.session.instructions
        child.resolve_model = self.session.resolve_model
        child.model_settings = self.context.live_model_overrides(child.model or _model_label(self.agent))
        child.model_defaults = self.context.model_defaults(child.model or _model_label(self.agent))
        child.on_setup_error = self.capability_failed
        if child.repl_states is not None:
            child.repl_states.copy(source=self.session.summary.id, target=child.summary.id)
        return child

    async def tasks_command(self, args: list[str]) -> str:
        if not args:
            return await open_tasks(self.tasks)
        if len(args) != 2 or args[0] not in ('stop', 'background', 'resume'):
            raise ValueError('Usage: /tasks [stop|background|resume ID]')
        record = self.tasks.resolve(args[1])
        if args[0] == 'stop':
            await self.tasks.stop(record.id)
            return f'Stopping task {record.id[:8]} and its descendants.'
        if args[0] == 'background':
            self.tasks.background(record.id)
            return f'Task {record.id[:8]} moved to background.'
        await self.tasks.owner.allow_resume(record.id)
        if self.editor is None:
            return f'Task {record.id} may now be resumed with delegate_task(resume={record.id!r}).'
        self.editor.submit(resume_prompt(record))
        return f'Resume requested for task {record.id[:8]}.'

    def submit_prompt(self, text: str) -> None:
        """Queue `text` to run as a turn once the running `/command` returns."""
        if self.submitted is None:
            raise RuntimeError('`submit_prompt` works only in a /command handler run between turns.')
        self.submitted.append(text)

    async def monty_command(self, args: list[str]) -> str:
        """`/monty CODE`: run a cell in Monty, code mode's sandbox; the next prompt tells the model."""
        code = args[0].strip() if args else ''
        if not code:
            raise ValueError("Usage: /monty CODE runs your own cell in Monty, this session's run_code sandbox.")
        if not (self.speculation.enabled or self.context.settings.code_mode):
            raise ValueError('Code mode is off, so there is no Monty sandbox. Turn it on with /set run.code_mode true.')
        if self.session.repl_states is None:
            raise ValueError('This CLAI saves no sessions, so no Monty state outlives a run.')
        self.session.plugins = self.run_plugins()
        cell = await self.session.run_in_monty(code)
        if cell.failed:
            raise ValueError(f'The cell failed:\n{cell.report()}')
        return cell.report() or '(no output)'

    def send_input(self, text: str) -> None:
        """Enter `text` as Enter at the prompt does: trimmed, bare `clear` as `/clear`, blank ignored.

        A `during_turn` command opens over a running turn; other input queues in the editor, or, while no
        editor reads keys yet or at all, in the inbox the read loop takes it from. A line-mode prompt,
        used when CLAI's output is not a terminal, reads only its own input, so it takes none.
        """
        if self.prompt is not None and self.editor is None:
            raise RuntimeError('This CLAI reads its prompts line by line, as its output is not a terminal.')
        text = expand_bare_command(text.strip())
        if not text or self.run_now(text):
            return
        if self.editor is not None:
            self.editor.submit(text)
        else:
            self._inbox.put_nowait(text)

    def request_reload(self, args: list[str]) -> str:
        if args:
            raise ValueError('Usage: /reload')
        self.reload_requested = True
        return 'Reloading CLAI2...'

    async def run(self) -> SessionEndReason:
        try:
            async with self.process.tasks.opened(self.tasks) as owner:
                if isinstance(self.agent, StockAgent):
                    self.session.delegations = owner
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
                    # Input a plugin entered while plugins loaded, before the editor existed.
                    while not self._inbox.empty():
                        self.editor.submit(self._inbox.get_nowait())
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

    def background_running(self) -> bool:
        """Whether a background fork or a delegated task of this session runs, using the plugins loaded now."""
        return any(record.status == 'running' for record in self.forks.records) or self.tasks.running()

    def plugins_busy(self, text: str) -> bool:
        """Keep plugin-owned transports alive until all children using them settle."""
        if text.split(maxsplit=1)[0] != '/plugins':
            return False
        if not self.tasks.running():
            return False
        self.console.print(_PLUGINS_WAIT, markup=False)
        return True

    async def _refused(self, text: str) -> bool:
        """`plugins_busy`, telling plugins the command was refused, as `send_input` callers wait for its end."""
        if not self.plugins_busy(text):
            return False
        await self.loader.fire(CommandEnd(text=text, error=RuntimeError(_PLUGINS_WAIT)))
        return True

    async def _run_mid_turn(self, text: str) -> None:
        async with self.screen.overlay():
            self.console.print(f'> {terminal_text(text)}', markup=False, highlight=False)
            self.console.print()
            if await self._refused(text):
                return
            await self.loader.fire(await _execute_command(self.commands, text, console=self.console))
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
                elif self.prompt is not None:
                    text = expand_bare_command((await self.prompt.prompt_async('> ')).strip())
                else:
                    text = (await self._inbox.get()).strip()
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
            # A line-mode prompt shows the line as it was typed; other input is echoed like the editor's.
            if self.prompt is None:
                self.console.print(f'> {terminal_text(text)}', markup=False, highlight=False)
            self.console.print()
            if await self._dispatch_input(text):
                return 'exit'

    async def _dispatch_input(self, text: str) -> bool:
        if (command := shell_command(text)) is not None:
            async with self.forks.busy(), self._released():
                context = await run_shell_command(
                    command, console=self.console, interrupts=self.interrupts, cwd=Path(self.session.workspace)
                )
                await self.session.commit_messages(
                    [*self.session.messages, ModelRequest(parts=[UserPromptPart(context)])]
                )
            return self.interrupts.exit_requested
        if is_command_input(text):
            return await self._command(text)
        return await self._prompt(text)

    async def _prompt(self, text: str) -> bool:
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
        if await self._refused(text):
            return False
        self.submitted = submitted = list[str]()
        ended: list[CommandEnd] = []
        try:
            if self.editor is not None and self.commands.runs_live(text):
                completed = await self._live_command(text, self.editor, ended)
                exiting = self.interrupts.exit_requested
            else:
                async with self.forks.busy(), self._released():
                    completed = await self.interrupts.run(self._execute(text, ended))
                exiting = (
                    text == '/exit'
                    or self.interrupts.exit_requested
                    or self.reload_requested
                    or self.updates.restart_required
                )
        finally:
            self.submitted = None
        # Plugins hear of the command once Esc no longer reaches it, so a slow handler cannot cancel it.
        for event in ended:
            await self.loader.fire(event)
        # A cancelled command's prompts are cancelled with it.
        while completed and submitted and not exiting:
            exiting = await self._prompt(submitted.pop(0))
        return exiting

    async def _execute(self, text: str, ended: list[CommandEnd]) -> None:
        """Run a command, adding its `CommandEnd` to `ended` unless it is cancelled first."""
        ended.append(await _execute_command(self.commands, text, console=self.console))

    async def _live_command(self, text: str, editor: LivePrompt, ended: list[CommandEnd]) -> bool:
        """Run a slow `live` command such as `/compact` with the spinner up and its cancel keys working.

        Suspending the editor instead would freeze its last frame, an idle prompt and a `ready`
        footer, for as long as the command runs, and echo keys raw into the terminal.

        Returns whether the command completed: `False` when the user cancelled it.
        """
        async with self.forks.busy():
            self.status.activity = 'working'
            try:
                completed = await self.interrupts.run(self._execute(text, ended))
            finally:
                self.status.activity = 'ready'
                await editor.output.drain()
        if not completed:
            self.console.print('Command cancelled.', style=theme.color(theme.MUTED))
            self.console.print()
        return completed

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
        self.sessions.namer.submit(self.session.summary.id, client=self.sessions.naming)
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


def _report_interrupt(completed: bool, console: Console) -> None:
    if not completed:
        console.print('Turn cancelled. Use /exit to quit.', style=theme.color(theme.MUTED), highlight=False)
        console.print()


async def _execute_command(commands: Commands, text: str, *, console: Console) -> CommandEnd:
    """Run a command and show its result; the caller tells plugins, outside the scope that Esc cancels."""
    ended = CommandEnd(text=text)
    try:
        result = await commands.execute_async(text)
        # The echoed command already ends in a blank line; a menu closed without changes adds nothing.
        if not is_silent(result):
            ended.output = result
            console.print(result, markup=False)
            console.print()
    except Exception as exc:  # noqa: BLE001 -- command failures must not exit the interactive shell.
        ended.error = exc
        console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
        console.print()
    return ended


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
        status.context_estimated = False

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
                status.context_tokens = context_tokens(message) or None
                status.context_estimated = False
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
