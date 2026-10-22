"""One live CLAI session: its conversation, plugins, commands, input, and the turn it runs."""

import asyncio
import math
from collections import deque
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Generic, TypeVar, cast

from anyio import CancelScope, Lock, create_memory_object_stream, create_task_group, get_cancelled_exc_class, to_thread
from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream
from prompt_toolkit import PromptSession
from pydantic import ValidationError
from rich.console import Console

from pydantic_ai import AgentRunResult, AgentStreamEvent
from pydantic_ai._context_size import context_tokens
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.models import Model
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
    QueuedInput,
    Renderer,
    SessionEndReason,
    SessionStart,
    TaskEvent,
    TurnEnd,
    TurnStart,
    UpdateProgress,
    bare_screen,
)
from pydantic_clai2.plugins.loader import PluginError, SessionPlugins
from pydantic_clai2.plugins.menus import SettingsMenu
from pydantic_clai2.plugins.process_plugins import LiveRuntime, handling
from pydantic_clai2.runtime import image_codec
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
from pydantic_clai2.ui.menus.key_menu import keys_command, keys_menu
from pydantic_clai2.ui.menus.model_picker import MODEL_SUBCOMMANDS, model_command, model_completions, models_menu
from pydantic_clai2.ui.menus.plugin_menu import open_plugins_menu
from pydantic_clai2.ui.menus.rewind import rewind
from pydantic_clai2.ui.menus.set_menu import set_command, settings_menu
from pydantic_clai2.ui.menus.spinner_picker import spinner_command, spinner_completions
from pydantic_clai2.ui.menus.system_prompt_menu import system_prompt_command, system_prompt_menu
from pydantic_clai2.ui.menus.task_menu import open_tasks
from pydantic_clai2.ui.menus.theme_picker import theme_command
from pydantic_clai2.ui.menus.update_menu import update_menu
from pydantic_clai2.ui.prompt.image_input import ImageInput
from pydantic_clai2.ui.prompt.input_history import input_history
from pydantic_clai2.ui.prompt.interrupts import Interrupts
from pydantic_clai2.ui.prompt.live_prompt import LivePrompt, PromptRewind, PromptWakeup
from pydantic_clai2.ui.prompt.prompt_queue import PromptInbox, QueuedPrompt, can_steer, find_queued, input_kind
from pydantic_clai2.ui.prompt.prompt_transcript import TranscriptBuffer
from pydantic_clai2.ui.prompt.screen import Screen
from pydantic_clai2.ui.prompt.terminal import Terminal
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
_RESTARTS = 'Restarts CLAI: enter it in the session the terminal shows'


async def _default_model(
    *, store: SettingsStore, models: ModelResolver, fallback: Model | str | None
) -> Model | str | None:
    """The model a new session starts with: the saved `model` setting, else the agent's own, `fallback`."""
    saved = (await to_thread.run_sync(store.overrides)).get('model')
    return await models.resolve(saved) if isinstance(saved, str) else fallback


_RESTART_ELSEWHERE = (
    '/reload and /update restart CLAI. Enter them in the session the terminal shows, '
    'or with no terminal, in the session CLAI started with.'
)


async def _default_model_name(*, store: SettingsStore, fallback: Model | str | None) -> str | None:
    """The name of the model a new session starts with, as `/model` shows it; nothing is built or signed in."""
    saved = (await to_thread.run_sync(store.overrides)).get('model')
    if isinstance(saved, str):
        return saved
    return fallback if fallback is None or isinstance(fallback, str) else fallback.model_name


def _refuse_restart(args: list[str]) -> str:
    raise ValueError(_RESTART_ELSEWHERE)


def _process(
    *, process: ClaiProcess[DepsT] | None, store: SettingsStore | None, builtin_plugins: Sequence[PluginSettings]
) -> ClaiProcess[DepsT]:
    """The process a session joins: `process`, else a new one over `store` with `builtin_plugins`."""
    if process is not None and store is not None:
        raise TypeError('create_shell takes `store` or `process`, not both: the process holds its store.')
    return process or ClaiProcess[DepsT].open(
        store, builtin_plugins=tuple(PluginSettings.model_validate(plugin.model_dump()) for plugin in builtin_plugins)
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
    launched: str | None = None,
    transcript: TranscriptBuffer | None = None,
    headless: bool = False,
    load_plugins: bool = True,
    workspace: Path | None = None,
    process: ClaiProcess[DepsT] | None = None,
    signals: bool = True,
) -> 'SessionRuntime[DepsT, OutputT]':
    """Build one session's runtime, without attaching terminal input in headless mode.

    `workspace` is the directory the session works in, by default the current one. `process` holds
    what the session shares with the others in this CLAI, its built-in plugins included; without one,
    it gets its own over `store` and `builtin_plugins`, which are otherwise unused.
    `signals` lets Ctrl-C cancel this session's turns; only the session the terminal shows takes it.
    """
    settings = (
        Settings.model_validate(settings.model_dump(exclude_unset=True))
        if settings is not None
        else Settings(model=None)
    )
    process = _process(process=process, store=store, builtin_plugins=builtin_plugins)
    store, conversations = process.store, process.conversations
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

    def apply_setting(key: str, settings: Settings) -> None:
        session_settings(key, settings)
        # A model chosen outside the prompt, as from the web UI, shows at once, not at the next read; a running
        # turn holds the change, and its model, until it ends.
        shell.show_model()

    context = CommandContext(settings=settings, store=store, apply_setting=apply_setting, project=project)
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

    def effort_menu() -> SettingsMenu:
        # Model settings load provider SDKs; keep them off the startup path.
        from pydantic_clai2.ui.menus.model_menu import model_settings_menu

        model = session.model or _model_label(agent)
        return model_settings_menu(store, model, settings_as=context.settings_model(model))

    sessions = Sessions(session=session, store=conversations, context=context, launched=launched, namer=process.namer)
    commands = Commands(workspace=lambda: Path(session.workspace))
    commands.register(
        Command(
            name='effort',
            description='View or set reasoning effort: /effort [VALUE|reset]',
            handler=lambda args: effort_command(context, args, model=session.model or _model_label(agent)),
            complete=lambda args: effort_completions(context, args, model=session.model or _model_label(agent)),
            menu=effort_menu,
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
    commands.register(
        Command(
            name='keys', description='Manage saved API keys', handler=keys_command, during_turn=True, menu=keys_menu
        )
    )
    commands.register(
        Command(
            name='login',
            description=(
                'Sign in: openai-codex, github-copilot, or one a plugin adds; NAME@PROFILE adds another account'
            ),
            handler=models.login,
            complete=lambda args: login_names(models.logins()) if len(args) <= 1 else (),
            during_turn=True,
            menu=models.accounts_menu,
        )
    )
    commands.register(
        Command(
            name='accounts',
            description='Add, rename, reorder, and sign out of accounts; MODEL@* tries them all in order',
            handler=models.accounts,
            during_turn=True,
            menu=models.accounts_menu,
        )
    )
    set_ = Command(
        name='set',
        description='Change settings; no arguments opens the menu',
        handler=lambda args: set_command(context, args),
        complete=lambda args: set_completions(args, plugin_models=context.plugin_models()),
        during_turn=True,
        menu=lambda: settings_menu(context),
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
            menu=lambda: settings_menu(context, keys=('display.theme',), title='Theme'),
            args_during_turn=True,
        )
    )
    commands.register(
        Command(
            name='system_prompt',
            description='View the system prompt and edit your own instructions, sent after the built-in ones',
            handler=lambda args: system_prompt_command(context, args, history=lambda: session.messages),
            during_turn=True,
            menu=lambda: system_prompt_menu(context, history=lambda: session.messages),
        )
    )
    commands.register(
        Command(
            name='model',
            description=(
                'Select any model or fallback chain, or open the picker; also /model add [NAME], '
                '/model settings [NAME], and /model chains'
            ),
            handler=lambda args: model_command(context, args, console=console),
            complete=lambda args: model_completions(context, args),
            during_turn=True,
            during_turn_subcommands=MODEL_SUBCOMMANDS,
            menu=lambda: models_menu(context),
        )
    )
    # Deprecated spellings of `/model add`, `/model settings`, and `/model chains`, kept working for existing habits.
    commands.register(
        Command(
            name='add_model',
            description='Deprecated: use /model add',
            handler=lambda args: model_command(context, ['add', *args], console=console),
            complete=lambda args: model_completions(context, ['add', *args]),
            during_turn=True,
        )
    )
    commands.register(
        Command(
            name='model_settings',
            description='Deprecated: use /model settings',
            handler=lambda args: model_command(context, ['settings', *args], console=console),
            complete=lambda args: model_completions(context, ['settings', *args]),
            during_turn=True,
        )
    )
    commands.register(
        Command(
            name='chain',
            description='Alias of /model chains: fallback chains live in the /model picker',
            handler=lambda args: model_command(context, ['chains', *args], console=console),
            during_turn=True,
        )
    )
    commands.register(
        Command(name='help', description='Show commands', handler=lambda args: f'{commands.help(args)}\n{SHELL_HELP}')
    )

    def reset_screen() -> None:
        console.clear()
        # Forget the old output too, or a resize or the exit printout would show it again.
        if shell.terminal is not None:
            shell.terminal.transcript.clear()
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
            handler=lambda args: shell.update(args),
            menu=lambda: update_menu(
                context, updates, run=lambda text: shell.send_input(text), refusal=lambda: shell.restart_refusal()
            ),
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
    loader = SessionPlugins[DepsT](
        process=process.plugins,
        console=console,
        commands=commands,
        session_start=lambda: SessionStart(agent=agent, settings=context.settings, conversation_chosen=sessions.chosen),
        full_screen=screen.full,
        project=tuple(PluginSettings.model_validate(plugin.model_dump()) for plugin in project.plugins),
        conversation=session,
        session_id=lambda: shell.session_id,
        status=status,
        enabled=load_plugins,
        submit_prompt=lambda text: shell.submit_prompt(text),
        send_input=lambda text, images: shell.send_input(text, images=images),
        interrupt=lambda: shell.interrupts.cancel(exit_on_repeat=False),
        sessions=sessions,
        setting=context.setting,
        tasks=lambda: shell.tasks if shell.session.delegations is not None else None,
        workspace=lambda: Path(session.workspace),
        running=lambda: shell.turning,
        default_model=partial(_default_model, store=store, models=models, fallback=agent.model),
        default_model_name=partial(_default_model_name, store=store, fallback=agent.model),
        inputs=lambda: shell,
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
            menu=lambda: settings_menu(context, keys=('display.spinner',), title='Working animation'),
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
        interrupts=Interrupts(signals=signals),
        screen=screen,
        sessions=sessions,
        speculation=Speculation(context=context, console=console, repl_states=session.repl_states),
        session_settings=session_settings,
        spinners=spinners,
        updates=updates,
    )
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
    sessions.show_live = shell.show_live
    shell.session.claim = lambda conversation_id: process.live.claim(conversation_id, holder=shell.live_session)
    process.live.add(shell)
    # Headless, a session still edits prompts in a terminal console, but reads no lines from a pipe.
    if not headless or console.is_terminal:
        _show_in_terminal(shell, transcript=transcript, shows_others=not headless)
    return shell


def _show_in_terminal(
    shell: 'SessionRuntime[DepsT, OutputT]', *, transcript: TranscriptBuffer | None, shows_others: bool
) -> None:
    """Show `shell` in a new `Terminal`, keeping the output `transcript` retains, as across a reload.

    With `shows_others`, `/resume` of a conversation another live session holds shows that session there;
    `clai2 --web` keeps its terminal on the session it started with.
    """
    Terminal[DepsT, OutputT](
        console=shell.console,
        shows_others=shows_others,
        transcript=transcript if transcript is not None else TranscriptBuffer(),
        history=input_history(shell.process.store.path.with_name('input-history')),
    ).attach(shell)
    # Mutate retained state only after the rebuild has succeeded, so reload failures can roll back.
    if transcript is not None:
        TranscriptBuffer.rebind(transcript)


def web_serving(loader: SessionPlugins[DepsT]) -> WebHandoff | None:
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


class Inbox:
    """Input waiting for a session's read loop, readable so the images its markers name are kept.

    `SessionRuntime` now queues through `PromptInbox`, whose entries can be edited and removed;
    this class stays so code that imports it keeps working.
    """

    def __init__(self) -> None:
        self.texts = deque[str]()
        self._arrived = asyncio.Event()

    def put(self, text: str) -> None:
        self.texts.append(text)
        self._arrived.set()

    async def get(self) -> str:
        while not self.texts:
            self._arrived.clear()
            await self._arrived.wait()
        return self.texts.popleft()


@dataclass(kw_only=True)
class SessionRuntime(Generic[DepsT, OutputT]):
    """One live session and its prompt loop; one turn is one `TurnStart`, one agent run, one `TurnEnd`.

    While a `Terminal` shows it, it reads the terminal's editor, or its line prompt when the console is
    not a terminal; otherwise it reads only what `send_input` gives it.
    """

    process: ClaiProcess[DepsT]
    agent: AbstractAgent[DepsT, OutputT]
    session: Session[DepsT, OutputT]
    plugins: tuple[AgentCapability[DepsT], ...]
    loader: SessionPlugins[DepsT]
    commands: Commands
    console: Console
    context: CommandContext
    status: Status
    interrupts: Interrupts
    screen: Screen
    sessions: Sessions[DepsT, OutputT]
    speculation: Speculation
    session_settings: SessionSettings[DepsT, OutputT]
    spinners: Spinners
    updates: Updates
    terminal: 'Terminal[DepsT, OutputT] | None' = None
    """The terminal showing this session; `None` while it shows another, or in a CLAI without one."""
    images: ImageInput = field(default_factory=ImageInput)
    """Images pasted for the next prompt: the terminal's while it shows this session."""
    submitted: list[str] | None = None
    """Prompts the running `/command` submitted with `host.submit_prompt`; `None` while no command runs."""
    reload_requested: bool = False
    editor: LivePrompt | None = None
    """The terminal's editor, while the terminal shows this session and its console is a terminal."""
    forks: Forks[DepsT, OutputT] = field(init=False)
    tasks: Tasks = field(init=False)
    _mid_turn_commands: MemoryObjectSendStream[str] | None = field(default=None, init=False, repr=False)
    _steering: MemoryObjectSendStream[str] | None = field(default=None, init=False, repr=False)
    _identity_pending: bool = field(default=False, init=False)
    _inbox: PromptInbox = field(default_factory=PromptInbox, init=False, repr=False)
    """Input `send_input` entered while no editor reads keys, as in a shell driven only by a plugin."""
    turning: bool = field(default=False, init=False)
    """Whether a turn runs: from its `TurnStart` until its run ends, before plugins hear `TurnEnd`."""
    _handling: bool = field(default=False, init=False, repr=False)
    """Whether input, a turn or a command, is being handled."""
    _reading: CancelScope | None = field(default=None, init=False, repr=False)
    """Cancelled to read again from where input comes now, as when the terminal starts or stops showing it."""

    @property
    def prompt(self) -> PromptSession[str] | None:
        """The terminal's line prompt, while the terminal shows this session and its console is not a terminal."""
        return self.terminal.prompt if self.terminal is not None else None

    @property
    def transcript(self) -> TranscriptBuffer | None:
        """The output the terminal showing this session retains for redrawing."""
        return self.terminal.transcript if self.terminal is not None else None

    @property
    def live_session(self) -> LiveRuntime:
        """This session as `host.live` lists it."""
        return self.loader.live_session

    @property
    def queued(self) -> int:
        """How many inputs wait for the running turn or command to end."""
        return len(self._queued_prompts())

    def _queued_prompts(self) -> list[QueuedPrompt]:
        """What waits where input comes from now: the terminal's editor while it shows this session, else the inbox."""
        return self.editor.queued_prompts() if self.editor is not None else self._inbox.entries

    @property
    def queue(self) -> list[QueuedInput]:
        """The inputs waiting for the running turn or command to end, oldest first, as `host.live` lists them."""
        return [QueuedInput(id=entry.id, text=entry.text) for entry in self._queued_prompts()]

    def edit_queued(self, input_id: str, text: str) -> None:
        """Rewrite a waiting input in place, as Enter on a recalled queued prompt does; blank text is refused."""
        text = expand_bare_command(text.strip())
        if not text:
            raise ValueError('Type a prompt or a /command.')
        entry = find_queued(self._queued_prompts(), input_id)
        if self.editor is not None:
            self.editor.rewrite(entry, text)  # which tells `host.live` through its `queue_changed`
        else:
            entry.text = text
            self.live_session.changed()

    def remove_queued(self, input_id: str) -> None:
        """Drop a waiting input, as clearing a recalled queued prompt before Enter does."""
        entry = find_queued(self._queued_prompts(), input_id)
        if self.editor is not None:
            self.editor.discard(entry)
        else:
            self._inbox.remove(entry)
            self.live_session.changed()

    @property
    def idle(self) -> bool:
        """Whether nothing runs or waits: no turn, command, queued input, background fork, or delegated task."""
        return not (self._handling or self.session.running or self.queued or self.background_running())

    async def serve(self, *, resume: str | None, ready: Callable[[], None]) -> SessionEndReason:
        """Run as a session with no terminal of its own, as one the browser opened, until it ends or is cancelled.

        Opens the agent, restores the conversation `resume`, loads the plugins, calls `ready`, then
        takes input. The plugins end on the way out, with `'exit'` when cancelled, as idle closing does.
        """
        reason: SessionEndReason = 'error'
        try:
            async with self.agent:
                if resume is not None and (notice := await self.sessions.start(resume=resume)):
                    self.console.print(notice, markup=False)
                await self.loader.load_all()
                if resume is not None:
                    self.session.record_resumed()
                for name in ('reload', 'update'):
                    # Both restart CLAI, which only the session the terminal shows, or CLAI started with, can do.
                    restart = next(command for command in self.commands if command.name == name)
                    self.commands.unregister([name])
                    self.commands.register(
                        Command(
                            name=name,
                            description=_RESTARTS,
                            handler=lambda args, restart=restart: (
                                restart.handler if self.terminal is not None else _refuse_restart
                            )(args),
                        )
                    )
                ready()
                reason = await self.run()
        except get_cancelled_exc_class():
            reason = 'exit'
            raise
        finally:
            with CancelScope(shield=True):
                await self.close(reason)
        return reason

    async def close(self, reason: SessionEndReason) -> None:
        """End this session's plugins and leave the process's live sessions."""
        self.process.live.leaving(self)
        try:
            await self.loader.close(reason)
        finally:
            self.process.live.remove(self)

    @contextmanager
    def _handling_input(self) -> Generator[None]:
        self._handling = True
        self.live_session.changed()
        try:
            yield
        finally:
            self._handling = False
            self.live_session.changed()

    def handoff(self) -> WebHandoff | None:
        """The web UI to serve again once `/update` restarts CLAI as this session; `None` when it does not, or none runs."""
        return None if self.updates.relaunch is None else web_serving(self.loader)

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

    def send_input(self, text: str, *, images: Sequence[BinaryContent] = ()) -> None:
        """Enter `text` as Enter at the prompt does: trimmed, bare `clear` as `/clear`, blank ignored.

        A `during_turn` command opens over a running turn; other input queues in the editor, or, while no
        editor reads keys yet or at all, in the inbox the read loop takes it from. A line-mode prompt,
        used when CLAI's output is not a terminal, reads only its own input, so it takes none.

        `images` are attached as pasted ones are, by markers in the text that the turn resolves.
        """
        if self.prompt is not None and self.editor is None:
            raise RuntimeError('This CLAI reads its prompts line by line, as its output is not a terminal.')
        text = expand_bare_command(text.strip())
        if images:
            if input_kind(text) != 'prompt':
                raise ValueError('Images go with a prompt, not with a /command or ! line.')
            if sum(len(image.data) for image in images) > image_codec.MAX_PENDING_BYTES:
                raise ValueError('These images exceed the 32 MiB limit; send fewer at a time.')
            try:
                markers = self.images.attach(images)
            except ValueError as exc:
                # The sender may not see the images waiting here, such as a terminal draft's.
                raise ValueError(
                    "Images waiting in this session's prompt leave no room for these (32 MiB limit)."
                ) from exc
            text = f'{text} {markers}'.strip()
        if not text or self.run_now(text):
            return
        if self.editor is not None:
            self.editor.submit(text)
        else:
            self._inbox.put(QueuedPrompt(text))
        self.live_session.changed()

    def request_reload(self, args: list[str]) -> str:
        if args:
            raise ValueError('Usage: /reload')
        self._refuse_restart_while_others_run()
        self.reload_requested = True
        return 'Reloading CLAI2...'

    async def update(self, args: list[str]) -> str:
        """`/update`, refused while another live session runs a turn, which the restart would end."""
        self._refuse_restart_while_others_run()
        return await self.updates.command(args)

    def _refuse_restart_while_others_run(self) -> None:
        if (refusal := self.restart_refusal()) is not None:
            raise ValueError(refusal)

    def restart_refusal(self) -> str | None:
        """Why `/reload` and `/update` cannot restart CLAI from this session now; `None` when they can."""
        if self.terminal is None and self.process.live.focused not in (None, self.live_session.id):
            # The session CLAI started with, while the terminal shows another: the restart starts from that one.
            return _RESTART_ELSEWHERE
        home = self.terminal.home if self.terminal is not None else None
        if home is not None and home is not self and self.session.workspace != home.session.workspace:
            # The restart resumes this conversation where CLAI started, which works only in its own folder.
            return (
                f'CLAI restarts in {home.session.workspace}, where it started, and this session works in '
                f'{self.session.workspace}; restart it from a session there.'
            )
        running = sum(
            1 for session in self.process.live.listing() if session.running and session.id != self.live_session.id
        )
        if running > 1:
            return f'{running} other sessions are running turns; try again when they finish.'
        if running:
            return '1 other session is running a turn; try again when it finishes.'
        return None

    async def run(self) -> SessionEndReason:
        """Take input until this session ends; with a terminal, until the session the terminal shows ends."""
        if self.terminal is not None:
            return await self.terminal.drive(self)
        return await self.loop()

    async def loop(self) -> SessionEndReason:
        """Take input until this session ends, telling the terminal when it shows this session."""
        reason: SessionEndReason = 'error'
        try:
            # Commands and turns run here: a process-scoped plugin's host follows this session meanwhile.
            with handling(self.loader.live_session):
                async with self.process.tasks.opened(view=self.tasks) as owner:
                    if isinstance(self.agent, StockAgent):
                        self.session.delegations = owner
                    # Reports a task left while no CLAI ran are known once the owner opened.
                    if self.editor is not None and owner.reports(conversation_id=self.session.summary.id):
                        self.editor.wake()
                    reason = await self._read_loop()
                    return reason
        finally:
            self.session.delegations = None
            await self.forks.close()
            if self.terminal is not None:
                self.terminal.ended(self, reason)

    def shown_in(self, terminal: 'Terminal[DepsT, OutputT] | None') -> None:
        """Read the input of `terminal`, which shows this session now, and take its Ctrl-C; `None` stops that.

        The terminal hands its console output over itself.
        """
        self.terminal = terminal
        previous, self.images = self.images, terminal.images if terminal is not None else ImageInput()
        # Queued prompts, and the one rejected for retry, keep their images as the terminal moves.
        self.images.pending.update(previous.pending)
        if previous.retry_text:
            # The terminal's goes with the session leaving it, and the one shown brings its own.
            self.images.retry_text, previous.retry_text = previous.retry_text, ''
        self.interrupts.focus(signals=terminal is not None)
        if terminal is not None:
            self.process.live.focused = self.live_session.id
        self.bind(terminal.editor if terminal is not None else None)
        # Both now hold every image; each keeps those its own texts use, so none counts twice against the limit.
        self._retain_images()
        self.live_session.changed()

    def _retain_images(self) -> None:
        """Let go of images that neither the draft nor a queued prompt of this session uses."""
        self.images.retain(
            [self.editor.buffer.text, *self.editor.queued_messages]
            if self.editor is not None
            else [entry.text for entry in self._inbox.entries]
        )

    def bind(self, editor: LivePrompt | None) -> None:
        """Read from `editor`, the open editor of the terminal showing this session; `None` stops that."""
        if self.editor is not None and editor is None:
            # Prompts queued for this session stay its own while the terminal shows another.
            for entry in self.editor.take_queued():
                self._inbox.put(entry)
        self.editor = editor
        self.screen.editor = editor.suspended if editor is not None else None
        self.tasks.wake = editor.wake if editor is not None else None
        if editor is not None:
            # Input a plugin entered while the editor read for another session, or did not exist yet.
            for entry in self._inbox.take():
                editor.submit(entry)
            if self.tasks.owner.reports(conversation_id=self.session.summary.id):
                editor.wake()
        if self._reading is not None:
            self._reading.cancel()

    async def show_live(self, conversation_id: str) -> str | None:
        """Show in the terminal the other live session holding `conversation_id`; `None` when none does.

        As `/resume` does, the terminal then shows that session's history; this one stays live.
        """
        other = self.process.live.runtime(conversation_id)
        if (
            self.terminal is None
            or not self.terminal.shows_others
            or other is None
            or other is self
            or not isinstance(other, SessionRuntime)
        ):
            return None
        # Every session of a process runs the same agent, so it has the same types.
        await self.terminal.show(cast('SessionRuntime[DepsT, OutputT]', other))
        return ''

    async def _next_input(self) -> str:
        """The next input from where it comes now, which moves when the terminal starts or stops showing this session."""
        while True:
            with CancelScope() as self._reading:
                try:
                    if self.editor is not None:
                        return await self.editor.read()
                    if self.prompt is not None:
                        return expand_bare_command((await self.prompt.prompt_async('> ')).strip())
                    return (await self._inbox.get()).strip()
                finally:
                    self._reading = None

    def steer(self, text: str) -> bool:
        """Enqueue steering in core and hand transcript feedback to the active turn.

        `False` when no run takes it, and for a `/command` (bare `clear` included) or `!` shell line, which never steer.
        """
        if not can_steer(expand_bare_command(text)):
            return False
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

    def show_model(self) -> None:
        """Show the model the next prompt runs on in the status row, forgetting the last one's window if it changed."""
        model = self.session.model or _model_label(self.agent)
        if model != self.status.model:
            self.status.context_window = None
            self.status.context_alert = False
        self.status.model = model

    def _show_status_segments(self) -> None:
        """Paint the status segments of the plugins loaded now."""
        self.status.status_segments = (*self.loader.status_segments(), self.updates.segment, self._live_segment)

    def _live_segment(self) -> str:
        """How many sessions are live in this CLAI, shown in the terminal once the browser opened others."""
        live = len(self.process.live.listing())
        return f'{live} live' if self.terminal is not None and live > 1 else ''

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
            self._retain_images()
            try:
                self.show_model()
                self.status.workspace = self.session.workspace
                self._show_status_segments()
                text = await self._next_input()
            except PromptRewind:
                await self._rewind()
                continue
            except PromptWakeup:
                if not self.tasks.owner.reports(conversation_id=self.session.summary.id):
                    continue
                with self._handling_input():
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
            with self._handling_input():
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
            # Before waiting for the terminal: once the prompt left the queue, only this holds its images.
            text, images = self.images.resolve(text)
        except ValueError as exc:
            self.console.print(str(exc), style=theme.color(theme.ERROR), markup=False)
            return False
        try:
            async with self.forks.busy():
                return await self._turn(text, images)
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

    async def _turn(self, text: str | None, images: Sequence[BinaryContent] = ()) -> bool:
        """Run a turn on `text` and its resolved `images`, or an automated one, as a task's report, on `None`."""
        automated = text is None
        start = TurnStart(text=text or '')
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
        self.turning = True
        self.live_session.changed()
        try:
            return await self._run_turn(start, images=images, headless=headless, automated=automated)
        finally:
            self.turning = False
            self.live_session.changed()

    async def _run_turn(
        self, start: TurnStart, *, images: Sequence[BinaryContent], headless: bool, automated: bool
    ) -> TurnEnd:
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
    # A session the terminal does not show yet prints nowhere: no footer of its own may draw over the editor later.
    status_line = StatusLine(console, status, enabled=screen.editor is None and console.is_terminal, spinner=spinner)

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
        total = session_usage(session.messages).total
        status.cost = total.cost
        status.cost_is_lower_bound = bool(total.unmeasured_requests)
        session.on_context_usage = None
        session.on_context_window = None
        await renderer.finish()
