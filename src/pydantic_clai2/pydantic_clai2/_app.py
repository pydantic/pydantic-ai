"""Interactive terminal shell around a capability-independent session."""

import asyncio
from collections.abc import Sequence
from dataclasses import replace
from typing import TypeVar

from prompt_toolkit import PromptSession
from pydantic_ai import Agent, AgentStreamEvent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.usage import UsageLimits
from pydantic_ai_harness.coder import Coder
from rich.console import Console

from ._branding import print_banner
from ._completion_adapter import COMPLETION_STYLE, PromptCompleter
from ._rendering import StreamRenderer
from ._session import Session
from .auth import CodexAuth
from .command_context import CommandContext, CommandProvider
from .commands import Command, Commands, config_command, config_completions, plugins_command, set_completions
from .config import Settings
from .input_history import input_history
from .interrupts import Interrupts
from .settings_store import SettingsStore
from .status import Status, StatusLine

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')


def create_agent(model: str | None = None) -> Agent[None, str]:
    """Build the default coding agent; custom agents need not use `Coder`."""
    return Agent(model, capabilities=[Coder(unrestricted_filesystem=True)])


async def chat(
    agent: AbstractAgent[DepsT, OutputT],
    *,
    deps: DepsT,
    plugins: Sequence[AbstractCapability[DepsT]] = (),
    usage_limits: UsageLimits | None = None,
    console: Console | None = None,
    settings: Settings | None = None,
    store: SettingsStore | None = None,
) -> None:
    """Start an asyncio terminal conversation with a caller-supplied agent.

    Ctrl-C cancels the current turn or clears input; Ctrl-D and `/exit` quit.
    Failed and cancelled turns are not added to the retained history.
    """
    console = console or Console()
    print_banner(console)
    console.print('/new clears history; /exit quits. Ctrl-C interrupts a turn.', style='dim')
    settings = settings or Settings(model=None)
    store = store or SettingsStore()
    session = Session(agent, deps=deps, plugins=plugins, usage_limits=usage_limits)
    session.model = settings.model
    auth = CodexAuth(console)
    session.resolve_model = lambda name: auth.model(name) if name.startswith('openai-codex:') else name
    if session.model is None and agent.model is None:
        console.print('Choose a model with /set model <Tab>.', style='cyan')

    def apply_setting(key: str, updated: Settings) -> None:
        if key == 'model':
            session.model = updated.model
        elif key == 'run.request_limit':
            session.usage_limits = replace(session.usage_limits or UsageLimits(), request_limit=updated.request_limit)

    context = CommandContext(settings=settings, store=store, clear_history=session.clear, apply_setting=apply_setting)

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
            description='View or change settings; Tab completes names and values',
            handler=context.set_setting,
            complete=set_completions,
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
    commands.register(Command(name='exit', description='Quit CLAI', handler=lambda _: 'Goodbye.'))
    commands.register(
        Command(
            name='config',
            description='show|get|set|reset settings',
            handler=lambda args: config_command(store, args),
            complete=config_completions,
        )
    )
    commands.register(
        Command(
            name='plugins',
            description='list|add|enable|disable plugins',
            handler=lambda args: plugins_command(store, args),
            complete=lambda args: (
                ('list', 'add', 'enable', 'disable') if len(args) <= 1 else (p.id for p in store.plugins())
            ),
        )
    )
    _register_plugin_commands(commands, plugins, context)
    status = Status()
    prompt = PromptSession[str](
        history=input_history(store.path.with_name('input-history')),
        completer=PromptCompleter(commands),
        complete_while_typing=True,
        style=COMPLETION_STYLE,
        reserve_space_for_menu=6,
        bottom_toolbar=lambda: status.text(),
    )
    interrupts = Interrupts()
    async with agent:
        while True:
            try:
                status.model = session.model or _model_label(agent)
                text = (await prompt.prompt_async('You > ')).strip()
            except KeyboardInterrupt:
                if interrupts.press():
                    return
                console.print('Input cleared. Press Ctrl-C again within 2 seconds to exit.', style='dim')
                continue
            except EOFError:
                return
            if not text:
                continue
            console.print()
            if text.startswith('/'):
                await interrupts.run(_execute_command(commands, text, console=console, status=status))
                if text == '/exit' or interrupts.exit_requested:
                    return
                continue
            if session.model is None and agent.model is None:
                console.print('Choose a model first: /set model <Tab>', style='yellow')
                continue
            completed = await interrupts.run(
                _run_prompt(session, text, console=console, settings=context.settings, status=status)
            )
            _report_interrupt(completed, console)
            if interrupts.exit_requested:
                return


def _register_plugin_commands(
    commands: Commands, plugins: Sequence[AbstractCapability[DepsT]], context: CommandContext
) -> None:
    for plugin in plugins:
        if isinstance(plugin, CommandProvider):
            commands.register_many(plugin.get_commands(context))


def _report_interrupt(completed: bool, console: Console) -> None:
    if not completed:
        console.print('Turn cancelled. Press Ctrl-C again within 2 seconds to exit.', style='dim')
        console.print()


async def _execute_command(commands: Commands, text: str, *, console: Console, status: Status) -> None:
    try:
        console.print(await commands.execute_async(text), markup=False)
    except Exception as exc:  # noqa: BLE001 -- command failures must not exit the interactive shell.
        console.print(str(exc), style='red', markup=False)
    console.print()
    _reset_status(text, status)


def _reset_status(command: str, status: Status) -> None:
    if command == '/new':
        status.context_tokens = None
        status.output_tokens = None
        status.streamed_chars = 0


def _model_label(agent: AbstractAgent[DepsT, OutputT]) -> str:
    model = agent.model
    if isinstance(model, str):  # pragma: no cover -- concrete Agent resolves string models before chat.
        return model
    return model.model_name if model else 'agent default'


async def _run_prompt(
    session: Session[DepsT, OutputT], text: str, *, console: Console, settings: Settings, status: Status
) -> None:
    renderer = StreamRenderer(
        console,
        stop_loading=lambda: None,
        show_thinking=settings.thinking,
        smooth_seconds=settings.smooth_seconds,
        shell_lines=settings.shell_lines,
        grep_lines=settings.grep_lines,
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
    except asyncio.CancelledError:
        await renderer.abort()
        raise
    except Exception as exc:  # noqa: BLE001 -- interactive boundary reports plugin/provider failures.
        await renderer.finish()
        console.print(f'{type(exc).__name__}: {exc}', style='red', markup=False)
        console.print('Turn not saved. External tool side effects may already have occurred.', style='dim')
        console.print()
    finally:
        status.activity = 'ready'
        session.on_context_usage = None
        await renderer.finish()
