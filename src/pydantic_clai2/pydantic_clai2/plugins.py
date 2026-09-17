"""Everything a plugin can register, recorded on one host per plugin."""

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, Never, Protocol, TypeVar, get_args, overload

from pydantic import BaseModel, JsonValue
from pydantic_ai import AgentRunResult, AgentStreamEvent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability, Hooks
from pydantic_ai.capabilities.hooks import (
    AfterModelRequestHookFunc,
    AfterNodeRunHookFunc,
    AfterOutputProcessHookFunc,
    AfterOutputValidateHookFunc,
    AfterRunHookFunc,
    AfterToolExecuteHookFunc,
    AfterToolValidateHookFunc,
    BeforeModelRequestHookFunc,
    BeforeNodeRunHookFunc,
    BeforeOutputProcessHookFunc,
    BeforeOutputValidateHookFunc,
    BeforeRunHookFunc,
    BeforeToolExecuteHookFunc,
    BeforeToolValidateHookFunc,
    HandleDeferredToolCallsHookFunc,
    OnEventHookFunc,
    OnModelRequestErrorHookFunc,
    OnNodeRunErrorHookFunc,
    OnOutputProcessErrorHookFunc,
    OnOutputValidateErrorHookFunc,
    OnRunErrorHookFunc,
    OnToolExecuteErrorHookFunc,
    OnToolValidateErrorHookFunc,
    PrepareOutputToolsHookFunc,
    PrepareToolsHookFunc,
    WrapModelRequestHookFunc,
    WrapNodeRunHookFunc,
    WrapOutputProcessHookFunc,
    WrapOutputValidateHookFunc,
    WrapRunEventStreamHookFunc,
    WrapRunHookFunc,
    WrapToolExecuteHookFunc,
    WrapToolValidateHookFunc,
)
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from rich.console import Console, RenderableType
from typing_extensions import TypeVar as DefaultTypeVar

from .commands import Commands
from .config import Settings
from .status import Status

DepsT = DefaultTypeVar('DepsT', default=None)
EventT = TypeVar('EventT', bound=AgentStreamEvent)
ModelT = TypeVar('ModelT', bound=BaseModel)

SessionEndReason = Literal['exit', 'eof', 'error']
TurnOutcome = Literal['completed', 'failed', 'cancelled']


class Conversation(Protocol):
    """The retained history as a plugin sees it. The shell's `Session` is one; `Transcript` is the plain one."""

    @property
    def messages(self) -> list[ModelMessage]:
        """A snapshot of the retained messages."""
        ...

    def replace_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Swap the retained history, as `/compact` does after summarising it."""
        ...

    async def resolved_model(self) -> Model | str | None:
        """The model the next run uses; `None` when nothing has been chosen yet."""
        ...


class Transcript:
    """An in-memory `Conversation` for hosts built outside the shell, such as in a plugin's tests."""

    def __init__(self, *, messages: Sequence[ModelMessage] = (), model: Model | str | None = None) -> None:
        """Start with `messages` retained and `model` as what `resolved_model` reports."""
        self._messages = list(messages)
        self.model = model

    @property
    def messages(self) -> list[ModelMessage]:
        """A snapshot of the retained messages, like `Session.messages`; edit through `replace_messages`."""
        return list(self._messages)

    def replace_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Swap the retained history."""
        self._messages = list(messages)

    async def resolved_model(self) -> Model | str | None:
        """The `model` given at construction."""
        return self.model


@dataclass(kw_only=True)
class SessionStart:
    """CLAI is ready for prompts. Each plugin receives this once, when it loads."""

    agent: AbstractAgent[Never, object]
    settings: Settings


@dataclass(kw_only=True)
class SessionEnd:
    """CLAI is quitting, or this plugin is being unloaded."""

    reason: SessionEndReason


@dataclass(kw_only=True)
class TurnStart:
    """A prompt was submitted. Edit `text` or call `cancel()` before the agent sees it."""

    text: str
    cancelled: bool = False
    cancel_reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        """Stop this turn before it starts; the reason is shown to the user."""
        self.cancelled = True
        self.cancel_reason = reason


@dataclass(kw_only=True)
class TurnEnd:
    """The turn finished. `result` is set when it completed, `error` when it failed."""

    text: str
    outcome: TurnOutcome
    result: AgentRunResult[object] | None = None
    error: BaseException | None = None


HostEvent = SessionStart | SessionEnd | TurnStart | TurnEnd
HostEventT = TypeVar('HostEventT', bound=HostEvent)
HostHandler = Callable[[HostEventT], Awaitable[None]]
Renderer = Callable[[EventT], RenderableType | None]

HostHookName = Literal['session_start', 'session_end', 'turn_start', 'turn_end']
CoreHookName = Literal[
    'before_run',
    'after_run',
    'run',
    'run_error',
    'before_node_run',
    'after_node_run',
    'node_run',
    'node_run_error',
    'run_event_stream',
    'event',
    'before_model_request',
    'after_model_request',
    'model_request',
    'model_request_error',
    'prepare_tools',
    'prepare_output_tools',
    'before_tool_validate',
    'after_tool_validate',
    'tool_validate',
    'tool_validate_error',
    'before_tool_execute',
    'after_tool_execute',
    'tool_execute',
    'tool_execute_error',
    'before_output_validate',
    'after_output_validate',
    'output_validate',
    'output_validate_error',
    'before_output_process',
    'after_output_process',
    'output_process',
    'output_process_error',
    'deferred_tool_calls',
]
HOST_HOOKS: dict[str, type[HostEvent]] = {
    'session_start': SessionStart,
    'session_end': SessionEnd,
    'turn_start': TurnStart,
    'turn_end': TurnEnd,
}
CORE_HOOK_NAMES: frozenset[str] = frozenset(get_args(CoreHookName))


class PluginHost(Generic[DepsT]):
    """The one object a plugin talks to. Discarding the host unloads the plugin."""

    def __init__(
        self,
        *,
        name: str,
        console: Console,
        settings: dict[str, JsonValue],
        conversation: Conversation | None = None,
        status: Status | None = None,
    ) -> None:
        """`settings` is the raw JSON from `plugins add`; validate it with `settings(Model)`.

        The shell passes its own `conversation` and `status`; a host built elsewhere gets a
        `Transcript` and a detached status row, so a plugin needs no special case for either.
        """
        self.name = name
        self.console = console
        self.conversation: Conversation = conversation if conversation is not None else Transcript()
        self.status = status if status is not None else Status()
        self.commands = Commands()
        self._settings = settings
        self._hooks: Hooks[DepsT] = Hooks()
        self._hooks_used = False
        self._capabilities: list[AgentCapability[DepsT]] = []
        self._handlers: list[Callable[[HostEvent], Awaitable[None]]] = []
        self._renderers: list[Renderer[AgentStreamEvent]] = []

    @property
    def capabilities(self) -> list[AgentCapability[DepsT]]:
        """Capabilities to bind on every run while this plugin is loaded."""
        return [*self._capabilities, *([self._hooks] if self._hooks_used else [])]

    @property
    def handlers(self) -> list[Callable[[HostEvent], Awaitable[None]]]:
        """Host-hook handlers; each ignores events it was not registered for."""
        return list(self._handlers)

    @property
    def renderers(self) -> list[Renderer[AgentStreamEvent]]:
        """Renderers; each returns `None` for events it was not registered for."""
        return list(self._renderers)

    def summary(self) -> str:
        """One line for the `/plugins` menu."""
        return (
            f'{len(list(self.commands))} commands, {len(self._handlers)} hooks, '
            f'{len(self._capabilities)} capabilities, {len(self._renderers)} renderers'
        )

    def settings(self, model: type[ModelT], /) -> ModelT:
        """Validate the JSON given to `plugins add` against the plugin's own model."""
        return model.model_validate(self._settings)

    def add(self, capability: AgentCapability[DepsT], /) -> None:
        """Give the agent tools, instructions, or a capability chosen per run."""
        self._capabilities.append(capability)

    def render(self, event_type: type[EventT], /) -> Callable[[Renderer[EventT]], Renderer[EventT]]:
        """Draw an event yourself; return `None` to fall back to the default display."""

        def decorator(func: Renderer[EventT]) -> Renderer[EventT]:
            def erased(event: AgentStreamEvent) -> RenderableType | None:
                return func(event) if isinstance(event, event_type) else None

            self._renderers.append(erased)
            return func

        return decorator

    @overload
    def on(
        self, name: Literal['session_start'], /
    ) -> Callable[[HostHandler[SessionStart]], HostHandler[SessionStart]]: ...
    @overload
    def on(self, name: Literal['session_end'], /) -> Callable[[HostHandler[SessionEnd]], HostHandler[SessionEnd]]: ...
    @overload
    def on(self, name: Literal['turn_start'], /) -> Callable[[HostHandler[TurnStart]], HostHandler[TurnStart]]: ...
    @overload
    def on(self, name: Literal['turn_end'], /) -> Callable[[HostHandler[TurnEnd]], HostHandler[TurnEnd]]: ...
    @overload
    def on(self, name: Literal['before_run'], /) -> Callable[[BeforeRunHookFunc], BeforeRunHookFunc]: ...
    @overload
    def on(self, name: Literal['after_run'], /) -> Callable[[AfterRunHookFunc], AfterRunHookFunc]: ...
    @overload
    def on(self, name: Literal['run'], /) -> Callable[[WrapRunHookFunc], WrapRunHookFunc]: ...
    @overload
    def on(self, name: Literal['run_error'], /) -> Callable[[OnRunErrorHookFunc], OnRunErrorHookFunc]: ...
    @overload
    def on(self, name: Literal['before_node_run'], /) -> Callable[[BeforeNodeRunHookFunc], BeforeNodeRunHookFunc]: ...
    @overload
    def on(self, name: Literal['after_node_run'], /) -> Callable[[AfterNodeRunHookFunc], AfterNodeRunHookFunc]: ...
    @overload
    def on(self, name: Literal['node_run'], /) -> Callable[[WrapNodeRunHookFunc], WrapNodeRunHookFunc]: ...
    @overload
    def on(self, name: Literal['node_run_error'], /) -> Callable[[OnNodeRunErrorHookFunc], OnNodeRunErrorHookFunc]: ...
    @overload
    def on(
        self, name: Literal['run_event_stream'], /
    ) -> Callable[[WrapRunEventStreamHookFunc], WrapRunEventStreamHookFunc]: ...
    @overload
    def on(
        self, name: Literal['event'], /
    ) -> Callable[[OnEventHookFunc[AgentStreamEvent]], OnEventHookFunc[AgentStreamEvent]]: ...
    @overload
    def on(
        self, name: Literal['before_model_request'], /
    ) -> Callable[[BeforeModelRequestHookFunc], BeforeModelRequestHookFunc]: ...
    @overload
    def on(
        self, name: Literal['after_model_request'], /
    ) -> Callable[[AfterModelRequestHookFunc], AfterModelRequestHookFunc]: ...
    @overload
    def on(
        self, name: Literal['model_request'], /
    ) -> Callable[[WrapModelRequestHookFunc], WrapModelRequestHookFunc]: ...
    @overload
    def on(
        self, name: Literal['model_request_error'], /
    ) -> Callable[[OnModelRequestErrorHookFunc], OnModelRequestErrorHookFunc]: ...
    @overload
    def on(self, name: Literal['prepare_tools'], /) -> Callable[[PrepareToolsHookFunc], PrepareToolsHookFunc]: ...
    @overload
    def on(
        self, name: Literal['prepare_output_tools'], /
    ) -> Callable[[PrepareOutputToolsHookFunc], PrepareOutputToolsHookFunc]: ...
    @overload
    def on(
        self, name: Literal['before_tool_validate'], /
    ) -> Callable[[BeforeToolValidateHookFunc], BeforeToolValidateHookFunc]: ...
    @overload
    def on(
        self, name: Literal['after_tool_validate'], /
    ) -> Callable[[AfterToolValidateHookFunc], AfterToolValidateHookFunc]: ...
    @overload
    def on(
        self, name: Literal['tool_validate'], /
    ) -> Callable[[WrapToolValidateHookFunc], WrapToolValidateHookFunc]: ...
    @overload
    def on(
        self, name: Literal['tool_validate_error'], /
    ) -> Callable[[OnToolValidateErrorHookFunc], OnToolValidateErrorHookFunc]: ...
    @overload
    def on(
        self, name: Literal['before_tool_execute'], /
    ) -> Callable[[BeforeToolExecuteHookFunc], BeforeToolExecuteHookFunc]: ...
    @overload
    def on(
        self, name: Literal['after_tool_execute'], /
    ) -> Callable[[AfterToolExecuteHookFunc], AfterToolExecuteHookFunc]: ...
    @overload
    def on(self, name: Literal['tool_execute'], /) -> Callable[[WrapToolExecuteHookFunc], WrapToolExecuteHookFunc]: ...
    @overload
    def on(
        self, name: Literal['tool_execute_error'], /
    ) -> Callable[[OnToolExecuteErrorHookFunc], OnToolExecuteErrorHookFunc]: ...
    @overload
    def on(
        self, name: Literal['before_output_validate'], /
    ) -> Callable[[BeforeOutputValidateHookFunc], BeforeOutputValidateHookFunc]: ...
    @overload
    def on(
        self, name: Literal['after_output_validate'], /
    ) -> Callable[[AfterOutputValidateHookFunc], AfterOutputValidateHookFunc]: ...
    @overload
    def on(
        self, name: Literal['output_validate'], /
    ) -> Callable[[WrapOutputValidateHookFunc], WrapOutputValidateHookFunc]: ...
    @overload
    def on(
        self, name: Literal['output_validate_error'], /
    ) -> Callable[[OnOutputValidateErrorHookFunc], OnOutputValidateErrorHookFunc]: ...
    @overload
    def on(
        self, name: Literal['before_output_process'], /
    ) -> Callable[[BeforeOutputProcessHookFunc], BeforeOutputProcessHookFunc]: ...
    @overload
    def on(
        self, name: Literal['after_output_process'], /
    ) -> Callable[[AfterOutputProcessHookFunc], AfterOutputProcessHookFunc]: ...
    @overload
    def on(
        self, name: Literal['output_process'], /
    ) -> Callable[[WrapOutputProcessHookFunc], WrapOutputProcessHookFunc]: ...
    @overload
    def on(
        self, name: Literal['output_process_error'], /
    ) -> Callable[[OnOutputProcessErrorHookFunc], OnOutputProcessErrorHookFunc]: ...
    @overload
    def on(
        self, name: Literal['deferred_tool_calls'], /
    ) -> Callable[[HandleDeferredToolCallsHookFunc], HandleDeferredToolCallsHookFunc]: ...
    @overload
    def on(self, name: type[EventT], /) -> Callable[[OnEventHookFunc[EventT]], OnEventHookFunc[EventT]]: ...

    def on(self, name: str | type[AgentStreamEvent], /) -> object:
        """Register a handler for a named moment or a typed event. Use as a decorator."""
        if isinstance(name, type):
            self._hooks_used = True
            return self._hooks.on.event(name)
        host_event = HOST_HOOKS.get(name)
        if host_event is not None:
            return self._host_decorator(host_event)
        if name not in CORE_HOOK_NAMES:
            raise ValueError(f'Unknown hook {name!r}. See PLUGINS.md for the list.')
        self._hooks_used = True
        return getattr(self._hooks.on, name)

    def _host_decorator(
        self, event_type: type[HostEventT]
    ) -> Callable[[HostHandler[HostEventT]], HostHandler[HostEventT]]:
        def decorator(func: HostHandler[HostEventT]) -> HostHandler[HostEventT]:
            async def erased(event: HostEvent) -> None:
                if isinstance(event, event_type):
                    await func(event)

            self._handlers.append(erased)
            return func

        return decorator
