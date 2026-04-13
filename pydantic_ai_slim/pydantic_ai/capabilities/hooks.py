"""Hooks capability for decorator-based hook registration.

Provides the [`Hooks`][pydantic_ai.capabilities.Hooks] class as an ergonomic
alternative to subclassing [`AbstractCapability`][pydantic_ai.capabilities.AbstractCapability]
for registering hook functions.

Hook functions are registered via the `hooks.on` namespace::

    hooks = Hooks()

    @hooks.on.before_model_request
    async def log_request(ctx, request_context):
        print(f'Request: {request_context}')
        return request_context

    agent = Agent('openai:gpt-5', capabilities=[hooks])
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, overload

import anyio
from pydantic import ValidationError

from pydantic_ai.exceptions import ModelRetry
from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from .abstract import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
    RawToolArgs,
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapNodeRunHandler,
    WrapRunHandler,
    WrapToolExecuteHandler,
    WrapToolValidateHandler,
)

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.run import AgentRunResult

_FuncT = TypeVar('_FuncT', bound=Callable[..., Any])


# --- Timeout exception ---


class HookTimeoutError(TimeoutError):
    """Raised when a hook function exceeds its configured timeout."""

    def __init__(self, hook_name: str, func_name: str, timeout: float):
        self.hook_name = hook_name
        self.func_name = func_name
        self.timeout = timeout
        super().__init__(f'Hook {hook_name!r} function {func_name!r} timed out after {timeout}s')


# --- Hook entries ---


@dataclass
class _HookEntry(Generic[_FuncT]):
    """A registered hook function with optional timeout."""

    func: _FuncT
    timeout: float | None = None


@dataclass
class _ToolHookEntry(_HookEntry[_FuncT]):
    """A registered tool hook function with optional tools filter and timeout."""

    tools: frozenset[str] | None = None


# fmt: off
# --- Hook function protocols ---
# These define the exact signatures users must implement for each hook type.
# Both sync and async functions are accepted (sync auto-wrapped at runtime).


class BeforeRunHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.before_run` hook functions."""
    def __call__(self, ctx: RunContext[Any], /) -> None | Awaitable[None]: ...

class AfterRunHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.after_run` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, result: AgentRunResult[Any]) -> AgentRunResult[Any] | Awaitable[AgentRunResult[Any]]: ...

class WrapRunHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.wrap_run` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, handler: WrapRunHandler) -> AgentRunResult[Any] | Awaitable[AgentRunResult[Any]]: ...

class OnRunErrorHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.on_run_error` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, error: BaseException) -> AgentRunResult[Any] | Awaitable[AgentRunResult[Any]]: ...

class BeforeNodeRunHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.before_node_run` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, node: AgentNode[Any]) -> AgentNode[Any] | Awaitable[AgentNode[Any]]: ...

class AfterNodeRunHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.after_node_run` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, node: AgentNode[Any], result: NodeResult[Any]) -> NodeResult[Any] | Awaitable[NodeResult[Any]]: ...

class WrapNodeRunHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.wrap_node_run` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, node: AgentNode[Any], handler: WrapNodeRunHandler[Any]) -> NodeResult[Any] | Awaitable[NodeResult[Any]]: ...

class OnNodeRunErrorHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.on_node_run_error` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, node: AgentNode[Any], error: Exception) -> NodeResult[Any] | Awaitable[NodeResult[Any]]: ...

class WrapRunEventStreamHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.wrap_run_event_stream` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, stream: AsyncIterable[AgentStreamEvent]) -> AsyncIterable[AgentStreamEvent]: ...

class OnEventHookFunc(Protocol):
    """Protocol for per-event hook functions (convenience over wrap_run_event_stream)."""
    def __call__(self, ctx: RunContext[Any], event: AgentStreamEvent, /) -> AgentStreamEvent | Awaitable[AgentStreamEvent]: ...

class BeforeModelRequestHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.before_model_request` hook functions."""
    def __call__(self, ctx: RunContext[Any], request_context: ModelRequestContext, /) -> ModelRequestContext | Awaitable[ModelRequestContext]: ...

class AfterModelRequestHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.after_model_request` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, request_context: ModelRequestContext, response: ModelResponse) -> ModelResponse | Awaitable[ModelResponse]: ...

class WrapModelRequestHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.wrap_model_request` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, request_context: ModelRequestContext, handler: WrapModelRequestHandler) -> ModelResponse | Awaitable[ModelResponse]: ...

class OnModelRequestErrorHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.on_model_request_error` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, request_context: ModelRequestContext, error: Exception) -> ModelResponse | Awaitable[ModelResponse]: ...

class PrepareToolsHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.prepare_tools` hook functions."""
    def __call__(self, ctx: RunContext[Any], tool_defs: list[ToolDefinition], /) -> list[ToolDefinition] | Awaitable[list[ToolDefinition]]: ...

class BeforeToolValidateHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.before_tool_validate` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs) -> RawToolArgs | Awaitable[RawToolArgs]: ...

class AfterToolValidateHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.after_tool_validate` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs) -> ValidatedToolArgs | Awaitable[ValidatedToolArgs]: ...

class WrapToolValidateHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.wrap_tool_validate` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs, handler: WrapToolValidateHandler) -> ValidatedToolArgs | Awaitable[ValidatedToolArgs]: ...

class OnToolValidateErrorHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.on_tool_validate_error` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs, error: ValidationError | ModelRetry) -> ValidatedToolArgs | Awaitable[ValidatedToolArgs]: ...

class BeforeToolExecuteHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.before_tool_execute` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs) -> ValidatedToolArgs | Awaitable[ValidatedToolArgs]: ...

class AfterToolExecuteHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.after_tool_execute` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, result: Any) -> Any | Awaitable[Any]: ...

class WrapToolExecuteHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.wrap_tool_execute` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, handler: WrapToolExecuteHandler) -> Any | Awaitable[Any]: ...

class OnToolExecuteErrorHookFunc(Protocol):
    """Protocol for :meth:`~AbstractCapability.on_tool_execute_error` hook functions."""
    def __call__(self, ctx: RunContext[Any], /, *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs, error: Exception) -> Any | Awaitable[Any]: ...
# fmt: on


# --- Helpers ---


async def _call_entry(entry: _HookEntry[Any], hook_name: str, *args: Any, **kwargs: Any) -> Any:
    """Call a hook entry's function, with optional timeout and sync auto-wrapping."""
    func = entry.func
    if entry.timeout is not None:
        try:
            with anyio.fail_after(entry.timeout):
                return await _call_func(func, *args, **kwargs)
        except TimeoutError:
            raise HookTimeoutError(
                hook_name=hook_name,
                func_name=getattr(func, '__name__', repr(func)),
                timeout=entry.timeout,
            ) from None
    return await _call_func(func, *args, **kwargs)


async def _call_func(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Call a function, auto-wrapping sync functions."""
    result = func(*args, **kwargs)
    if inspect.isawaitable(result):
        return await result
    return result


def _filter_tool_entries(entries: list[_HookEntry[Any]], *, call: ToolCallPart) -> list[_HookEntry[Any]]:
    """Filter entries by tool names."""
    return [
        entry
        for entry in entries
        if not (isinstance(entry, _ToolHookEntry) and entry.tools is not None and call.tool_name not in entry.tools)
    ]


# --- Registration decorator helpers ---


def _bare_or_parameterized(
    registry: dict[str, list[_HookEntry[Any]]],
    key: str,
    func: _FuncT | None,
    *,
    timeout: float | None = None,
) -> _FuncT | Callable[[_FuncT], _FuncT]:
    """Handle bare decorator or parameterized decorator for non-tool hooks."""
    if func is not None:
        registry.setdefault(key, []).append(_HookEntry(func, timeout=timeout))
        return func

    def decorator(f: _FuncT) -> _FuncT:
        registry.setdefault(key, []).append(_HookEntry(f, timeout=timeout))
        return f

    return decorator


def _tool_bare_or_parameterized(
    registry: dict[str, list[_HookEntry[Any]]],
    key: str,
    func: _FuncT | None,
    *,
    tools: Sequence[str] | None = None,
    timeout: float | None = None,
) -> _FuncT | Callable[[_FuncT], _FuncT]:
    """Handle bare decorator or parameterized decorator for tool hooks."""
    frozen_tools = frozenset(tools) if tools is not None else None
    if func is not None:
        registry.setdefault(key, []).append(_ToolHookEntry(func, timeout=timeout, tools=frozen_tools))
        return func

    def decorator(f: _FuncT) -> _FuncT:
        registry.setdefault(key, []).append(_ToolHookEntry(f, timeout=timeout, tools=frozen_tools))
        return f

    return decorator


# --- Hook registration namespace ---


class _HookRegistration(Generic[AgentDepsT]):
    """Decorator namespace for registering hooks on a :class:`Hooks` instance.

    Accessed via `hooks.on`. Each method corresponds to a lifecycle hook and
    can be used as a bare decorator or a parameterized decorator::

        @hooks.on.before_model_request
        async def my_hook(ctx, request_context):
            return request_context

        @hooks.on.before_tool_execute(tools=['dangerous'], timeout=5.0)
        async def guard(ctx, *, call, tool_def, args):
            return args
    """

    def __init__(self, hooks: Hooks[AgentDepsT]) -> None:
        self._hooks = hooks

    @property
    def _r(self) -> dict[str, list[_HookEntry[Any]]]:
        return self._hooks._registry  # pyright: ignore[reportPrivateUsage]

    # --- Run lifecycle ---

    @overload
    def before_run(self, func: BeforeRunHookFunc, /) -> BeforeRunHookFunc: ...
    @overload
    def before_run(self, *, timeout: float | None = None) -> Callable[[BeforeRunHookFunc], BeforeRunHookFunc]: ...
    def before_run(self, func: BeforeRunHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'before_run', func, timeout=timeout)

    @overload
    def after_run(self, func: AfterRunHookFunc, /) -> AfterRunHookFunc: ...
    @overload
    def after_run(self, *, timeout: float | None = None) -> Callable[[AfterRunHookFunc], AfterRunHookFunc]: ...
    def after_run(self, func: AfterRunHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'after_run', func, timeout=timeout)

    @overload
    def run(self, func: WrapRunHookFunc, /) -> WrapRunHookFunc: ...
    @overload
    def run(self, *, timeout: float | None = None) -> Callable[[WrapRunHookFunc], WrapRunHookFunc]: ...
    def run(self, func: WrapRunHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'wrap_run', func, timeout=timeout)

    @overload
    def run_error(self, func: OnRunErrorHookFunc, /) -> OnRunErrorHookFunc: ...
    @overload
    def run_error(self, *, timeout: float | None = None) -> Callable[[OnRunErrorHookFunc], OnRunErrorHookFunc]: ...
    def run_error(self, func: OnRunErrorHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'on_run_error', func, timeout=timeout)

    # --- Node lifecycle ---

    @overload
    def before_node_run(self, func: BeforeNodeRunHookFunc, /) -> BeforeNodeRunHookFunc: ...
    @overload
    def before_node_run(
        self, *, timeout: float | None = None
    ) -> Callable[[BeforeNodeRunHookFunc], BeforeNodeRunHookFunc]: ...
    def before_node_run(self, func: BeforeNodeRunHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'before_node_run', func, timeout=timeout)

    @overload
    def after_node_run(self, func: AfterNodeRunHookFunc, /) -> AfterNodeRunHookFunc: ...
    @overload
    def after_node_run(
        self, *, timeout: float | None = None
    ) -> Callable[[AfterNodeRunHookFunc], AfterNodeRunHookFunc]: ...
    def after_node_run(self, func: AfterNodeRunHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'after_node_run', func, timeout=timeout)

    @overload
    def node_run(self, func: WrapNodeRunHookFunc, /) -> WrapNodeRunHookFunc: ...
    @overload
    def node_run(self, *, timeout: float | None = None) -> Callable[[WrapNodeRunHookFunc], WrapNodeRunHookFunc]: ...
    def node_run(self, func: WrapNodeRunHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'wrap_node_run', func, timeout=timeout)

    @overload
    def node_run_error(self, func: OnNodeRunErrorHookFunc, /) -> OnNodeRunErrorHookFunc: ...
    @overload
    def node_run_error(
        self, *, timeout: float | None = None
    ) -> Callable[[OnNodeRunErrorHookFunc], OnNodeRunErrorHookFunc]: ...
    def node_run_error(self, func: OnNodeRunErrorHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'on_node_run_error', func, timeout=timeout)

    # --- Event stream ---

    def run_event_stream(self, func: WrapRunEventStreamHookFunc, /) -> WrapRunEventStreamHookFunc:
        """Register a `wrap_run_event_stream` hook. Timeout not supported for stream wrappers."""
        self._r.setdefault('wrap_run_event_stream', []).append(_HookEntry(func))
        return func

    @overload
    def event(self, func: OnEventHookFunc, /) -> OnEventHookFunc: ...
    @overload
    def event(self, *, timeout: float | None = None) -> Callable[[OnEventHookFunc], OnEventHookFunc]: ...
    def event(self, func: OnEventHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, '_on_event', func, timeout=timeout)

    # --- Model request ---

    @overload
    def before_model_request(self, func: BeforeModelRequestHookFunc, /) -> BeforeModelRequestHookFunc: ...
    @overload
    def before_model_request(
        self, *, timeout: float | None = None
    ) -> Callable[[BeforeModelRequestHookFunc], BeforeModelRequestHookFunc]: ...
    def before_model_request(
        self, func: BeforeModelRequestHookFunc | None = None, *, timeout: float | None = None
    ) -> Any:
        return _bare_or_parameterized(self._r, 'before_model_request', func, timeout=timeout)

    @overload
    def after_model_request(self, func: AfterModelRequestHookFunc, /) -> AfterModelRequestHookFunc: ...
    @overload
    def after_model_request(
        self, *, timeout: float | None = None
    ) -> Callable[[AfterModelRequestHookFunc], AfterModelRequestHookFunc]: ...
    def after_model_request(
        self, func: AfterModelRequestHookFunc | None = None, *, timeout: float | None = None
    ) -> Any:
        return _bare_or_parameterized(self._r, 'after_model_request', func, timeout=timeout)

    @overload
    def model_request(self, func: WrapModelRequestHookFunc, /) -> WrapModelRequestHookFunc: ...
    @overload
    def model_request(
        self, *, timeout: float | None = None
    ) -> Callable[[WrapModelRequestHookFunc], WrapModelRequestHookFunc]: ...
    def model_request(self, func: WrapModelRequestHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'wrap_model_request', func, timeout=timeout)

    @overload
    def model_request_error(self, func: OnModelRequestErrorHookFunc, /) -> OnModelRequestErrorHookFunc: ...
    @overload
    def model_request_error(
        self, *, timeout: float | None = None
    ) -> Callable[[OnModelRequestErrorHookFunc], OnModelRequestErrorHookFunc]: ...
    def model_request_error(
        self, func: OnModelRequestErrorHookFunc | None = None, *, timeout: float | None = None
    ) -> Any:
        return _bare_or_parameterized(self._r, 'on_model_request_error', func, timeout=timeout)

    # --- Tool preparation ---

    @overload
    def prepare_tools(self, func: PrepareToolsHookFunc, /) -> PrepareToolsHookFunc: ...
    @overload
    def prepare_tools(
        self, *, timeout: float | None = None
    ) -> Callable[[PrepareToolsHookFunc], PrepareToolsHookFunc]: ...
    def prepare_tools(self, func: PrepareToolsHookFunc | None = None, *, timeout: float | None = None) -> Any:
        return _bare_or_parameterized(self._r, 'prepare_tools', func, timeout=timeout)

    # --- Tool validation ---

    @overload
    def before_tool_validate(self, func: BeforeToolValidateHookFunc, /) -> BeforeToolValidateHookFunc: ...
    @overload
    def before_tool_validate(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[BeforeToolValidateHookFunc], BeforeToolValidateHookFunc]: ...
    def before_tool_validate(
        self,
        func: BeforeToolValidateHookFunc | None = None,
        *,
        tools: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return _tool_bare_or_parameterized(self._r, 'before_tool_validate', func, tools=tools, timeout=timeout)

    @overload
    def after_tool_validate(self, func: AfterToolValidateHookFunc, /) -> AfterToolValidateHookFunc: ...
    @overload
    def after_tool_validate(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[AfterToolValidateHookFunc], AfterToolValidateHookFunc]: ...
    def after_tool_validate(
        self,
        func: AfterToolValidateHookFunc | None = None,
        *,
        tools: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return _tool_bare_or_parameterized(self._r, 'after_tool_validate', func, tools=tools, timeout=timeout)

    @overload
    def tool_validate(self, func: WrapToolValidateHookFunc, /) -> WrapToolValidateHookFunc: ...
    @overload
    def tool_validate(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[WrapToolValidateHookFunc], WrapToolValidateHookFunc]: ...
    def tool_validate(
        self,
        func: WrapToolValidateHookFunc | None = None,
        *,
        tools: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return _tool_bare_or_parameterized(self._r, 'wrap_tool_validate', func, tools=tools, timeout=timeout)

    @overload
    def tool_validate_error(self, func: OnToolValidateErrorHookFunc, /) -> OnToolValidateErrorHookFunc: ...
    @overload
    def tool_validate_error(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[OnToolValidateErrorHookFunc], OnToolValidateErrorHookFunc]: ...
    def tool_validate_error(
        self,
        func: OnToolValidateErrorHookFunc | None = None,
        *,
        tools: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return _tool_bare_or_parameterized(self._r, 'on_tool_validate_error', func, tools=tools, timeout=timeout)

    # --- Tool execution ---

    @overload
    def before_tool_execute(self, func: BeforeToolExecuteHookFunc, /) -> BeforeToolExecuteHookFunc: ...
    @overload
    def before_tool_execute(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[BeforeToolExecuteHookFunc], BeforeToolExecuteHookFunc]: ...
    def before_tool_execute(
        self,
        func: BeforeToolExecuteHookFunc | None = None,
        *,
        tools: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return _tool_bare_or_parameterized(self._r, 'before_tool_execute', func, tools=tools, timeout=timeout)

    @overload
    def after_tool_execute(self, func: AfterToolExecuteHookFunc, /) -> AfterToolExecuteHookFunc: ...
    @overload
    def after_tool_execute(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[AfterToolExecuteHookFunc], AfterToolExecuteHookFunc]: ...
    def after_tool_execute(
        self,
        func: AfterToolExecuteHookFunc | None = None,
        *,
        tools: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return _tool_bare_or_parameterized(self._r, 'after_tool_execute', func, tools=tools, timeout=timeout)

    @overload
    def tool_execute(self, func: WrapToolExecuteHookFunc, /) -> WrapToolExecuteHookFunc: ...
    @overload
    def tool_execute(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[WrapToolExecuteHookFunc], WrapToolExecuteHookFunc]: ...
    def tool_execute(
        self,
        func: WrapToolExecuteHookFunc | None = None,
        *,
        tools: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return _tool_bare_or_parameterized(self._r, 'wrap_tool_execute', func, tools=tools, timeout=timeout)

    @overload
    def tool_execute_error(self, func: OnToolExecuteErrorHookFunc, /) -> OnToolExecuteErrorHookFunc: ...
    @overload
    def tool_execute_error(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[OnToolExecuteErrorHookFunc], OnToolExecuteErrorHookFunc]: ...
    def tool_execute_error(
        self,
        func: OnToolExecuteErrorHookFunc | None = None,
        *,
        tools: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        return _tool_bare_or_parameterized(self._r, 'on_tool_execute_error', func, tools=tools, timeout=timeout)


# --- The Hooks capability ---


class Hooks(AbstractCapability[AgentDepsT]):
    """Register hook functions via decorators or constructor kwargs.

    For extension developers building reusable capabilities, subclass
    :class:`AbstractCapability` directly. For application code that needs
    a few hooks without the ceremony of a subclass, use `Hooks`.

    Example using decorators::

        hooks = Hooks()

        @hooks.on.before_model_request
        async def log_request(ctx, request_context):
            print(f'Request: {request_context}')
            return request_context

        agent = Agent('openai:gpt-5', capabilities=[hooks])

    Example using constructor kwargs::

        agent = Agent('openai:gpt-5', capabilities=[
            Hooks(before_model_request=log_request)
        ])
    """

    _registry: dict[str, list[_HookEntry[Any]]]

    def __init__(
        self,
        *,
        # Run lifecycle
        before_run: BeforeRunHookFunc | None = None,
        after_run: AfterRunHookFunc | None = None,
        run: WrapRunHookFunc | None = None,
        run_error: OnRunErrorHookFunc | None = None,
        # Node lifecycle
        before_node_run: BeforeNodeRunHookFunc | None = None,
        after_node_run: AfterNodeRunHookFunc | None = None,
        node_run: WrapNodeRunHookFunc | None = None,
        node_run_error: OnNodeRunErrorHookFunc | None = None,
        # Event stream
        run_event_stream: WrapRunEventStreamHookFunc | None = None,
        event: OnEventHookFunc | None = None,
        # Model request
        before_model_request: BeforeModelRequestHookFunc | None = None,
        after_model_request: AfterModelRequestHookFunc | None = None,
        model_request: WrapModelRequestHookFunc | None = None,
        model_request_error: OnModelRequestErrorHookFunc | None = None,
        # Tool preparation
        prepare_tools: PrepareToolsHookFunc | None = None,
        # Tool validation
        before_tool_validate: BeforeToolValidateHookFunc | None = None,
        after_tool_validate: AfterToolValidateHookFunc | None = None,
        tool_validate: WrapToolValidateHookFunc | None = None,
        tool_validate_error: OnToolValidateErrorHookFunc | None = None,
        # Tool execution
        before_tool_execute: BeforeToolExecuteHookFunc | None = None,
        after_tool_execute: AfterToolExecuteHookFunc | None = None,
        tool_execute: WrapToolExecuteHookFunc | None = None,
        tool_execute_error: OnToolExecuteErrorHookFunc | None = None,
        # Ordering
        ordering: CapabilityOrdering | None = None,
    ):
        self._ordering = ordering
        self._registry = {}
        # Map constructor kwarg names to internal registry keys (AbstractCapability method names)
        _kwargs: dict[str, Any] = {
            'before_run': before_run,
            'after_run': after_run,
            'wrap_run': run,
            'on_run_error': run_error,
            'before_node_run': before_node_run,
            'after_node_run': after_node_run,
            'wrap_node_run': node_run,
            'on_node_run_error': node_run_error,
            'wrap_run_event_stream': run_event_stream,
            '_on_event': event,
            'before_model_request': before_model_request,
            'after_model_request': after_model_request,
            'wrap_model_request': model_request,
            'on_model_request_error': model_request_error,
            'prepare_tools': prepare_tools,
            'before_tool_validate': before_tool_validate,
            'after_tool_validate': after_tool_validate,
            'wrap_tool_validate': tool_validate,
            'on_tool_validate_error': tool_validate_error,
            'before_tool_execute': before_tool_execute,
            'after_tool_execute': after_tool_execute,
            'wrap_tool_execute': tool_execute,
            'on_tool_execute_error': tool_execute_error,
        }
        for key, func in _kwargs.items():
            if func is not None:
                self._registry.setdefault(key, []).append(_HookEntry(func))

    @cached_property
    def on(self) -> _HookRegistration[AgentDepsT]:
        """Decorator namespace for registering hook functions."""
        return _HookRegistration(self)

    def _get(self, key: str) -> list[_HookEntry[Any]]:
        return self._registry.get(key, [])

    @property
    def has_wrap_node_run(self) -> bool:
        return bool(self._get('wrap_node_run'))

    @property
    def has_wrap_run_event_stream(self) -> bool:
        return bool(self._get('wrap_run_event_stream') or self._get('_on_event'))

    def get_ordering(self) -> CapabilityOrdering | None:
        return self._ordering

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def __repr__(self) -> str:
        registered = {k: len(v) for k, v in self._registry.items() if v}
        return f'Hooks({registered})'

    # --- AbstractCapability method overrides ---
    # These dispatch to registered hook functions in self._registry.

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        for entry in self._get('before_run'):
            await _call_entry(entry, 'before_run', ctx)

    async def after_run(self, ctx: RunContext[AgentDepsT], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        for entry in self._get('after_run'):
            result = await _call_entry(entry, 'after_run', ctx, result=result)
        return result

    async def wrap_run(self, ctx: RunContext[AgentDepsT], *, handler: WrapRunHandler) -> AgentRunResult[Any]:
        entries = self._get('wrap_run')
        if not entries:
            return await handler()
        chain: Callable[..., Any] = handler
        for entry in reversed(entries):
            chain = _make_wrap_link(entry, 'wrap_run', ctx, {}, chain, None)
        return await chain()

    async def on_run_error(self, ctx: RunContext[AgentDepsT], *, error: BaseException) -> AgentRunResult[Any]:
        for entry in self._get('on_run_error'):
            try:
                return await _call_entry(entry, 'on_run_error', ctx, error=error)
            except BaseException as new_error:
                error = new_error
        raise error

    async def before_node_run(
        self, ctx: RunContext[AgentDepsT], *, node: AgentNode[AgentDepsT]
    ) -> AgentNode[AgentDepsT]:
        for entry in self._get('before_node_run'):
            node = await _call_entry(entry, 'before_node_run', ctx, node=node)
        return node

    async def after_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        result: NodeResult[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        for entry in self._get('after_node_run'):
            result = await _call_entry(entry, 'after_node_run', ctx, node=node, result=result)
        return result

    async def wrap_node_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        node: AgentNode[AgentDepsT],
        handler: WrapNodeRunHandler[AgentDepsT],
    ) -> NodeResult[AgentDepsT]:
        entries = self._get('wrap_node_run')
        if not entries:
            return await handler(node)
        chain: Callable[..., Any] = handler
        for entry in reversed(entries):
            chain = _make_wrap_link(entry, 'wrap_node_run', ctx, {}, chain, 'node')
        return await chain(node)

    async def on_node_run_error(
        self, ctx: RunContext[AgentDepsT], *, node: AgentNode[AgentDepsT], error: Exception
    ) -> NodeResult[AgentDepsT]:
        for entry in self._get('on_node_run_error'):
            try:
                return await _call_entry(entry, 'on_node_run_error', ctx, node=node, error=error)
            except Exception as new_error:
                error = new_error
        raise error

    async def wrap_run_event_stream(
        self, ctx: RunContext[AgentDepsT], *, stream: AsyncIterable[AgentStreamEvent]
    ) -> AsyncIterable[AgentStreamEvent]:
        # First, wrap with per-event callbacks (innermost)
        event_entries = self._get('_on_event')
        if event_entries:
            stream = _event_callback_stream(ctx, stream, event_entries)
        # Then chain explicit stream wrappers (outermost)
        for entry in reversed(self._get('wrap_run_event_stream')):
            stream = entry.func(ctx, stream=stream)
        async for event in stream:
            yield event

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        for entry in self._get('before_model_request'):
            request_context = await _call_entry(entry, 'before_model_request', ctx, request_context)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        for entry in self._get('after_model_request'):
            response = await _call_entry(
                entry, 'after_model_request', ctx, request_context=request_context, response=response
            )
        return response

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        entries = self._get('wrap_model_request')
        if not entries:
            return await handler(request_context)
        chain: Callable[..., Any] = handler
        for entry in reversed(entries):
            chain = _make_wrap_link(entry, 'wrap_model_request', ctx, {}, chain, 'request_context')
        return await chain(request_context)

    async def on_model_request_error(
        self, ctx: RunContext[AgentDepsT], *, request_context: ModelRequestContext, error: Exception
    ) -> ModelResponse:
        for entry in self._get('on_model_request_error'):
            try:
                return await _call_entry(
                    entry, 'on_model_request_error', ctx, request_context=request_context, error=error
                )
            except Exception as new_error:
                error = new_error
        raise error

    async def prepare_tools(self, ctx: RunContext[AgentDepsT], tool_defs: list[ToolDefinition]) -> list[ToolDefinition]:
        for entry in self._get('prepare_tools'):
            tool_defs = await _call_entry(entry, 'prepare_tools', ctx, tool_defs)
        return tool_defs

    async def before_tool_validate(
        self, ctx: RunContext[AgentDepsT], *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs
    ) -> RawToolArgs:
        for entry in _filter_tool_entries(self._get('before_tool_validate'), call=call):
            args = await _call_entry(entry, 'before_tool_validate', ctx, call=call, tool_def=tool_def, args=args)
        return args

    async def after_tool_validate(
        self, ctx: RunContext[AgentDepsT], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        for entry in _filter_tool_entries(self._get('after_tool_validate'), call=call):
            args = await _call_entry(entry, 'after_tool_validate', ctx, call=call, tool_def=tool_def, args=args)
        return args

    async def wrap_tool_validate(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        handler: WrapToolValidateHandler,
    ) -> ValidatedToolArgs:
        entries = _filter_tool_entries(self._get('wrap_tool_validate'), call=call)
        if not entries:
            return await handler(args)
        chain: Callable[..., Any] = handler
        for entry in reversed(entries):
            chain = _make_wrap_link(
                entry, 'wrap_tool_validate', ctx, {'call': call, 'tool_def': tool_def}, chain, 'args'
            )
        return await chain(args)

    async def on_tool_validate_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        error: ValidationError | ModelRetry,
    ) -> ValidatedToolArgs:
        for entry in _filter_tool_entries(self._get('on_tool_validate_error'), call=call):
            try:
                return await _call_entry(
                    entry, 'on_tool_validate_error', ctx, call=call, tool_def=tool_def, args=args, error=error
                )
            except (ValidationError, ModelRetry) as new_error:
                error = new_error
        raise error

    async def before_tool_execute(
        self, ctx: RunContext[AgentDepsT], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        for entry in _filter_tool_entries(self._get('before_tool_execute'), call=call):
            args = await _call_entry(entry, 'before_tool_execute', ctx, call=call, tool_def=tool_def, args=args)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        for entry in _filter_tool_entries(self._get('after_tool_execute'), call=call):
            result = await _call_entry(
                entry, 'after_tool_execute', ctx, call=call, tool_def=tool_def, args=args, result=result
            )
        return result

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        entries = _filter_tool_entries(self._get('wrap_tool_execute'), call=call)
        if not entries:
            return await handler(args)
        chain: Callable[..., Any] = handler
        for entry in reversed(entries):
            chain = _make_wrap_link(
                entry, 'wrap_tool_execute', ctx, {'call': call, 'tool_def': tool_def}, chain, 'args'
            )
        return await chain(args)

    async def on_tool_execute_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        for entry in _filter_tool_entries(self._get('on_tool_execute_error'), call=call):
            try:
                return await _call_entry(
                    entry, 'on_tool_execute_error', ctx, call=call, tool_def=tool_def, args=args, error=error
                )
            except Exception as new_error:
                error = new_error
        raise error


# --- Wrap chain helper ---


def _make_wrap_link(
    entry: _HookEntry[Any],
    hook_name: str,
    ctx: RunContext[Any],
    static_kwargs: dict[str, Any],
    inner_handler: Callable[..., Any],
    handler_arg: str | None,
) -> Callable[..., Any]:
    """Build one link in a wrap middleware chain."""
    frozen_kwargs = dict(static_kwargs)

    if handler_arg:

        async def wrapper(value: Any) -> Any:
            kw = dict(frozen_kwargs)
            kw[handler_arg] = value
            return await _call_entry(entry, hook_name, ctx, handler=inner_handler, **kw)

        return wrapper

    async def wrapper_no_arg() -> Any:
        return await _call_entry(entry, hook_name, ctx, handler=inner_handler, **frozen_kwargs)

    return wrapper_no_arg


# --- Event stream helper ---


async def _event_callback_stream(
    ctx: RunContext[Any],
    stream: AsyncIterable[AgentStreamEvent],
    entries: list[_HookEntry[Any]],
) -> AsyncIterable[AgentStreamEvent]:
    """Wrap a stream with per-event callbacks that can observe or modify events."""
    async for event in stream:
        for entry in entries:
            event = await _call_entry(entry, 'on_event', ctx, event)
        yield event
