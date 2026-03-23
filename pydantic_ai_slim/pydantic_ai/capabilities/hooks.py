"""Hooks capability for decorator-based hook registration.

Provides the `Hooks` class as an ergonomic alternative to subclassing
`AbstractCapability` for registering hook functions.

Hooks intentionally shadows AbstractCapability methods with _HookSlot instance
attributes. The framework invokes these through AbstractCapability-typed refs
(type-safe), while users access the _HookSlot type for decorator registration.
"""

from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, overload

import anyio

from pydantic_ai.messages import AgentStreamEvent, ModelResponse, ToolCallPart
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

from .abstract import (
    AbstractCapability,
    AgentNode,
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
    from pydantic import ValidationError

    from pydantic_ai.exceptions import ModelRetry
    from pydantic_ai.models import ModelRequestContext
    from pydantic_ai.run import AgentRunResult

# fmt: off
# --- Hook function protocols ---
# These define the exact signatures users must implement for each hook type.
# Both sync and async functions are accepted (sync auto-wrapped at runtime).
# Uses RunContext[Any] since AgentDepsT is contravariant (Protocol requires invariant).


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


# --- Hook slots ---


class _HookSlot(Generic[_FuncT]):
    """A hook registration slot that serves as both a decorator and framework dispatch.

    When accessed on a ``Hooks`` instance, this object shadows the inherited
    ``AbstractCapability`` method of the same name. Its ``__call__`` auto-detects
    whether it's being used as a decorator (first arg is a callable) or invoked
    by the framework (first arg is a ``RunContext``).
    """

    __slots__ = ('funcs', '_dispatch', '_default', '_name')

    def __init__(self, dispatch: Callable[..., Any], default: Callable[..., Any], name: str = ''):
        self.funcs: list[_HookEntry[Any]] = []
        self._dispatch = dispatch
        self._default = default
        self._name = name

    @overload
    def __call__(self, func: _FuncT, /) -> _FuncT: ...

    @overload
    def __call__(self, *, timeout: float | None = None) -> Callable[[_FuncT], _FuncT]: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not args:
            # Parameterized decorator: @hooks.before_model_request(timeout=5)
            timeout = kwargs.get('timeout')
            return self._make_decorator(timeout=timeout)

        first = args[0]
        if isinstance(first, RunContext):
            # Framework dispatch
            if not self.funcs:
                return self._default(*args, **kwargs)
            return self._dispatch(self.funcs, self._name, *args, **kwargs)

        # Bare decorator: @hooks.before_model_request
        self.funcs.append(self._make_entry(first))
        return first

    def _make_entry(self, func: _FuncT, *, timeout: float | None = None) -> _HookEntry[_FuncT]:
        """Create a hook entry. Override in subclasses to change the entry type."""
        return _HookEntry(func, timeout=timeout)

    def _make_decorator(self, *, timeout: float | None = None) -> Callable[[_FuncT], _FuncT]:
        def decorator(func: _FuncT) -> _FuncT:
            self.funcs.append(self._make_entry(func, timeout=timeout))
            return func

        return decorator


class _ToolHookSlot(_HookSlot[_FuncT]):
    """Hook slot for tool hooks — adds ``tools`` filtering parameter."""

    @overload
    def __call__(self, func: _FuncT, /) -> _FuncT: ...

    @overload
    def __call__(
        self, *, tools: Sequence[str] | None = None, timeout: float | None = None
    ) -> Callable[[_FuncT], _FuncT]: ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not args:
            # Parameterized decorator: @hooks.before_tool_execute(tools=['x'], timeout=5)
            timeout = kwargs.get('timeout')
            tools = kwargs.get('tools')
            return self._make_tool_decorator(timeout=timeout, tools=tools)
        # Delegate RunContext dispatch and bare-decorator to parent
        return super().__call__(*args, **kwargs)

    def _make_entry(self, func: _FuncT, *, timeout: float | None = None) -> _HookEntry[_FuncT]:
        """Override to create _ToolHookEntry for bare decorators."""
        return _ToolHookEntry(func, timeout=timeout)

    def _make_tool_decorator(
        self, *, timeout: float | None = None, tools: Sequence[str] | None = None
    ) -> Callable[[_FuncT], _FuncT]:
        frozen_tools = frozenset(tools) if tools is not None else None

        def decorator(func: _FuncT) -> _FuncT:
            self.funcs.append(_ToolHookEntry(func, timeout=timeout, tools=frozen_tools))
            return func

        return decorator


# --- Dispatch helpers ---


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


def _iter_tool_entries(entries: list[_HookEntry[Any]], *, call: ToolCallPart | None = None) -> list[_HookEntry[Any]]:
    """Filter entries by tool names if applicable."""
    if call is None:
        return entries
    return [
        entry
        for entry in entries
        if not (isinstance(entry, _ToolHookEntry) and entry.tools and call.tool_name not in entry.tools)
    ]


# --- Dispatch functions for each pattern ---
# These are called by _HookSlot when the framework invokes the hook.
# Signature: (entries, hook_name, ctx, *args, **kwargs) -> coroutine | async_iterable


async def _dispatch_observe(entries: list[_HookEntry[Any]], hook_name: str, ctx: RunContext[Any]) -> None:
    """before_run: call each for side effects."""
    for entry in entries:
        await _call_entry(entry, hook_name, ctx)


async def _dispatch_forward_positional(
    entries: list[_HookEntry[Any]], hook_name: str, ctx: RunContext[Any], value: Any, **kwargs: Any
) -> Any:
    """before_model_request, prepare_tools: chain through 2nd positional arg."""
    for entry in _iter_tool_entries(entries, call=kwargs.get('call')):
        value = await _call_entry(entry, hook_name, ctx, value, **kwargs)
    return value


def _make_dispatch_forward_keyword(return_key: str) -> Callable[..., Any]:
    """Create dispatch for hooks that chain through a keyword arg (after_*, before_node_run, etc.)."""

    async def dispatch(entries: list[_HookEntry[Any]], hook_name: str, ctx: RunContext[Any], **kwargs: Any) -> Any:
        for entry in _iter_tool_entries(entries, call=kwargs.get('call')):
            kwargs[return_key] = await _call_entry(entry, hook_name, ctx, **kwargs)
        return kwargs[return_key]

    return dispatch


def _make_dispatch_wrap(handler_arg: str | None) -> Callable[..., Any]:
    """Create dispatch for wrap_* hooks that build a middleware chain."""

    async def dispatch(entries: list[_HookEntry[Any]], hook_name: str, ctx: RunContext[Any], **kwargs: Any) -> Any:
        handler = kwargs.pop('handler')
        chain = handler
        filtered = _iter_tool_entries(entries, call=kwargs.get('call'))
        for entry in reversed(filtered):
            chain = _build_wrap_link(entry, hook_name, ctx, kwargs, chain, handler_arg)
        if handler_arg and handler_arg in kwargs:
            return await chain(kwargs[handler_arg])
        return await chain()

    return dispatch


def _build_wrap_link(
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


def _dispatch_wrap_stream(
    entries: list[_HookEntry[Any]],
    hook_name: str,
    ctx: RunContext[Any],
    *,
    stream: AsyncIterable[Any],
) -> AsyncIterable[Any]:
    """wrap_run_event_stream: chain async generators (sync return, not a coroutine).

    Timeout is not supported for stream wrapper hooks because they are async
    generators, not single coroutines.  Calls entry.func directly.
    """
    for entry in reversed(entries):
        stream = entry.func(ctx, stream=stream)
    return stream


async def _dispatch_error(entries: list[_HookEntry[Any]], hook_name: str, ctx: RunContext[Any], **kwargs: Any) -> Any:
    """on_*_error: try each handler, first recovery wins."""
    error = kwargs.pop('error')
    for entry in _iter_tool_entries(entries, call=kwargs.get('call')):
        try:
            return await _call_entry(entry, hook_name, ctx, error=error, **kwargs)
        except Exception as new_error:
            error = new_error
    raise error


# --- Default functions (when no hooks registered) ---


async def _noop_observe(ctx: RunContext[Any]) -> None:
    pass


async def _noop_forward_positional(ctx: RunContext[Any], value: Any, **kwargs: Any) -> Any:
    return value


def _make_noop_forward_keyword(return_key: str) -> Callable[..., Any]:
    async def noop(ctx: RunContext[Any], **kwargs: Any) -> Any:
        return kwargs[return_key]

    return noop


def _make_noop_wrap(handler_arg: str | None) -> Callable[..., Any]:
    async def noop(ctx: RunContext[Any], **kwargs: Any) -> Any:
        handler = kwargs['handler']
        if handler_arg and handler_arg in kwargs:
            return await handler(kwargs[handler_arg])
        return await handler()

    return noop


def _noop_wrap_stream(ctx: RunContext[Any], *, stream: AsyncIterable[Any]) -> AsyncIterable[Any]:
    return stream


async def _noop_error(ctx: RunContext[Any], **kwargs: Any) -> Any:
    raise kwargs['error']


# --- on_run_error needs BaseException handling ---


async def _dispatch_run_error(
    entries: list[_HookEntry[Any]], hook_name: str, ctx: RunContext[Any], *, error: BaseException
) -> Any:
    for entry in entries:
        try:
            return await _call_entry(entry, hook_name, ctx, error=error)
        except BaseException as new_error:
            error = new_error
    raise error


async def _noop_run_error(ctx: RunContext[Any], *, error: BaseException) -> Any:
    raise error


# --- The Hooks capability ---


class Hooks(AbstractCapability[AgentDepsT]):
    """Register hook functions via decorators or constructor kwargs.

    For extension developers building reusable capabilities, subclass
    :class:`AbstractCapability` directly. For application code that needs
    a few hooks without the ceremony of a subclass, use ``Hooks``.

    Example using decorators::

        hooks = Hooks()

        @hooks.before_model_request
        async def log_request(ctx, request_context):
            print(f'Request: {request_context}')
            return request_context

        agent = Agent('openai:gpt-5', capabilities=[hooks])

    Example using constructor kwargs::

        agent = Agent('openai:gpt-5', capabilities=[
            Hooks(before_model_request=log_request)
        ])
    """

    # --- Type annotations for hook slots ---
    # These intentionally shadow the inherited AbstractCapability methods on instances,
    # giving pyright the _HookSlot type for decorator usage while the framework sees
    # AbstractCapability methods through the base class type.

    # Run lifecycle
    before_run: _HookSlot[BeforeRunHookFunc]
    after_run: _HookSlot[AfterRunHookFunc]
    wrap_run: _HookSlot[WrapRunHookFunc]
    on_run_error: _HookSlot[OnRunErrorHookFunc]

    # Node lifecycle
    before_node_run: _HookSlot[BeforeNodeRunHookFunc]
    after_node_run: _HookSlot[AfterNodeRunHookFunc]
    wrap_node_run: _HookSlot[WrapNodeRunHookFunc]
    on_node_run_error: _HookSlot[OnNodeRunErrorHookFunc]

    # Event stream
    wrap_run_event_stream: _HookSlot[WrapRunEventStreamHookFunc]

    # Model request
    before_model_request: _HookSlot[BeforeModelRequestHookFunc]
    after_model_request: _HookSlot[AfterModelRequestHookFunc]
    wrap_model_request: _HookSlot[WrapModelRequestHookFunc]
    on_model_request_error: _HookSlot[OnModelRequestErrorHookFunc]

    # Tool preparation
    prepare_tools: _HookSlot[PrepareToolsHookFunc]

    # Tool validation
    before_tool_validate: _ToolHookSlot[BeforeToolValidateHookFunc]
    after_tool_validate: _ToolHookSlot[AfterToolValidateHookFunc]
    wrap_tool_validate: _ToolHookSlot[WrapToolValidateHookFunc]
    on_tool_validate_error: _ToolHookSlot[OnToolValidateErrorHookFunc]

    # Tool execution
    before_tool_execute: _ToolHookSlot[BeforeToolExecuteHookFunc]
    after_tool_execute: _ToolHookSlot[AfterToolExecuteHookFunc]
    wrap_tool_execute: _ToolHookSlot[WrapToolExecuteHookFunc]
    on_tool_execute_error: _ToolHookSlot[OnToolExecuteErrorHookFunc]

    def __init__(
        self,
        *,
        # Run lifecycle
        before_run: BeforeRunHookFunc | None = None,
        after_run: AfterRunHookFunc | None = None,
        wrap_run: WrapRunHookFunc | None = None,
        on_run_error: OnRunErrorHookFunc | None = None,
        # Node lifecycle
        before_node_run: BeforeNodeRunHookFunc | None = None,
        after_node_run: AfterNodeRunHookFunc | None = None,
        wrap_node_run: WrapNodeRunHookFunc | None = None,
        on_node_run_error: OnNodeRunErrorHookFunc | None = None,
        # Event stream
        wrap_run_event_stream: WrapRunEventStreamHookFunc | None = None,
        # Model request
        before_model_request: BeforeModelRequestHookFunc | None = None,
        after_model_request: AfterModelRequestHookFunc | None = None,
        wrap_model_request: WrapModelRequestHookFunc | None = None,
        on_model_request_error: OnModelRequestErrorHookFunc | None = None,
        # Tool preparation
        prepare_tools: PrepareToolsHookFunc | None = None,
        # Tool validation
        before_tool_validate: BeforeToolValidateHookFunc | None = None,
        after_tool_validate: AfterToolValidateHookFunc | None = None,
        wrap_tool_validate: WrapToolValidateHookFunc | None = None,
        on_tool_validate_error: OnToolValidateErrorHookFunc | None = None,
        # Tool execution
        before_tool_execute: BeforeToolExecuteHookFunc | None = None,
        after_tool_execute: AfterToolExecuteHookFunc | None = None,
        wrap_tool_execute: WrapToolExecuteHookFunc | None = None,
        on_tool_execute_error: OnToolExecuteErrorHookFunc | None = None,
    ):
        # --- Initialize hook slots ---

        # Run lifecycle
        self.before_run = _HookSlot(_dispatch_observe, _noop_observe, 'before_run')  # pyright: ignore[reportIncompatibleMethodOverride]
        self.after_run = _HookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_forward_keyword('result'), _make_noop_forward_keyword('result'), 'after_run'
        )
        self.wrap_run = _HookSlot(_make_dispatch_wrap(None), _make_noop_wrap(None), 'wrap_run')  # pyright: ignore[reportIncompatibleMethodOverride]
        self.on_run_error = _HookSlot(_dispatch_run_error, _noop_run_error, 'on_run_error')  # pyright: ignore[reportIncompatibleMethodOverride]

        # Node lifecycle
        self.before_node_run = _HookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_forward_keyword('node'), _make_noop_forward_keyword('node'), 'before_node_run'
        )
        self.after_node_run = _HookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_forward_keyword('result'), _make_noop_forward_keyword('result'), 'after_node_run'
        )
        self.wrap_node_run = _HookSlot(_make_dispatch_wrap('node'), _make_noop_wrap('node'), 'wrap_node_run')  # pyright: ignore[reportIncompatibleMethodOverride]
        self.on_node_run_error = _HookSlot(_dispatch_error, _noop_error, 'on_node_run_error')  # pyright: ignore[reportIncompatibleMethodOverride]

        # Event stream
        self.wrap_run_event_stream = _HookSlot(_dispatch_wrap_stream, _noop_wrap_stream, 'wrap_run_event_stream')  # pyright: ignore[reportIncompatibleMethodOverride]

        # Model request
        self.before_model_request = _HookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _dispatch_forward_positional, _noop_forward_positional, 'before_model_request'
        )
        self.after_model_request = _HookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_forward_keyword('response'),
            _make_noop_forward_keyword('response'),
            'after_model_request',
        )
        self.wrap_model_request = _HookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_wrap('request_context'), _make_noop_wrap('request_context'), 'wrap_model_request'
        )
        self.on_model_request_error = _HookSlot(_dispatch_error, _noop_error, 'on_model_request_error')  # pyright: ignore[reportIncompatibleMethodOverride]

        # Tool preparation
        self.prepare_tools = _HookSlot(_dispatch_forward_positional, _noop_forward_positional, 'prepare_tools')  # pyright: ignore[reportIncompatibleMethodOverride]

        # Tool validation
        self.before_tool_validate = _ToolHookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_forward_keyword('args'), _make_noop_forward_keyword('args'), 'before_tool_validate'
        )
        self.after_tool_validate = _ToolHookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_forward_keyword('args'), _make_noop_forward_keyword('args'), 'after_tool_validate'
        )
        self.wrap_tool_validate = _ToolHookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_wrap('args'), _make_noop_wrap('args'), 'wrap_tool_validate'
        )
        self.on_tool_validate_error = _ToolHookSlot(_dispatch_error, _noop_error, 'on_tool_validate_error')  # pyright: ignore[reportIncompatibleMethodOverride]

        # Tool execution
        self.before_tool_execute = _ToolHookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_forward_keyword('args'), _make_noop_forward_keyword('args'), 'before_tool_execute'
        )
        self.after_tool_execute = _ToolHookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_forward_keyword('result'), _make_noop_forward_keyword('result'), 'after_tool_execute'
        )
        self.wrap_tool_execute = _ToolHookSlot(  # pyright: ignore[reportIncompatibleMethodOverride]
            _make_dispatch_wrap('args'), _make_noop_wrap('args'), 'wrap_tool_execute'
        )
        self.on_tool_execute_error = _ToolHookSlot(_dispatch_error, _noop_error, 'on_tool_execute_error')  # pyright: ignore[reportIncompatibleMethodOverride]

        # --- Register constructor-provided functions ---
        _register_if_provided(self.before_run, before_run)
        _register_if_provided(self.after_run, after_run)
        _register_if_provided(self.wrap_run, wrap_run)
        _register_if_provided(self.on_run_error, on_run_error)
        _register_if_provided(self.before_node_run, before_node_run)
        _register_if_provided(self.after_node_run, after_node_run)
        _register_if_provided(self.wrap_node_run, wrap_node_run)
        _register_if_provided(self.on_node_run_error, on_node_run_error)
        _register_if_provided(self.wrap_run_event_stream, wrap_run_event_stream)
        _register_if_provided(self.before_model_request, before_model_request)
        _register_if_provided(self.after_model_request, after_model_request)
        _register_if_provided(self.wrap_model_request, wrap_model_request)
        _register_if_provided(self.on_model_request_error, on_model_request_error)
        _register_if_provided(self.prepare_tools, prepare_tools)
        _register_if_provided(self.before_tool_validate, before_tool_validate)
        _register_if_provided(self.after_tool_validate, after_tool_validate)
        _register_if_provided(self.wrap_tool_validate, wrap_tool_validate)
        _register_if_provided(self.on_tool_validate_error, on_tool_validate_error)
        _register_if_provided(self.before_tool_execute, before_tool_execute)
        _register_if_provided(self.after_tool_execute, after_tool_execute)
        _register_if_provided(self.wrap_tool_execute, wrap_tool_execute)
        _register_if_provided(self.on_tool_execute_error, on_tool_execute_error)

    @property
    def has_wrap_node_run(self) -> bool:
        return bool(self.wrap_node_run.funcs)

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None  # Not spec-serializable (contains callables)

    def __repr__(self) -> str:
        registered = {
            name: len(slot.funcs)
            for name in _ALL_HOOK_NAMES
            if (slot := getattr(self, name, None)) is not None and slot.funcs
        }
        return f'Hooks({registered})'


def _register_if_provided(slot: _HookSlot[Any], func: Callable[..., Any] | None) -> None:
    """Register a function on a slot if it's not None."""
    if func is not None:
        slot(func)


_ALL_HOOK_NAMES: tuple[str, ...] = (
    'before_run',
    'after_run',
    'wrap_run',
    'on_run_error',
    'before_node_run',
    'after_node_run',
    'wrap_node_run',
    'on_node_run_error',
    'wrap_run_event_stream',
    'before_model_request',
    'after_model_request',
    'wrap_model_request',
    'on_model_request_error',
    'prepare_tools',
    'before_tool_validate',
    'after_tool_validate',
    'wrap_tool_validate',
    'on_tool_validate_error',
    'before_tool_execute',
    'after_tool_execute',
    'wrap_tool_execute',
    'on_tool_execute_error',
)
