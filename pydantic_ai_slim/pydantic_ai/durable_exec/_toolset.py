from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any, Generic, Literal, Protocol, TypeAlias, cast

import anyio
from pydantic import Discriminator, Tag, ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError, to_jsonable_python
from typing_extensions import Self, assert_never

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai._agent_graph import build_validation_context
from pydantic_ai._cancel import RunCancellation
from pydantic_ai._enqueue import PendingMessage
from pydantic_ai._utils import is_str_dict
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, ToolFailed, UserError
from pydantic_ai.messages import InstructionPart, ToolReturn, ToolReturnContent
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.external import TOOL_SCHEMA_VALIDATOR
from pydantic_ai.toolsets.function import FunctionToolsetTool

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent
    from pydantic_ai.mcp import MCPToolset

DurableConfig: TypeAlias = Mapping[str, Any]
ToolConfig: TypeAlias = DurableConfig | Literal[False]
Lifecycle: TypeAlias = Literal['enter-outside-durable', 'enter-always', 'enter-never', 'enter-in-durable-unit']
Instructions: TypeAlias = str | InstructionPart | Sequence[str | InstructionPart] | None


class CallToolOperation(Protocol):
    async def __call__(
        self,
        name: str,
        tool_args: dict[str, Any],
        *,
        ctx: RunContext[Any],
        tool: ToolsetTool[Any],
        config: DurableConfig,
    ) -> Any: ...


"""Runs one tool call inside the engine's durable unit (activity/step/task)."""
ResolveToolConfig: TypeAlias = Callable[[ToolsetTool[Any] | None, str], ToolConfig]
"""Resolve a tool's per-tool durable config: a config mapping to merge, or `False` to run the tool inline.

Engines that restrict inline execution enforce it here, where the engine's own error
wording is available (e.g. Temporal requires async tools and forbids inline MCP tools).
"""
ValidationContextResolver: TypeAlias = Callable[[RunContext[Any]], Any]


def _serializable_validation_input(value: Any) -> Any:
    try:
        return to_jsonable_python(value)
    except PydanticSerializationError:
        try:
            representation = repr(value)
        except Exception:
            representation = '<repr failed>'
        return {'type': f'{type(value).__module__}.{type(value).__qualname__}', 'repr': representation}


def live_validation_context(ctx: RunContext[Any]) -> Any:
    """Return the run's live validation context for in-process durable units."""
    return object.__getattribute__(ctx, 'validation_context')


def validation_context_from_agent(agent: AbstractAgent[Any, Any] | None) -> ValidationContextResolver:
    """Rebuild a run's validation context inside a serialized durable unit."""

    def resolve(ctx: RunContext[Any]) -> Any:
        spec = agent._get_validation_context() if agent is not None else None  # pyright: ignore[reportPrivateUsage]
        return build_validation_context(spec, ctx)

    return resolve


@dataclass(kw_only=True)
class DynamicToolInfo:
    """Serializable tool information returned from dynamic tool discovery."""

    tool_def: ToolDefinition
    max_retries: int
    has_args_validator: bool = False
    """Whether the tool's validator needs its own unit; false decodes older recorded payloads."""


@dataclass(kw_only=True)
class DynamicToolsResult:
    """Serializable result of the dynamic toolset's tool discovery operation.

    Instructions are collected in the same durable unit (and thus against the same resolution and entry of
    the inner toolset) as the tools. For an MCP-backed dynamic toolset this keeps discovery to a single
    entry of the server rather than one for tools and another for instructions; the second entry would add
    a redundant `initialize` round-trip whose `notifications/initialized` races teardown.
    """

    tools: dict[str, DynamicToolInfo]
    instructions: Instructions


class RunHeldToolset(Generic[AgentDepsT]):
    """A toolset held entered for one durable run, entered lazily inside a durable unit.

    Each durable unit used to enter (and, for a dynamic toolset, build) its own toolset and tear it
    down again, so anything the toolset cached — such as
    [`MCPToolset.cache_tools`][pydantic_ai.mcp.MCPToolset.cache_tools] — was discarded before the
    next unit could use it, and an MCP server saw a fresh session per unit. The run holds one
    instead: the first durable unit that needs it enters it, where the engine's own retry policy
    covers a failed connection, and the run exits it at the end. That is the lifecycle a non-durable
    run gives a toolset.

    A toolset attached to the agent is the same object in the container and in the unit, so the run
    holds that toolset itself. A [`DynamicToolset`][pydantic_ai.toolsets.DynamicToolset] built with
    `per_run_step=False` is resolved once by the run, which then holds what it resolved: resolving is
    not connecting — an `MCPToolset` opens nothing until it is entered — but the factory is arbitrary
    user code that runs in the durable container rather than in a unit, so I/O inside it is not
    checkpointed and re-runs when the container replays. Like the capability factories that have
    always run there, it has to be deterministic given the run's dependencies and leave its I/O to
    the units — which is what the engine docs tell users.

    Only used where the durable unit runs in the same process as the container. Engines that
    serialize the run context across the boundary never see one and each unit enters its own, as
    they always have.
    """

    def __init__(self, id: str, toolset: AbstractToolset[AgentDepsT]):
        self.id = id
        """The toolset `id` a durable unit looks it up by."""
        self.toolset = toolset
        self._entered = False

    @functools.cached_property
    def _lock(self) -> anyio.Lock:
        # Created on first use so it binds to the running event loop, and so parallel tool-call
        # units in one run step can't both enter the toolset.
        return anyio.Lock()

    async def entered(self) -> AbstractToolset[AgentDepsT]:
        """Return the toolset the run holds, entering it the first time a unit needs it."""
        async with self._lock:
            if not self._entered:
                await self.toolset.__aenter__()
                # Only mark it entered once `__aenter__` succeeded, so a failed connection is
                # retried by the next unit rather than leaving a toolset nothing will exit.
                self._entered = True
            return self.toolset

    async def aclose(self, *args: Any) -> None:
        """Exit the toolset at the end of the run, if any unit entered it.

        Takes the run's own `__aexit__` arguments: the units that used the toolset each returned
        long ago, so how the run ended is the only thing that can tell a toolset whether to roll
        back or commit what it did.
        """
        async with self._lock:
            if self._entered:
                self._entered = False
                await self.toolset.__aexit__(*args)


def _run_held_toolset(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> RunHeldToolset[AgentDepsT] | None:
    """The toolset this run holds for this one, if it's reachable from this durable unit.

    The run context holds them without their dependencies type, which is the run's own.
    """
    held = ctx._run_held_toolsets  # pyright: ignore[reportPrivateUsage]
    if held is None or toolset.id is None:
        return None
    return cast('RunHeldToolset[AgentDepsT] | None', held.get(toolset.id))


@asynccontextmanager
async def toolset_for_unit(
    toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]
) -> AsyncGenerator[AbstractToolset[AgentDepsT]]:
    """Yield the toolset to run one durable unit against.

    Reuses the toolset the run holds entered when the unit can reach it, and otherwise resolves and
    enters one for this unit alone — the only option when the unit may run in another process, and
    what every unit did before run-held toolsets existed.
    """
    if (held := _run_held_toolset(toolset, ctx)) is not None:
        yield await held.entered()
        return
    run_toolset = await toolset.for_run(ctx)
    async with run_toolset:
        yield run_toolset


async def get_dynamic_tools(toolset: AbstractToolset[AgentDepsT], ctx: RunContext[AgentDepsT]) -> DynamicToolsResult:
    """Resolve a dynamic toolset and collect its tools and instructions in a single entry.

    Falls back to resolving the toolset for this unit alone when the run's own resolved toolset
    isn't reachable, so replay/recovery in a fresh process stays deterministic.
    """
    async with toolset_for_unit(toolset, ctx) as run_toolset:
        run_toolset = await run_toolset.for_run_step(ctx)
        tools = await run_toolset.get_tools(ctx)
        instructions = await run_toolset.get_instructions(ctx)
        return DynamicToolsResult(
            tools={
                name: DynamicToolInfo(
                    tool_def=tool.tool_def,
                    max_retries=tool.max_retries,
                    has_args_validator=tool.args_validator_func is not None,
                )
                for name, tool in tools.items()
            },
            instructions=instructions,
        )


async def _dynamic_tool(
    toolset: AbstractToolset[AgentDepsT],
    run_toolset: AbstractToolset[AgentDepsT],
    name: str,
    tool_def: ToolDefinition | None,
    ctx: RunContext[AgentDepsT],
) -> ToolsetTool[AgentDepsT]:
    """The tool to call, rebuilt from the definition the discovery unit recorded when there is one.

    A toolset that can build the tool from its definition alone answers without listing its tools;
    the default implementation lists them, as this always did.
    """
    try:
        if tool_def is None:
            tool = (await run_toolset.get_tools(ctx))[name]
        else:
            tool = await run_toolset.get_tool_for_tool_def(tool_def, ctx)
    except KeyError as e:  # pragma: no cover
        raise UserError(
            f'Tool {name!r} not found in dynamic toolset {toolset.id!r}. '
            'The dynamic toolset function may have returned a different toolset than expected.'
        ) from e
    if tool_def is None:
        return tool
    tool = replace(tool, tool_def=tool_def)
    if isinstance(tool, FunctionToolsetTool):
        tool = replace(tool, timeout=tool_def.timeout)
    return tool


async def call_dynamic_tool(
    toolset: AbstractToolset[AgentDepsT],
    name: str,
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    *,
    tool_def: ToolDefinition | None = None,
    validation_context: ValidationContextResolver = live_validation_context,
) -> Any:
    """Resolve a dynamic toolset, re-validate the tool args, and call the tool.

    The args were only parsed (not validated) on the workflow/flow side, where the real tool
    isn't available; validation happens here against the resolved tool's own validator.
    """
    async with toolset_for_unit(toolset, ctx) as run_toolset:
        run_toolset = await run_toolset.for_run_step(ctx)
        tool = await _dynamic_tool(toolset, run_toolset, name, tool_def, ctx)
        args = tool.args_validator.validate_python(tool_args, context=validation_context(ctx))
        return await run_toolset.call_tool(name, args, ctx, tool)


async def validate_dynamic_tool_args(
    toolset: AbstractToolset[AgentDepsT],
    name: str,
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    *,
    tool_def: ToolDefinition | None = None,
    validation_context: ValidationContextResolver = live_validation_context,
) -> None:
    """Resolve a dynamic toolset and validate arguments against its real tool."""
    async with toolset_for_unit(toolset, ctx) as run_toolset:
        run_toolset = await run_toolset.for_run_step(ctx)
        tool = await _dynamic_tool(toolset, run_toolset, name, tool_def, ctx)
        await validate_tool_args(tool, tool_args, ctx, validation_context=validation_context)


async def validate_tool_args(
    tool: ToolsetTool[AgentDepsT],
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    *,
    validation_context: ValidationContextResolver = live_validation_context,
) -> None:
    """Schema-validate arguments and run the tool's validator inside a durable unit."""
    args = tool.args_validator.validate_python(tool_args, context=validation_context(ctx))
    await run_args_validator(tool, args, ctx)


async def run_args_validator(tool: ToolsetTool[AgentDepsT], args: dict[str, Any], ctx: RunContext[AgentDepsT]) -> None:
    """Run a tool's validator on already schema-validated arguments."""
    args_validator_func = tool.args_validator_func
    if args_validator_func is None:
        raise UserError(
            f'Tool {tool.tool_def.name!r} has no `args_validator`. '
            'The dynamic toolset function may have returned a different toolset than expected.'
        )
    result = args_validator_func(ctx, **args)
    if inspect.isawaitable(result):
        await result


@dataclass
class _ApprovalRequired:
    metadata: dict[str, Any] | None = None
    _: KW_ONLY
    kind: Literal['approval_required'] = 'approval_required'


@dataclass
class _CallDeferred:
    metadata: dict[str, Any] | None = None
    _: KW_ONLY
    kind: Literal['call_deferred'] = 'call_deferred'


@dataclass
class _ModelRetry:
    message: str
    _: KW_ONLY
    kind: Literal['model_retry'] = 'model_retry'


@dataclass
class _ValidationErrorDetail:
    type: str
    _: KW_ONLY
    loc: list[str | int]
    msg: str
    input: Any


@dataclass
class _ValidationError:
    title: str
    _: KW_ONLY
    errors: list[_ValidationErrorDetail]
    kind: Literal['validation_error'] = 'validation_error'


@dataclass
class _ToolFailed:
    message: str
    _: KW_ONLY
    kind: Literal['tool_failed'] = 'tool_failed'


def _result_discriminator(value: Any) -> str:
    if isinstance(value, ToolReturn) or (is_str_dict(value) and value.get('kind') == 'tool-return'):
        return 'tool-return'
    return 'content'


_ToolReturnResult = Annotated[
    Annotated[ToolReturn, Tag('tool-return')] | Annotated[ToolReturnContent, Tag('content')],
    Discriminator(_result_discriminator),
]


@dataclass
class _ToolReturn:
    """Legacy wire shape retained for decoding in-flight durable executions."""

    result: _ToolReturnResult
    _: KW_ONLY
    kind: Literal['tool_return'] = 'tool_return'


@dataclass
class _ToolContentResult:
    # Emitted only when a user dict's `kind` collides with `'tool-return'`. Workers predating this
    # variant cannot decode it, but those payloads already failed to round-trip there; ordinary
    # results deliberately retain the legacy `tool_return` shape for rolling upgrades.
    result: ToolReturnContent
    _: KW_ONLY
    kind: Literal['tool_content_result'] = 'tool_content_result'


CallToolResult = Annotated[
    _ApprovalRequired | _CallDeferred | _ModelRetry | _ValidationError | _ToolReturn | _ToolContentResult | _ToolFailed,
    Discriminator('kind'),
]


async def wrap_tool_call_result(coro: Awaitable[Any]) -> CallToolResult:
    try:
        result = await coro
        if is_str_dict(result) and result.get('kind') == 'tool-return':
            return _ToolContentResult(result=result)
        return _ToolReturn(result=result)
    except ApprovalRequired as exc:
        return _ApprovalRequired(metadata=exc.metadata)
    except CallDeferred as exc:
        return _CallDeferred(metadata=exc.metadata)
    except ModelRetry as exc:
        return _ModelRetry(message=exc.message)
    except ToolFailed as exc:
        return _ToolFailed(message=exc.message)
    except ValidationError as exc:
        return _ValidationError(
            title=exc.title,
            errors=[
                _ValidationErrorDetail(
                    type=detail['type'],
                    loc=list(detail['loc']),
                    msg=detail['msg'],
                    input=_serializable_validation_input(detail.get('input')),
                )
                for detail in exc.errors(include_url=False, include_context=False)
            ],
        )


def unwrap_tool_call_result(result: CallToolResult) -> Any:
    if isinstance(result, _ToolReturn | _ToolContentResult):
        return result.result
    if isinstance(result, _ApprovalRequired):
        raise ApprovalRequired(metadata=result.metadata)
    if isinstance(result, _CallDeferred):
        raise CallDeferred(metadata=result.metadata)
    if isinstance(result, _ValidationError):
        raise ValidationError.from_exception_data(
            result.title,
            [
                {
                    'type': PydanticCustomError(
                        error.type,  # pyright: ignore[reportArgumentType]
                        '{message}',
                        {'message': error.msg},
                    ),
                    'loc': tuple(error.loc),
                    'input': error.input,
                }
                for error in result.errors
            ],
        )
    if isinstance(result, _ModelRetry):
        raise ModelRetry(result.message)
    if isinstance(result, _ToolFailed):
        raise ToolFailed(result.message)
    assert_never(result)


class EnqueueGuard(list[PendingMessage]):
    """Replaces `ctx.pending_messages` inside a durable unit, where enqueueing can't be supported.

    A durable unit's recorded output is replayed on recovery (DBOS), cache hit (Prefect), or
    across the activity boundary (Temporal) without re-running the code, so messages enqueued
    inside it would be silently dropped; enqueueing raises an explanatory `UserError` instead.
    """

    def __init__(self, message: str):
        super().__init__()
        self._message = message

    def append(self, pending: PendingMessage) -> None:
        raise UserError(self._message)


def enqueue_not_supported_message(unit_noun: str, container_noun: str) -> str:
    """The shared `ctx.enqueue()` error, worded for one engine's durable unit and container.

    `unit_noun` is the engine's durable unit (`'activity'`/`'step'`/`'task'`) and
    `container_noun` is its durable container (`'workflow'`/`'flow'`), so every engine
    raises the same explanation with its own vocabulary.
    """
    return (
        f'`ctx.enqueue()` is not supported inside a durable {unit_noun}: the durable runtime replays '
        f"the {unit_noun}'s recorded result without re-running your code, so the enqueued messages "
        f'would be dropped. Enqueue messages from {container_noun}-level code instead.'
    )


class CancelGuard(RunCancellation):
    """Replaces the run's live cancellation controller inside a durable unit.

    `ctx.cancel()` inside a durable unit would be replay-divergent: on recovery (DBOS) or
    cache hit (Prefect), the unit's recorded result is replayed without re-running the code, so
    the cancellation would silently not happen again; cancelling raises an explanatory
    `UserError` instead. (Temporal gets the same protection structurally: the live controller
    never crosses the activity serialization boundary.)
    """

    def __init__(self, message: str):
        super().__init__()
        self._guard_message = message

    def cancel(self) -> None:
        raise UserError(self._guard_message)


def cancel_not_supported_message(unit_noun: str, container_noun: str) -> str:
    """The shared `ctx.cancel()` error, worded for one engine's durable unit and container."""
    return (
        f'`cancel` is not supported inside a durable {unit_noun}: the durable runtime replays '
        f"the {unit_noun}'s recorded result without re-running your code, so the cancellation "
        f'would silently not happen again on recovery. Cancel the {container_noun} instead.'
    )


def guard_run_context(ctx: RunContext[AgentDepsT], *, unit_noun: str, container_noun: str) -> RunContext[AgentDepsT]:
    """Return a copy of `ctx` whose `enqueue()` and `cancel()` raise, for user code in a durable unit.

    Used by the in-process engines (DBOS steps, Prefect tasks) that pass the live context into
    the durable unit. Temporal reconstructs its context across the activity boundary and installs
    the enqueue guard in `deserialize_run_context` instead (its `cancel` protection is
    structural: the live controller is never serialized).
    """
    return replace(
        ctx,
        pending_messages=EnqueueGuard(enqueue_not_supported_message(unit_noun, container_noun)),
        _cancellation=CancelGuard(cancel_not_supported_message(unit_noun, container_noun)),
    )


def unwrap_recorded_tool_call_result(result: Any) -> Any:
    """Unwrap a durably-recorded tool result, passing raw pre-wrapper values through.

    Engines that replay recorded durable-unit outputs (DBOS step recovery, Prefect task
    caches) may hold outputs recorded before the unit wrapped control-flow exceptions as
    values; those recordings are the raw tool result and are returned unchanged.
    """
    if isinstance(
        result,
        _ToolReturn
        | _ToolContentResult
        | _ApprovalRequired
        | _CallDeferred
        | _ModelRetry
        | _ValidationError
        | _ToolFailed,
    ):
        return unwrap_tool_call_result(result)
    return result


def resolve_tool_durable_config(
    tool: ToolsetTool[Any] | None,
    tool_name: str,
    fallback_config: Mapping[str, ToolConfig],
    *,
    metadata_key: str,
    config_type_label: str,
) -> ToolConfig:
    """Resolve a tool's durable config: tool metadata under `metadata_key` first, then `fallback_config` by name."""
    if tool is not None and tool.tool_def.metadata is not None:
        metadata_config = tool.tool_def.metadata.get(metadata_key)
        if metadata_config is False:
            return False
        if metadata_config is not None:
            if not isinstance(metadata_config, dict):
                raise UserError(
                    f'Tool {tool_name!r} has invalid {metadata_key!r} metadata: expected a dict '
                    f'(`{config_type_label}`) or `False`, got {type(metadata_config).__name__}.'
                )
            return cast('DurableConfig', metadata_config)
    return fallback_config.get(tool_name, {})


def _dispatch_args_validator(
    operation: CallToolOperation, name: str, tool: ToolsetTool[Any], config: DurableConfig
) -> Callable[..., Awaitable[None]]:
    async def args_validator_func(ctx: RunContext[Any], **args: Any) -> None:
        await operation(name, args, ctx=ctx, tool=tool, config=config)

    return args_validator_func


class DurableToolsetBase(WrapperToolset[AgentDepsT]):
    """Shared workflow/flow-side scaffolding for the engines' durable toolset wrappers.

    Mirrors [`DurableModel`][pydantic_ai.durable_exec._utils.DurableModel]: everything
    engine-specific lives in the segment callables the engine supplies, each running one
    operation inside the engine's durable unit (activity/step/task).
    """

    def __init__(
        self,
        wrapped: AbstractToolset[AgentDepsT],
        *,
        in_durable_context: Callable[[], bool],
        lifecycle: Lifecycle,
        durable_registrations: list[Any] | None,
        durable_config: Mapping[str, Any] | None = None,
    ):
        super().__init__(wrapped)
        self._in_durable_context = in_durable_context
        self._lifecycle = lifecycle
        self._run_held: RunHeldToolset[AgentDepsT] | None = None
        self.durable_registrations = durable_registrations or []
        """Opaque engine handles that must be registered with the engine (e.g. Temporal activities)."""
        self.durable_config = durable_config
        """The engine's base per-operation config for this toolset (e.g. a Temporal `ActivityConfig`)."""

    @property
    def id(self) -> str | None:
        return self.wrapped.id

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if self._lifecycle == 'enter-outside-durable':
            return self
        return await super().for_run(ctx)

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if self._lifecycle == 'enter-outside-durable':
            return self
        return await super().for_run_step(ctx)

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        return self

    def _enters_wrapped(self) -> bool:
        """Whether this wrapper is the one that enters the wrapped toolset around the run.

        `enter-in-durable-unit` hands that to the run's units, but only inside the durable context:
        outside it there are no units, so the wrapper enters it as `enter-outside-durable` does.
        """
        if self._lifecycle == 'enter-always':
            return True
        if self._lifecycle == 'enter-never':
            return False
        return not self._in_durable_context()

    def _ctx_for_unit(self, ctx: RunContext[AgentDepsT]) -> RunContext[AgentDepsT]:
        """Attach the toolset the run holds so a durable unit that can reach it reuses it."""
        if (held := self._run_held) is None:
            return ctx
        existing = ctx._run_held_toolsets or {}  # pyright: ignore[reportPrivateUsage]
        return replace(ctx, _run_held_toolsets={**existing, held.id: held})

    async def __aenter__(self) -> Self:
        if self._enters_wrapped():
            await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        try:
            if self._enters_wrapped():
                return await self.wrapped.__aexit__(*args)
            return None
        finally:
            # Whichever unit entered the toolset the run holds left it entered for the rest of the
            # run, so the run is what closes it, passing on how the run ended. Its result is
            # ignored: a toolset's teardown doesn't get to suppress the run's exception.
            if (held := self._run_held) is not None:
                await held.aclose(*args)


class DurableFunctionToolset(DurableToolsetBase[AgentDepsT]):
    def __init__(
        self,
        wrapped: FunctionToolset[AgentDepsT],
        *,
        in_durable_context: Callable[[], bool],
        call_tool_operation: CallToolOperation,
        resolve_tool_config: ResolveToolConfig,
        lifecycle: Lifecycle,
        validate_args_operation: CallToolOperation | None = None,
        resolve_validation_config: ResolveToolConfig | None = None,
        durable_registrations: list[Any] | None = None,
        durable_config: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            wrapped,
            in_durable_context=in_durable_context,
            lifecycle=lifecycle,
            durable_registrations=durable_registrations,
            durable_config=durable_config,
        )
        self._call_tool_operation = call_tool_operation
        self._resolve_tool_config = resolve_tool_config
        self._validate_args_operation = validate_args_operation
        self._resolve_validation_config = resolve_validation_config or resolve_tool_config

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        tools = await super().get_tools(ctx)
        if not self._in_durable_context():
            return tools
        return {name: self._tool_with_durable_validation(name, tool) for name, tool in tools.items()}

    def _tool_with_durable_validation(self, name: str, tool: ToolsetTool[AgentDepsT]) -> ToolsetTool[AgentDepsT]:
        if tool.args_validator_func is None:
            return tool
        config = self._resolve_validation_config(tool, name)
        if config is False or (operation := self._validate_args_operation) is None:
            return tool
        return replace(tool, args_validator_func=_dispatch_args_validator(operation, name, tool, config))

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not self._in_durable_context():
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        config = self._resolve_tool_config(tool, name)
        if config is False:
            return await self.wrapped.call_tool(name, tool_args, ctx, tool)
        return await self._call_tool_operation(name, tool_args, ctx=ctx, tool=tool, config=config)


class DurableDynamicToolset(DurableToolsetBase[AgentDepsT]):
    def __init__(
        self,
        wrapped: DynamicToolset[AgentDepsT],
        *,
        in_durable_context: Callable[[], bool],
        get_tools_operation: Callable[[RunContext[AgentDepsT]], Awaitable[DynamicToolsResult]],
        call_tool_operation: CallToolOperation,
        resolve_tool_config: ResolveToolConfig,
        lifecycle: Lifecycle,
        validate_args_operation: CallToolOperation | None = None,
        resolve_validation_config: ResolveToolConfig | None = None,
        durable_registrations: list[Any] | None = None,
        durable_config: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            wrapped,
            in_durable_context=in_durable_context,
            lifecycle=lifecycle,
            durable_registrations=durable_registrations,
            durable_config=durable_config,
        )
        self._dynamic_toolset = wrapped
        self._get_tools_operation = get_tools_operation
        self._call_tool_operation = call_tool_operation
        self._resolve_tool_config = resolve_tool_config
        self._validate_args_operation = validate_args_operation
        self._resolve_validation_config = resolve_validation_config or resolve_tool_config
        self._run_instructions: Instructions = None

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if not self._in_durable_context():
            # Fully transparent outside the durable context: resolve the dynamic toolset
            # and hand the run its resolved form directly, without the durable dispatch.
            # (The wrapped `DynamicToolset` only resolves in `for_run`; delegating the
            # individual methods to the unresolved factory would silently yield no tools.)
            return await self.wrapped.for_run(ctx)
        # Per-run copy isolates `_run_instructions` and `_run_held` from the process-shared
        # instance. The shallow copy shares the engine-registered operations; this is only state
        # isolation.
        run_copy = copy.copy(self)
        run_copy._run_instructions = None
        run_copy._run_held = None
        if not self._dynamic_toolset.per_run_step and (toolset_id := self._dynamic_toolset.id) is not None:
            # `per_run_step=False` is the factory's own statement that one resolution covers the
            # run, so resolve it here like a non-durable run does, leaving entry to the first
            # durable unit that needs the toolset. This runs the factory in container code, where
            # it must be deterministic and leave its I/O to the units. A `per_run_step=True`
            # factory is re-evaluated per unit as before: its `for_run_step` swaps the inner
            # toolset in place, which parallel tool-call units must not share.
            run_copy._run_held = RunHeldToolset(toolset_id, await self._dynamic_toolset.for_run(ctx))
        return run_copy

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # The per-run copy is stable across steps: a `per_run_step=True` factory is re-evaluated
        # inside the durable units, not in workflow/flow code here, and a `per_run_step=False` one
        # was resolved once in `for_run`. (Outside the durable context this wrapper isn't in the
        # run's tree at all — `for_run` above replaced it with the resolved toolset.)
        return self

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        result = await self._get_tools_operation(self._ctx_for_unit(ctx))
        self._run_instructions = result.instructions
        return {name: self._tool_for_info(name, info) for name, info in result.tools.items()}

    def _tool_for_info(self, name: str, info: DynamicToolInfo) -> ToolsetTool[AgentDepsT]:
        tool = ToolsetTool[AgentDepsT](
            toolset=self,
            tool_def=info.tool_def,
            max_retries=info.max_retries,
            # Only parse here; the real tool validates again inside the durable unit.
            args_validator=TOOL_SCHEMA_VALIDATOR,
        )
        if not info.has_args_validator:
            return tool
        config = self._resolve_validation_config(tool, name)
        if config is False:

            async def args_validator_func(ctx: RunContext[AgentDepsT], **args: Any) -> None:
                await validate_dynamic_tool_args(
                    self.wrapped, name, args, self._ctx_for_unit(ctx), tool_def=tool.tool_def
                )

            return replace(tool, args_validator_func=args_validator_func)
        if (operation := self._validate_args_operation) is None:
            raise UserError(
                f'Tool {name!r} in dynamic toolset {self.id!r} has an `args_validator`, but the durable '
                'engine has no validation unit to run it in. An `args_validator` is a Python callable that '
                "cannot cross the durable boundary, so it can't be run in workflow/flow code against a tool "
                'that only exists inside the durable unit. Remove the `args_validator`, or validate the '
                'arguments in the tool function itself.'
            )
        dispatch = _dispatch_args_validator(operation, name, tool, config)

        async def dispatch_in_unit(ctx: RunContext[AgentDepsT], **args: Any) -> None:
            await dispatch(self._ctx_for_unit(ctx), **args)

        return replace(tool, args_validator_func=dispatch_in_unit)

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> Instructions:
        # Set by `get_tools`, which the framework runs earlier in each step.
        return self._run_instructions

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        config = self._resolve_tool_config(tool, name)
        if config is False:
            # The wrapped dynamic toolset is only a construction-time factory, so an
            # explicitly inline call resolves one in flow code — reusing the run's resolved
            # toolset when there is one, like the durable units do.
            return await call_dynamic_tool(
                self.wrapped, name, tool_args, self._ctx_for_unit(ctx), tool_def=tool.tool_def
            )
        return await self._call_tool_operation(name, tool_args, ctx=self._ctx_for_unit(ctx), tool=tool, config=config)


class DurableMCPToolset(DurableToolsetBase[AgentDepsT]):
    def __init__(
        self,
        wrapped: MCPToolset[AgentDepsT],
        *,
        in_durable_context: Callable[[], bool],
        get_tools_operation: Callable[[RunContext[AgentDepsT]], Awaitable[dict[str, ToolDefinition]]] | None,
        get_instructions_operation: Callable[[RunContext[AgentDepsT]], Awaitable[Instructions]] | None,
        call_tool_operation: CallToolOperation,
        resolve_tool_config: ResolveToolConfig,
        lifecycle: Lifecycle,
        durable_registrations: list[Any] | None = None,
        durable_config: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            wrapped,
            in_durable_context=in_durable_context,
            lifecycle=lifecycle,
            durable_registrations=durable_registrations,
            durable_config=durable_config,
        )
        self._mcp_toolset = wrapped
        self._get_tools_operation = get_tools_operation
        self._get_instructions_operation = get_instructions_operation
        self._call_tool_operation = call_tool_operation
        self._resolve_tool_config = resolve_tool_config

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        if (
            self._lifecycle != 'enter-in-durable-unit'
            or not self._in_durable_context()
            or (toolset_id := self.id) is None
        ):
            return await super().for_run(ctx)
        # Per-run copy isolates `_run_held` from the process-shared instance; the shallow copy
        # shares the engine-registered operations, so this is only state isolation.
        run_copy = copy.copy(self)
        # One server session covers the run, the way it does outside a durable container, so the
        # server is connected to once instead of once per unit and `cache_tools` survives between
        # units. Entry is refcounted by the toolset itself, so concurrent runs sharing this
        # process-wide toolset share its session, and it stays open until the last of them ends.
        run_copy._run_held = RunHeldToolset(toolset_id, self.wrapped)
        return run_copy

    async def get_tools(self, ctx: RunContext[AgentDepsT]) -> dict[str, ToolsetTool[AgentDepsT]]:
        if not self._in_durable_context() or self._get_tools_operation is None:
            return await self.wrapped.get_tools(ctx)
        cache_key = self.id or ''
        if self._mcp_toolset.cache_tools and (cached := ctx._mcp_tool_defs_cache.get(cache_key)) is not None:  # pyright: ignore[reportPrivateUsage]
            return {name: self._mcp_toolset.tool_for_tool_def(tool_def, ctx=ctx) for name, tool_def in cached.items()}
        tool_defs = await self._get_tools_operation(self._ctx_for_unit(ctx))
        if self._mcp_toolset.cache_tools:
            ctx._mcp_tool_defs_cache[cache_key] = tool_defs  # pyright: ignore[reportPrivateUsage]
        return {name: self._mcp_toolset.tool_for_tool_def(tool_def, ctx=ctx) for name, tool_def in tool_defs.items()}

    async def get_instructions(self, ctx: RunContext[AgentDepsT]) -> Instructions:
        if not self._mcp_toolset.include_instructions:
            return None
        if not self._in_durable_context() or self._get_instructions_operation is None:
            return await self._mcp_toolset.get_instructions(ctx)
        # Always route through the durable unit: deciding based on locally-cached state (e.g.
        # instructions a warm in-process MCP server already holds) would make the durable
        # schedule depend on process warmth and diverge on replay/recovery (#5884).
        return await self._get_instructions_operation(self._ctx_for_unit(ctx))

    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[AgentDepsT], tool: ToolsetTool[AgentDepsT]
    ) -> Any:
        if not self._in_durable_context():
            return await self._mcp_toolset.call_tool(name, tool_args, ctx, tool)
        config = self._resolve_tool_config(tool, name)
        if config is False:
            return await self._mcp_toolset.call_tool(name, tool_args, ctx, tool)
        return await self._call_tool_operation(name, tool_args, ctx=self._ctx_for_unit(ctx), tool=tool, config=config)
