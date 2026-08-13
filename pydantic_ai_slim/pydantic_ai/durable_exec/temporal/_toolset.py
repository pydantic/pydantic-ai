from __future__ import annotations

import asyncio
import copy
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from contextlib import asynccontextmanager, contextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast, get_type_hints

from pydantic import ConfigDict, TypeAdapter, ValidationError, with_config
from pydantic.errors import PydanticUserError
from temporalio import activity, workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
from temporalio.workflow import ActivityConfig
from typing_extensions import Self, TypedDict

from pydantic_ai import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.durable_exec._toolset import (
    CallToolResult,
    DurableToolsetBase,
    resolve_tool_durable_config,
    unwrap_tool_call_result,
    wrap_tool_call_result,
)
from pydantic_ai.exceptions import FallbackExceptionGroup, UnexpectedModelBehavior, UserError
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets._dynamic import DynamicToolset

from ._run_context import TemporalRunContext

if TYPE_CHECKING:
    from pydantic_ai.agent.abstract import AbstractAgent


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class GetToolsParams:
    serialized_run_context: Any


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class CallToolParams:
    name: str
    tool_args: dict[str, Any]
    serialized_run_context: Any
    tool_def: ToolDefinition | None
    original_name: str | None = None
    """The name the toolset holds the tool under, when a `prepare` function renamed it in `tool_def.name`."""


@asynccontextmanager
async def heartbeating() -> AsyncGenerator[None]:
    """Emit periodic activity heartbeats in the background while the wrapped activity body runs.

    Every activity we register beats, so that a long-but-healthy activity isn't mistaken for a
    crashed worker, and so workflow cancellation stays deliverable (cancellation reaches an
    activity as a response to a heartbeat).

    The beat interval is derived from the activity's configured `heartbeat_timeout` so a
    custom (shorter or longer) timeout keeps working; the SDK additionally throttles
    outgoing heartbeats on its own. Without a configured timeout, heartbeats are inert but
    harmless, so a plain 5-second cadence is fine.

    The heartbeat task is supervised: if `beat()` itself crashes, the failure surfaces
    once the wrapped body completes, so the activity fails loudly instead of having
    silently run without heartbeats (the server would have failed the attempt via
    `heartbeat_timeout` anyway had the crash come early). An exception from the wrapped
    body always wins — a heartbeat failure never replaces it.
    """

    async def beat() -> None:
        timeout = activity.info().heartbeat_timeout
        interval = timeout.total_seconds() / 2 if timeout else 5.0
        while True:
            activity.heartbeat()
            await asyncio.sleep(interval)

    task = asyncio.create_task(beat())
    try:
        yield
    except BaseException:
        # The body's exception is already propagating; a heartbeat failure must not replace it.
        task.cancel()
        with suppress(BaseException):
            await task
        raise
    else:
        task.cancel()
        with suppress(asyncio.CancelledError):
            # Anything but our own cancellation is a `beat()` crash — propagate it.
            await task


class TemporalWrapperToolset(WrapperToolset[AgentDepsT], ABC):
    @property
    def id(self) -> str:
        # An error is raised in `TemporalAgent` if no `id` is set.
        assert self.wrapped.id is not None
        return self.wrapped.id

    @property
    @abstractmethod
    def temporal_activities(self) -> list[Callable[..., Any]]:
        raise NotImplementedError

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # Temporal-wrapped toolsets manage their wrapped toolset's lifecycle
        # per-activity (inside activities), not per-run.
        return self  # pragma: no cover

    async def for_run_step(self, ctx: RunContext[AgentDepsT]) -> AbstractToolset[AgentDepsT]:
        # Temporal-wrapped toolsets manage their wrapped toolset's lifecycle
        # per-activity (inside activities), not per-run-step.
        return self

    def visit_and_replace(
        self, visitor: Callable[[AbstractToolset[AgentDepsT]], AbstractToolset[AgentDepsT]]
    ) -> AbstractToolset[AgentDepsT]:
        # Temporalized toolsets cannot be swapped out after the fact.
        return self  # pragma: no cover

    async def __aenter__(self) -> Self:
        if not workflow.in_workflow():
            await self.wrapped.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if not workflow.in_workflow():
            return await self.wrapped.__aexit__(*args)
        return None

    async def _wrap_call_tool_result(self, coro: Awaitable[Any]) -> CallToolResult:
        return await wrap_tool_call_result(coro)

    def _unwrap_call_tool_result(self, result: CallToolResult) -> Any:
        return unwrap_tool_call_result(result)


PAYLOAD_SIZE_ERROR_TYPE = 'PayloadSizeError'
"""The failure type Temporal stamps on an activity whose payload exceeds the server blob-size limit.

Matched by name rather than by class: the SDK raises a private `_PayloadSizeError` from inside its own
result-encoding step, after the activity function has returned, so only the converted failure type ever
reaches code we own.

The name is stamped by Temporal's `DefaultFailureConverter`, which special-cases that private class; any
other converter falls back to the class name. As `PydanticAIPlugin` deliberately preserves a custom
`failure_converter_class`, a user who supplies one that doesn't special-case it loses both this guard and
the non-retryable entry below.
"""


def with_non_retryable_errors(retry_policy: RetryPolicy | None) -> RetryPolicy:
    """Return a copy of `retry_policy` with the framework's non-retryable errors ensured."""
    retry_policy = copy.copy(retry_policy) if retry_policy else RetryPolicy()
    existing = retry_policy.non_retryable_error_types or []
    additional = [
        UserError.__name__,
        PydanticUserError.__name__,
        UnexpectedModelBehavior.__name__,
        FallbackExceptionGroup.__name__,
        # An over-limit payload is deterministic, so Temporal's default unlimited retries would resend the
        # same oversized result forever and hang the workflow instead of ever surfacing an error (#7110).
        PAYLOAD_SIZE_ERROR_TYPE,
    ]
    retry_policy.non_retryable_error_types = [*existing, *(name for name in additional if name not in existing)]
    return retry_policy


@contextmanager
def payload_size_errors(subject: str, remedy: str) -> Generator[None]:
    """Re-raise an over-limit activity payload as a `UserError` that points at the cause.

    Temporal rejects the payload during result encoding, so the failure that reaches the workflow names
    only a byte count. Without this, an activity returning a large `BinaryImage` fails with a bare
    `[TMPRL1103] ... Size: N bytes, Limit: M bytes` that mentions neither what produced it, nor the
    image, nor Pydantic AI, leaving no way to get from the error to the fix (#7110).

    The guard sits at the workflow side of the boundary rather than pre-checking the result size inside
    the activity, because the limit is a server setting an activity cannot read: Temporal reports it to
    the worker at startup and keeps it on a private data converter. Catching what Temporal actually
    rejected reports the real limit instead of a hardcoded guess at it.

    Nothing here establishes what made the payload large, so the message offers the base64 budget as a
    likely explanation rather than asserting it: an over-limit payload can just as well be a large JSON
    tool return.

    Args:
        subject: What produced the over-limit payload, as a sentence without its final period.
        remedy: What the user can do about it, specific to what produced the payload.
    """
    try:
        yield
    except ActivityError as exc:
        cause = exc.__cause__
        if not isinstance(cause, ApplicationError) or cause.type != PAYLOAD_SIZE_ERROR_TYPE:
            raise
        raise UserError(
            f'{subject}. {cause.message}. '
            'Binary content like an image is base64-encoded into the activity payload, so if that is the '
            f'cause, the raw-byte budget is about three quarters of the limit — roughly 1.5MB at the 2MB '
            f'default. {remedy} To keep large payloads out of the workflow history without changing what '
            'your tools or models return, configure Temporal external storage (or a claim-check '
            '`payload_codec`) on your `DataConverter` — `PydanticAIPlugin` preserves it, and it covers '
            'every payload in both directions. '
            'See https://ai.pydantic.dev/durable_execution/temporal/#large-payloads'
        ) from exc


@contextmanager
def tool_result_payload_errors(tool_name: str) -> Generator[None]:
    """Guard a tool-call activity's result against Temporal's payload size limit."""
    with payload_size_errors(
        f'Tool {tool_name!r} returned a result too large for Temporal',
        'Return a reference instead of the value itself, like a URL or a key your application resolves later.',
    ):
        yield


@contextmanager
def model_response_payload_errors(model_name: str) -> Generator[None]:
    """Guard a model-request activity's response against Temporal's payload size limit."""
    with payload_size_errors(
        f'The response from model {model_name!r} is too large for Temporal',
        'A generated image is the usual cause, so ask the model for a smaller one through the model '
        'settings; a streamed segment can also overflow on its buffered events alone.',
    ):
        yield


_ValidatedActivityConfig = with_config(ConfigDict(extra='forbid'))(
    TypedDict(
        '_ValidatedActivityConfig',
        # The functional syntax is intentionally dynamic so new Temporal keys are included.
        get_type_hints(ActivityConfig, include_extras=True),  # pyright: ignore[reportArgumentType]
        total=ActivityConfig.__total__,
    )
)
"""A `typing_extensions.TypedDict` copy of `ActivityConfig` with unknown keys forbidden.

The copy is derived so it stays in sync when Temporal adds keys. `typing_extensions.TypedDict` is
required because Pydantic cannot generate schemas for `typing.TypedDict` on Python before 3.12.
"""


# Pyright cannot see that the dynamic `TypedDict` has the exact `ActivityConfig` annotations.
_activity_config_adapter = cast('TypeAdapter[ActivityConfig]', TypeAdapter(_ValidatedActivityConfig))


def validate_activity_config(config: ActivityConfig, source: str) -> ActivityConfig:
    """Return `config` validated into Temporal's own types, or raise a `UserError`.

    Unknown keys survive `ActivityConfig` construction (it's a `total=False` `TypedDict`) and only
    fail once they're splatted into `workflow.start_activity()` inside the workflow, where the
    resulting `TypeError` isn't one of `PydanticAIPlugin`'s `workflow_failure_exception_types` and
    so fails the workflow *task*, which Temporal retries forever.

    The validated config is returned rather than discarded because validation also coerces: a
    `'PT5M'` that came back from a round trip becomes a `timedelta`, and only the coerced value is
    something `start_activity()` accepts.

    `source` names where the config came from, for example '`model_activity_config`'.
    """
    try:
        return _activity_config_adapter.validate_python(config)
    except ValidationError as e:
        raise UserError(f'Invalid Temporal `ActivityConfig` in {source}: {e}') from e


def resolve_tool_activity_config(
    tool: ToolsetTool[Any] | None,
    tool_name: str,
    tool_activity_config: Mapping[str, ActivityConfig | Literal[False]],
) -> ActivityConfig | Literal[False]:
    """Resolve per-tool Temporal activity config.

    Reads `tool.tool_def.metadata['temporal']` first, then falls back to the explicit
    `tool_activity_config` dict keyed by tool name. Returns an `ActivityConfig` dict
    (possibly empty), or `False` to skip activity wrapping.

    The config is validated back into Temporal's own types: a `DynamicToolset`'s tools are
    discovered inside the get-tools activity, so their `ToolDefinition.metadata` returns to the
    workflow as JSON, where `timedelta(minutes=5)` has become `'PT5M'`, a `RetryPolicy` a plain
    dict, and an `ActivityCancellationType` an int. Handing those to
    `workflow.execute_activity` fails the workflow *task*, which Temporal retries forever;
    a `UserError` for what validation can't restore fails the workflow instead.
    """
    config = cast(
        'ActivityConfig | Literal[False]',
        resolve_tool_durable_config(
            tool,
            tool_name,
            tool_activity_config,
            metadata_key='temporal',
            config_type_label='ActivityConfig',
        ),
    )
    if config is False:
        return False
    try:
        config = _activity_config_adapter.validate_python(config)
    except ValidationError as e:
        raise UserError(f'Tool {tool_name!r} has an invalid Temporal `ActivityConfig`: {e}') from e
    if 'retry_policy' in config:
        config['retry_policy'] = with_non_retryable_errors(config.get('retry_policy'))
    return config


def toolset_temporal_activities(toolset: AbstractToolset[Any]) -> list[Callable[..., Any]]:
    """The Temporal activities a durable-wrapped toolset needs registered with the worker."""
    if isinstance(toolset, DurableToolsetBase):
        return toolset.durable_registrations
    if isinstance(toolset, TemporalWrapperToolset):
        return toolset.temporal_activities
    return []


async def call_tool_in_activity(
    toolset: AbstractToolset[AgentDepsT],
    name: str,
    tool_args: dict[str, Any],
    ctx: RunContext[AgentDepsT],
    tool: ToolsetTool[AgentDepsT],
) -> CallToolResult:
    args = tool.args_validator.validate_python(tool_args)
    return await wrap_tool_call_result(toolset.call_tool(name, args, ctx, tool))


def temporalize_toolset(
    toolset: AbstractToolset[AgentDepsT],
    activity_name_prefix: str,
    activity_config: ActivityConfig,
    tool_activity_config: dict[str, ActivityConfig | Literal[False]],
    deps_type: type[AgentDepsT],
    run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    agent: AbstractAgent[AgentDepsT, Any] | None = None,
) -> AbstractToolset[AgentDepsT]:
    """Temporalize a toolset.

    Args:
        toolset: The toolset to temporalize.
        activity_name_prefix: Prefix for Temporal activity names.
        activity_config: The Temporal activity config to use.
        tool_activity_config: The Temporal activity config to use for specific tools identified by tool name.
        deps_type: The type of agent's dependencies object. It needs to be serializable using Pydantic's `TypeAdapter`.
        run_context_type: The `TemporalRunContext` (sub)class that's used to serialize and deserialize the run context.
        agent: The agent instance to attach to deserialized run contexts in activities.
    """
    if isinstance(toolset, FunctionToolset):
        from ._function_toolset import temporalize_function_toolset

        return temporalize_function_toolset(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
            agent=agent,
        )

    if isinstance(toolset, DynamicToolset):
        from ._dynamic_toolset import temporalize_dynamic_toolset

        return temporalize_dynamic_toolset(
            toolset,
            activity_name_prefix=activity_name_prefix,
            activity_config=activity_config,
            tool_activity_config=tool_activity_config,
            deps_type=deps_type,
            run_context_type=run_context_type,
            agent=agent,
        )

    try:
        from pydantic_ai.mcp import MCPToolset

        from ._mcp_toolset import temporalize_mcp_toolset
    except ImportError:
        pass
    else:
        if isinstance(toolset, MCPToolset):
            return temporalize_mcp_toolset(
                toolset,
                activity_name_prefix=activity_name_prefix,
                activity_config=activity_config,
                tool_activity_config=tool_activity_config,
                deps_type=deps_type,
                run_context_type=run_context_type,
                agent=agent,
            )

    return toolset
