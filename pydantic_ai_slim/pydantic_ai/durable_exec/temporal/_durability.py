from __future__ import annotations

from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypeAlias, cast

from pydantic import ConfigDict, with_config
from pydantic_core import PydanticSerializationError
from temporalio import workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai import messages as _messages
from pydantic_ai._agent_graph import set_agent_graph_sleep
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import (
    AbstractCapability,
    WrapRunHandler,
)
from pydantic_ai.durable_exec._base import (
    BaseDurabilityCapability,
    CancelSuspendedResponseOperationParams,
    EventStreamHandlerOperationParams as _SemanticEventStreamHandlerParams,
    ModelRequestOperationParams,
    ToolsetKind,
    _CallToolParams,  # pyright: ignore[reportPrivateUsage]
    _GetToolsParams,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.durable_exec._codec import IDENTITY_CODEC
from pydantic_ai.durable_exec._operation import DurableOperationId
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._toolset import CallToolResult, DurableToolsetBase, Lifecycle
from pydantic_ai.durable_exec._utils import StreamedActivityResult, disable_threads
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import AgentStreamEvent, ModelResponse
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, infer_model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.toolsets.function import FunctionToolsetTool

if TYPE_CHECKING:
    pass

from ._operation_backend import TemporalOperationBackend
from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import (
    CallToolParams,
    GetToolsParams,
    TemporalWrapperToolset,
    resolve_tool_activity_config,
    temporalize_toolset as _default_temporalize_toolset,
    tool_result_payload_errors,
    toolset_temporal_activities,
    validate_activity_config,
    with_non_retryable_errors,
)


class _FunctionCallTransport:
    wire_type = CallToolParams
    result_type = CallToolResult

    def __init__(self, durability: TemporalDurability[Any], toolset: FunctionToolset[Any]) -> None:
        self._durability = durability
        self._toolset = toolset

    def dump(self, params: _CallToolParams) -> tuple[CallToolParams, Any]:
        assert params.tool is not None
        tool = params.tool
        return (
            CallToolParams(
                name=params.name,
                tool_args=params.tool_args,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx),
                tool_def=tool.tool_def,
                original_name=tool.original_name if isinstance(tool, FunctionToolsetTool) else None,
            ),
            params.ctx.deps,
        )

    def load(self, payload: tuple[CallToolParams, Any], *, runtime: object) -> _CallToolParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        try:
            tool = (
                self._toolset.tool_for_tool_def(params.tool_def, ctx=ctx, original_name=params.original_name)
                if params.tool_def is not None
                else None
            )
        except KeyError as exc:
            raise UserError(
                f'Tool {params.name!r} not found in toolset {self._toolset.id!r}. '
                'Removing or renaming tools during an agent run is not supported with Temporal.'
            ) from exc
        return _CallToolParams(params.name, params.tool_args, ctx, tool)


class _GetToolsTransport:
    wire_type = GetToolsParams
    result_type = dict[str, ToolDefinition]

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: _GetToolsParams) -> tuple[GetToolsParams, Any]:
        return (
            GetToolsParams(serialized_run_context=self._durability.run_context_type.serialize_run_context(params.ctx)),
            params.ctx.deps,
        )

    def load(self, payload: tuple[GetToolsParams, Any], *, runtime: object) -> _GetToolsParams:
        params, deps = payload
        return _GetToolsParams(self._durability.deserialize_operation_run_context(params.serialized_run_context, deps))


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _RequestParams:
    """Serializable arguments for the model-request Temporal activity."""

    messages: list[_messages.ModelMessage]
    # `model_settings` can't be a `ModelSettings` because Temporal would end up dropping fields only defined on its subclasses.
    model_settings: dict[str, Any] | None
    model_request_parameters: ModelRequestParameters
    serialized_run_context: Any
    model_id: str | None = None


@dataclass
class _CancelParams:
    response: ModelResponse
    model_id: str | None = None
    serialized_run_context: Any = None


@dataclass
@with_config(ConfigDict(arbitrary_types_allowed=True))
class _EventStreamHandlerParams:
    event: AgentStreamEvent
    serialized_run_context: Any


# The `ModelResponse` arm decodes histories recorded by the deprecated `TemporalAgent`, whose
# stream activity returned the bare response. Remove it (and the workflow-side event synthesis
# in `request_stream_segment`) once those histories have aged out, along with `TemporalAgent`.
_StreamedActivityPayload: TypeAlias = StreamedActivityResult | ModelResponse


class _ModelRequestTransport:
    wire_type = _RequestParams

    def __init__(self, durability: TemporalDurability[Any], *, result_type: object) -> None:
        self._durability = durability
        self.result_type = result_type

    def dump(self, params: ModelRequestOperationParams) -> tuple[_RequestParams, Any]:
        ctx = params.run_context
        return (
            _RequestParams(
                messages=params.messages,
                model_settings=cast(dict[str, Any] | None, params.model_settings),
                model_request_parameters=params.model_request_parameters,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
                model_id=params.model_id,
            ),
            ctx.deps,
        )

    def load(self, payload: tuple[_RequestParams, Any], *, runtime: object) -> ModelRequestOperationParams:
        request, deps = payload
        ctx = self._durability.deserialize_operation_run_context(request.serialized_run_context, deps)
        return ModelRequestOperationParams(
            request.model_id,
            request.messages,
            cast(ModelSettings | None, request.model_settings),
            request.model_request_parameters,
            ctx,
        )


class _CancelTransport:
    wire_type = _CancelParams
    result_type = type(None)

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: CancelSuspendedResponseOperationParams) -> tuple[_CancelParams, Any]:
        ctx = params.run_context
        return (
            _CancelParams(
                response=params.response,
                model_id=params.model_id,
                serialized_run_context=(
                    self._durability.run_context_type.serialize_run_context(ctx) if ctx is not None else None
                ),
            ),
            ctx.deps if ctx is not None else None,
        )

    def load(self, payload: tuple[_CancelParams, Any], *, runtime: object) -> CancelSuspendedResponseOperationParams:
        params, deps = payload
        ctx = (
            self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
            if params.serialized_run_context is not None
            else None
        )
        return CancelSuspendedResponseOperationParams(params.model_id, params.response, ctx)


class _EventStreamHandlerTransport:
    wire_type = _EventStreamHandlerParams
    result_type = type(None)

    def __init__(self, durability: TemporalDurability[Any]) -> None:
        self._durability = durability

    def dump(self, params: _SemanticEventStreamHandlerParams) -> tuple[_EventStreamHandlerParams, Any]:
        ctx = params.run_context
        return (
            _EventStreamHandlerParams(
                event=params.event,
                serialized_run_context=self._durability.run_context_type.serialize_run_context(ctx),
            ),
            ctx.deps,
        )

    def load(
        self, payload: tuple[_EventStreamHandlerParams, Any], *, runtime: object
    ) -> _SemanticEventStreamHandlerParams:
        params, deps = payload
        ctx = self._durability.deserialize_operation_run_context(params.serialized_run_context, deps)
        return _SemanticEventStreamHandlerParams(params.event, ctx)


_DEFAULT_MODEL_HEARTBEAT_TIMEOUT = timedelta(seconds=30)
"""Default `heartbeat_timeout` for the model-request activities.

A model request activity can legitimately run for a long time while waiting for one
provider round trip, and heartbeating (see `heartbeating`) lets Temporal distinguish that
long-but-healthy activity from a crashed worker. Tool activities deliberately get no default:
a CPU-bound tool can starve the heartbeat task, and failing it for a missed heartbeat would
be a regression against no timeout at all.
"""


def serialization_user_error(error: PydanticSerializationError) -> UserError:
    """Explain a serialization failure that happened while scheduling a Temporal activity.

    The failing value isn't identifiable from here — activity arguments are encoded by
    Temporal's payload converter, which reports the offending type but not the argument it
    came from — so the message names the values the framework passes rather than claiming
    it was `deps`.
    """
    return UserError(
        f'A value passed to a Temporal activity failed to be serialized ({error}). '
        "Temporal requires all values that are passed to activities to be serializable using Pydantic's "
        '`TypeAdapter`. Besides `deps`, this includes `model_settings`, the `RunContext` `metadata` and '
        '`tool_call_metadata`, and tool `metadata`.'
    )


IMAGE_OUTPUT_UNSUPPORTED_MESSAGE = (
    'Image output is not supported with Temporal because the image would ride the activity payload, '
    'which is capped by the server blob-size limit (2MB by default, leaving about 1.5MB of raw image '
    'bytes once base64-encoded).'
)
"""Shared by the capability and the deprecated `TemporalModel`, which reject image output identically."""


@dataclass(init=False)
class TemporalDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through Temporal activities.

    When added to an agent, this capability intercepts model requests and
    wraps toolsets to route their I/O through Temporal activities.
    Outside of workflows, the capability is transparent.

    The capability discovers the agent's model, name, and toolsets
    automatically via `for_agent()`. Only Temporal-specific configuration
    needs to be passed to the constructor.

    Example:
        ```python {test="skip"}
        from pydantic_ai import Agent
        from pydantic_ai.durable_exec.temporal import TemporalDurability

        durability = TemporalDurability()
        agent = Agent('openai:gpt-5.6-sol', name='my_agent', capabilities=[durability])
        ```
    """

    engine_name = 'Temporal'
    _codec: ClassVar = IDENTITY_CODEC
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )
    _wrapped_toolset_kinds: ClassVar[frozenset[ToolsetKind]] = frozenset({'function', 'mcp', 'dynamic'})
    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-outside-durable',
        'mcp': 'enter-outside-durable',
        'dynamic': 'enter-never',
    }
    _tool_call_result_upgrade_lenient = False
    _journal_discovery = True

    _durable_unit_noun = 'activity'
    _durable_container_noun = 'workflow'
    _tool_config_key = 'temporal'

    run_context_type: type[TemporalRunContext[AgentDepsT]]
    """The `TemporalRunContext` subclass used to serialize/deserialize the run context."""

    activity_config: ActivityConfig
    """Base Temporal activity config used for all activities."""

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        deps_type: type[AgentDepsT] | None = None,
        activity_config: ActivityConfig | None = None,
        model_activity_config: ActivityConfig | None = None,
        event_stream_handler_activity_config: ActivityConfig | None = None,
        toolset_activity_config: dict[str, ActivityConfig] | None = None,
        run_context_type: type[TemporalRunContext[AgentDepsT]] = TemporalRunContext[AgentDepsT],
    ):
        """Create a TemporalDurability capability.

        The agent's model, name, and toolsets are discovered automatically
        when the capability is attached to an agent (via `for_agent()`).

        Args:
            models: Optional additional models keyed by ID for runtime model
                switching. The agent's primary model is always registered as
                `'default'`. A `Model` instance can't be serialized across the
                activity boundary, so a run-time model (via `agent.run(model=...)`
                / `agent.override(model=...)`, or swapped in by an outer capability)
                has to be registered here and referenced by key (or passed as the
                registered instance); an unregistered instance is rejected, because
                rebuilding it from its `model_id` would build a different model.
                Model-name strings never need registering: they cross as the string
                the caller wrote and are built on the worker by the agent's
                `resolve_model_id` capability chain, then `infer_model`. To build a
                specific instance on the worker from such a string — a custom
                provider, or per-user credentials carried on `deps` — use the
                [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability.
            event_stream_handler: Optional event stream handler. Model events are handled
                live inside model-request activities, and tool events are handled in
                per-event activities.
            name: Unique agent name used in the Temporal activity names. Defaults to the agent's
                `name` when the capability is bound.
            deps_type: The type of the agent's dependencies, needed for Temporal
                serialization of activity parameters. Defaults to the agent's own
                `deps_type`, discovered when the capability binds via `for_agent()`.
            activity_config: Base Temporal activity config for all activities.
                Defaults to a 60-second `start_to_close_timeout`.
            model_activity_config: Activity config merged on top of the base for
                model request activities.
            event_stream_handler_activity_config: Activity config merged on top of the base for
                event stream handler activities.
            toolset_activity_config: Per-toolset activity configs keyed by toolset ID,
                merged on top of the base config.
            run_context_type: The `TemporalRunContext` subclass for run context
                serialization/deserialization.

        Note:
            Per-tool activity config (custom timeouts, retry policies, or disabling
            activity wrapping entirely) is configured via tool metadata:

            ```python {test="skip" lint="skip"}
            @my_toolset.tool(metadata={'temporal': ActivityConfig(...)})
            async def my_slow_tool(...): ...
            ```

            or via the `SetToolMetadata` capability for selector-based config.
            Setting the `'temporal'` key to `False` skips activity wrapping
            (only valid for async tool functions).
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self.run_context_type = run_context_type
        self._deps_type = deps_type

        # An unknown key, or a value Temporal's own types don't accept, would only fail when the
        # config is splatted into `workflow.start_activity()` inside the workflow, where the
        # `TypeError` wedges the workflow task forever. Validation also *coerces* — a
        # round-tripped `'PT5M'` becomes a `timedelta` — so the validated config is what we keep.
        if activity_config is not None:
            activity_config = validate_activity_config(activity_config, '`activity_config`')
        if model_activity_config is not None:
            model_activity_config = validate_activity_config(model_activity_config, '`model_activity_config`')
        if event_stream_handler_activity_config is not None:
            event_stream_handler_activity_config = validate_activity_config(
                event_stream_handler_activity_config, '`event_stream_handler_activity_config`'
            )
        toolset_activity_config = {
            ts_id: validate_activity_config(config, f'`toolset_activity_config[{ts_id!r}]`')
            for ts_id, config in (toolset_activity_config or {}).items()
        }

        # Normalize the activity config on copies: mutating the caller's `ActivityConfig` or a
        # `RetryPolicy` shared with other activities would leak the non-retryable entries into
        # them, and repeated construction from the same config would accumulate duplicates.
        activity_config = (
            activity_config.copy() if activity_config else ActivityConfig(start_to_close_timeout=timedelta(seconds=60))
        )
        activity_config['retry_policy'] = with_non_retryable_errors(activity_config.get('retry_policy'))
        self.activity_config = activity_config
        # All activities heartbeat in the background (see `heartbeating`), but only the model
        # ones get a heartbeat timeout by default; an explicit `heartbeat_timeout` in either
        # config wins.
        self._model_activity_config: ActivityConfig = {
            'heartbeat_timeout': _DEFAULT_MODEL_HEARTBEAT_TIMEOUT,
            **activity_config,
            **(model_activity_config or {}),
        }
        # A `retry_policy` in `model_activity_config` would otherwise replace the normalized
        # base policy and drop the non-retryable entries.
        self._model_activity_config['retry_policy'] = with_non_retryable_errors(
            self._model_activity_config.get('retry_policy')
        )
        self._event_stream_handler_activity_config: ActivityConfig = {
            **activity_config,
            **(event_stream_handler_activity_config or {}),
        }
        self._event_stream_handler_activity_config['retry_policy'] = with_non_retryable_errors(
            self._event_stream_handler_activity_config.get('retry_policy')
        )
        self._toolset_activity_config = toolset_activity_config or {}

        # Populated by for_agent().
        self._operation_backend: TemporalOperationBackend | None = None

    def _check_bindable(self) -> None:
        if self.in_durable_context:
            raise UserError(
                'An agent with `TemporalDurability` must be constructed outside of a Temporal workflow, '
                'so its activities can be registered with the worker before the workflow runs. '
                'Construct the agent at module level (or in worker setup code) and reference it from the workflow.'
            )

    def deserialize_operation_run_context(self, serialized_run_context: Any, deps: Any) -> RunContext[AgentDepsT]:
        return deserialize_run_context(self.run_context_type, serialized_run_context, deps=deps, agent=self._agent)

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        # Discover the deps type from the agent unless explicitly configured.
        if self._deps_type is None:
            self._deps_type = cast('type[AgentDepsT]', agent.deps_type)

        assert self._deps_type is not None
        self._operation_backend = TemporalOperationBackend(
            agent_name=self.name,
            deps_type=self._deps_type,
            model_config=self._model_activity_config,
            event_config=self._event_stream_handler_activity_config,
            tool_config=self.activity_config,
            resolve_tool_config=self._resolve_temporal_tool_config,
            runtime=self,
        )
        self._register_activities(agent)

    def _register_activities(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        """Bind common model/event operations and adopt the existing toolset activities."""
        backend = self._operation_backend
        assert backend is not None

        default_model = self._models_by_id.get('default')
        model_name = self._default_model_id or (default_model.model_id if default_model is not None else 'default')
        self._bound_model_operations = self._bind_model_operations(backend, model_id=None, model_name=model_name)
        self.request_activity = self._bound_model_operations[0].registration
        self.request_stream_activity = self._bound_model_operations[1].registration
        self.cancel_suspended_response_activity = self._bound_model_operations[2].registration

        if self._event_stream_handler is not None:
            self._bound_event_operation = self._bind_event_operation(backend)
            self.event_stream_handler_activity = self._bound_event_operation.registration
            backend.move_registration_to_end(self.cancel_suspended_response_activity)

        # --- Toolset wrapping ---
        self._register_toolsets(agent)
        for wrapped in self._toolsets_by_id.values():
            backend.adopt_registrations(toolset_temporal_activities(wrapped))

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        if isinstance(ts, FunctionToolset):
            return self._build_function_toolset(ts)
        try:
            from pydantic_ai.mcp import MCPToolset
        except ImportError:  # pragma: no cover
            pass
        else:
            if isinstance(ts, MCPToolset):
                return self._build_temporal_mcp_toolset(ts)
        ts_id = ts.id
        toolset_activity_config = self.activity_config.copy()
        if ts_id is not None:
            toolset_activity_config.update(self._toolset_activity_config.get(ts_id, {}))
        toolset_activity_config['retry_policy'] = with_non_retryable_errors(toolset_activity_config.get('retry_policy'))
        assert self._deps_type is not None
        wrapped = _default_temporalize_toolset(
            ts,
            f'agent__{self.name}',
            toolset_activity_config,
            {},
            self._deps_type,
            self.run_context_type,
            self._agent,
        )
        return wrapped if isinstance(wrapped, (TemporalWrapperToolset, DurableToolsetBase)) else None

    def _build_temporal_mcp_toolset(self, toolset: Any) -> DurableToolsetBase[AgentDepsT]:
        from ._mcp_toolset import temporalize_mcp_toolset

        base_config = self._toolset_operation_config('mcp', cast(str, toolset.id))
        get_tools = self._bind_mcp_get_tools_operation(toolset)
        get_instructions = self._bind_mcp_get_instructions_operation(toolset)

        async def get_tools_operation(ctx: RunContext[AgentDepsT]):
            return await get_tools(_GetToolsParams(ctx), config=base_config)

        async def get_instructions_operation(ctx: RunContext[AgentDepsT]):
            return await get_instructions(_GetToolsParams(ctx), config=base_config)

        assert self._deps_type is not None
        return temporalize_mcp_toolset(
            toolset,
            activity_name_prefix=f'agent__{self.name}',
            activity_config=base_config,
            tool_activity_config={},
            deps_type=self._deps_type,
            run_context_type=self.run_context_type,
            agent=self._agent,
            get_tools_operation=get_tools_operation,
            get_tools_registration=get_tools.registration,
            get_instructions_operation=get_instructions_operation,
            get_instructions_registration=get_instructions.registration,
        )

    def _build_operation_backend(self) -> TemporalOperationBackend:
        backend = self._operation_backend
        assert backend is not None
        return backend

    def _toolset_base_config(self, kind: ToolsetKind) -> ActivityConfig:
        return self.activity_config

    def _toolset_operation_config(self, kind: ToolsetKind, toolset_id: str) -> ActivityConfig:
        config = self.activity_config.copy()
        config.update(self._toolset_activity_config.get(toolset_id, {}))
        config['retry_policy'] = with_non_retryable_errors(config.get('retry_policy'))
        return config

    @contextmanager
    def _tool_run_context_scope(self, ctx: RunContext[AgentDepsT]) -> Generator[RunContext[AgentDepsT], None, None]:
        # Temporal applies its durable-operation guards while deserializing the run context;
        # wrapping it again would replace activity-specific compatibility state.
        yield ctx

    def _resolve_temporal_tool_config(
        self, operation_id: DurableOperationId, tool: object | None, name: str
    ) -> ActivityConfig | Literal[False]:
        toolset_id = cast(Any, operation_id).toolset_id
        base_config = self._toolset_operation_config('function', toolset_id)
        config = resolve_tool_activity_config(cast(ToolsetTool[Any] | None, tool), name, {})
        if config is False:
            assert isinstance(tool, FunctionToolsetTool)
            if not tool.is_async:
                raise UserError(
                    f'Temporal activity config for tool {name!r} has been explicitly set to `False` (activity disabled), '
                    'but non-async tools are run in threads which are not supported outside of an activity. Make the tool function async instead.'
                )
            return False
        return cast(ActivityConfig, {**base_config, **config})

    def _function_call_parameter_transport(self, toolset: FunctionToolset[AgentDepsT]) -> _FunctionCallTransport:
        return _FunctionCallTransport(self, toolset)

    def _get_tools_parameter_transport(self, toolset: AbstractToolset[AgentDepsT]) -> _GetToolsTransport:
        return _GetToolsTransport(self)

    def _get_instructions_parameter_transport(self, toolset: AbstractToolset[AgentDepsT]) -> _GetToolsTransport:
        return _GetToolsTransport(self)

    def _bound_operation_registrations(self, *operations: object) -> list[Callable[..., Any]]:
        return [cast(Any, operation).registration for operation in operations]

    async def _prepare_function_call_params(
        self, toolset: FunctionToolset[AgentDepsT], params: _CallToolParams
    ) -> _CallToolParams:
        tool = params.tool
        if tool is None:
            try:
                tool = (await toolset.get_tools(params.ctx))[params.name]
            except KeyError as exc:
                raise UserError(
                    f'Tool {params.name!r} not found in toolset {toolset.id!r}. '
                    'Removing or renaming tools during an agent run is not supported with Temporal.'
                ) from exc
        args = tool.args_validator.validate_python(params.tool_args)
        return _CallToolParams(params.name, args, params.ctx, tool)

    def _tool_call_payload_errors(self, tool_name: str):
        return tool_result_payload_errors(tool_name)

    @property
    def temporal_activities(self) -> list[Callable[..., Any]]:
        """All Temporal activities registered by this capability.

        Register these with the Temporal worker, either directly or via
        `AgentPlugin`.
        """
        backend = self._operation_backend
        if backend is None:
            return []
        return list(backend.registrations())

    # --- Capability hooks ---

    @property
    def in_durable_context(self) -> bool:
        return workflow.in_workflow()

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Disable threads inside Temporal workflows."""
        if not self.in_durable_context:
            return await handler()

        with disable_threads(), set_agent_graph_sleep(workflow.sleep):
            return await handler()

    async def on_run_error(self, ctx: RunContext[AgentDepsT], *, error: BaseException) -> AgentRunResult[Any]:
        """Explain a serialization failure raised while scheduling an activity.

        This is the run's error-transformation hook: an exception raised from `wrap_run`
        would only be attached as the original error's `__context__`, never propagated.
        """
        if self.in_durable_context and isinstance(error, PydanticSerializationError):
            raise serialization_user_error(error) from error
        raise error

    def _validate_runtime_capabilities(
        self, ctx: RunContext[AgentDepsT], capabilities: Sequence[AbstractCapability[AgentDepsT]]
    ) -> None:
        """Reject per-run capabilities whose activities were not registered with the worker."""
        if self.in_durable_context:
            unsafe_capabilities = [capability for capability in capabilities if not capability._safe_at_runtime]
        else:
            unsafe_capabilities = []
        if unsafe_capabilities:
            names = ', '.join(sorted(type(capability).__name__ for capability in unsafe_capabilities))
            raise UserError(
                f'Capabilities added per-run inside a Temporal workflow are not supported: {names}. '
                'Temporal activities must be registered with the worker before the workflow runs. '
                'Attach all capabilities at agent construction time so `TemporalDurability.for_agent()` '
                'can register the activities for the toolsets they contribute.'
            )

    def _model_request_parameter_transport(self, result_type: object) -> _ModelRequestTransport:
        if result_type is StreamedActivityResult:
            result_type = _StreamedActivityPayload
        return _ModelRequestTransport(self, result_type=result_type)

    def _cancel_suspended_response_parameter_transport(self) -> _CancelTransport:
        return _CancelTransport(self)

    def _event_stream_handler_parameter_transport(self) -> _EventStreamHandlerTransport:
        return _EventStreamHandlerTransport(self)

    async def _load_streamed_activity_result(
        self, result: object, model_request_parameters: ModelRequestParameters
    ) -> StreamedActivityResult:
        if isinstance(result, ModelResponse):
            stream = CompletedStreamedResponse(
                result,
                model_request_parameters=model_request_parameters,
                replay_events=True,
            )
            return StreamedActivityResult(response=result, events=[event async for event in stream])
        return cast(StreamedActivityResult, result)

    async def _cancel_suspended_response_without_run_context(
        self, model_id: str | None, response: ModelResponse
    ) -> None:
        model = self._models_by_id.get(model_id or 'default')
        if model is None:
            assert model_id is not None
            model = infer_model(model_id)
        await model.cancel_suspended_response(response)

    def _validate_model_request_parameters(self, model_request_parameters: ModelRequestParameters) -> None:
        if model_request_parameters.allow_image_output:
            raise UserError(IMAGE_OUTPUT_UNSUPPORTED_MESSAGE)
