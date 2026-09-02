from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Literal, cast

from pydantic_core import PydanticSerializationError
from temporalio import workflow
from temporalio.workflow import ActivityConfig

from pydantic_ai._agent_graph import set_agent_graph_sleep
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.agent.abstract import AbstractAgent
from pydantic_ai.capabilities.abstract import WrapRunHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._capability_operation import CapabilityMethodDeclaration
from pydantic_ai.durable_exec._codec import IDENTITY_CODEC
from pydantic_ai.durable_exec._operation import (
    DurableOperationId,
    ToolsetCallToolId,
    ToolsetCallToolParams,
    ToolsetKind,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._spec import DurabilityEngineSpec
from pydantic_ai.durable_exec._toolset import DurableToolsetBase, validation_context_from_agent
from pydantic_ai.durable_exec._utils import StreamedActivityResult, disable_threads, managed_model_scope
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, infer_model
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset, ToolsetTool, WrapperToolset
from pydantic_ai.toolsets._dynamic import DynamicToolset
from pydantic_ai.toolsets.function import FunctionToolsetTool

from ._operation_backend import TemporalBoundOperation, TemporalOperationBackend
from ._run_context import TemporalRunContext, deserialize_run_context
from ._toolset import (
    TemporalWrapperToolset,
    resolve_tool_activity_config,
    temporalize_toolset as _default_temporalize_toolset,
    tool_result_payload_errors,
    validate_activity_config,
    with_non_retryable_errors,
)
from ._transports import (
    _CancelParams as _CancelParams,
    _CancelTransport,
    _CapabilityOperationTransport,
    _CompactMessagesTransport,
    _DynamicCallTransport,
    _DynamicGetToolsTransport,
    _EventStreamHandlerParams as _EventStreamHandlerParams,
    _EventStreamHandlerTransport,
    _FunctionCallTransport,
    _GetToolsTransport,
    _MCPCallTransport,
    _ModelRequestTransport,
    _RequestParams as _RequestParams,
    _StreamedActivityPayload,
)

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


@dataclass(init=False, kw_only=True)
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

    engine_spec = DurabilityEngineSpec(
        engine_name='Temporal',
        durable_unit_noun='activity',
        durable_unit_plural='activities',
        durable_container_noun='workflow',
        codec=IDENTITY_CODEC,
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        wrapped_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        toolset_lifecycles={
            'function': 'enter-outside-durable',
            'mcp': 'enter-outside-durable',
            'dynamic': 'enter-never',
        },
        tool_call_result_upgrade_lenient=False,
        journal_discovery=True,
        sequential_tools_in_durable_context=False,
        tool_config_key='temporal',
    )

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

    def _capability_operation_parameter_transport(
        self, declaration: CapabilityMethodDeclaration
    ) -> _CapabilityOperationTransport:
        return _CapabilityOperationTransport(self, declaration)

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
        model_name = self.default_model_id or (default_model.model_id if default_model is not None else 'default')
        self._bound_model_operations = self._bind_model_operations(backend, model_id=None, model_name=model_name)
        request = self._bound_model_operations.request
        request_stream = self._bound_model_operations.request_stream
        compact_messages = self._bound_model_operations.compact_messages
        cancel_suspended_response = self._bound_model_operations.cancel_suspended_response
        assert isinstance(request, TemporalBoundOperation)
        assert isinstance(request_stream, TemporalBoundOperation)
        assert isinstance(compact_messages, TemporalBoundOperation)
        assert isinstance(cancel_suspended_response, TemporalBoundOperation)
        self.request_activity = request.registration
        self.request_stream_activity = request_stream.registration
        self.compact_messages_activity = compact_messages.registration
        self.cancel_suspended_response_activity = cancel_suspended_response.registration

        if self._event_stream_handler is not None:
            self._bound_event_operation = self._bind_event_operation(backend)
            assert isinstance(self._bound_event_operation, TemporalBoundOperation)
            self.event_stream_handler_activity = self._bound_event_operation.registration
            backend.move_registration_to_end(self.cancel_suspended_response_activity)

        # --- Toolset wrapping ---
        self._register_toolsets(agent)

    def _wrap_other_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
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

    def get_durable_operation_backend(self) -> TemporalOperationBackend:
        backend = self._operation_backend
        assert backend is not None
        return backend

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
            from pydantic_ai.mcp import MCPToolset

            if (
                isinstance(operation_id, (ToolsetCallToolId, ToolsetValidateToolArgumentsId))
                and operation_id.toolset_kind == 'dynamic'
            ):
                raise UserError(
                    f'Temporal activity config for dynamic toolset tool {name!r} has been explicitly set to `False` '
                    '(activity disabled), but dynamic-toolset tools cannot run inside the workflow: resolving the '
                    'toolset and calling the tool may perform I/O. Remove the opt-out, or move the tool to a static '
                    '`FunctionToolset` (async tools there may opt out of activities).'
                )
            if isinstance(cast(ToolsetTool[Any], tool).toolset, MCPToolset):
                raise UserError(
                    f'Temporal activity config for MCP tool {name!r} has been explicitly set to `False` (activity disabled), '
                    'but MCP tools require the use of IO and so cannot be run outside of an activity.'
                )
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

    def _dynamic_get_tools_parameter_transport(self, toolset: DynamicToolset[AgentDepsT]) -> _DynamicGetToolsTransport:
        return _DynamicGetToolsTransport(self)

    def _dynamic_call_parameter_transport(self, toolset: DynamicToolset[AgentDepsT]) -> _DynamicCallTransport:
        return _DynamicCallTransport(self)

    def _validation_context(self, ctx: RunContext[Any]) -> Any:
        return validation_context_from_agent(self._agent)(ctx)

    def _mcp_call_parameter_transport(self, toolset: AbstractToolset[AgentDepsT]) -> _MCPCallTransport:
        return _MCPCallTransport(self, toolset)

    def _bound_operation_registrations(self, *operations: object) -> list[Callable[..., Any]]:
        return [cast(Any, operation).registration for operation in operations]

    async def _prepare_function_call_params(
        self, toolset: FunctionToolset[AgentDepsT], params: ToolsetCallToolParams
    ) -> ToolsetCallToolParams:
        tool = params.tool
        if tool is None:
            try:
                tool = (await toolset.get_tools(params.ctx))[params.name]
            except KeyError as exc:
                raise UserError(
                    f'Tool {params.name!r} not found in toolset {toolset.id!r}. '
                    'Removing or renaming tools during an agent run is not supported with Temporal.'
                ) from exc
        args = tool.args_validator.validate_python(params.tool_args, context=self._validation_context(params.ctx))
        return ToolsetCallToolParams(params.name, tool_args=args, ctx=params.ctx, tool=tool)

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

    def _model_request_parameter_transport(self, result_type: object) -> _ModelRequestTransport:
        if result_type is StreamedActivityResult:
            result_type = _StreamedActivityPayload
        return _ModelRequestTransport(self, result_type=result_type)

    def _cancel_suspended_response_parameter_transport(self) -> _CancelTransport:
        return _CancelTransport(self)

    def _compact_messages_parameter_transport(self) -> _CompactMessagesTransport:
        return _CompactMessagesTransport(self)

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
        owned = model is None
        if owned:
            assert model_id is not None
            model = infer_model(model_id)
        async with managed_model_scope(model, owned=owned) as active_model:
            await active_model.cancel_suspended_response(response)

    def _validate_model_request_parameters(self, model_request_parameters: ModelRequestParameters) -> None:
        if model_request_parameters.allow_image_output:
            raise UserError(IMAGE_OUTPUT_UNSUPPORTED_MESSAGE)
