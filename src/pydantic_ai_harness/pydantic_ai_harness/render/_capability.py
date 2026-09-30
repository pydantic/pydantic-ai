"""Render Workflows execution capability for Pydantic AI agents.

SDK assumptions verified 2026-09-29 against Render 1.2.0 and CLI 2.28.0:
public task definitions register before the worker starts, and `TaskContext.run`
executes child tasks in separate processes. The local runtime does not populate
`TaskContext.metadata`, so task lineage is not used as an authorization check.
Recheck with `tests/harness/render/test_local_runtime.py` and the SDK reference:
https://render.com/docs/workflows-sdk-python
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, ClassVar, Concatenate, Literal, ParamSpec, Protocol, TypeVar, overload

from pydantic_ai.agent import AbstractAgent, EventStreamHandler
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationBackend,
    DurableOperationId,
    RoleBasedOperationConfig,
    ToolsetCallToolId,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import InstructionPart
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, FunctionToolset, WrapperToolset

try:
    from render import Options, Retry, TaskContext, Workflows
    from render.workflows import TaskDefinition
except ModuleNotFoundError as _import_error:  # pragma: no cover
    if _import_error.name != 'render':
        raise
    raise ImportError(
        'Please install the `render` package to use the Render Workflows capability, '
        'for example with `pip install "pydantic-ai-harness[render]"`.'
    ) from _import_error

from ._compat import (
    CapabilityMethodDeclaration,
    RenderRunContextCodec,
    ToolsetCallToolParams,
    function_tool_original_name,
    prepare_function_call_params,
    reject_unidentified_operation_capabilities,
)
from ._context import activate_task_context, current_task_context
from ._operation_backend import RenderOperationBackend
from ._toolset_ids import prepare_capability_toolset_ids
from ._transports import (
    RenderCancelTransport,
    RenderCapabilityOperationTransport,
    RenderCompactMessagesTransport,
    RenderDynamicCallTransport,
    RenderDynamicGetToolsTransport,
    RenderEventStreamHandlerTransport,
    RenderFunctionCallTransport,
    RenderGetToolsTransport,
    RenderMCPCallTransport,
    RenderModelRequestTransport,
    RenderWorkspaceCallTransport,
)

P = ParamSpec('P')
R = TypeVar('R')

ToolOptionsResolver = Callable[
    [DurableOperationId, object | None, str],
    Options | Literal[False] | None,
]

Instructions = str | InstructionPart | Sequence[str | InstructionPart] | None


class TaskDecorator(Protocol):
    """The decorator `RenderWorkflows.task(...)` returns when it is given options.

    The Render SDK types this shape as `render.workflows.task.BoundTaskDecorator`, which
    neither `render` nor `render.workflows` exports. Restating it structurally keeps the
    runtime import surface to Render's public API while the two overloads stay compatible.
    """

    # The async overload must stay first: a coroutine function also matches the sync
    # signature, with `R` bound to the coroutine rather than to its result.
    @overload
    def __call__(self, func: Callable[Concatenate[TaskContext, P], Awaitable[R]], /) -> TaskDefinition[P, R]: ...

    @overload
    def __call__(self, func: Callable[Concatenate[TaskContext, P], R], /) -> TaskDefinition[P, R]: ...


@dataclass(init=False)
class RenderWorkflows(BaseDurabilityCapability[AgentDepsT]):
    """Route supported agent operations through an explicit Workflows app.

    Outside a Render task, the capability is transparent. Use this instance's
    `task` decorator for each workflow entry point that calls the agent.
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name='Render Workflows',
        durable_unit_noun='task',
        durable_container_noun='workflow',
        codec=JSON_CODEC,
        wrapped_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        toolset_lifecycles={
            'function': 'enter-outside-durable',
            'mcp': 'enter-outside-durable',
            'dynamic': 'enter-never',
        },
        journal_discovery=True,
        sequential_tools_in_durable_context=False,
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        tool_config_key='render_workflows',
    )

    def __init__(
        self,
        app: Workflows,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        deps_type: type[AgentDepsT] | None = None,
        model_options: Options | None = None,
        tool_options: Options | None = None,
        event_options: Options | None = None,
        capability_options: Options | None = None,
        resolve_tool_options: ToolOptionsResolver | None = None,
    ) -> None:
        """Create a capability and register its operation tasks on `app` when bound.

        Args:
            app: The exact Render `Workflows` app started by the worker.
            models: Additional models keyed by their registered model ID.
            event_stream_handler: Optional handler for agent stream events.
            name: Stable prefix for generated Render task names. Defaults to the agent name.
            deps_type: Dependency type used for task-boundary serialization. Defaults to the
                agent's dependency type when the capability is bound.
            model_options: Options for model operation tasks.
            tool_options: Options for tool operation tasks.
            event_options: Options for event handler tasks.
            capability_options: Options for tasks generated from other capabilities'
                `@durable_operation` methods.
            resolve_tool_options: Optional resolver for toolset registration options and
                function-tool opt-out. The framework asks for a toolset default with
                `tool=None`; each statically known function tool is then resolved with its
                concrete tool and name. `None` keeps `tool_options`, and `False` executes that
                function tool inline. Render fixes task options at registration, so returning
                different `Options` later raises a `UserError` rather than silently ignoring them.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self.app = app
        self._deps_type = deps_type
        base_tool_options = tool_options or Options()

        def resolve_options(
            operation_id: DurableOperationId, tool: object | None, tool_name: str
        ) -> Options | Literal[False]:
            if resolve_tool_options is None:
                return base_tool_options
            static_name = function_tool_original_name(tool) if tool is not None else None
            resolved = resolve_tool_options(operation_id, tool, static_name or tool_name)
            # Binding asks for the task definition's options before a concrete
            # tool is known. `False` remains a per-invocation function-tool
            # opt-out, so it cannot suppress registration of the shared task.
            if tool is None and resolved is False:
                return base_tool_options
            if resolved is False and (
                not isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId)
                or operation_id.toolset_kind != 'function'
            ):
                raise UserError(
                    '`resolve_tool_options` may return `False` only for function tools; '
                    'MCP and dynamic tools must run as Render child tasks.'
                )
            return base_tool_options if resolved is None else resolved

        self._operation_config = RoleBasedOperationConfig[Options | None](
            model=model_options or Options(),
            tool=base_tool_options,
            event=event_options or Options(),
            capability=capability_options or Options(),
            resolve_tool=resolve_options if resolve_tool_options is not None else None,
        )
        self._operation_backend: RenderOperationBackend[AgentDepsT] | None = None
        self._context_codec: RenderRunContextCodec[Any] | None = None
        self._inline_toolsets: frozenset[int] = frozenset()
        # Task context belongs to the Workflow app. Explicitly configured child agents using
        # another RenderWorkflows instance for the same app can therefore start nested tasks.
        self._owner_token = app

    @property
    def current_task_context(self) -> TaskContext | None:
        """Return the active Render task context for this capability."""
        return current_task_context(self._owner_token)

    @property
    def in_durable_context(self) -> bool:
        return self.current_task_context is not None

    def activate(self, context: TaskContext) -> AbstractContextManager[None]:
        """Activate a Render task context for an adapter or local test harness."""
        return activate_task_context(self._owner_token, context)

    def _check_bindable(self) -> None:
        if self.in_durable_context:
            raise UserError(
                'An agent with `RenderWorkflows` must be constructed outside a Render workflow so '
                'its operation tasks are registered before the worker starts.'
            )

    @classmethod
    def _check_single_capability(cls, agent: AbstractAgent[Any, Any]) -> None:
        """Refuse a second `RenderWorkflows` while the agent is still being built.

        `from_agent` is the engine's own lookup and already rejects a second instance, but
        only once something asks for the bound capability. Asking here means the agent is
        never constructed with two registered task sets and no defined answer for which
        `Workflows` app a run dispatches to.
        """
        try:
            cls.from_agent(agent)
        except UserError as exc:
            raise UserError(
                'Attach exactly one `RenderWorkflows` capability to an agent. Each one registers a '
                'full set of Render tasks against the `Workflows` app it was given, so a second '
                'leaves the agent with two sets and nothing to say which app a run dispatches to.'
            ) from exc

    # The async overload comes first so a coroutine function does not bind `R` to its coroutine.
    @overload
    def task(
        self,
        func: Callable[Concatenate[TaskContext, P], Awaitable[R]],
        /,
    ) -> TaskDefinition[P, R]: ...

    @overload
    def task(
        self,
        func: Callable[Concatenate[TaskContext, P], R],
        /,
    ) -> TaskDefinition[P, R]: ...

    @overload
    def task(
        self,
        *,
        name: str | None = ...,
        retry: Retry | None = ...,
        timeout_seconds: int | None = ...,
        plan: str | None = ...,
    ) -> TaskDecorator: ...

    def task(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        retry: Retry | None = None,
        timeout_seconds: int | None = None,
        plan: str | None = None,
    ) -> TaskDefinition[..., Any] | TaskDecorator:
        """Register a task that activates this capability's Render context."""

        def decorator(task_func: Callable[..., Any]) -> TaskDefinition[..., Any]:
            if inspect.iscoroutinefunction(task_func):

                @functools.wraps(task_func)
                async def async_wrapper(context: TaskContext, *args: Any, **kwargs: Any) -> Any:
                    with self.activate(context):
                        return await task_func(context, *args, **kwargs)

                wrapped = async_wrapper
            else:

                @functools.wraps(task_func)
                def sync_wrapper(context: TaskContext, *args: Any, **kwargs: Any) -> Any:
                    with self.activate(context):
                        return task_func(context, *args, **kwargs)

                wrapped = sync_wrapper
            return self.app.task(
                name=name,
                retry=retry,
                timeout_seconds=timeout_seconds,
                plan=plan,
            )(wrapped)

        if func is None:
            return decorator
        return decorator(func)

    def _bind_to_agent(self, agent: AbstractAgent[AgentDepsT, Any]) -> None:
        self._check_single_capability(agent)

        if self._deps_type is None:
            self._deps_type = agent.deps_type

        # Render task names are persisted workflow identity, so every leaf is named before
        # the codec, the operation backend, and the event operation below start registering
        # tasks: a name settled later would be a name nothing can go back and change.
        self._inline_toolsets = prepare_capability_toolset_ids(agent.toolsets)
        # Pydantic AI reaches the same check while registering capability operations, by
        # which point this app is already holding tasks that nothing can unregister.
        reject_unidentified_operation_capabilities(agent.root_capability)

        self._context_codec = RenderRunContextCodec(
            deps_type=self._deps_type,
            agent=agent,
            resolve_model=self._resolve_child_task_model,
        )
        self._operation_backend = RenderOperationBackend(
            self.app,
            runtime=self,
            agent_name=self.name,
            config=self._operation_config,
        )
        # Toolset identity was settled above, so the base bind cannot reject an unnamed or
        # duplicate ID after this event task is registered. Event binding must still come
        # first because the durable wrappers created by the base bind capture it.
        if self._event_stream_handler is not None:
            self._bound_event_operation = self._bind_event_operation(self._operation_backend)
        super()._bind_to_agent(agent)

    def _wrap_leaf_toolset(self, ts: AbstractToolset[AgentDepsT]) -> WrapperToolset[AgentDepsT] | None:
        if id(ts) in self._inline_toolsets:
            return None
        return super()._wrap_leaf_toolset(ts)

    def get_durable_operation_backend(self) -> DurableOperationBackend[Options | None]:
        backend = self._operation_backend
        if backend is None:
            raise UserError('`RenderWorkflows` must be bound to an agent before its operation backend is used.')
        return backend

    def _codec(self) -> RenderRunContextCodec[Any]:
        codec = self._context_codec
        if codec is None:
            raise UserError('`RenderWorkflows` must be bound to an agent before its operation transports are used.')
        return codec

    def _resolve_child_task_model(self, model_id: str | None) -> Model | None:
        """Resolve a child task's model from this worker's model registry."""
        registry_key = model_id or 'default'
        return self._models_by_id.get(registry_key)

    async def _prepare_function_call_params(
        self,
        toolset: FunctionToolset[AgentDepsT],
        params: ToolsetCallToolParams,
    ) -> ToolsetCallToolParams:
        """Revalidate JSON-decoded arguments against the worker-local tool."""
        agent = self._agent
        if agent is None:  # pragma: no cover - binding always supplies the agent
            raise UserError('`RenderWorkflows` must be bound before a function tool can run.')
        return await prepare_function_call_params(agent, toolset, params)

    def _workspace_call_transport(self) -> RenderWorkspaceCallTransport[AgentDepsT]:
        return RenderWorkspaceCallTransport(self._codec())

    def _capability_operation_parameter_transport(
        self, declaration: CapabilityMethodDeclaration
    ) -> RenderCapabilityOperationTransport[AgentDepsT]:
        return RenderCapabilityOperationTransport(self._codec(), declaration)

    def _function_call_parameter_transport(
        self, toolset: FunctionToolset[AgentDepsT]
    ) -> RenderFunctionCallTransport[AgentDepsT]:
        return RenderFunctionCallTransport(self._codec(), toolset)

    def _get_tools_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> RenderGetToolsTransport[AgentDepsT]:
        del toolset
        return RenderGetToolsTransport(self._codec(), result_type=dict[str, ToolDefinition])

    def _get_instructions_parameter_transport(
        self, toolset: AbstractToolset[AgentDepsT]
    ) -> RenderGetToolsTransport[AgentDepsT]:
        del toolset
        return RenderGetToolsTransport(self._codec(), result_type=Instructions)

    def _dynamic_get_tools_parameter_transport(
        self, toolset: DynamicToolset[AgentDepsT]
    ) -> RenderDynamicGetToolsTransport[AgentDepsT]:
        del toolset
        return RenderDynamicGetToolsTransport(self._codec())

    def _dynamic_call_parameter_transport(
        self, toolset: DynamicToolset[AgentDepsT]
    ) -> RenderDynamicCallTransport[AgentDepsT]:
        del toolset
        return RenderDynamicCallTransport(self._codec())

    def _mcp_call_parameter_transport(self, toolset: AbstractToolset[AgentDepsT]) -> RenderMCPCallTransport[AgentDepsT]:
        return RenderMCPCallTransport(self._codec(), toolset)

    def _model_request_parameter_transport(self, result_type: object) -> RenderModelRequestTransport[AgentDepsT]:
        return RenderModelRequestTransport(self._codec(), result_type=result_type)

    def _cancel_suspended_response_parameter_transport(
        self,
    ) -> RenderCancelTransport[AgentDepsT]:
        return RenderCancelTransport(self._codec())

    def _compact_messages_parameter_transport(self) -> RenderCompactMessagesTransport[AgentDepsT]:
        return RenderCompactMessagesTransport(self._codec())

    def _event_stream_handler_parameter_transport(
        self,
    ) -> RenderEventStreamHandlerTransport[AgentDepsT]:
        return RenderEventStreamHandlerTransport(self._codec())
