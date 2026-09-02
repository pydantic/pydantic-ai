from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from prefect.context import FlowRunContext

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._codec import IDENTITY_CODEC
from pydantic_ai.durable_exec._operation import (
    DurableOperationId,
    ToolsetCallToolId,
    ToolsetKind,
    ToolsetValidateToolArgumentsId,
)
from pydantic_ai.durable_exec._operation_backend import DurableOperationBackend
from pydantic_ai.durable_exec._spec import DurabilityEngineSpec
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import ToolsetTool

from ._model import _stamp_response_provenance  # pyright: ignore[reportPrivateUsage]
from ._operation_backend import PrefectOperationBackend, PrefectOperationConfig
from ._toolset import with_non_retryable_errors
from ._types import TaskConfig, default_task_config


@dataclass(init=False, kw_only=True)
class PrefectDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through Prefect tasks.

    Built on the declarative base: the base owns toolset/model/event assembly, and this
    capability contributes the Prefect operation backend, transparency gate, and task configuration.
    """

    engine_spec = DurabilityEngineSpec(
        engine_name='Prefect',
        durable_unit_noun='task',
        durable_unit_plural='tasks',
        durable_container_noun='flow',
        codec=IDENTITY_CODEC,  # object-passing: Prefect serializes/caches internally
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        wrapped_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        toolset_lifecycles={'function': 'enter-always', 'mcp': 'enter-always', 'dynamic': 'enter-never'},
        tool_call_result_upgrade_lenient=True,  # cached payloads may predate value-wrapping
        journal_discovery=True,
        sequential_tools_in_durable_context=False,
        tool_config_key='prefect',
    )

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        event_stream_handler_task_config: TaskConfig | None = None,
        model_task_config: TaskConfig | None = None,
        mcp_task_config: TaskConfig | None = None,
        tool_task_config: TaskConfig | None = None,
    ):
        """Create a PrefectDurability capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model
                switching. The agent's primary model is always registered as
                `'default'`. A `Model` instance can't be serialized across the
                task boundary, so a run-time model (via `agent.run(model=...)`
                / `agent.override(model=...)`, or swapped in by an outer capability)
                has to be registered here and referenced by key (or passed as the
                registered instance); an unregistered instance is rejected, because
                rebuilding it from its `model_id` would build a different model.
                Model-name strings never need registering: they cross as the string
                the caller wrote and are built inside the task by the agent's
                `resolve_model_id` capability chain, then `infer_model`. To build a
                specific instance inside the task from such a string — a custom
                provider, or per-user credentials carried on `deps` — use the
                [`ResolveModelId`][pydantic_ai.capabilities.ResolveModelId] capability.
            event_stream_handler: Optional event stream handler. Model events are handled
                live inside model-request tasks, and tool events are handled in per-event tasks.
            name: Unique agent name used in the Prefect task names. Defaults to the agent's
                `name` when the capability is bound.
            event_stream_handler_task_config: Prefect task config for event stream handler tasks.
            model_task_config: Prefect task config for model request tasks.
            mcp_task_config: Prefect task config for MCP server tasks.
            tool_task_config: Default Prefect task config for tool call tasks. Per-tool
                overrides are configured via tool metadata, e.g.
                `@my_toolset.tool(metadata={'prefect': TaskConfig(...)})` (or `False` to skip
                task wrapping), or via the
                [`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] capability.
        """
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        # Model and event-handler tasks compose the same non-retryable condition as tool tasks.
        self._model_task_config = with_non_retryable_errors(default_task_config | (model_task_config or {}))
        self._mcp_task_config = default_task_config | (mcp_task_config or {})
        self._tool_task_config = default_task_config | (tool_task_config or {})
        self._event_stream_handler_task_config = with_non_retryable_errors(
            default_task_config | (event_stream_handler_task_config or {})
        )

    # --- Behavioral hooks ---

    @property
    def in_durable_context(self) -> bool:
        return FlowRunContext.get() is not None

    def get_durable_operation_backend(self) -> DurableOperationBackend[TaskConfig]:
        def tool_config(
            operation_id: DurableOperationId, tool: object | None, tool_name: str
        ) -> TaskConfig | Literal[False]:
            assert isinstance(operation_id, ToolsetCallToolId | ToolsetValidateToolArgumentsId)
            kind = operation_id.toolset_kind
            config = self._build_resolve_tool_config(self._toolset_base_config(kind))(
                cast(ToolsetTool[Any] | None, tool), tool_name
            )
            return cast(TaskConfig | Literal[False], config)

        return PrefectOperationBackend(
            config=PrefectOperationConfig(
                model=self._model_task_config,
                event=self._event_stream_handler_task_config,
                capability=default_task_config,
                tool=self._tool_task_config,
                resolve_tool=tool_config,
            ),
            event_sequence_key=f'pydantic_ai_event_sequence:{self.name}',
        )

    # --- Config knobs ---

    def _toolset_base_config(self, kind: ToolsetKind) -> Any:
        return self._mcp_task_config if kind == 'mcp' else self._tool_task_config

    def _normalize_unit_config(self, config: Any) -> Any:
        return with_non_retryable_errors(config)

    def _stamp_response(self, response: ModelResponse, messages: list[ModelMessage]) -> None:
        _stamp_response_provenance(response, messages)
