from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, cast

from prefect import task
from prefect.context import FlowRunContext

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.durable_exec._base import BaseDurabilityCapability, ToolsetKind
from pydantic_ai.durable_exec._codec import IDENTITY_CODEC
from pydantic_ai.durable_exec._runtime_toolsets import RuntimeToolsetKind
from pydantic_ai.durable_exec._toolset import Lifecycle
from pydantic_ai.messages import AgentStreamEvent, ModelMessage, ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT, RunContext

from ._model import _stamp_response_provenance  # pyright: ignore[reportPrivateUsage]
from ._toolset import with_non_retryable_errors
from ._types import TaskConfig, default_task_config


@dataclass(init=False)
class PrefectDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by routing I/O through Prefect tasks.

    Ported onto the declarative Shape-D base: the base owns toolset/model/event assembly, and this
    capability contributes only the Prefect primitive (`run_durable_unit`), the transparency gate
    (`in_durable_context`), the naming scheme, the config knobs, and -- the one genuine behavioral
    override -- the hash-keyed event dispatch (Prefect keys replay on input hash, so identical events
    need a per-flow-run sequence number the sequence-keyed engines don't).
    """

    # --- Declarative surface ---
    engine_name = 'Prefect'
    _codec: ClassVar = IDENTITY_CODEC  # object-passing: Prefect serializes/caches internally
    _unsupported_runtime_toolset_kinds: ClassVar[frozenset[RuntimeToolsetKind]] = frozenset(
        {'function', 'mcp', 'dynamic'}
    )
    _wrapped_toolset_kinds: ClassVar[frozenset[ToolsetKind]] = frozenset({'function', 'mcp', 'dynamic'})
    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-always',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }
    _tool_call_result_upgrade_lenient: ClassVar[bool] = True  # cached payloads may predate value-wrapping
    _journal_discovery: ClassVar[bool] = False  # resolve MCP/dynamic toolsets in flow code, journal only calls
    _durable_unit_noun = 'task'
    _durable_container_noun = 'flow'
    _tool_config_key = 'prefect'

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

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        """Run `fn` as a Prefect task.

        `inputs` are passed as task arguments so the cache policy (`PrefectAgentInputs` -- hash-keyed) forks the cache key on them; `fn` (a closure) does the
        real work. `name` is prepended to the hashed inputs so operations with identical inputs but
        different identity keep distinct cache entries.
        """

        @task
        async def _unit(
            unit_name: str,
            a0: Any = None,
            a1: Any = None,
            a2: Any = None,
            a3: Any = None,
            a4: Any = None,
        ) -> Any:
            return await fn()

        options = cast(TaskConfig, config or {})
        return await _unit.with_options(name=name, **options)(name, *inputs)

    # --- Naming (compat surface): Prefect's human-readable task display names ---

    def _unit_name(self, kind: str, **parts: Any) -> str:
        label = parts.get('label')
        if (model_name := parts.get('model_name')) is not None:
            return f'{label}: {model_name}'
        if (tool_name := parts.get('tool_name')) is not None:
            return f'{label}: {tool_name}'
        assert isinstance(label, str)
        return label

    def _model_id_suffix(self, model_id: str | None) -> str:
        """Keep Prefect's existing display names unchanged for runtime model selection."""
        return ''

    # --- Config knobs ---

    def _model_unit_config(self) -> Any:
        return self._model_task_config

    def _event_unit_config(self) -> Any:
        return self._event_stream_handler_task_config

    def _toolset_base_config(self, kind: ToolsetKind) -> Any:
        return self._mcp_task_config if kind == 'mcp' else self._tool_task_config

    def _normalize_unit_config(self, config: Any) -> Any:
        return with_non_retryable_errors(config)

    def _stamp_response(self, response: ModelResponse, messages: list[ModelMessage]) -> None:
        _stamp_response_provenance(response, messages)

    # --- The one genuine behavioral override: hash-keyed event dispatch ---

    async def _dispatch_event_stream_event(self, ctx: RunContext[AgentDepsT], event: AgentStreamEvent) -> None:
        handler = self._event_stream_handler
        assert handler is not None

        # The sequence number makes content-identical events within one flow run each fire (distinct
        # task-cache keys) while a flow retry reproduces the same numbers and replays from cache.
        # `task_run_dynamic_keys` is Prefect's per-flow-run counter store.
        flow_context = FlowRunContext.get()
        assert flow_context is not None
        sequence_key = f'pydantic_ai_event_sequence:{self.name}'
        sequence = flow_context.task_run_dynamic_keys.get(sequence_key, 0)
        assert isinstance(sequence, int)
        flow_context.task_run_dynamic_keys[sequence_key] = sequence + 1

        async def fn() -> None:
            with self._durable_run_context_scope(ctx) as durable_ctx:
                await handler(durable_ctx, self._single_event_stream(event))
            return None

        await self._durable_operation(
            'Handle Stream Event',
            fn,
            tp=type(None),
            inputs=(event, sequence),
            config=self._event_unit_config(),
        )
