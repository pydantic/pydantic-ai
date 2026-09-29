"""Durable execution for Pydantic AI agents on the Restate engine.

Restate (`restate-sdk`) is a durable-execution engine. This module journals an agent's I/O --
model requests, MCP calls, function tool calls, and dynamic-toolset resolution -- into Restate
run steps (`ctx.run_typed(...)`), so a handler that crashes or is retried mid-run replays from
the journal instead of repeating the work.

Restate's durable primitive is async and its journal is positional: `ctx.run_typed(name, fn, ...)`
records `fn`'s result by encounter order, and on replay serves the recorded bytes without
re-running `fn`. The `name` is a label, not the journal identity. An agent run is async, so it
drives the primitive directly with no thread bridge. Outside a Restate context the capability is
transparent and the run is an ordinary, non-durable agent run.
"""

from __future__ import annotations

try:
    import restate
except ImportError as _import_error:
    raise ImportError(
        'Please install the `restate-sdk` package to use the Restate durability capability, '
        'you can use the `restate` optional group -- `pip install "pydantic-ai-harness[restate]"`'
    ) from _import_error

from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import ClassVar, Literal

from restate.context import RunOptions
from restate.exceptions import TerminalError
from restate.extensions import current_context
from restate.serde import JsonSerde

from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    BaseDurabilityCapability,
    DurabilityEngineSpec,
    DurableOperationId,
    JournalCallableOperationBackend,
    RoleBasedOperationConfig,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.models import Model
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import ToolsetTool

_ENGINE_NAME = 'Restate'
_TOOL_CONFIG_KEY = 'restate'
_MAX_ATTEMPTS = 2**32 - 1
_IN_RESTATE_STEP: ContextVar[bool] = ContextVar('pydantic_ai_harness.restate.in_step', default=False)


def _current_restate_context() -> restate.Context | None:
    """Return the active Restate context, or `None` outside a Restate invocation."""
    try:
        return current_context()
    except LookupError:
        return None


def _resolve_tool_config(
    operation_id: DurableOperationId, tool: object | None, tool_name: str
) -> Mapping[str, object] | Literal[False]:
    del operation_id, tool_name
    config = (tool.tool_def.metadata or {}).get(_TOOL_CONFIG_KEY) if isinstance(tool, ToolsetTool) else None
    if config is False:
        return False
    if config:
        raise UserError('Restate run steps take no per-tool options; remove the config.')
    return {}


class _RestateOperationBackend(JournalCallableOperationBackend[Mapping[str, object]]):
    def __init__(self, agent_name: str, default_model_id: str | None, max_attempts: int | None) -> None:
        super().__init__(
            agent_name=agent_name,
            default_model_id=default_model_id,
            config=RoleBasedOperationConfig(
                model={}, event={}, capability={}, tool={}, resolve_tool=_resolve_tool_config
            ),
        )
        self._run_options: RunOptions[object] = RunOptions(serde=JsonSerde(), max_attempts=max_attempts)

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: Mapping[str, object],
    ) -> object:
        del operation_id, cache_key
        assert not config
        context = current_context()
        assert context is not None

        async def body_in_step() -> object:
            token = _IN_RESTATE_STEP.set(True)
            try:
                return await body()
            finally:
                _IN_RESTATE_STEP.reset(token)

        return await context.run_typed(name, body_in_step, self._run_options)


@dataclass(init=False)
class RestateDurability(BaseDurabilityCapability[AgentDepsT]):
    """Capability that makes an agent durable by journaling its I/O into Restate run steps.

    Attach it via `capabilities=[RestateDurability()]` and call `agent.run()` inside a Restate
    service handler: every model request, MCP call, function tool call, and dynamic-toolset
    resolution is wrapped in `ctx.run_typed(...)`, so a handler that crashes or is retried mid-run
    replays from the journal instead of repeating the work. A completed step is served from its
    journal entry on replay instead of being recomputed, so tokens are not re-spent on work that
    already finished. A step is journaled after it runs, so a crash between a tool's side effect
    and its journal entry re-runs the tool on recovery: keep tool side effects idempotent. Outside
    a Restate context the capability is transparent and the run is a normal, non-durable agent run.

    The capability discovers the agent's model, name, and toolsets automatically when it is bound
    to the agent. Step results are written to the Restate journal as JSON bytes, so a journaled
    tool's return value must be JSON-serializable. Serialization failures are terminal so Restate
    does not retry them as transient invocation failures. Control-flow signals (`ModelRetry`,
    `ApprovalRequired`, `CallDeferred`, `ToolFailed`) cross the journal as values rather than
    exceptions, with their metadata preserved, so approval and deferred-tool flows work inside a
    durable run.

    Restate steps use a finite three-attempt retry policy by default. Pass `max_attempts=None` to
    opt in to Restate's invocation-level retry policy instead.

    Example:
        ```python {test="skip"}
        import restate
        from pydantic_ai import Agent
        from pydantic_ai_harness import RestateDurability

        agent = Agent('openai:gpt-5', name='analyst', capabilities=[RestateDurability()])
        analyst = restate.Service('analyst')


        @analyst.handler()
        async def analyse(ctx: restate.Context, prompt: str) -> str:
            result = await agent.run(prompt)
            return result.output


        app = restate.app([analyst])
        ```
    """

    engine_spec: ClassVar = DurabilityEngineSpec(
        engine_name=_ENGINE_NAME,
        durable_unit_noun='step',
        durable_container_noun='handler',
        codec=JSON_CODEC,
        serialization_failure=lambda exc: TerminalError(str(exc)),
        toolset_lifecycles={
            'function': 'enter-never',
            'mcp': 'enter-in-durable-unit',
            'dynamic': 'enter-never',
        },
        sequential_tools_in_durable_context=True,
        unsupported_runtime_toolset_kinds=frozenset({'function', 'mcp', 'dynamic'}),
        tool_config_key=_TOOL_CONFIG_KEY,
    )

    def __init__(
        self,
        *,
        models: Mapping[str, Model] | None = None,
        event_stream_handler: EventStreamHandler[AgentDepsT] | None = None,
        name: str | None = None,
        max_attempts: int | None = 3,
    ) -> None:
        """Create a `RestateDurability` capability.

        The agent's model, name, and toolsets are discovered automatically.

        Args:
            models: Optional additional models keyed by ID for runtime model switching via
                `agent.run(model='<id>')`. The agent's primary model is always registered as
                `'default'`; the ID is folded into the step name so a replay maps each journal
                entry back to the model it was recorded for.
            event_stream_handler: Optional event stream handler. Model events are handled live
                inside the model-request step; each agent-level event is handled in its own
                journaled step.
            name: Unique agent name used as the prefix for every step name. Defaults to the
                agent's `name` when the capability is bound.
            max_attempts: Maximum attempts for each Restate run step, including the initial
                attempt. Defaults to 3 so a persistently failing operation terminates instead of
                retrying indefinitely. Must be between 1 and 4,294,967,295; set to `None` to use
                Restate's invocation retry policy.
        """
        if max_attempts is not None and (
            isinstance(max_attempts, bool)
            or not isinstance(max_attempts, int)
            or not 1 <= max_attempts <= _MAX_ATTEMPTS
        ):
            raise UserError(f'`max_attempts` must be an integer from 1 to {_MAX_ATTEMPTS}, or `None`.')
        super().__init__(models=models, event_stream_handler=event_stream_handler, name=name)
        self._max_attempts = max_attempts

    @property
    def in_durable_context(self) -> bool:
        return _current_restate_context() is not None and not _IN_RESTATE_STEP.get()

    def get_durable_operation_backend(self) -> _RestateOperationBackend:
        return _RestateOperationBackend(self.name, self.default_model_id, self._max_attempts)
