"""The `Planning` capability: task planning with a cache-safe live reminder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.messages import CachePoint, ModelRequest, ModelResponse, UserPromptPart
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset

from pydantic_ai_harness.planning._store import InMemoryPlanStore, PlanStore
from pydantic_ai_harness.planning._toolset import (
    SUBTASK_TOOL_NAMES,
    PlanningToolset,
    available_tool_names,
    render_plan,
)
from pydantic_ai_harness.planning._types import PlanItem

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
    from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
    from pydantic_ai.models import ModelRequestContext

_WRITE_PLAN_GUIDANCE = (
    'You have a planning tool, `write_plan`. For multi-step work, call it first to lay out the '
    'steps, then keep it current: mark exactly one step `in_progress`, and mark a step `completed` '
    'as soon as it is fully done. Pass the full plan every time you call `write_plan`.'
)

_GRANULAR_GUIDANCE = (
    'Use `add_task` to append a single step, `update_task_status`/`update_task_statuses` to move '
    'steps between statuses, and `read_plan` to see step ids before a granular edit.'
)

_SUBTASK_GUIDANCE = (
    'Break a complex step into subtasks with `add_subtask`, and record ordering with '
    '`set_dependency`: a step stays `blocked` until every step it depends on is resolved '
    '(`completed` or `cancelled`). Call `get_available_tasks` to pick the next step that has no '
    'incomplete dependencies.'
)


@dataclass
class Planning(AbstractCapability[AgentDepsT]):
    """Structured task planning that never invalidates the prompt cache.

    The model owns the plan through a small toolset (`write_plan`, `read_plan`,
    `add_task`, `update_task_status`, `update_task_statuses`, `remove_task`, and
    -- when `enable_subtasks` is set -- `add_subtask`, `set_dependency`,
    `get_available_tasks`); `tools` narrows that surface to an allowlist. The
    current plan is surfaced back as an *ephemeral* reminder appended to the
    tail of each request. Its cache-stable opening tag precedes a `CachePoint`,
    so the cached prefix stays byte-identical across turns; only the mutable
    plan content is re-read each turn.

    By default the plan lives in memory for the duration of a single run (a
    fresh, isolated plan per run). Pass a `store` (or `store_resolver`) to
    persist it -- e.g. `SqlitePlanStore` or `PostgresPlanStore`.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.planning import Planning

    agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[Planning()])
    ```
    """

    guidance: str | None = None
    """Static planning guidance for the system prompt. Cache-stable.

    Three states, so opting out is something you do on purpose rather than by
    accident:

    - `None` (the default): use the built-in guidance.
    - `''`: no guidance at all.
    - any other string: use it instead of the built-in guidance.

    A single `str | None` where `None` meant "no guidance" would leave no way to
    ask for the default explicitly, and would turn a config that resolves to
    `None` into a silent opt-out. This matches `memory`, `exa` and
    `runtime_authoring`, which read the same way.
    """

    cache_ttl: Literal['5m', '1h'] = '5m'
    """TTL for the cache breakpoint placed after the stable plan-reminder opening tag."""

    store: PlanStore | None = None
    """Storage backend. `None` keeps a fresh in-memory plan per run (the original
    ephemeral behaviour). Pass a store to persist the plan across runs."""

    store_resolver: Callable[[RunContext[AgentDepsT]], PlanStore] | None = None
    """Optional per-run store resolver, e.g. `lambda ctx: ctx.deps.plan_store`."""

    enable_subtasks: bool = False
    """Add the subtask/dependency tools and the `blocked` status when true."""

    inject: bool = True
    """Surface the current plan as a cache-safe tail reminder each turn."""

    tools: Sequence[str] | None = None
    """Optional allowlist of tool names to register; `None` registers all of them.

    The full surface is `write_plan`, `read_plan`, `add_task`, `update_task_status`,
    `update_task_statuses`, `remove_task`, plus `add_subtask`, `set_dependency` and
    `get_available_tasks` under `enable_subtasks`. `tools=['write_plan']` is the smallest
    useful plan surface. Naming a tool this mode does not register raises `ValueError`.

    The built-in `guidance` follows the allowlist for the whole-plan/granular/subtask split;
    trimming within a group is better paired with a `guidance` string of your own.
    """

    descriptions: dict[str, str] | None = None
    """Optional per-tool description overrides, keyed by tool name. Unknown names raise `ValueError`."""

    # Override the inherited default ID because durable-operation recovery needs a stable identity.
    _: KW_ONLY
    id: str | None = 'planning'

    _resolved_store: PlanStore | None = field(default=None, init=False, repr=False, compare=False)

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Planning[AgentDepsT]:
        """Return a clone with this run's store resolved and cached (per-run isolation)."""
        clone = copy(self)
        clone._resolved_store = clone._resolve_store(ctx)
        return clone

    def resolve_store(self, ctx: RunContext[AgentDepsT]) -> PlanStore:
        """Return the cached run store, or resolve one for direct toolset use."""
        if self._resolved_store is not None:
            return self._resolved_store
        self._resolved_store = self._resolve_store(ctx)
        return self._resolved_store

    def _resolve_store(self, ctx: RunContext[AgentDepsT]) -> PlanStore:
        if self.store_resolver is not None:
            return self.store_resolver(ctx)
        if self.store is not None:
            return self.store
        return InMemoryPlanStore()

    def get_toolset(self) -> AgentToolset[AgentDepsT] | None:
        """Provide the `planning` toolset over this run's resolved store."""
        return PlanningToolset[AgentDepsT](self)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Provide static, cache-stable guidance on using the planning tools.

        A custom `guidance` string is used verbatim. The default is assembled from
        the tools actually registered -- the granular sentence is dropped when
        `tools` excludes them all, and the subtask/dependency workflow is added
        under `enable_subtasks` -- so the model is not told about tools it lacks.
        """
        if self.guidance is not None:
            return self.guidance or None
        registered = available_tool_names(subtasks=self.enable_subtasks) if self.tools is None else set(self.tools)
        parts: list[str] = []
        if 'write_plan' in registered:
            parts.append(_WRITE_PLAN_GUIDANCE)
        if registered & {'read_plan', 'add_task', 'update_task_status', 'update_task_statuses'}:
            parts.append(_GRANULAR_GUIDANCE)
        if registered & set(SUBTASK_TOOL_NAMES):
            parts.append(_SUBTASK_GUIDANCE)
        return ' '.join(parts) or None

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Append the current plan as an ephemeral tail reminder with a cache breakpoint."""
        if not self.inject:
            return await handler(request_context)
        items = await self._read_plan(ctx)
        if not items:
            return await handler(request_context)
        messages = request_context.messages
        last = messages[-1]
        if isinstance(last, ModelRequest):
            reminder = UserPromptPart(
                content=['<plan-reminder>\n', CachePoint(ttl=self.cache_ttl), _reminder_text(render_plan(items))]
            )
            messages[-1] = replace(last, parts=[*last.parts, reminder])
        return await handler(request_context)

    @durable_operation('read_plan')
    async def _read_plan(self, ctx: RunContext[AgentDepsT]) -> list[PlanItem]:
        return await self.resolve_store(ctx).get_items()

    @classmethod
    def from_spec(
        cls,
        *,
        backend: Literal['memory', 'sqlite'] = 'memory',
        database: str = '.agent-plan.db',
        session: str = 'default',
        enable_subtasks: bool = False,
        inject: bool = True,
        guidance: str | None = None,
        cache_ttl: Literal['5m', '1h'] = '5m',
        tools: list[str] | None = None,
    ) -> Planning[AgentDepsT]:
        """Construct a `Planning` capability from serializable options."""
        if backend != 'sqlite' and database != '.agent-plan.db':
            raise ValueError('database is only valid with backend="sqlite"')
        if backend == 'memory':
            store: PlanStore | None = None
        elif backend == 'sqlite':
            from pydantic_ai_harness.planning._store import SqlitePlanStore

            store = SqlitePlanStore(database, session=session)
        else:
            # `backend` is a `Literal`, but this is the deserialization entry point: the value
            # arrives from a YAML or JSON spec, where nothing has checked it. Falling through
            # to SQLite would silently write `postgres` plans to `.agent-plan.db`.
            raise ValueError(f"Unknown planning backend {backend!r}; expected 'memory' or 'sqlite'.")
        return cls(
            store=store,
            enable_subtasks=enable_subtasks,
            inject=inject,
            guidance=guidance,
            cache_ttl=cache_ttl,
            tools=tools,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'Planning'


def _reminder_text(plan: str) -> str:
    return f'Your current plan (keep it updated with the planning tools):\n\n{plan}\n</plan-reminder>'
