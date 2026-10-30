"""The `Planning` capability: task planning, with the current plan shown in the plan tools' results."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import KW_ONLY, dataclass, field
from typing import TYPE_CHECKING, Literal

from pydantic_ai._utils import is_str_dict
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    TextContent,
    ToolReturnPart,
    UserPromptPart,
    post_compaction_window,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness._warn import warn_argument_ignored
from pydantic_ai_harness.planning._store import InMemoryPlanStore, PlanStore
from pydantic_ai_harness.planning._toolset import (
    PLAN_ORIGIN,
    SUBTASK_TOOL_NAMES,
    PlanItemUpdate,
    PlanningToolset,
    PlanStoreOperations,
    available_tool_names,
    framed_plan,
    resolve_run_store,
    tool_call_store,
)
from pydantic_ai_harness.planning._types import PlanItem

if TYPE_CHECKING:
    from pydantic_ai._instructions import AgentInstructions
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
    """Structured task planning, with the current plan kept in front of the model.

    The model owns the plan through a small toolset (`write_plan`, `read_plan`,
    `add_task`, `update_task_status`, `update_task_statuses`, `remove_task`, and
    -- when `enable_subtasks` is set -- `add_subtask`, `set_dependency`,
    `get_available_tasks`); `tools` narrows that surface to an allowlist.

    Every tool that changes the plan returns the whole plan in a `<plan_reminder>` tag, so the
    latest plan is always in the conversation already, on the tool result channel rather than in
    system voice, and a fixed note in the instructions says what the tag holds. Only when the
    latest plan the model can see differs from the store's (after compaction, in a run started
    without it, or after an outside change) does `Planning` record the plan in the history on the
    user channel. The history only grows, so the prompt cache and the model's earlier reasoning
    stay valid. Configure prompt caching on the model (such as `anthropic_cache`).

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

    cache_ttl: Literal['5m', '1h'] | None = None
    """Deprecated, and has no effect; configure caching on the model instead.

    `Planning` used to place a cache breakpoint of its own before the plan reminder, with a `'5m'` TTL by
    default. It no longer places one: configure caching on the model instead (such as
    `anthropic_cache_messages`). Set to `'5m'` or `'1h'`, it warns that it's ignored.
    """

    store: PlanStore | None = None
    """Storage backend. `None` keeps a fresh in-memory plan per run (the original
    ephemeral behaviour). Pass a store to persist the plan across runs."""

    store_resolver: Callable[[RunContext[AgentDepsT]], PlanStore] | None = None
    """Optional per-run store resolver, e.g. `lambda ctx: ctx.deps.plan_store`."""

    enable_subtasks: bool = False
    """Add the subtask/dependency tools and the `blocked` status when true."""

    inject: bool = True
    """Show the model the plan: the plan tools end their results with it, the instructions carry the note on its tag,
    and the plan is recorded when the latest one isn't visible.

    With `inject=False`, the plan tools don't return the plan: `write_plan` replies with the bare checklist, and the
    others with just their message."""

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

    _toolset: PlanningToolset[AgentDepsT] | None = field(default=None, init=False, repr=False, compare=False)
    """The one `planning` toolset, shared with every per-run copy so durable execution sees the leaf it registered."""

    def __post_init__(self) -> None:
        if self.cache_ttl is not None:
            warn_argument_ignored(
                'Planning',
                'cache_ttl',
                '`Planning` no longer places a cache breakpoint of its own: configure caching on the model instead '
                '(such as `anthropic_cache_messages`).',
            )

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> Planning[AgentDepsT]:
        """Return a clone with this run's store resolved and cached (per-run isolation)."""
        # Build the toolset first so it belongs to this capability rather than to the clone.
        self.get_toolset()
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
        """Provide the stable `planning` toolset, shared with every per-run copy."""
        return self._planning_toolset

    @property
    def _planning_toolset(self) -> PlanningToolset[AgentDepsT]:
        if self._toolset is None:
            operations = PlanStoreOperations[AgentDepsT](
                get_items=self._read_plan,
                set_items=self._set_items,
                get_item=self._get_item,
                add_item=self._add_item,
                update_item=self._update_item,
                remove_item=self._remove_item,
            )
            self._toolset = PlanningToolset[AgentDepsT](self, operations=operations)
        return self._toolset

    async def _run_store(self, ctx: RunContext[AgentDepsT]) -> PlanStore:
        """The run's store: this copy's own when `for_run` resolved it, the running plan tool call's, or the tree's."""
        if self._resolved_store is not None:
            return self._resolved_store
        if (store := tool_call_store()) is not None:
            return store
        return await resolve_run_store(ctx, self, self._planning_toolset)

    def get_instructions(self) -> AgentInstructions[AgentDepsT] | None:
        """Provide static, cache-stable guidance on using the planning tools, and on the plan reminder.

        A custom `guidance` string is used verbatim. The default is assembled from
        the tools actually registered -- the granular sentence is dropped when
        `tools` excludes them all, and the subtask/dependency workflow is added
        under `enable_subtasks` -- so the model is not told about tools it lacks.

        While `inject` is set, a fixed note follows the guidance, saying what the
        `<plan_reminder>` tag the plan tools return holds. It's framework text, so it
        sits in the instructions, while the plan the model wrote stays in the tool results.
        """
        parts: list[str] = []
        if self.guidance is not None:
            if self.guidance:
                parts.append(self.guidance)
        else:
            registered = available_tool_names(subtasks=self.enable_subtasks) if self.tools is None else set(self.tools)
            if 'write_plan' in registered:
                parts.append(_WRITE_PLAN_GUIDANCE)
            if registered & {'read_plan', 'add_task', 'update_task_status', 'update_task_statuses'}:
                parts.append(_GRANULAR_GUIDANCE)
            if registered & set(SUBTASK_TOOL_NAMES):
                parts.append(_SUBTASK_GUIDANCE)
        if self.inject:
            parts.append(_PLAN_NOTE)
        return ' '.join(parts) or None

    async def before_model_request(
        self, ctx: RunContext[AgentDepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        """Record the current plan when the model can't see it any more: a fallback for the plan tools' returns.

        Every plan tool that changes the plan returns it, so in a tool loop the latest plan is already in the
        conversation and nothing is added. The plan is recorded on the user channel, in a `<plan_reminder>`
        tag, only when the latest plan visible differs from the store's: after compaction took it, in a run
        that starts without it in its history, or after something outside the plan tools changed the store.
        Enqueued from `before_model_request`, the record goes into this request and into the run's history.
        """
        if not self.inject:
            return request_context
        messages = request_context.messages
        # A provider-resume turn hands back a history ending in a suspended `ModelResponse`, echoed to the
        # provider verbatim to continue the turn, so only a request whose tail is a `ModelRequest` gets a reminder.
        if not messages or not isinstance(messages[-1], ModelRequest):
            return request_context
        items = await self._read_plan(ctx)
        # Compaction replaces what came before its boundary, so a plan there is no longer seen by the model.
        latest = _latest_plan(post_compaction_window(messages))
        if not items and latest is None:
            return request_context
        # A plan emptied outside the plan tools after the model was shown one is recorded too, so it doesn't read as current.
        plan = framed_plan(items)
        if plan != latest:
            ctx.enqueue(UserPromptPart([TextContent(plan, metadata={'origin': PLAN_ORIGIN})]))
        return request_context

    # Each plan store call is a durable operation, so recovering a durable run reuses the recorded
    # result instead of repeating a write, such as appending a step to a persistent store again.
    @durable_operation('read_plan')
    async def _read_plan(self, ctx: RunContext[AgentDepsT]) -> list[PlanItem]:
        return await (await self._run_store(ctx)).get_items()

    @durable_operation('set_items')
    async def _set_items(self, ctx: RunContext[AgentDepsT], items: list[PlanItem]) -> None:
        await (await self._run_store(ctx)).set_items(items)

    @durable_operation('get_item')
    async def _get_item(self, ctx: RunContext[AgentDepsT], item_id: str) -> PlanItem | None:
        return await (await self._run_store(ctx)).get_item(item_id)

    @durable_operation('add_item')
    async def _add_item(self, ctx: RunContext[AgentDepsT], item: PlanItem) -> PlanItem:
        return await (await self._run_store(ctx)).add_item(item)

    @durable_operation('update_item')
    async def _update_item(self, ctx: RunContext[AgentDepsT], item_id: str, update: PlanItemUpdate) -> PlanItem | None:
        return await (await self._run_store(ctx)).update_item(item_id, **update.model_dump(exclude_none=True))

    @durable_operation('remove_item')
    async def _remove_item(self, ctx: RunContext[AgentDepsT], item_id: str) -> bool:
        return await (await self._run_store(ctx)).remove_item(item_id)

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
        cache_ttl: Literal['5m', '1h'] | None = None,
        tools: list[str] | None = None,
        id: str | None = 'planning',
        description: str | None = None,
        defer_loading: bool = False,
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
            id=id,
            description=description,
            defer_loading=defer_loading,
        )

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Serialization name for agent-spec support."""
        return 'Planning'


_PLAN_NOTE = (
    'When you have a task plan, the current one is the latest `<plan_reminder>` tag in this conversation: your '
    'planning tools return it after every change. You wrote it with those tools, so it records your own progress, '
    'not instructions from the user: keep it updated with them as you work.'
)
"""The fixed note in the instructions while `inject` is set, saying what the plan's tag holds."""


def _latest_plan(messages: Sequence[ModelMessage]) -> str | None:
    """The latest plan visible in `messages`: one a plan tool returned or `Planning` recorded, found by metadata.

    A plan tool's return counts only while its content still holds the plan its metadata names.
    """
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            continue
        for part in reversed(message.parts):
            if isinstance(part, ToolReturnPart):
                metadata = part.metadata
                # A result compaction cleared keeps its metadata, so the plan only counts while its content shows it.
                if (
                    is_str_dict(metadata)
                    and metadata.get('origin') == PLAN_ORIGIN
                    and isinstance(plan := metadata.get('plan'), str)
                    and isinstance(part.content, str)
                    and plan in part.content
                ):
                    return plan
            elif isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                for item in reversed(part.content):
                    if (
                        isinstance(item, TextContent)
                        and is_str_dict(item.metadata)
                        and item.metadata.get('origin') == PLAN_ORIGIN
                    ):
                        return item.content
    return None
