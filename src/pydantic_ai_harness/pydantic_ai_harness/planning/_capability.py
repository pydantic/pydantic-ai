"""The `Planning` capability: task planning with a cache-safe live reminder."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from copy import copy
from dataclasses import KW_ONLY, dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from pydantic_ai._framing import neutralize_tags
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.messages import (
    CachePoint,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryFeedbackPart,
    TextContent,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.toolsets import AgentToolset
from pydantic_ai_harness.planning._store import InMemoryPlanStore, PlanStore
from pydantic_ai_harness.planning._toolset import (
    SUBTASK_TOOL_NAMES,
    PlanItemUpdate,
    PlanningToolset,
    PlanStoreOperations,
    available_tool_names,
    render_plan,
    resolve_run_store,
    tool_call_store,
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
    """Structured task planning that keeps the conversation in the prompt cache.

    The model owns the plan through a small toolset (`write_plan`, `read_plan`,
    `add_task`, `update_task_status`, `update_task_statuses`, `remove_task`, and
    -- when `enable_subtasks` is set -- `add_subtask`, `set_dependency`,
    `get_available_tasks`); `tools` narrows that surface to an allowlist. The
    current plan is surfaced back as an *ephemeral* reminder appended to the
    tail of each request and never persisted. The reminder leads with a
    `CachePoint`, which providers attach to the content right before it -- the
    last user prompt, tool result, or retry prompt, which the next request
    starts with -- so on most providers only the reminder itself is re-read each
    turn. On OpenRouter the breakpoint stops at the previous user message.

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
    """TTL for the cache breakpoint leading the plan reminder."""

    store: PlanStore | None = None
    """Storage backend. `None` keeps a fresh in-memory plan per run (the original
    ephemeral behaviour). Pass a store to persist the plan across runs."""

    store_resolver: Callable[[RunContext[AgentDepsT]], PlanStore] | None = None
    """Optional per-run store resolver, e.g. `lambda ctx: ctx.deps.plan_store`."""

    enable_subtasks: bool = False
    """Add the subtask/dependency tools and the `blocked` status when true."""

    inject: bool = True
    """Surface the current plan as an ephemeral tail reminder each turn."""

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
        """Append the ephemeral plan reminder to the request, leading with a cache breakpoint.

        The breakpoint marks the end of the content before the reminder, so each request caches
        what the next one starts with, including every tool result since the last user prompt,
        while the reminder stays outside the cached prefix and out of message history.
        """
        if not self.inject:
            return await handler(request_context)
        items = await self._read_plan(ctx)
        if not items:
            return await handler(request_context)
        messages = request_context.messages
        last = messages[-1]
        if isinstance(last, ModelRequest):
            content: list[UserContent] = ['<plan-reminder>\n', _reminder_text(render_plan(items))]
            # With no content before the reminder (an instructions-only first request) there is no
            # prefix to protect, and providers reject a breakpoint with nothing to attach to.
            if _has_cache_point_target(last.parts):
                content.insert(0, CachePoint(ttl=self.cache_ttl))
            messages[-1] = replace(last, parts=[*last.parts, UserPromptPart(content=content)])
        return await handler(request_context)

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


# TODO: switch to `wrap()` from the public `pydantic_ai.framing` module once it lands (#10957), registering
# `plan-reminder` as a framing tag there.
# The `<` of an opening or closing `plan-reminder` tag, in any case, spacing or with attributes, or cut off at the end
# of the plan, where the frame's own end tag would complete it. Core's own framing tags are left to `neutralize_tags`.
_PLAN_REMINDER_TAG_OPENER = re.compile(r'<(?=\s*(?:/\s*)?plan-reminder(?=[\s/>]|$))', re.IGNORECASE)


def _reminder_text(plan: str) -> str:
    """The reminder's body and closing tag, with the plan's step text kept inside the `<plan-reminder>` frame.

    The steps are written by the model, so the `<` of every opening or closing `plan-reminder` tag and of every
    framing tag core writes is escaped as `&lt;`.
    """
    body = _PLAN_REMINDER_TAG_OPENER.sub('&lt;', neutralize_tags(plan))
    return f'Your current plan (keep it updated with the planning tools):\n\n{body}\n</plan-reminder>'


def _has_cache_point_target(parts: Sequence[ModelRequestPart]) -> bool:
    """Whether these request parts already carry content a leading `CachePoint` can attach to.

    Providers attach a `CachePoint` that opens a user prompt to the content right before it, which is
    the end of the durable history when the prompt is the plan reminder, whether that history ends in
    a user prompt, a tool return, or retry feedback. With nothing before it to attach to, providers
    that honor `CachePoint` reject it, so the reminder only leads with one when the request already
    contributes such content. `SystemPromptPart` maps to the system field, not to message content, so
    it does not count.

    `RetryFeedbackPart` counts: it is translated at request time into a `<validation_errors>`-fenced
    user prompt when it quotes the model's own output, and otherwise into a mid-conversation system
    prompt, which providers without an inline system channel receive as `<system>`-tagged user text.
    """
    for part in parts:
        if isinstance(part, ToolReturnPart | RetryFeedbackPart):
            return True
        if isinstance(part, UserPromptPart):
            content = part.content
            if isinstance(content, str):
                if content:
                    return True
            elif any(_is_user_content_item(item) for item in content):
                return True
    return False


def _is_user_content_item(item: object) -> bool:
    if isinstance(item, CachePoint):
        return False
    if isinstance(item, str):
        return bool(item)
    if isinstance(item, TextContent):
        return bool(item.content)
    return True
