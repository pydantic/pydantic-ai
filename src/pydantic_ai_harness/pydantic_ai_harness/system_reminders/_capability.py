"""System reminders capability: re-inject behavioral guidance without busting the cache."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic, Literal

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import (
    CachePoint,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    TextContent,
    TextPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.tools import AgentDepsT, RunContext

from pydantic_ai_harness._usage import reserved_usage_limits

if TYPE_CHECKING:
    from pydantic_ai.capabilities.abstract import WrapModelRequestHandler
    from pydantic_ai.models import ModelRequestContext


@dataclass
class Reminder(Generic[AgentDepsT]):
    r"""A static reminder injected on a cadence during an agent run."""

    content: str
    """The reminder text."""

    interval: int = 1
    """Fire every N model requests within a run. `interval=3` fires on the 3rd, 6th, 9th, ... request."""

    first_after: int | None = None
    """Request number of the first fire. `None` (the default) fires on the first multiple of
    `interval` (plain modulo). When set, the reminder fires at `first_after`, then every
    `interval` requests after that."""

    trigger: Callable[[RunContext[AgentDepsT]], bool] | None = None
    """Optional predicate over the current `RunContext`. When set, the reminder fires only when
    the trigger returns `True` *and* the cadence condition is met."""

    max_fires: int | None = None
    """Maximum number of times this reminder may fire within a run. `None` means no limit."""

    tag: str | None = 'system-reminder'
    r"""When set, wrap the content in an XML tag: `<tag>\ncontent\n</tag>`. Defaults to
    `'system-reminder'` (Claude Code's convention); set `None` to emit the raw content."""

    def __post_init__(self) -> None:
        if self.interval < 1:
            raise ValueError(f'interval must be >= 1, got {self.interval}')
        if self.first_after is not None and self.first_after < 1:
            raise ValueError(f'first_after must be >= 1, got {self.first_after}')
        if self.max_fires is not None and self.max_fires < 1:
            raise ValueError(f'max_fires must be >= 1, got {self.max_fires}')


DynamicReminder = Callable[[RunContext[AgentDepsT]], 'str | None']
"""A callable returning reminder text (or `None` to skip), evaluated every model request.

The callable reads the current `RunContext`, so it subsumes token-budget, post-compaction,
mode-switch, and other conditions without hardcoded detectors: the user writes the condition.
"""

AsyncDynamicReminder = Callable[[RunContext[AgentDepsT]], 'Awaitable[str | None]']
"""Async variant of `DynamicReminder`."""


@dataclass
class SystemReminders(AbstractCapability[AgentDepsT]):
    r"""Inject periodic or conditional reminders to counter instruction fade in long sessions.

    Long multi-turn runs suffer instruction fade: after many tool-use turns the model
    progressively ignores start-of-session guidance. `SystemReminders` re-injects targeted
    guidance mid-run, either on a fixed cadence (`Reminder`) or reactively from a callable
    (`dynamic_reminders`).

    Cache safety is the design constraint. Reminders are appended to the *tail* of each
    request as an ephemeral `UserPromptPart` behind a `CachePoint`, inside `wrap_model_request`
    (which runs after core persists the durable history). So reminders reach the model but
    never enter `message_history`: no stale reminders accumulate, and the cached prefix stays
    byte-identical across turns -- only the small reminder falls outside the cache. Injecting
    into the system prompt or a persisted part instead would bust the cache prefix on every
    fire and let reminders pile up.

    ```python
    from pydantic_ai import Agent
    from pydantic_ai_harness.system_reminders import SystemReminders, Reminder

    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[
            SystemReminders(
                reminders=[Reminder('Stay focused on the original request.', interval=5)],
            )
        ],
    )
    ```
    """

    reminders: Sequence[Reminder[AgentDepsT]] = ()
    """Static reminders injected on a cadence."""

    dynamic_reminders: Sequence[DynamicReminder[AgentDepsT] | AsyncDynamicReminder[AgentDepsT]] = ()
    """Callables evaluated every model request; return text to inject or `None` to skip."""

    cache_ttl: Literal['5m', '1h'] = '5m'
    """TTL for the cache breakpoint placed before the tail reminder."""

    on_fire: Callable[[str], None] | None = None
    """Optional observability callback invoked with each rendered reminder as it fires."""

    _request_count: int = field(default=0, init=False, repr=False, compare=False)
    # Keyed by `id(reminder)`, not list index: a user callback that inserts or removes a
    # reminder mid-run must not shift another reminder onto a stale budget.
    _fire_counts: dict[int, int] = field(default_factory=dict[int, int], init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.reminders and not self.dynamic_reminders:
            raise ValueError('At least one static or dynamic reminder must be provided.')

    async def for_run(self, ctx: RunContext[AgentDepsT]) -> SystemReminders[AgentDepsT]:
        """Return a fresh per-run instance with reset counters (config preserved).

        `replace` builds a new instance whose generated `__init__` re-initializes the
        `init=False` fields -- `_request_count` back to `0` and `_fire_counts` to an empty
        dict -- so concurrent runs on the same agent never share fire state.
        """
        return replace(self)

    async def wrap_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        """Append fired reminders to the request tail behind a cache breakpoint, then call the model.

        Runs after core persists the durable history; the per-request message list mutated
        here is never written back, so the reminder and its `CachePoint` reach the model but
        never enter `ctx.state.message_history`.
        """
        messages = request_context.messages
        # A provider-resume turn (`_prepare_resume_request`) hands back a message list whose
        # tail is a suspended `ModelResponse`, and that exact history is echoed to the provider
        # verbatim to continue the turn -- injecting into it would corrupt the continuation. So
        # only a real `ModelRequest` tail carries a reminder, and only such a turn spends a
        # cadence slot (the counter advances inside the guard, not before it).
        if messages and isinstance(last := messages[-1], ModelRequest):
            self._request_count += 1
            # Select which reminders fire before mutating any state: `_eligible_static` is
            # pure, and `_collect_dynamic` can raise (a user callback, a model call). Fire
            # state and `on_fire` are committed only once the reminder is actually appended, so
            # a raising dynamic reminder never leaves a static reminder's `max_fires` budget
            # spent or reports a fire that never reached the model.
            fired = self._eligible_static(ctx)
            dynamic_texts = await self._collect_dynamic(ctx)
            texts = [text for _, text in fired] + dynamic_texts
            if texts:
                content: list[CachePoint | str] = []
                # A leading `CachePoint` is only valid when the request already carries a
                # user-content block for it to attach to; Anthropic and Bedrock raise otherwise.
                if _has_user_content(last.parts):
                    content.append(CachePoint(ttl=self.cache_ttl))
                content.append('\n\n'.join(texts))
                messages[-1] = replace(last, parts=[*last.parts, UserPromptPart(content=content)])
                for key, _text in fired:
                    self._fire_counts[key] = self._fire_counts.get(key, 0) + 1
                if self.on_fire is not None:
                    for text in texts:
                        self.on_fire(text)
        return await handler(request_context)

    def _eligible_static(self, ctx: RunContext[AgentDepsT]) -> list[tuple[int, str]]:
        """Static reminders whose cadence, trigger, and `max_fires` allow firing this request.

        Returns `(id(reminder), rendered_text)` pairs. Pure: it reads `_fire_counts` but does
        not mutate it, so the caller commits fire state only after the reminders are injected.
        The `reminders` sequence is snapshotted so a `trigger` that mutates it mid-iteration
        cannot desync this pass.
        """
        eligible: list[tuple[int, str]] = []
        pending: dict[int, int] = {}
        for reminder in tuple(self.reminders):
            if not _should_fire(reminder, self._request_count):
                continue
            if reminder.trigger is not None and not reminder.trigger(ctx):
                continue
            key = id(reminder)
            # `pending` counts this reminder's fires already selected in this same pass, so the
            # same instance listed more than once still respects `max_fires` (its budget is not
            # committed until after injection).
            if (
                reminder.max_fires is not None
                and self._fire_counts.get(key, 0) + pending.get(key, 0) >= reminder.max_fires
            ):
                continue
            pending[key] = pending.get(key, 0) + 1
            eligible.append((key, _render_content(reminder)))
        return eligible

    async def _collect_dynamic(self, ctx: RunContext[AgentDepsT]) -> list[str]:
        texts: list[str] = []
        # Snapshot so a callback that appends to `dynamic_reminders` cannot extend the loop.
        for dynamic in tuple(self.dynamic_reminders):
            result = dynamic(ctx)
            if isinstance(result, Awaitable):
                result = await result
            if result is not None:
                texts.append(result)
        return texts

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: reminders take arbitrary callables."""
        return None


@dataclass
class GoalReanchor(Generic[AgentDepsT]):
    """Zero-cost dynamic reminder that re-states the run's first user request as the anchor.

    No model call and no dependencies: it reads the first user message from `ctx.messages`
    and asks the model to check its next action advances that goal. Falls back to a static
    line when there is no user message yet. Add it to `SystemReminders.dynamic_reminders`.
    """

    fallback: str = 'Stay on task.'

    def __call__(self, ctx: RunContext[AgentDepsT]) -> str | None:
        goal = _first_user_text(ctx.messages)
        if goal is None:
            return self.fallback
        return (
            f'Your original request was: "{goal}". Check that your next action advances it. '
            'If it is already satisfied, produce the final answer.'
        )


_LLM_INSTRUCTIONS = (
    'You write a short stay-on-task reminder for an AI agent mid-run. Given the original '
    'goal and recent activity, produce at most two sentences that refocus the agent on the '
    'goal. Output only the reminder text.'
)


@dataclass
class LLMReminder(Generic[AgentDepsT]):
    """Dynamic reminder whose text a model generates from a compact transcript.

    Opt-in and dependency-free (it uses `pydantic_ai.Agent`). `model` is required and has no
    default -- pass an explicit model. On any error it falls back to `GoalReanchor` text, so a
    failed generation never blocks the run. Add it to `SystemReminders.dynamic_reminders`.

    Like every dynamic reminder it is evaluated on every model request, so it issues one extra
    model call per turn; its usage is threaded onto the parent run (`ctx.usage`) and it runs
    under the parent's `usage_limits` minus one reserved request, so the reminder cannot push a
    run past its `request_limit`. Once the budget is that tight the generation is skipped and
    `GoalReanchor` text is used instead. Gate it on a cadence (see the docs) if per-turn
    generation is too costly.

    The generation runs inside `wrap_model_request`, so under durable execution (Temporal,
    DBOS, Prefect) it executes in orchestration context rather than a durable step: the model
    call is non-deterministic on replay and is not checkpointed, and its errors fall back
    silently to `GoalReanchor`. For durable runs prefer `GoalReanchor`, which makes no model
    call, or gate `LLMReminder` off.
    """

    model: Model | KnownModelName | str
    max_context_messages: int = 10
    instructions: str = _LLM_INSTRUCTIONS
    _agent: Agent[None, str] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.max_context_messages < 1:
            raise ValueError(f'max_context_messages must be >= 1, got {self.max_context_messages}')

    async def __call__(self, ctx: RunContext[AgentDepsT]) -> str | None:
        try:
            agent = self._agent
            if agent is None:
                agent = Agent(self.model, instructions=self.instructions, output_type=str)
                self._agent = agent
            transcript = _build_compact_transcript(ctx.messages, self.max_context_messages)
            result = await agent.run(transcript, usage=ctx.usage, usage_limits=reserved_usage_limits(ctx.usage_limits))
            text = result.output.strip()
            return text or None
        except Exception:
            return GoalReanchor[AgentDepsT]()(ctx)


def _should_fire(reminder: Reminder[AgentDepsT], count: int) -> bool:
    """Whether `reminder`'s cadence fires on request number `count` (1-based)."""
    base = reminder.interval if reminder.first_after is None else reminder.first_after
    return count >= base and (count - base) % reminder.interval == 0


def _render_content(reminder: Reminder[AgentDepsT]) -> str:
    """Return `reminder`'s content, XML-wrapped when its `tag` is set."""
    if reminder.tag is not None:
        return f'<{reminder.tag}>\n{reminder.content}\n</{reminder.tag}>'
    return reminder.content


def _has_user_content(parts: Sequence[ModelRequestPart]) -> bool:
    """Whether these request parts already carry a block a `CachePoint` can attach to.

    Anthropic and Bedrock reject a `CachePoint` that is the first content of a user message,
    so the tail reminder leads with one only when the request already contributes user-mappable
    content: a non-empty user prompt, a tool return, or a retry prompt. `SystemPromptPart` maps
    to the system field, not user content, so it does not count.
    """
    for part in parts:
        if isinstance(part, (ToolReturnPart, RetryPromptPart)):
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


def _first_user_text(messages: Sequence[ModelMessage]) -> str | None:
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    text = _prompt_text(part.content)
                    if text:
                        return text
    return None


def _prompt_text(content: str | Sequence[object]) -> str:
    if isinstance(content, str):
        return content
    texts: list[str] = []
    for item in content:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, TextContent):
            texts.append(item.content)
    return ' '.join(texts)


def _build_compact_transcript(messages: Sequence[ModelMessage], max_messages: int) -> str:
    goal = _first_user_text(messages)
    recent = _recent_texts(messages, max_messages)
    sections: list[str] = []
    if goal is not None:
        sections.append(f'Original goal: {goal}')
    if recent:
        sections.append('Recent activity:\n' + '\n'.join(recent))
    return '\n\n'.join(sections) if sections else 'No activity yet.'


def _recent_texts(messages: Sequence[ModelMessage], max_messages: int) -> list[str]:
    fragments: list[str] = []
    for msg in messages:
        for part in msg.parts:
            if isinstance(part, UserPromptPart):
                text = _prompt_text(part.content)
                if text:
                    fragments.append(f'user: {text}')
            elif isinstance(part, TextPart) and part.content:
                fragments.append(f'assistant: {part.content}')
    return fragments[-max_messages:]
