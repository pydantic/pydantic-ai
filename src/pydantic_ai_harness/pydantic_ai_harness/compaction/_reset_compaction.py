"""`ResetCompaction` -- clear the history in one step and point the model at its work log."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import KW_ONLY, dataclass, replace
from typing import TYPE_CHECKING

from pydantic_ai._capability_loads import deferred_capability_ids, keep_loaded_state
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering, durable_operation
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    DataCapabilityPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    OperatorSource,
    ToolResultSource,
)
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai_harness._work_log_kinds import WORK_LOG_READ_KIND, WORK_LOG_SEARCH_KIND
from pydantic_ai_harness.compaction._context_window import DEFAULT_CONTEXT_WINDOW
from pydantic_ai_harness.compaction._pinning import reinject_pinned
from pydantic_ai_harness.compaction._receipts import receipt_capability_id
from pydantic_ai_harness.compaction._shared import (
    compact_with_span,
    context_for_request,
    estimate_token_count,
    exceeds,
    find_first_user_message,
    find_safe_cutoff,
    is_user_part,
    iter_tool_pairs,
    prepend_first_user_message,
    record_compaction_reclaim,
    record_span_attributes,
    require_trigger,
    resolve_token_trigger,
    user_turn_starts,
    validate_token_trigger,
)
from pydantic_ai_harness.planning import Planning

if TYPE_CHECKING:
    from pydantic_ai.models import ModelRequestContext

    # Not imported at runtime: importing `work_log` imports this package, so a runtime import would cycle.
    from pydantic_ai_harness.work_log import WorkLog


@dataclass(kw_only=True)
class ResetNoticePart(
    DataCapabilityPart[str],
    namespace='reset_compaction',
    name='notice',
    tag='context_cleared',
    # Without it, a model asked who wrote the frame can credit it to the user.
    note=(
        'The accompanying {frame} is from the application, not the user: it marks where earlier messages were cleared.'
    ),
):
    """The notice `ResetCompaction` leaves where it cleared the conversation, on the user channel.

    Its text is the capability's own, from `ResetCompaction.notice`. It goes with a note in the application's
    voice saying the frame is the application's. Find notices in history by this type: a later reset reads its
    typed fields, which are never sent to the model.
    """

    number: int
    """How many times the conversation has been cleared, this time included."""

    work_log: str | None
    """The work log's handle the text names: `None` for a store the agent can't open, such as
    `InMemoryWorkLogStore`, or when the conversation is not known."""

    kept_responses: int
    """Model responses the reset kept after the notice, so a later reset can tell whether anything is new."""


@dataclass(kw_only=True)
class CurrentStatePart(DataCapabilityPart[str], namespace='reset_compaction', name='state', tag='current_state'):
    """The task's current state, as `ResetCompaction.restate` returns it, right after the notice.

    Return one from `restate` to give the text its real source, such as `OperatorSource()` for text the
    application wrote; a plain string from `restate` is recorded as the callable's output, with a
    `ToolResultSource` named after it. A later reset replaces it.
    """


@dataclass(frozen=True, kw_only=True)
class ResetNotice:
    """What a reset did, handed to `ResetCompaction.notice` to write the text of the notice it leaves."""

    number: int
    """How many times the conversation has been reset, this reset included."""

    work_log: str | None
    """Where the work log is, from its store's `locate`: `None` for a store the agent can't open, such as
    `InMemoryWorkLogStore`, or when the conversation is not known."""

    search_tool: str
    """The name the model sees for `search_work_log`."""

    read_tool: str
    """The name the model sees for `read_work_log`."""

    replies_logged: bool
    """Whether the work log records the model's replies, as `WorkLog.record_text` sets."""

    cleared_messages: int
    """Messages this reset removed."""

    kept_first_request: bool
    """Whether the conversation's first user request is still in it."""

    kept_user_turns: int
    """User turns after the first request still in the conversation, besides one in the kept messages."""

    kept_messages: int
    """Messages kept after the notice, at the end of the conversation."""

    kept_tool_calls: int
    """Tool calls kept with their results among those messages."""


def default_reset_notice(notice: ResetNotice) -> str:
    """The default text of the notice a reset leaves: that the conversation was cleared, and where the earlier work is.

    It says the task continues and how many times the conversation has been cleared, names the work log and the tools
    that search and read it, with a first move: list the newest entries or search, then read only the entries needed.
    It asks the model to read a file again before editing it, since an entry shows the file as it was, and says what
    the reset kept. It leaves out the model's replies when the work log doesn't record them.
    """
    cleared = 'once' if notice.number == 1 else f'{notice.number} times'
    log = 'the work log' if notice.work_log is None else f'the work log `{notice.work_log}`'
    work = (
        'Your earlier tool calls with full results, including calls made inside code, and your earlier replies are'
        if notice.replies_logged
        else 'Your earlier tool calls with full results, including calls made inside code, are'
    )
    requests = "The user's requests" if notice.kept_user_turns else "The user's original request"
    calls = (
        'latest tool call and its result'
        if notice.kept_tool_calls == 1
        else f'{notice.kept_tool_calls} latest tool calls and their results'
    )
    if notice.kept_first_request and notice.kept_tool_calls:
        kept = f' {requests} and your {calls} are still in this conversation.'
    elif notice.kept_first_request:
        kept = f' {requests} {"are" if notice.kept_user_turns else "is"} still in this conversation.'
    elif notice.kept_tool_calls:
        kept = f' Your {calls} are still in this conversation.'
    else:
        kept = ''
    return (
        'The conversation before this point was cleared to free context; you are continuing the same task '
        f'(cleared {cleared} in this conversation). {work} in {log}. To pick up where you left off, call '
        f'`{notice.search_tool}` with no query to list your most recent entries, or with keywords to find one, then '
        f'read only the entries you need in full with `{notice.read_tool}`; both work as tools and from code. The '
        f'log shows files as they were when read: read a file again before you edit it.{kept}'
    )


@dataclass
class ResetCompaction(AbstractCapability[AgentDepsT]):
    """Clear the conversation down to a small kept shape, and leave a notice pointing the model at its work log.

    When the history passes a trigger, everything is cleared except the first user request, what
    the user wrote in later turns, pinned content, loaded capabilities and the last `keep_messages`
    messages. A notice on the user channel says the conversation was cleared and names the
    `WorkLog`'s tools, so the model can search and read the full results of its earlier work instead
    of running it again. No model is called. A later reset replaces the notice rather than adding
    another.

    `work_log` must be the `WorkLog` registered on the same agent. It is passed explicitly because
    used as a strategy, such as through `compact_now`, the reset has no agent to find it on.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_harness.compaction import ResetCompaction
        from pydantic_ai_harness.work_log import FileWorkLogStore, WorkLog

        work_log = WorkLog(FileWorkLogStore('work-logs'))
        agent = Agent('openai:gpt-5', capabilities=[work_log, ResetCompaction(work_log, max_fraction=0.6)])
        ```
    """

    work_log: WorkLog[AgentDepsT]
    """The `WorkLog` registered on the same agent, whose log the notice points at."""

    max_messages: int | None = None
    """Reset when the history has more messages than this. `None` disables."""

    max_tokens: int | None = None
    """Reset when the history's estimated tokens exceed this. `None` disables."""

    _: KW_ONLY

    max_fraction: float | None = None
    """Reset when estimated tokens exceed this fraction of the request's model's context window.

    Resolved per request from the request's model. Mutually exclusive with `max_tokens`. Registered on an
    agent, it needs one of `max_messages`, `max_tokens` or `max_fraction`, and a run raises `UserError` at its
    start without one; as a strategy (a `TieredCompaction` tier, a `FallbackCompaction` entry, or the strategy
    of `CompactOnRequest` or `compact_now`) it needs none."""

    context_window: int | None = None
    """Window override in tokens. `None` resolves it from the request's model.

    Only consulted alongside `max_fraction`."""

    fallback_context_window: int = DEFAULT_CONTEXT_WINDOW
    """Window assumed when the request's model is not in the pricing registry.

    Only consulted alongside `max_fraction`."""

    keep_messages: int = 2
    """Messages kept at the end of the conversation, after the notice.

    Tool calls stay with their results, and the newest message is always kept, so `0` keeps one."""

    keep_user_turns: bool = True
    """Keep what the user wrote in each turn after the first, ahead of the kept messages, when they don't hold it.

    Only the user's own parts, all in one request, so the kept history grows by no message per turn: tool
    results and harness data in the same requests are cleared. With `False`, only the first request and the
    kept messages hold the user's words."""

    notice: Callable[[ResetNotice], str] = default_reset_notice
    """Write the notice's text from what the reset did. Defaults to `default_reset_notice`."""

    restate: Callable[[RunContext[AgentDepsT]], Awaitable[str | CurrentStatePart | None]] | None = None
    """Return the task's current state to put after the notice, in a `<current_state>` frame.

    A string is recorded as the output of the `restate` callable (a `ToolResultSource` named after it), so text in
    it that looks like a frame gets a note. Return a `CurrentStatePart` to give it another source, such as
    `OperatorSource()` for text your application wrote. `None` or an empty string adds nothing. Runs as a durable
    operation, and an error it raises fails the request."""

    preserve_loaded_capabilities: bool = True
    """When `True`, a deferred capability loaded in the cleared messages stays loaded.

    Its load is recorded again ahead of the kept messages. When `False`, a note names the capabilities that
    were unloaded, so the model can load the ones it still needs."""

    tokenizer: Callable[[str], int] | None = None
    """Optional tokenizer for the token trigger and the span. When `None`, uses a ~4 characters-per-token heuristic."""

    id: str | None = 'reset_compaction'
    """Stable capability ID: durable execution names the `restate` operation by it."""

    def __post_init__(self) -> None:
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError('max_messages must be positive.')
        validate_token_trigger(self.max_tokens, self.max_fraction, self.fallback_context_window, self.context_window)
        if self.keep_messages < 0:
            raise ValueError('keep_messages must be non-negative.')

    @classmethod
    def get_serialization_name(cls) -> str | None:
        """Not spec-serializable: `work_log` is a live capability, and `notice` and `restate` are callables."""
        return None

    def get_ordering(self) -> CapabilityOrdering:
        # Outside `Planning`: the reset runs first in `before_model_request`, so Planning finds the plan gone
        # and records it again in the same request.
        return CapabilityOrdering(wraps=[Planning], requires=[type(self.work_log)])

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Reject a registration without a trigger, or with a `work_log` the agent doesn't register under its id.

        With another `WorkLog`, the notice would name a log that nothing writes to.
        """
        require_trigger('ResetCompaction', self.max_messages, self.max_tokens, self.max_fraction)
        work_log = self.work_log
        if work_log.id is None:
            raise UserError(
                "`ResetCompaction` needs a `WorkLog` with an `id`, such as the default `'work_log'`, to find it on the "
                'agent: with `id=None` it cannot check that the notice names the log the agent writes.'
            )
        registered = ctx.capabilities.get(work_log.id)
        if registered != work_log:
            raise UserError(
                f'`ResetCompaction.work_log` is {work_log!r}, but the agent registers {registered!r} under id '
                f'{work_log.id!r}. Pass the `WorkLog` registered on the agent, so the notice names the log it writes.'
            )

    async def compact(self, messages: list[ModelMessage], ctx: RunContext[AgentDepsT]) -> list[ModelMessage]:
        """Clear all but the first request, the notice, pins, loaded state, the user's later turns and the tail.

        The notice is built here, not enqueued, so a use as a strategy gets it too.
        """
        cutoff = find_safe_cutoff(messages, self.keep_messages)
        if cutoff <= 0:
            return messages
        first = find_first_user_message(messages)
        head = prepend_first_user_message(messages, cutoff, [])
        kept_turns = [
            message
            for index in (user_turn_starts(messages) if self.keep_user_turns else [])
            if index < cutoff and (message := messages[index]) is not first and isinstance(message, ModelRequest)
        ]
        # One request for every kept turn, so a message-count trigger can't be crossed by the turns alone. The
        # user's parts only: tool results merged into a turn's request would have lost their calls.
        turns: list[ModelMessage] = (
            [replace(kept_turns[-1], parts=[part for turn in kept_turns for part in turn.parts if is_user_part(part)])]
            if kept_turns
            else []
        )
        tail = without_reset_parts(messages[cutoff:])
        log_id = ctx.conversation_id or next(
            (message.conversation_id for message in reversed(messages) if message.conversation_id), None
        )
        facts = ResetNotice(
            number=reset_count(messages) + 1,
            work_log=None if log_id is None else self.work_log.store.locate(log_id),
            search_tool=ctx.tool_name_for_kind(WORK_LOG_SEARCH_KIND) or 'search_work_log',
            read_tool=ctx.tool_name_for_kind(WORK_LOG_READ_KIND) or 'read_work_log',
            replies_logged=self.work_log.record_text,
            cleared_messages=cutoff - len(head) - len(kept_turns),
            kept_first_request=first is not None,
            kept_user_turns=len(kept_turns),
            kept_messages=len(tail),
            kept_tool_calls=len(iter_tool_pairs(tail)),
        )
        record_span_attributes({'compaction.reset.number': facts.number})
        kept_responses = sum(isinstance(message, ModelResponse) for message in tail)
        parts: list[ModelRequestPart] = [
            ResetNoticePart(
                content=self.notice(facts),
                source=OperatorSource(),
                capability_id=receipt_capability_id(ctx, self),
                number=facts.number,
                work_log=facts.work_log,
                kept_responses=kept_responses,
            )
        ]
        if self.restate is not None and (state := await self._restate(ctx)):
            if isinstance(state, str):
                # Text the code ran rather than wrote, such as command output: the callable's output.
                name = getattr(self.restate, '__name__', None)
                source = ToolResultSource(tool_name=name if isinstance(name, str) else 'restate')
                state = CurrentStatePart(content=state, source=source)
            parts.append(state)
        compacted = reinject_pinned(
            messages, [*head, ModelRequest(parts=parts), *turns, *tail], insert_at=len(head) + 1
        )
        return keep_loaded_state(
            messages,
            compacted,
            kept_from=len(compacted) - len(tail) - len(turns),
            deferred_ids=deferred_capability_ids(ctx),
            preserve=self.preserve_loaded_capabilities,
        )

    @durable_operation('restate')
    async def _restate(self, ctx: RunContext[AgentDepsT]) -> str | CurrentStatePart | None:
        """The caller's `restate`, as a durable operation, so a replay reuses its recorded result."""
        assert self.restate is not None
        return await self.restate(ctx)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Reset when the history passes the trigger and the model has responded since the last reset."""
        messages: list[ModelMessage] = list(ctx.messages)
        request_ctx = context_for_request(ctx, request_context)
        token_trigger = resolve_token_trigger(
            self.max_tokens, self.max_fraction, request_ctx.model, self.fallback_context_window, self.context_window
        )
        if not exceeds(
            messages,
            self.max_messages,
            token_trigger,
            self.tokenizer,
            model_request_parameters=request_context.model_request_parameters,
        ):
            return request_context
        # Nothing new since the last reset: resetting again would clear only what that reset kept.
        new_responses = responses_past_kept(messages)
        if new_responses is not None and new_responses <= 0:
            return request_context
        compacted = await compact_with_span(
            request_ctx,
            strategy='ResetCompaction',
            messages=messages,
            compact=lambda: self.compact(messages, request_ctx),
            tokenizer=self.tokenizer,
        )
        ctx.messages[:] = compacted
        record_compaction_reclaim(
            ctx,
            estimate_token_count(messages, self.tokenizer),
            estimate_token_count(compacted, self.tokenizer),
        )
        return replace(request_context, messages=compacted)


def is_reset_part(part: ModelRequestPart) -> bool:
    """Whether *part* is a reset notice or the state `restate` added with it."""
    return isinstance(part, ResetNoticePart | CurrentStatePart)


def newest_reset_notice(messages: Sequence[ModelMessage]) -> ResetNoticePart | None:
    """The reset notice in *messages* with the highest number, including one merged into another request, or `None`.

    The highest rather than the last: `result.conversation` places a notice after the messages the run was given,
    so a notice from a reset in a continued run comes after the responses that reset kept, and need not come last.
    """
    notices = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ResetNoticePart)
    ]
    return max(notices, key=lambda notice: notice.number, default=None)


def reset_count(messages: Sequence[ModelMessage]) -> int:
    """How many times the conversation in *messages* has been cleared: the newest notice's number, or 0."""
    notice = newest_reset_notice(messages)
    return 0 if notice is None else notice.number


def without_reset_parts(messages: Sequence[ModelMessage]) -> list[ModelMessage]:
    """*messages* without an earlier reset's parts, and without the requests that leaves empty."""
    out: list[ModelMessage] = []
    for message in messages:
        if not (isinstance(message, ModelRequest) and any(is_reset_part(part) for part in message.parts)):
            out.append(message)
        elif parts := [part for part in message.parts if not is_reset_part(part)]:
            out.append(replace(message, parts=parts))
    return out


def responses_past_kept(messages: Sequence[ModelMessage]) -> int | None:
    """Model responses in *messages* beyond those the newest reset kept, or `None` before any reset.

    Counted over the whole history, not after the notice: a reset's own history holds no responses but the ones it
    kept, while `result.conversation` also holds the ones it cleared and can place the notice after the kept ones.
    Responses rather than messages: core merges consecutive requests in a continued history, never responses.
    """
    notice = newest_reset_notice(messages)
    if notice is None:
        return None
    return sum(isinstance(message, ModelResponse) for message in messages) - notice.kept_responses
