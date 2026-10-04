"""Prompt-cache collapse detection, shared by core's `Instrumentation` and harness's `WarnOnCacheBusts`.

Both answer the same question about the same provider cache: did a request read back far less of the
cacheable prefix than the conversation had already established, and if so, is that explained by the
provider's retention window? This module is the single answer. Its callers are only outputs over it:
`Instrumentation` writes span attributes and a span event, `WarnOnCacheBusts` a Python warning.

Each caller owns its own marks (its own `ConversationCacheMarkStore`), so when both are active on one
agent neither sees marks the other has already advanced past a collapse.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Set as AbstractSet
from dataclasses import KW_ONLY, dataclass, field, replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

from . import _utils
from .messages import CompactionPart, ModelResponse, NativeToolCallPart
from .profiles import ModelProfile, _expected_cache_retention  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from .models import ModelRequestContext

CollapseReason: TypeAlias = Literal['unexpected', 'ttl_expired', 'compacted', 'unknown', 'unreported']
"""Why a cached prefix collapsed, as far as the history, the provider's usage, and its retention window can tell.

- `'unexpected'`: the retention window should still have been active, so the prefix moved.
- `'ttl_expired'`: the gap since the last request for the same cache exceeded the retention window.
- `'compacted'`: a [`CompactionPart`][pydantic_ai.messages.CompactionPart] replaced the history since the
  last request, which shrinks the prefix by design.
- `'unknown'`: the provider publishes no retention window, so the collapse can't be attributed.
- `'unreported'`: the response reported no cache usage at all, so the cause can't be determined.
"""

MIN_MISSED_RATIO = 0.05
"""A request collapsed when it falls short of the established prefix by more than this fraction of it..."""
MIN_MISSED_TOKENS = 2000
"""...and by at least this many tokens: the thresholds Claude Code uses for a prompt-cache miss.

Message history is append-only, so any real shortfall means the prefix moved or the cache expired. The
thresholds only keep provider rounding and small partial misses out; the classification is what keeps
expiries from alerting.
"""

CacheKey: TypeAlias = tuple[str | None, str | None, str | None]
"""A response's `(provider_name, provider_url, model_name)`: which provider cache its tokens came from."""


@dataclass(frozen=True)
class CacheMark:
    """What one provider cache held for a conversation as of its last reported response."""

    established_tokens: int
    """The cached-prefix size later requests are judged against."""
    last_seen: datetime
    """When the provider last reported cache usage for this key, which starts its retention clock."""
    run_id: str | None
    """The run that made that request."""
    compactions: int = 0
    """How many `CompactionPart`s the history had as of that request, to tell a compaction since."""
    alerted: bool = False
    """Whether a collapse was alerted on and the cache hasn't re-stabilized since, so a sustained collapse
    alerts once rather than on every request."""


CacheMarks: TypeAlias = dict[CacheKey, CacheMark]


class ConversationCacheMarkStore:
    """Cache marks keyed by `RunContext.conversation_id`, shared by the runs of each conversation.

    The first request of a run that continues a conversation re-sends the prefix the previous run
    cached, so that is where a moved prefix most often shows. Judging it needs the previous run's marks,
    which is why marks outlive the run. Conversation ids are unique, and a history serialized and loaded
    back carries its id along, so keying by them serves every way of running an agent.

    Memory is bounded twice over: conversations are kept in least-recently-updated order, and the oldest
    are forgotten once idle for longer than any provider documents keeping a cache, or once there are
    more than `max_conversations`. Forgetting one only loses the chance to report a collapse the
    provider's cache expiry already explains. A conversation's marks stay bound to the runs using them,
    and go back in the store on their next update.
    """

    max_conversations: ClassVar[int] = 4096
    horizon: ClassVar[timedelta] = timedelta(hours=24)
    """The longest documented prompt-cache retention (OpenAI's extended retention)."""

    def __init__(self) -> None:
        self._conversations: OrderedDict[str, tuple[datetime, CacheMarks]] = OrderedDict()
        # Runs on different threads (each with its own event loop) can share the store.
        self._lock = threading.Lock()

    def get(self, conversation_id: str | None) -> CacheMarks:
        """The conversation's marks; a new, unstored set for a new conversation or a run without one."""
        with self._lock:
            stored = self._conversations.get(conversation_id) if conversation_id is not None else None
        return stored[1] if stored is not None else {}

    def update(self, conversation_id: str | None, marks: CacheMarks, now: datetime) -> CacheMarks:
        """Record that the conversation's marks were updated at `now`, forgetting stale conversations.

        Returns the conversation's marks to keep using. Runs of a new conversation that started
        concurrently each got their own set from `get`; the first to update stores its set, and the
        others merge theirs into it, so no run's marks are lost.
        """
        if conversation_id is None:
            return marks
        with self._lock:
            conversations = self._conversations
            stored = conversations.get(conversation_id)
            if stored is not None and stored[1] is not marks:
                stored[1].update(marks)
                marks = stored[1]
            conversations[conversation_id] = (now, marks)
            conversations.move_to_end(conversation_id)
            while (
                len(conversations) > self.max_conversations
                or now - next(iter(conversations.values()))[0] > self.horizon
            ):
                conversations.popitem(last=False)
        return marks


@dataclass(frozen=True)
class CacheCollapse:
    """A request that read back far less of its cache key's established prefix than expected."""

    reason: CollapseReason
    previous: CacheMark
    """The mark the request was judged against."""
    cache_read_tokens: int
    missed_tokens: int
    """Previously established tokens that were not read back."""
    idle: timedelta
    """Time since `previous` was recorded."""
    retention: timedelta | None
    """The retention window the collapse was classified against, when one is known."""
    alert: bool
    """Whether this collapse should be surfaced: its reason is one the caller alerts on, and the cache
    re-stabilized since the last collapse alerted on."""


@dataclass(frozen=True)
class CacheHealth:
    """How one response used its provider's prompt cache."""

    hit_ratio: float
    """Fraction of input tokens read from the cache."""
    established_tokens: int
    """The established prefix after this response: later requests are judged against it."""
    collapse: CacheCollapse | None


def cache_hit_ratio(cache_read_tokens: int, input_tokens: int) -> float:
    return cache_read_tokens / input_tokens if input_tokens else 0.0


def _count_compactions(request_context: ModelRequestContext, response: ModelResponse) -> int:
    """How many `CompactionPart`s the request's history and its response carry.

    Anthropic and OpenAI's server-side compaction return the part in the response that compacted; OpenAI's
    stateless compaction adds it to the history before the request. Either way the count grows.
    """
    responses = [message for message in request_context.messages if isinstance(message, ModelResponse)]
    return sum(isinstance(part, CompactionPart) for message in [*responses, response] for part in message.parts)


def _sums_cache_usage(response: ModelResponse) -> bool:
    """Whether the response's cache usage may be summed over several internal model calls.

    A native tool such as web search runs the model several times inside one request, and some
    providers report the cache reads of every pass, so the total can be several times the prefix the
    next request will read back. A reported pass count settles it: Anthropic's `message_iterations`
    (compaction passes count too). Without one, a native tool call is treated as summed usage unless
    the provider counts tool-use prompt tokens separately from cache reads, as Google does.
    """
    details = response.usage.details
    passes = details.get('message_iterations')
    if passes is not None:
        return passes + details.get('compaction_iterations', 0) > 1
    return 'tool_use_prompt_tokens' not in details and any(
        isinstance(part, NativeToolCallPart) for part in response.parts
    )


def _cache_retention(request_context: ModelRequestContext) -> timedelta | None:
    """How long the provider is expected to keep this request's cached prefix, or `None` when unknown.

    The retention the request's settings ask for, else the provider's documented default, extended by any
    cache points in the history -- the same boundary `prompt_cache_outlook` predicts with.
    """
    model = request_context.model
    profile: ModelProfile | None
    try:
        profile = model.profile
    except NotImplementedError:
        # `FallbackModel` has no profile of its own: it resolves a model per request and applies that
        # model's profile during dispatch, and the resolved model isn't reachable from here -- the
        # response only carries its provider and model *names*. Without a retention window the
        # collapse is classified `unknown` rather than failing an otherwise successful run.
        profile = None
    return _expected_cache_retention(
        request_context.messages,
        profile=profile,
        retention=model.resolve_cache_retention(request_context.model_settings),
    )


@dataclass
class CacheHealthDetector:
    """Judges one run's responses against its conversation's cache marks, and advances them.

    Bind one per run with the store it shares marks through. The thresholds and `alert_on` are the
    caller's policy; the collapse definition, keying, retention source, and classification are not.
    """

    store: ConversationCacheMarkStore
    conversation_id: str | None
    run_id: str | None
    _: KW_ONLY
    alert_on: AbstractSet[CollapseReason]
    """The collapse reasons worth surfacing. Every collapse is classified and reported either way."""
    min_missed_ratio: float = MIN_MISSED_RATIO
    min_missed_tokens: int = MIN_MISSED_TOKENS
    min_prefix_tokens: int = 0
    """Only judge an established prefix at least this large (kept for harness's deprecated `min_prefix_tokens`)."""
    marks: CacheMarks = field(init=False)

    def __post_init__(self) -> None:
        self.marks = self.store.get(self.conversation_id)

    def observe(
        self,
        request_context: ModelRequestContext,
        response: ModelResponse,
        *,
        final_segment: ModelResponse | None = None,
    ) -> CacheHealth | None:
        """Judge `response` against its cache key's mark and update the mark; `None` when caching isn't in play.

        When `response` merges a continuation chain (Anthropic `pause_turn`, ...), its usage sums every
        segment's request, which no single later request can read back. Pass the chain's `final_segment`
        to judge its cache usage instead: its prompt carries the whole prefix, earlier segments included.
        """
        measured = final_segment or response
        usage = measured.usage
        read = usage.cache_read_tokens
        write = usage.cache_write_tokens
        # Keyed on the response: `FallbackModel` resolves the model inside `request()`, so only the
        # response says which provider, endpoint, and model actually served this request. A switch
        # therefore starts a fresh mark (never a collapse), and switching back is judged against the old one.
        key = (response.provider_name, response.provider_url, response.model_name)
        mark = self.marks.get(key)
        established = mark.established_tokens if mark else 0

        # A response reporting neither reads nor writes never engaged the provider's cache.
        unreported = not read and not write
        if unreported and not established:
            return None

        now = _utils.now_utc()
        compactions = _count_compactions(request_context, response)
        missed = established - read
        collapse: CacheCollapse | None = None
        if (
            mark is not None
            and established >= self.min_prefix_tokens
            and missed >= self.min_missed_tokens
            and missed > established * self.min_missed_ratio
        ):
            collapse = self._classify(request_context, mark, read, now, unreported=unreported, compactions=compactions)

        updated_established = established
        if not unreported and _sums_cache_usage(measured):
            # A summed read can still prove a collapse (every pass read at least what the first did), but
            # a high one proves nothing: it neither raises the mark nor re-arms the alert, and the mark
            # keeps the run that established it. Only the retention clock restarts, since the provider
            # did read the cache.
            if mark is not None:
                alerted = mark.alerted or (collapse is not None and collapse.alert)
                self.marks[key] = replace(mark, last_seen=now, alerted=alerted)
                self.marks = self.store.update(self.conversation_id, self.marks, now)
        elif not unreported:
            if collapse is None:
                # A healthy read-back re-stabilizes the cache, re-arming the alert.
                updated_established, alerted = max(established, read + write), False
            else:
                # After a collapse the mark re-baselines to what this request established, so a
                # deliberate bust (compaction, a rewritten prompt) is judged once rather than against
                # a stale high-water mark on every later request.
                updated_established, alerted = read + write, collapse.previous.alerted or collapse.alert
            self.marks[key] = CacheMark(updated_established, now, self.run_id, compactions, alerted)
            self.marks = self.store.update(self.conversation_id, self.marks, now)
        # An unreported response tells us nothing about the provider's copy of the prefix -- it may
        # still be sitting there, aging toward its TTL -- so the mark, its idle clock, and its alert
        # latch stay put.

        return CacheHealth(
            hit_ratio=cache_hit_ratio(read, usage.input_tokens),
            established_tokens=updated_established,
            collapse=collapse,
        )

    def _classify(
        self,
        request_context: ModelRequestContext,
        mark: CacheMark,
        read: int,
        now: datetime,
        *,
        unreported: bool,
        compactions: int,
    ) -> CacheCollapse:
        idle = now - mark.last_seen
        retention: timedelta | None = None
        reason: CollapseReason
        if unreported:
            # Ambiguous by construction: providers that report cache writes (Anthropic, Bedrock) show
            # `0/0` when the cache wasn't engaged at all -- caching disabled for this request, or a
            # prompt below the minimum cacheable size -- while providers that only report reads
            # (OpenAI's implicit caching) show `0/0` for a full cache miss. The established prefix was
            # re-sent uncached either way, so the miss is real, but the cause isn't knowable from
            # usage alone.
            reason = 'unreported'
        elif compactions > mark.compactions:
            # Provider-native compaction (Anthropic's, OpenAI's) replaces the history before the
            # `CompactionPart` with the compacted summary, so the prefix shrinks by design.
            reason = 'compacted'
        elif (retention := _cache_retention(request_context)) is None:
            reason = 'unknown'
        else:
            reason = 'ttl_expired' if idle > retention else 'unexpected'
        return CacheCollapse(
            reason=reason,
            previous=mark,
            cache_read_tokens=read,
            missed_tokens=mark.established_tokens - read,
            idle=idle,
            retention=retention,
            alert=reason in self.alert_on and not mark.alerted,
        )
