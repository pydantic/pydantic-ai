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
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias

from . import _utils
from .profiles import ModelProfile, _expected_cache_retention  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from .messages import ModelResponse
    from .models import ModelRequestContext

CollapseReason: TypeAlias = Literal['unexpected', 'ttl-expired', 'unknown', 'unreported']
"""Why a cached prefix collapsed, as far as the provider's usage and retention window can tell.

- `'unexpected'`: the retention window should still have been active, so the prefix moved.
- `'ttl-expired'`: the gap since the last request for the same cache exceeded the retention window.
- `'unknown'`: the provider publishes no retention window, so the collapse can't be attributed.
- `'unreported'`: the response reported no cache usage at all, so the cause can't be determined.
"""

COLLAPSE_RATIO = 0.5
"""A request reading back less than this fraction of the established prefix counts as a collapse."""
MIN_PREFIX_TOKENS = 1024
"""Only judge collapse once the established prefix reaches this size (Anthropic's minimum cacheable prefix)."""

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

    def update(self, conversation_id: str | None, marks: CacheMarks, now: datetime) -> None:
        """Record that the conversation's marks were updated at `now`, forgetting stale conversations."""
        if conversation_id is None:
            return
        with self._lock:
            conversations = self._conversations
            conversations[conversation_id] = (now, marks)
            conversations.move_to_end(conversation_id)
            while (
                len(conversations) > self.max_conversations
                or now - next(iter(conversations.values()))[0] > self.horizon
            ):
                conversations.popitem(last=False)


@dataclass(frozen=True)
class CacheCollapse:
    """A request that read back far less of its cache key's established prefix than expected."""

    reason: CollapseReason
    previous: CacheMark
    """The mark the request was judged against."""
    cache_read_tokens: int
    wasted_tokens: int
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
    collapse_ratio: float = COLLAPSE_RATIO
    min_prefix_tokens: int = MIN_PREFIX_TOKENS
    marks: CacheMarks = field(init=False)

    def __post_init__(self) -> None:
        self.marks = self.store.get(self.conversation_id)

    def observe(self, request_context: ModelRequestContext, response: ModelResponse) -> CacheHealth | None:
        """Judge `response` against its cache key's mark and update the mark; `None` when caching isn't in play."""
        usage = response.usage
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
        collapse: CacheCollapse | None = None
        if mark is not None and established >= self.min_prefix_tokens and read < established * self.collapse_ratio:
            collapse = self._classify(request_context, mark, read, now, unreported=unreported)

        updated_established = established
        if not unreported:
            if collapse is None:
                # A healthy read-back re-stabilizes the cache, re-arming the alert.
                updated_established, alerted = max(established, read + write), False
            else:
                # After a collapse the mark re-baselines to what this request established, so a
                # deliberate bust (compaction, a rewritten prompt) is judged once rather than against
                # a stale high-water mark on every later request.
                updated_established, alerted = read + write, collapse.previous.alerted or collapse.alert
            self.marks[key] = CacheMark(updated_established, now, self.run_id, alerted)
            self.store.update(self.conversation_id, self.marks, now)
        # An unreported response tells us nothing about the provider's copy of the prefix -- it may
        # still be sitting there, aging toward its TTL -- so the mark, its idle clock, and its alert
        # latch stay put.

        return CacheHealth(
            hit_ratio=cache_hit_ratio(read, usage.input_tokens),
            established_tokens=updated_established,
            collapse=collapse,
        )

    def _classify(
        self, request_context: ModelRequestContext, mark: CacheMark, read: int, now: datetime, *, unreported: bool
    ) -> CacheCollapse:
        idle = now - mark.last_seen
        retention: timedelta | None = None
        reason: CollapseReason
        if unreported:
            # Ambiguous by construction: providers that report cache writes (Anthropic, Bedrock) show
            # `0/0` when the cache wasn't engaged at all -- caching disabled for this request, or a
            # prompt below the minimum cacheable size -- while providers that only report reads
            # (OpenAI's implicit caching) show `0/0` for a full cache miss. The established prefix was
            # re-sent uncached either way, so the waste is real, but the cause isn't knowable from
            # usage alone.
            reason = 'unreported'
        elif (retention := _cache_retention(request_context)) is None:
            reason = 'unknown'
        else:
            reason = 'ttl-expired' if idle > retention else 'unexpected'
        return CacheCollapse(
            reason=reason,
            previous=mark,
            cache_read_tokens=read,
            wasted_tokens=mark.established_tokens - read,
            idle=idle,
            retention=retention,
            alert=reason in self.alert_on and not mark.alerted,
        )
