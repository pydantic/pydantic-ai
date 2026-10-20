"""KeepCacheWarm: refresh a conversation's prompt cache before the provider expires it."""

from __future__ import annotations

import asyncio
import warnings
from dataclasses import KW_ONLY, dataclass, field
from datetime import timedelta
from typing import Any

from opentelemetry import context as otel_context
from opentelemetry.trace import StatusCode, Tracer

from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.run import AgentRunResult
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT, RunContext
from pydantic_ai.usage import RunUsage

_MIN_READ_BACK_RATIO = 0.95
"""A refresh must read back at least this fraction of the prefix the real request cached.

An exact replay reads the whole prefix back. Reading back less means the replay wrote part of the prefix
again, at the cache-write price, so further refreshes would keep paying for writes instead of reads.
"""

_MAX_REFRESH_OUTPUT_TOKENS = 16
"""A refresh producing more output tokens than this means the provider ignored `max_tokens=1`.

An honoured cap yields at most 1 token; 16 leaves slack for providers that round small caps up.
"""


class CacheWarmingStoppedWarning(UserWarning):
    """Warned when `KeepCacheWarm` stops warming a conversation because a refresh request didn't work as a cheap read.

    That's a refresh request that raised, such as with Anthropic extended thinking, whose `budget_tokens`
    `max_tokens` must exceed; one that read back less of the cached prefix than the real request cached, so it
    paid to write the cache again; or one that produced more output than `max_tokens=1` allows, so the provider
    ignored the cap. The conversation isn't warmed again until its next model request.
    """


@dataclass
class _Refresh:
    """The request a refresh replays, and when the conversation last sent a real one."""

    model: Model
    messages: list[ModelMessage]
    model_settings: ModelSettings
    model_request_parameters: ModelRequestParameters
    retention: float
    """How long the provider keeps the cached prefix, in seconds."""
    cached_tokens: int
    """The prefix the real request cached: its cache reads plus writes."""
    requested_at: float
    """When the real request started, on the event loop's clock."""


@dataclass
class KeepCacheWarm(AbstractCapability[AgentDepsT]):
    """Keep a conversation's prompt cache from expiring between model requests, by replaying the last request before it does.

    After each model request, a background task waits until shortly before the provider's cache retention
    window ends, then sends the same request again with `max_tokens=1` and discards the response. Reading the
    cached prefix back restarts the provider's retention timer, so the next real request reads the cache
    instead of paying to write it again. The conversation's next model request cancels the pending refresh.

    By default the cache is kept warm only while a run is in progress, such as during a long tool call, where
    the next model request is certain to come. With `between_runs=True` it's also kept warm after a run ends,
    for the conversation's next turn.

    Warming applies to models whose cache expires a fixed time after its last use, which each read restarts:
    a profile with a `'5m'` cache tier, as on Anthropic, Claude on Amazon Bedrock, and OpenRouter's Anthropic
    routes. The retention is the one the request's settings ask for, else the provider's
    `default_cache_retention`. It stops `max_idle` after the conversation's last real request.

    The refresh runs on the event loop the run ran on, so it needs that loop to keep running, as in an async
    application; between `run_sync()` calls nothing is refreshed. It isn't for durable execution, where model
    requests run in a separate activity or step.
    """

    _: KW_ONLY

    id: str | None = 'keep_cache_warm'
    """One-off: two on one agent would each replay the same request, doubling the refresh cost.

    Two of them resolve to one via `combine`, which keeps the last. Pass a distinct `id` to keep both.
    """

    between_runs: bool = False
    """Whether to keep the cache warm after a run ends, until the conversation's next run or `max_idle`.

    Off by default: whether the next turn comes before `max_idle` is unknown, and each refresh costs a cache
    read of the whole prefix. At Anthropic's 0.1x read and 1.25x write prices, the refreshes of a 30-minute
    `max_idle` cost about half of the one rewrite they avoid.
    """

    max_idle: timedelta = timedelta(minutes=30)
    """How long after a conversation's last real model request to keep its cache warm.

    A refresh is only sent while the cache would otherwise expire before then, so a retention window at least
    this long (such as Anthropic's `'1h'` against the default) is never refreshed.
    """

    margin: timedelta = timedelta(seconds=30)
    """How long before the cache would expire to send the refresh request.

    A refresh that can't start within half of this margin of its due time is dropped and warming stops, since
    the cache may already have expired and the refresh would pay to write it again.
    """

    _warmers: dict[str, asyncio.Task[None]] = field(
        init=False, default_factory=dict[str, 'asyncio.Task[None]'], compare=False, repr=False
    )
    _started: dict[str, float] = field(init=False, default_factory=dict[str, float], compare=False, repr=False)
    _usage: RunUsage = field(init=False, default_factory=RunUsage, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_idle <= timedelta(0):
            raise ValueError('max_idle must be positive')
        if self.margin <= timedelta(0):
            raise ValueError('margin must be positive')

    @property
    def usage(self) -> RunUsage:
        """The combined usage of the refresh requests sent so far, which no run's usage includes."""
        return self._usage

    async def stop(self) -> None:
        """Cancel every pending refresh and wait for it to finish."""
        warmers = list(self._warmers.values())
        self._warmers.clear()
        for warmer in warmers:
            warmer.cancel()
        await asyncio.gather(*warmers, return_exceptions=True)

    async def wrap_run(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        """Stop warming when the run ends, unless `between_runs` is set."""
        try:
            return await handler()
        finally:
            if not self.between_runs and (warmer := self._warmers.pop(_conversation_key(ctx), None)) is not None:
                warmer.cancel()

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        """Cancel the conversation's pending refresh: this request uses the cache itself."""
        key = _conversation_key(ctx)
        if (warmer := self._warmers.pop(key, None)) is not None:
            warmer.cancel()
        # The provider times its retention from the start of the request that used the cache, not its end.
        self._started[key] = asyncio.get_running_loop().time()
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        """Start keeping warm the cache this request used, if the model's cache retention is known."""
        key = _conversation_key(ctx)
        started = self._started.pop(key, asyncio.get_running_loop().time())
        retention = _retention(request_context)
        usage = response.usage
        if (
            retention is None
            or not self.margin < retention < self.max_idle
            or not (usage.cache_read_tokens or usage.cache_write_tokens)
        ):
            return response
        refresh = _Refresh(
            model=request_context.model,
            messages=list(request_context.messages),
            model_settings={**(request_context.model_settings or {}), 'max_tokens': 1},
            model_request_parameters=request_context.model_request_parameters,
            retention=retention.total_seconds(),
            cached_tokens=usage.cache_read_tokens + usage.cache_write_tokens,
            requested_at=started,
        )
        if (previous := self._warmers.pop(key, None)) is not None:
            previous.cancel()  # pragma: no cover -- only when runs of one conversation overlap
        warmer = asyncio.create_task(self._warm(refresh, ctx.tracer), name=f'keep cache warm ({key})')
        self._warmers[key] = warmer
        warmer.add_done_callback(lambda task: self._forget(key, task))
        return response

    async def on_model_request_error(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        """Forget the failed request's start time: there's no cache use to keep warm."""
        self._started.pop(_conversation_key(ctx), None)
        raise error

    def _forget(self, key: str, task: asyncio.Task[None]) -> None:
        if self._warmers.get(key) is task:
            del self._warmers[key]

    async def _warm(self, refresh: _Refresh, tracer: Tracer) -> None:
        margin = self.margin.total_seconds()
        idle_until = refresh.requested_at + self.max_idle.total_seconds()
        last_use = refresh.requested_at
        while last_use + refresh.retention < idle_until:
            due = last_use + refresh.retention - margin
            await asyncio.sleep(max(due - asyncio.get_running_loop().time(), 0))
            started = asyncio.get_running_loop().time()
            if started > due + margin / 2:
                # The loop didn't run on time (blocked, or the process was suspended): the cache may be gone.
                return
            # The refresh outlives the run that started it, so its span is a root span, not the run's child.
            with tracer.start_as_current_span('keep_cache_warm', context=otel_context.Context()) as span:
                try:
                    response = await refresh.model.request(
                        refresh.messages, refresh.model_settings, refresh.model_request_parameters
                    )
                except Exception as e:
                    span.record_exception(e)
                    span.set_status(StatusCode.ERROR)
                    warnings.warn(
                        f'Stopped keeping the prompt cache warm: the refresh request failed with {e!r}.',
                        CacheWarmingStoppedWarning,
                        stacklevel=1,
                    )
                    return
                self._usage.incr(response.usage)
                self._usage.requests += 1
                if span.is_recording():
                    span.set_attributes(
                        {
                            'keep_cache_warm.cache_read_tokens': response.usage.cache_read_tokens,
                            'keep_cache_warm.cache_write_tokens': response.usage.cache_write_tokens,
                            'keep_cache_warm.retention_seconds': refresh.retention,
                        }
                    )
            if (stopped := _stop_reason(response, refresh)) is not None:
                warnings.warn(
                    f'Stopped keeping the prompt cache warm: {stopped}.', CacheWarmingStoppedWarning, stacklevel=1
                )
                return
            last_use = started


def _conversation_key(ctx: RunContext[AgentDepsT]) -> str:
    return ctx.conversation_id or ctx.run_id or ''


def _retention(request_context: ModelRequestContext) -> timedelta | None:
    """How long the provider keeps this request's cached prefix, or `None` when it isn't known."""
    model = request_context.model
    try:
        profile = model.profile
    except NotImplementedError:
        # `FallbackModel` resolves its model per request, so which provider's cache a replay would hit isn't known.
        return None
    # A provider whose cache lives *at least* some time, like OpenAI's, may keep it longer anyway, so refreshing
    # it buys nothing certain. A `'5m'` tier marks a cache that expires a fixed time after its last use.
    if not profile.get('supports_cache', False) or '5m' not in profile.get('supported_cache_retentions', ()):
        return None
    return model.resolve_cache_retention(request_context.model_settings) or profile.get('default_cache_retention')


def _stop_reason(response: ModelResponse, refresh: _Refresh) -> str | None:
    usage = response.usage
    if usage.output_tokens > _MAX_REFRESH_OUTPUT_TOKENS:
        return (
            f'the refresh request produced {usage.output_tokens} output tokens, so the provider ignored `max_tokens=1`'
        )
    if usage.cache_read_tokens < refresh.cached_tokens * _MIN_READ_BACK_RATIO:
        return (
            f'the refresh request read back {usage.cache_read_tokens} of the {refresh.cached_tokens} cached tokens, '
            'so it wrote the cache again instead of refreshing it'
        )
    return None
