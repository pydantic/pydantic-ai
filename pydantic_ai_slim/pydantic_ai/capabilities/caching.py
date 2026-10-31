from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from datetime import timedelta
from typing import Any, Literal, TypeAlias, cast

from pydantic_ai._run_context import RunContext
from pydantic_ai.messages import idle_gaps
from pydantic_ai.models import Model
from pydantic_ai.settings import CacheConfig, CacheRetention, ModelSettings

from .abstract import AbstractCapability

CachePolicyName: TypeAlias = Literal['auto', 'sticky']
"""A named [`CachePolicy`][pydantic_ai.capabilities.CachePolicy] for [`Caching.retention`][pydantic_ai.capabilities.Caching.retention].

- `'sticky'`: [`StickyCachePolicy()`][pydantic_ai.capabilities.StickyCachePolicy]. Stable between releases.
- `'auto'`: [`AutoCachePolicy()`][pydantic_ai.capabilities.AutoCachePolicy], the policy we currently measure as cheapest.
  It may change between releases as new measurements come in.
"""


class CachePolicy(ABC):
    """Chooses the cache retention for each model request.

    [`Caching`][pydantic_ai.capabilities.Caching] calls [`retention`][pydantic_ai.capabilities.CachePolicy.retention]
    before every request, with the run's message history in `ctx.messages`, and sends the result as the unified
    [`cache`][pydantic_ai.settings.ModelSettings.cache] setting. The model snaps a retention to the nearest tier its
    provider supports, so a policy can ask for `'1h'` without knowing whether the provider offers it.
    The built-in policies are implemented on this interface, and a policy of your own plugs in the same way.
    """

    @abstractmethod
    def retention(self, ctx: RunContext[Any]) -> bool | CacheRetention:
        """The retention for the request about to be sent: `True` for the provider's default, or a tier."""
        raise NotImplementedError


@dataclass
class StickyCachePolicy(CachePolicy):
    """A short retention until the conversation first idles, then a long one for the rest of it.

    A conversation driven by tool calls sends its requests seconds apart, so a short-lived cache is enough and its
    writes are cheaper. Once a person (or a slow background job) has paced it, with a gap longer than `idle`
    between a response and the next request, gaps like that tend to recur, and a long-lived cache saves the full
    re-write each one would cost.
    """

    idle: timedelta = timedelta(minutes=5)
    """The gap, from a response to the next request, after which the conversation is treated as human-paced."""

    short: CacheRetention = '5m'
    """The retention before the first such gap."""

    long: CacheRetention = '1h'
    """The retention from the first such gap on."""

    def retention(self, ctx: RunContext[Any]) -> bool | CacheRetention:
        return self.long if any(gap > self.idle for gap in idle_gaps(ctx.messages)) else self.short


@dataclass
class AutoCachePolicy(CachePolicy):
    """The retention policy we currently measure as cheapest. It may change between releases.

    Today: [`StickyCachePolicy`][pydantic_ai.capabilities.StickyCachePolicy] on models whose profile offers both a
    short and a long tier (`'5m'` and `'1h'`), and the provider's default retention everywhere else.
    """

    def retention(self, ctx: RunContext[Any]) -> bool | CacheRetention:
        # The `cast` only pins `Model`'s client type parameter, which `isinstance` can't recover.
        ctx_model = ctx.model
        model = cast('Model[Any]', ctx_model) if isinstance(ctx_model, Model) else None
        tiers = model.profile.get('supported_cache_retentions', ()) if model is not None else ()
        if '5m' in tiers and '1h' in tiers:
            return StickyCachePolicy().retention(ctx)
        return True


@dataclass
class Caching(AbstractCapability[Any]):
    """Enables and configures prompt caching.

    Uses the unified `cache` setting in
    [`ModelSettings`][pydantic_ai.settings.ModelSettings] to work portably across providers.
    On models that cache prompts implicitly, it has no effect.
    Provider-specific cache settings (e.g., `anthropic_cache`, `bedrock_cache_instructions`) take
    precedence when both are set.
    """

    retention: bool | CacheRetention | CachePolicyName | CachePolicy = 'auto'
    """The prompt-cache configuration.

    - `'auto'` (the default): the retention policy we currently measure as cheapest,
      [`AutoCachePolicy`][pydantic_ai.capabilities.AutoCachePolicy]. It may change between releases.
    - `'sticky'`: [`StickyCachePolicy`][pydantic_ai.capabilities.StickyCachePolicy], 5 minutes until the
      conversation first idles for longer than that, then 1 hour.
    - A [`CachePolicy`][pydantic_ai.capabilities.CachePolicy] instance: your own policy.
    - `True`: Enable prompt caching with the provider's default retention.
    - `False`: Disable library-managed prompt caching.
    - `'5m'`/`'30m'`/`'1h'`: Enable prompt caching with a specific retention, snapped to the nearest
      tier the provider supports.
    """

    _: KW_ONLY

    messages: bool = True
    """Whether to also cache the growing conversation, not only the stable prompt prefix.

    With `False`, only the tool definitions and static instructions are cached: useful for many short,
    one-off conversations that share long instructions or tools. See
    [`CacheConfig.messages`][pydantic_ai.settings.CacheConfig.messages].
    """

    id: str | None = 'caching'
    """One-off: an agent has a single caching configuration, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine],
    which keeps the last. Pass a distinct `id` to keep both, or `id=None` for derived ids.
    """

    def get_model_settings(self) -> ModelSettings | Callable[[RunContext[Any]], ModelSettings]:
        retention = self.retention
        if retention == 'auto':
            retention = AutoCachePolicy()
        elif retention == 'sticky':
            retention = StickyCachePolicy()
        if isinstance(retention, CachePolicy):
            policy = retention
            return lambda ctx: self._settings(policy.retention(ctx))
        return self._settings(retention)

    def _settings(self, retention: bool | CacheRetention) -> ModelSettings:
        if self.messages or retention is False:
            return ModelSettings(cache=retention)
        if retention is True:
            return ModelSettings(cache=CacheConfig(messages=False))
        return ModelSettings(cache=CacheConfig(retention=retention, messages=False))
