from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Any

from pydantic_ai.settings import CacheConfig, CacheRetention, ModelSettings

from .abstract import AbstractCapability


@dataclass
class Caching(AbstractCapability[Any]):
    """Enables and configures prompt caching.

    Uses the unified `cache` setting in
    [`ModelSettings`][pydantic_ai.settings.ModelSettings] to work portably across providers.
    On models that cache prompts implicitly, it has no effect.
    Provider-specific cache settings (e.g., `anthropic_cache`, `bedrock_cache_instructions`) take
    precedence when both are set.
    """

    retention: bool | CacheRetention = True
    """The prompt-cache configuration.

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

    def get_model_settings(self) -> ModelSettings | None:
        if self.messages or self.retention is False:
            return ModelSettings(cache=self.retention)
        if self.retention is True:
            return ModelSettings(cache=CacheConfig(messages=False))
        return ModelSettings(cache=CacheConfig(retention=self.retention, messages=False))
