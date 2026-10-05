from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Any

from pydantic_ai.settings import CacheSetting, ModelSettings

from .abstract import AbstractCapability


@dataclass
class Caching(AbstractCapability[Any]):
    """Configures prompt caching.

    Uses the unified `cache` setting in
    [`ModelSettings`][pydantic_ai.settings.ModelSettings] to work portably across providers.
    Prompt caching is already on by default on the models that need it to be configured, so this
    capability is mostly useful to request a longer retention, or to turn library-managed caching off.
    Provider-specific cache settings (e.g., `anthropic_cache`, `bedrock_cache_instructions`) take
    precedence when both are set.
    """

    retention: CacheSetting = True
    """The prompt-cache configuration.

    - `True`: Enable prompt caching with the provider's default retention.
    - `False`: Disable library-managed prompt caching.
    - `'5m'`/`'30m'`/`'1h'`: Enable prompt caching with a specific retention, snapped to the nearest
      tier the provider supports.
    """

    _: KW_ONLY

    id: str | None = 'caching'
    """One-off: an agent has a single caching configuration, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine],
    which keeps the last. Pass a distinct `id` to keep both, or `id=None` for derived ids.
    """

    def get_model_settings(self) -> ModelSettings | None:
        return ModelSettings(cache=self.retention)
