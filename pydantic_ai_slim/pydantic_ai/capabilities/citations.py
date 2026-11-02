from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Any

from pydantic_ai.settings import ModelSettings

from .abstract import AbstractCapability


@dataclass
class Citations(AbstractCapability[Any]):
    """Asks for citations from providers that only cite when asked.

    Sets the [`citations`][pydantic_ai.settings.ModelSettings.citations] model setting. Providers that always cite,
    such as OpenAI and Google for web search, don't need it. Anthropic returns web fetch results without citations
    unless it's set. On models with the `20260209` web tools, it also has the model call web search and web fetch
    directly, so their results are cited, unless the request needs code execution; see
    [Web Tool Callers](../models/anthropic.md#web-tool-callers). The setting is only used when the output is text.
    """

    _: KW_ONLY

    id: str | None = 'citations'
    """One-off: an agent either asks for citations or doesn't, so the id is fixed by default.

    Two of them resolve to one via [`combine`][pydantic_ai.capabilities.AbstractCapability.combine],
    which keeps the last. Pass a distinct `id` to keep both, or `id=None` for derived ids.
    """

    def get_model_settings(self) -> ModelSettings | None:
        return ModelSettings(citations=True)
