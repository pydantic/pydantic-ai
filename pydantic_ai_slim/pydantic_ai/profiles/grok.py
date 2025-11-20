from __future__ import annotations as _annotations

from dataclasses import dataclass

from . import ModelProfile


@dataclass(kw_only=True)
class GrokModelProfile(ModelProfile):
    """Profile for Grok models (used with both GrokProvider and XaiProvider).

    ALL FIELDS MUST BE `grok_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    grok_supports_builtin_tools: bool = False
    """Whether the model always has the web search built-in tool available."""


def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    return GrokModelProfile(
        # Support tool calling for building tools
        grok_supports_builtin_tools=model_name.startswith('grok-4'),
    )
