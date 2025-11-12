from __future__ import annotations as _annotations

from dataclasses import dataclass

from . import ModelProfile

@dataclass(kw_only=True)
class GrokModelProfile(ModelProfile):
    """Profile for models used with GroqModel.

    ALL FIELDS MUST BE `groq_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    builtin_tool: bool = False
    """Whether the model always has the web search built-in tool available."""



def grok_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Grok model."""
    return GrokModelProfile(
        # Support tool calling for building tools
        builtin_tool=model_name.startswith('grok-4'),
    )
