from __future__ import annotations as _annotations

from . import ModelProfile


def perplexity_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Perplexity model.

    Perplexity's chat models perform web search natively, so [`WebSearchTool`][pydantic_ai.builtin_tools.WebSearchTool]
    is enabled at the provider level rather than per-model. Reasoning models in the Sonar family stream a
    leading newline that needs to be ignored, matching DeepSeek R1 behaviour.
    """
    is_reasoning = 'reasoning' in model_name
    return ModelProfile(
        ignore_streamed_leading_whitespace=is_reasoning,
        supports_thinking=is_reasoning,
    )
