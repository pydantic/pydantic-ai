from __future__ import annotations as _annotations

from . import ModelProfile


def perplexity_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Perplexity model.

    Reasoning models in the Sonar family (`sonar-reasoning*` and `sonar-deep-research`) emit thinking and
    stream a leading newline that needs to be ignored, matching DeepSeek R1 behaviour.
    """
    is_reasoning = 'reasoning' in model_name or 'deep-research' in model_name
    return ModelProfile(
        ignore_streamed_leading_whitespace=is_reasoning,
        supports_thinking=is_reasoning,
    )
