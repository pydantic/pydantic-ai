from __future__ import annotations as _annotations

from . import ModelProfile

# Models that support extended thinking
_THINKING_MODELS = (
    'claude-3-7',
    'claude-sonnet-4',
    'claude-opus-4',
    'claude-haiku-4-5',
)


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    models_that_support_json_schema_output = (
        'claude-haiku-4-5',
        'claude-sonnet-4-5',
        'claude-opus-4-1',
        'claude-opus-4-5',
        'claude-opus-4-6',
    )
    """These models support both structured outputs and strict tool calling."""
    # TODO update when new models are released that support structured outputs
    # https://docs.claude.com/en/docs/build-with-claude/structured-outputs#example-usage

    supports_json_schema_output = any(name in model_name for name in models_that_support_json_schema_output)

    # Check if model supports extended thinking
    supports_thinking = any(name in model_name for name in _THINKING_MODELS)

    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        supports_thinking=supports_thinking,
    )
