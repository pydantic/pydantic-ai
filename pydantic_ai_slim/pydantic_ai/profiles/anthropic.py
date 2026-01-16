from __future__ import annotations as _annotations

from . import ModelProfile

# Anthropic effort-to-budget mapping aligns with TOKEN_HISTOGRAM_BOUNDARIES
ANTHROPIC_EFFORT_TO_BUDGET: dict[str, int] = {
    'low': 1024,
    'medium': 4096,
    'high': 16384,
}

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
    )
    """These models support both structured outputs and strict tool calling."""
    # TODO update when new models are released that support structured outputs
    # https://docs.claude.com/en/docs/build-with-claude/structured-outputs#example-usage

    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)

    # Check if model supports extended thinking
    supports_thinking = model_name.startswith(_THINKING_MODELS)

    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        supports_thinking=supports_thinking,
        default_thinking_budget=4096 if supports_thinking else None,
        effort_to_budget_map=ANTHROPIC_EFFORT_TO_BUDGET if supports_thinking else None,
    )
