from __future__ import annotations as _annotations

from dataclasses import dataclass

from . import ModelProfile

# Effort-to-budget mapping for Anthropic-style models (Claude on Anthropic and Bedrock).
# Budget-based thinking uses a token budget instead of effort levels.
EFFORT_TO_BUDGET: dict[str, int] = {
    'low': 1024,
    'medium': 4096,
    'high': 16384,
}
DEFAULT_THINKING_BUDGET = 4096


@dataclass(kw_only=True)
class AnthropicModelProfile(ModelProfile):
    """Profile for models used with AnthropicModel.

    ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    anthropic_supports_adaptive_thinking: bool = False
    """Whether the model supports adaptive thinking (``type: 'adaptive'``) instead of budget-based thinking."""


# Models that support extended thinking
_THINKING_MODELS = (
    'claude-3-7',
    'claude-sonnet-4',
    'claude-opus-4',
    'claude-haiku-4-5',
)

# Models that support adaptive thinking (type: "adaptive" with output_config.effort).
# Only Opus 4.6 and Sonnet 4.6 support this; older models use type: "enabled" with budget_tokens.
_ADAPTIVE_THINKING_MODELS = (
    'claude-opus-4-6',
    'claude-sonnet-4-6',
)


def anthropic_model_profile(model_name: str) -> AnthropicModelProfile | None:
    """Get the model profile for an Anthropic model."""
    models_that_support_json_schema_output = (
        'claude-haiku-4-5',
        'claude-sonnet-4-5',
        'claude-sonnet-4-6',
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

    # Check if model supports adaptive thinking
    supports_adaptive = any(name in model_name for name in _ADAPTIVE_THINKING_MODELS)

    return AnthropicModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        supports_thinking=supports_thinking,
        anthropic_supports_adaptive_thinking=supports_adaptive,
    )
