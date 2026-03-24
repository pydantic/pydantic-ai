from __future__ import annotations as _annotations

from dataclasses import dataclass

from . import ModelProfile


@dataclass(kw_only=True)
class AnthropicModelProfile(ModelProfile):
    """Profile for models used with `AnthropicModel`.

    ALL FIELDS MUST BE `anthropic_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    anthropic_supports_adaptive_thinking: bool = False
    """Whether the model supports adaptive thinking (Opus 4.6+).

    When True, unified `thinking` translates to `{'type': 'adaptive'}`.
    When False, it translates to `{'type': 'enabled', 'budget_tokens': N}`.
    """


ANTHROPIC_THINKING_BUDGET_MAP: dict[bool | str, int] = {True: 10000, 'low': 2048, 'medium': 10000, 'high': 16384}
"""Maps unified thinking values to Anthropic budget_tokens for non-adaptive models."""


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
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

    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)

    # Opus 4.6+ supports adaptive thinking; older models use budget-based
    supports_adaptive = model_name.startswith('claude-opus-4-6')

    return AnthropicModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        supports_thinking=True,
        anthropic_supports_adaptive_thinking=supports_adaptive,
    )
