from __future__ import annotations as _annotations

from . import ModelProfile


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

    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)
    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
    )
