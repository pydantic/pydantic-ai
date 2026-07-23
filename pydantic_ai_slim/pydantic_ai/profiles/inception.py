from __future__ import annotations as _annotations

from . import ModelProfile


def inception_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Inception model."""
    # Mercury 2 supports schema-aligned JSON output via OpenAI-style `response_format` with `json_schema`,
    # and thinking tunable via `reasoning_effort`, including the `'none'` value that the unified `thinking`
    # setting maps `False` to; the legacy mercury and mercury-coder models don't. See https://docs.inceptionlabs.ai/.
    is_mercury_2 = model_name.startswith('mercury-2')
    return ModelProfile(supports_json_schema_output=is_mercury_2, supports_thinking=is_mercury_2)
