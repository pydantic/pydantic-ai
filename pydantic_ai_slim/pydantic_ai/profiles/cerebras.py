from __future__ import annotations as _annotations

from . import ModelProfile
from .harmony import harmony_model_profile
from .meta import meta_model_profile
from .openai import OpenAIJsonSchemaTransformer, OpenAIModelProfile
from .qwen import qwen_model_profile


def cerebras_provider_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a model routed through Cerebras provider.

    This function handles model profiling for models that use Cerebras's API,
    and applies Cerebras-specific settings like unsupported model parameters.
    """
    prefix_to_profile = {'llama': meta_model_profile, 'qwen': qwen_model_profile, 'gpt-oss': harmony_model_profile}

    profile = None
    for prefix, profile_func in prefix_to_profile.items():
        model_name = model_name.lower()
        if model_name.startswith(prefix):
            profile = profile_func(model_name)

    # According to https://inference-docs.cerebras.ai/resources/openai#currently-unsupported-openai-features,
    # Cerebras doesn't support some model settings.
    unsupported_model_settings = (
        'frequency_penalty',
        'logit_bias',
        'presence_penalty',
        'parallel_tool_calls',
        'service_tier',
    )
    return OpenAIModelProfile(
        json_schema_transformer=OpenAIJsonSchemaTransformer,
        openai_unsupported_model_settings=unsupported_model_settings,
    ).update(profile)
