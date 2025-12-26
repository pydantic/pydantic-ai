from __future__ import annotations as _annotations

from dataclasses import dataclass

from . import ModelProfile


@dataclass(kw_only=True)
class GroqModelProfile(ModelProfile):
    """Profile for models used with GroqModel.

    ALL FIELDS MUST BE `groq_` PREFIXED SO YOU CAN MERGE THEM WITH OTHER MODELS.
    """

    groq_always_has_web_search_builtin_tool: bool = False
    """Whether the model always has the web search built-in tool available."""


def groq_model_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a Groq model."""
    return GroqModelProfile(
        groq_always_has_web_search_builtin_tool=model_name.startswith('compound-'),
    )


def groq_gpt_oss_model_profile(model_name: str) -> ModelProfile:
    """Get profile for OpenAI GPT-OSS models on Groq.

    GPT-OSS models (gpt-oss-20b, gpt-oss-120b) support strict native structured output
    with 100% schema adherence.
    """
    from .openai import OpenAIJsonSchemaTransformer

    return GroqModelProfile(
        supports_json_schema_output=True,
        supports_json_object_output=True,
        default_structured_output_mode='native',
        json_schema_transformer=OpenAIJsonSchemaTransformer,
    )
