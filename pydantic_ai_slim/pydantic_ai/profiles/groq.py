from __future__ import annotations as _annotations

from dataclasses import dataclass

from . import ModelProfile
from .deepseek import deepseek_model_profile
from .google import google_model_profile
from .meta import meta_model_profile
from .mistral import mistral_model_profile
from .moonshotai import moonshotai_model_profile
from .openai import openai_model_profile
from .qwen import qwen_model_profile


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


def groq_moonshotai_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an MoonshotAI model used with the Groq provider."""
    return ModelProfile(supports_json_object_output=True, supports_json_schema_output=True).update(
        moonshotai_model_profile(model_name)
    )


def meta_groq_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model used with the Groq provider."""
    if model_name in {'llama-4-maverick-17b-128e-instruct', 'llama-4-scout-17b-16e-instruct'}:
        return ModelProfile(supports_json_object_output=True, supports_json_schema_output=True).update(
            meta_model_profile(model_name)
        )
    else:
        return meta_model_profile(model_name)


def groq_provider_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a model routed through Groq provider.

    This function handles model profiling for models that use Groq's API,
    including various model families like Llama, Gemma, Qwen, etc.
    """
    prefix_to_profile = {
        'llama': meta_model_profile,
        'meta-llama/': meta_groq_model_profile,
        'gemma': google_model_profile,
        'qwen': qwen_model_profile,
        'deepseek': deepseek_model_profile,
        'mistral': mistral_model_profile,
        'moonshotai/': groq_moonshotai_model_profile,
        'compound-': groq_model_profile,
        'openai/': openai_model_profile,
    }

    for prefix, profile_func in prefix_to_profile.items():
        model_name = model_name.lower()
        if model_name.startswith(prefix):
            if prefix.endswith('/'):
                model_name = model_name[len(prefix) :]
            return profile_func(model_name)

    return None
