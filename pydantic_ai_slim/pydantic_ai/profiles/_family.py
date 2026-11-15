"""The model family named by the vendor or organization in a `vendor/model` id, shared by the gateway providers."""

from __future__ import annotations as _annotations

from collections.abc import Callable, Mapping

from . import ModelProfile
from .amazon import amazon_model_profile
from .anthropic import anthropic_model_profile
from .cohere import cohere_model_profile
from .deepseek import deepseek_model_profile
from .google import google_model_profile
from .grok import grok_model_profile
from .harmony import harmony_model_profile
from .meta import meta_model_profile
from .mistral import mistral_model_profile
from .moonshotai import moonshotai_model_profile
from .openai import openai_model_profile
from .qwen import qwen_model_profile
from .zai import _glm_5_3_schema_profile  # pyright: ignore[reportPrivateUsage]

ModelProfileFunc = Callable[[str], ModelProfile | None]


def _openai_family_profile(model_name: str) -> ModelProfile | None:
    """gpt-oss uses the Harmony format, unlike the models OpenAI serves itself."""
    if model_name.startswith('gpt-oss'):
        return harmony_model_profile(model_name)
    return openai_model_profile(model_name)


_VENDOR_FAMILY_PROFILES: dict[str, ModelProfileFunc] = {
    'alibaba': qwen_model_profile,
    'amazon': amazon_model_profile,
    'anthropic': anthropic_model_profile,
    'cohere': cohere_model_profile,
    'deepseek': deepseek_model_profile,
    'deepseek-ai': deepseek_model_profile,
    'google': google_model_profile,
    # Not `muse_model_profile` for Muse ids: it describes Meta's own API (reasoning efforts, web search, phase),
    # which gateways may not accept.
    'meta': meta_model_profile,
    'meta-llama': meta_model_profile,
    'meta-models': meta_model_profile,
    'mistral': mistral_model_profile,
    'mistralai': mistral_model_profile,
    'moonshotai': moonshotai_model_profile,
    'openai': _openai_family_profile,
    'qwen': qwen_model_profile,
    'x-ai': grok_model_profile,
    'xai': grok_model_profile,
    # Gateways only get the GLM-5.3 schema fix, not Z.AI's native thinking settings, which they may not accept.
    'z-ai': _glm_5_3_schema_profile,
    'zai': _glm_5_3_schema_profile,
    'zai-org': _glm_5_3_schema_profile,
}
"""Lowercase vendor and organization names, as Hugging Face repo ids and gateway slugs spell them, to the family
profile for the model after the `/`. Only families with a profile function have an entry."""


def vendor_family_model_profile(
    model_name: str, *, overrides: Mapping[str, ModelProfileFunc] | None = None
) -> ModelProfile | None:
    """The family profile for a `vendor/model` id, matching the vendor case-insensitively.

    The family function gets the lowercased model name after the first `/`. `overrides` are a provider's own
    vendor names, or its own profile for a vendor in the shared map, keyed in lowercase; they apply only to that
    provider. Returns `None` for a name without a `/`, an unknown vendor, or a model the family doesn't describe.
    """
    if '/' not in model_name:
        return None
    vendor, family_model_name = model_name.lower().split('/', 1)
    family_profile = (overrides or {}).get(vendor) or _VENDOR_FAMILY_PROFILES.get(vendor)
    return family_profile(family_model_name) if family_profile else None
