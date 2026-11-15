from __future__ import annotations as _annotations

from ..native_tools import WebSearchTool
from . import InlineDefsJsonSchemaTransformer, ModelProfile
from .openai import OpenAIModelProfile

_MUSE_TEXT_MODEL_PREFIXES = ('muse-spark-', 'muse-wonder-')


def meta_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Meta model."""
    return ModelProfile(json_schema_transformer=InlineDefsJsonSchemaTransformer)


def muse_model_profile(model_name: str) -> OpenAIModelProfile | None:
    """Get the model profile for a Meta Muse model, or `None` if `model_name` isn't one."""
    if not model_name.startswith(_MUSE_TEXT_MODEL_PREFIXES):
        return None
    # Muse always reasons: `api.meta.ai` rejects `reasoning.effort='none'` and accepts `minimal` through `max`,
    # so `thinking=False` sends `minimal`. `openai_supports_reasoning` stays off because it makes `OpenAI*Model`
    # drop sampling parameters while reasoning, which Meta accepts.
    return OpenAIModelProfile(
        supports_thinking=True,
        thinking_always_enabled=True,
        thinking_enabled_by_default=True,
        openai_reasoning_efforts=frozenset(('minimal', 'low', 'medium', 'high', 'xhigh', 'max')),
        openai_supports_encrypted_reasoning_content=True,
        # Muse labels progress preambles `commentary`; sending the phase back keeps the model from reading its
        # own preambles as final answers in long tool-calling runs.
        openai_supports_phase=True,
        supports_json_schema_output=True,
        supports_json_object_output=True,
        openai_chat_supports_web_search=True,
        supported_native_tools=frozenset({WebSearchTool}),
        context_window=1_048_576,
    )
