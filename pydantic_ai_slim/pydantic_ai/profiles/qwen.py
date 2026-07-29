from __future__ import annotations as _annotations

import re

from ..profiles.openai import OpenAIModelProfile
from . import InlineDefsJsonSchemaTransformer, ModelProfile

_QWEN_3_5_RE = re.compile(r'qwen-?3[\.\-]5')
# Gateways disagree on whether the generation number is hyphenated: Cerebras serves
# `qwen-3-coder-480b`, while Bedrock, Heroku and OpenRouter all use `qwen3-coder-*`. Accept both,
# the same way `_QWEN_3_5_RE` does.
_QWEN_3_CODER_RE = re.compile(r'^qwen-?3-coder')


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    if _QWEN_3_CODER_RE.match(model_name):
        return OpenAIModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            openai_supports_tool_choice_required=False,
            openai_supports_strict_tool_definition=False,
            ignore_streamed_leading_whitespace=True,
        )
    if _QWEN_3_5_RE.search(model_name):
        return ModelProfile(
            json_schema_transformer=InlineDefsJsonSchemaTransformer,
            ignore_streamed_leading_whitespace=True,
            supports_json_schema_output=True,
            supports_json_object_output=True,
        )
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        ignore_streamed_leading_whitespace=True,
    )
