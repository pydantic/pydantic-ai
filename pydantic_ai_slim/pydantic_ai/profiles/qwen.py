from __future__ import annotations as _annotations

import re

from ..profiles.openai import OpenAIModelProfile
from . import InlineDefsJsonSchemaTransformer, ModelProfile

_QWEN_3_5_RE = re.compile(r'qwen-?3[\.\-]5')


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    if model_name.startswith('qwen-3-coder'):
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
