from __future__ import annotations as _annotations

from ..profiles.openai import OpenAIModelProfile
from . import InlineDefsJsonSchemaTransformer, ModelProfile
from ._json_schema import JsonSchema


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    if model_name.startswith('qwen-3-coder'):
        return OpenAIModelProfile(
            openai_supports_tool_choice_required=False,
            json_schema_transformer=NotStrictCompatibleJsonSchemaTransformer,
            ignore_streamed_leading_whitespace=True,
        )
    return ModelProfile(
        json_schema_transformer=InlineDefsJsonSchemaTransformer,
        ignore_streamed_leading_whitespace=True,
    )


class NotStrictCompatibleJsonSchemaTransformer(InlineDefsJsonSchemaTransformer):
    """Transforms the JSON Schema to be not strict compatible."""

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict)
        self.is_strict_compatible = False

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema
