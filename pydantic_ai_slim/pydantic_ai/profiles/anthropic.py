from __future__ import annotations as _annotations

from dataclasses import dataclass

from .._json_schema import JsonSchema, JsonSchemaTransformer
from . import ModelProfile


def anthropic_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for an Anthropic model."""
    models_that_support_json_schema_output = (
        'claude-haiku-4-5',
        'claude-sonnet-4-5',
        'claude-opus-4-1',
        'claude-opus-4-5',
    )
    """These models support both structured outputs and strict tool calling."""
    # TODO update when new models are released that support structured outputs
    # https://docs.claude.com/en/docs/build-with-claude/structured-outputs#example-usage

    supports_json_schema_output = model_name.startswith(models_that_support_json_schema_output)
    return ModelProfile(
        thinking_tags=('<thinking>', '</thinking>'),
        supports_json_schema_output=supports_json_schema_output,
        json_schema_transformer=AnthropicJsonSchemaTransformer,
        supports_tool_search=True,
    )


@dataclass(init=False)
class AnthropicJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms schemas to the subset supported by Anthropic structured outputs.

    Transformation is applied when:
    - `NativeOutput` is used as the `output_type` of the Agent
    - `strict=True` is set on the `Tool`

    The behavior of this transformer differs from the OpenAI one in that it sets `Tool.strict=False` by default when not explicitly set to True.

    Example:
        ```python
        from pydantic_ai import Agent

        agent = Agent('anthropic:claude-sonnet-4-5')

        @agent.tool_plain  # -> defaults to strict=False
        def my_tool(x: str) -> dict[str, int]:
            ...
        ```

    Anthropic's SDK `transform_schema()` automatically:
    - Adds `additionalProperties: false` to all objects (required by API)
    - Removes unsupported constraints (minLength, pattern, etc.)
    - Moves removed constraints to description field
    - Removes title and $schema fields
    """

    def walk(self) -> JsonSchema:
        from anthropic import transform_schema

        schema = super().walk()

        # The caller (pydantic_ai.models._customize_tool_def or _customize_output_object) coalesces
        # - output_object.strict = self.is_strict_compatible
        # - tool_def.strict = self.is_strict_compatible
        # the reason we don't default to `strict=True` is that the transformation could be lossy
        # so in order to change the behavior (default to True), we need to come up with logic that will check for lossiness
        # https://github.com/pydantic/pydantic-ai/issues/3541
        self.is_strict_compatible = self.strict is True  # not compatible when strict is False/None

        return transform_schema(schema) if self.strict is True else schema

    def transform(self, schema: JsonSchema) -> JsonSchema:
        schema.pop('title', None)
        schema.pop('$schema', None)
        return schema
