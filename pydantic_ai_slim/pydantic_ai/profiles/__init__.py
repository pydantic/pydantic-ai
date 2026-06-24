from __future__ import annotations as _annotations

from collections.abc import Callable
from textwrap import dedent
from typing import TypeAlias

from typing_extensions import TypedDict

from .._json_schema import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer
from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from ..output import StructuredOutputMode

__all__ = [
    'ModelProfile',
    'ModelProfileSpec',
    'DEFAULT_PROFILE',
    'DEFAULT_PROMPTED_OUTPUT_TEMPLATE',
    'DEFAULT_THINKING_TAGS',
    'InlineDefsJsonSchemaTransformer',
    'JsonSchemaTransformer',
    'merge_profile',
]


DEFAULT_PROMPTED_OUTPUT_TEMPLATE = dedent(
    """
    Always respond with a JSON object that's compatible with this schema:

    {schema}

    Don't include any text or Markdown fencing before or after.
    """
)
"""Default instructions template for prompted structured output. The `{schema}` placeholder is replaced with the JSON schema for the output."""

DEFAULT_THINKING_TAGS: tuple[str, str] = ('<think>', '</think>')
"""Default `(start_tag, end_tag)` pair for parsing thinking content out of text responses."""


class ModelProfile(TypedDict, total=False):
    """Describes how requests to and responses from specific models or families of models need to be constructed and processed to get the best results, independent of the model and provider classes used.

    All fields are optional; absent keys mean "use the documented default" (defaults are documented per field below and applied at access sites).

    Subclasses (`OpenAIModelProfile`, `AnthropicModelProfile`, ...) add provider-specific keys; cross-class merging via dict-spread is supported.
    """

    supports_tools: bool
    """Whether the model supports tools. Default: `True`."""

    supports_tool_return_schema: bool
    """Whether the model natively supports tool return schemas. Default: `False`.

    When True, the model's API accepts a structured return schema alongside each tool definition.
    When False, return schemas are injected as JSON text into tool descriptions as a fallback.
    """

    supports_json_schema_output: bool
    """Whether the model supports JSON schema output. Default: `False`.

    This is also referred to as 'native' support for structured output.
    Relates to the `NativeOutput` output type.
    """

    supports_json_object_output: bool
    """Whether the model supports a dedicated mode to enforce JSON output, without necessarily sending a schema. Default: `False`.

    E.g. [OpenAI's JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode)
    Relates to the `PromptedOutput` output type.
    """

    supports_image_output: bool
    """Whether the model supports image output. Default: `False`."""

    supports_inline_system_prompts: bool
    """Whether the provider's API accepts `SystemPromptPart`s inline at any position. Default: `False`.

    When `False`, non-leading `SystemPromptPart`s are wrapped as `UserPromptPart`s with
    `<system>...</system>` content in `Model.prepare_messages`. Leading ones still hoist to the
    provider's top-level system parameter.
    """

    default_structured_output_mode: StructuredOutputMode
    """The default structured output mode to use for the model. Default: `'tool'`."""

    prompted_output_template: str
    """The instructions template to use for prompted structured output. The `{schema}` placeholder will be replaced with the JSON schema for the output. Default: `DEFAULT_PROMPTED_OUTPUT_TEMPLATE`."""

    native_output_requires_schema_in_instructions: bool
    """Whether to add prompted output template in native structured output mode. Default: `False`."""

    json_schema_transformer: type[JsonSchemaTransformer] | None
    """The transformer to use to make JSON schemas for tools and structured output compatible with the model. Default: `None`."""

    supports_thinking: bool
    """Whether the model supports thinking/reasoning configuration. Default: `False`.

    When False, the unified `thinking` setting in `ModelSettings` is silently ignored.
    """

    thinking_always_enabled: bool
    """Whether the model always uses thinking/reasoning (e.g., OpenAI o-series, DeepSeek R1). Default: `False`.

    When True, `thinking=False` is silently ignored since the model cannot disable thinking.
    Implies `supports_thinking=True`.
    """

    thinking_tags: tuple[str, str]
    """The tags used to indicate thinking parts in the model's output. Default: [`DEFAULT_THINKING_TAGS`][pydantic_ai.profiles.DEFAULT_THINKING_TAGS]."""

    ignore_streamed_leading_whitespace: bool
    """Whether to ignore leading whitespace when streaming a response. Default: `False`.

    This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
    which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.

    This is currently only used by `OpenAIChatModel`, `HuggingFaceModel`, and `GroqModel`.
    """

    supported_native_tools: frozenset[type[AbstractNativeTool]]
    """The set of native tool types that this model/profile supports. Default: `SUPPORTED_NATIVE_TOOLS` (all)."""


DEFAULT_PROFILE: ModelProfile = {
    'supports_tools': True,
    'supports_tool_return_schema': False,
    'supports_json_schema_output': False,
    'supports_json_object_output': False,
    'supports_image_output': False,
    'default_structured_output_mode': 'tool',
    'prompted_output_template': DEFAULT_PROMPTED_OUTPUT_TEMPLATE,
    'native_output_requires_schema_in_instructions': False,
    'json_schema_transformer': None,
    'supports_thinking': False,
    'thinking_always_enabled': False,
    'thinking_tags': DEFAULT_THINKING_TAGS,
    'ignore_streamed_leading_whitespace': False,
    'supported_native_tools': SUPPORTED_NATIVE_TOOLS,
}
"""Fully populated default `ModelProfile`. Used as the base layer when resolving a model's effective profile."""


ModelProfileSpec: TypeAlias = ModelProfile | Callable[['ModelProfile'], 'ModelProfile']
"""Acceptable shapes for the `profile=` argument on a `Model`.

- A `ModelProfile` dict — a partial profile, merged on top of the provider's resolved default.
- A `Callable[[ModelProfile], ModelProfile]` — receives the provider's resolved default (with `DEFAULT_PROFILE` already merged in) and returns the final profile (full control: replace, derive, ignore the default).

Provider classes still expose `Provider.model_profile(model_name)` (`Callable[[str], ModelProfile | None]`) — that's a separate concept used internally by `Model.profile` to resolve the provider's default for a given model name.
"""


def merge_profile(base: ModelProfile | None, *overrides: ModelProfile | None) -> ModelProfile:
    """Merge profiles via dict-spread. Later arguments override earlier ones; `None` is treated as empty.

    This is the canonical way to layer profiles in providers and tests; replaces the old `ModelProfile.update()` method.
    """
    result: ModelProfile = {}
    if base:
        result = {**result, **base}
    for override in overrides:
        if override:
            result = {**result, **override}
    return result
