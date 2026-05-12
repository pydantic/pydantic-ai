from __future__ import annotations as _annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from textwrap import dedent
from typing import Any, ClassVar

from typing_extensions import Self

from .._json_schema import InlineDefsJsonSchemaTransformer, JsonSchemaTransformer
from .._utils import install_deprecated_kwarg_alias
from .._warnings import PydanticAIDeprecationWarning
from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from ..output import StructuredOutputMode

__all__ = [
    'ModelProfile',
    'ModelProfileSpec',
    'DEFAULT_PROFILE',
    'InlineDefsJsonSchemaTransformer',
    'JsonSchemaTransformer',
]


# Maps deprecated kwarg/attribute names to their renamed targets for `ModelProfile`.
# Used by `ModelProfile.__getattr__` for read access. Constructor aliasing is installed
# lazily on first instantiation of each subclass via `ModelProfile.__new__`, since
# `@dataclass` regenerates `__init__` on each subclass and overwrites a single base wrap.
# Subclasses extend this map by declaring their own `_deprecated_kwarg_aliases` class
# attribute; `__new__` walks the MRO to collect entries from every level.
_MODEL_PROFILE_DEPRECATED_FIELD_ALIASES: dict[str, str] = {
    'supported_builtin_tools': 'supported_native_tools',
}

# Tracks which subclasses have already had their deprecated-kwarg aliases installed,
# so the `__new__` lazy-install runs exactly once per class.
_DEPRECATED_KWARG_ALIASES_INSTALLED: set[type] = set()


@dataclass(kw_only=True)
class ModelProfile:
    """Describes how requests to and responses from specific models or families of models need to be constructed and processed to get the best results, independent of the model and provider classes used."""

    # Maps deprecated `__init__` kwarg names to their renamed targets. Subclasses can extend
    # this by declaring their own `_deprecated_kwarg_aliases = {...}`; `__new__` walks the
    # MRO at first instantiation to collect every level's entries.
    _deprecated_kwarg_aliases: ClassVar[dict[str, str]] = {
        'supported_builtin_tools': 'supported_native_tools',
    }

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        # Lazy install of deprecated-kwarg aliases on first instantiation of each subclass.
        # `@dataclass` regenerates `__init__` on each subclass, so a base-level wrap is lost.
        # `__new__` runs before `__init__` (via `type.__call__`), so we can wrap the
        # subclass's `__init__` exactly once before the constructor receives the legacy kwarg.
        if cls not in _DEPRECATED_KWARG_ALIASES_INSTALLED:
            _DEPRECATED_KWARG_ALIASES_INSTALLED.add(cls)
            collected: dict[str, str] = {}
            for klass in reversed(cls.__mro__):
                collected.update(getattr(klass, '_deprecated_kwarg_aliases', None) or {})
            for old, new in collected.items():
                install_deprecated_kwarg_alias(cls, old=old, new=new)
        return super().__new__(cls)

    supports_tools: bool = True
    """Whether the model supports tools."""
    supports_tool_return_schema: bool = False
    """Whether the model natively supports tool return schemas.

    When True, the model's API accepts a structured return schema alongside each tool definition.
    When False, return schemas are injected as JSON text into tool descriptions as a fallback.
    """
    supports_json_schema_output: bool = False
    """Whether the model supports JSON schema output.

    This is also referred to as 'native' support for structured output.
    Relates to the `NativeOutput` output type.
    """
    supports_json_object_output: bool = False
    """Whether the model supports a dedicated mode to enforce JSON output, without necessarily sending a schema.

    E.g. [OpenAI's JSON mode](https://platform.openai.com/docs/guides/structured-outputs#json-mode)
    Relates to the `PromptedOutput` output type.
    """
    supports_image_output: bool = False
    """Whether the model supports image output."""
    default_structured_output_mode: StructuredOutputMode = 'tool'
    """The default structured output mode to use for the model."""
    prompted_output_template: str = dedent(
        """
        Always respond with a JSON object that's compatible with this schema:

        {schema}

        Don't include any text or Markdown fencing before or after.
        """
    )
    """The instructions template to use for prompted structured output. The '{schema}' placeholder will be replaced with the JSON schema for the output."""
    native_output_requires_schema_in_instructions: bool = False
    """Whether to add prompted output template in native structured output mode"""
    json_schema_transformer: type[JsonSchemaTransformer] | None = None
    """The transformer to use to make JSON schemas for tools and structured output compatible with the model."""

    supports_thinking: bool = False
    """Whether the model supports thinking/reasoning configuration.

    When False, the unified `thinking` setting in `ModelSettings` is silently ignored.
    """

    thinking_always_enabled: bool = False
    """Whether the model always uses thinking/reasoning (e.g., OpenAI o-series, DeepSeek R1).

    When True, `thinking=False` is silently ignored since the model cannot disable thinking.
    Implies `supports_thinking=True`.
    """

    thinking_tags: tuple[str, str] = ('<think>', '</think>')
    """The tags used to indicate thinking parts in the model's output. Defaults to ('<think>', '</think>')."""

    ignore_streamed_leading_whitespace: bool = False
    """Whether to ignore leading whitespace when streaming a response.

    This is a workaround for models that emit `<think>\n</think>\n\n` or an empty text part ahead of tool calls (e.g. Ollama + Qwen3),
    which we don't want to end up treating as a final result when using `run_stream` with `str` a valid `output_type`.

    This is currently only used by `OpenAIChatModel`, `HuggingFaceModel`, and `GroqModel`.
    """

    supported_native_tools: frozenset[type[AbstractNativeTool]] = field(default_factory=lambda: SUPPORTED_NATIVE_TOOLS)
    """The set of native tool types that this model/profile supports.

    Defaults to ALL native tools. Profile functions should explicitly
    restrict this based on model capabilities.
    """

    @classmethod
    def from_profile(cls, profile: ModelProfile | None) -> Self:
        """Build a ModelProfile subclass instance from a ModelProfile instance."""
        if isinstance(profile, cls):
            return profile
        return cls().update(profile)

    def update(self, profile: ModelProfile | None) -> Self:
        """Update this ModelProfile (subclass) instance with the non-default values from another ModelProfile instance."""
        if not profile:
            return self
        field_names = set(f.name for f in fields(self))
        non_default_attrs = {
            f.name: getattr(profile, f.name)
            for f in fields(profile)
            if f.name in field_names and getattr(profile, f.name) != f.default
        }
        return replace(self, **non_default_attrs)

    def __getattr__(self, name: str) -> Any:
        # Deprecated alias for read access to a renamed field. Only warns when the renamed
        # target field actually exists on this class — otherwise raise `AttributeError` so
        # genuine typos still surface clearly.
        new_name = _MODEL_PROFILE_DEPRECATED_FIELD_ALIASES.get(name)
        if new_name is not None and new_name in {f.name for f in fields(type(self))}:
            warnings.warn(
                f'`{type(self).__name__}.{name}` is deprecated, use `.{new_name}` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            return getattr(self, new_name)
        raise AttributeError(name)


ModelProfileSpec = ModelProfile | Callable[[str], ModelProfile | None]

DEFAULT_PROFILE = ModelProfile()
