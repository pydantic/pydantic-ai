"""Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the
specific LLM being used.
"""

from __future__ import annotations as _annotations

import base64
import json
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime
from functools import cache, cached_property
from types import TracebackType
from typing import Annotated, Any, Generic, Literal, TypeVar, cast, get_args, overload

import httpx
import pydantic
from typing_extensions import Self, TypeAliasType, TypedDict, deprecated
from typing_inspection.introspection import get_literal_values

from .. import _deferred_capabilities, _utils
from .._deprecated_callable import deprecated_callable_property
from .._json_schema import JsonSchemaTransformer
from .._output import StructuredTextOutputSchema
from .._parts_manager import ModelResponsePartsManager
from .._run_context import RunContext
from .._warnings import PydanticAIDeprecationWarning
from ..exceptions import UserError
from ..messages import (
    BaseToolCallPart,
    BinaryImage,
    FilePart,
    FileUrl,
    FinalResultEvent,
    FinishReason,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponsePart,
    ModelResponseState,
    ModelResponseStreamEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
    VideoUrl,
)
from ..native_tools import AbstractNativeTool
from ..native_tools._tool_search import ToolSearchTool
from ..output import OutputMode, OutputObjectDefinition, StructuredOutputMode
from ..profiles import DEFAULT_PROFILE, ModelProfile, ModelProfileSpec
from ..providers import InterfaceClient, Provider, infer_provider, infer_provider_class
from ..settings import ModelSettings, ThinkingLevel, merge_model_settings
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._known_model_names import KnownModelName as KnownModelName

DEFAULT_HTTP_TIMEOUT: int = 600
"""Default HTTP timeout in seconds for API requests.

This matches the default timeout used by OpenAI's Python client.
See https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9
"""


@cache
def known_model_names() -> tuple[str, ...]:
    """Return every model name known to [`KnownModelName`][pydantic_ai.models.KnownModelName].

    This is the public, stable way to enumerate the known model ids. Prefer it over introspecting
    the `KnownModelName` type alias directly (e.g. `get_args(KnownModelName.__value__)`), which is
    not part of the public API and would break if the alias were ever recomposed.
    """
    return tuple(get_literal_values(KnownModelName.__value__, unpack_type_aliases='eager'))


OpenAIChatCompatibleProvider = TypeAliasType(
    'OpenAIChatCompatibleProvider',
    Literal[
        'alibaba',
        'azure',
        'cerebras',
        'deepseek',
        'fireworks',
        'github',
        'grok',
        'heroku',
        'litellm',
        'moonshotai',
        'nebius',
        'ollama',
        'openrouter',
        'ovhcloud',
        'sambanova',
        'together',
        'vercel',
    ],
)
OpenAIResponsesCompatibleProvider = TypeAliasType(
    'OpenAIResponsesCompatibleProvider',
    Literal[
        'azure',
        'deepseek',
        'fireworks',
        'grok',
        'nebius',
        'openrouter',
        'ovhcloud',
        'sambanova',
        'together',
    ],
)


@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and output handling."""

    function_tools: list[ToolDefinition] = field(default_factory=list[ToolDefinition])
    native_tools: Annotated[
        list[AbstractNativeTool],
        # Accept the pre-rename `builtin_tools` key when validating from a dict (e.g. through
        # `pydantic.TypeAdapter`). The dump uses the new name only.
        pydantic.Field(validation_alias=pydantic.AliasChoices('native_tools', 'builtin_tools')),
    ] = field(default_factory=list[AbstractNativeTool])

    output_mode: OutputMode = 'text'
    output_object: OutputObjectDefinition | None = None
    output_tools: list[ToolDefinition] = field(default_factory=list[ToolDefinition])
    prompted_output_template: str | Literal[False] | None = None
    allow_text_output: bool = True
    allow_image_output: bool = False

    instruction_parts: list[InstructionPart] | None = None
    """Structured instruction parts with metadata about their origin (static vs dynamic).

    Static instructions (`dynamic=False`) come from literal strings passed to `Agent(instructions=...)`.
    Dynamic instructions (`dynamic=True`) come from `@agent.instructions` functions, `TemplateStr`,
    or toolset `get_instructions()` methods.

    Models that support granular caching (e.g. Anthropic, Bedrock) use this to place cache
    boundaries at the static/dynamic instruction boundary.
    """

    thinking: ThinkingLevel | None = None
    """Resolved thinking/reasoning configuration for this request.

    `None` means the model should use its default behavior. Set by the base
    `Model.prepare_request()` from the unified `thinking` field in `ModelSettings`,
    after checking that the model's profile supports thinking.
    """

    @cached_property
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {tool_def.name: tool_def for tool_def in [*self.function_tools, *self.output_tools]}

    @property
    def builtin_tools(self) -> list[AbstractNativeTool]:
        """Deprecated: use [`native_tools`][pydantic_ai.models.ModelRequestParameters.native_tools] instead."""
        warnings.warn(
            '`ModelRequestParameters.builtin_tools` is deprecated, use `ModelRequestParameters.native_tools` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        return self.native_tools

    @cached_property
    def prompted_output_instructions(self) -> str | None:
        if self.prompted_output_template and self.output_object:
            return StructuredTextOutputSchema.build_instructions(self.prompted_output_template, self.output_object)
        return None

    def with_default_output_mode(self, output_mode: StructuredOutputMode) -> ModelRequestParameters:
        """Set the default output mode if the current mode is 'auto', atomically updating allow_text_output.

        No-op if the current output_mode is not 'auto'. This ensures the two fields stay in sync —
        output_mode='tool' implies allow_text_output=False, while 'native' and 'prompted' imply
        allow_text_output=True.
        """
        if self.output_mode != 'auto':
            return self
        return replace(self, output_mode=output_mode, allow_text_output=output_mode in ('native', 'prompted'))

    __repr__ = _utils.dataclasses_no_defaults_repr


# Wrap the dataclass-generated `__init__` so direct construction still accepts a
# deprecated `builtin_tools=` kwarg. (Pydantic deserialization is handled by the
# `validation_alias` on the `native_tools` field above.)
_utils.install_deprecated_kwarg_alias(ModelRequestParameters, old='builtin_tools', new='native_tools')


@dataclass(kw_only=True)
class ModelRequestContext:
    """Context for model request hooks.

    Wrapping these parameters in a dataclass instead of a tuple makes the signature
    future-proof: new fields can be added without breaking existing implementations.
    """

    model: Model
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters


class Model(ABC, Generic[InterfaceClient]):
    """Abstract class for a model."""

    _provider: Provider[InterfaceClient]
    _profile: ModelProfileSpec | None = None
    _settings: ModelSettings | None = None

    def __init__(
        self,
        *,
        settings: ModelSettings | None = None,
        profile: ModelProfileSpec | None = None,
    ) -> None:
        """Initialize the model with optional settings and profile.

        Args:
            settings: Model-specific settings that will be used as defaults for this model.
            profile: The model profile to use.
        """
        self._settings = settings
        self._profile = profile

    @property
    def provider(self) -> Provider[InterfaceClient] | None:
        """The provider for this model, if any."""
        return self._provider

    async def __aenter__(self) -> Self:
        """Enter the model context, delegating to the provider to manage its HTTP client lifecycle."""
        if self.provider is not None:
            await self.provider.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        """Exit the model context, closing the provider's HTTP client if it owns one."""
        if self.provider is not None:
            await self.provider.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def settings(self) -> ModelSettings | None:
        """Get the model settings."""
        return self._settings

    @abstractmethod
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Make a request to the model.

        This is ultimately called by `pydantic_ai._agent_graph.ModelRequestNode._make_request(...)`.
        """
        raise NotImplementedError()

    async def count_tokens(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> RequestUsage:
        """Make a request to the model for counting tokens."""
        # This method is not required, but you need to implement it if you want to support `UsageLimits.count_tokens_before_request`.
        raise NotImplementedError(f'Token counting ahead of the request is not supported by {self.__class__.__name__}')

    async def compact_messages(
        self,
        request_context: ModelRequestContext,
        *,
        instructions: str | None = None,
    ) -> ModelResponse:
        """Compact messages to reduce conversation context size.

        This method is optional and only supported by specific providers
        (e.g. OpenAI Responses API). Providers that support compaction
        override this method with their implementation.
        """
        raise NotImplementedError(f'Message compaction is not supported by {self.__class__.__name__}')

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        """Make a request to the model and return a streaming response."""
        # This method is not required, but you need to implement it if you want to support streamed responses
        raise NotImplementedError(f'Streamed requests not supported by this {self.__class__.__name__}')
        # yield is required to make this a generator for type checking
        # noinspection PyUnreachableCode
        yield  # pragma: no cover

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        """Customize the request parameters for the model.

        This method can be overridden by subclasses to modify the request parameters before sending them to the model.
        In particular, this method can be used to make modifications to the generated tool JSON schemas if necessary
        for vendor/model-specific reasons.
        """
        if transformer := self.profile.json_schema_transformer:
            model_request_parameters = replace(
                model_request_parameters,
                function_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.function_tools],
                output_tools=[_customize_tool_def(transformer, t) for t in model_request_parameters.output_tools],
            )
            if output_object := model_request_parameters.output_object:
                model_request_parameters = replace(
                    model_request_parameters,
                    output_object=_customize_output_object(transformer, output_object),
                )

        return model_request_parameters

    def prepare_request(
        self,
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> tuple[ModelSettings | None, ModelRequestParameters]:
        """Prepare request inputs before they are passed to the provider.

        This merges the given `model_settings` with the model's own `settings` attribute and ensures
        `customize_request_parameters` is applied to the resolved
        [`ModelRequestParameters`][pydantic_ai.models.ModelRequestParameters]. Subclasses can override this method if
        they need to customize the preparation flow further, but most implementations should simply call
        `self.prepare_request(...)` at the start of their `request` (and related) methods.
        """
        model_settings = merge_model_settings(self.settings, model_settings)

        params = self.customize_request_parameters(model_request_parameters)
        params = _prepare_return_schemas(params, self.profile)

        # Resolve unified thinking setting and strip from model_settings
        if model_settings and 'thinking' in model_settings:
            thinking_value = model_settings['thinking']
            if self.profile.supports_thinking or self.profile.thinking_always_enabled:
                if not (thinking_value is False and self.profile.thinking_always_enabled):
                    params = replace(params, thinking=thinking_value)
            stripped = {k: v for k, v in model_settings.items() if k != 'thinking'}
            model_settings = cast(ModelSettings, stripped) if stripped else None

        if native_tools := params.native_tools:
            # Deduplicate native tools
            params = replace(
                params,
                native_tools=list({tool.unique_id: tool for tool in native_tools}.values()),
            )

        params = params.with_default_output_mode(self.profile.default_structured_output_mode)

        # Reset irrelevant fields
        if params.output_tools and params.output_mode != 'tool':
            params = replace(params, output_tools=[])
        if params.output_object and params.output_mode not in ('native', 'prompted'):
            params = replace(params, output_object=None)
        if params.prompted_output_template and params.output_mode not in ('prompted', 'native'):
            params = replace(params, prompted_output_template=None)  # pragma: no cover

        # Set default prompted output template
        if (
            params.output_mode == 'prompted'
            or (params.output_mode == 'native' and self.profile.native_output_requires_schema_in_instructions)
        ) and params.prompted_output_template is None:
            params = replace(params, prompted_output_template=self.profile.prompted_output_template)

        # Append prompted_output_instructions to instruction_parts so models that use structured
        # instruction parts (for per-part system messages or cache placement) also get them.
        # Done here (after customize_request_parameters) so it uses the final resolved template.
        if output_instr := params.prompted_output_instructions:
            parts = [*(params.instruction_parts or []), InstructionPart(content=output_instr)]
            params = replace(params, instruction_parts=InstructionPart.sorted(parts))

        # Check if output mode is supported
        if params.output_mode == 'native' and not self.profile.supports_json_schema_output:
            raise UserError('Native structured output is not supported by this model.')
        if params.output_mode == 'tool' and not self.profile.supports_tools:
            raise UserError('Tool output is not supported by this model.')
        if params.allow_image_output and not self.profile.supports_image_output:
            raise UserError('Image output is not supported by this model.')

        # Check native tools and handle fallback swap
        if params.native_tools or any(t.unless_native or t.with_native for t in params.function_tools):
            params = self._resolve_native_tool_swap(params)

        return model_settings, params

    def prepare_messages(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Pre-process the message history before it's handed to the adapter's message-prep step.

        Currently translates any typed `NativeToolSearch*Part` instances carried over from a
        prior native turn (e.g. Anthropic / OpenAI Responses) into the local-shape
        `ToolSearch*Part` instances when the active model's profile doesn't support
        `ToolSearchTool` — splitting the single `ModelResponse(call+return)` carrying the
        inline server-side result into `ModelResponse(call) + ModelRequest(return)` so the
        adapter sees a normal function-call exchange against `search_tools`.

        Also wraps non-leading `SystemPromptPart`s as `<system>`-tagged `UserPromptPart`s when
        the profile's `supports_inline_system_prompts` is `False`.

        Subclasses normally don't need to override this; the framework calls it on the
        agent's behalf in `_agent_graph._make_request` so per-adapter message-prep code
        sees a homogeneous shape regardless of which provider produced the prior turn.
        """
        if ToolSearchTool not in self.profile.supported_native_tools:
            from .._tool_search import synthesize_local_tool_search_messages

            messages = synthesize_local_tool_search_messages(messages)

        if not self.profile.supports_inline_system_prompts:
            messages = _wrap_non_leading_system_prompts(messages)

        return messages

    def _resolve_native_tool_swap(self, params: ModelRequestParameters) -> ModelRequestParameters:
        """Swap native tools and function-tool fallbacks/corpus based on profile support.

        Four rules drive the per-tool filter:

        1. `unless_native` matches a supported native tool → drop from wire.
        2. `with_native` matches a supported native tool → keep on wire; the adapter
           applies any native-tool-specific format (e.g. Anthropic / OpenAI's wire-side
           `defer_loading` flag for `ToolSearchTool`).
        3. `with_native` matches an *unsupported* native tool AND `defer_loading=True`
           → drop from wire (the corpus member is currently undiscovered, so the model has
           no way to call it on this provider).
        4. Otherwise → keep.

        On top of the four-rule filter, two narrower drops apply, kept independent:

        * `optional=True` only governs the *unsupported-on-this-model* path: an unsupported
          optional native tool is silently dropped (no error raised). It does NOT govern the
          corpus-empty drop below.
        * The corpus-empty drop is specific to the framework-managed tool-search native tool's
          corpus-management role: an *optional* `ToolSearchTool` is dropped when its
          corpus ends up empty after filtering, since sending it with no deferred tools
          to discover would waste a tool slot. A non-optional `ToolSearchTool` stays —
          the user asked explicitly. Other native tools don't have a corpus and aren't subject
          to this drop, so making `optional` a base-class field doesn't accidentally cause
          e.g. `WebSearchTool(optional=True)` to be dropped here.
        """
        supported_types = self.profile.supported_native_tools

        supported_natives = [t for t in params.native_tools if isinstance(t, tuple(supported_types))]
        unsupported_natives = [t for t in params.native_tools if not isinstance(t, tuple(supported_types))]

        supported_ids = {t.unique_id for t in supported_natives}
        unsupported_ids = {t.unique_id for t in unsupported_natives}
        optional_ids = {t.unique_id for t in unsupported_natives if t.optional}
        fallback_ids = {t.unless_native for t in params.function_tools if t.unless_native}

        without_fallback = unsupported_ids - fallback_ids - optional_ids
        if without_fallback:
            unsupported_names = [type(t).__name__ for t in unsupported_natives if t.unique_id in without_fallback]
            supported_names = [t.__name__ for t in supported_types]
            raise UserError(
                f'Native tool(s) {unsupported_names} not supported by this model. '
                f'Supported: {supported_names}. '
                f'To use these tools with this model, provide a local fallback via '
                f'NativeOrLocalTool(native=..., local=...) or the `local` parameter '
                f"of the capability (e.g. WebSearch(local='duckduckgo'), WebFetch(local=True), "
                f'MCP(local=True), ImageGeneration(local=my_func)). '
                f'Some capabilities require an optional install group for the local fallback '
                f'(e.g. `pip install "pydantic-ai-slim[mcp]"` for MCP).'
            )

        tool_search_resolution = _resolve_tool_search_native_for_capability_owned_corpus(
            supported_natives, params.function_tools
        )
        supported_natives = tool_search_resolution.native_tools
        tool_search_kept_local = tool_search_resolution.keep_search_tools_local

        function_tools: list[ToolDefinition] = []
        for t in params.function_tools:
            # Rule 1: drop local fallback when the native tool is supported — except for
            # `search_tools` when tool search was kept local for capability visibility,
            # where the local function tool is the callback the client-executed native
            # surface dispatches to.
            if t.unless_native and t.unless_native in supported_ids:
                if not (tool_search_kept_local and t.unless_native == ToolSearchTool.kind):
                    continue
            # Rule 3: drop undiscovered corpus members when the native tool is unsupported.
            if t.with_native and t.with_native not in supported_ids and t.defer_loading:
                continue
            # Rules 2 + 4: keep.
            function_tools.append(t)

        # Drop optional `ToolSearchTool` whose managed corpus is empty after filtering —
        # nothing to discover, sending it would waste a tool slot. The `isinstance` check
        # confines this to ToolSearchTool specifically: other native tools don't carry a corpus,
        # so making `optional` a base-class field doesn't accidentally drop e.g.
        # `WebSearchTool(optional=True)` here on absence of dependents.
        remaining_corpus_ids = {t.with_native for t in function_tools if t.with_native}
        supported_natives = [
            t
            for t in supported_natives
            if not (isinstance(t, ToolSearchTool) and t.optional) or t.unique_id in remaining_corpus_ids
        ]
        return replace(params, native_tools=supported_natives, function_tools=function_tools)

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model name."""
        raise NotImplementedError()

    @property
    def model_id(self) -> str:
        """The fully qualified model name in `'provider:model_name'` format."""
        return f'{self.system}:{self.model_name}'

    @property
    def label(self) -> str:
        """Human-friendly display label for the model.

        Handles common patterns:
        - gpt-5 -> GPT 5
        - claude-sonnet-4-5 -> Claude Sonnet 4.5
        - gemini-2.5-pro -> Gemini 2.5 Pro
        - meta-llama/llama-3-70b -> Llama 3 70b (OpenRouter style)
        """
        label = self.model_name
        # Handle OpenRouter-style names with / (e.g., meta-llama/llama-3-70b)
        if '/' in label:
            label = label.split('/')[-1]

        parts = label.split('-')
        result: list[str] = []

        for i, part in enumerate(parts):
            if i == 0 and part.lower() == 'gpt':
                result.append(part.upper())
            elif part.replace('.', '').isdigit():
                if result and result[-1].replace('.', '').isdigit():
                    result[-1] = f'{result[-1]}.{part}'
                else:
                    result.append(part)
            else:
                result.append(part.capitalize())

        return ' '.join(result)

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """Return the set of native tool types this model class can handle.

        Subclasses should override this to reflect their actual capabilities.
        Default is empty set - subclasses must explicitly declare support.
        """
        return frozenset()

    @classmethod
    def supported_builtin_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """Deprecated: use [`supported_native_tools`][pydantic_ai.models.Model.supported_native_tools] instead."""
        warnings.warn(
            '`Model.supported_builtin_tools()` is deprecated, use `Model.supported_native_tools()` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        return cls.supported_native_tools()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # If a subclass overrides only the deprecated `supported_builtin_tools` classmethod
        # (and not the new `supported_native_tools`), wire the legacy override through so
        # the framework still picks up the user's declared tools — with a warning.
        own = cls.__dict__
        if 'supported_builtin_tools' in own and 'supported_native_tools' not in own:
            legacy: Any = own['supported_builtin_tools']
            warnings.warn(
                f'{cls.__name__} overrides `supported_builtin_tools()`, which is deprecated — '
                'override `supported_native_tools()` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )

            # Promote the legacy override to be this class's `supported_native_tools`, and
            # replace its `supported_builtin_tools` with a stub that warns and delegates to
            # the modern method. This way a further subclass overriding only the modern
            # method still wins when callers reach for the legacy name (mixed-generation
            # MRO case): `Sub.supported_builtin_tools()` → modern stub → `cls.supported_native_tools()`
            # → modern override on `Sub`.
            if isinstance(legacy, classmethod):
                legacy_func: Any = legacy.__func__  # type: ignore[reportUnknownMemberType]
            else:
                legacy_func = legacy

            def _supported_native_tools_via_legacy(
                _cls: type[Model[Any]],
                _legacy_func: Any = legacy_func,
            ) -> frozenset[type[AbstractNativeTool]]:
                return _legacy_func(_cls)

            def _supported_builtin_tools_delegating(
                _cls: type[Model[Any]],
            ) -> frozenset[type[AbstractNativeTool]]:
                warnings.warn(
                    '`Model.supported_builtin_tools()` is deprecated, use `Model.supported_native_tools()` instead.',
                    PydanticAIDeprecationWarning,
                    stacklevel=2,
                )
                return _cls.supported_native_tools()

            setattr(cls, 'supported_native_tools', classmethod(_supported_native_tools_via_legacy))
            setattr(cls, 'supported_builtin_tools', classmethod(_supported_builtin_tools_delegating))

    @cached_property
    def profile(self) -> ModelProfile:
        """The model profile.

        We use this to compute the intersection of the profile's supported_native_tools
        and the model's implemented tools, ensuring model.profile.supported_native_tools
        is the single source of truth for what native tools are actually usable.
        """
        _profile = self._profile
        if callable(_profile):
            _profile = _profile(self.model_name)

        if _profile is None:
            _profile = DEFAULT_PROFILE

        # Compute intersection: profile's allowed tools & model's implemented tools
        model_supported = self.__class__.supported_native_tools()
        profile_supported = _profile.supported_native_tools
        effective_tools = profile_supported & model_supported

        if effective_tools != profile_supported:
            _profile = replace(_profile, supported_native_tools=effective_tools)

        return _profile

    @property
    @abstractmethod
    def system(self) -> str:
        """The model provider, ex: openai.

        Use to populate the `gen_ai.system` OpenTelemetry semantic convention attribute,
        so should use well-known values listed in
        https://opentelemetry.io/docs/specs/semconv/attributes-registry/gen-ai/#gen-ai-system
        when applicable.
        """
        raise NotImplementedError()

    @property
    def base_url(self) -> str | None:
        """The base URL for the provider API, if available."""
        return None

    @staticmethod
    def _get_instruction_parts(
        messages: Sequence[ModelMessage], model_request_parameters: ModelRequestParameters
    ) -> list[InstructionPart] | None:
        """Get structured instruction parts for the current request.

        Uses `model_request_parameters.instruction_parts` when set (normal agent flow).
        Falls back to synthesizing from `ModelRequest.instructions` in message history
        when `instruction_parts` is `None` (e.g. direct `model.request()` calls).
        """
        if model_request_parameters.instruction_parts is not None:
            return model_request_parameters.instruction_parts or None

        # Fallback: synthesize from message history for direct model.request() callers.
        # Mirrors the last-two-requests logic from `pydantic_ai._instrumentation.get_instructions`:
        # if the most recent request only has tool-return/retry-prompt parts (a "mock" request
        # for result tools), use the instructions from the second-to-most-recent request.
        last_two_requests: list[ModelRequest] = []
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                last_two_requests.append(message)
                if len(last_two_requests) == 2:
                    break
                if message.instructions is not None:
                    return [InstructionPart(content=message.instructions)]

        if len(last_two_requests) == 2:
            most_recent = last_two_requests[0]
            second = last_two_requests[1]
            if (
                all(p.part_kind == 'tool-return' or p.part_kind == 'retry-prompt' for p in most_recent.parts)
                and second.instructions is not None
            ):
                return [InstructionPart(content=second.instructions)]

        return None


@dataclass
class StreamedResponse(ABC):
    """Streamed response from an LLM when calling a tool."""

    model_request_parameters: ModelRequestParameters

    final_result_event: FinalResultEvent | None = field(default=None, init=False)

    provider_response_id: str | None = field(default=None, init=False)
    provider_details: dict[str, Any] | None = field(default=None, init=False)
    finish_reason: FinishReason | None = field(default=None, init=False)

    _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
    _usage: RequestUsage = field(default_factory=RequestUsage, init=False)
    _cancelled: bool = field(default=False, init=False)
    _finished: bool = field(default=False, init=False)

    @cached_property
    def _parts_manager(self) -> ModelResponsePartsManager:
        # Built lazily so subclasses don't need to remember `super().__post_init__()`.
        # `model_request_parameters` is handed in so streamed `ToolCallPart`s auto-promote
        # to their typed subclasses (via `ToolDefinition.tool_kind`) from the first
        # `PartStartEvent` — consumers see typed parts throughout the stream rather than
        # only after a post-stream pass.
        return ModelResponsePartsManager(model_request_parameters=self.model_request_parameters)

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:  # noqa: C901
        """Stream the response as an async iterable of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This proxies the `_event_iterator()` and emits all events, while also checking for matches
        on the result schema and emitting a [`FinalResultEvent`][pydantic_ai.messages.FinalResultEvent] if/when the
        first match is found.
        """
        if self._event_iterator is None:

            async def iterator_with_final_event(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                async for event in iterator:
                    yield event
                    if (
                        final_result_event := _get_final_result_event(event, self.model_request_parameters)
                    ) is not None:
                        self.final_result_event = final_result_event
                        yield final_result_event
                        break

                # If we broke out of the above loop, we need to yield the rest of the events
                # If we didn't, this will just be a no-op
                async for event in iterator:
                    yield event

            async def iterator_with_part_end(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                last_start_event: PartStartEvent | None = None

                def part_end_event(next_part: ModelResponsePart | None = None) -> PartEndEvent | None:
                    if not last_start_event:
                        return None

                    index = last_start_event.index
                    part = self._parts_manager.get_parts()[index]
                    if not isinstance(part, TextPart | ThinkingPart | BaseToolCallPart):
                        # Parts other than these 3 don't have deltas, so don't need an end part.
                        return None

                    return PartEndEvent(
                        index=index,
                        part=part,
                        next_part_kind=next_part.part_kind if next_part else None,
                    )

                async for event in iterator:
                    if isinstance(event, PartStartEvent):
                        if last_start_event:
                            end_event = part_end_event(event.part)
                            if end_event:
                                yield end_event

                            event.previous_part_kind = last_start_event.part.part_kind
                        last_start_event = event

                    yield event

                end_event = part_end_event()
                if end_event:
                    yield end_event

            async def iterator_with_cancel_guard(
                iterator: AsyncIterator[ModelResponseStreamEvent],
            ) -> AsyncIterator[ModelResponseStreamEvent]:
                # Suppress transport errors caused by `cancel()` tearing down the
                # connection mid-stream. The try/except has to live inside an
                # async generator body so it's active at every `await` during
                # iteration.
                try:
                    async for event in iterator:
                        yield event
                except self.get_stream_cancel_errors():
                    if not self.cancelled:
                        raise
                else:
                    # Only natural `StopAsyncIteration` flips `_finished`. Early
                    # `break` / `aclose()` (raising `GeneratorExit` at the suspended
                    # `yield`) and any in-flight exception leave `_finished=False`
                    # so `get()` reports the truncated response as `'incomplete'`
                    # rather than silently stamping it `'complete'`. The cancel
                    # branch above explicitly sets `_cancelled` (→ `'interrupted'`).
                    self._finished = True

            self._event_iterator = iterator_with_cancel_guard(
                iterator_with_part_end(iterator_with_final_event(self._get_event_iterator()))
            )
        return self._event_iterator

    async def cancel(self) -> None:
        """Cancel the stream, stopping token generation.

        Sets `self._cancelled = True` before delegating to `close_stream()`
        so the flag is visible to any iterator that observes the transport error
        raised when the underlying connection is torn down, even if
        `close_stream()` itself raises.
        """
        if self.cancelled:
            return
        self._cancelled = True
        await self.close_stream()

    def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
        """Return transport errors caused by `cancel()` tearing down the stream.

        The default covers model classes whose SDKs iterate `httpx` responses
        directly (Anthropic, OpenAI, Groq, Mistral, Google GenAI, HuggingFace,
        and the custom Gemini client), since they let bare `httpx` errors
        propagate from chunk reads. Model classes that use other transports
        (for example gRPC or botocore) should override this method.
        """
        return (httpx.StreamError, httpx.TransportError)

    async def close_stream(self) -> None:
        """Close the underlying HTTP/gRPC connection.

        Model classes must override this to stop token generation (and billing)
        on the remote side. Integrations that cannot support cancellation should
        leave the default implementation so `cancel()` fails clearly rather than
        silently reporting successful cancellation while generation continues.
        """
        raise NotImplementedError(
            f'Stream cancellation is not implemented for {type(self).__name__}. '
            'This model class must override `close_stream()` to support streaming cancellation.'
        )

    # TODO: (v2) We should not have public private methods which need to be overwritten.
    @abstractmethod
    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        """Return an async iterator of [`ModelResponseStreamEvent`][pydantic_ai.messages.ModelResponseStreamEvent]s.

        This method should be implemented by subclasses to translate the vendor-specific stream of events into
        pydantic_ai-format events.

        It should use the `_parts_manager` to handle deltas, and should update the `_usage` attributes as it goes.
        """
        raise NotImplementedError()
        # noinspection PyUnreachableCode
        yield

    def get(self) -> ModelResponse:
        """Build a [`ModelResponse`][pydantic_ai.messages.ModelResponse] from the data received from the stream so far."""
        if self._cancelled:
            state: ModelResponseState = 'interrupted'
        elif self._finished:
            state = 'complete'
        else:
            state = 'incomplete'
        return ModelResponse(
            parts=self._parts_manager.get_parts(),
            model_name=self.model_name,
            timestamp=self.timestamp,
            usage=self._usage,
            provider_name=self.provider_name,
            provider_url=self.provider_url,
            provider_response_id=self.provider_response_id,
            provider_details=self.provider_details,
            finish_reason=self.finish_reason,
            state=state,
        )

    @deprecated_callable_property(
        '`StreamedResponse.usage` is no longer a method; access it as a property (drop the parentheses).'
    )
    def usage(self) -> RequestUsage:
        """Get the usage of the response so far. This will not be the final usage until the stream is exhausted."""
        return self._usage

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name of the response."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def provider_name(self) -> str | None:
        """Get the provider name."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def provider_url(self) -> str | None:
        """Get the provider base URL."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def timestamp(self) -> datetime:
        """Get the timestamp of the response."""
        raise NotImplementedError()

    @property
    def cancelled(self) -> bool:
        """Whether the stream has been cancelled via `cancel()`."""
        return self._cancelled


ALLOW_MODEL_REQUESTS = True
"""Whether to allow requests to models.

This global setting allows you to disable request to most models, e.g. to make sure you don't accidentally
make costly requests to a model during tests.

The testing models [`TestModel`][pydantic_ai.models.test.TestModel] and
[`FunctionModel`][pydantic_ai.models.function.FunctionModel] are no affected by this setting.
"""


def check_allow_model_requests() -> None:
    """Check if model requests are allowed.

    If you're defining your own models that have costs or latency associated with their use, you should call this in
    [`Model.request`][pydantic_ai.models.Model.request] and [`Model.request_stream`][pydantic_ai.models.Model.request_stream].

    Raises:
        RuntimeError: If model requests are not allowed.
    """
    if not ALLOW_MODEL_REQUESTS:
        raise RuntimeError('Model requests are not allowed, since ALLOW_MODEL_REQUESTS is False')


@contextmanager
def override_allow_model_requests(allow_model_requests: bool) -> Generator[None]:
    """Context manager to temporarily override [`ALLOW_MODEL_REQUESTS`][pydantic_ai.models.ALLOW_MODEL_REQUESTS].

    Args:
        allow_model_requests: Whether to allow model requests within the context.
    """
    global ALLOW_MODEL_REQUESTS
    old_value = ALLOW_MODEL_REQUESTS
    ALLOW_MODEL_REQUESTS = allow_model_requests  # pyright: ignore[reportConstantRedefinition]
    try:
        yield
    finally:
        ALLOW_MODEL_REQUESTS = old_value  # pyright: ignore[reportConstantRedefinition]


_LEGACY_MODEL_PREFIXES: dict[str, str] = {
    'gpt': 'openai',
    'o1': 'openai',
    'o3': 'openai',
    'claude': 'anthropic',
    'gemini': 'google',
}
"""Backward compat: allows prefix-only model names like `gpt-4` without `provider:`."""


def parse_model_id(model: str) -> tuple[str | None, str]:
    """Parse a model id string into its provider and model name components.

    Handles both the modern `provider:model` format and legacy model names
    that start with known prefixes (e.g., `gpt-4`, `claude-3`).

    Emits a `DeprecationWarning` when a legacy prefix-based model name is used.

    Args:
        model: A model identifier string, either `provider:model_name` or a legacy
            prefix-based name.

    Returns:
        A tuple of `(provider_name, model_name)`. If the provider can't be inferred,
        returns `(None, model)` so callers can decide how to handle unknown providers.
    """
    if ':' in model:
        provider_name, model_name = model.split(':', maxsplit=1)
        return provider_name, model_name

    # Legacy model names without provider prefix
    for prefix, provider_name in _LEGACY_MODEL_PREFIXES.items():
        if model.startswith(prefix):
            warnings.warn(
                f'Specifying a model name without a provider prefix is deprecated. '
                f"Instead of {model!r}, use '{provider_name}:{model}'.",
                DeprecationWarning,
                stacklevel=2,
            )
            return provider_name, model

    # Unknown prefix: let callers decide how to handle this case.
    return None, model


def infer_model_profile(model: str) -> ModelProfile:
    """Infer the model profile from a model id string without constructing a provider.

    Uses `Provider.model_profile` to look up the profile for the given model.
    Returns `DEFAULT_PROFILE` for unknown or unrecognized providers.

    Note: This returns the raw provider profile **without** intersecting with
    `Model.supported_native_tools()`, unlike `Model.profile`. This means the returned
    profile may claim support for native tools that a specific `Model` subclass doesn't
    implement. This is acceptable for best-effort scenarios (e.g. `TemporalModel` with
    unregistered model strings) where the actual `Model` class isn't available.

    Args:
        model: A model identifier string (e.g. `'openai:gpt-5'`, `'anthropic:claude-sonnet-4-5'`).

    Returns:
        The inferred `ModelProfile`, or `DEFAULT_PROFILE` if the provider is unknown.
    """
    provider, model_name = parse_model_id(model)
    if provider is None:
        return DEFAULT_PROFILE

    try:
        provider_class = infer_provider_class(provider)
    except ValueError:
        return DEFAULT_PROFILE

    try:
        return provider_class.model_profile(model_name) or DEFAULT_PROFILE
    except (ValueError, UserError):
        return DEFAULT_PROFILE


def infer_model(  # noqa: C901
    model: Model | KnownModelName | str, provider_factory: Callable[[str], Provider[Any]] = infer_provider
) -> Model:
    """Infer the model from the name.

    Args:
        model:
            Model name to instantiate, in the format of `provider:model`. Use the string "test" to instantiate TestModel.
        provider_factory:
            Function that instantiates a provider object. The provider name is passed into the function parameter. Defaults to `provider.infer_provider`.
    """
    if isinstance(model, Model):
        return model
    elif model == 'test':
        from .test import TestModel

        return TestModel()

    provider_name, model_name = parse_model_id(model)
    if provider_name is None:
        raise UserError(f'Unknown model: {model}')

    if provider_name == 'vertexai':  # pragma: no cover
        warnings.warn(
            "The 'vertexai' provider name is deprecated. Use 'google-cloud' instead.",
            PydanticAIDeprecationWarning,
        )
        provider_name = 'google-cloud'

    provider = provider_factory(provider_name)

    model_kind = provider_name
    if model_kind.startswith('gateway/'):
        from ..providers.gateway import normalize_gateway_provider

        model_kind = normalize_gateway_provider(model_kind)

    # OpenRouter, Cerebras and Ollama need to be checked before OpenAI,
    # as they are in `OpenAIChatCompatibleProvider` but have their own model classes.
    if model_kind == 'openrouter':
        from .openrouter import OpenRouterModel

        return OpenRouterModel(model_name, provider=provider)
    elif model_kind == 'cerebras':
        from .cerebras import CerebrasModel

        return CerebrasModel(model_name, provider=provider)
    elif model_kind == 'ollama':
        from .ollama import OllamaModel

        return OllamaModel(model_name, provider=provider)
    elif model_kind in ('openai-chat', 'openai', *get_args(OpenAIChatCompatibleProvider.__value__)):
        from .openai import OpenAIChatModel

        if provider_name in ('openai', 'gateway/openai'):
            warnings.warn(
                "In v2.0, 'openai:' will resolve to the OpenAI Responses API by default. "
                "Use 'openai-chat:' to keep current Chat Completions behavior, or "
                "'openai-responses:' to opt in early.",
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
        return OpenAIChatModel(model_name, provider=provider)
    elif model_kind == 'openai-responses':
        from .openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name, provider=provider)
    elif model_kind in ('google', 'google-gla', 'google-vertex', 'google-cloud'):
        from .google import GoogleModel

        return GoogleModel(model_name, provider=provider)
    elif model_kind == 'groq':
        from .groq import GroqModel

        return GroqModel(model_name, provider=provider)
    elif model_kind == 'cohere':
        from .cohere import CohereModel

        return CohereModel(model_name, provider=provider)
    elif model_kind == 'mistral':
        from .mistral import MistralModel

        return MistralModel(model_name, provider=provider)
    elif model_kind == 'anthropic':
        from .anthropic import AnthropicModel

        return AnthropicModel(model_name, provider=provider)
    elif model_kind == 'bedrock':
        from .bedrock import BedrockConverseModel

        return BedrockConverseModel(model_name, provider=provider)
    elif model_kind == 'huggingface':
        from .huggingface import HuggingFaceModel

        return HuggingFaceModel(model_name, provider=provider)
    elif model_kind == 'xai':
        from .xai import XaiModel

        return XaiModel(model_name, provider=provider)
    else:
        raise UserError(f'Unknown model: {model}')  # pragma: no cover


def create_async_http_client(*, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5) -> httpx.AsyncClient:
    """Create an HTTPX async client.

    Each call creates a new client instance. When used via a [`Provider`][pydantic_ai.providers.Provider],
    the client's lifecycle is managed automatically — it will be closed when the provider (or agent) exits.

    The default timeouts match those of OpenAI,
    see <https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9>.
    """
    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


@deprecated('`cached_async_http_client` is deprecated, use `create_async_http_client` instead.')
def cached_async_http_client(
    *, provider: str | None = None, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5
) -> httpx.AsyncClient:
    """Use [`create_async_http_client`][pydantic_ai.models.create_async_http_client] instead."""
    return create_async_http_client(timeout=timeout, connect=connect)


DataT = TypeVar('DataT', str, bytes)


class DownloadedItem(TypedDict, Generic[DataT]):
    """The downloaded data and its type."""

    data: DataT
    """The downloaded data."""

    data_type: str
    """The type of data that was downloaded.

    Extracted from header "content-type", but defaults to the media type inferred from the file URL if content-type is "application/octet-stream".
    """


@overload
async def download_item(
    item: FileUrl,
    data_format: Literal['bytes'],
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[bytes]: ...


@overload
async def download_item(
    item: FileUrl,
    data_format: Literal['base64', 'base64_uri', 'text'],
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[str]: ...


async def download_item(
    item: FileUrl,
    data_format: Literal['bytes', 'base64', 'base64_uri', 'text'] = 'bytes',
    type_format: Literal['mime', 'extension'] = 'mime',
) -> DownloadedItem[str] | DownloadedItem[bytes]:
    """Download an item by URL and return the content as a bytes object or a (base64-encoded) string.

    This function includes SSRF (Server-Side Request Forgery) protection:
    - Only http:// and https:// protocols are allowed
    - Private/internal IP addresses are blocked by default
    - Cloud metadata endpoints (169.254.169.254) are always blocked
    - Hostnames are resolved before requests to prevent DNS rebinding

    Set `item.force_download='allow-local'` to allow private IP addresses.

    Args:
        item: The item to download.
        data_format: The format to return the content in:
            - `bytes`: The raw bytes of the content.
            - `base64`: The base64-encoded content.
            - `base64_uri`: The base64-encoded content as a data URI.
            - `text`: The content as a string.
        type_format: The format to return the media type in:
            - `mime`: The media type as a MIME type.
            - `extension`: The media type as an extension.

    Raises:
        UserError: If the URL points to a YouTube video.
        ValueError: If the URL uses an unsupported protocol or targets a private/internal
            IP address (unless allow-local is set).
    """
    if isinstance(item, VideoUrl) and item.is_youtube:
        raise UserError('Downloading YouTube videos is not supported.')

    from .._ssrf import safe_download

    allow_local = item.force_download == 'allow-local'
    response = await safe_download(item.url, allow_local=allow_local)

    if content_type := response.headers.get('content-type'):
        content_type = content_type.split(';')[0]
        if content_type == 'application/octet-stream':
            content_type = None

    media_type = content_type or item.media_type

    data_type = media_type
    if type_format == 'extension':
        data_type = item.format

    data = response.content
    if data_format in ('base64', 'base64_uri'):
        data = base64.b64encode(data).decode('utf-8')
        if data_format == 'base64_uri':
            data = f'data:{media_type};base64,{data}'
        return DownloadedItem[str](data=data, data_type=data_type)
    elif data_format == 'text':
        return DownloadedItem[str](data=data.decode('utf-8'), data_type=data_type)
    else:
        return DownloadedItem[bytes](data=data, data_type=data_type)


@cache
def get_user_agent() -> str:
    """Get the user agent string for the HTTP client."""
    from .. import __version__

    return f'pydantic-ai/{__version__}'


def _customize_tool_def(transformer: type[JsonSchemaTransformer], tool_def: ToolDefinition):
    """Customize the tool definition using the given transformer.

    If the tool definition has `strict` set to None, the strictness will be inferred from the transformer.
    """
    schema_transformer = transformer(tool_def.parameters_json_schema, strict=tool_def.strict)
    parameters_json_schema = schema_transformer.walk()
    return replace(
        tool_def,
        parameters_json_schema=parameters_json_schema,
        strict=schema_transformer.is_strict_compatible if tool_def.strict is None else tool_def.strict,
    )


def _customize_output_object(transformer: type[JsonSchemaTransformer], output_object: OutputObjectDefinition):
    schema_transformer = transformer(output_object.json_schema, strict=output_object.strict)
    json_schema = schema_transformer.walk()
    return replace(
        output_object,
        json_schema=json_schema,
        strict=schema_transformer.is_strict_compatible if output_object.strict is None else output_object.strict,
    )


@dataclass
class _ToolSearchNativeResolution:
    native_tools: list[AbstractNativeTool]
    keep_search_tools_local: bool


def _resolve_tool_search_native_for_capability_owned_corpus(
    supported_natives: Sequence[AbstractNativeTool], function_tools: Sequence[ToolDefinition]
) -> _ToolSearchNativeResolution:
    """Resolve tool search's native mode when a deferred capability owns a corpus tool.

    Provider-side tool search (Anthropic `bm25`/`regex`, OpenAI server-managed `tool_search`)
    is a black box: it indexes whatever we send and returns matches. It can't honor "this tool
    is only visible after its owning capability has been loaded." Our local search loop in
    `ToolSearchToolset._search_tools` *can* — it filters the corpus by
    `ctx.available_capability_ids`. So whenever a capability-owned tool sits in the corpus,
    search must run client-side or hidden tools will leak.

    Two switches make that happen: (1) flip `ToolSearchTool(strategy=None)` to `'custom'` so
    the adapter wires the client-executed native surface (Anthropic tool-reference blocks,
    OpenAI `execution='client'`) which dispatches into our local `search_tools` callback;
    (2) the caller keeps `search_tools` in the request parameters — that callback is what
    the client-executed surface invokes. Adapters may still render that callback as a
    native client-executed tool-search item rather than as a regular function tool on the
    provider wire. Named-native strategies (`'bm25'`/`'regex'`) have no client-executed
    equivalent, so we raise rather than silently substitute a different algorithm.
    """
    capability_owns_corpus = any(
        t.with_native == ToolSearchTool.kind
        and (t.metadata or {}).get(_deferred_capabilities.DEFERRED_CAPABILITY_TOOL_METADATA_KEY) is True
        for t in function_tools
    )
    if not capability_owns_corpus:
        return _ToolSearchNativeResolution(list(supported_natives), keep_search_tools_local=False)

    resolved_natives: list[AbstractNativeTool] = []
    keep_search_tools_local = False
    for t in supported_natives:
        if not isinstance(t, ToolSearchTool):
            resolved_natives.append(t)
            continue
        if t.strategy not in (None, 'custom'):
            raise UserError(
                f'`ToolSearch(strategy={t.strategy!r})` is incompatible with deferred-loading '
                "capabilities. Server-side strategies can't "
                "honor capability gating and would reveal tools whose owning capability hasn't "
                'been loaded yet. Use `strategy=None` (auto: client-executed local search when a '
                "deferred capability is present), `strategy='keywords'`, or a custom callable."
            )
        keep_search_tools_local = True
        if t.strategy is None:
            t = replace(t, strategy='custom')
        resolved_natives.append(t)
    return _ToolSearchNativeResolution(resolved_natives, keep_search_tools_local=keep_search_tools_local)


def _prepare_return_schemas(params: ModelRequestParameters, profile: ModelProfile) -> ModelRequestParameters:
    """Resolve return schemas: clear on tools that haven't opted in, inject into descriptions for non-native models.

    For tools with `include_return_schema=True` and a non-empty schema, models that natively support
    return schemas keep the schema as-is; other models get it injected into the tool description.
    Tools that haven't opted in have their `return_schema` cleared.
    """
    inject = not profile.supports_tool_return_schema
    resolved: list[ToolDefinition] = []
    changed = False
    for td in params.function_tools:
        if not td.include_return_schema and td.return_schema is not None:
            td = replace(td, return_schema=None)
            changed = True
        elif td.include_return_schema and not td.return_schema:
            warnings.warn(
                f'Tool {td.name!r} has `include_return_schema` enabled but no meaningful return schema'
                f' was generated. Set `include_return_schema=False` on this tool to suppress this warning.',
                UserWarning,
                stacklevel=1,
            )
            td = replace(td, return_schema=None)
            changed = True
        elif inject and td.return_schema:
            parts: list[str] = []
            if td.description:
                parts.append(td.description)
            parts.append('Return schema:')
            parts.append(json.dumps(td.return_schema, indent=2))
            td = replace(td, description='\n\n'.join(parts), return_schema=None)
            changed = True
        resolved.append(td)
    if changed:
        return replace(params, function_tools=resolved)
    return params


def _get_final_result_event(e: ModelResponseStreamEvent, params: ModelRequestParameters) -> FinalResultEvent | None:
    """Return an appropriate FinalResultEvent if `e` corresponds to a part that will produce a final result."""
    if isinstance(e, PartStartEvent):
        new_part = e.part
        if (isinstance(new_part, TextPart) and params.allow_text_output) or (
            isinstance(new_part, FilePart) and params.allow_image_output and isinstance(new_part.content, BinaryImage)
        ):
            return FinalResultEvent(tool_name=None, tool_call_id=None)
        elif isinstance(new_part, ToolCallPart) and (tool_def := params.tool_defs.get(new_part.tool_name)):
            if tool_def.kind == 'output':
                return FinalResultEvent(tool_name=new_part.tool_name, tool_call_id=new_part.tool_call_id)
            elif tool_def.defer:
                return FinalResultEvent(tool_name=None, tool_call_id=None)


def _wrap_non_leading_system_prompts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Wrap `SystemPromptPart`s outside the first `ModelRequest` as `<system>`-tagged `UserPromptPart`s.

    `SystemPromptPart`s in the first `ModelRequest` aren't transformed; the provider's `_map_messages` hoists them.
    Returns the original list when nothing changed so the identity check in `_make_request` can skip the
    redundant `_clean_message_history` pass.
    """
    first_request_idx = next(
        (i for i, m in enumerate(messages) if isinstance(m, ModelRequest)),
        None,
    )
    if first_request_idx is None:
        return messages

    new_messages: list[ModelMessage] = list(messages[: first_request_idx + 1])
    changed = False
    for msg in messages[first_request_idx + 1 :]:
        if isinstance(msg, ModelRequest) and any(isinstance(p, SystemPromptPart) for p in msg.parts):
            new_parts = [
                UserPromptPart(content=f'<system>{part.content}</system>', timestamp=part.timestamp)
                if isinstance(part, SystemPromptPart)
                else part
                for part in msg.parts
            ]
            new_messages.append(replace(msg, parts=new_parts))
            changed = True
        else:
            new_messages.append(msg)

    return new_messages if changed else messages
