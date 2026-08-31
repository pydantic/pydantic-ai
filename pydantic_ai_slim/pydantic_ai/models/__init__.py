"""Logic related to making requests to an LLM.

The aim here is to make a common interface for different LLMs, so that the rest of the code can be agnostic to the
specific LLM being used.
"""

from __future__ import annotations as _annotations

import base64
import hashlib
import json
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from difflib import get_close_matches
from functools import cache, cached_property
from types import TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Literal, TypeVar, cast, get_args, overload

import httpx2
from typing_extensions import Self, TypeAliasType, TypedDict, deprecated
from typing_inspection.introspection import get_literal_values

from .. import _utils
from .._cost import preload_pricing_data
from .._http import DEFAULT_HTTP_TIMEOUT as DEFAULT_HTTP_TIMEOUT, legacy_httpx
from .._json_schema import JsonSchemaTransformer
from .._output import StructuredTextOutputSchema
from .._parts_manager import ModelResponsePartsManager
from .._run_context import RunContext
from .._warnings import PydanticAIDeprecationWarning as PydanticAIDeprecationWarning
from ..exceptions import UserError
from ..messages import (
    STANDING_PROMPT_PLANTED_KEY,
    BaseToolCallPart,
    BaseToolReturnPart,
    BinaryImage,
    CompactionPart,
    FilePart,
    FileUrl,
    FinalResultEvent,
    FinishReason,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponseState,
    ModelResponseStreamEvent,
    NativeToolSearchReturnPart as NativeToolSearchReturnPart,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
    _compaction_part_is_wire_boundary,  # pyright: ignore[reportPrivateUsage]
    _tool_results_first_sort_key,  # pyright: ignore[reportPrivateUsage]
)
from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from ..native_tools._tool_search import TOOL_SEARCH_FUNCTION_TOOL_NAME, ToolSearchTool
from ..output import OutputMode, OutputObjectDefinition, StructuredOutputMode
from ..profiles import (
    DEFAULT_PROFILE,
    DEFAULT_PROMPTED_OUTPUT_TEMPLATE,
    ModelProfile,
    ModelProfileSpec,
    ToolAdditionMode,
    ToolDeferralMode,
    _translate_legacy_profile_keys,  # pyright: ignore[reportPrivateUsage]
    merge_profile,
)
from ..providers import InterfaceClient, Provider, infer_provider, infer_provider_class
from ..settings import ModelSettings, ThinkingLevel, merge_model_settings
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._abstract import AbstractModel as AbstractModel
from ._known_model_names import KnownModelName as KnownModelName

if TYPE_CHECKING:
    from httpx import AsyncClient

    from ..agent.abstract import AbstractAgent
    from ..usage import RunUsage
else:
    # Legacy HTTPX is optional, so this module has to import without it. The annotation then degrades
    # to `object`, dropping static checking of what `create_async_http_client` hands back — the client
    # its caller owns and closes. Calling it without legacy HTTPX still raises (see its body).
    AsyncClient = legacy_httpx.AsyncClient if legacy_httpx is not None else object

_MAX_FILE_URL_DOWNLOAD_BYTES = 50 * 1024 * 1024
"""Default maximum response body size when downloading a [`FileUrl`][pydantic_ai.messages.FileUrl]."""


ModelContextDepsT = TypeVar('ModelContextDepsT')


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
        'crusoe',
        'deepseek',
        'fireworks',
        'github',
        'heroku',
        'litellm',
        'moonshotai',
        'nebius',
        'ollama',
        'openrouter',
        'ovhcloud',
        'sambanova',
        'snowflake',
        'together',
        'vercel',
        'vllm',
        'zai',
    ],
)
OpenAIResponsesCompatibleProvider = TypeAliasType(
    'OpenAIResponsesCompatibleProvider',
    Literal[
        'azure',
        'deepseek',
        'fireworks',
        'nebius',
        'openrouter',
        'ovhcloud',
        'sambanova',
        'together',
    ],
)

ToolVisibility = Literal['visible', 'deferred', 'withheld', 'via_history']
"""How a function tool is represented on the request a provider actually receives.

- `'visible'`: an ordinary entry in the provider's `tools` collection, schema included.
- `'deferred'`: a declared `tools` entry whose schema is withheld behind the provider's
  schema-deferral flag until something reveals it.
- `'withheld'`: absent from the request entirely.
- `'via_history'`: absent from the `tools` collection; the full definition travels on the
  provider's mid-conversation tool-addition channel instead.

Resolved per tool name into [`ModelRequestParameters.tool_visibility`][pydantic_ai.models.ModelRequestParameters.tool_visibility]."""


@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and output handling."""

    function_tools: list[ToolDefinition] = field(default_factory=list[ToolDefinition])
    native_tools: list[AbstractNativeTool] = field(default_factory=list[AbstractNativeTool])
    tool_visibility: dict[str, ToolVisibility] | None = None
    """Maps each function tool name to its resolved [`ToolVisibility`][pydantic_ai.models.ToolVisibility].

    `None` on authored parameters; [`Model.prepare_request`][pydantic_ai.models.Model.prepare_request]
    populates an entry for every function tool, so a resolved request always carries a dict — empty
    exactly when there are no function tools. Output tools never get entries because they are always
    plain `tools` entries; [`visibility_of`][pydantic_ai.models.ModelRequestParameters.visibility_of]
    treats their absent entries like `'visible'`. The no-defaults `repr` omits the field until
    resolution, so authored parameters print as authored and resolved state stays visible.
    """
    revealed_tool_names: set[str] = field(default_factory=set[str], repr=False)
    """Names history has revealed so far, derived from the outgoing message list before each request.

    Discovered means evidenced by history; revealed means represented on this request's wire state.

    Input to visibility resolution: `ToolDefinition.defer_loading` records what the author asked
    for and stays set after a reveal, so this answers the separate question of what the model can
    see *now*. History can name tools that no longer exist in the current run's definitions; those
    are dropped where this is derived, so it is a subset of `function_tools`' names by construction.
    """

    deferred_capability_ids: set[str] = field(default_factory=set[str], repr=False)
    """IDs of the run's capabilities that defer their loading.

    Read from the capability instances themselves, so it means what it says. It cannot be derived
    from the function tools: `ToolDefinition.capability_id` records which capability *contributed* a
    tool, and `defer_loading` is set both by a deferred capability and by a search-gated tool inside
    an always-on one — so the two cases are indistinguishable from the definitions alone.

    Used to answer "may this tool be revealed yet?": a tool whose `capability_id` is in this set is
    gated on that capability being loaded, while one whose owner is absent here is gated only on its
    own discovery.
    """

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

    def visibility_of(self, tool_name: str) -> ToolVisibility:
        """The resolved [`ToolVisibility`][pydantic_ai.models.ToolVisibility] for `tool_name`.

        For parameters constructed directly rather than resolved by [`Model.prepare_request`][pydantic_ai.models.Model.prepare_request],
        deferred function tools default to `'withheld'` and every other name defaults to `'visible'`.
        """
        if visibility := (self.tool_visibility or {}).get(tool_name):
            return visibility
        # `tool_defs` is a cached dict, so the fallback stays O(1) — adapters call this in
        # per-tool loops.
        tool_def = self.tool_defs.get(tool_name)
        return 'withheld' if tool_def is not None and tool_def.defer_loading else 'visible'

    @cached_property
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {tool_def.name: tool_def for tool_def in [*self.function_tools, *self.output_tools]}

    @cached_property
    def declared_tool_defs(self) -> dict[str, ToolDefinition]:
        """Definitions represented in the provider's ordinary `tools` collection.

        The visibility filter applies to function tools only: output tools are always plain
        `tools` entries, so they are included unconditionally rather than keyed through a
        name-indexed filter a hidden function tool could shadow.
        """
        return {tool_def.name: tool_def for tool_def in [*self.declared_function_tools, *self.output_tools]}

    @cached_property
    def declared_function_tools(self) -> list[ToolDefinition]:
        """Function tools represented in the provider's ordinary `tools` collection."""
        return [
            tool for tool in self.function_tools if self.visibility_of(tool.name) not in ('withheld', 'via_history')
        ]

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


@dataclass(kw_only=True)
class ModelRequestContext:
    """Context for model request hooks.

    Wrapping these parameters in a dataclass instead of a tuple makes the signature
    future-proof: new fields can be added without breaking existing implementations.

    A [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] hook
    returns a context, so modifying one is normally `dataclasses.replace(request_context, ...)`. Every
    field is therefore settable through `replace()`, including `model_id` and `streaming`: the agent
    graph sets those two immediately before calling the hook, and `replace()` re-initializes any
    `init=False` field to its default, so declaring them that way silently zeroed both for any hook
    that copied its context — costing a streamed run its `streaming` flag and a durable-execution
    worker the selection token it re-resolves an aliased model from. They are still set by the graph
    rather than by a caller; passing either to a fresh `ModelRequestContext` is not meaningful.
    """

    model: Model
    messages: list[ModelMessage]
    model_settings: ModelSettings | None

    model_request_parameters: ModelRequestParameters
    """The tool, output, and instruction configuration for this request.

    [`instruction_parts`][pydantic_ai.models.ModelRequestParameters.instruction_parts] is the source of
    truth for the instructions in the agent flow: a
    [`before_model_request`][pydantic_ai.capabilities.AbstractCapability.before_model_request] hook that
    rewrites them changes what the model receives, and the [`ModelRequest`][pydantic_ai.messages.ModelRequest]
    recorded in message history is re-rendered from them afterwards, so history and traces keep showing
    what was sent. Assigning to that message's `instructions` instead is not propagated back into the
    parts, and so does not reach the model.
    """

    model_id: str | None = None
    """The model-name string this request's model was selected/resolved from, if any.

    This is the *selection* token — e.g. `'openai:gpt-5.6-sol'`, or an alias like `'tenant-x'` that a
    [`resolve_model_id`][pydantic_ai.capabilities.AbstractCapability.resolve_model_id] capability
    turned into a concrete model — so it can differ from the resolved model's own
    [`model_id`][pydantic_ai.models.Model.model_id]. `None` when the model was supplied as an
    instance rather than resolved from a string.

    Durable-execution capabilities carry this across the activity/step/task boundary in preference
    to the resolved model's own `model_id`, so an aliased model round-trips as the original string
    the worker-side resolution chain can re-resolve. Only meaningful while `model` is still the run's
    resolved model — a model swapped in by a hook invalidates it.
    """

    streaming: bool = False
    """Whether the agent loop expects to iterate the model response as a stream.

    Set for streamed runs — `run_stream()`, `run_stream_events()`, `iter()`'s node streaming — and
    for `run()` when an `event_stream_handler` is set or a capability overrides
    `wrap_run_event_stream` (e.g. `ProcessEventStream`, or a durability capability's
    `event_stream_handler=`). There is no separate `before_model_request_stream` hook — streaming
    and non-streaming requests share the same hooks — so this field is how a hook can tell them
    apart. Read-only from hooks: reassigning it doesn't change how the loop consumes the response.
    """


@dataclass(frozen=True, kw_only=True)
class ModelResolutionContext(Generic[ModelContextDepsT]):
    """Context used to resolve a model ID before a model is available.

    This is narrower than [`RunContext`][pydantic_ai.tools.RunContext] because model
    resolution happens before a run context can contain its resolved model.
    """

    agent: AbstractAgent[ModelContextDepsT, Any]
    """The agent whose model is being resolved."""

    deps: ModelContextDepsT
    """The dependencies supplied for this run."""


@dataclass(frozen=True, kw_only=True)
class ModelSelectionContext(ModelResolutionContext[ModelContextDepsT]):
    """Context used by a capability to select the model for a request step."""

    model: Model | None
    """The lower-precedence model on the first step, then the model used for the previous step."""

    run_step: int
    """The request step being selected, starting at `1`."""

    messages: list[ModelMessage]
    """The message history available before this request step."""

    usage: RunUsage
    """Usage accumulated by the run before this request step."""


class Model(AbstractModel, Generic[InterfaceClient]):
    """Abstract class for a model."""

    supported_tool_deferral_modes: ClassVar[frozenset[ToolDeferralMode]] = frozenset()
    """`tool_deferral_mode` values this adapter's renderer implements.

    A profile may claim a mode for the model family, but the claim only takes effect when the
    adapter class declares it here: `Model.tool_deferral_mode` intersects the two, so a `Model`
    subclass that declares nothing (the default) never resolves tools to a wire shape it cannot
    render, no matter what a pass-through vendor profile claims.
    """
    supported_tool_addition_modes: ClassVar[frozenset[ToolAdditionMode]] = frozenset()
    """`tool_addition_mode` values this adapter's renderer implements. See `supported_tool_deferral_modes`."""
    compaction_requires_encrypted_content: ClassVar[bool] = False
    """Whether this adapter's API only honors a [`CompactionPart`][pydantic_ai.messages.CompactionPart]
    that carries encrypted content.

    When set, a part without it isn't a wire boundary: the adapter would omit it, so letting it hide
    the earlier history would send nothing in its place.

    Declared by the adapter rather than the model profile: how an API carries compaction state is a
    property of the API, not of the model behind it — the same model reached through OpenAI's Chat
    Completions and Responses APIs answers differently, and eight providers route a profile of their
    own through `OpenAIResponsesModel`. Independent of `compaction_retains_standing_prompt`, which
    today's two adapters happen to answer the same way."""
    compaction_retains_standing_prompt: ClassVar[bool] = False
    """Whether this adapter's compaction item keeps serving the leading system items of the window
    it replaced.

    When set, re-sending the standing prompt after the boundary would duplicate it. When not (the
    default), the standing prompt travels in a per-request channel rebuilt from those items, so the
    trim has to re-insert them or it is silently dropped from every subsequent request. See
    `compaction_requires_encrypted_content` for why this is declared here and not on the profile."""

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
        preload_pricing_data()

    @property
    def provider(self) -> Provider[InterfaceClient] | None:
        """The provider for this model, if any."""
        return getattr(self, '_provider', None)

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

    def resolve_prompt_cache_retention(self, model_settings: ModelSettings | None) -> timedelta | None:
        """Resolve prompt cache retention requested by provider-specific model settings.

        The model's default settings are merged with the per-request `model_settings`. Only provider-specific settings
        are currently considered; a future unified cache setting is not yet an input. If multiple active settings
        request different retention periods, the longest period wins because any longer-lived cache breakpoint can
        keep the corresponding prompt prefix available. Models without a provider-specific retention setting return
        `None`.
        """
        return None

    @staticmethod
    def _max_prompt_cache_retention(
        *cache_settings: bool | Literal['5m', '1h'] | None,
    ) -> timedelta | None:
        if '1h' in cache_settings:
            return timedelta(hours=1)
        if any(cache_settings):
            return timedelta(minutes=5)
        return None

    @property
    def tool_deferral_mode(self) -> ToolDeferralMode | None:
        """The effective schema-deferral mode: the profile's claim, if this adapter renders it."""
        mode = self.profile.get('tool_deferral_mode')
        return mode if mode in self.supported_tool_deferral_modes else None

    @property
    def tool_addition_mode(self) -> ToolAdditionMode | None:
        """The effective tool-addition mode: the profile's claim, if this adapter renders it."""
        mode = self.profile.get('tool_addition_mode')
        return mode if mode in self.supported_tool_addition_modes else None

    def _trim_before_compaction(
        self,
        messages: list[ModelMessage],
        *,
        standing_prompt_retained: bool | None = None,
    ) -> list[ModelMessage]:
        """Drop history before the latest compaction boundary this adapter's API honors.

        Called only by adapters that render `CompactionPart`s on the wire, and the one place their
        declared `compaction_*` facts are turned into trim behavior — so an adapter states what its
        API does rather than what to do about it. See `_trim_messages_before_compaction` for what
        the trim preserves.

        `standing_prompt_retained` defaults to `compaction_retains_standing_prompt`. A caller passes
        an explicit value where its window is not an ordinary one: re-compaction plants the standing
        prompt afresh, since retention decays across a second compaction.
        """
        return _trim_messages_before_compaction(
            messages,
            self.system,
            requires_encrypted_content=self.compaction_requires_encrypted_content,
            standing_prompt_retained=self.compaction_retains_standing_prompt
            if standing_prompt_retained is None
            else standing_prompt_retained,
        )

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

    async def cancel_suspended_response(self, response: ModelResponse) -> None:
        """Cancel a server-side suspended/background response (e.g. an OpenAI background job).

        Called when a continuation is abandoned via cancellation or error. No-op by default;
        model classes with cancellable server-side jobs override this.
        """
        return None

    def continuation_delay(self, response: ModelResponse) -> float | None:
        """Seconds to wait before continuing a suspended response, or `None` to continue immediately.

        Called between the segments of a suspended turn. `None` by default (e.g. Anthropic `pause_turn`
        continues immediately); a model that polls a server-side job (e.g. OpenAI background mode)
        overrides this to return a poll interval so the graph doesn't busy-poll.
        """
        return None

    def customize_request_parameters(self, model_request_parameters: ModelRequestParameters) -> ModelRequestParameters:
        """Customize the request parameters for the model.

        This method can be overridden by subclasses to modify the request parameters before sending them to the model.
        In particular, this method can be used to make modifications to the generated tool JSON schemas if necessary
        for vendor/model-specific reasons.
        """
        if transformer := self.profile.get('json_schema_transformer'):
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
        params = prepare_return_schemas(
            params, supports_tool_return_schema=self.profile.get('supports_tool_return_schema', False)
        )

        # Resolve unified thinking setting and strip from model_settings
        if model_settings and 'thinking' in model_settings:
            thinking_value = model_settings['thinking']
            supports_thinking = self.profile.get('supports_thinking', False)
            thinking_always_enabled = self.profile.get('thinking_always_enabled', False)
            if supports_thinking or thinking_always_enabled:
                if not (thinking_value is False and thinking_always_enabled):
                    params = replace(params, thinking=thinking_value)
            stripped = {k: v for k, v in model_settings.items() if k != 'thinking'}
            model_settings = cast(ModelSettings, stripped) if stripped else None

        if native_tools := params.native_tools:
            # Deduplicate native tools
            params = replace(
                params,
                native_tools=list({tool.unique_id: tool for tool in native_tools}.values()),
            )

        params = params.with_default_output_mode(self.profile.get('default_structured_output_mode', 'tool'))

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
            or (
                params.output_mode == 'native'
                and self.profile.get('native_output_requires_schema_in_instructions', False)
            )
        ) and params.prompted_output_template is None:
            params = replace(
                params,
                prompted_output_template=self.profile.get('prompted_output_template', DEFAULT_PROMPTED_OUTPUT_TEMPLATE),
            )

        # Append prompted_output_instructions to instruction_parts so models that use structured
        # instruction parts (for per-part system messages or cache placement) also get them.
        # Done here (after customize_request_parameters) so it uses the final resolved template.
        if output_instr := params.prompted_output_instructions:
            parts = [*(params.instruction_parts or []), InstructionPart(content=output_instr)]
            params = replace(params, instruction_parts=InstructionPart.sorted(parts))

        # Check if output mode is supported
        if params.output_mode == 'native' and not self.profile.get('supports_json_schema_output', False):
            raise UserError('Native structured output is not supported by this model.')
        if params.output_mode == 'tool' and not self.profile.get('supports_tools', True):
            raise UserError('Tool output is not supported by this model.')
        if params.allow_image_output and not self.profile.get('supports_image_output', False):
            raise UserError('Image output is not supported by this model.')

        # Check native tools, handle fallback swap, and resolve deferred-tool visibility. A deferred
        # tool has to get here on its own account: one gated by an on-demand capability belongs to no
        # native tool's corpus, so a run whose deferred tools are all capability-gated reaches this
        # point with neither a native tool nor a `with_native` between them.
        if params.native_tools or any(
            t.unless_native or t.with_native or t.defer_loading for t in params.function_tools
        ):
            params = self._resolve_request_tools(params)
        else:
            # Nothing native and nothing deferred: every function tool is plainly visible. Stamped
            # unconditionally so `tool_visibility` is a dict — `None` means unresolved — after
            # `prepare_request` on this path too, not only when the full swap resolution runs.
            params = replace(
                params,
                tool_visibility={t.name: 'visible' for t in params.function_tools},
            )

        return model_settings, params

    def prepare_messages(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters | None = None,
    ) -> list[ModelMessage]:
        """Pre-process the message history before it's handed to the adapter's message-prep step.

        Translates typed `NativeToolSearch*Part` instances carried over from a
        different provider (e.g. Anthropic to OpenAI Responses), or any native
        provider when the active model doesn't support `ToolSearchTool`, into the
        local-shape `ToolSearch*Part` instances. This splits the single
        `ModelResponse(call+return)` carrying the inline server-side result into
        `ModelResponse(call) + ModelRequest(return)` so the adapter can render the
        provider-agnostic exchange.

        Also wraps non-leading `SystemPromptPart`s as `<system>`-tagged `UserPromptPart`s when
        the profile's `supports_inline_system_prompts` is `False`, and converts
        `SpeechPart`s from realtime session history into `UserPromptPart`s /
        `TextPart`s that any model can consume.

        Subclasses normally don't need to override this; the framework calls it on the
        agent's behalf in `_agent_graph._make_request` so per-adapter message-prep code
        sees a homogeneous shape regardless of which provider produced the prior turn.

        Args:
            messages: The history to pre-process.
            model_request_parameters: The parameters this history will be sent with. Optional, and
                only needed to render a `ToolAvailabilityDeltaPart` on a model with no native way to
                express one: whether that reveal has to be a mechanism or can just be a statement
                depends on whether any tool actually goes on the wire with its schema withheld, which
                the profile alone can't answer. Omitting it falls back to the adapter's effective mode,
                which differs only for a corpus mixing capability-gated and standalone deferred tools.
                Framework callers pass it.
        """
        messages = _convert_speech_parts(messages, include_audio=self.profile.get('supports_audio_input', False))

        supports_tool_addition = self.tool_addition_mode is not None
        messages = self._translate_legacy_tool_reveals(messages, model_request_parameters)
        delta_parts = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, ToolAvailabilityDeltaPart)
        ]
        supports_native_tool_search = ToolSearchTool in self.profile.get(
            'supported_native_tools', SUPPORTED_NATIVE_TOOLS
        )
        if delta_parts and not supports_tool_addition:
            # `None` means "no definitions to validate against, render as recorded": the bare
            # `prepare_messages(messages)` form has no parameters, and filtering everything there
            # would erase legitimate announcements. The agent path always passes parameters, so
            # only names matching a currently-served *authored-deferred* function tool render —
            # a forged-but-well-shaped name must not reach system voice, and an always-visible
            # tool named by a delta has no "now available" news and no exchange to fabricate:
            # the delta is a no-op for it on this channel just as on the native ones.
            deferred_tool_names = (
                {tool.name for tool in model_request_parameters.function_tools if tool.defer_loading}
                if model_request_parameters is not None
                else None
            )
            # Two different jobs hide behind "render the delta", and which applies turns on whether this
            # model can withhold a tool's schema at all.
            #
            # Where it can, the revealed tool is already on the wire with `'deferred'` visibility, and the
            # tool-search exchange is what takes the flag off again: Anthropic renders the return as the
            # `tool_reference` block that unhides the schema. Announcing the change in prose there would
            # leave the tool hidden for good, which `test_anthropic_defer_loading_needs_a_reveal_mechanism`
            # pins as "the reveal and the flag travel together".
            #
            # Where it can't, the tool has `'visible'` visibility from the turn it's revealed and the
            # exchange carries no mechanism, only the news. Stating that beats fabricating a
            # `search_tools` call the model never made, and beats naming a `search_tools` tool the
            # corpus-empty drop may have removed from the wire entirely.
            #
            # "Can withhold a schema" is narrower than "has native tool search". OpenAI has tool search
            # but rejects `defer_loading` without a `tool_search` tool on the wire, and a capability-only
            # corpus has nothing to put there — so its gated tools aren't declared until revealed, and
            # arrive visible. Anthropic takes `defer_loading` with no search surface at all, so its gated
            # tools do arrive hidden and do need the reveal.
            if self._hides_deferred_schemas(model_request_parameters):
                messages = _synthesize_tool_availability_delta_messages(messages, deferred_tool_names)
            else:
                messages = _announce_tool_availability_delta_messages(messages, deferred_tool_names)

        from .._tool_search import synthesize_local_tool_search_messages

        target_provider_name = self.system if supports_native_tool_search else None
        messages = synthesize_local_tool_search_messages(messages, target_provider_name=target_provider_name)

        if not self.profile.get('supports_inline_system_prompts', False):
            messages = _wrap_non_leading_system_prompts(messages)

        return messages

    def _translate_legacy_tool_reveals(
        self,
        messages: list[ModelMessage],
        model_request_parameters: ModelRequestParameters | None,
    ) -> list[ModelMessage]:
        """Upgrade framework-fabricated legacy reveal exchanges onto this model's reveal channel.

        Pre-delta lazy-capability code stored a fabricated `search_tools` exchange after each
        `load_capability` call. Where this model has a native reveal channel, that fabrication is
        upgraded to the availability delta it always represented, and renders as `tool_addition` /
        `additional_tools`. On a channel-less target the exchange already replays byte-stably as
        plain tool parts and the revealed tool reaches the wire regardless (deferred entry or
        visible definition), so it is left alone — translating would change the replayed prefix
        for no gain. Deciding on the adapter's effective mode alone also keeps this from resolving native tools
        here, which would preempt `prepare_request`'s more specific unsupported-tool errors.

        Genuine search exchanges — native or local, from any provider — are never rewritten: a
        real search is evidence of what the model did, and the cross-provider local-search
        projection already carries its reveal. This changes only the outgoing copy; stored history
        remains untouched.
        """
        if model_request_parameters is None or self.tool_addition_mode is None:
            return messages

        translated_call_ids = _legacy_fabricated_tool_search_reveals(messages, model_request_parameters)
        if not translated_call_ids:
            return messages

        return _replace_tool_search_exchanges_with_deltas(messages, translated_call_ids)

    def _hides_deferred_schemas(self, params: ModelRequestParameters | None) -> bool:
        """Whether this request puts a tool on the wire with its schema withheld."""
        if params is None:
            return self.tool_deferral_mode == 'standalone'
        # Mirrors `prepare_request`'s guard so this can't raise where that wouldn't: with nothing
        # native and nothing deferred there is no schema to withhold anyway.
        if not (
            params.native_tools
            or any(t.unless_native or t.with_native or t.defer_loading for t in params.function_tools)
        ):
            return False
        # TODO(#7196): reorder the stages so message projection always receives resolved
        # parameters, at which point this on-demand resolution can be removed.
        resolved = params if params.tool_visibility is not None else self._resolve_request_tools(params)
        return any(visibility == 'deferred' for visibility in (resolved.tool_visibility or {}).values())

    def _resolve_request_tools(self, params: ModelRequestParameters) -> ModelRequestParameters:
        """Resolve native tools, their local fallbacks, and deferred-tool visibility for this model."""
        return resolve_request_tools(
            params,
            self.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS),
            can_withhold_tool_schemas=self._can_withhold_tool_schemas,
            tool_addition_mode=self.tool_addition_mode,
        )

    def _can_withhold_tool_schemas(self, native_tools: Sequence[AbstractNativeTool]) -> bool:
        """Whether this request can declare a function tool while withholding its schema.

        `'standalone'` always permits it. `'with_tool_search'` permits it only when a
        [`ToolSearchTool`][pydantic_ai.native_tools.ToolSearchTool] survives request resolution.
        The result feeds the single `tool_visibility` decision table; `defer_loading` is unchanged.
        """
        tool_deferral_mode = self.tool_deferral_mode
        if tool_deferral_mode == 'standalone':
            return True
        if tool_deferral_mode == 'with_tool_search':
            return any(isinstance(t, ToolSearchTool) for t in native_tools)
        return False

    @classmethod
    def supported_native_tools(cls) -> frozenset[type[AbstractNativeTool]]:
        """Return the set of native tool types this model class can handle.

        Subclasses should override this to reflect their actual capabilities.
        Default is empty set - subclasses must explicitly declare support.
        """
        return frozenset()

    @cached_property
    def profile(self) -> ModelProfile:
        """The model profile.

        Resolution order (later layers override earlier ones):
          1. `DEFAULT_PROFILE` — base values for every key in `ModelProfile`.
          2. The provider's `model_profile(model_name)` result — provider-specific defaults
             for this model.
          3. The user's `profile=` argument — partial dict merged on top, OR a callable
             `(default) -> profile` for full control.

        After resolution we compute the intersection of the profile's `supported_native_tools`
        and the model class's implemented tools, ensuring `model.profile['supported_native_tools']`
        is the single source of truth for what's actually usable.
        """
        # Step 1+2: provider default merged with base default
        provider_profile: ModelProfile = {}
        if (provider := self.provider) is not None:
            provider_profile = provider.model_profile(self.model_name) or {}
        resolved = merge_profile(DEFAULT_PROFILE, provider_profile)

        # Step 3: user override
        user = self._profile
        if user is None:
            pass
        elif callable(user):
            # The callable form's result bypasses `merge_profile`, so translate deprecated key
            # spellings here too.
            resolved = _translate_legacy_profile_keys(user(resolved))
        else:
            # Partial dict — merge on top
            resolved = merge_profile(resolved, user)

        # Step 4: native tools intersection — profile's allowed tools & model's implemented tools
        model_supported = self.__class__.supported_native_tools()
        profile_supported = resolved.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)
        effective_tools = profile_supported & model_supported
        if effective_tools != profile_supported:
            resolved = merge_profile(resolved, ModelProfile(supported_native_tools=effective_tools))

        return resolved

    def _validate_uploaded_file_provider(self, item: UploadedFile) -> None:
        """Raise `UserError` if an `UploadedFile` references a different provider than this model."""
        if item.provider_name != self.system:
            raise UserError(
                f'UploadedFile with `provider_name={item.provider_name!r}` cannot be used with {type(self).__name__}. '
                f'Expected `provider_name` to be `{self.system!r}`.'
            )

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
    state: ModelResponseState = field(default='complete', init=False)
    """Lifecycle state of the response."""
    metadata: dict[str, Any] | None = field(default=None, init=False)

    _event_iterator: AsyncIterator[ModelResponseStreamEvent] | None = field(default=None, init=False)
    _usage: RequestUsage = field(default_factory=RequestUsage, init=False)
    _cancelled: bool = field(default=False, init=False)
    _finished: bool = field(default=False, init=False)
    _first_chunk_monotonic: float | None = field(default=None, init=False)
    """`time.perf_counter()` stamped on the first event surfaced to the consumer, or `None` if nothing
    was yielded; surfaced as a duration by the `time_to_first_chunk` method."""

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
                        if self._first_chunk_monotonic is None:
                            # First event surfaced to the consumer: stamp the monotonic clock.
                            self._first_chunk_monotonic = time.perf_counter()
                        yield event
                except self.get_stream_cancel_errors():
                    if not self.cancelled:
                        raise
                else:
                    # Only natural `StopAsyncIteration` on a stream that wasn't
                    # cancelled flips `_finished`. Early `break` / `aclose()` (raising
                    # `GeneratorExit` at the suspended `yield`) and any in-flight error
                    # leave `_finished=False` so `get()` reports the truncated response
                    # as `'incomplete'` rather than silently stamping it `'complete'`.
                    # A `cancel()` mid-stream that still drains to a natural completion
                    # (e.g. a local model with no live connection to tear down) must not
                    # be recorded as finished either: `_cancelled` wins so `get()`
                    # reports `'interrupted'`. A defensive `cancel()` *after* the stream
                    # already finished naturally leaves `_finished=True` (set here before
                    # `_cancelled`), so `get()` keeps `'complete'`.
                    if not self._cancelled:
                        self._finished = True

            self._event_iterator = iterator_with_cancel_guard(
                iterator_with_part_end(iterator_with_final_event(self._get_event_iterator()))
            )
        return self._event_iterator

    async def cancel(self) -> None:
        """Cancel local stream consumption and request provider shutdown.

        Sets `self._cancelled = True` before delegating to `close_stream()`
        so the flag is visible to any iterator that observes the transport error
        raised when the underlying connection is torn down, even if
        `close_stream()` itself raises.
        """
        if self.cancelled:
            return
        self._cancelled = True
        # A stream that finished naturally stays 'complete': get() checks _finished
        # before _cancelled, and there's no live connection left to tear down.
        if self._finished:
            return
        await self.close_stream()

    def get_stream_cancel_errors(self) -> tuple[type[BaseException], ...]:
        """Return transport errors caused by `cancel()` tearing down the stream.

        The default covers model classes whose SDKs iterate HTTP responses
        directly (Anthropic, OpenAI, Groq, Mistral, Google GenAI, and HuggingFace),
        since they let bare `httpx2` (or legacy `httpx`) errors propagate from
        chunk reads. Model classes that use other transports (for example gRPC or
        botocore) should override this method.
        """
        try:
            import httpx
        except ImportError:
            return (httpx2.StreamError, httpx2.TransportError)

        return (httpx2.StreamError, httpx2.TransportError, httpx.StreamError, httpx.TransportError)

    async def close_stream(self) -> None:
        """Close the provider stream and any exposed HTTP or gRPC transport.

        Model classes must override this to close the local stream and, where the
        provider SDK exposes one, its transport. Integrations that cannot support
        local cancellation should leave the default implementation so `cancel()`
        fails clearly.
        """
        raise NotImplementedError(
            f'Stream cancellation is not implemented for {type(self).__name__}. '
            'This model class must override `close_stream()` to support streaming cancellation.'
        )

    # TODO: We should not have public private methods which need to be overwritten.
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
        # `'suspended'` is the one state a provider stamps that `get()` can't otherwise derive, so it wins.
        # A finished iteration only means `'complete'` if the provider didn't leave an explicit `'incomplete'`
        # hint (e.g. a foreground OpenAI Responses stream that EOF'd without a terminal event). An explicit
        # `cancel()` outranks that in-flight `'incomplete'` hint, so a cancelled foreground stream reports
        # `'interrupted'` rather than `'incomplete'`.
        state: ModelResponseState
        if self.state == 'suspended':
            state = 'suspended'
        elif self._finished and self.state != 'incomplete':
            state = 'complete'
        elif self._cancelled:
            state = 'interrupted'
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
            metadata=self.metadata,
        )

    @property
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

    def time_to_first_chunk(self, request_start: float) -> float | None:
        """Seconds from `request_start` to the first chunk surfaced to the consumer, or `None` if nothing was yielded.

        `request_start` must be a `time.perf_counter()` reading taken when the request was issued.
        The first-chunk instant is stamped on the first `async for` pull, so the result reflects when
        the consumer *received* the first event: it includes any consumer-side iteration delay
        (debouncing, batching, or awaiting other work) on top of the chunk's transit time, which for
        eager consumers is negligible.
        """
        first_chunk = self._first_chunk_monotonic
        return first_chunk - request_start if first_chunk is not None else None


class CompletedStreamedResponse(StreamedResponse):
    """A `StreamedResponse` that wraps an already-completed `ModelResponse`.

    Used when a [`StreamedResponse`][pydantic_ai.models.StreamedResponse] is needed but no
    live stream is available — for example, when an agent run is short-circuited by
    [`SkipModelRequest`][pydantic_ai.exceptions.SkipModelRequest], when a capability's
    [`wrap_model_request`][pydantic_ai.capabilities.AbstractCapability.wrap_model_request]
    short-circuits without calling the handler, or when a durable-execution capability drains
    the real stream inside an activity/step/task and only surfaces the final
    [`ModelResponse`][pydantic_ai.messages.ModelResponse] to the workflow.

    What the stream yields is controlled by `replay_events`:

    - `False` (default): yield no events — the response is complete and no streaming
      consumer needs to observe it.
    - `True`: synthesize `PartStartEvent` + `PartDeltaEvent` sequences from the response
      parts, so streaming consumers (`event_stream_handler`, `run_stream_events`, ...)
      keep working when only a complete `ModelResponse` exists.
    - a list of events: replay events that were captured off the live stream elsewhere
      (e.g. inside a durable-execution activity/step/task), preserving the real
      event granularity.
    """

    @overload
    def __init__(
        self,
        response: ModelResponse,
        *,
        model_request_parameters: ModelRequestParameters,
        replay_events: bool | list[ModelResponseStreamEvent] = False,
    ) -> None: ...

    @overload
    @deprecated('Pass the response first and `model_request_parameters` as a keyword argument.')
    def __init__(
        self,
        model_request_parameters: ModelRequestParameters,
        response: ModelResponse,
        /,
        *,
        replay_events: bool | list[ModelResponseStreamEvent] = False,
    ) -> None: ...

    @overload
    @deprecated('Use `replay_events` instead of `events`.')
    def __init__(
        self,
        response: ModelResponse,
        *,
        model_request_parameters: ModelRequestParameters,
        events: bool | list[ModelResponseStreamEvent] = False,
    ) -> None: ...

    @overload
    @deprecated('Use `replay_events` instead of `events`.')
    def __init__(
        self,
        model_request_parameters: ModelRequestParameters,
        response: ModelResponse,
        /,
        *,
        events: bool | list[ModelResponseStreamEvent] = False,
    ) -> None: ...

    def __init__(
        self,
        response: ModelResponse | ModelRequestParameters,
        model_request_parameters: ModelRequestParameters | ModelResponse | None = None,
        *,
        replay_events: bool | list[ModelResponseStreamEvent] | _utils.Unset = _utils.UNSET,
        events: bool | list[ModelResponseStreamEvent] | None = None,
    ):
        # TODO(v3): remove the `events` alias and its deprecated `__init__` overloads
        if events is not None:
            warnings.warn(
                '`events` is deprecated; use `replay_events` instead.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            # The deprecated alias only fills the gap: an explicit `replay_events` wins.
            if isinstance(replay_events, _utils.Unset):
                replay_events = events
        if isinstance(replay_events, _utils.Unset):
            replay_events = False
        # TODO(v3): remove the positional `(model_request_parameters, response)` order and its deprecated overloads
        if isinstance(response, ModelRequestParameters):
            # The positional `(model_request_parameters, response)` order predates the move
            # from `pydantic_ai.models.wrapper` to `pydantic_ai.models`.
            warnings.warn(
                '`CompletedStreamedResponse(model_request_parameters, response)` is deprecated; pass the response '
                'first and `model_request_parameters` as a keyword argument: '
                '`CompletedStreamedResponse(response, model_request_parameters=...)`.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            response, model_request_parameters = cast(ModelResponse, model_request_parameters), response
        assert isinstance(model_request_parameters, ModelRequestParameters)
        super().__init__(model_request_parameters)
        self.response = response
        self.state = response.state
        self._replay_events = replay_events

    def __aiter__(self) -> AsyncIterator[ModelResponseStreamEvent]:
        if not isinstance(self._replay_events, list):
            return super().__aiter__()
        # Buffered events were already produced by the live stream's `__aiter__`,
        # which means they include `PartEndEvent`s. Yield them directly so the
        # parent `__aiter__` doesn't re-inject PartEnds.
        if self._event_iterator is None:
            self._event_iterator = self._iter_buffered(self._replay_events)
        return self._event_iterator

    async def _iter_buffered(self, events: list[ModelResponseStreamEvent]) -> AsyncIterator[ModelResponseStreamEvent]:
        for event in events:
            self._parts_manager.apply_event(event)
            yield event
        self._finished = True

    async def _get_event_iterator(self) -> AsyncIterator[ModelResponseStreamEvent]:
        # Only reached when `replay_events` is a bool — `__aiter__` short-circuits the
        # buffered-list path above.
        if self._replay_events is False:
            return
        for part in self.response.parts:
            # Register the complete part with the parts manager, which yields a single
            # `PartStartEvent` carrying its full content — exactly like a real stream that
            # delivers the part in one chunk. We deliberately do NOT follow it with a
            # `PartDeltaEvent` for the same content: a consumer that reduces the stream applies
            # `PartStartEvent.part` as the initial state and then each `PartDeltaEvent`, so a full
            # start plus a full delta would double the text/thinking/args. `PartEndEvent` is added
            # automatically by `StreamedResponse.__aiter__`.
            start_event = self._parts_manager.handle_part(vendor_part_id=None, part=part)
            assert isinstance(start_event, PartStartEvent)
            yield start_event

    async def close_stream(self) -> None:
        # No live stream to close — the response was produced without (or outside of) one.
        pass

    def get(self) -> ModelResponse:
        if isinstance(self._replay_events, list):
            return replace(
                self.response,
                parts=self._parts_manager.get_parts(),
                state=super().get().state,
            )
        return self.response

    @property
    def usage(self) -> RequestUsage:
        return self.response.usage

    @property
    def model_name(self) -> str:
        return self.response.model_name or ''

    @property
    def provider_name(self) -> str | None:
        return self.response.provider_name

    @property
    def provider_url(self) -> str | None:
        return self.response.provider_url

    @property
    def timestamp(self) -> datetime:
        return self.response.timestamp


ALLOW_MODEL_REQUESTS = True
"""Whether to allow requests to models.

This global setting allows you to disable request to most models, e.g. to make sure you don't accidentally
make costly requests to a model during tests.

The testing models [`TestModel`][pydantic_ai.models.test.TestModel],
[`FunctionModel`][pydantic_ai.models.function.FunctionModel] and
[`TestEmbeddingModel`][pydantic_ai.embeddings.TestEmbeddingModel] are not affected by this setting, nor is
[`SentenceTransformerEmbeddingModel`][pydantic_ai.embeddings.sentence_transformers.SentenceTransformerEmbeddingModel],
which runs inference locally and so has no per-call provider cost.
"""


def check_allow_model_requests() -> None:
    """Check if model requests are allowed.

    If you're defining your own models that have costs or latency associated with their use, you should call this at the
    top of each method that sends a request to the provider: [`Model.request`][pydantic_ai.models.Model.request],
    [`Model.request_stream`][pydantic_ai.models.Model.request_stream],
    [`Model.count_tokens`][pydantic_ai.models.Model.count_tokens],
    [`Model.compact_messages`][pydantic_ai.models.Model.compact_messages],
    [`EmbeddingModel.embed`][pydantic_ai.embeddings.EmbeddingModel.embed] and
    [`EmbeddingModel.count_tokens`][pydantic_ai.embeddings.EmbeddingModel.count_tokens].

    Methods that produce their result locally don't need it — for example
    [`OpenAIEmbeddingModel`][pydantic_ai.embeddings.openai.OpenAIEmbeddingModel]'s `count_tokens`, which tokenizes with
    `tiktoken` and never calls the provider. Neither does
    [`Model.cancel_suspended_response`][pydantic_ai.models.Model.cancel_suspended_response], which deliberately omits it
    so an already-started job can still be cancelled after the flag is flipped.

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


def parse_model_id(model: str) -> tuple[str | None, str]:
    """Parse a model id string into its provider and model name components.

    Args:
        model: A model identifier string in the form `provider:model_name`.

    Returns:
        A tuple of `(provider_name, model_name)`. If the model string contains no
        `provider:` prefix, returns `(None, model)` so callers can decide how to
        handle the unknown provider.
    """
    if ':' in model:
        provider_name, model_name = model.split(':', maxsplit=1)
        return provider_name, model_name

    return None, model


def _suggest_known_model_name(model: str, model_name: str, known_model_ids: Sequence[str] | None = None) -> str | None:
    if known_model_ids is None:
        known_model_ids = known_model_names()
    known_ids = sorted(known_model_ids, key=lambda name: (name.startswith('gateway/'), name))
    normalized_ids: list[str] = [known_id.replace(':', '-', 1) for known_id in known_ids if ':' in known_id]
    normalized_model = model.replace(':', '-', 1)
    if matches := get_close_matches(normalized_model, normalized_ids, n=1, cutoff=0.9):
        return next(known_id for known_id in known_ids if known_id.replace(':', '-', 1) == matches[0])

    known_names: list[str] = [known_id.split(':', maxsplit=1)[1] for known_id in known_ids if ':' in known_id]
    matches = get_close_matches(model_name, known_names, n=1, cutoff=0.8)
    if not matches:
        matches = get_close_matches(normalized_model, known_names, n=1, cutoff=0.7)
    if matches:
        return next(known_id for known_id in known_ids if known_id.endswith(f':{matches[0]}'))
    return None


def _suggest_known_model_id_from_provider_error(  # pyright: ignore[reportUnusedFunction]
    model_id_namespace: str, model_name: str
) -> str | None:
    """The closest known model ID for a name the provider itself rejected, or `None`.

    The result rides on `ModelHTTPError.suggested_model_id` rather than a dedicated exception type.
    Only some model classes carry a not-found signal at all — `MistralModel`, `CohereModel`,
    `HuggingFaceModel` and `XaiModel` map their errors without one — so a distinct type would assert
    a taxonomy that holds for part of the matrix only. A hint that is sometimes absent degrades
    harmlessly; an exception type that is sometimes absent misclassifies.
    """
    model_id = f'{model_id_namespace}:{model_name}'
    provider_prefix = f'{model_id_namespace}:'
    known_model_ids = [name for name in known_model_names() if name.startswith(provider_prefix)]
    suggestion = _suggest_known_model_name(model_id, model_name, known_model_ids)
    return suggestion if suggestion != model_id else None


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
        message = f'Unknown model: {model}'
        if suggested_name := _suggest_known_model_name(model, model_name):
            message += f". Did you mean '{suggested_name}'?"
        raise UserError(message)

    if provider_factory is infer_provider:
        try:
            infer_provider_class(provider_name)
        except ValueError:
            message = f'Unknown model: {model}'
            if suggested_name := _suggest_known_model_name(model, model_name):
                message += f". Did you mean '{suggested_name}'?"
            raise UserError(message) from None

    provider = provider_factory(provider_name)

    model_kind = provider_name
    if model_kind.startswith('gateway/'):
        from ..providers.gateway import normalize_gateway_provider

        model_kind = normalize_gateway_provider(model_kind)

    if provider_name == 'bedrock-mantle':
        from ..providers.bedrock_mantle import BedrockMantleProvider, bedrock_mantle_model_profile
        from .bedrock_mantle import BedrockMantleChatModel, BedrockMantleResponsesModel

        if not isinstance(provider, BedrockMantleProvider):
            raise UserError('Bedrock Mantle models require a `BedrockMantleProvider`.')
        # The profile carries the endpoint family (and raises for non-OpenAI models), so routing reads
        # it rather than re-deriving the interface here.
        if bedrock_mantle_model_profile(model_name).get('bedrock_mantle_interface') == 'chat':
            return BedrockMantleChatModel(model_name, provider=provider)
        return BedrockMantleResponsesModel(model_name, provider=provider)

    # OpenRouter, Cerebras, Crusoe, Ollama, Z.AI and Snowflake need to be checked before OpenAI,
    # as they are in `OpenAIChatCompatibleProvider` but have their own model classes.
    if model_kind == 'openrouter':
        from .openrouter import OpenRouterModel

        return OpenRouterModel(model_name, provider=provider)
    elif model_kind == 'cerebras':
        from .cerebras import CerebrasModel

        return CerebrasModel(model_name, provider=provider)
    elif model_kind == 'crusoe':
        from .crusoe import CrusoeModel

        return CrusoeModel(model_name, provider=provider)
    elif model_kind == 'snowflake':
        from .snowflake import SnowflakeModel

        return SnowflakeModel(model_name, provider=provider)
    elif model_kind == 'ollama':
        from .ollama import OllamaModel

        return OllamaModel(model_name, provider=provider)
    elif model_kind == 'zai':
        from .zai import ZaiModel

        return ZaiModel(model_name, provider=provider)
    elif model_kind in ('openai', 'openai-responses', 'azure-responses'):
        from .openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name, provider=provider)
    elif model_kind in ('openai-chat', *get_args(OpenAIChatCompatibleProvider.__value__)):
        from .openai import OpenAIChatModel

        return OpenAIChatModel(model_name, provider=provider)
    elif model_kind in ('google', 'google-cloud'):
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


def create_async_http_client(*, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5) -> AsyncClient:
    """Create a legacy HTTPX async client.

    This factory serves the providers whose SDKs still require a legacy `httpx.AsyncClient`;
    providers migrated to `httpx2` build their own `httpx2.AsyncClient` instead.

    Each call creates a new client instance. When used via a [`Provider`][pydantic_ai.providers.Provider],
    the client's lifecycle is managed automatically — it will be closed when the provider (or agent) exits.

    The default timeouts match those of OpenAI,
    see <https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9>.

    Raises:
        ImportError: If legacy `httpx` is not installed.
    """
    try:
        import httpx
    except ImportError as _import_error:
        raise ImportError(
            'Please install `httpx` to create a legacy HTTPX client with this factory, '
            'you can use the `retries` optional group — `pip install "pydantic-ai-slim[retries]"`. '
            'Providers otherwise build their own `httpx2.AsyncClient`, which you can also pass in yourself.'
        ) from _import_error

    return httpx.AsyncClient(
        timeout=httpx.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


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
    - Response bodies are limited to 50 MiB

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
            IP address (unless allow-local is set), or the body exceeds 50 MiB.
    """
    if isinstance(item, VideoUrl) and item.is_youtube:
        raise UserError('Downloading YouTube videos is not supported.')

    from .._ssrf import safe_download

    allow_local = item.force_download == 'allow-local'
    response = await safe_download(item.url, allow_local=allow_local, max_bytes=_MAX_FILE_URL_DOWNLOAD_BYTES)

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


def _customize_tool_def(transformer: type[JsonSchemaTransformer], tool_def: ToolDefinition) -> ToolDefinition:
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


def _customize_output_object(
    transformer: type[JsonSchemaTransformer], output_object: OutputObjectDefinition
) -> OutputObjectDefinition:
    schema_transformer = transformer(output_object.json_schema, strict=output_object.strict)
    json_schema = schema_transformer.walk()
    return replace(
        output_object,
        json_schema=json_schema,
        strict=schema_transformer.is_strict_compatible if output_object.strict is None else output_object.strict,
    )


def resolve_request_tools(
    params: ModelRequestParameters,
    supported_types: frozenset[type[AbstractNativeTool]],
    *,
    can_withhold_tool_schemas: Callable[[Sequence[AbstractNativeTool]], bool] | None = None,
    tool_addition_mode: ToolAdditionMode | None = None,
) -> ModelRequestParameters:
    """Resolve native tools, their local fallbacks, and deferred-tool visibility for the given supported-native-tool set.

    Three rules drive the per-tool filter:

    1. `unless_native` matches a supported native tool → drop from wire.
    2. `with_native` matches an *unsupported* native tool → shed `with_native`. The tool is
       a member of a corpus the native tool would have managed; with that native tool absent
       the membership means nothing, and an adapter deriving a wire flag from it would emit
       the flag unpaired and earn a rejection.
    3. `defer_loading` remains authored intent; this function resolves its provider representation
       into `tool_visibility` exactly once. A caller without a `can_withhold_tool_schemas` answer
       (the realtime session path) can't withhold schemas at all.

    On top of the filter, two narrower drops apply, kept independent:

    * `optional=True` only governs the *unsupported-on-this-model* path: an unsupported
      optional native tool is silently dropped (no error raised). It does NOT govern the
      corpus-empty drop.
    * The corpus-empty drop is specific to the framework-managed tool-search native tool's
      corpus-management role: an *optional* `ToolSearchTool` is dropped when nothing is
      searchable, since sending it with no corpus to search would waste a tool slot. A
      non-optional `ToolSearchTool` stays — the user asked explicitly. Other native tools
      don't have a corpus and aren't subject to this drop, so making `optional` a base-class
      field doesn't accidentally cause e.g. `WebSearchTool(optional=True)` to be dropped here.

    This is a module-level function rather than a `Model` method so both the classic agent-run
    path (via `Model._resolve_request_tools`, which passes its profile-derived
    `_can_withhold_tool_schemas` and `tool_addition_mode`) and the realtime session path can
    share it — `RealtimeModel` is not a `Model` subclass.
    """
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

    # Drop an optional `ToolSearchTool` with nothing to search. `ToolSearchToolset` marks only
    # the searchable deferred tools as corpus members, so a run whose deferred tools are all
    # gated by on-demand capabilities arrives here with an empty corpus and no search surface
    # is sent at all. The `isinstance` check confines this to `ToolSearchTool`: other native
    # tools don't carry a corpus, so making `optional` a base-class field doesn't accidentally
    # drop e.g. `WebSearchTool(optional=True)` here on absence of dependents.
    corpus_ids = {t.with_native for t in params.function_tools if t.with_native}
    supported_natives = [
        t for t in supported_natives if not (isinstance(t, ToolSearchTool) and t.optional) or t.unique_id in corpus_ids
    ]

    # Recomputed after the two steps above so it names the native tools this request really
    # sends: rule 1 must not drop a local fallback for a native tool that just left.
    supported_ids = {t.unique_id for t in supported_natives}

    can_defer = can_withhold_tool_schemas(supported_natives) if can_withhold_tool_schemas is not None else False
    tool_search_on_wire = any(isinstance(native, ToolSearchTool) for native in supported_natives)

    function_tools: list[ToolDefinition] = []
    visibility_by_name: dict[str, ToolVisibility] = {}
    for t in params.function_tools:
        # Rule 1: drop local fallback when the native tool is supported.
        if t.unless_native and t.unless_native in supported_ids:
            continue
        # Rule 2: a corpus member whose native tool is unsupported can't be paired with it here.
        if t.with_native and t.with_native not in supported_ids:
            t = replace(t, with_native=None)
        if not t.defer_loading:
            visibility = 'visible'
        else:
            revealed = t.name in params.revealed_tool_names
            corpus_member = t.with_native is not None and t.with_native in supported_ids
            if corpus_member and can_defer:
                visibility = 'deferred'
            elif revealed:
                if tool_addition_mode == 'with_definitions':
                    visibility = 'via_history'
                elif can_defer:
                    visibility = 'deferred'
                else:
                    visibility = 'visible'
            elif corpus_member:
                visibility = 'withheld'
            elif tool_search_on_wire:
                # A hidden non-corpus tool must stay off any wire carrying a search surface,
                # since server-side search indexes the request's deferred tool declarations.
                visibility = 'withheld'
            elif tool_addition_mode == 'with_definitions':
                visibility = 'withheld'
            elif can_defer:
                # Capability-only Anthropic runs pre-advertise from turn one: with no search
                # surface there is nothing that can leak the hidden tool, and the stable
                # declaration avoids a reveal-time deferred-preamble transition.
                visibility = 'deferred'
            else:
                visibility = 'withheld'
        function_tools.append(t)
        visibility_by_name[t.name] = visibility

    return replace(
        params,
        native_tools=supported_natives,
        function_tools=function_tools,
        tool_visibility=visibility_by_name,
    )


def prepare_return_schemas(
    params: ModelRequestParameters, *, supports_tool_return_schema: bool
) -> ModelRequestParameters:
    """Resolve return schemas: clear on tools that haven't opted in, inject into descriptions for non-native models.

    For tools with `include_return_schema=True` and a non-empty schema, models that natively support
    return schemas keep the schema as-is; other models get it injected into the tool description.
    Tools that haven't opted in have their `return_schema` cleared.

    A module-level function taking the profile flag rather than a `Model` method so both the classic
    path (via `Model.prepare_request`) and the realtime session path can share it — `RealtimeModel` is
    not a `Model` subclass and carries its own profile type.
    """
    inject = not supports_tool_return_schema
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


def _convert_speech_parts(messages: list[ModelMessage], *, include_audio: bool) -> list[ModelMessage]:
    """Convert `SpeechPart`s from realtime session history into parts any model can consume.

    User-speaker parts become `UserPromptPart`s carrying the retained audio (when `include_audio` is
    `True` and audio was retained) or the transcript text; assistant-speaker parts become `TextPart`s
    carrying the transcript. Parts without usable content are dropped, as are messages left without
    parts. Returns the original list when nothing changed so the identity check in `_make_request`
    can skip the redundant `_clean_message_history` pass.
    """
    if not any(isinstance(part, SpeechPart) for message in messages for part in message.parts):
        return messages

    new_messages: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            request_parts: list[ModelRequestPart] = []
            for part in message.parts:
                if isinstance(part, SpeechPart):
                    if include_audio and part.audio is not None:
                        request_parts.append(UserPromptPart(content=[part.audio]))
                    elif part.transcript:
                        request_parts.append(UserPromptPart(content=part.transcript))
                    # A part with neither retained audio nor transcript has nothing to send.
                else:
                    request_parts.append(part)
            if request_parts:
                new_messages.append(replace(message, parts=request_parts))
        else:
            # A barge-in cuts the model off mid-sentence, so the last speech part's transcript stops
            # short. Without an inline `[Interrupted]` marker, a standard model reads the fragment as a
            # complete utterance and may repeat itself. The marker is written here, on the way to the
            # model, and never persisted: history keeps the interruption on `SpeechPart.interrupted_at_ms`.
            last_speech = max(
                (index for index, part in enumerate(message.parts) if isinstance(part, SpeechPart)),
                default=None,
            )
            response_parts: list[ModelResponsePart] = []
            for index, part in enumerate(message.parts):
                if isinstance(part, SpeechPart):
                    lines = [part.transcript] if part.transcript else []
                    if part.interrupted_at_ms is not None:
                        lines.append(f'[Interrupted after {part.interrupted_at_ms} ms]')
                    elif message.state == 'interrupted' and index == last_speech:
                        # The provider reported the interruption without an offset.
                        lines.append('[Interrupted]')
                    if lines:
                        response_parts.append(TextPart(content='\n'.join(lines)))
                    # Assistant audio without a transcript has nothing to send.
                else:
                    response_parts.append(part)
            if response_parts:
                new_messages.append(replace(message, parts=response_parts))
    return new_messages


def _standing_system_prompt_count(request: ModelRequest) -> int:
    """How many of a request's opening parts belong to the run's standing system prompt.

    The standing prompt is authored before the run starts, so it is whatever `SystemPromptPart`s the
    first request *opens* with. One sitting after a user prompt or a tool return in that same request
    got there later: enqueued mid-run, or carried in from its own `ModelRequest` when
    `_clean_message_history` merged two adjacent requests that no assistant turn separated. Position
    is the only thing that tells them apart, and it is worth getting right — hoisting a
    mid-conversation instruction into the provider's top-level system parameter rewrites the first
    cache section of every later request, which is the exact invalidation that leaving it in place
    exists to avoid.
    """
    count = 0
    for part in request.parts:
        if not isinstance(part, SystemPromptPart):
            break
        count += 1
    return count


def _trim_messages_before_compaction(
    messages: list[ModelMessage],
    system: str,
    *,
    requires_encrypted_content: bool = False,
    standing_prompt_retained: bool = False,
) -> list[ModelMessage]:
    """Drop history before the latest same-provider compaction part the request will send.

    Reached through [`Model._trim_before_compaction`][pydantic_ai.models.Model._trim_before_compaction],
    which derives both flags from the adapter's declarations; adapters call that from their own
    message-prep step, since where in a request build the trim belongs is provider mechanics.
    Anthropic ignores (and doesn't bill) pre-boundary blocks,
    so there the trim only saves request size; the OpenAI Responses API processes and bills
    replayed items that precede a compaction item (live-verified), so there it is what makes
    compaction actually compact. `requires_encrypted_content` is this caller's own render condition,
    passed to the shared wire-boundary predicate: a part the adapter would omit must not act as a
    boundary either, or the history is dropped with nothing sent to stand in for it.

    The standing prompt survives via `_standing_prompt_request`; nothing else from the prefix does.
    `standing_prompt_retained` mirrors where the calling API carries the standing prompt: on
    Anthropic the top-level `system` parameter is rebuilt from the opening `SystemPromptPart`s on
    every request, so they must be re-inserted (`False`, the default) or the standing prompt is
    silently dropped from all subsequent requests. On OpenAI Responses those parts render as
    `system` input items *inside* the compacted window, and the compaction item demonstrably
    retains them — a latent directive that never fired before the boundary still governs
    post-compaction replies without the item being re-sent (live-verified) — so ordinary requests
    pass `True` and skip re-sending what the model would receive twice. `True` is honored only for
    a boundary part stamped with `STANDING_PROMPT_PLANTED_KEY`: retention presumes the compacted
    window contained the standing prompt, which only our own compact call guarantees — an
    externally produced or spliced-in item gets the standing prompt re-inserted as before.
    Retention is also only reliable for a single hop: a directive carried solely by a previous
    compaction item decayed when compacted again (live-verified), so the re-compaction call itself
    passes `False` to plant the standing prompt explicitly in every freshly built window (and
    stamps the result). Recovered instructions are re-sent either way — they travel as a
    per-request parameter, never inside the window.
    The Messages API accepts a request whose messages start with the assistant compaction block
    (live-verified), and keeping e.g. the original first user message can 400 when it carries a
    `tool_result` whose `tool_use` was trimmed away — validation runs even on ignored content.
    Idempotent: re-applying to an already-trimmed list is a no-op.
    """
    for message_index in range(len(messages) - 1, -1, -1):
        message = messages[message_index]
        if not isinstance(message, ModelResponse):
            continue
        for part_index in range(len(message.parts) - 1, -1, -1):
            part = message.parts[part_index]
            if not isinstance(part, CompactionPart) or not _compaction_part_is_wire_boundary(
                part, system, requires_encrypted_content=requires_encrypted_content
            ):
                continue
            tail = [replace(message, parts=message.parts[part_index:]), *messages[message_index + 1 :]]
            retained = standing_prompt_retained and bool(
                part.provider_details and part.provider_details.get(STANDING_PROMPT_PLANTED_KEY)
            )
            return [
                *_standing_prompt_request(messages[:message_index], include_system_parts=not retained),
                *tail,
            ]
    return messages


def _standing_prompt_request(prefix: list[ModelMessage], *, include_system_parts: bool = True) -> list[ModelRequest]:
    """The standing prompt from a trimmed-away prefix, as a request of its own.

    System parts come from the first `ModelRequest` wherever it appears (a history may open with a
    `ModelResponse`), sliced by `_standing_system_prompt_count` — the same opening-parts rule the
    hoisting adapters use; they are skipped entirely when the caller's compaction carrier already
    retains them (see `_trim_messages_before_compaction`). Instructions come from the latest prefix
    request that carried any: what the instruction fallback for direct `Model.request()` callers
    would otherwise have recovered from the dropped history; a kept-tail request with its own
    instructions still wins, being more recent. Mid-conversation `SystemPromptPart`s render inline
    as conversation content, which the compaction summary replaces, so they are deliberately not
    preserved.
    """
    first_request = next((m for m in prefix if isinstance(m, ModelRequest)), None)
    opening: Sequence[ModelRequestPart] = (
        first_request.parts[: _standing_system_prompt_count(first_request)]
        if first_request and include_system_parts
        else []
    )
    instructions = next(
        (m.instructions for m in reversed(prefix) if isinstance(m, ModelRequest) and m.instructions is not None),
        None,
    )
    if not opening and instructions is None:
        return []
    return [ModelRequest(parts=list(opening), instructions=instructions)]


def _wrap_non_leading_system_prompts(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Wrap mid-conversation `SystemPromptPart`s as `<system>`-tagged `UserPromptPart`s.

    The run's standing system prompt is left alone; the provider's `_map_messages` hoists it. Which
    parts those are is `_standing_system_prompt_count`'s
    question, and it is not simply "everything in the first request".

    Returns the original list when nothing changed so the identity check in `_make_request` can skip the
    redundant `_clean_message_history` pass.
    """
    first_request_idx = next(
        (i for i, m in enumerate(messages) if isinstance(m, ModelRequest)),
        None,
    )
    if first_request_idx is None:
        return messages

    new_messages: list[ModelMessage] = list(messages[:first_request_idx])
    changed = False
    for offset, msg in enumerate(messages[first_request_idx:]):
        start = _standing_system_prompt_count(msg) if offset == 0 and isinstance(msg, ModelRequest) else 0
        if isinstance(msg, ModelRequest) and any(isinstance(p, SystemPromptPart) for p in msg.parts[start:]):
            new_parts = [
                UserPromptPart(content=f'<system>{part.content}</system>', timestamp=part.timestamp)
                if index >= start and isinstance(part, SystemPromptPart)
                else part
                for index, part in enumerate(msg.parts)
            ]
            new_messages.append(replace(msg, parts=new_parts))
            changed = True
        else:
            new_messages.append(msg)

    return new_messages if changed else messages


def _unsynthesized_tool_availability_delta_error() -> UserError:  # pyright: ignore[reportUnusedFunction]
    """The error for a `ToolAvailabilityDeltaPart` that reached an adapter with no way to render it.

    `prepare_messages` projects every delta to the local tool-search exchange unless the profile
    advertises native support, so an adapter that doesn't support the part natively only sees one
    when that projection didn't run. Running a model through an agent always runs it, but
    [`Model.request`][pydantic_ai.models.Model.request] and
    [`Model.count_tokens`][pydantic_ai.models.Model.count_tokens] are public and don't, so a caller
    driving a model directly can reach this with a history that is otherwise perfectly valid. Hence
    a `UserError` naming the missing step, rather than an assertion about an internal invariant.

    Raising beats dropping the part: silently discarding it would tell the model nothing about the
    tools that appeared, and it would then fail to call a tool it was supposed to have gained.
    """
    return UserError(
        '`ToolAvailabilityDeltaPart` cannot be rendered by this model. '
        'Call `model.prepare_messages(messages)` first and pass the result — that projects the part '
        'into the tool-search exchange every model understands. `Agent` does this for you; a direct '
        '`Model.request()` or `Model.count_tokens()` call has to do it itself.'
    )


def _unconverted_speech_part_error() -> UserError:  # pyright: ignore[reportUnusedFunction]
    """The error for a realtime `SpeechPart` that reached an adapter unconverted.

    `prepare_messages` turns every `SpeechPart` from realtime session history into the
    `UserPromptPart`s / `TextPart`s any model can consume, so an adapter only sees one when that
    conversion didn't run. Running a model through an agent always runs it, but
    [`Model.request`][pydantic_ai.models.Model.request] and
    [`Model.count_tokens`][pydantic_ai.models.Model.count_tokens] are public and don't, so a caller
    driving a model directly can reach this with a history that is otherwise perfectly valid. Hence
    a `UserError` naming the missing step, rather than an assertion about an internal invariant.

    Raising beats dropping the part: silently discarding it would erase the turn's speech — possibly
    the entire user message — from what the model sees.
    """
    return UserError(
        '`SpeechPart` cannot be sent to this model as-is. '
        'Call `model.prepare_messages(messages)` first and pass the result — that converts realtime '
        'speech into the text and audio parts every model understands. `Agent` does this for you; a '
        'direct `Model.request()` or `Model.count_tokens()` call has to do it itself.'
    )


TOOL_AVAILABILITY_ANNOUNCEMENT = 'The following tool(s) are now available: {names}'
"""What a tool-availability change says to a model whose API can't express one itself.

Deliberately states only the fact. The tools appear in the request's `tools` list on this path, so
the model can already see their schemas; what it can't see is *when* they appeared, which is what
leaves it unable to explain a list that grew mid-conversation. Naming them is enough, and anything
more — urging the model to use them, explaining why they arrived — is an instruction nobody asked
for, on a turn the user didn't write.
"""


def _legacy_fabricated_tool_search_reveals(
    messages: list[ModelMessage], model_request_parameters: ModelRequestParameters
) -> dict[str, list[str]]:
    """Recognize pre-delta framework-fabricated `search_tools` exchanges.

    All three confidence signals are required: a framework-prefixed id, direct adjacency to a
    `load_capability` return, and discoveries confined to that capability's current tools.
    """
    capability_by_load_call_id = _load_capability_ids_by_call(messages)
    tools_by_capability: dict[str, set[str]] = {}
    for tool in [*model_request_parameters.function_tools, *model_request_parameters.output_tools]:
        if tool.capability_id is not None:
            tools_by_capability.setdefault(tool.capability_id, set()).add(tool.name)

    recognized: dict[str, list[str]] = {}
    for index, message in enumerate(messages):
        if index < 2 or not isinstance(message, ModelRequest) or len(message.parts) != 1:
            continue
        search_return = message.parts[0]
        if not isinstance(search_return, ToolReturnPart) or search_return.tool_name != TOOL_SEARCH_FUNCTION_TOOL_NAME:
            continue
        tool_call_id = search_return.tool_call_id
        if not tool_call_id.startswith(_utils.TOOL_CALL_ID_PREFIX):
            continue

        search_call_message = messages[index - 1]
        load_return_message = messages[index - 2]
        if (
            not isinstance(search_call_message, ModelResponse)
            or len(search_call_message.parts) != 1
            or not isinstance(search_call_message.parts[0], ToolCallPart)
            or search_call_message.parts[0].tool_name != TOOL_SEARCH_FUNCTION_TOOL_NAME
            or search_call_message.parts[0].tool_call_id != tool_call_id
            or not isinstance(load_return_message, ModelRequest)
            or not load_return_message.parts
        ):
            continue
        load_return = load_return_message.parts[-1]
        if not isinstance(load_return, ToolReturnPart) or load_return.tool_name != 'load_capability':
            continue
        capability_id = capability_by_load_call_id.get(load_return.tool_call_id)
        capability_tools = tools_by_capability.get(capability_id) if capability_id is not None else None
        if not capability_tools:
            continue

        discovered = _search_return_discovered_names(search_return)
        if discovered is None:
            continue
        if discovered and set(discovered) <= capability_tools:
            recognized[tool_call_id] = discovered

    return recognized


def _load_capability_ids_by_call(messages: list[ModelMessage]) -> dict[str, str]:
    capability_by_call_id: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, ModelResponse):
            continue
        for part in message.parts:
            if not isinstance(part, ToolCallPart) or part.tool_name != 'load_capability':
                continue
            try:
                args = part.args_as_dict(raise_if_invalid=True)
            except (AssertionError, ValueError):
                continue
            capability_id = args.get('id')
            if isinstance(capability_id, str):
                capability_by_call_id[part.tool_call_id] = capability_id
    return capability_by_call_id


def _search_return_discovered_names(part: ToolReturnPart) -> list[str] | None:
    if isinstance(part, ToolSearchReturnPart):
        return [match['name'] for match in part.discovered_tools]
    metadata = part.metadata
    discovered = metadata.get('discovered_tools') if metadata is not None else None
    if not isinstance(discovered, list):
        return None
    values = cast(list[Any], discovered)
    if not all(isinstance(name, str) for name in values):
        return None
    return cast(list[str], values)


def _replace_tool_search_exchanges_with_deltas(
    messages: list[ModelMessage], translated_call_ids: dict[str, list[str]]
) -> list[ModelMessage]:
    """Replace selected search call/return pairs with wire-only availability deltas."""
    transformed: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelResponse):
            parts = [
                part
                for part in message.parts
                if not (
                    isinstance(part, ToolCallPart)
                    and part.tool_name == TOOL_SEARCH_FUNCTION_TOOL_NAME
                    and part.tool_call_id in translated_call_ids
                )
            ]
        else:
            parts = [
                ToolAvailabilityDeltaPart(
                    tools_added=translated_call_ids[part.tool_call_id], tool_call_id=part.tool_call_id
                )
                if (
                    isinstance(part, ToolReturnPart)
                    and part.tool_name == TOOL_SEARCH_FUNCTION_TOOL_NAME
                    and part.tool_call_id in translated_call_ids
                )
                else part
                for part in message.parts
            ]
        if parts:
            transformed.append(replace(message, parts=parts))
    return transformed


def _announce_tool_availability_delta_messages(
    messages: list[ModelMessage], deferred_tool_names: set[str] | None
) -> list[ModelMessage]:
    """Render tool availability changes as a mid-conversation system instruction.

    Providers with a native way to say "these tools just appeared" get it rendered natively. The rest
    used to get a fabricated `search_tools` call/return pair, which told the model it had run a search
    it never ran. That was wrong in three ways, and all three go away by stating the fact instead:

    * It attributed an action to the model. In a mixed corpus — some tools searchable, some gated
      behind a capability — a capability load rendered as a search claims the wrong cause.
    * It could reference a `search_tools` tool that isn't on the wire, since the corpus-empty drop
      removes it when nothing is searchable. Some providers reject a history naming an undeclared tool.
    * It had to fabricate a `tool_call_id`, and two deltas over the same tool names produced the same
      one — duplicate ids in a history that providers requiring uniqueness reject.

    A `SystemPromptPart` also stays inside the delta's own message, where the pair had to be spliced
    across two messages: the fabricated `ModelResponse` went in ahead of the rebuilt `ModelRequest`,
    so a delta sharing a request with a user prompt put the assistant's turn before it and reordered
    the conversation. Within that message the announcements render after the request's tool results.

    On a model that takes a mid-conversation system message this lands as a real one, carrying the
    operator authority the statement deserves; elsewhere `_wrap_non_leading_system_prompts` — which
    runs after this — degrades it to `<system>`-tagged user text. Either way it's append-only, so the
    cached prefix ahead of it survives.
    """
    # If truncation promotes a never-sent delta request to first position, its announcement may
    # hoist with the standing prompt. All parts still precede the same assistant response, and the
    # rendering is deterministic; no finer positional fidelity is required within one request.
    transformed: list[ModelMessage] = []
    changed = False
    is_first_kept_request = True
    for message in messages:
        if not isinstance(message, ModelRequest):
            transformed.append(message)
            continue
        if not any(isinstance(part, ToolAvailabilityDeltaPart) for part in message.parts):
            transformed.append(message)
            is_first_kept_request = False
            continue

        changed = True
        replacement_parts: list[ModelRequestPart] = []
        for part in message.parts:
            if not isinstance(part, ToolAvailabilityDeltaPart):
                replacement_parts.append(part)
                continue
            # A delta that adds nothing has nothing to announce, so it drops out entirely.
            added = [name for name in part.tools_added if deferred_tool_names is None or name in deferred_tool_names]
            if added:
                replacement_parts.append(
                    SystemPromptPart(
                        content=TOOL_AVAILABILITY_ANNOUNCEMENT.format(names=', '.join(f'`{name}`' for name in added))
                    )
                )
        # A request whose only part was an empty delta would otherwise reach the adapter with no
        # parts at all, which providers reject.
        if replacement_parts:
            # Anthropic requires the tool results answering the previous turn to open the message,
            # so the announcements sort to the back. One exception: system prompts opening the
            # history's first request are the agent's standing prompt, which the adapters lift into
            # the provider's dedicated system field based on exactly this position, so they stay at
            # the front.
            request = replace(message, parts=replacement_parts)
            keep = _standing_system_prompt_count(request) if is_first_kept_request else 0
            head, tail = replacement_parts[:keep], replacement_parts[keep:]
            tail.sort(key=_tool_results_first_sort_key)
            transformed.append(replace(request, parts=[*head, *tail]))
            is_first_kept_request = False

    return transformed if changed else messages


def _synthesize_tool_availability_delta_messages(
    messages: list[ModelMessage], deferred_tool_names: set[str] | None
) -> list[ModelMessage]:
    """Render tool availability changes as the local tool-search exchange.

    For a model that can withhold a tool's schema, this exchange is the mechanism rather than the
    news: the return is what Anthropic renders as the `tool_reference` block that unhides the schema
    `defer_loading` is holding shut. A model without that ability gets
    `_announce_tool_availability_delta_messages` instead, which states the change without claiming
    the model ran a search.

    The exchange spans a turn boundary — an assistant call, then its return — so a request holding
    other parts alongside the delta has to be split at the delta's position. Emitting the whole
    rebuilt request after the synthetic `ModelResponse` instead would hoist an assistant turn ahead
    of a user prompt that originally preceded the delta, reordering the conversation.
    """
    transformed: list[ModelMessage] = []
    changed = False
    # Counts deltas that had an id fabricated, so two can't collide. The digest is taken over the tool
    # names, and the same names legitimately recur in one conversation — a tool withdrawn and re-added,
    # or a UI adapter replaying the same frontend tool set — which without this produced one id for both
    # exchanges. Duplicate ids are rejected by providers that require uniqueness, and mis-pair a call
    # with the wrong return for anything matching on id.
    #
    # The ordinal is stable across requests, which it has to be or the ids would move the prefix they
    # exist to protect: the projection reruns over the whole history each turn, and history is
    # append-only, so a delta already in it keeps its position and its id.
    synthesized_count = 0
    synthesized_ids: set[str] = set()
    # A client-authored delta may carry the id of the call that triggered it — an id that is
    # typically still in the surrounding history (https://github.com/pydantic/pydantic-ai/issues/7187).
    # Passing it through would emit a second assistant call part with the same id, which providers
    # requiring globally unique call ids reject or mis-pair with the wrong return. Seeding the
    # uniqueness check with the ids already present keeps the synthesized exchange distinct, while a
    # delta reusing the id of an exchange the projection collapsed away still passes through.
    history_call_ids = {
        part.tool_call_id
        for message in messages
        for part in message.parts
        if isinstance(part, BaseToolCallPart | BaseToolReturnPart | RetryPromptPart)
    }
    for message in messages:
        if not isinstance(message, ModelRequest) or not any(
            isinstance(part, ToolAvailabilityDeltaPart) for part in message.parts
        ):
            transformed.append(message)
            continue

        changed = True
        # Parts accumulated since the last split; flushed as their own `ModelRequest` before each
        # synthetic assistant turn so everything keeps the order it was authored in.
        pending: list[ModelRequestPart] = []
        for part in message.parts:
            if not isinstance(part, ToolAvailabilityDeltaPart):
                pending.append(part)
                continue
            added = [name for name in part.tools_added if deferred_tool_names is None or name in deferred_tool_names]
            if not added:
                continue

            tool_call_id = part.tool_call_id
            if tool_call_id is None or tool_call_id in synthesized_ids or tool_call_id in history_call_ids:
                while True:
                    digest = hashlib.blake2s(
                        '\x00'.join([str(synthesized_count), *added]).encode(),
                        digest_size=8,
                        usedforsecurity=False,
                    ).hexdigest()
                    synthesized_count += 1
                    tool_call_id = f'{_utils.TOOL_CALL_ID_PREFIX}{digest}'
                    # Loop-back on `synthesized_ids` needs a blake2s collision between distinct
                    # inputs (`synthesized_count` changes every iteration); a client-authored
                    # history part can carry a fabricated-shape id, so `history_call_ids` can loop.
                    if tool_call_id not in synthesized_ids and tool_call_id not in history_call_ids:
                        break
            synthesized_ids.add(tool_call_id)
            if pending:
                transformed.append(replace(message, parts=pending))
                pending = []
            transformed.append(
                ModelResponse(parts=[ToolSearchCallPart(args={'queries': added}, tool_call_id=tool_call_id)])
            )
            pending.append(
                ToolSearchReturnPart(
                    content={'discovered_tools': [{'name': name} for name in added]},
                    tool_call_id=tool_call_id,
                )
            )
        if pending:
            transformed.append(replace(message, parts=pending))

    return transformed if changed else messages
