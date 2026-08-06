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
from datetime import datetime
from functools import cache, cached_property
from types import TracebackType
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast, get_args, overload

import httpx
from typing_extensions import Self, TypeAliasType, TypedDict, deprecated
from typing_inspection.introspection import get_literal_values

from .. import _utils
from .._json_schema import JsonSchemaTransformer
from .._output import StructuredTextOutputSchema
from .._parts_manager import ModelResponsePartsManager
from .._run_context import RunContext
from .._warnings import PydanticAIDeprecationWarning as PydanticAIDeprecationWarning
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
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    ModelResponseState,
    ModelResponseStreamEvent,
    PartEndEvent,
    PartStartEvent,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    UploadedFile,
    UserPromptPart,
    VideoUrl,
)
from ..native_tools import SUPPORTED_NATIVE_TOOLS, AbstractNativeTool
from ..native_tools._tool_search import ToolSearchTool
from ..output import OutputMode, OutputObjectDefinition, StructuredOutputMode
from ..profiles import DEFAULT_PROFILE, DEFAULT_PROMPTED_OUTPUT_TEMPLATE, ModelProfile, ModelProfileSpec, merge_profile
from ..providers import InterfaceClient, Provider, infer_provider, infer_provider_class
from ..settings import ModelSettings, ThinkingLevel, merge_model_settings

if TYPE_CHECKING:
    from ..agent.abstract import AbstractAgent
from ..tools import ToolDefinition
from ..usage import RequestUsage
from ._known_model_names import KnownModelName as KnownModelName

if TYPE_CHECKING:
    from ..agent.abstract import AbstractAgent
    from ..usage import RunUsage

DEFAULT_HTTP_TIMEOUT: int = 600
"""Default HTTP timeout in seconds for API requests.

This matches the default timeout used by OpenAI's Python client.
See https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9
"""

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
        'together',
        'vercel',
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


@dataclass(repr=False, kw_only=True)
class ModelRequestParameters:
    """Configuration for an agent's request to a model, specifically related to tools and output handling."""

    function_tools: list[ToolDefinition] = field(default_factory=list[ToolDefinition])
    native_tools: list[AbstractNativeTool] = field(default_factory=list[AbstractNativeTool])
    revealed_tool_names: set[str] = field(default_factory=set[str], repr=False)
    """Names of the deferred tools tool search or capability loading has revealed so far.

    A subset of `function_tools`' names. `ToolDefinition.defer_loading` records what the author asked for
    and stays set after a reveal, so this answers the separate question of what the model can see *now* —
    which is what an adapter needs in order to decide what to put on the wire.
    """

    deferred_capability_ids: set[str] = field(default_factory=set[str], repr=False)
    """IDs of the run's capabilities configured with `defer_loading=True`.

    The whole configured set, not the loaded subset, so it doesn't change as capabilities load. Adapters
    use it to recognize a tool as capability-owned — `ToolDefinition.capability_id` in this set — and so
    to tell a corpus a capability gates apart from one the model may search freely.
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

    @cached_property
    def tool_defs(self) -> dict[str, ToolDefinition]:
        return {tool_def.name: tool_def for tool_def in [*self.function_tools, *self.output_tools]}

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
    """

    model: Model
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters

    model_id: str | None = field(default=None, init=False)
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

    streaming: bool = field(default=False, init=False)
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
        params = _prepare_return_schemas(params, self.profile)

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
            params = self._resolve_native_tool_swap(params)

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
        the profile's `supports_inline_system_prompts` is `False`.

        Subclasses normally don't need to override this; the framework calls it on the
        agent's behalf in `_agent_graph._make_request` so per-adapter message-prep code
        sees a homogeneous shape regardless of which provider produced the prior turn.

        Args:
            messages: The history to pre-process.
            model_request_parameters: The parameters this history will be sent with. Optional, and
                only needed to render a `ToolAvailabilityDeltaPart` on a model with no native way to
                express one: whether that reveal has to be a mechanism or can just be a statement
                depends on whether any tool actually goes on the wire with its schema withheld, which
                the profile alone can't answer. Omitting it falls back to the profile-level answer,
                which differs only for a corpus mixing capability-gated and standalone deferred tools.
                Framework callers pass it.
        """
        supports_tool_addition = self.profile.get('tool_additions') is not None
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
        # Gated on a delta actually being present, not just on support, because answering
        # `_hides_deferred_schemas` resolves the request's native tools — and resolution raises for an
        # unsupported native tool. Doing that here on every request would preempt `prepare_request`,
        # which raises the same condition with the adapter's more specific message.
        if delta_parts and not supports_tool_addition:
            # `None` means "no definitions to validate against, render as recorded": the bare
            # `prepare_messages(messages)` form has no parameters, and filtering everything there
            # would erase legitimate announcements. The agent path always passes parameters, so
            # names that don't match a currently-served tool (function or output) render nothing —
            # a forged-but-well-shaped name must not reach system voice.
            available_tool_names = (
                set(model_request_parameters.tool_defs) if model_request_parameters is not None else None
            )
            # Two different jobs hide behind "render the delta", and which applies turns on whether this
            # model can withhold a tool's schema at all.
            #
            # Where it can, the revealed tool is already on the wire behind `defer_loading`, and the
            # tool-search exchange is what takes the flag off again: Anthropic renders the return as the
            # `tool_reference` block that unhides the schema. Announcing the change in prose there would
            # leave the tool hidden for good, which `test_anthropic_defer_loading_needs_a_reveal_mechanism`
            # pins as "the reveal and the flag travel together".
            #
            # Where it can't, the tool is simply present in `tools` from the turn it's revealed and the
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
                messages = _synthesize_tool_availability_delta_messages(messages, available_tool_names)
            else:
                messages = _announce_tool_availability_delta_messages(messages, available_tool_names)

        from .._tool_search import synthesize_local_tool_search_messages

        target_provider_name = self.system if supports_native_tool_search else None
        messages = synthesize_local_tool_search_messages(messages, target_provider_name=target_provider_name)

        if not self.profile.get('supports_inline_system_prompts', False):
            messages = _wrap_non_leading_system_prompts(messages)

        return messages

    def _resolve_native_tool_swap(self, params: ModelRequestParameters) -> ModelRequestParameters:
        """Resolve native tools, their local fallbacks, and deferred-tool visibility for this model.

        Three rules drive the per-tool filter:

        1. `unless_native` matches a supported native tool → drop from wire.
        2. `with_native` matches an *unsupported* native tool → shed `with_native`. The tool is
           a member of a corpus the native tool would have managed; with that native tool absent
           the membership means nothing, and an adapter deriving a wire flag from it would emit
           the flag unpaired and earn a rejection.
        3. `defer_loading` → the tool is hidden until something reveals it, and `_can_defer_tool_schemas`
           decides how that reaches the wire: kept declared-but-deferred where the model can withhold a
           schema, demoted to a plain visible tool where it can't but something has already revealed the
           tool, and withheld entirely where it can't and nothing has.

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
        """
        supported_types = self.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS)

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
            t
            for t in supported_natives
            if not (isinstance(t, ToolSearchTool) and t.optional) or t.unique_id in corpus_ids
        ]

        tool_search_resolution = _resolve_tool_search_native_for_capability_gated_tools(supported_natives, params)
        supported_natives = tool_search_resolution.native_tools
        tool_search_kept_local = tool_search_resolution.keep_search_tools_local
        # Recomputed after the two steps above so it names the native tools this request really
        # sends: rule 1 must not drop a local fallback for a native tool that just left.
        supported_ids = {t.unique_id for t in supported_natives}

        can_defer = self._can_defer_tool_schemas(supported_natives)

        function_tools: list[ToolDefinition] = []
        for t in params.function_tools:
            # Rule 1: drop local fallback when the native tool is supported — except for
            # `search_tools` when tool search was kept local for capability visibility,
            # where the local function tool is the callback the client-executed native
            # surface dispatches to.
            if t.unless_native and t.unless_native in supported_ids:
                if not (tool_search_kept_local and t.unless_native == ToolSearchTool.kind):
                    continue
            # Rule 2: a corpus member whose native tool is unsupported can't be paired with it here.
            if t.with_native and t.with_native not in supported_ids:
                t = replace(t, with_native=None)
            # Rule 3: a hidden tool this request has no way to hide is either already revealed —
            # and so a plain visible tool — or still hidden, in which case it stays off the wire
            # and arrives only if and when something reveals it.
            if t.defer_loading and not can_defer:
                if t.name not in params.revealed_tool_names:
                    continue
                t = replace(t, defer_loading=False)
            function_tools.append(t)

        return replace(params, native_tools=supported_natives, function_tools=function_tools)

    def _hides_deferred_schemas(self, params: ModelRequestParameters | None) -> bool:
        """Whether this request puts a tool on the wire with its schema withheld.

        That's what decides how a `ToolAvailabilityDeltaPart` has to be rendered on a model with no
        native way to express one. If a schema is withheld, the tool-search exchange is the mechanism
        that unhides it — Anthropic renders the return as a `tool_reference` block — and replacing it
        with prose would leave the tool unreachable. If nothing is withheld, the revealed tool is
        plainly in `tools` and the exchange is only news.

        Read off the resolved parameters rather than re-derived, because the answer is a property of
        the request and not of the model: the same model gives different answers for a capability-only
        corpus (nothing searchable, so on OpenAI no `tool_search` tool survives and the deferral flag
        can't be sent) and for a corpus that also holds a standalone deferred tool (search surface
        back, flag sent, reveal load-bearing again).

        Falls back to the profile-level answer when parameters weren't passed.
        """
        if params is None:
            # An empty native-tools sequence asks for the profile-only answer: no search tool survived.
            return self._can_defer_tool_schemas(())
        # Mirrors `prepare_request`'s guard so this can't raise where that wouldn't: with nothing
        # native and nothing deferred there is no schema to withhold anyway.
        if not (
            params.native_tools
            or any(t.unless_native or t.with_native or t.defer_loading for t in params.function_tools)
        ):
            return False
        resolved = self._resolve_native_tool_swap(params)
        # After resolution `defer_loading` means exactly "render the provider's deferral flag", so a
        # single tool carrying it is the whole question.
        return any(t.defer_loading for t in resolved.function_tools)

    def _can_defer_tool_schemas(self, native_tools: Sequence[AbstractNativeTool]) -> bool:
        """Whether this request can declare a function tool while withholding its schema.

        That's the wire form of "hidden until revealed": the tool occupies its `tools` entry from the
        first turn, and a reveal unlocks it in place, so the block the provider caches never changes.
        It needs the model to support deferred tools at all, and — on an API that only accepts the
        deferral flag alongside a tool-search tool — a tool-search tool actually surviving into this
        request.

        Callers turn the answer into the resolved `defer_loading` on each function tool, which is
        what makes it one decision: both adapters used to re-derive their own version of it, from
        their own inputs, and could disagree about the same request. After `prepare_request`,
        `defer_loading` means exactly "render the provider's deferral flag for this tool" and an
        adapter has only to read it.
        """
        if ToolSearchTool not in self.profile.get('supported_native_tools', SUPPORTED_NATIVE_TOOLS):
            return False
        if not self.profile.get('deferred_tools_require_tool_search', False):
            return True
        return any(isinstance(t, ToolSearchTool) for t in native_tools)

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
            # New v2 form: (default profile) -> final profile
            resolved = user(resolved)
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
        """Cancel the stream, stopping token generation.

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

    # OpenRouter, Cerebras, Ollama and Z.AI need to be checked before OpenAI,
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


@dataclass
class _ToolSearchNativeResolution:
    native_tools: list[AbstractNativeTool]
    keep_search_tools_local: bool


def _resolve_tool_search_native_for_capability_gated_tools(
    supported_natives: Sequence[AbstractNativeTool], params: ModelRequestParameters
) -> _ToolSearchNativeResolution:
    """Resolve tool search's native mode when a deferred capability gates a hidden tool.

    A capability-gated tool is never a corpus member, but on the wire the provider's deferral flag
    is the same flag corpus members carry, so provider-side tool search (Anthropic `bm25`/`regex`,
    OpenAI server-managed `tool_search`) would index it along with everything else and hand it back
    as a match. It's a black box: it can't honor "this tool is only visible after its owning
    capability has been loaded." Our local search loop in `ToolSearchToolset._search_tools` *can* —
    it only ever sees the searchable tools. So a request that both hides a capability-gated tool and
    sends a search surface has to run the search client-side, or the gate leaks.

    Two switches make that happen: (1) flip `ToolSearchTool(strategy=None)` to `'custom'` so
    the adapter wires the client-executed native surface (Anthropic tool-reference blocks,
    OpenAI `execution='client'`) which dispatches into our local `search_tools` callback;
    (2) the caller keeps `search_tools` in the request parameters — that callback is what
    the client-executed surface invokes. Adapters may still render that callback as a
    native client-executed tool-search item rather than as a regular function tool on the
    provider wire. Named-native strategies (`'bm25'`/`'regex'`) have no client-executed
    equivalent, so we raise rather than silently substitute a different algorithm.
    """
    capability_gates_a_tool = any(t.capability_id in params.deferred_capability_ids for t in params.function_tools)
    if not capability_gates_a_tool:
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
    inject = not profile.get('supports_tool_return_schema', False)
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


TOOL_AVAILABILITY_ANNOUNCEMENT = 'The following tool(s) are now available: {names}'
"""What a tool-availability change says to a model whose API can't express one itself.

Deliberately states only the fact. The tools appear in the request's `tools` list on this path, so
the model can already see their schemas; what it can't see is *when* they appeared, which is what
leaves it unable to explain a list that grew mid-conversation. Naming them is enough, and anything
more — urging the model to use them, explaining why they arrived — is an instruction nobody asked
for, on a turn the user didn't write.
"""


def _announce_tool_availability_delta_messages(
    messages: list[ModelMessage], available_tool_names: set[str] | None
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

    A `SystemPromptPart` also replaces the delta *in place*, where the pair had to be spliced across
    two messages: the fabricated `ModelResponse` went in ahead of the rebuilt `ModelRequest`, so a
    delta sharing a request with a user prompt put the assistant's turn before it and reordered the
    conversation.

    On a model that takes a mid-conversation system message this lands as a real one, carrying the
    operator authority the statement deserves; elsewhere `_wrap_non_leading_system_prompts` — which
    runs after this — degrades it to `<system>`-tagged user text. Either way it's append-only, so the
    cached prefix ahead of it survives.
    """
    transformed: list[ModelMessage] = []
    changed = False
    for message in messages:
        if not isinstance(message, ModelRequest) or not any(
            isinstance(part, ToolAvailabilityDeltaPart) for part in message.parts
        ):
            transformed.append(message)
            continue

        changed = True
        replacement_parts: list[ModelRequestPart] = []
        for part in message.parts:
            if not isinstance(part, ToolAvailabilityDeltaPart):
                replacement_parts.append(part)
                continue
            # A delta that adds nothing has nothing to announce, so it drops out entirely.
            added = [name for name in part.added if available_tool_names is None or name in available_tool_names]
            if added:
                replacement_parts.append(
                    SystemPromptPart(
                        content=TOOL_AVAILABILITY_ANNOUNCEMENT.format(names=', '.join(f'`{name}`' for name in added))
                    )
                )
        # A request whose only part was an empty delta would otherwise reach the adapter with no
        # parts at all, which providers reject.
        if replacement_parts:
            transformed.append(replace(message, parts=replacement_parts))

    return transformed if changed else messages


def _synthesize_tool_availability_delta_messages(
    messages: list[ModelMessage], available_tool_names: set[str] | None
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
            added = [name for name in part.added if available_tool_names is None or name in available_tool_names]
            if not added:
                continue

            tool_call_id = part.tool_call_id
            if tool_call_id is None or tool_call_id in synthesized_ids:
                while True:
                    digest = hashlib.blake2s(
                        '\x00'.join([str(synthesized_count), *added]).encode(),
                        digest_size=8,
                        usedforsecurity=False,
                    ).hexdigest()
                    synthesized_count += 1
                    tool_call_id = f'{_utils.TOOL_CALL_ID_PREFIX}{digest}'
                    # Loop-back needs a blake2s collision between distinct inputs (`synthesized_count`
                    # changes every iteration) — kept as a guarantee, not an expected path.
                    if tool_call_id not in synthesized_ids:  # pragma: no branch
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
