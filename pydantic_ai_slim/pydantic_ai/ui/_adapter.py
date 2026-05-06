from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import KW_ONLY, Field, dataclass, replace
from functools import cached_property
from http import HTTPStatus
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    cast,
    runtime_checkable,
)
from urllib.parse import urlparse

from pydantic import BaseModel, ValidationError
from typing_extensions import Self, TypeVar

from pydantic_ai import DeferredToolRequests, DeferredToolResults, _instructions
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.agent.abstract import AgentMetadata
from pydantic_ai.builtin_tools import AbstractBuiltinTool
from pydantic_ai.capabilities import AbstractCapability, ReinjectSystemPrompt
from pydantic_ai.messages import (
    BaseToolCallPart,
    FileUrl,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelResponsePart,
    SystemPromptPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits

from ._event_stream import NativeEvent, OnCompleteFunc, UIEventStream

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse


__all__ = [
    'UIAdapter',
    'StateHandler',
    'StateDeps',
]

RunInputT = TypeVar('RunInputT')
"""Type variable for protocol-specific run input types."""

MessageT = TypeVar('MessageT')
"""Type variable for protocol-specific message types."""

EventT = TypeVar('EventT')
"""Type variable for protocol-specific event types."""

StateT = TypeVar('StateT', bound=BaseModel)
"""Type variable for the state type, which must be a subclass of `BaseModel`."""

DispatchDepsT = TypeVar('DispatchDepsT')
"""TypeVar for deps to avoid awkwardness with unbound classvar deps."""

DispatchOutputDataT = TypeVar('DispatchOutputDataT')
"""TypeVar for output data to avoid awkwardness with unbound classvar output data."""


@runtime_checkable
class StateHandler(Protocol):
    """Protocol for state handlers in agent runs. Requires the class to be a dataclass with a `state` field."""

    # Has to be a dataclass so we can use `replace` to update the state.
    # From https://github.com/python/typeshed/blob/9ab7fde0a0cd24ed7a72837fcb21093b811b80d8/stdlib/_typeshed/__init__.pyi#L352
    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    @property
    def state(self) -> Any:
        """Get the current state of the agent run."""
        ...

    @state.setter
    def state(self, state: Any) -> None:
        """Set the state of the agent run.

        This method is called to update the state of the agent run with the
        provided state.

        Args:
            state: The run state.
        """
        ...


@dataclass
class StateDeps(Generic[StateT]):
    """Dependency type that holds state.

    This class is used to manage the state of an agent run. It allows setting
    the state of the agent run with a specific type of state model, which must
    be a subclass of `BaseModel`.

    The state is set using the `state` setter by the `Adapter` when the run starts.

    Implements the `StateHandler` protocol.
    """

    state: StateT


@dataclass
class UIAdapter(ABC, Generic[RunInputT, MessageT, EventT, AgentDepsT, OutputDataT]):
    """Base class for UI adapters.

    This class is responsible for transforming agent run input received from the frontend into arguments for [`Agent.run_stream_events()`][pydantic_ai.agent.Agent.run_stream_events], running the agent, and then transforming Pydantic AI events into protocol-specific events.

    The event stream transformation is handled by a protocol-specific [`UIEventStream`][pydantic_ai.ui.UIEventStream] subclass.
    """

    agent: AbstractAgent[AgentDepsT, OutputDataT]
    """The Pydantic AI agent to run."""

    run_input: RunInputT
    """The protocol-specific run input object."""

    _: KW_ONLY

    accept: str | None = None
    """The `Accept` header value of the request, used to determine how to encode the protocol-specific events for the streaming response."""

    manage_system_prompt: Literal['server', 'client'] = 'server'
    """Who owns the system prompt.

    Only affects `system_prompt` — [`instructions`][pydantic_ai.Agent.instructions]
    are always injected by the agent on every request regardless of this setting.

    `'server'` (default): the agent's configured `system_prompt` is authoritative.
    Any `SystemPromptPart` sent by the frontend is stripped with a warning (since a
    malicious client could otherwise inject arbitrary instructions via crafted API
    requests), and the agent's own system prompt is reinjected at the head of the
    first request via the
    [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] capability.

    `'client'`: the frontend owns the system prompt. Frontend `SystemPromptPart`s
    are preserved as-is, and the agent's configured `system_prompt` is not injected
    — the caller is fully responsible for sending it on every turn if desired. To
    opt into the same fallback-to-configured behavior as server mode, add the
    [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] capability
    to your agent.
    """

    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'})
    """URL schemes that are allowed for [`FileUrl`][pydantic_ai.messages.FileUrl] parts
    ([`ImageUrl`][pydantic_ai.messages.ImageUrl], [`DocumentUrl`][pydantic_ai.messages.DocumentUrl],
    [`VideoUrl`][pydantic_ai.messages.VideoUrl], [`AudioUrl`][pydantic_ai.messages.AudioUrl])
    in client-submitted messages.

    Defaults to `{'http', 'https'}`. Parts whose URL scheme is not in this set are
    dropped with a warning before the messages are passed to the agent.

    Non-HTTP schemes like `s3://` (Bedrock) or `gs://` (Vertex AI) cause the model
    provider to fetch the object using the server-side IAM role or service account,
    so a client that can supply arbitrary URLs can read anything that identity can
    reach. HTTPS URLs are safe to forward because the provider fetches them with
    its own public credentials, and the library's own [`download_item`][pydantic_ai.models.download_item]
    path applies SSRF protection when it has to download them itself.

    For uploads initiated in the browser, prefer pre-signed `https://` URLs over
    cloud-storage schemes. To opt into a cloud-storage scheme after auditing your
    frontend, add it to this set, e.g. `frozenset({'http', 'https', 's3'})`.
    """

    @classmethod
    async def from_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        manage_system_prompt: Literal['server', 'client'] = 'server',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        **kwargs: Any,
    ) -> Self:
        """Create an adapter from a request.

        Extra keyword arguments are forwarded to the adapter constructor, allowing subclasses
        to accept additional adapter-specific parameters.
        """
        return cls(
            agent=agent,
            run_input=cls.build_run_input(await request.body()),
            accept=request.headers.get('accept'),
            manage_system_prompt=manage_system_prompt,
            allowed_file_url_schemes=allowed_file_url_schemes,
            **kwargs,
        )

    @classmethod
    @abstractmethod
    def build_run_input(cls, body: bytes) -> RunInputT:
        """Build a protocol-specific run input object from the request body."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load_messages(cls, messages: Sequence[MessageT]) -> list[ModelMessage]:
        """Transform protocol-specific messages into Pydantic AI messages."""
        raise NotImplementedError

    @classmethod
    def dump_messages(cls, messages: Sequence[ModelMessage]) -> list[MessageT]:
        """Transform Pydantic AI messages into protocol-specific messages."""
        raise NotImplementedError

    @abstractmethod
    def build_event_stream(self) -> UIEventStream[RunInputT, EventT, AgentDepsT, OutputDataT]:
        """Build a protocol-specific event stream transformer."""
        raise NotImplementedError

    @cached_property
    @abstractmethod
    def messages(self) -> list[ModelMessage]:
        """Pydantic AI messages from the protocol-specific run input."""
        raise NotImplementedError

    @cached_property
    def toolset(self) -> AbstractToolset[AgentDepsT] | None:
        """Toolset representing frontend tools from the protocol-specific run input."""
        return None

    @cached_property
    def state(self) -> dict[str, Any] | None:
        """Frontend state from the protocol-specific run input."""
        return None

    @cached_property
    def deferred_tool_results(self) -> DeferredToolResults | None:
        """Deferred tool results extracted from the request, used for tool approval workflows."""
        return None

    @cached_property
    def conversation_id(self) -> str | None:
        """Conversation ID extracted from the protocol-specific run input.

        Used to correlate multiple agent runs that share message history. Returned as
        the `gen_ai.conversation.id` OpenTelemetry span attribute on each run.

        Subclasses for protocols that carry a conversation/thread/chat ID should override this
        (e.g. AG-UI's `RunAgentInput.threadId`, Vercel AI's top-level chat `id`).
        """
        return None

    def sanitize_messages(
        self,
        messages: Sequence[ModelMessage],
        *,
        deferred_tool_results: DeferredToolResults | None = None,
    ) -> list[ModelMessage]:
        """Strip parts of client-submitted messages that aren't trusted from the client.

        Called on the messages produced from the protocol-specific run input before
        they're passed to the agent. Caller-supplied `message_history` is not passed
        through this method — it is trusted as coming from server-side persistence.

        Currently strips:

        - [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s when
          [`manage_system_prompt`][pydantic_ai.ui.UIAdapter.manage_system_prompt] is
          `'server'`. The agent's configured `system_prompt` is reinjected by
          [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] on
          the next model request. If stripping leaves a `ModelRequest` with no parts,
          the request is dropped from history entirely.
        - [`FileUrl`][pydantic_ai.messages.FileUrl] parts whose URL scheme is not in
          [`allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes].
          Non-HTTP schemes like `s3://` or `gs://` cause the model provider to fetch
          the object using the server-side IAM role, so they should only be accepted
          from trusted frontends.
        - [`ToolCallPart`][pydantic_ai.messages.ToolCallPart] and
          [`BuiltinToolCallPart`][pydantic_ai.messages.BuiltinToolCallPart] entries at
          the end of the history that don't have a matching entry in
          `deferred_tool_results`. Tool calls are produced by the model on the server
          side, so an unresolved tool call at the end of client-supplied history doesn't
          correspond to a paused agent run and shouldn't be executed. Tool calls that
          correspond to a resolution in `deferred_tool_results` are preserved so that
          human-in-the-loop resumption continues to work. If stripping leaves the final
          response with no parts, the response is dropped from history entirely.
        """
        resolved_tool_call_ids: set[str] = set()
        if deferred_tool_results is not None:
            resolved_tool_call_ids.update(deferred_tool_results.approvals)
            resolved_tool_call_ids.update(deferred_tool_results.calls)

        strip_system_prompt = self.manage_system_prompt == 'server'
        stripped_system_prompt = False
        disallowed_url_schemes: set[str] = set()
        dangling_tool_call_names: list[str] = []
        last_index = len(messages) - 1

        sanitized: list[ModelMessage] = []
        for index, message in enumerate(messages):
            if isinstance(message, ModelRequest):
                new_request_parts, request_stripped_system_prompt = self._sanitize_request_parts(
                    message.parts, strip_system_prompt=strip_system_prompt, disallowed_schemes=disallowed_url_schemes
                )
                stripped_system_prompt = stripped_system_prompt or request_stripped_system_prompt
                if new_request_parts:
                    sanitized.append(replace(message, parts=new_request_parts))
                # Otherwise drop the request entirely so we don't leave an empty
                # `ModelRequest(parts=[])` in history.
            elif isinstance(message, ModelResponse) and index == last_index:
                new_response_parts = self._sanitize_last_response_parts(
                    message.parts,
                    resolved_tool_call_ids=resolved_tool_call_ids,
                    dangling_names=dangling_tool_call_names,
                )
                if new_response_parts:
                    sanitized.append(replace(message, parts=new_response_parts))
                # Otherwise drop the final response entirely so we don't leave an empty
                # `ModelResponse(parts=[])` in history.
            else:
                sanitized.append(message)

        if stripped_system_prompt:
            warnings.warn(
                "Client-submitted system prompts were stripped because `manage_system_prompt` is `'server'` "
                "(the default). Set `manage_system_prompt='client'` to let the frontend own the system prompt.",
                UserWarning,
                stacklevel=2,
            )

        if disallowed_url_schemes:
            warnings.warn(
                f'Client-submitted file URLs with scheme(s) {sorted(disallowed_url_schemes)!r} '
                f'were dropped because those schemes are not in `allowed_file_url_schemes` '
                f'(currently {sorted(self.allowed_file_url_schemes)!r}). Non-HTTP schemes like '
                f'`s3://` or `gs://` are fetched by the model provider using the server-side IAM role, '
                f'so they should only be accepted from trusted frontends. To allow a scheme, add it to '
                f'`allowed_file_url_schemes` on the adapter.',
                UserWarning,
                stacklevel=2,
            )

        if dangling_tool_call_names:
            warnings.warn(
                f'Client-submitted history ended with unresolved tool call(s) '
                f'{sorted(set(dangling_tool_call_names))!r}, which were stripped. Tool calls are '
                f'produced by the model on the server side, so an unresolved tool call at the end '
                f'of client-supplied history does not correspond to a paused agent run. For '
                f'human-in-the-loop resumption, pass matching `deferred_tool_results` to the run '
                f'method.',
                UserWarning,
                stacklevel=2,
            )

        return sanitized

    def _sanitize_request_parts(
        self,
        parts: Sequence[ModelRequestPart],
        *,
        strip_system_prompt: bool,
        disallowed_schemes: set[str],
    ) -> tuple[list[ModelRequestPart], bool]:
        """Sanitize the parts of a client-submitted [`ModelRequest`][pydantic_ai.messages.ModelRequest].

        `disallowed_schemes` is updated in place with any non-allowlisted file URL schemes encountered.
        Returns the kept parts and whether any [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s were stripped.
        """
        stripped_system_prompt = False
        new_parts: list[ModelRequestPart] = []
        for part in parts:
            if strip_system_prompt and isinstance(part, SystemPromptPart):
                stripped_system_prompt = True
                continue
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                new_parts.append(replace(part, content=self._filter_user_content(part.content, disallowed_schemes)))
            else:
                new_parts.append(part)
        return new_parts, stripped_system_prompt

    def _filter_user_content(
        self,
        content: Sequence[UserContent],
        disallowed_schemes: set[str],
    ) -> list[UserContent]:
        """Drop [`FileUrl`][pydantic_ai.messages.FileUrl] items whose scheme isn't in the allowlist.

        `disallowed_schemes` is updated in place with any disallowed schemes encountered.
        """
        filtered: list[UserContent] = []
        for item in content:
            if isinstance(item, FileUrl):
                scheme = urlparse(item.url).scheme.lower()
                if scheme and scheme not in self.allowed_file_url_schemes:
                    disallowed_schemes.add(scheme)
                    continue
            filtered.append(item)
        return filtered

    def _sanitize_last_response_parts(
        self,
        parts: Sequence[ModelResponsePart],
        *,
        resolved_tool_call_ids: set[str],
        dangling_names: list[str],
    ) -> list[ModelResponsePart]:
        """Sanitize the parts of the trailing client-submitted [`ModelResponse`][pydantic_ai.messages.ModelResponse].

        Drops tool calls that aren't resolved by `deferred_tool_results`. `dangling_names`
        is appended to with the names of any stripped calls.
        """
        new_parts: list[ModelResponsePart] = []
        for part in parts:
            if isinstance(part, BaseToolCallPart) and part.tool_call_id not in resolved_tool_call_ids:
                dangling_names.append(part.tool_name)
                continue
            new_parts.append(part)
        return new_parts

    def transform_stream(
        self,
        stream: AsyncIterator[NativeEvent],
        on_complete: OnCompleteFunc[EventT] | None = None,
    ) -> AsyncIterator[EventT]:
        """Transform a stream of Pydantic AI events into protocol-specific events.

        Args:
            stream: The stream of Pydantic AI events to transform.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
        """
        return self.build_event_stream().transform_stream(stream, on_complete=on_complete)

    def encode_stream(self, stream: AsyncIterator[EventT]) -> AsyncIterator[str]:
        """Encode a stream of protocol-specific events as strings according to the `Accept` header value.

        Args:
            stream: The stream of protocol-specific events to encode.
        """
        return self.build_event_stream().encode_stream(stream)

    def streaming_response(self, stream: AsyncIterator[EventT]) -> StreamingResponse:
        """Generate a streaming response from a stream of protocol-specific events.

        Args:
            stream: The stream of protocol-specific events to encode.
        """
        return self.build_event_stream().streaming_response(stream)

    def run_stream_native(
        self,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
    ) -> AsyncIterator[NativeEvent]:
        """Run the agent with the protocol-specific run input and stream Pydantic AI events.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools to use for this run.
        """
        if deferred_tool_results is None:
            deferred_tool_results = self.deferred_tool_results
        if conversation_id is None:
            conversation_id = self.conversation_id

        frontend_messages = self.sanitize_messages(self.messages, deferred_tool_results=deferred_tool_results)
        message_history = [*(message_history or []), *frontend_messages]

        toolset = self.toolset
        if toolset:
            output_type = [output_type or self.agent.output_type, DeferredToolRequests]
            toolsets = [*(toolsets or []), toolset]

        if isinstance(deps, StateHandler):
            raw_state = self.state or {}
            if isinstance(deps.state, BaseModel):
                state = type(deps.state).model_validate(raw_state)
            else:
                state = raw_state

            deps.state = state
        elif self.state:
            warnings.warn(
                f'State was provided but `deps` of type `{type(deps).__name__}` does not implement the `StateHandler` protocol, so the state was ignored. Use `StateDeps[...]` or implement `StateHandler` to receive AG-UI state.',
                UserWarning,
                stacklevel=2,
            )

        capabilities: list[AbstractCapability[AgentDepsT]] = []
        if self.manage_system_prompt == 'server':
            capabilities.append(ReinjectSystemPrompt(replace_existing=True))

        return self.agent.run_stream_events(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            conversation_id=conversation_id,
            model=model,
            deps=deps,
            model_settings=model_settings,
            instructions=instructions,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            builtin_tools=builtin_tools,
            capabilities=capabilities,
        )

    def run_stream(
        self,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[EventT] | None = None,
    ) -> AsyncIterator[EventT]:
        """Run the agent with the protocol-specific run input and stream protocol-specific events.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools to use for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
        """
        return self.transform_stream(
            self.run_stream_native(
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
            ),
            on_complete=on_complete,
        )

    @classmethod
    async def dispatch_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[DispatchDepsT, DispatchOutputDataT],
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[DispatchDepsT] = None,
        deps: DispatchDepsT = None,
        output_type: OutputSpec[Any] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[DispatchDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[DispatchDepsT]] | None = None,
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        on_complete: OnCompleteFunc[EventT] | None = None,
        manage_system_prompt: Literal['server', 'client'] = 'server',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        **kwargs: Any,
    ) -> Response:
        """Handle a protocol-specific HTTP request by running the agent and returning a streaming response of protocol-specific events.

        Extra keyword arguments are forwarded to [`from_request`][pydantic_ai.ui.UIAdapter.from_request],
        allowing subclasses to accept additional adapter-specific parameters.

        Args:
            request: The incoming Starlette/FastAPI request.
            agent: The agent to run.
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            builtin_tools: Optional additional builtin tools to use for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
            manage_system_prompt: Who owns the system prompt. See
                [`UIAdapter.manage_system_prompt`][pydantic_ai.ui.UIAdapter.manage_system_prompt].
            allowed_file_url_schemes: URL schemes allowed for file URL parts from the client. See
                [`UIAdapter.allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes].
            **kwargs: Additional keyword arguments forwarded to [`from_request`][pydantic_ai.ui.UIAdapter.from_request].

        Returns:
            A streaming Starlette response with protocol-specific events encoded per the request's `Accept` header value.
        """
        try:
            from starlette.responses import Response
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'Please install the `starlette` package to use `dispatch_request()` method, '
                'you can use the `ui` optional group — `pip install "pydantic-ai-slim[ui]"`'
            ) from e

        try:
            # The DepsT and OutputDataT come from `agent`, not from `cls`; the cast is necessary to explain this to pyright
            adapter = cast(
                UIAdapter[RunInputT, MessageT, EventT, DispatchDepsT, DispatchOutputDataT],
                await cls.from_request(
                    request,
                    agent=cast(AbstractAgent[AgentDepsT, OutputDataT], agent),
                    manage_system_prompt=manage_system_prompt,
                    allowed_file_url_schemes=allowed_file_url_schemes,
                    **kwargs,
                ),
            )
        except ValidationError as e:  # pragma: no cover
            return Response(
                content=e.json(),
                media_type='application/json',
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        return adapter.streaming_response(
            adapter.run_stream(
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                deps=deps,
                output_type=output_type,
                model=model,
                instructions=instructions,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                infer_name=infer_name,
                toolsets=toolsets,
                builtin_tools=builtin_tools,
                on_complete=on_complete,
            ),
        )
