from __future__ import annotations

import re
import warnings
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
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
    Self,
    cast,
    runtime_checkable,
)

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from typing_extensions import TypeVar

from pydantic_ai import CancellationToken, DeferredToolRequests, DeferredToolResults, _instructions
from pydantic_ai._utils import is_str_dict
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.agent.abstract import AgentMetadata
from pydantic_ai.capabilities import AbstractCapability, ReinjectSystemPrompt
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    CompactionPart,
    ForceDownloadMode,
    InstructionBaselinePart,
    InstructionDeltaPart,
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    ModelSettingsDeltaPart,
    RetryFeedbackPart,
    RetryInfo,
    StructuredContent,
    SystemPromptPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolRetryCause,
    UserPromptPart,
    _dangling_tool_calls_by_response,  # pyright: ignore[reportPrivateUsage]
    _drop_compaction_parts,  # pyright: ignore[reportPrivateUsage]
    _is_tool_result_part,  # pyright: ignore[reportPrivateUsage]
    _sanitize_messages,  # pyright: ignore[reportPrivateUsage]
)
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputDataT, OutputSpec
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef

from ._event_stream import NativeEvent, OnCancelFunc, OnCompleteFunc, UIEventStream
from ._utils import get_ui_message_id, set_ui_message_id, without_ui_message_id

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response, StreamingResponse


__all__ = [
    'UIAdapter',
    'StateHandler',
    'StateDeps',
    'DEFAULT_ALLOWED_CONTENT_TYPES',
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

_TOOL_NAME_PATTERN = re.compile(r'[a-zA-Z0-9_-]{1,64}')
"""The tool-name shape accepted by the strictest supported model providers."""
_TOOL_AVAILABILITY_DELTA_PART_ADAPTER = TypeAdapter(ToolAvailabilityDeltaPart)
_MODEL_SETTINGS_DELTA_PART_ADAPTER = TypeAdapter(ModelSettingsDeltaPart)
_RETRY_FEEDBACK_PART_ADAPTER = TypeAdapter(RetryFeedbackPart)


DEFAULT_ALLOWED_CONTENT_TYPES = frozenset({'application/json'})
"""Request media types [`UIAdapter.from_request()`][pydantic_ai.ui.UIAdapter.from_request] accepts by default.

This is a CSRF control, not content negotiation. A browser can send the three CORS-safelisted
content types (`text/plain`, `multipart/form-data`, `application/x-www-form-urlencoded`) — or no
content type at all — cross-origin with no preflight, and every one of them can carry a JSON body.
An endpoint that parses the body regardless of the header can therefore be driven by any page the
caller happens to have open, using whatever ambient credentials the browser attaches. Authenticating
the endpoint does not prevent that; that is what makes CSRF a distinct concern from the client-trust
boundary described in the [UI adapter trust model](https://pydantic.dev/docs/ai/ui/overview/#trust-model-for-client-submitted-messages).

`application/json` is not safelisted, so requiring it forces a preflight, which a cross-origin page
can only pass if the application's own CORS policy grants it. Legitimate frontends are unaffected:
both the AG-UI and Vercel AI SDK default transports send `application/json`, so a genuinely
cross-origin frontend is preflighted and admitted by the CORS middleware the application already runs.

This is an allowlist rather than a denylist of the safelisted types, because a denylist would miss
the no-content-type case and would silently stop covering anything added to the safelist later.
"""


def _check_content_type(request: Request, allowed_content_types: frozenset[str] | None) -> None:
    """Reject a request whose media type is not allowed, before its body is read.

    Raises a Starlette `HTTPException` with status 415, which Starlette and FastAPI render as a
    response on their own — unlike the `ValidationError` from body parsing, which
    [`dispatch_request`][pydantic_ai.ui.UIAdapter.dispatch_request] has to convert itself.

    Does nothing when `allowed_content_types` is `None`.
    """
    if allowed_content_types is None:
        return

    # Media types are case-insensitive, so normalize the configured entries too, not just the
    # request's. Comparing a lowercased request value against a raw allowlist rejects a valid
    # request whenever the caller wrote `APPLICATION/JSON`, and the 415 then names the very media
    # type the request already sent. Normalizing here also keeps that message honest, since it is
    # built from the same set the comparison uses.
    allowed = {allowed_type.strip().lower() for allowed_type in allowed_content_types}

    media_type = request.headers.get('content-type', '').split(';')[0].strip().lower()
    if media_type in allowed:
        return

    from starlette.exceptions import HTTPException

    expected = ', '.join(sorted(allowed))
    raise HTTPException(
        status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        detail=f'Expected `Content-Type: {expected}`, got {media_type or "no content type"}',
    )


def validation_error_response(e: ValidationError) -> Response:
    """Build the 422 response returned when a UI request body fails validation."""
    from starlette.responses import Response

    try:
        content = e.json()
    except ValueError:
        # A body that isn't valid UTF-8 leaves the raw bytes on `input_value`, which
        # `e.json()` can't serialize — drop the echoed input so the client still gets its
        # 422 rather than a 500.
        content = e.json(include_input=False)
    return Response(
        content=content,
        media_type='application/json',
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
    )


# TODO(v3): remove this helper along with the Vercel AI adapter's deprecated `preserve_file_data` alias (AG-UI's `preserve_file_data` is a separate, non-deprecated setting)
def resolve_allow_uploaded_files(
    allow_uploaded_files: bool, preserve_file_data: bool | None, *, stacklevel: int = 3
) -> bool:
    """Map the deprecated `preserve_file_data` argument onto `allow_uploaded_files`.

    Returns `allow_uploaded_files` unchanged when `preserve_file_data` is omitted (`None`).
    When `preserve_file_data` is passed, emits a [`PydanticAIDeprecationWarning`][pydantic_ai.exceptions.PydanticAIDeprecationWarning]
    and returns its value. Used by adapters that exposed the old `preserve_file_data` argument for
    honoring client-submitted uploaded files (now `allow_uploaded_files`).

    `stacklevel` selects the frame the warning points at. The default `3` is right for the
    `from_request`/`dispatch_request` classmethod paths (user → method → helper → `warn`). The
    constructor path (`__post_init__`) has an extra generated-`__init__` frame in between
    (user → `__init__` → `__post_init__` → helper → `warn`), so it passes `stacklevel=4`.
    """
    if preserve_file_data is None:
        return allow_uploaded_files
    warnings.warn(
        '`preserve_file_data` is deprecated; use `allow_uploaded_files` to honor client-submitted '
        'uploaded file references.',
        PydanticAIDeprecationWarning,
        stacklevel=stacklevel,
    )
    return preserve_file_data


_COMPACTION_PART_ADAPTER = TypeAdapter(CompactionPart)


def enqueued_messages_for_client(
    messages: Sequence[ModelMessage], *, manage_system_prompt: Literal['server', 'client']
) -> list[ModelMessage]:
    """The enqueued messages to stream to the frontend.

    When the server owns the system prompt, enqueued `SystemPromptPart`s stay on the server: the
    frontend's copy would be stripped on the way back in anyway, so sending it would only disclose it.
    Request-scoped ones stay on the server either way, as in `dump_messages`: they're for the model's
    current request only, and loaded back they'd become lasting system prompts. So do instruction changes
    and baselines, which are server-side operator state.
    """
    kept: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = [
                part
                for part in message.parts
                if not isinstance(part, (InstructionDeltaPart, InstructionBaselinePart))
                and (
                    not isinstance(part, SystemPromptPart)
                    or (manage_system_prompt == 'client' and part.scope != 'request')
                )
            ]
            if not parts:
                continue
            if len(parts) != len(message.parts):
                message = replace(message, parts=parts)
        kept.append(message)
    return kept


def compaction_payload(part: CompactionPart) -> dict[str, Any]:
    """Serialize a compaction part as a UI payload, omitting fields whose value is `None`.

    The payload is faithful — `provider_details` travels verbatim, provenance stamp included.
    Trust is enforced on the load side: client-submitted messages pass through
    [`sanitize_messages`][pydantic_ai.messages.sanitize_messages], which strips the stamp.
    """
    return {
        key: value
        for key, value in {
            'content': part.content,
            'id': part.id,
            'provider_name': part.provider_name,
            'provider_details': part.provider_details,
            'metadata': part.metadata,
        }.items()
        if value is not None
    }


def compaction_part_from_payload(payload: Mapping[str, Any]) -> CompactionPart | None:
    """Build a [`CompactionPart`][pydantic_ai.messages.CompactionPart] from a UI payload.

    Like `tool_availability_delta_from_payload`, validation here is shape hygiene at the UI
    boundary, not a security gate: malformed data returns `None` and the part is skipped, rather
    than raising out of `load_messages` and taking the whole request with it. Skipping — instead of
    degrading to an empty part — is deliberate: even an empty `CompactionPart` acts as a visibility
    boundary for [`post_compaction_window`][pydantic_ai.messages.post_compaction_window] (which
    ignores `provider_name`), resetting derived state like tool discovery, so it would not be
    inert. The cost is that a corrupted-in-transit valid boundary un-compacts the conversation —
    acceptable, since the protocol history still holds the plaintext messages and the window merely
    re-inflates until the next compaction.
    """
    try:
        return _COMPACTION_PART_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


_STRUCTURED_CONTENT_ADAPTER: TypeAdapter[StructuredContent] = TypeAdapter(StructuredContent)


def structured_content_payload(item: StructuredContent) -> dict[str, Any]:
    """The part of a structured content item a UI protocol has no field for, to carry beside its text.

    The item is sent as an ordinary text part holding its `content`, so a frontend shows it like any other user
    text; this payload rides along in that part's metadata. The value is dumped as JSON-compatible data, and
    `tag` and `description` are omitted when unset. Like `TextContent.metadata`, `metadata` does not travel.
    """
    payload: dict[str, Any] = {'value': to_jsonable_python(item.value)}
    if item.tag is not None:
        payload['tag'] = item.tag
    if item.description is not None:
        payload['description'] = item.description
    return payload


def structured_content_from_payload(payload: object, text: str) -> StructuredContent | str:
    """Rebuild a [`StructuredContent`][pydantic_ai.messages.StructuredContent] from a text part and its payload.

    The text part's own text is the item's `content`, so what the frontend showed is what the model is sent.
    Like `compaction_part_from_payload`, validation here is shape hygiene at the UI boundary, not a security gate:
    the item is user content, which the client could send as text anyway. A malformed payload leaves the plain
    text, rather than raising out of `load_messages` and taking the whole request with it.
    """
    if not is_str_dict(payload):
        return text
    fields = {key: payload[key] for key in ('value', 'tag', 'description') if key in payload}
    try:
        return _STRUCTURED_CONTENT_ADAPTER.validate_python({**fields, 'content': text})
    except ValidationError:
        return text


def is_request_compaction_part(part: CompactionPart) -> bool:
    """Whether *part* can sit in a `ModelRequest`: a summary no provider produced, with no provider data."""
    return part.provider_name is None and part.id is None and part.provider_details is None


def tool_availability_delta_from_payload(payload: Mapping[str, Any]) -> ToolAvailabilityDeltaPart:
    """Build a [`ToolAvailabilityDeltaPart`][pydantic_ai.messages.ToolAvailabilityDeltaPart] from a UI payload.

    Client-driven tool availability is intentional. Validation here is shape hygiene at the UI
    boundary, not a security gate: malformed data renders an empty change rather than raising out of
    `load_messages` and taking the whole request with it.
    """
    try:
        part = _TOOL_AVAILABILITY_DELTA_PART_ADAPTER.validate_python(payload)
    except ValidationError:
        return ToolAvailabilityDeltaPart()
    part.tools_added = [name for name in part.tools_added if _TOOL_NAME_PATTERN.fullmatch(name)]
    if part.tool_call_id is not None and not part.tool_call_id.strip():
        part.tool_call_id = None
    return part


def model_settings_delta_payload(part: ModelSettingsDeltaPart) -> dict[str, Any]:
    """Serialize a [`ModelSettingsDeltaPart`][pydantic_ai.messages.ModelSettingsDeltaPart] for a UI data channel.

    The part rides through the client like a `ToolAvailabilityDeltaPart`, because the request after a round trip
    has to send the settings changes, and the request-level settings, that the earlier requests were sent with:
    leaving it out would change the prompt cache prefix.
    """
    return {'settings': dict(part.settings), 'top_level': part.top_level}


def model_settings_delta_from_payload(payload: Mapping[str, Any]) -> ModelSettingsDeltaPart | None:
    """Build a [`ModelSettingsDeltaPart`][pydantic_ai.messages.ModelSettingsDeltaPart] from a UI payload, or `None` if it's malformed.

    Client-driven like `tool_availability_delta_from_payload`, and bounded: each setting is validated against
    `RecordedModelSettings` (an effort against the effort levels; unknown keys are dropped), and a forged change
    is superseded by the one the next request records for the settings the server actually resolves.
    """
    try:
        part = _MODEL_SETTINGS_DELTA_PART_ADAPTER.validate_python(
            {'settings': payload.get('settings'), 'top_level': payload.get('top_level', False)}
        )
    except ValidationError:
        return None
    # A change records at least one setting, and only a window's opening part goes back to no effort level.
    if not part.settings or ('thinking' in part.settings and part.settings['thinking'] is None and not part.top_level):
        return None
    return part


def retry_feedback_payload(part: RetryFeedbackPart) -> dict[str, Any]:
    """Serialize a [`RetryFeedbackPart`][pydantic_ai.messages.RetryFeedbackPart] for a metadata channel.

    Both protocols dump harness retry feedback as a plain message, whose text is lossy — it says
    nothing about the `retry.cause` that chose its voice — and indistinguishable from a message the client
    wrote. So the part itself rides alongside it under a `retry_feedback` key in the metadata channel
    each protocol already uses for Pydantic AI continuity claims (Vercel AI's `providerMetadata`,
    AG-UI's `encrypted_value`), and that is what `load_messages` reconstructs from.

    Each error's `ctx` is dropped, so this channel discloses no more than the text beside it: `ctx`
    can hold whatever a `field_validator` was given as context, the rendered feedback never carries
    it, and the client reads and echoes this channel. Nothing reads it back — the part is
    reconstructed only to be rendered.
    """
    return _RETRY_FEEDBACK_PART_ADAPTER.dump_python(part, mode='json', exclude={'content': {'__all__': {'ctx'}}})


_TOOL_RETRY_INFO_ADAPTER: TypeAdapter[RetryInfo[ToolRetryCause]] = TypeAdapter(RetryInfo[ToolRetryCause])


def tool_retry_payload(retry: RetryInfo[ToolRetryCause]) -> dict[str, Any]:
    """Serialize a retried tool return's [`retry`][pydantic_ai.messages.ToolReturnPart.retry] for a metadata channel."""
    return _TOOL_RETRY_INFO_ADAPTER.dump_python(retry, mode='json', exclude_none=True)


def parse_tool_retry(value: object) -> RetryInfo[ToolRetryCause] | None:
    """Read a retried tool return's `retry` claim from a client-echoed metadata channel.

    The claim is untrusted coming back in, so anything that doesn't validate, such as an unknown
    cause, reads as `None`. It's metadata for the application only and never reaches the model, so a
    forged one can mislead an audit but not the run.
    """
    try:
        return _TOOL_RETRY_INFO_ADAPTER.validate_python(value)
    except ValidationError:
        return None


def retry_feedback_from_payload(payload: object) -> RetryFeedbackPart | None:
    """Rebuild the `RetryFeedbackPart` a dumped message claims to have been made from, or `None`.

    `None` for anything that doesn't validate — including the absent payload of a system message the
    client wrote — so only a message carrying the adapter's own marker can load as harness feedback.

    Like every claim in that channel the marker is client-echoed, so it proves origin no more than a
    `tool_kind` claim does: it separates our own dumped feedback from ordinary client system text, and
    [`sanitize_messages`][pydantic_ai.messages.sanitize_messages] is what keeps a forged one from
    reaching the model when the server owns the system prompt.
    """
    try:
        return _RETRY_FEEDBACK_PART_ADAPTER.validate_python(payload)
    except ValidationError:
        return None


def _warn_on_repeated_client_messages(
    message_history: Sequence[ModelMessage], client_messages: Sequence[ModelMessage]
) -> None:
    """Warn under `manage_message_history='client'` when client messages repeat ones `message_history` has, by UI id.

    Only a history that went through an adapter carries UI message ids, so one built any other way,
    such as by hand or from `agent.run()`, never warns: nothing tells its messages apart from the client's.
    """
    stored_ids = {get_ui_message_id(message) for message in message_history} - {None}
    if repeated := sum(get_ui_message_id(message) in stored_ids for message in client_messages):
        warnings.warn(
            f'{repeated} client-submitted message(s) already appear in `message_history`, so the model sees '
            f"them twice. If `message_history` holds the whole conversation, set `manage_message_history='server'` "
            f"to use only the client's new turn.",
            UserWarning,
            stacklevel=4,
        )


@dataclass
class _ServerHeldTurn:
    """What a run under `manage_message_history='server'` continues from, and what the client adds to it."""

    history: list[ModelMessage]
    """The stored history the run continues: all of it, or the part before a resubmission's cut.

    After a cut, it ends with a request holding the tool results the cut request carried, if it had any.
    """
    new_turn: list[ModelMessage]
    """The client's new turn, without tool results for calls that aren't pending or compaction parts.

    Not yet sanitized: `UIAdapter.sanitize_messages` still has to run on it.
    """
    cut_at: int | None = None
    """The index of the stored request a resubmitted message replaces, or `None` when the new turn is appended.

    `history[:cut_at]` is the part of the stored history the run keeps as it is.
    """


def _take_new_turn(
    message_history: Sequence[ModelMessage], client_messages: Sequence[ModelMessage], *, regenerate: bool = False
) -> _ServerHeldTurn:
    """Take the client's new turn under `manage_message_history='server'`, and the stored history it continues.

    The new turn is the client's requests after its last response. When its first message has the UI
    message id of a stored request with a user prompt, the client resubmitted that message, as a
    regenerate, an edit or a retried submit does, and the stored history is cut just before it, so the
    run replaces that turn. Only a user prompt can be resubmitted: a client can't cut the history at any
    other request whose id it knows. The tool results the cut request carries stay, as the server
    recorded them: a run merges the results of the calls before it into the client's next request.

    `regenerate` says the request asks to regenerate an answer, which is only possible for a message
    the stored history holds. When it doesn't, a `UserError` says so rather than adding a turn: the
    server never stored that message, as when its run failed or was stopped and the client retries it,
    or the client holds a different conversation than the server, as a client in a stale tab does.
    Without the client's assistant ids, the two can't be told apart.

    Tool results in the new turn are kept only for calls the stored history's last response is still
    waiting on. Copies of the results kept from the cut request are dropped without a warning, as an
    honest client resubmits them too, and other results with one. Compaction parts are never kept, as
    the client doesn't own the history they would cut off. See `_drop_compaction_parts`.
    """
    last_response = max((i for i, m in enumerate(client_messages) if isinstance(m, ModelResponse)), default=-1)
    new_turn = client_messages[last_response + 1 :]

    history = list(message_history)
    cut_at: int | None = None
    resubmitted_results: set[str] = set()
    ui_message_id = get_ui_message_id(new_turn[0]) if new_turn else None
    if ui_message_id is not None:
        cut_at = next(
            (
                i
                for i, m in enumerate(history)
                if isinstance(m, ModelRequest)
                and get_ui_message_id(m) == ui_message_id
                and any(isinstance(part, UserPromptPart) for part in m.parts)
            ),
            None,
        )
    if cut_at is not None:
        cut, history = history[cut_at], history[:cut_at]
        if results := [part for part in cut.parts if _is_tool_result_part(part)]:
            history.append(ModelRequest(parts=results))
            resubmitted_results = {part.tool_call_id for part in results}
    elif regenerate:
        raise UserError(
            f'The message to regenerate ({ui_message_id!r}) is not in `message_history`. Either the server never '
            f'stored it, as when the run that answered it failed or was stopped, or the client holds a different '
            f'conversation than the server, as a client in a stale tab does. Send the message again as a new one, '
            f'or reload the conversation from the server.'
        )

    pending = {call.tool_call_id for call in pending_tool_calls(history)}
    kept: list[ModelMessage] = []
    dropped = 0
    for message in new_turn:
        assert isinstance(message, ModelRequest)
        parts: list[ModelRequestPart] = []
        for part in message.parts:
            if _is_tool_result_part(part):
                if part.tool_call_id not in pending:
                    dropped += part.tool_call_id not in resubmitted_results
                    continue
                pending.discard(part.tool_call_id)
            parts.append(part)
        if len(parts) == len(message.parts):
            kept.append(message)
        elif parts:
            kept.append(replace(message, parts=parts))
    if dropped:
        warn_dropped_tool_results(dropped, stacklevel=5)
    return _ServerHeldTurn(history, _drop_compaction_parts(kept), cut_at)


def pending_tool_calls(message_history: Sequence[ModelMessage]) -> list[ToolCallPart]:
    """The tool calls of `message_history`'s last response that no request after it answers."""
    history = list(message_history)
    last_response = max((i for i, m in enumerate(history) if isinstance(m, ModelResponse)), default=-1)
    return _dangling_tool_calls_by_response(history).get(last_response, [])


def warn_dropped_tool_results(dropped: int, *, stacklevel: int) -> None:
    """Warn that `manage_message_history='server'` dropped client tool results for calls that aren't pending.

    `stacklevel` counts from the caller, as `warnings.warn` does.
    """
    warnings.warn(
        f'{dropped} client-submitted tool result(s) were dropped because they answer no tool call that '
        f"`message_history` is waiting on. With `manage_message_history='server'`, the client can only "
        f'return results for the tool calls at the end of the stored history.',
        UserWarning,
        stacklevel=stacklevel,
    )


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
    """Who owns the system prompt and request-level instructions in client-submitted messages.

    The agent's own [`instructions`][pydantic_ai.agent.Agent.instructions] are always
    injected on every request regardless of this setting.

    `'server'` (default): the agent's configured `system_prompt` is authoritative.
    Any `SystemPromptPart` sent by the frontend is stripped with a warning (since a
    malicious client could otherwise inject arbitrary instructions via crafted API
    requests), and the agent's own system prompt is reinjected at the head of the
    first request via the
    [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] capability.
    Request-level [`instructions`][pydantic_ai.messages.ModelRequest.instructions] on
    frontend messages are reset to `None` as well.

    `'client'`: the frontend owns the system prompt. Frontend `SystemPromptPart`s
    and request-level `instructions` are preserved as-is, and the agent's configured
    `system_prompt` is not injected — the caller is fully responsible for sending it
    on every turn if desired. To opt into the same fallback-to-configured behavior as
    server mode, add the
    [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] capability
    to your agent.
    """

    manage_message_history: Literal['client', 'server'] = 'client'
    """Who owns the conversation history: the client's messages, or the `message_history` passed to a run method.

    `'client'` (default): the client's messages are the conversation. A `message_history` passed to a
    run method is prepended to them unchanged, which suits server-only context such as a pinned
    preamble. If a client message's UI message id already appears in `message_history`, the adapter
    warns, because the model sees that message twice. Only a history that went through an adapter
    carries those ids, so for one built any other way the adapter can't tell: passing it while the
    client sends its whole conversation sends that conversation to the model twice, without a warning.

    `'server'`: `message_history` is the conversation, persisted by you, typically from
    [`AgentRunResult.all_messages()`][pydantic_ai.agent.AgentRunResult.all_messages] in `on_complete`.
    Only the client's new turn is used: its messages after the last assistant response. This works
    whether the client sends its whole history or only its new messages.

    - If the new turn starts with a user message whose UI message id is already in `message_history`,
      as on a regenerate, an edit or a retried submit, `message_history` is cut just before that
      message, so the run replaces that turn instead of adding a copy. Tool results stored alongside
      that message stay. Otherwise the new turn is appended, except that a regenerate of a message
      `message_history` doesn't hold is rejected with a protocol error event: the server never stored
      that message (its run failed or was stopped), or the client holds a different conversation than
      the server. Only a history an adapter run produced carries the UI message ids this matches on;
      for one persisted another way, a regenerate is always rejected and an edit is appended.
    - A client tool result is kept only if it answers a tool call that is still unanswered at the end
      of `message_history`. Others are dropped with a warning.
    - Tool approvals still come from the request, and the new turn is sanitized as usual.

    See [Keeping the history on the server](https://pydantic.dev/docs/ai/integrations/ui/overview/#keeping-the-history-on-the-server).
    """

    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'})
    """URL schemes that are allowed for [`FileUrl`][pydantic_ai.messages.FileUrl] parts
    ([`ImageUrl`][pydantic_ai.messages.ImageUrl], [`DocumentUrl`][pydantic_ai.messages.DocumentUrl],
    [`VideoUrl`][pydantic_ai.messages.VideoUrl], [`AudioUrl`][pydantic_ai.messages.AudioUrl])
    in client-submitted messages.

    Defaults to `{'http', 'https'}`. Parts whose URL scheme is not in this set are
    dropped with a warning before the messages are passed to the agent. This applies
    both to file URLs in user content and to those nested in tool return parts.

    Non-HTTP schemes like `s3://` (Bedrock) or `gs://` (Google Cloud) cause the model
    provider to fetch the object using the server-side IAM role or service account,
    so a client that can supply arbitrary URLs can read anything that identity can
    reach. HTTPS URLs are safe to forward because the provider fetches them with
    its own public credentials, and the library's own [`download_item`][pydantic_ai.models.download_item]
    path applies SSRF protection when it has to download them itself.

    For uploads initiated in the browser, prefer pre-signed `https://` URLs over
    cloud-storage schemes. To opt into a cloud-storage scheme after auditing your
    frontend, add it to this set, e.g. `frozenset({'http', 'https', 's3'})`.
    """

    allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset()
    """Additional [`FileUrl.force_download`][pydantic_ai.messages.FileUrl.force_download] values
    allowed on [`FileUrl`][pydantic_ai.messages.FileUrl] parts in client-submitted messages.

    `False` (the safe default that the sanitizer resets to) is always permitted regardless of
    whether it appears in this set. Values listed here are the *additional* `force_download`
    values that are trusted from the client. Defaults to `frozenset()`, so by default both
    `True` and `'allow-local'` are reset to `False` with a warning before the messages are
    passed to the agent. This applies both to file URLs in user content and to those nested in
    tool return parts.

    `force_download=True` makes the server download the file itself instead of letting the
    model provider fetch it. `force_download='allow-local'` additionally opts the URL out of
    the SSRF private-IP block in [`download_item`][pydantic_ai.models.download_item], which
    lets a client probe internal services. Neither is safe to honor from untrusted client
    input by default.

    To opt into a value after auditing your frontend, add it to this set, e.g.
    `frozenset({True})` or `frozenset({True, 'allow-local'})`.
    """

    allow_uploaded_files: bool = False
    """Whether to honor [`UploadedFile`][pydantic_ai.messages.UploadedFile] references from
    client-submitted messages.

    Defaults to `False`. By default, `UploadedFile` items in client-submitted messages are
    dropped with a warning before the messages are passed to the agent, mirroring how
    [`allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes] filters
    [`FileUrl`][pydantic_ai.messages.FileUrl] parts. This applies both to uploaded files in
    user content and to those nested in tool return parts.

    Like a non-HTTP `FileUrl`, an `UploadedFile` references an object that the model provider
    fetches using the server-side IAM role or service account, so a client that can supply
    arbitrary file references can read anything that identity can reach. Uploaded files should
    therefore only be accepted from trusted frontends.

    Set to `True` to honor client-submitted uploaded files after auditing your frontend.

    This is a purely inbound, security-oriented setting. It does not affect what the adapter
    sends *to* the client: file content the agent produces is always serialized on the way out.
    """

    strip_workspace_refs: bool = True
    """Whether to reset [`ModelResponse.workspace_ref`][pydantic_ai.messages.ModelResponse.workspace_ref]
    to `None` on client-submitted messages.

    Defaults to `True`. The most recent reference in history is offered to a capability's
    [`get_workspace`][pydantic_ai.capabilities.AbstractCapability.get_workspace], so a client that
    can set it could point a reconnecting capability at an environment it attaches to using
    server-side provider credentials. Reconnect explicitly by passing an authorized `workspace=` to
    the run method instead.

    The Vercel AI and AG-UI protocols do not carry workspace references. Setting this to `False`
    only affects `sanitize_messages` and custom adapters; it does not round-trip a reference through
    those protocols. Persist the ref server-side and pass an authorized `workspace=` instead.
    """

    @classmethod
    async def from_request(
        cls,
        request: Request,
        *,
        agent: AbstractAgent[AgentDepsT, OutputDataT],
        manage_system_prompt: Literal['server', 'client'] = 'server',
        manage_message_history: Literal['client', 'server'] = 'client',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset(),
        allow_uploaded_files: bool = False,
        strip_workspace_refs: bool = True,
        allowed_content_types: frozenset[str] | None = DEFAULT_ALLOWED_CONTENT_TYPES,
        **kwargs: Any,
    ) -> Self:
        """Create an adapter from a request.

        Extra keyword arguments are forwarded to the adapter constructor, allowing subclasses
        to accept additional adapter-specific parameters.

        Args:
            request: The incoming Starlette/FastAPI request.
            agent: The agent the adapter will run.
            manage_system_prompt: Who owns the system prompt. See
                [`UIAdapter.manage_system_prompt`][pydantic_ai.ui.UIAdapter.manage_system_prompt].
            manage_message_history: Who owns the conversation history. See
                [`UIAdapter.manage_message_history`][pydantic_ai.ui.UIAdapter.manage_message_history].
            allowed_file_url_schemes: URL schemes allowed for file URL parts from the client. See
                [`UIAdapter.allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes].
            allowed_file_url_force_download: Additional `FileUrl.force_download` values allowed on file URL parts
                from the client. See
                [`UIAdapter.allowed_file_url_force_download`][pydantic_ai.ui.UIAdapter.allowed_file_url_force_download].
            allow_uploaded_files: Whether to honor `UploadedFile` references from client-submitted messages. See
                [`UIAdapter.allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files].
            strip_workspace_refs: Whether to reset `ModelResponse.workspace_ref` on client-submitted messages. See
                [`UIAdapter.strip_workspace_refs`][pydantic_ai.ui.UIAdapter.strip_workspace_refs].
            allowed_content_types: Request media types to accept, as a CSRF control. Defaults to
                [`DEFAULT_ALLOWED_CONTENT_TYPES`][pydantic_ai.ui.DEFAULT_ALLOWED_CONTENT_TYPES]
                (`application/json`); anything else is rejected with a `415` before the body is read.
                Pass a wider set to admit another media type your frontend sends, or `None` to skip the
                check entirely when the route is already covered by CSRF protection of your own.
            **kwargs: Additional keyword arguments forwarded to the adapter constructor.
        """
        _check_content_type(request, allowed_content_types)

        return cls(
            agent=agent,
            run_input=cls.build_run_input(await request.body()),
            accept=request.headers.get('accept'),
            manage_system_prompt=manage_system_prompt,
            manage_message_history=manage_message_history,
            allowed_file_url_schemes=allowed_file_url_schemes,
            allowed_file_url_force_download=allowed_file_url_force_download,
            allow_uploaded_files=allow_uploaded_files,
            strip_workspace_refs=strip_workspace_refs,
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
        """Deferred tool results extracted from the request, used for tool approval workflows.

        An implementation may raise `UserError` to reject a malformed request: it is resolved inside
        the stream returned by [`run_stream_native`][pydantic_ai.ui.UIAdapter.run_stream_native]
        rather than when it is called, so the error reaches the client as a protocol error event
        instead of escaping the streaming response. Any other exception escapes as it normally
        would, so a bug in an implementation still surfaces as a server error.
        """
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
        through this method — it is trusted as coming from server-side persistence. Use
        [`sanitize_messages`][pydantic_ai.messages.sanitize_messages] before
        passing `message_history` that came from an untrusted client.

        Delegates to
        [`sanitize_messages`][pydantic_ai.messages.sanitize_messages] — see its
        docstring for the full list of what's stripped — with these adapter-specific settings:

        - [`SystemPromptPart`][pydantic_ai.messages.SystemPromptPart]s and request-level
          [`instructions`][pydantic_ai.messages.ModelRequest.instructions] are stripped only when
          [`manage_system_prompt`][pydantic_ai.ui.UIAdapter.manage_system_prompt] is `'server'`,
          and the agent's configured `system_prompt` is reinjected by
          [`ReinjectSystemPrompt`][pydantic_ai.capabilities.ReinjectSystemPrompt] on the next model
          request.
        - File URL schemes and `force_download` values are checked against
          [`allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes] and
          [`allowed_file_url_force_download`][pydantic_ai.ui.UIAdapter.allowed_file_url_force_download],
          and [`UploadedFile`][pydantic_ai.messages.UploadedFile]s are kept only when
          [`allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files] is `True`.
        - [`ModelResponse.workspace_ref`][pydantic_ai.messages.ModelResponse.workspace_ref] is reset
          unless [`strip_workspace_refs`][pydantic_ai.ui.UIAdapter.strip_workspace_refs] is `False`.
        - Tool calls at the end of the history are kept when they correspond to a resolution in
          `deferred_tool_results`, so human-in-the-loop resumption continues to work.
        - The UI message id that [`load_messages`][pydantic_ai.ui.UIAdapter.load_messages] keeps on each
          message survives, so [`dump_messages`][pydantic_ai.ui.UIAdapter.dump_messages] can still give it
          back. It's the client's own id, and the adapters read nothing else from that reserved namespace.
        """
        resolved_tool_call_ids: set[str] = set()
        if deferred_tool_results is not None:
            resolved_tool_call_ids.update(deferred_tool_results.approvals)
            resolved_tool_call_ids.update(deferred_tool_results.calls)

        # The reserved namespace is dropped from client messages, but the UI message id in it is the one
        # `load_messages` wrote itself, so it's set aside here and put back on whatever message survives.
        ui_message_ids = [get_ui_message_id(message) for message in messages]
        sanitized, sources = _sanitize_messages(
            [without_ui_message_id(message) for message in messages],
            strip_system_prompts=self.manage_system_prompt == 'server',
            strip_instructions=self.manage_system_prompt == 'server',
            allowed_file_url_schemes=self.allowed_file_url_schemes,
            allowed_file_url_force_download=self.allowed_file_url_force_download,
            allow_uploaded_files=self.allow_uploaded_files,
            strip_workspace_refs=self.strip_workspace_refs,
            resolved_tool_call_ids=resolved_tool_call_ids,
        )
        for message, source in zip(sanitized, sources):
            if (ui_message_id := ui_message_ids[source]) is not None:
                set_ui_message_id(message, ui_message_id)
        return sanitized

    def _regenerates(self) -> bool:
        """Whether the request asks to regenerate an answer, for protocols that say so.

        Under `manage_message_history='server'`, the message to regenerate has to be in `message_history`.
        """
        return False

    def _server_held_client_messages(self, message_history: Sequence[ModelMessage]) -> list[ModelMessage]:
        """The client's messages under `manage_message_history='server'`, loaded against the stored history.

        A protocol whose messages only load in the context of earlier ones, as an AG-UI tool result needs
        the call it answers, can fill that context in from `message_history` for a client that sends only
        its new messages.
        """
        return self.messages

    def _run_message_history(
        self, message_history: Sequence[ModelMessage] | None, deferred_tool_results: DeferredToolResults | None
    ) -> list[ModelMessage]:
        """The history to run on: `message_history` and the client's messages, per `manage_message_history`."""
        if self.manage_message_history == 'server':
            stored = message_history or []
            client_messages = self._server_held_client_messages(stored)
            turn = _take_new_turn(stored, client_messages, regenerate=self._regenerates())
            message_history, frontend_messages = turn.history, turn.new_turn
        else:
            frontend_messages = self.messages
            if message_history:
                _warn_on_repeated_client_messages(message_history, frontend_messages)
                # A client-supplied compaction part would trim the trusted server-side history off the
                # wire, so only the server's own boundaries are honored. See `_drop_compaction_parts`.
                frontend_messages = _drop_compaction_parts(frontend_messages)
        # `sanitize_messages` must be the last step that drops or merges client messages. Its final step
        # strips unresolved tool calls from the end of the history, and a run without a new prompt
        # executes the tool calls of whatever response ends up last. Dropping or merging a message after
        # this call can move an earlier, unchecked response to the end. Do any such change before it.
        frontend_messages = self.sanitize_messages(frontend_messages, deferred_tool_results=deferred_tool_results)
        return [*(message_history or []), *frontend_messages]

    def transform_stream(
        self,
        stream: AsyncIterator[NativeEvent],
        on_complete: OnCompleteFunc[EventT] | None = None,
        on_cancel: OnCancelFunc[EventT] | None = None,
    ) -> AsyncIterator[EventT]:
        """Transform a stream of Pydantic AI events into protocol-specific events.

        Args:
            stream: The stream of Pydantic AI events to transform.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
            on_cancel: Optional callback function called when the agent run ends in first-party cancellation.
                The callback receives the [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] and can optionally yield additional protocol-specific events.
        """
        return self.build_event_stream().transform_stream(stream, on_complete=on_complete, on_cancel=on_cancel)

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
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        timeout: float | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AbstractCapability[AgentDepsT]] | None = None,
        workspace: WorkspaceBackend | WorkspaceRef | Literal['new'] | None = None,
    ) -> AsyncIterator[NativeEvent]:
        """Run the agent with the protocol-specific run input and stream Pydantic AI events.

        A `UserError` from [`deferred_tool_results`][pydantic_ai.ui.UIAdapter.deferred_tool_results],
        or the one [`manage_message_history='server'`][pydantic_ai.ui.UIAdapter.manage_message_history]
        raises for a regenerate of a message `message_history` doesn't hold, is re-raised on the first
        iteration of the returned stream rather than at the call site, so a request rejected this way
        surfaces as a protocol error event on the stream. Any other exception still raises at the call
        site, and nothing else about the run input is deferred.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far. Prepended to the client's messages, unless
                [`manage_message_history`][pydantic_ai.ui.UIAdapter.manage_message_history] is `'server'`.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7, except that an agent with a `TemporalDurability`, `DBOSDurability` or `PrefectDurability` capability, run inside a workflow or flow, gets one derived from the `run_id` so it stays the same when the engine re-executes the run.
            run_id: Optional ID for this agent run. Unlike `conversation_id`, never inherited from `message_history`. Passing an empty string, or a value that already appears on `message_history`, raises `UserError` because both break `new_messages()`; use `conversation_id` to correlate across turns or deferred-tool resume. If omitted, a fresh UUID7 is generated, except that an agent with a `TemporalDurability`, `DBOSDurability` or `PrefectDurability` capability, run inside a workflow or flow, gets one derived from the workflow or flow run so it stays the same when the engine re-executes the run.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            cancellation_token: Optional token for cancelling this run from another task.
            timeout: Optional maximum duration of the whole run, in seconds. When it passes, the run is
                cancelled and raises [`RunTimedOut`][pydantic_ai.exceptions.RunTimedOut], keeping what it completed.
                See [Bounding a whole run](../../timeouts.md#bounding-a-whole-run).
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            capabilities: Optional additional [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) for this run, merged with the agent's configured capabilities.
                Use `capabilities=[NativeTool(...)]` to add provider-side native tools per request.
            workspace: Optional [workspace](../../workspace.md) for this run: a backend or `Workspace` to use as is, a `WorkspaceRef` to continue in, or `'new'` for a fresh one instead of the one in `message_history`.
        """
        if conversation_id is None:
            conversation_id = self.conversation_id

        # A `UserError` rejecting the request is handed back as a failing stream so it reaches
        # `transform_stream`'s error handling; the early return keeps the run setup below, like
        # writing `deps.state`, from running for a request that never starts.
        try:
            if deferred_tool_results is None:
                deferred_tool_results = self.deferred_tool_results
            message_history = self._run_message_history(message_history, deferred_tool_results)
        except UserError as e:
            # Rebound because Python clears the `except ... as` name when the block exits, leaving
            # the generator below with an unresolvable free variable.
            error = e

            async def rejected_stream() -> AsyncIterator[NativeEvent]:
                raise error
                yield  # pragma: no cover

            return rejected_stream()

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

        run_capabilities: list[AbstractCapability[AgentDepsT]] = []
        if self.manage_system_prompt == 'server':
            # `id=None` opts this adapter-owned instance out of the fixed default `id`, so it never
            # occupies the `reinject_system_prompt` slot a user's own reinjector may hold. Keeping the
            # default here would make a user reinjector supplied for this run a hard duplicate-id
            # error, and silently supersede an agent-level one. Server mode stays authoritative
            # either way: this instance always runs, deriving a distinct id when it has to.
            run_capabilities.append(ReinjectSystemPrompt(replace_existing=True, id=None))
        if capabilities:
            run_capabilities.extend(capabilities)

        async def stream_events() -> AsyncIterator[NativeEvent]:
            async with self.agent.run_stream_events(
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                run_id=run_id,
                cancellation_token=cancellation_token,
                timeout=timeout,
                model=model,
                deps=deps,
                model_settings=model_settings,
                instructions=instructions,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                infer_name=infer_name,
                toolsets=toolsets,
                capabilities=run_capabilities,
                workspace=workspace,
            ) as events:
                async for event in events:
                    yield event

        return stream_events()

    def run_stream(
        self,
        *,
        output_type: OutputSpec[Any] | None = None,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[AgentDepsT] = None,
        deps: AgentDepsT = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        timeout: float | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[AgentDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        capabilities: Sequence[AbstractCapability[AgentDepsT]] | None = None,
        workspace: WorkspaceBackend | WorkspaceRef | Literal['new'] | None = None,
        on_complete: OnCompleteFunc[EventT] | None = None,
        on_cancel: OnCancelFunc[EventT] | None = None,
    ) -> AsyncIterator[EventT]:
        """Run the agent with the protocol-specific run input and stream protocol-specific events.

        Args:
            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far. Prepended to the client's messages, unless
                [`manage_message_history`][pydantic_ai.ui.UIAdapter.manage_message_history] is `'server'`.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7, except that an agent with a `TemporalDurability`, `DBOSDurability` or `PrefectDurability` capability, run inside a workflow or flow, gets one derived from the `run_id` so it stays the same when the engine re-executes the run.
            run_id: Optional ID for this agent run. Unlike `conversation_id`, never inherited from `message_history`. Passing an empty string, or a value that already appears on `message_history`, raises `UserError` because both break `new_messages()`; use `conversation_id` to correlate across turns or deferred-tool resume. If omitted, a fresh UUID7 is generated, except that an agent with a `TemporalDurability`, `DBOSDurability` or `PrefectDurability` capability, run inside a workflow or flow, gets one derived from the workflow or flow run so it stays the same when the engine re-executes the run.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            cancellation_token: Optional token for cancelling this run from another task.
            timeout: Optional maximum duration of the whole run, in seconds. When it passes, the run is
                cancelled and raises [`RunTimedOut`][pydantic_ai.exceptions.RunTimedOut], which is reported like
                any other first-party cancellation, including to `on_cancel`. See [Bounding a whole run](../../timeouts.md#bounding-a-whole-run).
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            capabilities: Optional additional [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) for this run, merged with the agent's configured capabilities.
                Use `capabilities=[NativeTool(...)]` to add provider-side native tools per request.
            workspace: Optional [workspace](../../workspace.md) for this run: a backend or `Workspace` to use as is, a `WorkspaceRef` to continue in, or `'new'` for a fresh one instead of the one in `message_history`.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
            on_cancel: Optional callback function called when the agent run ends in first-party cancellation.
                The callback receives the [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] and can optionally yield additional protocol-specific events.
        """
        return self.transform_stream(
            self.run_stream_native(
                output_type=output_type,
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                run_id=run_id,
                cancellation_token=cancellation_token,
                timeout=timeout,
                model=model,
                instructions=instructions,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                metadata=metadata,
                infer_name=infer_name,
                toolsets=toolsets,
                capabilities=capabilities,
                workspace=workspace,
            ),
            on_complete=on_complete,
            on_cancel=on_cancel,
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
        run_id: str | None = None,
        model: Model | KnownModelName | str | None = None,
        instructions: _instructions.AgentInstructions[DispatchDepsT] = None,
        deps: DispatchDepsT = None,
        output_type: OutputSpec[Any] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        cancellation_token: CancellationToken | None = None,
        timeout: float | None = None,
        usage: RunUsage | None = None,
        metadata: AgentMetadata[DispatchDepsT] | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[DispatchDepsT]] | None = None,
        capabilities: Sequence[AbstractCapability[DispatchDepsT]] | None = None,
        workspace: WorkspaceBackend | WorkspaceRef | Literal['new'] | None = None,
        on_complete: OnCompleteFunc[EventT] | None = None,
        on_cancel: OnCancelFunc[EventT] | None = None,
        manage_system_prompt: Literal['server', 'client'] = 'server',
        manage_message_history: Literal['client', 'server'] = 'client',
        allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
        allowed_file_url_force_download: frozenset[ForceDownloadMode] = frozenset(),
        allow_uploaded_files: bool = False,
        strip_workspace_refs: bool = True,
        allowed_content_types: frozenset[str] | None = DEFAULT_ALLOWED_CONTENT_TYPES,
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
            message_history: History of the conversation so far. Prepended to the client's messages, unless
                [`manage_message_history`][pydantic_ai.ui.UIAdapter.manage_message_history] is `'server'`.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7, except that an agent with a `TemporalDurability`, `DBOSDurability` or `PrefectDurability` capability, run inside a workflow or flow, gets one derived from the `run_id` so it stays the same when the engine re-executes the run.
            run_id: Optional ID for this agent run. Unlike `conversation_id`, never inherited from `message_history`. Passing an empty string, or a value that already appears on `message_history`, raises `UserError` because both break `new_messages()`; use `conversation_id` to correlate across turns or deferred-tool resume. If omitted, a fresh UUID7 is generated, except that an agent with a `TemporalDurability`, `DBOSDurability` or `PrefectDurability` capability, run inside a workflow or flow, gets one derived from the workflow or flow run so it stays the same when the engine re-executes the run.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            instructions: Optional additional instructions to use for this run.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            cancellation_token: Optional token for cancelling this run from another task.
            timeout: Optional maximum duration of the whole run, in seconds. When it passes, the run is
                cancelled and raises [`RunTimedOut`][pydantic_ai.exceptions.RunTimedOut], which is reported like
                any other first-party cancellation, including to `on_cancel`. See [Bounding a whole run](../../timeouts.md#bounding-a-whole-run).
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
                [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            capabilities: Optional additional [capabilities](https://pydantic.dev/docs/ai/capabilities/overview/) for this run, merged with the agent's configured capabilities.
                Use `capabilities=[NativeTool(...)]` to add provider-side native tools per request.
            workspace: Optional [workspace](../../workspace.md) for this run: a backend or `Workspace` to use as is, a `WorkspaceRef` to continue in, or `'new'` for a fresh one instead of the one in `message_history`.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can optionally yield additional protocol-specific events.
            on_cancel: Optional callback function called when the agent run ends in first-party cancellation.
                The callback receives the [`RunCancelled`][pydantic_ai.exceptions.RunCancelled] and can optionally yield additional protocol-specific events.
            manage_system_prompt: Who owns the system prompt. See
                [`UIAdapter.manage_system_prompt`][pydantic_ai.ui.UIAdapter.manage_system_prompt].
            manage_message_history: Who owns the conversation history. See
                [`UIAdapter.manage_message_history`][pydantic_ai.ui.UIAdapter.manage_message_history].
            allowed_file_url_schemes: URL schemes allowed for file URL parts from the client. See
                [`UIAdapter.allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes].
            allowed_file_url_force_download: Additional `FileUrl.force_download` values allowed on file URL parts from
                the client (beyond `False`, which is always allowed). See
                [`UIAdapter.allowed_file_url_force_download`][pydantic_ai.ui.UIAdapter.allowed_file_url_force_download].
            allow_uploaded_files: Whether to honor `UploadedFile` references from client-submitted messages. See
                [`UIAdapter.allow_uploaded_files`][pydantic_ai.ui.UIAdapter.allow_uploaded_files].
            strip_workspace_refs: Whether to reset `ModelResponse.workspace_ref` on client-submitted messages. See
                [`UIAdapter.strip_workspace_refs`][pydantic_ai.ui.UIAdapter.strip_workspace_refs].
            allowed_content_types: Request media types to accept, as a CSRF control. See
                [`from_request`][pydantic_ai.ui.UIAdapter.from_request].
            **kwargs: Additional keyword arguments forwarded to [`from_request`][pydantic_ai.ui.UIAdapter.from_request].

        Returns:
            A streaming Starlette response with protocol-specific events encoded per the request's `Accept` header value.
        """
        try:
            from starlette.responses import Response  # noqa: F401  # pyright: ignore[reportUnusedImport]
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
                    manage_message_history=manage_message_history,
                    allowed_file_url_schemes=allowed_file_url_schemes,
                    allowed_file_url_force_download=allowed_file_url_force_download,
                    allow_uploaded_files=allow_uploaded_files,
                    strip_workspace_refs=strip_workspace_refs,
                    allowed_content_types=allowed_content_types,
                    **kwargs,
                ),
            )
        except ValidationError as e:
            return validation_error_response(e)

        return adapter.streaming_response(
            adapter.run_stream(
                message_history=message_history,
                deferred_tool_results=deferred_tool_results,
                conversation_id=conversation_id,
                run_id=run_id,
                cancellation_token=cancellation_token,
                timeout=timeout,
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
                capabilities=capabilities,
                workspace=workspace,
                on_complete=on_complete,
                on_cancel=on_cancel,
            ),
        )
