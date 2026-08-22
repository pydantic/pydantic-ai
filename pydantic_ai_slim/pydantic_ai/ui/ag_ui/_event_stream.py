"""AG-UI protocol adapter for Pydantic AI agents.

This module provides classes for integrating Pydantic AI agents with the AG-UI protocol,
enabling streaming event-based communication for interactive AI applications.
"""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator, Iterable
from dataclasses import KW_ONLY, dataclass, field
from uuid import uuid4

from pydantic_core import to_json

from ..._utils import now_utc
from ..._uuid import uuid7
from ...exceptions import RunCancelled
from ...messages import (
    CompactionPart,
    FunctionToolResultEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    OutputToolResultEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolAvailabilityDeltaEvent,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from ...output import OutputDataT
from ...tools import AgentDepsT, DeferredToolRequests
from .. import SSE_CONTENT_TYPE, NativeEvent, UIEventStream
from .._adapter import compaction_payload
from ._interrupt import (
    HAS_INTERRUPTS,
    RunFinishedInterruptOutcome,
    RunFinishedSuccessOutcome,
    approval_to_interrupt,
)
from ._utils import (
    ACTIVITY_EVENTS_VERSION,
    BUILTIN_TOOL_CALL_ID_PREFIX,
    COMPACTION_ACTIVITY_TYPE,
    DEFAULT_AG_UI_VERSION,
    INTERRUPTS_VERSION,
    REASONING_VERSION,
    TOOL_AVAILABILITY_DELTA_ACTIVITY_TYPE,
    dump_tool_return_content,
    parse_ag_ui_version,
    tool_kind_encrypted_value,
)

try:
    from ag_ui.core import (
        BaseEvent,
        EventType,
        RunAgentInput,
        RunErrorEvent,
        RunFinishedEvent,
        RunStartedEvent,
        TextMessageContentEvent,
        TextMessageEndEvent,
        TextMessageStartEvent,
        ToolCallArgsEvent,
        ToolCallEndEvent,
        ToolCallResultEvent,
        ToolCallStartEvent,
    )
    from ag_ui.encoder import EventEncoder

except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` package to use AG-UI integration, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e


__all__ = [
    'AGUIEventStream',
    'DEFAULT_AG_UI_VERSION',
    'RunAgentInput',
    'RunStartedEvent',
    'RunFinishedEvent',
]


class _GeneratedID(str):
    """An ID the stream minted for itself, as opposed to one the caller passed in.

    Warning that a caller's ID lost to `run_input` means telling those two apart, and comparing
    against the run input's ID can't: a generated ID differs from it too. Marking the generated one
    keeps `thread_id` and `run_id` typed `str` and valid from construction, where an unset sentinel
    would push a non-`str` into every event that emits them.
    """


def _generate_id() -> str:
    return _GeneratedID(uuid7())


@dataclass
class AGUIEventStream(UIEventStream[RunAgentInput, BaseEvent, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the Agent-User Interaction (AG-UI) protocol."""

    ag_ui_version: str = DEFAULT_AG_UI_VERSION

    _: KW_ONLY

    thread_id: str = field(default_factory=_generate_id)
    """The AG-UI thread ID to report on `RUN_STARTED` and `RUN_FINISHED`.

    A [`run_input`][pydantic_ai.ui.UIEventStream.run_input] takes precedence: when one is given, its
    thread ID replaces whatever was passed here, with a `UserWarning`. Without a run input, set it to
    the ID the conversation already has in your own transport, or leave it to default to a new UUID —
    but note that the default is minted per stream, so a conversation that spans more than one run
    needs to pass its own.

    This identifies the conversation to the frontend. It is what
    [`AGUIAdapter`][pydantic_ai.ui.ag_ui.AGUIAdapter] maps onto the agent's `conversation_id` on the
    request path, so passing the conversation ID the agent run itself uses keeps the frontend and
    the agent's traces correlated.
    """

    run_id: str = field(default_factory=_generate_id)
    """The AG-UI run ID to report on `RUN_STARTED` and `RUN_FINISHED`.

    A [`run_input`][pydantic_ai.ui.UIEventStream.run_input] takes precedence: when one is given, its
    run ID replaces whatever was passed here, with a `UserWarning`. Without a run input, set it to
    the ID the run already has in your own transport, or leave it to default to a new UUID.

    This is the protocol's run ID, not the agent run ID that
    [`UIAdapter.run_stream()`][pydantic_ai.ui.UIAdapter.run_stream] takes as `run_id`; the two are
    never wired together.
    """

    _use_reasoning: bool = field(default=False, init=False)
    _reasoning_message_id: str | None = None
    _reasoning_started: bool = False
    _reasoning_text: bool = False
    _builtin_tool_call_ids: dict[str, str] = field(default_factory=dict[str, str])
    _error: bool = False
    _cancelled_run: bool = False

    def __post_init__(self) -> None:
        self._use_reasoning = parse_ag_ui_version(self.ag_ui_version) >= REASONING_VERSION
        if (run_input := self.run_input) is not None:
            # A request's own identity wins: the frontend picked these and correlates the run by them,
            # so they're not something the server gets to substitute.
            if overridden := [
                name
                for name, value in (('thread_id', self.thread_id), ('run_id', self.run_id))
                if not isinstance(value, _GeneratedID)
            ]:
                names = ' and '.join(f'`{name}`' for name in overridden)
                warnings.warn(
                    f'{names} {"is" if len(overridden) == 1 else "are"} ignored when a `run_input` is given; '
                    'the run input carries the identity the frontend correlates the run by.',
                    UserWarning,
                    stacklevel=3,
                )
            self.thread_id = run_input.thread_id
            self.run_id = run_input.run_id

    @property
    def _event_encoder(self) -> EventEncoder:
        return EventEncoder(accept=self.accept or SSE_CONTENT_TYPE)

    @property
    def content_type(self) -> str:
        return self._event_encoder.get_content_type()

    def encode_event(self, event: BaseEvent) -> str:
        return self._event_encoder.encode(event)

    @staticmethod
    def _get_timestamp() -> int:
        return int(now_utc().timestamp() * 1_000)

    async def handle_event(self, event: NativeEvent) -> AsyncIterator[BaseEvent]:
        """Override to set timestamps on all AG-UI events."""
        async for agui_event in super().handle_event(event):
            if agui_event.timestamp is None:
                agui_event.timestamp = self._get_timestamp()
            yield agui_event

    async def before_stream(self) -> AsyncIterator[BaseEvent]:
        yield RunStartedEvent(
            thread_id=self.thread_id,
            run_id=self.run_id,
            timestamp=self._get_timestamp(),
        )

    async def before_response(self) -> AsyncIterator[BaseEvent]:
        # Prevent parts from a subsequent response being tied to parts from an earlier response.
        # See https://github.com/pydantic/pydantic-ai/issues/3316
        self.new_message_id()
        return
        yield  # Make this an async generator

    async def after_stream(self) -> AsyncIterator[BaseEvent]:
        if self._error:
            return

        if self._cancelled_run:
            # AG-UI has no cancelled outcome; revisit when the protocol fills this spec gap:
            # https://github.com/ag-ui-protocol/ag-ui/issues/880
            yield RunFinishedEvent(
                thread_id=self.thread_id,
                run_id=self.run_id,
                timestamp=self._get_timestamp(),
            )
            return

        # `RunFinishedEvent.outcome` only exists in ag-ui-protocol >= 0.1.19. `ConfiguredBaseModel`
        # allows extra fields, so passing `outcome=None` on the old path wouldn't raise — but it
        # would serialize an `outcome` field that pre-interrupt clients don't expect, so we branch
        # to omit it entirely.
        if HAS_INTERRUPTS:
            yield RunFinishedEvent(
                thread_id=self.thread_id,
                run_id=self.run_id,
                outcome=self._build_outcome(),
                timestamp=self._get_timestamp(),
            )
        else:
            yield RunFinishedEvent(
                thread_id=self.thread_id,
                run_id=self.run_id,
                timestamp=self._get_timestamp(),
            )

    def _build_outcome(self) -> RunFinishedInterruptOutcome | RunFinishedSuccessOutcome | None:
        """Build the `RunFinishedEvent.outcome` from the final agent result.

        Returns `None` when the negotiated AG-UI version predates interrupts, so an old
        client doesn't receive a field it doesn't understand even if the server SDK
        does.
        """
        if parse_ag_ui_version(self.ag_ui_version) < INTERRUPTS_VERSION:
            # `outcome=None` only reaches an old client as a bare `RUN_FINISHED` because the SDK's
            # `EventEncoder` serializes with `exclude_none=True`; the field is valid on this SDK.
            return None
        output = self._result.output if self._result else None
        if isinstance(output, DeferredToolRequests) and output.approvals:
            return RunFinishedInterruptOutcome(
                interrupts=[approval_to_interrupt(call, output.metadata) for call in output.approvals],
            )
        return RunFinishedSuccessOutcome()

    async def on_error(self, error: Exception) -> AsyncIterator[BaseEvent]:
        self._error = True
        yield RunErrorEvent(message=str(error), timestamp=self._get_timestamp())

    async def on_cancelled(self, cancelled: RunCancelled) -> AsyncIterator[BaseEvent]:
        self._cancelled_run = True
        return
        yield

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[BaseEvent]:
        if follows_text:
            message_id = self.message_id
        else:
            message_id = self.new_message_id()
            yield TextMessageStartEvent(message_id=message_id)

        if part.content:  # pragma: no branch
            yield TextMessageContentEvent(message_id=message_id, delta=part.content)

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[BaseEvent]:
        if delta.content_delta:  # pragma: no branch
            yield TextMessageContentEvent(message_id=self.message_id, delta=delta.content_delta)

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[BaseEvent]:
        if not followed_by_text:
            yield TextMessageEndEvent(message_id=self.message_id)

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[BaseEvent]:
        self._reasoning_message_id = str(uuid4())
        self._reasoning_started = False

        if self._use_reasoning:
            from ._thinking_0_13 import handle_thinking_start as _impl
        else:
            from ._thinking_0_10 import handle_thinking_start as _impl
        async for event in _impl(self, part):
            yield event

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[BaseEvent]:
        if not delta.content_delta:
            return  # pragma: no cover

        assert self._reasoning_message_id is not None, (
            'handle_thinking_start must be called before handle_thinking_delta'
        )

        if self._use_reasoning:
            from ._thinking_0_13 import handle_thinking_delta as _impl
        else:
            from ._thinking_0_10 import handle_thinking_delta as _impl
        async for event in _impl(self, delta):
            yield event

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[BaseEvent]:
        assert self._reasoning_message_id is not None, 'handle_thinking_start must be called before handle_thinking_end'

        if self._use_reasoning:
            from ._thinking_0_13 import handle_thinking_end as _impl
        else:
            from ._thinking_0_10 import handle_thinking_end as _impl
        async for event in _impl(self, part):
            yield event

    def handle_tool_call_start(self, part: ToolCallPart | NativeToolCallPart) -> AsyncIterator[BaseEvent]:
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: NativeToolCallPart) -> AsyncIterator[BaseEvent]:
        tool_call_id = part.tool_call_id
        builtin_tool_call_id = '|'.join([BUILTIN_TOOL_CALL_ID_PREFIX, part.provider_name or '', tool_call_id])
        self._builtin_tool_call_ids[tool_call_id] = builtin_tool_call_id
        tool_call_id = builtin_tool_call_id

        return self._handle_tool_call_start(part, tool_call_id)

    async def _handle_tool_call_start(
        self, part: ToolCallPart | NativeToolCallPart, tool_call_id: str | None = None
    ) -> AsyncIterator[BaseEvent]:
        tool_call_id = tool_call_id or part.tool_call_id
        parent_message_id = self.message_id

        yield ToolCallStartEvent(
            tool_call_id=tool_call_id, tool_call_name=part.tool_name, parent_message_id=parent_message_id
        )
        if self._use_reasoning and (encrypted_value := tool_kind_encrypted_value(part.tool_kind)):
            # Clients echo this back as `ToolCall.encrypted_value`, so `tool_kind` survives
            # streaming-built histories. The event is 0.1.13+, hence the gated import.
            from ag_ui.core import ReasoningEncryptedValueEvent

            yield ReasoningEncryptedValueEvent(
                subtype='tool-call', entity_id=tool_call_id, encrypted_value=encrypted_value
            )
        if part.args:
            # A `str` is emitted raw: the args this first event carries can be a partial JSON fragment
            # that only becomes valid once the following deltas are concatenated, and
            # `args_as_json_str()` would degrade it to the `INVALID_JSON` wrapper. `dict` args always
            # arrive complete, so the helper is still the right encoder for them.
            yield ToolCallArgsEvent(
                tool_call_id=tool_call_id,
                delta=part.args if isinstance(part.args, str) else part.args_as_json_str(),
            )

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[BaseEvent]:
        tool_call_id = delta.tool_call_id
        assert tool_call_id, '`ToolCallPartDelta.tool_call_id` must be set'
        if tool_call_id in self._builtin_tool_call_ids:
            tool_call_id = self._builtin_tool_call_ids[tool_call_id]
        yield ToolCallArgsEvent(
            tool_call_id=tool_call_id,
            delta=delta.args_delta if isinstance(delta.args_delta, str) else to_json(delta.args_delta).decode(),
        )

    async def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[BaseEvent]:
        yield ToolCallEndEvent(tool_call_id=part.tool_call_id)

    async def handle_builtin_tool_call_end(self, part: NativeToolCallPart) -> AsyncIterator[BaseEvent]:
        builtin_id = self._builtin_tool_call_ids[part.tool_call_id]
        yield ToolCallEndEvent(tool_call_id=builtin_id)

    async def handle_builtin_tool_return(self, part: NativeToolReturnPart) -> AsyncIterator[BaseEvent]:
        tool_call_id = self._builtin_tool_call_ids[part.tool_call_id]
        # Use a one-off message ID instead of `self.new_message_id()` to avoid
        # mutating `self.message_id`, which is used as `parent_message_id` for
        # subsequent tool calls in the same response.
        message_id = str(uuid4())
        yield ToolCallResultEvent(
            message_id=message_id,
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=tool_call_id,
            content=_tool_return_content(part),
        )
        async for event in self._handle_tool_return_outcome(part, message_id):
            yield event

    async def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[BaseEvent]:
        async for e in self._handle_tool_result(event.part):
            yield e

    async def handle_tool_availability_delta(self, event: ToolAvailabilityDeltaEvent) -> AsyncIterator[BaseEvent]:
        if parse_ag_ui_version(self.ag_ui_version) < ACTIVITY_EVENTS_VERSION:
            return

        from ag_ui.core import ActivitySnapshotEvent

        part = event.part
        yield ActivitySnapshotEvent(
            message_id=str(uuid4()),
            activity_type=TOOL_AVAILABILITY_DELTA_ACTIVITY_TYPE,
            content={'added': part.tools_added, 'tool_call_id': part.tool_call_id},
        )

    async def handle_compaction(self, part: CompactionPart) -> AsyncIterator[BaseEvent]:
        if parse_ag_ui_version(self.ag_ui_version) < ACTIVITY_EVENTS_VERSION:
            return

        from ag_ui.core import ActivitySnapshotEvent

        yield ActivitySnapshotEvent(
            message_id=str(uuid4()),
            activity_type=COMPACTION_ACTIVITY_TYPE,
            content=compaction_payload(part),
        )

    async def handle_output_tool_result(self, event: OutputToolResultEvent) -> AsyncIterator[BaseEvent]:
        async for e in self._handle_tool_result(event.part):
            yield e

    async def _handle_tool_result(self, result: ToolReturnPart | RetryPromptPart) -> AsyncIterator[BaseEvent]:
        if isinstance(result, RetryPromptPart):
            output = result.model_response()
        else:
            output = _tool_return_content(result)

        # Regular tool results arrive after `ToolCallEvent` moved the stream to the request turn.
        # The next model response starts with `PartStartEvent`, whose request-to-response transition
        # replaces this ID in `before_response()`. Native tool returns differ: another native call can
        # follow inside the same response, so that path must use a one-off ID without mutating this one.
        message_id = self.new_message_id()
        yield ToolCallResultEvent(
            message_id=message_id,
            type=EventType.TOOL_CALL_RESULT,
            role='tool',
            tool_call_id=result.tool_call_id,
            content=output,
        )

        # ToolCallResultEvent.content may hold user parts (e.g. text, images) that AG-UI does not currently have events for

        if isinstance(result, ToolReturnPart):
            async for event in self._handle_tool_return_outcome(result, message_id):
                yield event

            # Check for AG-UI events returned by tool calls.
            possible_event = result.metadata or result.content
            if isinstance(possible_event, BaseEvent):
                yield possible_event
            elif isinstance(possible_event, str | bytes):  # pragma: no branch
                # Avoid iterable check for strings and bytes.
                pass
            elif isinstance(possible_event, Iterable):  # pragma: no branch
                for item in possible_event:  # type: ignore[reportUnknownMemberType]
                    if isinstance(item, BaseEvent):  # pragma: no branch
                        yield item

    async def _handle_tool_return_outcome(
        self, part: NativeToolReturnPart | ToolReturnPart, message_id: str
    ) -> AsyncIterator[BaseEvent]:
        # `ToolCallResultEvent` cannot express an outcome. This is the only standard event whose
        # reducer can attach continuity data to the resulting `ToolMessage`; it must follow the
        # result event because the reducer does not queue metadata for an entity that does not exist.
        if self._use_reasoning and (encrypted_value := tool_kind_encrypted_value(None, part.outcome)):
            from ag_ui.core import ReasoningEncryptedValueEvent

            yield ReasoningEncryptedValueEvent(subtype='message', entity_id=message_id, encrypted_value=encrypted_value)


def _tool_return_content(part: NativeToolReturnPart | ToolReturnPart) -> str:
    """Serialize a tool return's full content for a `ToolCallResultEvent`.

    Uses the same serialization as history `dump_messages` (see
    [`dump_tool_return_content`][pydantic_ai.ui.ag_ui._utils.dump_tool_return_content]), so files a tool
    returns ride inline in the streamed `ToolCallResultEvent.content` and survive the round-trip: a frontend
    that echoes the content back on the next request gets `BinaryContent`/`ImageUrl`/... rehydrated on load,
    so the file can be sent to the model again rather than collapsing to a text placeholder.
    """
    return dump_tool_return_content(part.content)
