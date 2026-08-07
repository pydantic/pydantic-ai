"""Claude Code `stream-json` event stream implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import KW_ONLY, dataclass, field
from datetime import datetime
from typing import Literal
from uuid import uuid4

from ..._utils import now_utc
from ...exceptions import RunCancelled, UsageLimitExceeded
from ...messages import (
    CompactionPart,
    FilePart,
    FinishReason,
    FunctionToolResultEvent,
    NativeToolCallPart,
    NativeToolReturnPart,
    OutputToolResultEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
)
from ...output import OutputDataT
from ...run import AgentRunResultEvent
from ...tools import AgentDepsT
from .. import UIEventStream
from ._types import (
    AssistantMessage,
    AssistantRecord,
    ClaudeCodeEvent,
    CompactBoundaryRecord,
    ContentBlock,
    InitRecord,
    JSONValue,
    ResultRecord,
    StreamEventRecord,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
    UserRecord,
)

__all__ = ['NDJSON_CONTENT_TYPE', 'ClaudeCodeEventStream']

NDJSON_CONTENT_TYPE = 'application/x-ndjson'
"""Content type header value for newline-delimited JSON, which is what `stream-json` is."""

# Anthropic's `stop_reason` vocabulary, which the `result` record reports. `'error'` has no
# counterpart: an errored run says so through `is_error` and `terminal_reason` instead.
_STOP_REASON_MAP: dict[FinishReason, str | None] = {
    'stop': 'end_turn',
    'length': 'max_tokens',
    'tool_call': 'tool_use',
    'content_filter': 'refusal',
    'error': None,
}


def _uuid_str() -> str:
    return str(uuid4())


@dataclass
class ClaudeCodeEventStream(UIEventStream[None, ClaudeCodeEvent, AgentDepsT, OutputDataT]):
    """UI event stream transformer for the Claude Code CLI's `stream-json` output format.

    The format is what `claude -p --output-format stream-json --verbose` writes to stdout: one JSON
    object per line, opening with a `system`/`init` record and closing with a terminal `result`
    record. Emitting it lets any Pydantic AI agent stand in for the Claude Code CLI in tooling that
    consumes that stream, like [GitHub Agentic Workflows](https://github.com/github/gh-aw).

    Unlike the web-chat protocols, `stream-json` is message-level: each line carries one complete
    content block rather than a delta, so nothing is emitted until a part ends. Set
    `include_partial_messages` to additionally interleave the CLI's `stream_event` records.

    !!! note
        The format is not a versioned public spec. This implementation mirrors fixtures captured
        from Claude Code CLI 2.1.222, and models only the fields it can fill honestly.
    """

    run_input: None = None
    """Unused: `stream-json` is an output format, so there is no protocol-specific run input to accept."""

    _: KW_ONLY

    session_id: str = field(default_factory=_uuid_str)
    """Session identifier reported by the `init` and `result` records, and by every line's envelope."""
    model: str | None = None
    """Model name to report on the `init` record and on every `assistant` message, if known.

    Not knowable from the event stream, which is why it's a parameter: a run's model is only
    reported once the run has finished, by which time the `init` record has long been written.

    Free-form: the CLI reports vendor model ids like `claude-haiku-4-5-20251001`, but nothing
    consuming the stream parses the value, so a Pydantic AI model name is equally valid.
    """
    cwd: str = ''
    """Working directory reported by the `init` record, disclosed only if you opt in by setting it.

    An absolute server-side path says more about the machine than a consumer needs, so it's empty
    by default: consumers treat an empty `cwd` as absent and simply don't report one.
    """
    include_partial_messages: bool = False
    """Whether to additionally emit `stream_event` records carrying Anthropic-shaped streaming deltas.

    Mirrors the CLI's `--include-partial-messages` flag, which is a superset: the whole-block
    `assistant` records are emitted either way.
    """

    _started_at: datetime = field(default_factory=now_utc)
    _response_count: int = 0
    _block_index: int = 0
    _response_message_id: str = ''
    _text_buffer: str = ''
    _thinking_buffer: str = ''
    _thinking_signature: str = ''
    _args_fragment_emitted: bool = False
    _final_text: str = ''
    _usage: Usage | None = None
    _cost: float | None = None
    _stop_reason: str | None = None
    _error: Exception | None = None
    _cancellation: RunCancelled | None = None

    @property
    def content_type(self) -> str:
        return NDJSON_CONTENT_TYPE

    def encode_event(self, event: ClaudeCodeEvent) -> str:
        return json.dumps(event, separators=(',', ':')) + '\n'

    async def before_stream(self) -> AsyncIterator[ClaudeCodeEvent]:
        # Every field the stream accumulates is reset here rather than only initialized, so that
        # streaming a second run through the same instance reports that run and not the first one's.
        self._started_at = now_utc()
        self._response_count = 0
        self._block_index = 0
        self._response_message_id = ''
        self._text_buffer = ''
        self._thinking_buffer = ''
        self._thinking_signature = ''
        self._args_fragment_emitted = False
        self._final_text = ''
        self._usage = None
        self._cost = None
        self._stop_reason = None
        self._error = None
        self._cancellation = None
        record: InitRecord = {
            'type': 'system',
            'subtype': 'init',
            'cwd': self.cwd,
            'session_id': self.session_id,
            'tools': [],
            'mcp_servers': [],
            'permissionMode': 'default',
            'slash_commands': [],
            'output_style': 'default',
            'uuid': _uuid_str(),
        }
        if self.model is not None:
            record['model'] = self.model
        yield record

    async def after_stream(self) -> AsyncIterator[ClaudeCodeEvent]:
        # The `result` record is emitted here rather than from `handle_run_result` because gh-aw's
        # parser reads the last raw JSON line of the stream for its whole Information section: any
        # line after `result` (a partial-mode `message_stop`, say) collapses it to "No information
        # available". `after_stream` is the only hook nothing can follow.
        yield self._result_record()

    async def on_error(self, error: Exception) -> AsyncIterator[ClaudeCodeEvent]:
        self._error = error
        return
        yield  # Make this an async generator

    async def on_cancelled(self, cancelled: RunCancelled) -> AsyncIterator[ClaudeCodeEvent]:
        # Recorded apart from `on_error`, which the base implementation would otherwise delegate to:
        # a cancellation is a pause the caller asked for, not a failure, so the `result` record names
        # it with its own `terminal_reason` and reports no `errors`.
        self._cancellation = cancelled
        return
        yield  # Make this an async generator

    async def before_response(self) -> AsyncIterator[ClaudeCodeEvent]:
        self._response_count += 1
        self._response_message_id = f'msg_{_uuid_str()}'
        self._block_index = -1
        if self.include_partial_messages:
            message: dict[str, JSONValue] = {
                'id': self._response_message_id,
                'type': 'message',
                'role': 'assistant',
                'content': [],
                'stop_reason': None,
                'stop_sequence': None,
            }
            if self.model is not None:
                message['model'] = self.model
            yield self._stream_event({'type': 'message_start', 'message': message})

    async def after_response(self) -> AsyncIterator[ClaudeCodeEvent]:
        if self.include_partial_messages:
            yield self._stream_event({'type': 'message_delta', 'delta': {'stop_reason': None, 'stop_sequence': None}})
            yield self._stream_event({'type': 'message_stop'})

    async def handle_text_start(self, part: TextPart, follows_text: bool = False) -> AsyncIterator[ClaudeCodeEvent]:
        # A model can split one logical answer across adjacent text parts (interleaved citations do
        # this), which `stream-json` has no way to express: a content block is whole or it isn't. So
        # a run of adjacent parts stays inside one block, buffered until the last of them ends.
        if not follows_text:
            self._text_buffer = ''
            for event in self._open_block({'type': 'text', 'text': ''}):
                yield event
        if self.include_partial_messages and part.content:
            yield self._content_block_delta({'type': 'text_delta', 'text': part.content})

    async def handle_text_delta(self, delta: TextPartDelta) -> AsyncIterator[ClaudeCodeEvent]:
        if self.include_partial_messages and delta.content_delta:
            yield self._content_block_delta({'type': 'text_delta', 'text': delta.content_delta})

    async def handle_text_end(self, part: TextPart, followed_by_text: bool = False) -> AsyncIterator[ClaudeCodeEvent]:
        self._text_buffer += part.content
        if followed_by_text:
            return
        # `result.result` reports the run's final answer, which is the last text block it produced.
        self._final_text = self._text_buffer
        block: TextBlock = {'type': 'text', 'text': self._text_buffer}
        for event in self._close_block(block):
            yield event

    async def handle_thinking_start(
        self, part: ThinkingPart, follows_thinking: bool = False
    ) -> AsyncIterator[ClaudeCodeEvent]:
        if not follows_thinking:
            self._thinking_buffer = ''
            self._thinking_signature = ''
            for event in self._open_block({'type': 'thinking', 'thinking': '', 'signature': ''}):
                yield event
        if self.include_partial_messages and part.content:
            yield self._content_block_delta({'type': 'thinking_delta', 'thinking': part.content})

    async def handle_thinking_delta(self, delta: ThinkingPartDelta) -> AsyncIterator[ClaudeCodeEvent]:
        if not self.include_partial_messages:
            return
        if delta.content_delta:
            yield self._content_block_delta({'type': 'thinking_delta', 'thinking': delta.content_delta})
        if delta.signature_delta:
            yield self._content_block_delta({'type': 'signature_delta', 'signature': delta.signature_delta})

    async def handle_thinking_end(
        self, part: ThinkingPart, followed_by_thinking: bool = False
    ) -> AsyncIterator[ClaudeCodeEvent]:
        self._thinking_buffer += part.content
        if part.signature:
            # A run of adjacent thinking parts is signed by whichever of them carried a signature
            # last: one block can only claim one.
            self._thinking_signature = part.signature
        if followed_by_thinking:
            return
        block: ThinkingBlock = {'type': 'thinking', 'thinking': self._thinking_buffer}
        if self._thinking_signature:
            # Omitted rather than emitted empty for models that don't sign their thinking: an empty
            # signature is a value Anthropic rejects, whereas an absent one is simply unsigned.
            block['signature'] = self._thinking_signature
        for event in self._close_block(block):
            yield event

    async def handle_file(self, part: FilePart) -> AsyncIterator[ClaudeCodeEvent]:
        # Dropped deliberately: a model-generated file has no `stream-json` counterpart, whose
        # assistant content blocks are only ever text, thinking or tool_use.
        return
        yield  # Make this an async generator

    def handle_tool_call_start(self, part: ToolCallPart) -> AsyncIterator[ClaudeCodeEvent]:
        return self._handle_tool_call_start(part)

    def handle_builtin_tool_call_start(self, part: NativeToolCallPart) -> AsyncIterator[ClaudeCodeEvent]:
        return self._handle_tool_call_start(part)

    async def _handle_tool_call_start(self, part: ToolCallPart | NativeToolCallPart) -> AsyncIterator[ClaudeCodeEvent]:
        self._args_fragment_emitted = False
        for event in self._open_block(
            {'type': 'tool_use', 'id': part.tool_call_id, 'name': part.tool_name, 'input': {}}
        ):
            yield event
        # A `str` is the head of the JSON the deltas continue, so a client reassembling `partial_json`
        # needs it announced as the first fragment, raw: re-encoding would corrupt what it rebuilds.
        # `dict` args aren't a fragment of anything and are emitted whole once the call ends instead.
        if self.include_partial_messages and isinstance(part.args, str) and part.args:
            yield self._input_json_delta(part.args)
            self._args_fragment_emitted = True

    async def handle_tool_call_delta(self, delta: ToolCallPartDelta) -> AsyncIterator[ClaudeCodeEvent]:
        if self.include_partial_messages and isinstance(delta.args_delta, str) and delta.args_delta:
            yield self._input_json_delta(delta.args_delta)
            self._args_fragment_emitted = True

    def handle_tool_call_end(self, part: ToolCallPart) -> AsyncIterator[ClaudeCodeEvent]:
        return self._handle_tool_call_end(part)

    def handle_builtin_tool_call_end(self, part: NativeToolCallPart) -> AsyncIterator[ClaudeCodeEvent]:
        return self._handle_tool_call_end(part)

    async def _handle_tool_call_end(self, part: ToolCallPart | NativeToolCallPart) -> AsyncIterator[ClaudeCodeEvent]:
        # `args_as_json_str` rather than `args_as_dict` because a `dict` arg value is only guaranteed
        # to be JSON-encodable once Pydantic AI has encoded it: a `datetime` reaches `json.dumps` as
        # a `TypeError` that would break the line rather than the value it stands for.
        args_json = part.args_as_json_str()
        # Concatenating a call's `input_json_delta` fragments has to yield exactly its `input`, so a
        # call whose args never arrived as JSON text emits them here as one complete fragment. That
        # covers `dict` args, which Pydantic AI merges by key rather than by concatenation. A call
        # can't mix the two: applying a delta of the other kind is rejected upstream.
        if self.include_partial_messages and not self._args_fragment_emitted and part.args:
            yield self._input_json_delta(args_json)
        block: ToolUseBlock = {
            'type': 'tool_use',
            'id': part.tool_call_id,
            'name': part.tool_name,
            'input': json.loads(args_json),
        }
        for event in self._close_block(block):
            yield event

    def handle_function_tool_result(self, event: FunctionToolResultEvent) -> AsyncIterator[ClaudeCodeEvent]:
        return self._handle_tool_result(event.part)

    def handle_output_tool_result(self, event: OutputToolResultEvent) -> AsyncIterator[ClaudeCodeEvent]:
        return self._handle_tool_result(event.part)

    def handle_builtin_tool_return(self, part: NativeToolReturnPart) -> AsyncIterator[ClaudeCodeEvent]:
        return self._handle_tool_result(part)

    async def _handle_tool_result(
        self, part: ToolReturnPart | NativeToolReturnPart | RetryPromptPart
    ) -> AsyncIterator[ClaudeCodeEvent]:
        if isinstance(part, RetryPromptPart):
            content: str | list[TextBlock] = part.model_response()
            is_error = True
        else:
            # Multimodal returns become the array content form. The files themselves are dropped:
            # the text references them by identifier, and mapping them onto Anthropic's binary
            # content blocks is a separate concern from the event mapping.
            text, files = part.model_response_str_and_user_content(wrap_if_error=False)
            content = [{'type': 'text', 'text': text}] if files else text
            # Only `'failed'` is an error. An `'interrupted'` call (cut off before it produced a
            # result) and a `'denied'` one are not, and must not render as failures.
            is_error = part.outcome == 'failed'

        block: ToolResultBlock = {'type': 'tool_result', 'tool_use_id': part.tool_call_id, 'content': content}
        if is_error:
            block['is_error'] = True

        record: UserRecord = {
            'type': 'user',
            'message': {'role': 'user', 'content': [block]},
            'parent_tool_use_id': None,
            'session_id': self.session_id,
            'uuid': _uuid_str(),
            'timestamp': self._timestamp(),
        }
        yield record

    async def handle_compaction(self, part: CompactionPart) -> AsyncIterator[ClaudeCodeEvent]:
        record: CompactBoundaryRecord = {
            'type': 'system',
            'subtype': 'compact_boundary',
            # Compaction in Pydantic AI is provider-driven, never a user's `/compact`, so it's
            # always `'auto'`. The CLI also reports `pre_tokens`, which the seam doesn't expose.
            'compact_metadata': {'trigger': 'auto'},
            'session_id': self.session_id,
            'uuid': _uuid_str(),
            'timestamp': self._timestamp(),
        }
        yield record

    async def handle_run_result(self, event: AgentRunResultEvent) -> AsyncIterator[ClaudeCodeEvent]:
        # The `result` record itself is emitted by `after_stream`; this only captures what it needs.
        usage = event.result.usage
        self._usage = {
            'input_tokens': usage.input_tokens,
            'output_tokens': usage.output_tokens,
            'cache_creation_input_tokens': usage.cache_write_tokens,
            'cache_read_input_tokens': usage.cache_read_tokens,
        }
        if usage.cost:
            self._cost = float(usage.cost)
        if finish_reason := event.result.response.finish_reason:
            self._stop_reason = _STOP_REASON_MAP.get(finish_reason)
        return
        yield  # Make this an async generator

    def _timestamp(self) -> str:
        return now_utc().isoformat(timespec='milliseconds').removesuffix('+00:00') + 'Z'

    def _stream_event(self, event: dict[str, JSONValue]) -> StreamEventRecord:
        return {
            'type': 'stream_event',
            'event': event,
            'parent_tool_use_id': None,
            'session_id': self.session_id,
            'uuid': _uuid_str(),
        }

    def _content_block_delta(self, delta: dict[str, JSONValue]) -> StreamEventRecord:
        return self._stream_event({'type': 'content_block_delta', 'index': self._block_index, 'delta': delta})

    def _input_json_delta(self, partial_json: str) -> StreamEventRecord:
        return self._content_block_delta({'type': 'input_json_delta', 'partial_json': partial_json})

    def _open_block(self, content_block: dict[str, JSONValue]) -> list[ClaudeCodeEvent]:
        """Advance to the next content block, announcing it in partial-messages mode."""
        self._block_index += 1
        if not self.include_partial_messages:
            return []
        return [
            self._stream_event(
                {'type': 'content_block_start', 'index': self._block_index, 'content_block': content_block}
            )
        ]

    def _close_block(self, block: ContentBlock) -> list[ClaudeCodeEvent]:
        """Emit the whole-block `assistant` record, then close the block in partial-messages mode.

        The CLI emits these in this order: the `assistant` record precedes its `content_block_stop`.
        """
        message: AssistantMessage = {
            'id': self._response_message_id,
            'type': 'message',
            'role': 'assistant',
            'content': [block],
            # A response's blocks are written as they complete, before the model has reported why it
            # stopped, so the CLI leaves this `null` on every `assistant` line. The run's finish
            # reason surfaces on the `result` record instead.
            'stop_reason': None,
            'stop_sequence': None,
        }
        if self.model is not None:
            message['model'] = self.model
        # `message.usage` is omitted: Pydantic AI only reports usage once the run has finished, and
        # fabricating zeros per response would misreport it as free.
        record: AssistantRecord = {
            'type': 'assistant',
            'message': message,
            'parent_tool_use_id': None,
            'session_id': self.session_id,
            'uuid': _uuid_str(),
            'timestamp': self._timestamp(),
        }
        events: list[ClaudeCodeEvent] = [record]
        if self.include_partial_messages:
            events.append(self._stream_event({'type': 'content_block_stop', 'index': self._block_index}))
        return events

    def _result_record(self) -> ResultRecord:
        error = self._cancellation if self._cancellation is not None else self._error
        subtype: Literal['success', 'error_max_turns'] = 'success'
        if self._cancellation is not None:
            # The CLI has no cancellation record, so this is our own vocabulary: a cancelled run is
            # reported as an error so nothing reads a stopped run as a finished one, under a
            # `terminal_reason` that says it was stopped rather than that it broke.
            terminal_reason = 'cancelled'
        elif error is None:
            terminal_reason = 'completed'
        elif isinstance(error, UsageLimitExceeded):
            subtype = 'error_max_turns'
            terminal_reason = 'max_turns'
        else:
            # Matching the CLI, whose non-turn-limit failures keep `subtype: 'success'` and report
            # the failure through `is_error` and `terminal_reason`.
            terminal_reason = 'error'

        record: ResultRecord = {
            'type': 'result',
            'subtype': subtype,
            'is_error': error is not None,
            'terminal_reason': terminal_reason,
            # One turn per model response, which is what the CLI counts and what gh-aw compares
            # against its max-turns budget, so it must never be inflated.
            'num_turns': self._response_count,
            'duration_ms': int((now_utc() - self._started_at).total_seconds() * 1000),
            'result': str(error) if error is not None else self._final_text,
            'session_id': self.session_id,
            'uuid': _uuid_str(),
        }
        if self._stop_reason is not None:
            record['stop_reason'] = self._stop_reason
        if self._usage is not None:
            record['usage'] = self._usage
        if self._cost is not None:
            # A `0` cost is indistinguishable from an absent one downstream, so only a real cost is
            # ever reported.
            record['total_cost_usd'] = self._cost
        if self._error is not None:
            # The CLI's own `success`-subtype failures carry no `errors`, but gh-aw's parser reads
            # `lastEntry.errors` to render its Errors block (`log_parser_shared.cjs:336`), so a real
            # failure reports one. A cancellation doesn't: nothing went wrong.
            record['errors'] = [str(self._error)]
        return record
