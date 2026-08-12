"""Record types for the Claude Code CLI `stream-json` output format.

The format is not a versioned public spec, so these shapes mirror the golden fixtures captured from
Claude Code CLI 2.1.222 in `tests/assets/claude_code_stream_json/`. Only the fields
[`ClaudeCodeEventStream`][pydantic_ai.ui.claude_code.ClaudeCodeEventStream] can fill honestly are
modelled: the CLI emits considerably more, and consumers treat every field as optional.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

from typing_extensions import NotRequired, TypedDict

JSONValue: TypeAlias = 'str | int | float | bool | None | list[JSONValue] | dict[str, JSONValue]'
"""Any JSON value, used where the payload's shape is the caller's (or another vendor's) to decide."""


class Usage(TypedDict):
    """Anthropic-native token counts, carried by the terminal `result` record."""

    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


class TextBlock(TypedDict):
    """An assistant text block, or one element of a `tool_result`'s content array."""

    type: Literal['text']
    text: str


class ThinkingBlock(TypedDict):
    """An assistant thinking block."""

    type: Literal['thinking']
    thinking: str
    signature: NotRequired[str]


class ToolUseBlock(TypedDict):
    """An assistant tool call block. `id` pairs with the matching `ToolResultBlock.tool_use_id`."""

    type: Literal['tool_use']
    id: str
    name: str
    input: dict[str, JSONValue]


ContentBlock: TypeAlias = TextBlock | ThinkingBlock | ToolUseBlock
"""A block of an assistant message's content."""


class ToolResultBlock(TypedDict):
    """A tool result block, carried by a `user` record."""

    type: Literal['tool_result']
    tool_use_id: str
    content: str | list[TextBlock]
    is_error: NotRequired[bool]


class AssistantMessage(TypedDict):
    """The Anthropic-shaped message of an `assistant` record."""

    id: str
    type: Literal['message']
    role: Literal['assistant']
    model: NotRequired[str]
    content: list[ContentBlock]
    stop_reason: None
    stop_sequence: None


class UserMessage(TypedDict):
    """The message of a `user` record, only ever carrying tool results."""

    role: Literal['user']
    content: list[ToolResultBlock]


class CompactMetadata(TypedDict):
    """Metadata of a `compact_boundary` record."""

    trigger: Literal['auto', 'manual']


class InitRecord(TypedDict):
    """The `system`/`init` record, always the first line of the stream."""

    type: Literal['system']
    subtype: Literal['init']
    cwd: str
    session_id: str
    model: NotRequired[str]
    tools: list[str]
    mcp_servers: list[JSONValue]
    # The CLI spells this key camelCase among its snake_case siblings; consumers match on it verbatim.
    permissionMode: str
    slash_commands: list[str]
    output_style: str
    uuid: str


class AssistantRecord(TypedDict):
    """An `assistant` record, carrying exactly one content block."""

    type: Literal['assistant']
    message: AssistantMessage
    parent_tool_use_id: None
    session_id: str
    uuid: str
    timestamp: str


class UserRecord(TypedDict):
    """A `user` record, carrying exactly one tool result block."""

    type: Literal['user']
    message: UserMessage
    parent_tool_use_id: None
    session_id: str
    uuid: str
    timestamp: str


class CompactBoundaryRecord(TypedDict):
    """A `system`/`compact_boundary` record, marking a history compaction."""

    type: Literal['system']
    subtype: Literal['compact_boundary']
    compact_metadata: CompactMetadata
    session_id: str
    uuid: str
    timestamp: str


class StreamEventRecord(TypedDict):
    """A `stream_event` record, wrapping a raw Anthropic server-sent event.

    Only emitted when [`ClaudeCodeEventStream.include_partial_messages`][pydantic_ai.ui.claude_code.ClaudeCodeEventStream.include_partial_messages]
    is set, mirroring the CLI's `--include-partial-messages` flag.
    """

    type: Literal['stream_event']
    event: dict[str, JSONValue]
    parent_tool_use_id: None
    session_id: str
    uuid: str


class ResultRecord(TypedDict):
    """The terminal `result` record, always the last line of the stream."""

    type: Literal['result']
    subtype: Literal['success', 'error_max_turns']
    is_error: bool
    terminal_reason: str
    num_turns: int
    duration_ms: int
    result: str
    session_id: str
    uuid: str
    stop_reason: NotRequired[str]
    usage: NotRequired[Usage]
    total_cost_usd: NotRequired[float]
    errors: NotRequired[list[str]]


ClaudeCodeEvent: TypeAlias = (
    InitRecord | AssistantRecord | UserRecord | CompactBoundaryRecord | StreamEventRecord | ResultRecord
)
"""One line of Claude Code `stream-json` output."""
