"""What the browser sees of a CLAI session: transcript items, the messages that change them, and how events become them.

Pure translation, with no server or plugin in it. Wire payloads are kw-only dataclasses, serialized by
`dump` with Pydantic; the browser's `web-ui/src/protocol.ts` mirrors these shapes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

from pydantic import TypeAdapter

from pydantic_ai import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.ask_user import AskUserRequest
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent

RESULT_CHARS = 20_000
"""The most of a tool result the browser receives; the model's copy is unchanged."""

ItemKind = Literal['user', 'assistant', 'thinking', 'tool']
ToolState = Literal['running', 'done', 'error', 'interrupted']


@dataclass(kw_only=True)
class Diff:
    """One file change a tool made, as a unified diff."""

    path: str
    diff: str
    truncated: bool = False


@dataclass(kw_only=True)
class Item:
    """One transcript entry: a prompt, an answer, a thinking block, or a tool call with what it produced."""

    id: str
    kind: ItemKind
    text: str = ''
    """The prompt, answer, or thinking text; a tool's result."""
    name: str = ''
    """The tool's name."""
    args: str = ''
    """The tool's arguments as JSON."""
    state: ToolState = 'done'
    output: str = ''
    """What a shell command printed while it ran."""
    exit_code: int | None = None
    diffs: list[Diff] = field(default_factory=list[Diff])


@dataclass(kw_only=True)
class Option:
    """One choice a question offers."""

    label: str
    description: str | None = None


@dataclass(kw_only=True)
class QuestionInfo:
    """One multiple-choice question, as the browser draws it."""

    header: str
    question: str
    options: list[Option]
    multi_select: bool = False


@dataclass(kw_only=True)
class Questions:
    """An open `ask_user_question` request the browser can answer."""

    id: str
    questions: list[QuestionInfo]
    asker: str | None = None
    """Who asks when it is not the conversation itself, such as a delegated task."""


@dataclass(kw_only=True)
class SessionInfo:
    """The conversation the shell holds."""

    id: str
    title: str | None
    workspace: str


@dataclass(kw_only=True)
class Usage:
    """Tokens and requests one turn used."""

    input_tokens: int = 0
    output_tokens: int = 0
    requests: int = 0


@dataclass(kw_only=True)
class StatusInfo:
    """The status row's figures, sent every `STATUS_INTERVAL` while a turn streams."""

    model: str
    activity: str
    context_tokens: int | None = None
    context_window: int | None = None
    cost: str | None = None
    streamed_tokens: int = 0
    """The turn's output so far, estimated from the characters streamed, as the terminal's `~N streamed tokens`."""
    output_tokens: int | None = None
    """The last turn's reported output tokens, once it ends."""
    run: Usage | None = None
    """What the running turn's model requests have reported so far; it grows as each request completes."""
    code_mode: bool | None = None
    """The `run.code_mode` setting; `None` without CLAI's settings."""
    speculation: bool | None = None
    """The `run.speculative_code_mode` setting, which turns code mode on for its runs too."""


@dataclass(kw_only=True)
class Palette:
    """A termflow palette as `/theme` applies it to the terminal: background, foreground, and the 16 ANSI slots."""

    bg: str
    fg: str
    ansi: list[str]


@dataclass(kw_only=True)
class ThemeInfo:
    """The `/theme` choice: `default` is Pydantic's own colours, in the browser's light or dark mode."""

    name: str
    names: list[str]
    palette: Palette | None = None


@dataclass(kw_only=True)
class Snapshot:
    """Everything a newly connected browser needs: sent first on every connection and conversation switch."""

    type: Literal['snapshot'] = 'snapshot'
    session: SessionInfo
    items: list[Item]
    running: bool
    status: StatusInfo
    questions: list[Questions]
    theme: ThemeInfo


@dataclass(kw_only=True)
class ItemAdded:
    """A new transcript item."""

    type: Literal['item'] = 'item'
    item: Item


@dataclass(kw_only=True)
class TextDelta:
    """More text for an answer or thinking block."""

    type: Literal['delta'] = 'delta'
    id: str
    text: str


@dataclass(kw_only=True)
class ToolOutput:
    """More of what a running shell command printed."""

    type: Literal['tool_output'] = 'tool_output'
    id: str
    text: str


@dataclass(kw_only=True)
class ToolDone:
    """A tool call finished, with its result."""

    type: Literal['tool_done'] = 'tool_done'
    id: str
    text: str
    state: ToolState


@dataclass(kw_only=True)
class ShellDone:
    """A shell command exited."""

    type: Literal['shell_done'] = 'shell_done'
    id: str
    exit_code: int | None


@dataclass(kw_only=True)
class DiffAdded:
    """A file change a tool call made."""

    type: Literal['diff'] = 'diff'
    id: str
    diff: Diff


@dataclass(kw_only=True)
class TurnStarted:
    """A turn started; `item` is its prompt, or `None` for a turn a finished task woke."""

    type: Literal['turn_start'] = 'turn_start'
    item: Item | None = None


@dataclass(kw_only=True)
class TurnEnded:
    """The turn finished, failed, or was cancelled."""

    type: Literal['turn_end'] = 'turn_end'
    outcome: Literal['completed', 'failed', 'cancelled']
    error: str | None = None
    usage: Usage | None = None


@dataclass(kw_only=True)
class StatusChanged:
    """The status row's figures changed."""

    type: Literal['status'] = 'status'
    status: StatusInfo


@dataclass(kw_only=True)
class QuestionAsked:
    """The model asks the user something; answer it with `POST /api/questions/ID`."""

    type: Literal['question'] = 'question'
    questions: Questions


@dataclass(kw_only=True)
class QuestionClosed:
    """A question was answered, here or in the terminal, or withdrawn."""

    type: Literal['question_closed'] = 'question_closed'
    id: str


@dataclass(kw_only=True)
class Notice:
    """A `/command`'s result, or another line the browser should show once."""

    type: Literal['notice'] = 'notice'
    text: str
    level: Literal['info', 'error'] = 'info'
    command: str | None = None


@dataclass(kw_only=True)
class TitleChanged:
    """The current session got a new title; a switch to another session sends a `Snapshot` instead."""

    type: Literal['title'] = 'title'
    title: str | None


@dataclass(kw_only=True)
class ThemeChanged:
    """`/theme` or `/set display.theme` chose another palette."""

    type: Literal['theme'] = 'theme'
    theme: ThemeInfo


@dataclass(kw_only=True)
class SessionsChanged:
    """The saved sessions changed; the sidebar should list them again."""

    type: Literal['sessions'] = 'sessions'


ItemMessage = ItemAdded | TextDelta | ToolOutput | ToolDone | ShellDone | DiffAdded
ServerMessage = (
    Snapshot
    | ItemMessage
    | TurnStarted
    | TurnEnded
    | StatusChanged
    | QuestionAsked
    | QuestionClosed
    | Notice
    | TitleChanged
    | ThemeChanged
    | SessionsChanged
)
_WIRE: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def dump(message: ServerMessage) -> str:
    """The JSON text of one message, as the WebSocket sends it."""
    return _WIRE.dump_json(message).decode()


SessionState = Literal['ready', 'running', 'busy', 'completed', 'failed', 'cancelled']


@dataclass(kw_only=True)
class SessionEntry:
    """A saved session as the sidebar lists it."""

    id: str
    title: str
    workspace: str
    updated_at: datetime
    state: SessionState
    """`running` is this session's own turn; `busy` a turn another CLAI is running."""
    messages: int
    model: str | None
    current: bool


@dataclass(kw_only=True)
class Project:
    """Saved sessions grouped by repository, as the `/resume` browser groups them."""

    key: str
    name: str
    current: bool
    sessions: list[SessionEntry]


@dataclass(kw_only=True)
class Transcript:
    """A saved session's history, shown read-only until the shell switches to it."""

    id: str
    title: str
    workspace: str
    items: list[Item]
    openable: bool
    """Whether `/resume` can switch to it here; it refuses a session saved in another directory."""


@dataclass(kw_only=True)
class CompletionInfo:
    """One Tab suggestion: `text` replaces the `start` characters before the cursor."""

    text: str
    start: int
    display: str
    meta: str | None = None


def tool_result_text(part: ToolReturnPart | RetryPromptPart) -> str:
    """What the model got back from a tool, cut to `RESULT_CHARS` for display."""
    text = part.model_response_str() if isinstance(part, ToolReturnPart) else part.model_response()
    if len(text) > RESULT_CHARS:
        return text[:RESULT_CHARS] + f'\n… {len(text) - RESULT_CHARS:,} more characters'
    return text


def user_text(part: UserPromptPart) -> str:
    """A prompt's text, naming the attachments the browser does not show."""
    if isinstance(part.content, str):
        return part.content
    return '\n'.join(item if isinstance(item, str) else f'[{type(item).__name__}]' for item in part.content)


def history_items(messages: Sequence[ModelMessage]) -> list[Item]:
    """The transcript items of a saved or finished history; tool results join their calls."""
    items: list[Item] = []
    tools: dict[str, Item] = {}
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    items.append(Item(id=f'h{len(items)}', kind='user', text=user_text(part)))
                elif isinstance(part, ToolReturnPart | RetryPromptPart) and (tool := tools.get(part.tool_call_id)):
                    tool.text = tool_result_text(part)
                    tool.state = 'done' if isinstance(part, ToolReturnPart) else 'error'
            continue
        for part in message.parts:
            if isinstance(part, TextPart) and part.content:
                items.append(Item(id=f'h{len(items)}', kind='assistant', text=part.content))
            elif isinstance(part, ThinkingPart) and part.content:
                items.append(Item(id=f'h{len(items)}', kind='thinking', text=part.content))
            elif isinstance(part, ToolCallPart):
                tool = Item(
                    id=part.tool_call_id,
                    kind='tool',
                    name=part.tool_name,
                    args=part.args_as_json_str(),
                    state='interrupted',
                )
                tools[part.tool_call_id] = tool
                items.append(tool)
    return items


def questions_info(request: AskUserRequest, *, asker: str | None) -> Questions:
    """The browser's copy of an `ask_user_question` request."""
    return Questions(
        id=request.id,
        asker=asker,
        questions=[
            QuestionInfo(
                header=question.header,
                question=question.question,
                multi_select=question.multi_select,
                options=[Option(label=option.label, description=option.description) for option in question.options],
            )
            for question in request.questions
        ],
    )


class EventTranslator:
    """Turn one conversation's stream events into item messages, keeping what a later event needs.

    A part's index is only unique within one model response, so each `PartStartEvent` starts a new
    item and takes the index over. A file write's diff arrives before the write and is shown once it lands.
    """

    def __init__(self) -> None:
        self._parts: dict[int, str] = {}
        self._count = 0
        self._writes: dict[tuple[str | None, str], FileChangeRequestEvent] = {}

    def translate(self, event: AgentStreamEvent) -> list[ItemMessage]:
        """The item messages `event` makes; none for an event the browser does not show."""
        if isinstance(event, PartStartEvent):
            return self._start(event)
        if isinstance(event, PartDeltaEvent):
            delta = event.delta
            item_id = self._parts.get(event.index)
            if isinstance(delta, TextPartDelta | ThinkingPartDelta) and delta.content_delta and item_id is not None:
                return [TextDelta(id=item_id, text=delta.content_delta)]
            return []
        if isinstance(event, FunctionToolCallEvent):
            part = event.part
            item = Item(id=part.tool_call_id, kind='tool', name=part.tool_name, args=part.args_as_json_str())
            return [ItemAdded(item=replace(item, state='running'))]
        if isinstance(event, FunctionToolResultEvent):
            result = event.part
            state: ToolState = 'done' if isinstance(result, ToolReturnPart) else 'error'
            return [ToolDone(id=result.tool_call_id, text=tool_result_text(result), state=state)]
        return self._harness(event)

    def _start(self, event: PartStartEvent) -> list[ItemMessage]:
        part = event.part
        if not isinstance(part, TextPart | ThinkingPart):
            return []
        self._count += 1
        item_id = f'live{self._count}'
        self._parts[event.index] = item_id
        kind: ItemKind = 'assistant' if isinstance(part, TextPart) else 'thinking'
        return [ItemAdded(item=Item(id=item_id, kind=kind, text=part.content))]

    def _harness(self, event: AgentStreamEvent) -> list[ItemMessage]:
        """Shell output and file diffs from harness capability events, tied to their tool call."""
        if isinstance(event, CommandOutputEvent) and event.tool_call_id is not None:
            return [ToolOutput(id=event.tool_call_id, text=event.text)]
        if isinstance(event, CommandFinishedEvent) and event.tool_call_id is not None:
            return [ShellDone(id=event.tool_call_id, exit_code=event.exit_code)]
        if isinstance(event, FileChangeRequestEvent) and event.operation == 'write':
            self._writes[event.tool_call_id, event.path] = event
        elif isinstance(event, FileEditedEvent) and event.tool_call_id is not None:
            diff = Diff(path=event.path, diff=event.diff, truncated=event.truncated)
            return [DiffAdded(id=event.tool_call_id, diff=diff)]
        elif isinstance(event, FileWrittenEvent) and event.tool_call_id is not None:
            request = self._writes.pop((event.tool_call_id, event.path), None)
            if request is not None and not request.cancelled:
                diff = Diff(path=event.path, diff=request.diff, truncated=request.truncated)
                return [DiffAdded(id=event.tool_call_id, diff=diff)]
        return []


class ItemLog:
    """The current turn's items, kept in step with what browsers were sent, for one that connects mid-turn."""

    def __init__(self) -> None:
        self.items: dict[str, Item] = {}

    def clear(self) -> None:
        self.items.clear()

    def apply(self, message: ItemMessage) -> None:
        """Change the logged items as a browser applies `message`."""
        if isinstance(message, ItemAdded):
            self.items[message.item.id] = replace(message.item, diffs=list(message.item.diffs))
            return
        item = self.items.get(message.id)
        if item is None:
            return
        if isinstance(message, TextDelta):
            item.text += message.text
        elif isinstance(message, ToolOutput):
            item.output += message.text
        elif isinstance(message, ToolDone):
            item.text, item.state = message.text, message.state
        elif isinstance(message, ShellDone):
            item.exit_code = message.exit_code
        else:
            item.diffs.append(message.diff)
