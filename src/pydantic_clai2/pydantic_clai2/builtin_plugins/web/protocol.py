"""What the browser sees of a CLAI session: transcript items, the messages that change them, and how events become them.

Pure translation, with no server or plugin in it. Wire payloads are kw-only dataclasses, serialized by
`dump` with Pydantic; the browser's `web-ui/src/protocol.ts` mirrors these shapes.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

from pydantic import JsonValue, TypeAdapter

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
    ToolCallPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai_harness.ask_user import AskUserRequest
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent
from pydantic_ai_harness.subagents import DelegationOutcome
from pydantic_clai2.plugins import PluginOrigin, PluginStatus
from pydantic_clai2.runtime.eager_timing import NESTED_CALL
from pydantic_clai2.runtime.repl_cells import NOTE_PREFIX
from pydantic_clai2.runtime.sandbox_calls import SandboxCallFinishedEvent, SandboxCallStartedEvent

REPL_TOOL = 'run_code'
"""The code mode tool whose cells make up the REPL panel."""

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
class SubCall:
    """A tool `run_code` called from inside its sandbox, as CLAI's code mode reports it."""

    id: str
    name: str
    args: str
    state: ToolState = 'running'
    text: str = ''
    """What the tool returned to the script."""
    output: str = ''
    """What a shell command printed while it ran."""


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
    calls: list[SubCall] = field(default_factory=list[SubCall])
    """The tools a `run_code` cell called from its sandbox; live runs only, as history keeps none."""
    by_user: bool = False
    """A `run_code` cell the user ran with `/repl`, rather than one the model wrote."""


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
class TaskEntry:
    """A delegated task as the sidebar's Workers tree shows it, like a `/tasks` row."""

    id: str
    parent_id: str | None
    agent: str
    model: str | None
    prompt: str
    status: Literal['running', 'finished']
    outcome: DelegationOutcome | None
    started_at: datetime
    finished_at: datetime | None
    background: bool
    resumable: bool
    activity: str
    """Its last action, such as `thinking` or `running: shell`."""
    usage: Usage


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
    tasks: list[TaskEntry]
    """The conversation's delegated tasks, each parent before its children."""
    task_items: dict[str, list[Item]]
    """Each running task's transcript so far, by task ID."""


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
class ArgsDelta:
    """More of a `run_code` cell's arguments as the model writes them, as raw JSON text."""

    type: Literal['args_delta'] = 'args_delta'
    id: str
    text: str


@dataclass(kw_only=True)
class ToolArgs:
    """A streamed `run_code` cell's complete arguments, replacing what its deltas built."""

    type: Literal['tool_args'] = 'tool_args'
    id: str
    args: str


@dataclass(kw_only=True)
class SubCallAdded:
    """A `run_code` cell called a tool from its sandbox, possibly while the cell was still being written."""

    type: Literal['call'] = 'call'
    id: str
    call: SubCall


@dataclass(kw_only=True)
class SubCallOutput:
    """More of what a shell command called from a `run_code` cell printed."""

    type: Literal['call_output'] = 'call_output'
    id: str
    call_id: str
    text: str


@dataclass(kw_only=True)
class SubCallDone:
    """A tool called from a `run_code` cell returned or failed."""

    type: Literal['call_done'] = 'call_done'
    id: str
    call_id: str
    text: str
    state: ToolState


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


@dataclass(kw_only=True)
class TaskChanged:
    """A delegated task started, or its row changed: its activity, usage, or status."""

    type: Literal['task'] = 'task'
    task: TaskEntry


@dataclass(kw_only=True)
class PluginsChanged:
    """A command ran, which may have changed the plugins or their settings; an open plugins panel lists them again."""

    type: Literal['plugins'] = 'plugins'


ItemMessage = (
    ItemAdded
    | TextDelta
    | ArgsDelta
    | ToolArgs
    | ToolOutput
    | ToolDone
    | ShellDone
    | DiffAdded
    | SubCallAdded
    | SubCallOutput
    | SubCallDone
)


@dataclass(kw_only=True)
class TaskItems:
    """What a running task's own run did, as item messages for its transcript; their IDs are unlike the session's."""

    type: Literal['task_event'] = 'task_event'
    id: str
    """The task's ID."""
    messages: list[ItemMessage]


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
    | PluginsChanged
    | TaskChanged
    | TaskItems
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
class TaskTranscript:
    """A delegated task with its history, as it saved it."""

    task: TaskEntry
    items: list[Item]


@dataclass(kw_only=True)
class PluginForm:
    """A plugin's settings as a form: its settings model's JSON Schema, by alias, and the saved values."""

    json_schema: dict[str, JsonValue]
    values: dict[str, JsonValue]


@dataclass(kw_only=True)
class PluginRow:
    """A plugin as the plugins panel shows it, like a `/plugins` row."""

    name: str
    description: str
    origin: PluginOrigin
    source: str
    status: PluginStatus
    included_in: str | None
    error: str | None
    notice: str | None
    provides: str
    has_menu: bool
    """Whether it has a settings menu of its own, which opens in the terminal."""
    terminal: bool
    """Whether this CLAI has a terminal for that menu to open in; `clai2 --web` has none."""
    form: PluginForm | None
    """Its settings while it is on, unless it takes none or its model has no JSON Schema."""
    serving: bool
    """Whether it is the plugin serving this page, which the page must not turn off or reload."""


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
    """A prompt's text, naming the attachments the browser does not show.

    Notes telling the model about `/repl` cells are left out: the REPL panel shows those cells.
    """
    if isinstance(part.content, str):
        return part.content
    return '\n'.join(
        item if isinstance(item, str) else f'[{type(item).__name__}]'
        for item in part.content
        if not (isinstance(item, str) and item.startswith(NOTE_PREFIX))
    )


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
    item and takes the index over. A `run_code` cell is shown as soon as the model starts writing it,
    with its source streaming in, because code mode runs its statements and their tool calls
    (`SandboxCallStartedEvent`) while it is still being written. A file write's diff arrives before
    the write and is shown once it lands.
    """

    def __init__(self, *, scope: str = '') -> None:
        """`scope` starts every item ID, so a delegated task's items cannot take the session's IDs."""
        self._scope = scope
        self._parts: dict[int, str] = {}
        self._count = 0
        self._writes: dict[tuple[str | None, str], FileChangeRequestEvent] = {}
        self._streamed: set[str] = set()
        """`run_code` cells already shown from their streamed part, so their call event only completes them."""

    def translate(self, event: AgentStreamEvent) -> list[ItemMessage]:
        """The item messages `event` makes; none for an event the browser does not show."""
        messages = self._translate(event)
        if not self._scope:
            return messages
        return [
            ItemAdded(item=replace(message.item, id=self._scope + message.item.id))
            if isinstance(message, ItemAdded)
            else replace(message, id=self._scope + message.id)
            for message in messages
        ]

    def _translate(self, event: AgentStreamEvent) -> list[ItemMessage]:
        if isinstance(event, PartStartEvent):
            return self._start(event)
        if isinstance(event, PartDeltaEvent):
            return self._delta(event)
        if isinstance(event, SandboxCallStartedEvent):
            call = event.call
            parent = _parent(call.tool_call_id)
            sub = SubCall(id=call.tool_call_id, name=call.tool_name, args=call.args_as_json_str())
            return [] if parent is None else [SubCallAdded(id=parent, call=sub)]
        if isinstance(event, SandboxCallFinishedEvent):
            result = event.result
            parent = _parent(result.tool_call_id)
            state: ToolState = 'done' if isinstance(result, ToolReturnPart) else 'error'
            done = SubCallDone(id=parent or '', call_id=result.tool_call_id, text=tool_result_text(result), state=state)
            return [] if parent is None else [done]
        if isinstance(event, FunctionToolCallEvent):
            part = event.part
            if part.tool_call_id in self._streamed:
                return [ToolArgs(id=part.tool_call_id, args=part.args_as_json_str())]
            item = Item(id=part.tool_call_id, kind='tool', name=part.tool_name, args=part.args_as_json_str())
            return [ItemAdded(item=replace(item, state='running'))]
        if isinstance(event, FunctionToolResultEvent):
            result = event.part
            state = 'done' if isinstance(result, ToolReturnPart) else 'error'
            return [ToolDone(id=result.tool_call_id, text=tool_result_text(result), state=state)]
        return self._harness(event)

    def _start(self, event: PartStartEvent) -> list[ItemMessage]:
        part = event.part
        if isinstance(part, ToolCallPart):
            if part.tool_name != REPL_TOOL:
                return []
            self._parts[event.index] = part.tool_call_id
            self._streamed.add(part.tool_call_id)
            args = part.args if isinstance(part.args, str) else part.args_as_json_str() if part.args else ''
            item = Item(id=part.tool_call_id, kind='tool', name=REPL_TOOL, args=args, state='running')
            return [ItemAdded(item=item)]
        if not isinstance(part, TextPart | ThinkingPart):
            return []
        self._count += 1
        item_id = f'live{self._count}'
        self._parts[event.index] = item_id
        kind: ItemKind = 'assistant' if isinstance(part, TextPart) else 'thinking'
        return [ItemAdded(item=Item(id=item_id, kind=kind, text=part.content))]

    def _delta(self, event: PartDeltaEvent) -> list[ItemMessage]:
        delta = event.delta
        item_id = self._parts.get(event.index)
        if item_id is None:
            return []
        if isinstance(delta, TextPartDelta | ThinkingPartDelta) and delta.content_delta:
            return [TextDelta(id=item_id, text=delta.content_delta)]
        if isinstance(delta, ToolCallPartDelta) and isinstance(delta.args_delta, str) and delta.args_delta:
            # A provider that streams arguments as objects sends them whole with the call event instead.
            return [ArgsDelta(id=item_id, text=delta.args_delta)]
        return []

    def _harness(self, event: AgentStreamEvent) -> list[ItemMessage]:
        """Shell output and file diffs from harness capability events, tied to their tool call."""
        if isinstance(event, CommandOutputEvent) and event.tool_call_id is not None:
            if (parent := _parent(event.tool_call_id)) is not None:
                return [SubCallOutput(id=parent, call_id=event.tool_call_id, text=event.text)]
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


def _parent(tool_call_id: str) -> str | None:
    """The `run_code` cell a sandbox call belongs to, from its `PARENT__N` id; `None` for a top-level call."""
    nested = NESTED_CALL.fullmatch(tool_call_id)
    return None if nested is None else nested['parent']


class ItemLog:
    """The current turn's items, kept in step with what browsers were sent, for one that connects mid-turn."""

    def __init__(self) -> None:
        self.items: dict[str, Item] = {}

    def clear(self) -> None:
        self.items.clear()

    def apply(self, message: ItemMessage) -> None:
        """Change the logged items as a browser applies `message`."""
        if isinstance(message, ItemAdded):
            self.items[message.item.id] = replace(
                message.item, diffs=list(message.item.diffs), calls=list(message.item.calls)
            )
            return
        item = self.items.get(message.id)
        if item is None:
            return
        if isinstance(message, TextDelta):
            item.text += message.text
        elif isinstance(message, ArgsDelta):
            item.args += message.text
        elif isinstance(message, ToolArgs):
            item.args = message.args
        elif isinstance(message, ToolOutput):
            item.output += message.text
        elif isinstance(message, ToolDone):
            item.text, item.state = message.text, message.state
        elif isinstance(message, ShellDone):
            item.exit_code = message.exit_code
        elif isinstance(message, DiffAdded):
            item.diffs.append(message.diff)
        elif isinstance(message, SubCallAdded):
            item.calls.append(replace(message.call))
        else:
            for call in item.calls:
                if call.id == message.call_id:
                    if isinstance(message, SubCallOutput):
                        call.output += message.text
                    else:
                        call.text, call.state = message.text, message.state
