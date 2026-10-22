"""What the browser sees of a CLAI session: transcript items, the messages that change them, and how events become them.

Pure translation, with no server or plugin in it. Wire payloads are kw-only dataclasses, serialized by
`dump` with Pydantic; the browser's `web-ui/src/protocol.ts` mirrors these shapes.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Literal

from pydantic import JsonValue, TypeAdapter
from typing_extensions import TypeIs

from pydantic_ai import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
)
from pydantic_ai.messages import (
    BinaryContent,
    EnqueuedMessagesEvent,
    ModelMessage,
    ModelRequest,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    ToolReturnPart,
    UserContent,
    UserPromptPart,
)
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.ask_user import AskUserRequest
from pydantic_ai_harness.filesystem import FileChangeRequestEvent, FileEditedEvent, FileWrittenEvent
from pydantic_ai_harness.shell import CommandFinishedEvent, CommandOutputEvent
from pydantic_ai_harness.subagents import DelegationOutcome
from pydantic_clai2.plugins import PanelIcon, PluginOrigin, PluginStatus, QueuedInput
from pydantic_clai2.runtime._session import STEERING
from pydantic_clai2.runtime.eager_timing import NESTED_CALL
from pydantic_clai2.runtime.monty_cells import NOTE_PREFIX
from pydantic_clai2.runtime.sandbox_calls import SandboxCallFinishedEvent, SandboxCallStartedEvent
from pydantic_clai2.ui.prompt.prompt_queue import InputKind, input_kind
from pydantic_clai2.ui.rendering.usage_report import SessionUsage, format_cost

MONTY_TOOL = 'run_code'
"""The code mode tool whose cells make up the Monty panel."""

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
    """A `run_code` cell the user ran with `/monty`, rather than one the model wrote."""
    images: list[str] = field(default_factory=list[str])
    """The IDs of a prompt's images, in order, which the session's `images/ID` route serves."""


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
class Choice:
    """One value a settings row or a dialog offers."""

    value: str
    label: str
    description: str = ''


DialogKind = Literal['choose', 'text', 'confirm', 'wait', 'view']


@dataclass(kw_only=True)
class Dialog:
    """What a plugin's settings menu asks, open until `POST /api/dialogs/ID` answers or cancels it.

    Which fields apply depends on `kind`, after the `Dialogs` method that asks it.
    """

    id: str
    kind: DialogKind
    title: str
    description: str = ''
    options: list[Choice] = field(default_factory=list[Choice])
    """`choose`: what to pick from."""
    current: str | None = None
    """`choose`: the value marked as the current one."""
    label: str = ''
    """`text`: the input's label."""
    placeholder: str = ''
    initial: str = ''
    secret: bool = False
    """`text`: mask what is typed; the answer is never sent to any browser."""
    multiline: bool = False
    """`text`: take text with line breaks, in a text area."""
    yes: str = ''
    """`confirm`: the answer that confirms; `no`, the other, is the safe one."""
    no: str = ''
    shown: str = ''
    """`wait`: the latest text the waiting task showed, such as a sign-in link."""
    text: str = ''
    """`view`: the read-only text to show; any answer closes it."""


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


def usage_info(usage: RunUsage) -> Usage:
    """The browser's copy of a run's usage."""
    return Usage(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, requests=usage.requests)


@dataclass(kw_only=True)
class StatusInfo:
    """The status row's figures, sent every `STATUS_INTERVAL` while a turn streams."""

    model: str
    activity: str
    context_tokens: int | None = None
    context_window: int | None = None
    cost: str | None = None
    cost_is_lower_bound: bool = False
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


@dataclass(frozen=True, kw_only=True)
class UpdateInfo:
    """`/update` installing a newer CLAI, as the page shows it until the install ends or CLAI is back."""

    target: str
    """The version, or on the `main` channel the short commit, being installed."""
    line: str = ''
    """The last line uv printed."""
    restarting: bool = False
    """Installed: CLAI is restarting as the new build, and the page reconnects once it is back."""


@dataclass(frozen=True, kw_only=True)
class QueuedItem:
    """An input waiting for the session's running turn or command to end, editable until it runs."""

    id: str
    text: str
    kind: InputKind
    """A prompt can steer the running turn; a `/command` or a `!` shell line, queued from the terminal, cannot."""


def queued_item(entry: QueuedInput) -> QueuedItem:
    """A waiting input as the page lists it, with whether it can steer."""
    return QueuedItem(id=entry.id, text=entry.text, kind=input_kind(entry.text))


@dataclass(frozen=True, kw_only=True)
class QueueChanged:
    """The session's queued input changed: one joined, ran, was edited, or was removed, here or in the terminal."""

    type: Literal['queue'] = 'queue'
    queue: list[QueuedItem]


def delivered_steering(event: EnqueuedMessagesEvent) -> list[UserPromptPart]:
    """The user's steering an enqueue delivered into the run, for the transcript to show.

    Input a capability enqueues, such as a context file it found, is the model's to read, not the user's to see.
    """
    return [
        part
        for message in event.messages
        if isinstance(message, ModelRequest) and message.metadata and message.metadata.get(STEERING)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]


@dataclass(kw_only=True)
class Snapshot:
    """Everything a browser needs to show one live session: sent on connect, on `watch`, and on a conversation switch.

    `session.id` names the session, so a snapshot carries no `session` string of its own.
    """

    type: Literal['snapshot'] = 'snapshot'
    session: SessionInfo
    items: list[Item]
    running: bool
    status: StatusInfo
    questions: list[Questions]
    dialogs: list[Dialog]
    theme: ThemeInfo
    tasks: list[TaskEntry]
    """The conversation's delegated tasks, each parent before its children."""
    task_items: dict[str, list[Item]]
    """Each running task's transcript so far, by task ID."""
    queue: list[QueuedItem] = field(default_factory=list[QueuedItem])
    """The input waiting for the session's running turn or command, oldest first."""
    follows_workspace: bool = False
    """Whether each session runs in its own directory, as under `clai2 --web`, so a new one asks where."""
    version: str = ''
    """The running CLAI's version, or a `main` build's short commit; a new one after a reconnect means it updated."""
    update: UpdateInfo | None = None


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
class DialogOpened:
    """A plugin's settings menu asks something; answer it with `POST /api/dialogs/ID`."""

    type: Literal['dialog'] = 'dialog'
    dialog: Dialog


@dataclass(kw_only=True)
class DialogShown:
    """A `wait` dialog's task showed new text, such as a sign-in code."""

    type: Literal['dialog_shown'] = 'dialog_shown'
    id: str
    text: str


@dataclass(kw_only=True)
class DialogClosed:
    """A dialog was answered or cancelled, or its menu closed it."""

    type: Literal['dialog_closed'] = 'dialog_closed'
    id: str


@dataclass(frozen=True, kw_only=True)
class TurnUsage:
    """Tokens, requests, and cost of one turn, or of all of them, as a `/usage` row shows them."""

    requests: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost: str | None
    """In dollars, as `/cost` formats it; `None` when nothing could be priced."""


@dataclass(frozen=True, kw_only=True)
class UsageReport:
    """What `/usage` and `/cost` report about the retained history, from the same `session_usage`."""

    turns: list[TurnUsage]
    total: TurnUsage
    unpriced: list[str]
    """The models without price data, whose responses no cost counts."""
    unmeasured_requests: int = 0


def turn_usage(usage: RunUsage) -> TurnUsage:
    """The browser's copy of one `/usage` row."""
    return TurnUsage(
        requests=usage.requests,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cost=None if usage.cost is None else format_cost(usage.cost),
    )


def usage_report(usage: SessionUsage) -> UsageReport:
    """The browser's copy of what `/usage` prints."""
    return UsageReport(
        turns=[turn_usage(turn) for turn in usage.turns],
        total=turn_usage(usage.total),
        unpriced=list(usage.unpriced),
        unmeasured_requests=usage.total.unmeasured_requests,
    )


@dataclass(kw_only=True)
class Notice:
    """A `/command`'s result, or another line the browser should show once."""

    type: Literal['notice'] = 'notice'
    text: str
    level: Literal['info', 'error'] = 'info'
    command: str | None = None
    usage: UsageReport | None = None
    """For `/usage` and `/cost`: the figures their text reports, which the page shows as a card."""


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
class UpdateChanged:
    """`/update` started installing, uv printed a line, CLAI is restarting, or the install ended (`None`)."""

    type: Literal['update'] = 'update'
    update: UpdateInfo | None


@dataclass(kw_only=True)
class PluginsChanged:
    """A command ran, which may have changed the plugins or their settings; an open plugins panel lists them again."""

    type: Literal['plugins'] = 'plugins'


@dataclass(frozen=True, kw_only=True)
class SettingsChanged:
    """A command ran, or a command's menu was edited, which may have changed CLAI's settings; open menus read them again."""

    type: Literal['settings'] = 'settings'


LiveState = Literal['running', 'queued', 'asking', 'idle']
"""`queued`: input waits while the session's turn runs; `asking`: a question of its turn waits for the user."""


@dataclass(frozen=True, kw_only=True)
class LiveEntry:
    """A session open in this CLAI, which the browser can watch and drive, named by its conversation ID."""

    id: str
    state: LiveState
    title: str | None
    workspace: str


@dataclass(frozen=True, kw_only=True)
class LiveChanged:
    """The live sessions, sent on connect and whenever one opens, closes, switches conversation, or changes state."""

    type: Literal['live'] = 'live'
    sessions: list[LiveEntry]


@dataclass(frozen=True, kw_only=True)
class Watch:
    """From the browser: the live sessions it shows, replacing the ones it watched; each new one gets a `Snapshot`.

    A connection that never sends one watches the default session, as before CLAI ran several.
    """

    type: Literal['watch'] = 'watch'
    sessions: list[str]


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


@dataclass(frozen=True, kw_only=True)
class OrchestratorState:
    """The orchestrator chat, sent on connect, as a turn starts, and on a new chat; `GET /api/orchestrator/state` too."""

    type: Literal['orchestrator_state'] = 'orchestrator_state'
    enabled: bool
    """Whether the `workspaces` plugin is on; the orchestrator runs only then."""
    running: bool
    model: str | None
    """The model its next turn runs on: the saved default a new session starts with, not a session's own choice."""
    items: list[Item]
    questions: list[Questions]


OrchestratorEvent = ItemMessage | TurnStarted | TurnEnded | QuestionAsked | QuestionClosed | OrchestratorState


@dataclass(frozen=True, kw_only=True)
class OrchestratorMessage:
    """A message for the orchestrator chat, on the session's WebSocket."""

    type: Literal['orchestrator'] = 'orchestrator'
    message: OrchestratorEvent


ServerMessage = (
    Snapshot
    | OrchestratorMessage
    | ItemMessage
    | TurnStarted
    | TurnEnded
    | StatusChanged
    | QuestionAsked
    | QuestionClosed
    | DialogOpened
    | DialogShown
    | DialogClosed
    | Notice
    | TitleChanged
    | ThemeChanged
    | SessionsChanged
    | PluginsChanged
    | SettingsChanged
    | TaskChanged
    | TaskItems
    | QueueChanged
    | UpdateChanged
    | LiveChanged
)
SessionMessage = (
    ItemMessage
    | TurnStarted
    | TurnEnded
    | StatusChanged
    | QuestionAsked
    | QuestionClosed
    | Notice
    | TitleChanged
    | TaskChanged
    | TaskItems
    | QueueChanged
)
"""The messages about one live session, which only its watchers get, each with a `session` naming it."""
_WIRE: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def dump(message: ServerMessage) -> str:
    """The JSON text of one message, as the WebSocket sends it."""
    return _WIRE.dump_json(message).decode()


def dump_session(message: SessionMessage, *, session: str) -> str:
    """The JSON text of a message about one live session, led by `session`, its conversation ID."""
    # Every message is a JSON object with at least its `type`, so the conversation ID goes in first.
    return f'{{"session":{json.dumps(session)},{dump(message)[1:]}'


SessionState = Literal['ready', 'running', 'busy', 'completed', 'failed', 'cancelled']


@dataclass(kw_only=True)
class SessionEntry:
    """A saved session as the sidebar lists it."""

    id: str
    title: str
    workspace: str
    updated_at: datetime
    state: SessionState
    """`running`: a turn of a session live in this CLAI; `busy`: a turn another CLAI process is running."""
    messages: int
    model: str | None
    current: bool
    archived: bool
    """Hidden from the sidebar unless it lists archived sessions too."""


@dataclass(frozen=True, kw_only=True)
class Worktree:
    """A project's checkout, or a folder outside Git, with the sessions saved in it."""

    root: str
    label: str
    """Its branch, or its folder's name outside Git or when the branch is unknown."""
    current: bool
    missing: bool
    """The folder is gone, as with a removed worktree."""
    sessions: list[SessionEntry]


@dataclass(frozen=True, kw_only=True)
class Project:
    """Saved sessions grouped by repository, then by checkout, as the `/resume` browser groups them."""

    key: str
    name: str
    current: bool
    worktrees: list[Worktree]


@dataclass(kw_only=True)
class Transcript:
    """A saved session's history, shown read-only until the shell switches to it."""

    id: str
    title: str
    workspace: str
    items: list[Item]
    openable: bool
    """Whether `/resume` can switch to it: saved here, or, while CLAI follows sessions' directories, in one that exists."""
    missing: bool = False
    """Whether CLAI would follow it, but its directory is gone, as with a deleted worktree."""


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
    """Whether it has a settings menu."""
    settings_menu: bool
    """Whether that menu is declared, so this page shows it; otherwise it opens in the terminal."""
    terminal: bool
    """Whether this CLAI has a terminal for an undeclared menu to open in; `clai2 --web` has none."""
    form: PluginForm | None
    """Its settings while it is on, unless it declares a menu, takes none, or its model has no JSON Schema."""
    serving: bool
    """Whether it is the plugin serving this page, which the page must not turn off or reload."""


@dataclass(kw_only=True)
class PanelInfo:
    """A plugin's side panel, as the page shows it now."""

    plugin: str
    id: str
    title: str
    icon: PanelIcon
    url: str | None
    """The page to show; `None` shows `note` alone."""
    note: str
    embed: bool
    """Whether `url` lets this page show it in a frame; otherwise it opens in a window of its own."""


@dataclass(kw_only=True)
class MenuField:
    """One row of a declared settings menu, as `FieldRow` describes it, with its current value."""

    key: str
    label: str
    description: str
    value: str
    """The current value as the user would type it, as the terminal list shows it: never a secret itself."""
    default: str
    choices: list[Choice]
    allow_custom: bool
    """Whether a value other than `choices` may be typed."""
    secret: bool
    note: str
    """Where the value comes from when not from the user."""
    action: bool
    """Whether choosing it runs an action, with dialogs, instead of editing a value."""
    menu: bool
    """Whether choosing it opens another menu, which the routes reach with `?menu=KEY`, instead of editing a value."""
    button: str = ''
    """What the action's button says; when empty, `Change…` for a row with a value or a secret, else `Open…`."""


@dataclass(kw_only=True)
class MenuInfo:
    """A plugin's declared settings menu, or a command's, with every row's current value."""

    title: str
    fields: list[MenuField]


@dataclass(kw_only=True)
class MenuEdited:
    """What an edit through a settings menu said, and the menu as it is now."""

    messages: list[str]
    menu: MenuInfo | None
    """`None` when the plugin no longer declares a menu, such as after it failed to load again."""


WorkspaceState = Literal['ready', 'running', 'stopped']


@dataclass(frozen=True, kw_only=True)
class WorkspaceRow:
    """A workspace as the sidebar lists it; never its token."""

    id: str
    kind: str
    repo: str
    branch: str
    path: str
    status: WorkspaceState
    url: str | None
    """Its page's address without the token, while its CLAI runs."""


@dataclass(frozen=True, kw_only=True)
class WorkspacesInfo:
    """The sidebar's Workspaces section: shown while the `workspaces` plugin is on."""

    enabled: bool
    workspaces: list[WorkspaceRow]


@dataclass(frozen=True, kw_only=True)
class WorkspaceOpened:
    """A workspace's page, with its token, for this page to open."""

    url: str


@dataclass(kw_only=True)
class CompletionInfo:
    """One Tab suggestion: `text` replaces the `start` characters before the cursor."""

    text: str
    start: int
    display: str
    meta: str | None = None


def tool_result_text(part: ToolReturnPart) -> str:
    """What the model got back from a tool, cut to `RESULT_CHARS` for display."""
    text = part.model_response_str(wrap_if_error=part.outcome != 'retried')
    if len(text) > RESULT_CHARS:
        return text[:RESULT_CHARS] + f'\n… {len(text) - RESULT_CHARS:,} more characters'
    return text


def tool_result_state(part: ToolReturnPart) -> ToolState:
    """A tool that has to be retried shows as an error; any other result as done."""
    return 'error' if part.outcome == 'retried' else 'done'


def image_id(image: BinaryContent) -> str:
    """An image's ID in the transcript: the SHA-256 of its bytes, so it stays the same after a reload."""
    return hashlib.sha256(image.data).hexdigest()


class ImageIds:
    """The transcript IDs of a live session's images, each hashed once while the session holds the image."""

    def __init__(self) -> None:
        """Start knowing no image."""
        self._known: dict[int, tuple[BinaryContent, str]] = {}
        """By `id()`, which stays unique as the image is kept here with its ID."""

    def __call__(self, image: BinaryContent) -> str:
        """The ID of `image`."""
        known = self._known.get(id(image))
        if known is None:
            known = self._known[id(image)] = (image, image_id(image))
        return known[1]

    def find(self, wanted: str) -> BinaryContent | None:
        """The image whose ID is `wanted`, of those the transcript listed and the session still holds."""
        return next((image for image, known in self._known.values() if known == wanted), None)

    def keep(self, images: Sequence[BinaryContent]) -> None:
        """Forget every image but `images`, those the session holds now."""
        held = {id(image) for image in images}
        self._known = {key: known for key, known in self._known.items() if key in held}


def user_images(part: UserPromptPart) -> list[BinaryContent]:
    """The images a prompt carries, which the browser shows as thumbnails."""
    if isinstance(part.content, str):
        return []
    return [item for item in part.content if _is_image(item)]


def _is_image(item: UserContent) -> TypeIs[BinaryContent]:
    return isinstance(item, BinaryContent) and item.is_image


def user_text(part: UserPromptPart, *, thumbnails: bool = False) -> str:
    """A prompt's text, naming the attachments the browser does not show: images too, unless as `thumbnails`.

    Notes telling the model about `/monty` cells are left out: the Monty panel shows those cells.
    """
    if isinstance(part.content, str):
        return part.content
    return '\n'.join(
        item if isinstance(item, str) else '[image]' if _is_image(item) else f'[{type(item).__name__}]'
        for item in part.content
        # A prompt sent as images alone has an empty text.
        if not (isinstance(item, str) and (not item or item.startswith(NOTE_PREFIX)))
        and not (thumbnails and _is_image(item))
    )


def user_item(part: UserPromptPart, *, id: str, image_ids: ImageIds | None = None) -> Item:
    """A prompt's transcript item: its text, and its images by `image_ids` for a session that serves them.

    Without `image_ids`, as in a saved session's preview, each image shows as `[image]`.
    """
    images = [] if image_ids is None else [image_ids(image) for image in user_images(part)]
    return Item(id=id, kind='user', text=user_text(part, thumbnails=image_ids is not None), images=images)


def history_images(messages: Sequence[ModelMessage]) -> list[BinaryContent]:
    """Every image the prompts of a history carry."""
    return [
        image
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
        for image in user_images(part)
    ]


def history_items(messages: Sequence[ModelMessage], *, image_ids: ImageIds | None = None) -> list[Item]:
    """The transcript items of a saved or finished history; tool results join their calls.

    A live session passes its `image_ids`, so prompts list their images; see `user_item`.
    """
    items: list[Item] = []
    tools: dict[str, Item] = {}
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    items.append(user_item(part, id=f'h{len(items)}', image_ids=image_ids))
                elif isinstance(part, ToolReturnPart) and (tool := tools.get(part.tool_call_id)):
                    tool.text = tool_result_text(part)
                    tool.state = tool_result_state(part)
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
            state = tool_result_state(result)
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
            state = tool_result_state(result)
            return [ToolDone(id=result.tool_call_id, text=tool_result_text(result), state=state)]
        return self._harness(event)

    def _start(self, event: PartStartEvent) -> list[ItemMessage]:
        part = event.part
        if isinstance(part, ToolCallPart):
            if part.tool_name != MONTY_TOOL:
                return []
            self._parts[event.index] = part.tool_call_id
            self._streamed.add(part.tool_call_id)
            args = part.args if isinstance(part.args, str) else part.args_as_json_str() if part.args else ''
            item = Item(id=part.tool_call_id, kind='tool', name=MONTY_TOOL, args=args, state='running')
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
