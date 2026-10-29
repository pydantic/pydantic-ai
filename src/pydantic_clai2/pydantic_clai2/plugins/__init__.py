"""The declarative plugin API: subclass `Plugin` and override what it contributes, as with `AbstractCapability`."""

import re
import warnings
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from itertools import islice
from pathlib import Path
from typing import ClassVar, Generic, Literal, Never, Protocol, Self, TypeVar, cast, get_args, get_origin
from uuid import uuid4

from anyio import to_thread
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from rich.console import Console, RenderableType
from termflow.tui.completion import CompleteEvent, Completion, Document
from typing_extensions import TypeVar as DefaultTypeVar, deprecated, get_original_bases

from pydantic_ai import AgentRunResult, AgentStreamEvent
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.exceptions import PydanticAIDeprecationWarning
from pydantic_ai.messages import BinaryContent, ModelMessage
from pydantic_ai.models import Model
from pydantic_ai.providers import infer_provider_class
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness.step_persistence import StepStore
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SavedConversation
from pydantic_ai_harness.subagents import DelegationOutcome
from pydantic_clai2.commands import Command, Commands
from pydantic_clai2.config import Settings
from pydantic_clai2.config.plugin_requirements import Requirements, declared_requirements
from pydantic_clai2.models import CLAI_PROVIDERS, LOGIN_ALIASES, LOGINS
from pydantic_clai2.plugins.menus import SettingsMenu
from pydantic_clai2.ui.rendering.spinners import Spinner
from pydantic_clai2.ui.rendering.status import Status, StatusSegment

DepsT = DefaultTypeVar('DepsT', default=None)
ModelT = TypeVar('ModelT', bound=BaseModel)
ResultT = TypeVar('ResultT')
_SAVED_SETTINGS = TypeAdapter(dict[str, JsonValue])

SessionEndReason = Literal['exit', 'eof', 'error']
TurnOutcome = Literal['completed', 'failed', 'cancelled']


class Conversation(Protocol):
    """The retained history as a plugin sees it. The shell's `Session` is one; `Transcript` is the plain one."""

    step_store: StepStore | None

    @property
    def conversation_id(self) -> str:
        """The ID `/resume` and `--resume` restore this conversation by."""
        ...

    @property
    def title(self) -> str | None:
        """The conversation's title; `None` until its first prompt is saved."""
        ...

    async def commit_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Persist and publish a between-turn history replacement."""
        ...

    @property
    def messages(self) -> list[ModelMessage]:
        """A snapshot of the retained messages."""
        ...

    def replace_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Swap the retained history, as `/compact` does after summarising it."""
        ...

    async def resolved_model(self) -> Model | str | None:
        """The model CLAI or the user selected for the next run; `None` when neither has chosen one.

        A capability that selects a model can replace CLAI's default per request, so a run may use another.
        """
        ...

    async def resume(self, conversation_id: str) -> str:
        """Switch to a saved conversation, as `/resume ID` does, and return the notice to show.

        Only restores the saved history: nothing runs and no tool is replayed. Call it between
        turns, such as from `on_session_start` or a command. It raises `RuntimeError` during a run,
        `LookupError` for an unknown ID, and `ValueError` for a conversation saved in another
        directory, when persistence is not configured, or when another live session of this CLAI holds
        it. A conversation still running in another process is not taken over: a fork of it is resumed,
        and the notice says so.
        `on_conversation_changed` follows when the conversation or its title changed.
        """
        ...


class SessionCatalog(Protocol):
    """The saved conversations `/resume` lists, for a plugin that shows or manages them; the shell's `Sessions` is one.

    Switching to one goes through the prompt, as `/resume ID` typed there does, with `PluginHost.send_input`.
    """

    @property
    def follows_workspace(self) -> bool:
        """Whether `/resume ID` moves the session into the directory it was saved in, as under `clai2 --web`.

        Otherwise it refuses a session saved in another directory than the one CLAI runs in.
        """
        ...

    async def listing(
        self, *, query: str = '', limit: int = 200, archived: bool | None = False
    ) -> list[ConversationSummary]:
        """Saved conversations from every directory, newest first, matching `query` when it is not empty.

        Archived ones are left out unless `archived` is `True`, which lists only them, or `None`, which lists both.
        As when `/resume` opens, the newest untitled ones are queued for background naming, which
        calls the naming model, so poll it no more often than a person would look.
        """
        ...

    async def saved(self, conversation_id: str) -> SavedConversation:
        """A saved conversation and its history, without switching to it; raises `LookupError` for an unknown ID."""
        ...

    async def rename(self, conversation_id: str, title: str) -> None:
        """Retitle a saved conversation, as a rename in `/resume` does; raises `ValueError` if it changed meanwhile.

        For the current conversation, plugins hear `on_conversation_changed` at once, even while a turn
        runs, unlike background naming, which waits for the turn to end.
        """
        ...

    async def delete(self, conversation_id: str) -> None:
        """Delete a saved conversation; raises `ValueError` for the current one or one running elsewhere."""
        ...

    async def fork(self, conversation_id: str) -> ConversationSummary:
        """Save a copy of a conversation under a new ID, leaving the original as it was, and return the copy."""
        ...

    async def archive(self, conversation_id: str) -> ConversationSummary:
        """Hide a saved conversation from `listing` and `/resume`, deleting nothing, and return its summary.

        Raises `ValueError` for the current conversation or one running in another process. Resuming one
        unarchives it.
        """
        ...

    async def unarchive(self, conversation_id: str) -> ConversationSummary:
        """List an archived conversation again, and return its summary."""
        ...


PanelIcon = Literal['panel', 'traces']
"""The icons a side panel can show beside its title."""


@dataclass(frozen=True, kw_only=True)
class PanelView:
    """What a side panel shows now: a page, or a note saying why there is none, such as how to set it up."""

    url: str | None = None
    """An `http` or `https` page; `None` shows `note` alone."""
    note: str = ''


@dataclass(frozen=True, kw_only=True)
class SidePanel:
    """A page shown beside the conversation in the web UI; return it from `Plugin.get_side_panels`.

    `view` is awaited whenever the page asks, such as when a turn starts, so its page can follow
    the conversation. Keep it quick: cache anything that needs the network.
    """

    id: str
    title: str
    view: Callable[[], Awaitable[PanelView]]
    icon: PanelIcon = 'panel'


PluginOrigin = Literal['built-in', 'project', 'drop-in', 'installed']
"""Where a plugin's declaration came from, as `/plugins` labels it."""
PluginStatus = Literal['on', 'off', 'idle', 'failed', 'included']
"""Loaded; turned off; on but not loaded; on but failed to load; kept off because another loaded plugin includes it."""


@dataclass(frozen=True, kw_only=True)
class PluginInfo:
    """One `/plugins` row, for a plugin that shows or manages plugins, such as a remote UI."""

    name: str
    description: str
    """The first paragraph of the plugin's docstring, read without running its code."""
    origin: PluginOrigin
    source: str
    """The drop-in file, or the import string."""
    status: PluginStatus
    included_in: str | None = None
    """The loaded plugin that includes this one, when `status` is `included`."""
    error: str | None = None
    """Why it failed to load, when `status` is `failed`."""
    notice: str | None = None
    """Saved settings this build ignored at the last load."""
    provides: str = ''
    """What it contributes while loaded, such as `2 commands, 1 capability`."""
    has_menu: bool = False
    """Whether it has a settings menu: a declared one (`settings_menu`), or its own `configure`, which runs in the terminal."""
    settings_menu: bool = False
    """Whether it declares its settings menu, which `PluginCatalog.edit_settings` can edit from anywhere."""
    settings_model: type[BaseModel] | None = None
    """Its settings model while loaded, unless it takes none; validate new settings with it for `save_settings`."""
    settings: BaseModel | None = None
    """Its saved settings while loaded, as the plugin reads them with `host.settings`."""


class PluginCatalog(Protocol):
    """The plugins `/plugins` lists, for a plugin that shows or manages them; the shell's `SessionPlugins` is one.

    Each action applies at once and is saved, as the same `/plugins` action is, in every live session
    of this CLAI; a running turn keeps the plugins it started with. Adding and removing declarations
    goes through the prompt, with `PluginHost.send_input`, as does a settings menu that is not declared
    (`/plugins configure NAME`).
    Changes run one at a time, each waiting for one in progress, so never await one from a plugin's
    own `on_session_start` or `on_session_end`, which a change runs.
    """

    def listing(self) -> list[PluginInfo]:
        """Every plugin, sorted by name; reads the plugin folder and parses descriptions, so call it when shown."""
        ...

    async def enable(self, name: str) -> None:
        """Turn a plugin on and load it, without opening its settings menu; raises `PluginError` if loading fails."""
        ...

    async def disable(self, name: str) -> None:
        """Unload a plugin and turn it off."""
        ...

    async def reload(self, name: str) -> None:
        """Unload a plugin, import its module again, and load it; raises `ValueError` when it is off."""
        ...

    async def save_settings(self, name: str, settings: BaseModel) -> None:
        """Save a loaded plugin's settings, as its settings menu does, then load it again if they changed.

        `settings` is an instance of `PluginInfo.settings_model`. Raises `ValueError` when the plugin is
        not loaded or takes no settings, and `TypeError` for another model.
        """
        ...

    def settings_menu(self, name: str) -> SettingsMenu | None:
        """A loaded plugin's declared settings menu, to show; `None` when it is not loaded or declares none."""
        ...

    def side_panels(self) -> dict[str, tuple[SidePanel, ...]]:
        """The side panels of every loaded plugin that declares some, by plugin name, in load order."""
        ...

    async def edit_settings(self, name: str, edit: Callable[[SettingsMenu], Awaitable[ResultT]]) -> ResultT:
        """Run `edit` on a loaded plugin's declared settings menu, then load the plugin again if its settings changed.

        This is what closing the menu after an edit does, so call it once per edit: a value applied
        or reset through the menu's `source`, or one of its `actions` run with your own `Dialogs`.
        Raises `ValueError` when the plugin is not loaded, declares no menu, or its menu is already
        open, in the terminal or for another edit.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class QueuedInput:
    """One input waiting for a live session's running turn or command to end, as its prompt shows it queued."""

    id: str
    """Stable while it waits, also when the terminal starts or stops showing the session."""
    text: str
    """The prompt or `/command`, as it will run."""


SendInput = Callable[[str, Sequence[BinaryContent]], None]
"""How a host enters input at its session's prompt: the text, and the images attached to it."""


PluginScope = Literal['session', 'process']
"""How many instances of a plugin CLAI builds: one per live session, or one for the whole process."""


class LiveSession(Protocol):
    """One live session of this CLAI process, as a process-scoped plugin reaches it through `PluginHost.live`."""

    @property
    def id(self) -> str:
        """Stable while the session is live; its conversation can change, as with `/new` or `/resume`."""
        ...

    @property
    def session_id(self) -> str | None:
        """The session's current saved conversation ID, as `PluginHost.session_id` reads it."""
        ...

    @property
    def workspace(self) -> Path:
        """The directory the session works in."""
        ...

    @property
    def conversation(self) -> Conversation:
        """The session's retained history."""
        ...

    @property
    def status(self) -> Status:
        """The session's status row."""
        ...

    @property
    def sessions(self) -> SessionCatalog | None:
        """The saved conversations, as the session's `/resume` lists them."""
        ...

    @property
    def tasks(self) -> 'TaskCatalog | None':
        """The session's delegated tasks, while a task owner runs them."""
        ...

    @property
    def forks(self) -> 'tuple[ForkInfo, ...]':
        """The session's background `/fork` runs, oldest first, as `/forks` lists them."""
        ...

    @property
    def running(self) -> bool:
        """Whether a turn runs in the session now."""
        ...

    def setting(self, key: str, /) -> JsonValue:
        """The effective value of a CLAI setting in the session, as `PluginHost.setting` reads it."""
        ...

    def command_menu(self, name: str, /) -> SettingsMenu | None:
        """The menu bare `/NAME` opens in the session, built now, as `PluginHost.command_menu` builds it."""
        ...

    @property
    def queued(self) -> int:
        """How many inputs wait for the session's running turn or command to end."""
        ...

    @property
    def queue(self) -> list[QueuedInput]:
        """The inputs waiting for the session's running turn or command to end, oldest first, the order they run in."""
        ...

    def edit_queued(self, input_id: str, text: str, /) -> None:
        """Replace a waiting input's text, keeping its place, as editing a recalled one at the prompt does.

        Raises `LookupError` once it has run or was removed, and `ValueError` for blank text.
        """
        ...

    def remove_queued(self, input_id: str, /) -> None:
        """Drop a waiting input; raises `LookupError` once it has run or was removed."""
        ...

    def steer(self, text: str, /) -> bool:
        """Send `text` to the running turn now, as Alt+Enter at the prompt does; its model reads it at its next request.

        Returns `False` when nothing takes it: no turn runs, or it is a `/command` or `!` shell line, which never steer.
        """
        ...

    @property
    def watchers(self) -> int:
        """How many `watch` calls have not stopped watching yet."""
        ...

    def watch(self) -> Callable[[], None]:
        """Count a watcher, such as a browser showing the session, until the returned function is called.

        A watched session is never closed for being idle. Calling the returned function again does nothing.
        """
        ...

    def send_input(self, text: str, /, *, images: Sequence[BinaryContent] = ()) -> None:
        """Enter `text` at the session's prompt, with `images` attached, as `PluginHost.send_input` does."""
        ...

    def interrupt(self) -> bool:
        """Cancel the session's running turn or command; `False` when nothing was running."""
        ...

    async def complete(self, text: str, /) -> list[Completion]:
        """What Tab offers at the session's prompt for `text`, as `PluginHost.complete` returns it."""
        ...


class LiveSessions(Protocol):
    """The live sessions of this CLAI process, for a process-scoped plugin; see `PluginHost.live`."""

    def listing(self) -> list[LiveSession]:
        """Every live session, oldest first."""
        ...

    def get(self, live_id: str, /) -> LiveSession:
        """The live session with this `LiveSession.id`; raises `LookupError` when none is live."""
        ...

    def current(self) -> LiveSession | None:
        """The session whose moment, command, or run is being handled now; `None` outside one.

        A task started while a session is handled, such as a server started from a command, inherits it.
        """
        ...

    async def open(self, conversation_id: str, /) -> LiveSession:
        """The live session holding a saved conversation, resuming it in a new one, in its own directory, if none does.

        Concurrent calls for one conversation return the same session. A conversation another process runs
        continues in a fork of it, reusing one saved before, so the session's conversation ID differs.
        Raises `LookupError` for an unknown conversation, `ValueError` when its directory is gone, and
        `RuntimeError` where CLAI opens no sessions.
        Do not await it from `on_session_start`: the new session waits for the plugins still loading.
        """
        ...

    async def new(self, *, workspace: Path | None = None) -> LiveSession:
        """A new live session with a new conversation, in `workspace`, by default the directory CLAI started in.

        Do not await it from `on_session_start`: the new session waits for the plugins still loading.
        """
        ...

    async def close(self, live_id: str, /) -> None:
        """Close a session CLAI opened with `open` or `new`, cancelling its turn; its plugins end with `'exit'`.

        Raises `LookupError` when none is live and `ValueError` for one its own driver runs, such as the terminal's.
        """
        ...

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        """Call `listener` whenever a session opens or closes, or its running, queued, or watched state may change.

        It is called synchronously, so it must be quick and not raise. The returned function unsubscribes.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class TaskInfo:
    """One delegated task, as `/tasks` lists it, for a plugin that shows or controls them, such as a remote UI."""

    id: str
    parent_id: str | None
    """The task that delegated this one; `None` for one the conversation itself started."""
    agent: str
    """The sub-agent's name: `general-purpose` when the running agent delegated to itself."""
    model: str | None
    """The model its run streams from, once known; before that, the model it was asked to use, if any."""
    prompt: str
    status: Literal['running', 'finished']
    outcome: DelegationOutcome | None
    """How it ended, once `finished`."""
    started_at: datetime
    finished_at: datetime | None
    background: bool
    """Whether it runs in the background, so the run that started it does not wait for it."""
    resumable: bool
    """Whether `TaskCatalog.resume` and `prompt` can continue it once it has finished; one-shot agents such as
    `Explore` cannot."""
    activity: str
    """Its last action, as the `/tasks` panel shows it, such as `starting`, `thinking`, or `running: shell`."""
    usage: RunUsage
    """What its model requests have reported so far."""
    backgroundable: bool = False
    """Whether `TaskCatalog.background` can let the run that waits for it go on: it runs in the foreground of
    a run, and its workspace can outlive that run. A task the user prompted is waited for by no run."""


class TaskCatalog(Protocol):
    """The delegated tasks `/tasks` shows, for a plugin that shows or controls them; the shell's `Tasks` is one.

    Only the current conversation's tasks are listed, with their descendants. Each change to one
    arrives as `on_task_progress`. A task ID from elsewhere raises `LookupError`.
    """

    def listing(self) -> list[TaskInfo]:
        """Every task of the current conversation, each parent before its children, oldest first."""
        ...

    def history(self, task_id: str) -> list[ModelMessage]:
        """A task's messages: the run's so far while it runs, then what it saved."""
        ...

    async def stop(self, task_id: str) -> None:
        """Stop a task and its descendants, as `/tasks stop ID` does; a stopped task resumes only when asked to."""
        ...

    def background(self, task_id: str) -> None:
        """Let the run that waits for a task go on without it, as `/tasks background ID` does.

        Raises `ValueError` for a task whose workspace cannot outlive the run that started it, and for one
        the user prompted, which no run waits for.
        """
        ...

    async def resume(self, task_id: str) -> None:
        """Ask the agent to resume a finished task, as `/tasks resume ID` does, entering the request as a prompt.

        Raises `ValueError` for a task that is running or cannot resume, and `RuntimeError` on a CLAI with
        no prompt to enter it at; the task stays as it was then.
        """
        ...

    async def prompt(self, task_id: str, text: str) -> None:
        """Run a finished task again with `text` as the user's message, on its own, returning once it runs.

        The task continues its saved history with the agent, tools, and model it ran with, and saves what it
        does to it. It runs apart from the conversation: nothing reaches the conversation's history, the
        agent hears no report, and no turn waits for it, so it cannot be backgrounded. Each change arrives
        as `on_task_progress`, as for any task. Prompting a task the user stopped lets it resume. Raises
        `ValueError` for blank text, a task that is running, starting from a prompt, or cannot resume, or one
        the agent can no longer delegate to or whose delegate call fails or is denied, and `RuntimeError` on a
        CLAI that cannot run it; the task stays as it was then.
        """
        ...

    def steer(self, task_id: str, text: str) -> bool:
        """Send a running task a user message, which its run reads at its next model request.

        Returns `False` when the task is not running; `prompt` or `resume` continues a finished one. Raises
        `ValueError` for blank text.
        """
        ...


class Transcript:
    """An in-memory `Conversation` for hosts built outside the shell, such as in a plugin's tests."""

    def __init__(
        self,
        *,
        messages: Sequence[ModelMessage] = (),
        model: Model | str | None = None,
        conversation_id: str | None = None,
        title: str | None = None,
    ) -> None:
        """Start with `messages` retained and `model` as what `resolved_model` reports.

        Without a `conversation_id`, a new random one identifies the transcript.
        """
        self._messages = list(messages)
        self.step_store: StepStore | None = None
        self.model = model
        self.conversation_id = conversation_id or str(uuid4())
        self.title = title

    @property
    def messages(self) -> list[ModelMessage]:
        """A snapshot of the retained messages, like `Session.messages`; edit through `replace_messages`."""
        return list(self._messages)

    def replace_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Swap the retained history."""
        self._messages = list(messages)

    async def commit_messages(self, messages: Sequence[ModelMessage]) -> None:
        """Publish an in-memory history replacement."""
        self.replace_messages(messages)

    async def resolved_model(self) -> Model | str | None:
        """The `model` given at construction."""
        return self.model

    async def resume(self, conversation_id: str) -> str:
        """An in-memory transcript has no saved conversations to switch to."""
        raise LookupError(f'No saved session: {conversation_id}')


@dataclass(kw_only=True)
class SessionStart:
    """CLAI is ready for prompts. Each plugin receives this once, when it loads."""

    agent: AbstractAgent[Never, object]
    settings: Settings
    conversation_chosen: bool = False
    """Whether launch options (`--resume`, `--continue`, `--session-id`, `--fork-session`) picked `host.conversation`.

    `--continue` counts only when it found a saved session to restore.

    A plugin that restores a conversation of its own choosing on start should then leave it alone.
    """


@dataclass(kw_only=True)
class ConversationChanged:
    """The shell switched to another conversation, or the current one got a new title.

    Fires after `/new`, `/clear`, and `/resume`, when the first prompt titles a new conversation,
    and when background naming, a rename in `/resume`, or `host.sessions.rename` retitles the current one. A startup
    `--resume SESSION-ID`, `--session-id`, or `--fork-session` is already in place when
    `on_session_start` runs, so it fires nothing.
    """

    conversation_id: str
    title: str | None
    """`None` until the conversation's first prompt is saved."""


@dataclass(kw_only=True)
class CommandEnd:
    """A `/command` finished, typed at the prompt or entered with `PluginHost.send_input`.

    `output` is the text the command returned, which the shell printed, empty when it returned none
    or a silent one-line no-op such as `API key unchanged.`; what the command printed itself, such as
    the history `/resume` replays, is not in it. `error` is why it failed.
    """

    text: str
    output: str = ''
    error: BaseException | None = None


@dataclass(frozen=True, kw_only=True)
class TaskProgress:
    """A delegated task of the current conversation started, streamed an event, or finished."""

    task: TaskInfo
    """The task as it is now."""
    event: AgentStreamEvent | None = None
    """The event from the task's own run, its `DelegationStartEvent` or `DelegationEndEvent`, or one of its
    `sub_agents` task events, such as `TaskQuestionEvent`; `None` when only the task changed, such as when it
    started, finished, or began waiting for an answer."""


ForkStatus = Literal['running', 'done', 'failed', 'cancelled']


@dataclass(frozen=True, kw_only=True)
class ForkInfo:
    """One background `/fork`, as a `/forks` row lists it, for a plugin that shows them, such as a remote UI."""

    id: int
    """The number `/fork cancel ID` takes, counting from 1 in each live session."""
    model: str
    """The model it runs, or `agent default`."""
    prompt: str
    status: ForkStatus
    elapsed: float
    """Seconds it has run so far, or ran in all once it ended."""
    session_id: str | None
    """The saved conversation `/resume` continues it in, once it has saved one."""


@dataclass(frozen=True, kw_only=True)
class ForkEnd:
    """A background `/fork` finished, failed, or was cancelled."""

    fork: ForkInfo
    """The fork as it ended."""
    output: str | None = None
    """Its answer, once it is `done`; a structured output as `str` makes it."""
    error: BaseException | None = None
    """Why it `failed`."""


UpdateStage = Literal['installing', 'output', 'failed', 'installed', 'restarting']


@dataclass(frozen=True, kw_only=True)
class InstallProgress:
    """`/update` is installing a newer CLAI with uv.

    `installing` comes first, then `output` for each line uv prints, then one of `failed` when uv fails,
    `installed` when CLAI must be started again by hand, or `restarting` when CLAI is about to exit and
    start the new build in its place.
    """

    stage: UpdateStage
    target: str
    """The version, or on the `main` channel the short commit, being installed."""
    text: str = ''
    """The line uv printed, for `output`."""


@dataclass(kw_only=True)
class PluginLoadFailed:
    """A plugin failed to load; delivered to loaded plugins after startup loading finishes."""

    plugin: str
    error: BaseException


@dataclass(kw_only=True)
class SessionEnd:
    """CLAI is quitting, or this plugin is being unloaded."""

    reason: SessionEndReason


@dataclass(kw_only=True)
class TurnStart:
    """A prompt was submitted. Edit `text` or call `cancel()` before the agent sees it."""

    text: str
    cancelled: bool = False
    cancel_reason: str | None = None

    def cancel(self, reason: str | None = None) -> None:
        """Stop this turn before it starts; the reason is shown to the user."""
        self.cancelled = True
        self.cancel_reason = reason


@dataclass(kw_only=True)
class TurnEnd:
    """The turn finished. `result` is set when it completed, `error` when it failed."""

    text: str
    outcome: TurnOutcome
    result: AgentRunResult[object] | None = None
    error: BaseException | None = None


SettingsProvider = Literal['anthropic', 'google', 'openai', 'openai-chat']
"""Providers whose `/model settings` controls a plugin's models can take."""


@dataclass(frozen=True, kw_only=True)
class ModelProvider:
    """Models a plugin runs under its own `PREFIX:`; return it from `Plugin.get_model_providers`.

    `resolve` gets NAME without the prefix. CLAI calls it in a worker thread before every run with
    one of these models, so it may read the keyring; raise `UserError` saying how to set up when it
    cannot build the model. A prefix Pydantic AI or CLAI already runs is rejected; when two plugins
    offer one prefix, the later one wins. Unloading the plugin removes the prefix, and a run with
    a model under it then fails as an unknown provider until the plugin is enabled again.

    When `resolve` returns that provider's model class, such as an `AnthropicModel` subclass, set
    `settings_from` so `/model settings` offers its controls (thinking, effort) for these models.
    """

    prefix: str
    resolve: Callable[[str], Model]
    """Build the model for a name given without its prefix."""
    models: tuple[str, ...] = ()
    """Names without the prefix, offered by `/model add` and `/set model`."""
    settings_from: SettingsProvider | None = None
    """The provider whose `/model settings` controls, and known context window, these models take; `None` offers the generic ones."""
    resolve_profile: Callable[[str, str], Model] | None = None
    """Build NAME for another account: called as `resolve_profile(NAME, PROFILE)` for `PREFIX@PROFILE:NAME`.

    Called in a worker thread like `resolve`. `None` means the prefix has one account, and a model
    naming a profile fails with a message saying so.
    """

    def __post_init__(self) -> None:
        """Reject a malformed prefix, one CLAI already runs, or an unknown `settings_from`."""
        _require_name('Model prefix', self.prefix)
        if _runs_already(self.prefix):
            raise ValueError(f'Model prefix {self.prefix!r} is a provider CLAI already runs; choose your own.')
        if self.settings_from is not None and self.settings_from not in get_args(SettingsProvider):
            raise ValueError(
                f'settings_from must be one of {", ".join(get_args(SettingsProvider))}; got {self.settings_from!r}.'
            )

    @property
    def names(self) -> tuple[str, ...]:
        """`models` as CLAI shows and saves them, with the prefix."""
        return tuple(f'{self.prefix}:{name}' for name in self.models)


@dataclass(frozen=True, kw_only=True)
class UsageWindow:
    """One limit on an account, such as a five-hour or weekly window, as `/accounts` shows it."""

    label: str
    """A short name for the window: `5h`, `7d`, or `premium`."""
    used_percent: float
    """How much of the limit is used, from 0 to 100."""
    resets_at: datetime | None = None
    """When the window resets, timezone-aware; `None` when the service does not say."""


@dataclass(frozen=True, kw_only=True)
class AccountUsage:
    """An account's current usage: its limits, most pressing first, and its plan when known."""

    windows: tuple[UsageWindow, ...]
    plan: str | None = None


@dataclass(frozen=True, kw_only=True)
class PluginLogin:
    """A sign-in a plugin adds as `/login NAME`; return it from `Plugin.get_logins`.

    For sign-ins that store credentials, such as the subscription behind a `ModelProvider`. Keep
    secrets in the keyring, never in plugin settings, and raise `UserError` when signing in fails.
    Name it after the model prefix it unlocks, as CLAI's own `openai-codex` and `github-copilot` are.
    NAME cannot be a sign-in CLAI ships (including the earlier `codex` and `copilot`); when two
    plugins add one name, the later one wins. Unloading the plugin removes it. Once the sign-in
    succeeds, `models` (as `PREFIX:NAME`, such as a `ModelProvider`'s `names`) are added to the saved
    model list, so `/model` and `/model settings` offer them without `/model add`.
    """

    name: str
    handler: Callable[[], Awaitable[str]]
    """Sign in and return the message to show."""
    models: tuple[str, ...] = ()
    """Models, as `PREFIX:NAME`, added to the saved model list once the sign-in succeeds."""
    profile_handler: Callable[[str], Awaitable[str]] | None = None
    """Sign in to another account: called as `profile_handler(PROFILE)` for `/login NAME@PROFILE`.

    On success `models` are saved with the profile, as `PREFIX@PROFILE:NAME`. `None` means the
    sign-in has one account, and `/login NAME@PROFILE` says so.
    """
    usage: Callable[[str | None], Awaitable[AccountUsage]] | None = None
    """The account's current usage, for `/accounts`: called as `usage(PROFILE)`, `None` for the default.

    `/accounts` calls it in the background for each listed account while the menu is open, with a
    short timeout. Raise `UserError` with a short reason when usage is unavailable.
    """

    def __post_init__(self) -> None:
        """Reject a malformed name, or one CLAI already signs in to, before any plugin can offer it."""
        _require_name('Login name', self.name)
        if self.name in LOGINS or self.name in LOGIN_ALIASES:
            raise ValueError(f'Login name {self.name!r} is a sign-in CLAI already has; choose your own.')


_PROVIDER_PREFIX = re.compile(r'[a-z][a-z0-9-]*')


def _require_name(kind: str, name: str) -> None:
    """Model prefixes and login names share one format: what users type after `/login` or before `:`."""
    if not _PROVIDER_PREFIX.fullmatch(name):
        raise ValueError(
            f'{kind} {name!r} must start with a lowercase letter, followed by lowercase letters, digits, and hyphens.'
        )


def _runs_already(prefix: str) -> bool:
    """Whether CLAI or Pydantic AI already runs `prefix:` models, aliases such as `openai-chat` included.

    `infer_model` accepts exactly the prefixes `infer_provider_class` knows, so that is the source of truth.
    An unknown prefix raises `ValueError` without importing anything; a known one whose SDK is missing
    raises `ImportError`, and is still taken.
    """
    if prefix in CLAI_PROVIDERS:
        return True
    try:
        infer_provider_class(prefix)
    except ValueError:
        return False
    except ImportError:
        return True
    return True


COMPLETION_LIMIT = 100
"""The most suggestions `PluginHost.complete` returns, as many as the editor shows."""

HostEvent = (
    SessionStart
    | SessionEnd
    | TurnStart
    | TurnEnd
    | ConversationChanged
    | CommandEnd
    | PluginLoadFailed
    | TaskProgress
    | ForkEnd
    | InstallProgress
)
Renderer = Callable[[AgentStreamEvent], RenderableType | None]
"""Draws an event, or returns `None` to fall back to the default display; see `Plugin.render`."""
FullScreen = Callable[[], AbstractAsyncContextManager[None]]
"""Enter it to own the whole terminal for a widget while the agent runs; see `PluginHost.full_screen`."""


@asynccontextmanager
async def bare_screen() -> AsyncGenerator[None]:
    """The `FullScreen` of a host with no shell around it: nothing is streaming, so nothing to pause."""
    yield


class PluginHost(Generic[DepsT]):
    """What CLAI gives a plugin to talk to: the terminal, the conversation, and its saved settings.

    A plugin declares what it contributes on its `Plugin` subclass; the host only carries context.
    """

    def __init__(
        self,
        *,
        name: str,
        console: Console,
        settings: dict[str, JsonValue],
        full_screen: FullScreen = bare_screen,
        conversation: Conversation | None = None,
        settings_database: Path | None = None,
        session_id: Callable[[], str | None] = lambda: None,
        status: Status | None = None,
        save_settings: Callable[[dict[str, JsonValue]], None] = lambda _settings: None,
        requirements: Requirements | None = None,
        submit_prompt: Callable[[str], None] | None = None,
        send_input: SendInput | None = None,
        interrupt: Callable[[], bool] = lambda: False,
        commands: Commands | None = None,
        sessions: SessionCatalog | None = None,
        setting: Callable[[str], JsonValue] | None = None,
        plugins: PluginCatalog | None = None,
        tasks: Callable[[], TaskCatalog | None] = lambda: None,
        workspace: Callable[[], Path] = Path.cwd,
        live: LiveSessions | None = None,
        default_model: Callable[[], Awaitable[Model | str | None]] | None = None,
        default_model_name: Callable[[], Awaitable[str | None]] | None = None,
        resolve_model: Callable[[str], Awaitable[Model | str]] | None = None,
        model_settings: Callable[[str], ModelSettings | None] | None = None,
    ) -> None:
        """`settings` is the raw JSON from `plugins add`; validate it with `settings(Model)`.

        `settings_database` is the settings store's exact path, if the host has one.
        `save_settings` writes changed settings back to the plugin's declaration; without it they
        last until the plugin unloads. `submit_prompt` runs a prompt as the next turn, and
        `send_input` and `interrupt` drive the prompt as the keyboard does; see the methods of the
        same names. `commands` is the shell's registry, which `complete` reads; `sessions` its saved
        conversations; `setting` reads its settings, as the method of the same name does; `plugins`
        is what `/plugins` manages; `tasks` returns what `/tasks` shows, and `workspace` the session's
        directory, each read each time, as the properties of the same names are. `live` is given to a
        process-scoped plugin's host only. `default_model` resolves the model a new session starts with,
        and `default_model_name` names it without building it.

        The shell passes its own `conversation` and `status`; a host built elsewhere gets a
        `Transcript` and a detached status row, so a plugin needs no special case for either.
        """
        self.name = name
        self.console = console
        self.full_screen = full_screen
        """Own the whole terminal for a widget mid-run.

        `async with host.full_screen():` flushes streamed output and pauses the status row until
        the block exits, so a full-screen menu opened from inside a tool call draws on a settled
        screen. Between turns it is a no-op.
        """
        self.conversation: Conversation = conversation if conversation is not None else Transcript()
        self.settings_database = settings_database
        """The settings database used for this process; `None` on a host built outside the shell."""
        self._session_id = session_id
        self.status = status if status is not None else Status()
        self._settings = settings
        self._persist = save_settings
        self._requirements: Requirements = dict(requirements or {})
        self._submit_prompt = submit_prompt
        self._send_input = send_input
        self._interrupt = interrupt
        self._commands = commands
        self.sessions = sessions
        """The saved conversations `/resume` lists; `None` on a host without session persistence."""
        self._setting = setting
        self.plugins = plugins
        """The plugins `/plugins` manages; `None` on a host built outside the loader, such as in a test."""
        self._tasks = tasks
        self._workspace = workspace
        self._default_model = default_model
        self._default_model_name = default_model_name
        self._resolve_model = resolve_model
        self._model_settings = model_settings
        self.live = live
        """Every live session of this CLAI, for a plugin with `scope = 'process'`; `None` for any other.

        Such a plugin is built once for the whole process. Its host's `session_id`, `workspace`, `tasks`,
        `plugins`, `setting`, `send_input`, `submit_prompt`, and `interrupt` follow `live.current()`, the
        session whose moment, command, or run is being handled, else the session that loaded it (once
        that one has closed, the oldest live session). Its other attributes, such as `conversation` and
        `status`, always name the session that loaded it.
        """

    async def resolve_model(self, name: str) -> Model | str:
        """Resolve a model ID, saved chain, or account through the session's providers.

        A host outside CLAI without a resolver returns the ID unchanged for core to resolve.
        """
        return await self._resolve_model(name) if self._resolve_model is not None else name

    def model_settings(self, name: str) -> ModelSettings | None:
        """The session's defaults and saved settings for this model or chain, without the parent's run overrides."""
        return self._model_settings(name) if self._model_settings is not None else None

    @property
    def workspace(self) -> Path:
        """The directory this session works in: where its runs' tools work and its project files are read.

        Use it instead of the process's working directory, which CLAI does not change after startup,
        and pass it as `cwd=` to a subprocess. Under `clai2 --web` each session has its own. A host
        built outside the shell, such as in a test, has the current directory.
        """
        return self._workspace()

    @property
    def tasks(self) -> TaskCatalog | None:
        """The delegated tasks `/tasks` shows; `None` while no task owner runs them.

        The shell runs one for its stock agent (`create_stock_agent`) while its prompt is open. Before
        it opens and after it closes, a headless `-p` run, another agent, and a host built outside the
        shell have none.
        """
        return self._tasks()

    def submit_prompt(self, text: str, /) -> None:
        """Run `text` as the next turn, as if typed at the prompt but without echoing it.

        Call it from a `/command` handler that starts a turn, such as a skill command sending the
        skill's instructions. The prompt runs once the command returns, before input queued in the
        editor, and goes through `on_turn_start` like any prompt. A `during_turn` command opened over a
        running turn must not call it. Text starting with `/` or `!` is a prompt here, not
        a command. Raises `RuntimeError` outside a command handler run between turns, and on a host
        with no shell to run it, such as in a test, unless the host was built with `submit_prompt=`.
        """
        if self._submit_prompt is None:
            raise RuntimeError(f'Plugin {self.name!r} has no shell to run a prompt in.')
        self._submit_prompt(text)

    def send_input(self, text: str, /, *, images: Sequence[BinaryContent] = ()) -> None:
        """Enter `text` at the prompt as Enter does, as if it was typed there, at any time from the shell's event loop.

        For a plugin that is another way to type, such as a remote UI. The text is trimmed, a bare
        `clear` runs `/clear`, and blank text does nothing. A prompt runs as a turn and a `/command` as a
        command, each echoed in the transcript like typed input; while a turn runs it queues behind it,
        as a follow-up typed then would, and a `during_turn` command opens over it at once. Input
        entered before the prompt opens, such as from `on_session_start`, waits for it. Unlike
        `submit_prompt`, it is not tied to a command handler. A headless `-p` run reads no prompts, so
        what it is sent is never run. Raises `RuntimeError` on a host whose shell reads no such input:
        one built without `send_input=`, or a CLAI whose output is not a terminal, which reads its
        prompts line by line. A bare command that opens a menu, such as `/model`, opens it in the
        terminal, where only the keyboard or `interrupt` closes it.

        `images` go with the prompt, as images pasted at the prompt do: the model gets them after the
        text, and the text may be blank. Raises `ValueError` when `text` is a `/command` or `!` line,
        which take no images, or when the images would exceed the 32 MiB that may wait to be sent.
        """
        if self._send_input is None:
            raise RuntimeError(f'Plugin {self.name!r} has no prompt to enter input at.')
        self._send_input(text, images)

    async def default_model(self) -> Model | str | None:
        """The model a new session starts with: the saved `model` setting, resolved as `/model NAME` does.

        Without a saved one, the agent's own model, or `None` when it has none. For a plugin whose
        own agent works for the whole process rather than one session, such as the web UI's
        orchestrator. A host without CLAI's settings, such as in a test, returns `None` unless built
        with `default_model=`.
        """
        return None if self._default_model is None else await self._default_model()

    async def default_model_name(self) -> str | None:
        """The name of the model `default_model` resolves, as `/model` shows it, such as `chain:NAME`.

        Nothing is built, so it never needs a sign-in and never fails on a saved model that no longer
        resolves; use it to show the model. `None` where `default_model` returns `None`.
        """
        return None if self._default_model_name is None else await self._default_model_name()

    def setting(self, key: str, /) -> JsonValue:
        """The effective value of a CLAI setting, such as `run.code_mode`, read now, as `/set KEY` shows it.

        Read-only: change one as the user would, with `send_input('/set KEY VALUE')`. Raises `KeyError`
        for an unknown key, and on a host without CLAI's settings, such as in a test, unless the host
        was built with `setting=`.
        """
        if self._setting is None:
            raise KeyError(key)
        return self._setting(key)

    def interrupt(self) -> bool:
        """Cancel the running turn or command, as Esc does; `False` when nothing was running.

        A menu open over the cancelled turn or command closes with it, as it does for Ctrl-C; a
        headless `-p` run is not cancelled.
        """
        return self._interrupt()

    async def complete(self, text: str, /) -> list[Completion]:
        """What Tab offers at the prompt for `text`: command names with descriptions, their arguments, `@` paths.

        Each `Completion` replaces the `-start_position` characters before the end of `text`. As in the
        editor, they are computed in a worker thread, since `@` paths read the disk, at most
        `COMPLETION_LIMIT` of them, and none when a completer fails. A host without a shell offers nothing.
        """
        commands = self._commands
        if commands is None:
            return []

        def complete() -> list[Completion]:
            found = commands.get_completions(Document(text), CompleteEvent(text_inserted=True))
            return list(islice(found, COMPLETION_LIMIT))

        try:
            return await to_thread.run_sync(complete)
        except Exception:  # noqa: BLE001 -- optional suggestions, as in the editor; a broken completer offers none.
            return []

    def command_menu(self, name: str, /) -> SettingsMenu | None:
        """The settings menu bare `/NAME` opens, as the command declares it with `Command.menu`, built now.

        For a plugin that is another way to use CLAI, such as a remote UI, to show that menu itself
        instead of opening it in the terminal. `None` for a command that declares none, an unknown or
        unavailable one, and on a host without a shell.
        """
        command = None if self._commands is None else self._commands.get(name)
        return None if command is None or command.menu is None else command.menu()

    @property
    def session_id(self) -> str | None:
        """The current saved conversation's ID, or `None` for a host without session persistence."""
        return self._session_id()

    @property
    def requirements(self) -> Requirements:
        """Feature names each setting key needs, as declared with `settings(Model, requires=...)`."""
        return dict(self._requirements)

    def settings(self, model: type[ModelT], /, *, requires: Mapping[str, Iterable[str]] | None = None) -> ModelT:
        """Validate the JSON given to `plugins add` against the plugin's own model.

        `requires` names the features a setting's saved value depends on, such as
        `{'sub_agents': ['stock-bound-delegation']}`. CLAI stores the tags beside the declaration
        whenever the settings are saved, and a build missing a feature ignores that setting and uses
        its default. Keys are the saved names (aliases included); see `PLUGINS.md`.
        """
        if requires is not None:
            declared = declared_requirements(requires)
            known = {name for key, info in model.model_fields.items() for name in (key, info.alias) if name}
            unknown = declared.keys() - known
            if unknown:
                raise ValueError(
                    f'{model.__name__} has no setting {", ".join(sorted(unknown))} to require features for.'
                )
            self._requirements = declared
        return model.model_validate(self._settings)

    def save_settings(self, settings: BaseModel, /) -> None:
        """Remember new settings for this plugin, as `plugins add` would; they are stored in plaintext.

        Never save a secret: keep it in `/keys` and save a `KeyReference` naming it. A host built
        outside the loader keeps the change for this load only. Raises `ValueError`, saving nothing,
        when the dump is not a JSON object that validates back into the model, as `host.settings`
        will need on the next load.
        """
        try:
            saved = _SAVED_SETTINGS.validate_python(settings.model_dump(mode='json', by_alias=True))
            type(settings).model_validate(saved)
        except ValidationError as exc:
            raise ValueError(
                f'{type(settings).__name__} cannot be saved as plugin settings: they must dump to a JSON'
                ' object that validates back into the model.'
            ) from exc
        self._persist(saved)
        self._settings = saved


class NoSettings(BaseModel):
    """The settings of a plugin that takes none: any key in its declaration is an error."""

    model_config = ConfigDict(extra='forbid', frozen=True)


SettingsT = DefaultTypeVar('SettingsT', bound=BaseModel, default=NoSettings, covariant=True)


def _settings_argument(cls: type) -> object:
    """`Plugin`'s settings argument as `cls` binds it, in terms of `cls`'s own type parameters."""
    bases = ((base, get_origin(base) or base) for base in get_original_bases(cls))
    base, origin = next(
        (base, origin) for base, origin in bases if isinstance(origin, type) and Plugin in origin.__mro__
    )
    settings = SettingsT if origin is Plugin else _settings_argument(origin)
    parameters: tuple[object, ...] = getattr(origin, '__parameters__', ())
    return _bind(settings, dict(zip(parameters, get_args(base))))


def _bind(argument: object, bound: dict[object, object]) -> object:
    """Substitute `bound` type parameters into `argument`, including a generic settings model's own."""
    model = _as_model(argument)
    if model is None:
        return bound.get(argument, argument)
    parameters = model.__pydantic_generic_metadata__['parameters']
    arguments = tuple(_bind(parameter, bound) for parameter in parameters)
    if arguments == parameters:
        return model
    specialized: object = model[arguments]
    return specialized


def _as_model(argument: object) -> type[BaseModel] | None:
    return argument if isinstance(argument, type) and issubclass(argument, BaseModel) else None


_RENAMED_HANDLERS = {'on_task_event': 'on_task_progress', 'on_update_progress': 'on_install_progress'}
"""Deprecated `on_*` names a plugin may still define, each routed to the hook that replaced it."""


class Plugin(Generic[SettingsT, DepsT]):
    """A CLAI plugin, declared by overriding what it contributes, as an `AbstractCapability` is.

    Parametrize with the settings model, `class Linear(Plugin[LinearSettings])`: the loader validates
    the declaration's JSON against it and passes the result as `settings`. When the plugin loads,
    CLAI calls every `get_*` method once and keeps what they return until it unloads, so build
    tools, commands, and other contributions there or in `__init__`, not per call. Every method has
    a default that contributes nothing, so a plugin overrides only what it needs.

    Hooks into the agent run itself (`before_model_request`, typed events, and the rest) belong on
    a capability returned from `get_capabilities`: `Hooks`, or an `AbstractCapability` subclass.
    """

    settings_type: ClassVar[type[BaseModel]] = NoSettings
    """The settings model, read from the class's `Plugin[...]` parameter; there is no need to set it."""
    scope: ClassVar[PluginScope] = 'session'
    """`'session'` builds the plugin once per live session; `'process'` once for the whole CLAI.

    A process-scoped plugin gets `on_session_start` and `on_session_end` once, when it loads and unloads.
    Its capabilities and commands join every live session, and each other moment reaches it once per
    session, with `host.live.current()` naming that session.
    """

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Record the settings model named in `Plugin[SettingsModel, ...]`."""
        super().__init_subclass__(**kwargs)
        settings = _as_model(_settings_argument(cls))
        if settings is not None:
            cls.settings_type = settings
        for old, new in _RENAMED_HANDLERS.items():
            # The nearest class defining either name decides, so a legacy handler a mixin brings is routed too.
            owner = next(base for base in cls.__mro__ if old in base.__dict__ or new in base.__dict__)
            renamed = new in owner.__dict__
            if owner is Plugin or old not in owner.__dict__ or (renamed and owner is not cls):
                continue
            warnings.warn(
                f'`{cls.__name__}.{old}` is deprecated; rename it to `{new}`. '
                'The old name will be removed in a future release.',
                PydanticAIDeprecationWarning,
                stacklevel=2,
            )
            if not renamed:
                setattr(cls, new, owner.__dict__[old])

    def __init__(self, host: PluginHost[DepsT], settings: SettingsT) -> None:
        """Keep the host and validated settings; override to set up state the `get_*` methods share."""
        self.host = host
        self._settings = settings

    @classmethod
    def from_host(cls, host: PluginHost[DepsT]) -> Self:
        """Build the plugin from the settings saved in its declaration, as the loader does."""
        return cls(host, cast(SettingsT, host.settings(cls.settings_type)))

    @property
    def settings(self) -> SettingsT:
        """The settings the plugin loaded with. A settings menu reads fresh ones with `host.settings`."""
        return self._settings

    async def prepare(self) -> None:
        """Read what the `get_*` methods need, such as files on disk, before CLAI calls them.

        Called once per load, after `__init__` and before the `get_*` methods, so commands and
        capabilities can depend on what it finds. A failure fails the load. `load_plugin` raises
        `TypeError` for a plugin that overrides it; a test calls `from_host`, awaits `prepare`, then `collect`.
        """

    def get_capabilities(self) -> Sequence[AgentCapability[DepsT]]:
        """Tools, instructions, hooks, or a capability chosen per run, bound on every run."""
        return ()

    def get_commands(self) -> Sequence[Command]:
        """Slash commands, registered while the plugin is loaded."""
        return ()

    def get_status_segments(self) -> Sequence[StatusSegment]:
        """Short fragments for the status row, such as the working directory.

        The shell repaints the row about ten times a second while it works and once a second while
        the editor sits idle, so keep each fragment cheap and synchronous: it is called for every
        frame, not once per turn. Fragments are appended in load order, painted `MUTED`, and
        truncated from the right on a narrow terminal.
        """
        return ()

    def get_spinners(self) -> Sequence[Spinner]:
        """Working animations for `/spinner`; build each with `build_spinner`.

        A plugin spinner replaces a builtin of the same name, and the user's `spinners.json` replaces
        both. Unloading the plugin removes it; a selected spinner that is gone shows `working`.
        """
        return ()

    def get_model_providers(self) -> Sequence[ModelProvider]:
        """Model prefixes this plugin runs, offered in `/model add` and `/set model`."""
        return ()

    def get_logins(self) -> Sequence[PluginLogin]:
        """Sign-ins this plugin adds to `/login`, such as for the subscription behind a model provider."""
        return ()

    def get_side_panels(self) -> Sequence[SidePanel]:
        """Pages the web UI offers beside the conversation, such as a live view of this session's traces."""
        return ()

    def render(self, event: AgentStreamEvent) -> RenderableType | None:
        """Draw an event yourself; return `None` to fall back to the default display."""
        return None

    def settings_menu(self) -> SettingsMenu | None:
        """This plugin's settings menu, declared rather than run, so any surface can show it; `None` for none.

        Its `FieldSource` saves each change with `host.save_settings` as the user makes it, and its
        actions ask through the `Dialogs` they are given, so the menu works wherever it was opened.
        When the settings changed, the loader loads the plugin again afterwards, so it builds from them.
        """
        return None

    async def configure(self) -> str:
        """Show the settings menu in the terminal: `/plugins configure NAME`, `c` in `/plugins`, and on enable or add.

        By default it shows `settings_menu` and returns what changed. Override it only for a menu
        that cannot be declared, which then opens in the terminal alone.
        """
        menu = self.settings_menu()
        if menu is None:
            raise NotImplementedError(f'{type(self).__name__} has no settings menu.')
        if not self.host.console.is_terminal:
            return f'Configure {menu.source.title} from a terminal: /plugins configure {self.host.name}'
        # Imported here: the terminal menus import this package, and `/reload` refuses an import cycle.
        from pydantic_clai2.ui.menus.dialogs import run_settings_menu

        messages = await run_settings_menu(menu)
        return '\n'.join(messages) or f'{menu.source.title}: nothing changed.'

    async def on_session_start(self, event: SessionStart) -> None:
        """CLAI is ready for prompts; called once, when the plugin loads."""

    async def on_session_end(self, event: SessionEnd) -> None:
        """CLAI is quitting, the plugin is unloading, or it failed to load after it was built."""

    async def on_plugin_load_failed(self, event: PluginLoadFailed) -> None:
        """A startup plugin failed to load; called after all enabled plugins have been tried."""

    async def on_turn_start(self, event: TurnStart) -> None:
        """A prompt was submitted; edit `event.text` or call `event.cancel()`. A failure cancels the turn."""

    async def on_turn_end(self, event: TurnEnd) -> None:
        """The turn finished."""

    async def on_conversation_changed(self, event: ConversationChanged) -> None:
        """The shell switched conversations, or the current one got a new title."""

    async def on_command_end(self, event: CommandEnd) -> None:
        """A `/command` finished, with the text the shell showed for it."""

    async def on_task_progress(self, event: TaskProgress) -> None:
        """A delegated task of the current conversation started, streamed an event, or finished.

        It fires for every stream event of every running task, down to each text and thinking delta, so
        keep it cheap: the task waits for it.
        """

    async def on_fork_end(self, event: ForkEnd) -> None:
        """A background `/fork` finished, failed, or was cancelled."""

    async def on_install_progress(self, event: InstallProgress) -> None:
        """`/update` started installing a newer CLAI, uv printed a line, or the install ended."""

    @deprecated('`on_task_event` has been renamed to `on_task_progress`.', category=PydanticAIDeprecationWarning)
    async def on_task_event(self, event: TaskProgress) -> None:
        """The former name of `on_task_progress`; a plugin that defines it still receives `TaskProgress`."""

    @deprecated(
        '`on_update_progress` has been renamed to `on_install_progress`.', category=PydanticAIDeprecationWarning
    )
    async def on_update_progress(self, event: InstallProgress) -> None:
        """The former name of `on_install_progress`; a plugin that defines it still receives `InstallProgress`."""

    @property
    def has_configure(self) -> bool:
        """Whether this plugin offers a settings menu, by overriding `settings_menu` or `configure`."""
        cls = type(self)
        return cls.settings_menu is not Plugin.settings_menu or cls.configure is not Plugin.configure

    @property
    def has_render(self) -> bool:
        """Whether this plugin overrides `render`."""
        return type(self).render is not Plugin.render

    @property
    def hook_count(self) -> int:
        """How many host events this plugin overrides a handler for."""
        return sum(getattr(type(self), name) is not getattr(Plugin, name) for name in _HANDLERS.values())


_HANDLERS: dict[type[HostEvent], str] = {
    SessionStart: 'on_session_start',
    SessionEnd: 'on_session_end',
    TurnStart: 'on_turn_start',
    TurnEnd: 'on_turn_end',
    ConversationChanged: 'on_conversation_changed',
    CommandEnd: 'on_command_end',
    PluginLoadFailed: 'on_plugin_load_failed',
    TaskProgress: 'on_task_progress',
    ForkEnd: 'on_fork_end',
    InstallProgress: 'on_install_progress',
}


_RENAMED_EVENTS: dict[str, type[HostEvent]] = {'TaskEvent': TaskProgress, 'UpdateProgress': InstallProgress}
"""Deprecated names of host events, kept importable."""


def __getattr__(name: str) -> object:
    if event := _RENAMED_EVENTS.get(name):
        warnings.warn(
            f'`pydantic_clai2.plugins.{name}` has been renamed to `{event.__name__}`. '
            'Update your imports; this deprecated alias will be removed in a future release.',
            PydanticAIDeprecationWarning,
            stacklevel=2,
        )
        return event
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


@dataclass(frozen=True, kw_only=True)
class LoadedPlugin(Generic[DepsT]):
    """What a loaded plugin contributes, collected once from its `get_*` methods."""

    plugin: Plugin[BaseModel, DepsT]
    capabilities: tuple[AgentCapability[DepsT], ...]
    commands: Commands
    status_segments: tuple[StatusSegment, ...]
    spinners: tuple[Spinner, ...]
    model_providers: tuple[ModelProvider, ...]
    logins: tuple[PluginLogin, ...]
    side_panels: tuple[SidePanel, ...]

    @property
    def host(self) -> PluginHost[DepsT]:
        """The host the plugin was built with."""
        return self.plugin.host

    async def dispatch(self, event: HostEvent) -> None:
        """Hand a host event to the plugin's matching `on_*` method."""
        await getattr(self.plugin, _HANDLERS[type(event)])(event)

    def handles(self, event: type[HostEvent]) -> bool:
        """Whether the plugin overrides the `on_*` method for `event`, rather than inheriting the no-op."""
        name = _HANDLERS[event]
        return getattr(type(self.plugin), name) is not getattr(Plugin, name)

    def summary(self) -> str:
        """One line for the `/plugins` menu, naming only what the plugin contributes."""
        counts = (
            (len(list(self.commands)), 'command', 'commands'),
            (self.plugin.hook_count, 'hook', 'hooks'),
            (len(self.capabilities), 'capability', 'capabilities'),
            (int(self.plugin.has_render), 'renderer', 'renderers'),
            (len(self.status_segments), 'status segment', 'status segments'),
        )
        parts = [f'{count} {one if count == 1 else many}' for count, one, many in counts if count]
        return ', '.join(parts) or 'nothing yet'


def collect(plugin: Plugin[BaseModel, DepsT]) -> LoadedPlugin[DepsT]:
    """Call each of `plugin`'s `get_*` methods once; a command name collision raises `ValueError`."""
    commands = Commands()
    commands.register_many(plugin.get_commands())
    return LoadedPlugin(
        plugin=plugin,
        capabilities=tuple(plugin.get_capabilities()),
        commands=commands,
        status_segments=tuple(plugin.get_status_segments()),
        spinners=tuple(replace(spinner, source='plugin') for spinner in plugin.get_spinners()),
        model_providers=tuple(plugin.get_model_providers()),
        logins=tuple(plugin.get_logins()),
        side_panels=tuple(plugin.get_side_panels()),
    )


def load_plugin(plugin_type: type[Plugin[BaseModel, DepsT]], host: PluginHost[DepsT]) -> LoadedPlugin[DepsT]:
    """Build `plugin_type` from `host`'s settings and collect what it contributes, as the loader does.

    This is also how a plugin's tests load it without a shell; fire host events with `dispatch`. It is
    synchronous, so a plugin that overrides `prepare` raises `TypeError`: load it with `from_host`,
    `await plugin.prepare()`, then `collect`.
    """
    if plugin_type.prepare is not Plugin.prepare:
        name = plugin_type.__name__
        raise TypeError(
            f'{name} overrides `prepare`, which `load_plugin` cannot await. Use '
            f'`plugin = {name}.from_host(host)`, `await plugin.prepare()`, then `collect(plugin)`.'
        )
    return collect(plugin_type.from_host(host))
