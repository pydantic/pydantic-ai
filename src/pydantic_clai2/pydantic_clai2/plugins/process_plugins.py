"""The plugins of one CLAI process: declarations, the changes `/plugins` makes, imports, and process-scoped plugins.

Each live session's `SessionPlugins` (`loader.py`) joins the process's one `ProcessPlugins`. A change
made from one session, such as `/plugins disable NAME`, runs there under the process's change lock;
every other live session then follows it, each keeping the plugins its running turn started with.

A plugin with `scope = 'process'` is built once, by the first session that loads it, and joins every
live session; `on_session_start` and `on_session_end` reach it once. `LiveSessions` lists the sessions.
"""

import importlib
import logging
from collections.abc import Awaitable, Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from anyio import Lock
from pydantic import JsonValue
from rich.console import Console
from termflow.tui.completion import Completion

from pydantic_ai.messages import BinaryContent
from pydantic_clai2.config import PluginSettings
from pydantic_clai2.config.settings_store import SettingsStore, canonical_plugin_declarations, canonical_plugin_id
from pydantic_clai2.plugins import (
    Conversation,
    DepsT,
    LiveSession,
    LoadedPlugin,
    PluginHost,
    QueuedInput,
    SessionCatalog,
    SessionEndReason,
    TaskCatalog,
)
from pydantic_clai2.plugins._factories import import_file
from pydantic_clai2.plugins.menus import SettingsMenu
from pydantic_clai2.ui.prompt.prompt_queue import NOT_QUEUED
from pydantic_clai2.ui.rendering import theme
from pydantic_clai2.ui.rendering.status import Status

if TYPE_CHECKING:
    from pydantic_clai2.plugins.loader import SessionPlugins

T = TypeVar('T')
_LOGGER = logging.getLogger(__name__)

# Local settings databases from before the package move, or from before plugins were declared as
# `Plugin` classes, can still name the old factories.
MOVED_FACTORIES = {
    'pydantic_ai_harness.coder:Coder': 'pydantic_clai2.builtin_plugins.coder',
    'pydantic_ai_harness:Coder': 'pydantic_clai2.builtin_plugins.coder',
    'pydantic_clai2.sessions': 'pydantic_clai2.runtime.sessions',
    'pydantic_clai2.ask_user_menu:activate': 'pydantic_clai2.builtin_plugins.ask_user_menu',
    'pydantic_clai2.builtin_plugins.ask_user_menu:activate': 'pydantic_clai2.builtin_plugins.ask_user_menu',
    **{
        f'pydantic_clai2.{name}': f'pydantic_clai2.builtin_plugins.{name}'
        for name in (
            'repo_context',
            'compaction',
            'logfire',
            'notifications',
            'github',
            'pylon',
            'google_workspace',
            'day_ai',
            'ordinal',
            'notion',
            'slack',
            'logfire_mcp',
            'posthog',
            'grain',
            'linear',
        )
    },
}
RETIRED_BUILTINS: dict[str, PluginSettings] = {
    'google_workspace': PluginSettings(
        id='google_workspace', factory='pydantic_ai_harness.google_workspace:GoogleWorkspace', enabled=False
    ),
    'ordinal': PluginSettings(id='ordinal', factory='pydantic_ai_harness.ordinal:Ordinal', enabled=False),
    'slack': PluginSettings(id='slack', factory='pydantic_ai_harness.slack:Slack', enabled=False),
    'grain': PluginSettings(id='grain', factory='pydantic_ai_harness.grain:Grain', enabled=False),
}
"""Former built-in declarations. A stored copy of one loads the built-in now declared under its id."""


def same_plugin(declaration: PluginSettings, shipped: PluginSettings | None) -> bool:
    """Whether `declaration` is `shipped` itself, or the store's enabled or disabled copy of it."""
    if shipped is None:
        return False
    return declaration.model_copy(update={'enabled': True}) == shipped.model_copy(update={'enabled': True})


class PromptInputs(Protocol):
    """A session's prompt as `LiveRuntime` reaches it: the inputs waiting for its turn, and the turn to steer."""

    @property
    def queue(self) -> list[QueuedInput]: ...

    def edit_queued(self, input_id: str, text: str) -> None: ...

    def remove_queued(self, input_id: str) -> None: ...

    def steer(self, text: str) -> bool: ...


class LiveRuntime:
    """A live session as `PluginHost.live` lists it, read through the host its session plugins get."""

    def __init__(
        self,
        *,
        live_id: str,
        host: PluginHost[None],
        running: Callable[[], bool],
        queued: Callable[[], int] = lambda: 0,
        changed: Callable[[], None] = lambda: None,
        inputs: Callable[[], PromptInputs | None] = lambda: None,
    ) -> None:
        """`host` carries the session's context; `running` says whether a turn runs in it.

        `changed` tells the process's `LiveSessions` that its state may have changed. `inputs` is the session's
        prompt, which holds the input waiting for its turn and steers that turn; `None` holds none and steers nothing.
        `queued` counts the waiting input when `inputs` returns `None`; otherwise the count is the prompt's own
        queue length.
        """
        self._id = live_id
        self._host = host
        self._running = running
        self._queued = queued
        self._inputs = inputs
        self._changed = changed
        self._watchers = 0

    @property
    def id(self) -> str:
        """Stable while the session is live."""
        return self._id

    @property
    def session_id(self) -> str | None:
        """The session's current saved conversation ID."""
        return self._host.session_id

    @property
    def workspace(self) -> Path:
        """The directory the session works in."""
        return self._host.workspace

    @property
    def conversation(self) -> Conversation:
        """The session's retained history."""
        return self._host.conversation

    @property
    def status(self) -> Status:
        """The session's status row."""
        return self._host.status

    @property
    def sessions(self) -> SessionCatalog | None:
        """The saved conversations."""
        return self._host.sessions

    @property
    def tasks(self) -> TaskCatalog | None:
        """The session's delegated tasks."""
        return self._host.tasks

    @property
    def running(self) -> bool:
        """Whether a turn runs in the session now."""
        return self._running()

    def setting(self, key: str, /) -> JsonValue:
        """The effective value of a CLAI setting in the session."""
        return self._host.setting(key)

    def command_menu(self, name: str, /) -> SettingsMenu | None:
        """The menu bare `/NAME` opens in the session, built now from the session's own commands."""
        return self._host.command_menu(name)

    @property
    def queued(self) -> int:
        """How many inputs wait for the session's running turn or command to end."""
        inputs = self._inputs()
        return self._queued() if inputs is None else len(inputs.queue)

    @property
    def queue(self) -> list[QueuedInput]:
        """The inputs waiting for the session's running turn or command to end, oldest first."""
        inputs = self._inputs()
        return [] if inputs is None else inputs.queue

    def edit_queued(self, input_id: str, text: str, /) -> None:
        """Replace a waiting input's text, keeping its place."""
        if (inputs := self._inputs()) is None:
            raise LookupError(NOT_QUEUED)
        inputs.edit_queued(input_id, text)

    def remove_queued(self, input_id: str, /) -> None:
        """Drop a waiting input."""
        if (inputs := self._inputs()) is None:
            raise LookupError(NOT_QUEUED)
        inputs.remove_queued(input_id)

    def steer(self, text: str, /) -> bool:
        """Send `text` to the running turn now; `False` when nothing takes it."""
        inputs = self._inputs()
        return inputs is not None and inputs.steer(text)

    @property
    def watchers(self) -> int:
        """How many `watch` calls have not stopped watching yet."""
        return self._watchers

    def watch(self) -> Callable[[], None]:
        """Count a watcher, such as a browser showing the session, until the returned function is called."""
        self._watchers += 1
        self.changed()
        watching = True

        def unwatch() -> None:
            nonlocal watching
            if watching:
                watching = False
                self._watchers -= 1
                self.changed()

        return unwatch

    def changed(self) -> None:
        """Tell the listeners of the process's live sessions that this one's state may have changed."""
        self._changed()

    def send_input(self, text: str, /, *, images: Sequence[BinaryContent] = ()) -> None:
        """Enter `text` at the session's prompt, with `images` attached."""
        self._host.send_input(text, images=images)

    def interrupt(self) -> bool:
        """Cancel the session's running turn or command."""
        return self._host.interrupt()

    async def complete(self, text: str, /) -> list[Completion]:
        """What Tab offers at the session's prompt for `text`."""
        return await self._host.complete(text)


_CURRENT: ContextVar[LiveRuntime | None] = ContextVar('clai2_live_session', default=None)
"""The session whose moment, command, or run is being handled, for `LiveSessions.current`."""


@contextmanager
def handling(session: LiveRuntime) -> Generator[None]:
    """Name `session` as the one being handled for the block."""
    token = _CURRENT.set(session)
    try:
        yield
    finally:
        _CURRENT.reset(token)


class JoinedSessions:
    """`LiveSessions` over a list of sessions, by default those joined to one `ProcessPlugins`.

    It opens and closes none: a `ClaiProcess` replaces it with its `LiveRuntimes`, which do.
    """

    def __init__(self, sessions: Callable[[], list[LiveRuntime]]) -> None:
        self._sessions = sessions
        self._listeners: list[Callable[[], None]] = []

    def listing(self) -> list[LiveSession]:
        return list(self._sessions())

    def get(self, live_id: str, /) -> LiveSession:
        found = next((session for session in self._sessions() if session.id == live_id), None)
        if found is None:
            raise LookupError(f'No live session {live_id}')
        return found

    def current(self) -> LiveSession | None:
        current = _CURRENT.get()
        return current if any(session is current for session in self._sessions()) else None

    async def open(self, conversation_id: str, /) -> LiveSession:
        raise RuntimeError(_OPENS_NONE)

    async def new(self, *, workspace: Path | None = None) -> LiveSession:
        raise RuntimeError(_OPENS_NONE)

    async def close(self, live_id: str, /) -> None:
        raise RuntimeError(_OPENS_NONE)

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def changed(self, session: LiveRuntime) -> None:
        """Call every listener: `session`, or the listing, may have changed. A failing listener is logged."""
        for listener in list(self._listeners):
            try:
                listener()
            except Exception:
                # Called from a session's own loop, which a plugin's listener must not end.
                _LOGGER.exception('A live sessions listener failed')


_OPENS_NONE = 'This CLAI opens no other live sessions.'


class ProcessPlugins(Generic[DepsT]):
    """What every live session's plugins share: declarations, the change lock, imports, and process-scoped plugins.

    The `PluginCatalog` behind `host.plugins` stays per session (`SessionPlugins`), since what is loaded,
    included, or failed can differ between sessions; its changes come through `change` and `follow`.
    """

    def __init__(self, *, store: SettingsStore, builtin: Sequence[PluginSettings] = ()) -> None:
        """`builtin` ships with CLAI; the store overrides it, as it does each session's project declarations."""
        self.store = store
        self.builtin = canonical_plugin_declarations(builtin)
        self.staged: dict[str, PluginSettings] = {}
        """A `/plugins add` declaration being tried before it is saved, so rejected settings never reach the store."""
        self.editing: set[str] = set()
        """Plugins whose settings menu is open, in the terminal or for an edit from elsewhere, so one edits at a time."""
        self.changing = Lock()
        """Held by each `PluginCatalog` change and `remove`, which the panel and the terminal can run at once."""
        self._sessions: list[SessionPlugins[DepsT]] = []
        self._loading: dict[str, Lock] = {}
        self._shared: dict[str, LoadedPlugin[DepsT]] = {}
        # A process-scoped plugin unloaded while turns run ends after the last of them: the plugin and its
        # pending turns, by `id`. Holding the plugin keeps its `id` from being reused meanwhile.
        self._ending: dict[int, tuple[LoadedPlugin[DepsT], int]] = {}
        self.live: JoinedSessions = JoinedSessions(lambda: [session.live_session for session in self._sessions])
        """The live sessions, as process-scoped plugins see them through `host.live`; `ClaiProcess` sets its own."""

    @property
    def plugins_dir(self) -> Path:
        """Folder scanned for drop-in plugin files."""
        return self.store.plugins_dir

    def declared(self, project: Mapping[str, PluginSettings], *, console: Console) -> dict[str, PluginSettings]:
        """Every declaration a session with `project` sees: saved, staged, drop-in, project, then built-in."""
        folder = self._discover(console)
        declared = {declaration.id: self.upgrade(declaration) for declaration in self.store.plugins()}
        declared.update(self.staged)
        for name in folder.keys() - declared.keys():
            declared[name] = PluginSettings(id=name, factory=name, path=str(folder[name]))
        for shipped in (project, self.builtin):
            for name in shipped.keys() - declared.keys():
                declared[name] = shipped[name]
        return declared

    def upgrade(self, saved: PluginSettings) -> PluginSettings:
        """Resolve moved plugin paths and retired harness built-ins in stored declarations.

        Preserve custom settings when a local database uses an old CLAI module path.
        """
        old = saved
        factory = MOVED_FACTORIES.get(saved.factory)
        if factory is not None:
            saved = saved.model_copy(update={'factory': factory})
        current = self.builtin.get(saved.id)
        if current is not None and same_plugin(old, RETIRED_BUILTINS.get(saved.id)):
            return current.model_copy(update={'enabled': saved.enabled})
        return saved

    def stored(self, name: str) -> PluginSettings | None:
        """The stored declaration as it is now: another CLAI process may have replaced it."""
        return next((saved for saved in self.store.plugins() if saved.id == name), None)

    def ensure_plugins_dir(self, console: Console) -> None:
        """Create the drop-in folder, so installing a plugin is one copy into a folder that already exists."""
        try:
            self.store.plugins_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            console.print(f'Cannot create the plugins folder: {exc}', style=theme.color(theme.ERROR), markup=False)

    def _discover(self, console: Console) -> dict[str, Path]:
        folder = self.store.plugins_dir
        if not folder.is_dir():
            return {}
        found: dict[str, Path] = {}
        try:
            children = sorted(folder.iterdir())
        except OSError as exc:
            console.print(f'Cannot discover plugins: {exc}', style=theme.color(theme.ERROR), markup=False)
            return {}
        for child in children:
            name = child.stem if child.suffix == '.py' else child.name
            if not name.isidentifier() or name.startswith('_'):
                continue
            name = canonical_plugin_id(name)
            if child.is_file() and child.suffix == '.py':
                found[name] = child
            elif (child / '__init__.py').is_file():
                found[name] = child / '__init__.py'
        return found

    def import_module(self, name: str, declaration: PluginSettings, *, fresh: bool) -> ModuleType:
        """The declaration's module: a drop-in file afresh, else the installed module, reloaded when `fresh`."""
        if declaration.path is not None:
            path = Path(declaration.path)
            if not path.is_absolute():
                # Saved approvals are shared by every repository, so a relative path would run whatever
                # file sits at it under the launch directory. Project files are anchored before approval.
                raise ValueError(
                    f'{path} is relative, so it would load from whichever directory CLAI starts in. '
                    f'Use /plugins remove {name} and approve it again, or give it an absolute path.'
                )
            return import_file(name, path)
        module = importlib.import_module(declaration.factory.partition(':')[0])
        return importlib.reload(module) if fresh else module

    @property
    def sessions(self) -> tuple['SessionPlugins[DepsT]', ...]:
        """The live sessions' plugins, oldest first."""
        return tuple(self._sessions)

    def join(self, session: 'SessionPlugins[DepsT]') -> None:
        """Count `session` among the live ones, so it follows changes and gets process-scoped plugins."""
        if session not in self._sessions:
            self._sessions.append(session)
            self.live.changed(session.live_session)

    def leave(self, session: 'SessionPlugins[DepsT]') -> None:
        """Stop counting `session` among the live ones."""
        if session in self._sessions:
            self._sessions = [joined for joined in self._sessions if joined is not session]
            self.live.changed(session.live_session)

    def loading(self, name: str) -> Lock:
        """Held while a session loads `name`, so sessions load it one at a time and build a process plugin once."""
        return self._loading.setdefault(name, Lock())

    def here(self, origin: 'SessionPlugins[DepsT]') -> 'SessionPlugins[DepsT]':
        """The session being handled now, if it is live here; else `origin`, the one that loaded a plugin.

        Once `origin` has left, the oldest live session stands in for it, so a closed one never joins again.
        """
        current = _CURRENT.get()
        found = next((session for session in self._sessions if session.live_session is current), None)
        if found is not None or origin in self._sessions:
            return found if found is not None else origin
        return self._sessions[0] if self._sessions else origin

    async def change(
        self, name: str, apply: Callable[[], Awaitable[T]], *, origin: 'SessionPlugins[DepsT]', again: bool
    ) -> T:
        """Apply a change to `name` in `origin` under the change lock, then have every other live session follow it.

        They follow also when `origin` fails, since what it saved first, such as `enabled`, may load elsewhere.
        `again` loads it again where it stays loaded, as after new settings or a reload.
        """
        async with self.changing:
            try:
                result = await apply()
            except Exception:
                await self.follow(name, origin=origin, again=again)
                raise
            await self.follow(name, origin=origin, again=again)
            return result

    async def follow(self, name: str, *, origin: 'SessionPlugins[DepsT]', again: bool) -> None:
        """Have every live session but `origin` catch up with a change `origin` made to `name`."""
        for session in list(self._sessions):
            if session is not origin:
                await session.follow(name, again=again)

    def shared(self, name: str) -> LoadedPlugin[DepsT] | None:
        """The loaded process-scoped plugin `name`, if there is one."""
        return self._shared.get(name)

    def shared_plugins(self) -> list[tuple[str, LoadedPlugin[DepsT]]]:
        """Every loaded process-scoped plugin, in load order."""
        return list(self._shared.items())

    def register_shared(self, name: str, *, loaded: LoadedPlugin[DepsT]) -> None:
        """Record a process-scoped plugin as it starts; `forget_shared` drops it if starting fails."""
        self._shared[name] = loaded

    def forget_shared(self, name: str, *, loaded: LoadedPlugin[DepsT] | None) -> None:
        """Drop a process-scoped plugin that failed to start; no other session joined it, as `loading` held them."""
        if loaded is not None and self._shared.get(name) is loaded:
            del self._shared[name]

    async def share(self, name: str, loaded: LoadedPlugin[DepsT], *, origin: 'SessionPlugins[DepsT]') -> None:
        """Join a process-scoped plugin `origin` just loaded to every other live session."""
        for session in list(self._sessions):
            if session is not origin:
                await session.attach(name, loaded)

    async def unload_shared(self, name: str, *, reason: SessionEndReason, origin: 'SessionPlugins[DepsT]') -> None:
        """Take a process-scoped plugin out of every session, and end it once every running turn is over."""
        loaded = self._shared.pop(name)
        holders = [session for session in self._sessions if session.holds(name, loaded)]
        for session in holders:
            session.detach(name, loaded)
        # Only a turn that bound it may still use it.
        waiting = [session for session in holders if session.in_turn]
        if not waiting:
            await origin.end(name, loaded, reason=reason)
            return
        self._ending[id(loaded)] = (loaded, len(waiting))
        for session in waiting:
            session.retire(name, loaded, reason=reason)

    def last_end(self, loaded: LoadedPlugin[DepsT]) -> bool:
        """Whether ending `loaded` now is due: not a process-scoped plugin another running turn still holds."""
        ending = self._ending.get(id(loaded))
        if ending is None:
            return True
        if ending[1] > 1:
            self._ending[id(loaded)] = (loaded, ending[1] - 1)
            return False
        del self._ending[id(loaded)]
        return True

    async def close(self, reason: SessionEndReason, *, origin: 'SessionPlugins[DepsT]') -> None:
        """End every process-scoped plugin, last loaded first, as the process stops; `origin` reports failures.

        Closed from inside a turn of `origin`, they end once that turn is over, as an unload would.
        """
        for name in reversed(list(self._shared)):
            loaded = self._shared.pop(name)
            for session in self._sessions:
                session.detach(name, loaded)
            if not origin.retire(name, loaded, reason=reason):
                await origin.end(name, loaded, reason=reason)
