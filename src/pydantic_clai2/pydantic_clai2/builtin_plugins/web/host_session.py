"""The plugin's own host as the one live session, for a `web` plugin built outside CLAI's loader, such as in a test."""

from collections.abc import Callable, Sequence
from pathlib import Path

from pydantic import JsonValue
from termflow.tui.completion import Completion

from pydantic_ai.messages import BinaryContent
from pydantic_clai2.plugins import (
    Conversation,
    ForkInfo,
    LiveSession,
    PluginHost,
    QueuedInput,
    SessionCatalog,
    TaskCatalog,
)
from pydantic_clai2.plugins.menus import SettingsMenu
from pydantic_clai2.ui.prompt.prompt_queue import NOT_QUEUED
from pydantic_clai2.ui.rendering.status import Status

_OPENS_NONE = 'This CLAI opens no other live sessions.'


class HostSession:
    """`LiveSession` over a plugin host: the session it was built with, never idle-closed."""

    def __init__(self, host: PluginHost[None]) -> None:
        self._host = host

    @property
    def id(self) -> str:
        return 'host'

    @property
    def session_id(self) -> str | None:
        return self._host.session_id

    @property
    def workspace(self) -> Path:
        return self._host.workspace

    @property
    def conversation(self) -> Conversation:
        return self._host.conversation

    @property
    def status(self) -> Status:
        return self._host.status

    @property
    def sessions(self) -> SessionCatalog | None:
        return self._host.sessions

    @property
    def tasks(self) -> TaskCatalog | None:
        return self._host.tasks

    @property
    def forks(self) -> tuple[ForkInfo, ...]:
        return ()

    @property
    def running(self) -> bool:
        return False

    @property
    def queued(self) -> int:
        return 0

    @property
    def queue(self) -> list[QueuedInput]:
        return []

    def edit_queued(self, input_id: str, text: str, /) -> None:
        raise LookupError(NOT_QUEUED)

    def remove_queued(self, input_id: str, /) -> None:
        raise LookupError(NOT_QUEUED)

    def steer(self, text: str, /) -> bool:
        return False

    @property
    def watchers(self) -> int:
        return 0

    def watch(self) -> Callable[[], None]:
        return lambda: None

    def setting(self, key: str, /) -> JsonValue:
        return self._host.setting(key)

    def command_menu(self, name: str, /) -> SettingsMenu | None:
        return self._host.command_menu(name)

    def send_input(self, text: str, /, *, images: Sequence[BinaryContent] = ()) -> None:
        self._host.send_input(text, images=images)

    def interrupt(self) -> bool:
        return self._host.interrupt()

    async def complete(self, text: str, /) -> list[Completion]:
        return await self._host.complete(text)


class HostSessions:
    """`LiveSessions` holding only a `HostSession`; it opens and closes none."""

    def __init__(self, host: PluginHost[None]) -> None:
        self._session = HostSession(host)

    def listing(self) -> list[LiveSession]:
        return [self._session]

    def get(self, live_id: str, /) -> LiveSession:
        if live_id != self._session.id:
            raise LookupError(f'No live session {live_id}')
        return self._session

    def current(self) -> LiveSession | None:
        return None

    async def open(self, conversation_id: str, /) -> LiveSession:
        raise RuntimeError(_OPENS_NONE)

    async def new(self, *, workspace: Path | None = None) -> LiveSession:
        raise RuntimeError(_OPENS_NONE)

    async def close(self, live_id: str, /) -> None:
        raise RuntimeError(_OPENS_NONE)

    def subscribe(self, listener: Callable[[], None]) -> Callable[[], None]:
        return lambda: None
