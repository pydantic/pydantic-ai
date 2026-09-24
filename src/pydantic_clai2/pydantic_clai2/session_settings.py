"""Apply saved settings to the live session, never to a turn that is already running."""

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from typing import Generic, TypeVar

from pydantic_ai.usage import UsageLimits
from rich.console import Console

from . import theme
from ._session import Session
from .config import Settings

DepsT = TypeVar('DepsT')
OutputT = TypeVar('OutputT')
_SESSION_KEYS = ('model', 'run.tool_retries', 'run.request_limit')


class SessionSettings(Generic[DepsT, OutputT]):
    """`CommandContext.apply_setting` for the shell.

    A menu opened mid-turn saves its changes at once, but the running turn has
    already captured its model and limits, and still reads some of them from the
    session until it ends. Session changes made during a turn wait for it to end.
    """

    def __init__(self, *, session: Session[DepsT, OutputT], console: Console, settings: Settings) -> None:
        """Remember the theme in effect, so an unchanged theme is not repainted."""
        self.session = session
        self.console = console
        self._theme = settings.theme
        self._turn = False
        self._pending: dict[str, Settings] = {}

    def __call__(self, key: str, updated: Settings) -> None:
        """Apply `key` from `updated`, or hold a session change until the running turn ends."""
        if key in _SESSION_KEYS and self._turn:
            self._pending[key] = updated
        elif key in _SESSION_KEYS:
            self._apply(key, updated)
        elif key == 'display.theme' and self.console.is_terminal and self._theme != updated.theme:
            theme.apply(updated.theme, output=self.console.file)
        self._theme = updated.theme

    @contextmanager
    def turn(self) -> Generator[None]:
        """Hold session changes for the length of one turn, then apply the latest of each."""
        self._turn = True
        try:
            yield
        finally:
            self._turn = False
            pending, self._pending = self._pending, {}
            for key, updated in pending.items():
                self._apply(key, updated)

    def _apply(self, key: str, updated: Settings) -> None:
        if key == 'model':
            self.session.model = updated.model
        elif key == 'run.tool_retries':
            self.session.tool_retries = updated.tool_retries
        else:
            self.session.usage_limits = replace(
                self.session.usage_limits or UsageLimits(), request_limit=updated.request_limit
            )
