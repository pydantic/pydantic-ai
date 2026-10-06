"""Usage in `/accounts`: fetched in the background while the menu is open, drawn as each one arrives.

The fetches run on the event loop and only record results; the menu thread redraws itself when
`changed` is set, so nothing touches the screen from outside the menu.
"""

from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from threading import Event

import anyio
import anyio.lowlevel
from anyio import CancelScope, fail_after
from anyio.abc import TaskGroup

from pydantic_clai2.errors import error_message
from pydantic_clai2.models.accounts import Account
from pydantic_clai2.models.usage import TIMEOUT, UsageFetch
from pydantic_clai2.plugins import AccountUsage, UsageWindow

UsageState = AccountUsage | str | None
"""An account's usage once loaded, why it is unavailable, or `None` while it loads."""

_BAR = 10


class UsageBoard:
    """Each listed account's usage state, keyed by its `/login` name."""

    def __init__(self, *, now: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        """`now` dates reset times; tests pass a fixed clock."""
        self._states: dict[str, UsageState] = {}
        self._running: dict[str, tuple[CancelScope, anyio.Event]] = {}
        """Each fetch still in flight: the scope that cancels it and the event set once it has ended."""
        self._now = now
        self.changed = Event()
        """Set when a result arrives; the menu clears it and redraws."""

    def load(self, tasks: TaskGroup, items: Iterable[Account], fetcher: Callable[[Account], UsageFetch | None]) -> None:
        """Start fetching each signed-in account not loaded yet, in `tasks`, without waiting."""
        for item in items:
            if not item.signed_in or item.login in self._states:
                continue
            fetch = fetcher(item)
            if fetch is not None:
                self._states[item.login] = None
                # Registered before the task starts, so `stop` can cancel a fetch that has not begun.
                running = self._running[item.login] = (CancelScope(), anyio.Event())
                tasks.start_soon(self._fetch, item.login, fetch, running)

    def forget(self, login: str) -> None:
        """Drop an account's usage, after it signs in again, so the next `load` fetches it afresh."""
        self._states.pop(login, None)

    async def stop(self, login: str) -> None:
        """Cancel the account's fetch and wait until it has ended, then drop its usage.

        Call before signing the account out, so its usage is not fetched, or shown, for an account that
        is gone. A refresh the fetch started cannot sign it back in either way: the credential store only
        replaces a login that still exists (`replace_credentials`).
        """
        self.forget(login)
        running = self._running.pop(login, None)
        if running is not None:
            scope, ended = running
            scope.cancel()
            await ended.wait()

    async def _fetch(self, login: str, fetch: UsageFetch, running: tuple[CancelScope, anyio.Event]) -> None:
        scope, ended = running
        try:
            with scope:
                await anyio.lowlevel.checkpoint()  # a fetch stopped before it began never starts
                state: UsageState
                try:
                    with fail_after(TIMEOUT + 5):
                        state = await fetch()
                except TimeoutError:
                    state = 'timed out'
                except Exception as exc:  # noqa: BLE001 -- a plugin's failing fetch must not close the menu.
                    state = ' '.join(error_message(exc).split()) or type(exc).__name__
                if scope.cancel_called:
                    return  # stopped while a thread finished: its account is no longer shown
                self._states[login] = state
                self.changed.set()
        finally:
            if self._running.get(login) is running:
                del self._running[login]
            ended.set()

    def summary(self, login: str) -> str:
        """The row's usage, such as `5h 92% · 7d 43%`: `…` while loading, `?` when unavailable."""
        if login not in self._states:
            return ''
        state = self._states[login]
        if state is None:
            return '…'
        if isinstance(state, str):
            return '?'
        return ' · '.join(f'{window.label} {window.used_percent:.0f}%' for window in state.windows[:2])

    def details(self, login: str) -> list[str]:
        """The details panel's usage lines: a bar and reset time per window."""
        if login not in self._states:
            return []
        state = self._states[login]
        if state is None:
            return ['usage    loading…']
        if isinstance(state, str):
            return ['usage    unavailable', state]
        heading = 'usage' if state.plan is None else f'usage    {state.plan} plan'
        if not state.windows:
            return [heading, 'no metered limits']
        return [heading, *(line for window in state.windows for line in self._window(window))]

    def _window(self, window: UsageWindow) -> list[str]:
        filled = round(min(max(window.used_percent, 0), 100) / 100 * _BAR)
        lines = [f'{window.label:<8} {"█" * filled}{"░" * (_BAR - filled)} {window.used_percent:3.0f}%']
        if window.resets_at is not None:
            lines.append(f'         resets {self._until(window.resets_at)}')
        return lines

    def _until(self, moment: datetime) -> str:
        minutes = max(0, int((moment - self._now()).total_seconds() // 60))
        days, minutes = divmod(minutes, 24 * 60)
        hours, minutes = divmod(minutes, 60)
        if days:
            return f'in {days}d {hours}h'
        return f'in {hours}h {minutes}m' if hours else f'in {minutes}m'
