"""Shared lifetime bookkeeping for waits exposed by live-session adapters."""

from collections.abc import Callable


class UserWaits:
    """Overlapping user waits, each ended independently by its own idempotent callback."""

    def __init__(self, *, changed: Callable[[], None] = lambda: None) -> None:
        self._waits: dict[object, str] = {}
        self._changed = changed

    @property
    def current(self) -> str | None:
        """The most recently opened wait's reason, or `None`."""
        return next(reversed(self._waits.values()), None)

    def start(self, reason: str) -> Callable[[], None]:
        """Report a nonempty reason until the returned callback ends that wait."""
        if not reason:
            raise ValueError('User-wait reason must not be empty.')
        token = object()
        self._waits[token] = reason
        self._changed()

        def finished() -> None:
            if token in self._waits:
                del self._waits[token]
                self._changed()

        return finished
