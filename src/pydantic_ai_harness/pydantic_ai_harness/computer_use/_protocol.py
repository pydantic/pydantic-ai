"""The protocols a screen implements for `ComputerUse`: `Display`, its optional extras, and `SupportsDisplay`.

A workspace backend with a screen implements `SupportsDisplay`, and `ComputerUse` drives the `Display` it returns. A
backend doesn't have to inherit from these classes or import them: any class with the right methods counts.

These protocols don't change once released, because a new method would break every existing display. A new ability
gets its own optional protocol, like `SupportsZoom`. A new kind of input gets a new action type in `DisplayAction`,
and an older display refuses it with `UnsupportedActionError`.
"""

from __future__ import annotations as _annotations

from typing import Annotated, Protocol, TypeAlias, runtime_checkable

from pydantic import Field

from pydantic_ai.computer_use import (
    Click,
    Drag,
    HoldKeys,
    MouseDown,
    MouseMove,
    MouseUp,
    Point,
    PressKeys,
    Region,
    Scroll,
    TypeText,
)
from pydantic_ai.workspaces import WorkspaceError

__all__ = (
    'Display',
    'DisplayAction',
    'SupportsCursorPosition',
    'SupportsDisplay',
    'SupportsLiveView',
    'SupportsZoom',
    'UnsupportedActionError',
)

DisplayAction: TypeAlias = Annotated[
    Click | MouseMove | Drag | MouseDown | MouseUp | Scroll | TypeText | PressKeys | HoldKeys,
    Field(discriminator='type'),
]
"""A mouse or keyboard action for a `Display` to run, with positions in the display's own pixels.

Later releases may add action types, so a display raises
[`UnsupportedActionError`][pydantic_ai_harness.computer_use.UnsupportedActionError] for a type it doesn't know."""


class UnsupportedActionError(WorkspaceError):
    """Raised by a [`Display`][pydantic_ai_harness.computer_use.Display] that can't run an action, or this value of it.

    Raise it only before the action has changed anything on the screen. `ComputerUse` tells the model the action
    failed, and the actions after it in the call don't run.
    """


@runtime_checkable
class Display(Protocol):
    """A screen that takes screenshots, mouse input and keyboard input.

    Positions are in screenshot pixels, counted from the top left corner. A display may start its screen on the first
    method call. It can also implement [`SupportsZoom`][pydantic_ai_harness.computer_use.SupportsZoom] and
    [`SupportsCursorPosition`][pydantic_ai_harness.computer_use.SupportsCursorPosition], and
    [`SupportsLiveView`][pydantic_ai_harness.computer_use.SupportsLiveView].
    """

    async def display_size(self) -> tuple[int, int]:
        """The `(width, height)` of the next screenshot, in pixels. Every action's positions are in this space.

        The size can change between calls.
        """
        ...

    async def screenshot(self) -> bytes:
        """Capture the whole screen as PNG bytes, at the size `display_size` reports."""
        ...

    async def act(self, action: DisplayAction) -> None:
        """Run one action.

        Raise [`UnsupportedActionError`][pydantic_ai_harness.computer_use.UnsupportedActionError] for an action, or a
        value of it, that the display can't run exactly. That includes action types added after the display was
        written. Raise it only before the action has changed anything on the screen.
        """
        ...


@runtime_checkable
class SupportsDisplay(Protocol):
    """A workspace backend with a [`Display`][pydantic_ai_harness.computer_use.Display] for `ComputerUse` to drive."""

    @property
    def display(self) -> Display | None:
        """This workspace's display, or `None` when it has no screen. Reading it must not do I/O.

        It returns the same object for as long as the backend exists, even after the screen restarts or reconnects.
        """
        ...


@runtime_checkable
class SupportsZoom(Protocol):
    """An optional extra for a [`Display`][pydantic_ai_harness.computer_use.Display]: capture part of the screen."""

    async def zoom(self, region: Region) -> bytes:
        """Capture the region as PNG bytes. The region and the image are in screenshot pixels."""
        ...


@runtime_checkable
class SupportsCursorPosition(Protocol):
    """An optional extra for a [`Display`][pydantic_ai_harness.computer_use.Display]: report where the pointer is."""

    async def cursor_position(self) -> Point:
        """The pointer's position in screenshot pixels."""
        ...


@runtime_checkable
class SupportsLiveView(Protocol):
    """An optional extra for a [`Display`][pydantic_ai_harness.computer_use.Display]: a web page that shows the screen.

    `ComputerUse` doesn't use it. It is for an app that lets a person watch the agent work.
    """

    async def live_view_url(self) -> str:
        """A URL a browser can open to watch the screen live, view-only where the provider allows it.

        Anyone with the URL may be able to see the screen, so treat it like a password. Raise `WorkspaceError` when
        the URL can't be made.
        """
        ...
