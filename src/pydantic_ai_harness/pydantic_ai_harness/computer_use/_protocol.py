"""The protocols a screen implements for `ComputerUse`: `Display`, its optional abilities, and `SupportsDisplay`.

A workspace backend with a screen, such as a sandbox started with a desktop, implements `SupportsDisplay`, and
`ComputerUse` drives the `Display` it returns. They are structural: a backend conforms by having the members, without
importing anything from here.

These protocols don't change once released, since adding a member would break every display written before it. A new
ability is a new optional protocol, like `SupportsZoom`, and a new kind of input is a new action type in
`DisplayAction`, which a display written earlier refuses with `UnsupportedActionError`.
"""

from __future__ import annotations as _annotations

from typing import Annotated, Protocol, TypeAlias, runtime_checkable

from pydantic import Field

from pydantic_ai.computer_use import (
    Click,
    Drag,
    HoldKeys,
    MouseDown,
    MouseUp,
    MoveMouse,
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
    'SupportsZoom',
    'UnsupportedActionError',
)

DisplayAction: TypeAlias = Annotated[
    Click | MoveMouse | Drag | MouseDown | MouseUp | Scroll | TypeText | PressKeys | HoldKeys,
    Field(discriminator='type'),
]
"""A mouse or keyboard action a `Display` performs, in its own pixels. New types may be added, so a display refuses
one it doesn't know with [`UnsupportedActionError`][pydantic_ai_harness.computer_use.UnsupportedActionError]."""


class UnsupportedActionError(WorkspaceError):
    """A [`Display`][pydantic_ai_harness.computer_use.Display] can't perform this action, or this value of it, exactly.

    `ComputerUse` tells the model, and the actions after it don't run.
    """


@runtime_checkable
class Display(Protocol):
    """A screen: screenshots, mouse and keyboard.

    Coordinates are pixels of the screenshot, with the origin at the top left. A display may start on its first call.
    It can also implement [`SupportsZoom`][pydantic_ai_harness.computer_use.SupportsZoom] and
    [`SupportsCursorPosition`][pydantic_ai_harness.computer_use.SupportsCursorPosition].
    """

    async def display_size(self) -> tuple[int, int]:
        """The next screenshot's `(width, height)` in pixels: the coordinate space of every action.

        It can change between calls, for example when a phone rotates.
        """
        ...

    async def screenshot(self) -> bytes:
        """Capture the whole screen as PNG bytes, at the size `display_size` reports."""
        ...

    async def act(self, action: DisplayAction) -> None:
        """Perform one action.

        Raise [`UnsupportedActionError`][pydantic_ai_harness.computer_use.UnsupportedActionError] for an action, or a
        value, the display can't perform exactly, including action types added after it was written.
        """
        ...


@runtime_checkable
class SupportsDisplay(Protocol):
    """A workspace backend with a [`Display`][pydantic_ai_harness.computer_use.Display] for `ComputerUse` to drive."""

    @property
    def display(self) -> Display | None:
        """This environment's display, or `None` for one without a screen, decided without I/O.

        It is the same object for the backend's lifetime, through restarts and reconnects.
        """
        ...


@runtime_checkable
class SupportsZoom(Protocol):
    """Optional on a [`Display`][pydantic_ai_harness.computer_use.Display]: a closer look at part of the screen."""

    async def zoom(self, region: Region) -> bytes:
        """Capture the region as PNG bytes, at the screen's full resolution.

        The region is in screenshot pixels.
        """
        ...


@runtime_checkable
class SupportsCursorPosition(Protocol):
    """Optional on a [`Display`][pydantic_ai_harness.computer_use.Display]: where the pointer is."""

    async def cursor_position(self) -> Point:
        """The pointer's position in screenshot pixels."""
        ...
