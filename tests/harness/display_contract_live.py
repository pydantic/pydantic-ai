"""What every live desktop's display must do, checked once per backend against a real screen."""

from __future__ import annotations as _annotations

from pydantic_ai.computer_use import MoveMouse, Point, Region
from pydantic_ai_harness._png import png_size
from pydantic_ai_harness.computer_use import (
    Display,
    SupportsCursorPosition,
    SupportsZoom,
)


async def check_display_contract(display: Display) -> None:
    """Check that screenshots match the display size, zoom gives the size asked for, and the pointer moves.

    The display must be running, with a screenshot taken since it started.
    """
    width, height = await display.display_size()
    assert png_size(await display.screenshot()) == (width, height)
    if isinstance(display, SupportsZoom):
        assert png_size(await display.zoom(Region(left=10, top=20, right=110, bottom=70))) == (100, 50)
    if isinstance(display, SupportsCursorPosition):
        await display.act(MoveMouse(x=width // 3, y=height // 3))
        assert await display.cursor_position() == Point(x=width // 3, y=height // 3)
