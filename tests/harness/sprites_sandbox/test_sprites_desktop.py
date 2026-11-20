"""Tests for `SpritesSandbox(desktop=True)`, against the fake Sprites SDK with a scripted X display."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.workspaces import WorkspaceUnavailableError
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox, SpritesSandboxBackend

from .._fake_x_display import FakeDisplay
from ..display_fakes import FAKE_SCREENSHOT, look_click_then_finish, screen
from .fake_sprites import SpriteTransport


@pytest.fixture
def display(transport: SpriteTransport) -> FakeDisplay:
    """Answers the display's commands; the Sprite's fake runs every other command."""
    display = FakeDisplay()

    def answer(argv: list[str]) -> tuple[str, str, int] | None:
        command = ' '.join(argv)
        return display(command, None) if display.answers(command) else None

    transport.answer = answer
    return display


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_sprite(display: FakeDisplay) -> None:
    agent = Agent(
        FunctionModel(look_click_then_finish), capabilities=[SpritesSandbox(desktop=True), ComputerUse(native=False)]
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert len(display.starts) == 1
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)


async def test_a_shell_command_finds_the_desktop_running(display: FakeDisplay) -> None:
    # The shell is told a program it starts opens on the screen, before anything has looked at the screen.
    backend = SpritesSandboxBackend(desktop=True, env={'FOO': 'bar'})
    try:
        result = await backend.run('printf %s "$DISPLAY $FOO"', shell=True)
    finally:
        await backend.aclose()

    assert result.stdout == ':1 bar'
    assert len(display.starts) == 1


async def test_a_desktop_size_sets_the_display_it_starts(display: FakeDisplay) -> None:
    backend = SpritesSandboxBackend(desktop=(1440, 900))
    try:
        await screen(backend).display_size()
    finally:
        await backend.aclose()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    # The desktop: its programs installed when missing, and the window manager started.
    assert 'fluxbox tint2 xterm' in start and 'setsid fluxbox' in start


async def test_a_display_that_does_not_start_names_the_sprite_and_what_it_needs(transport: SpriteTransport) -> None:
    def answer(argv: list[str]) -> tuple[str, str, int] | None:
        needs = "the display needs Xvfb, and there's no apt-get to install them with"
        return ('', needs, 1) if 'Xvfb' in ' '.join(argv) else None

    transport.answer = answer
    backend = SpritesSandboxBackend(desktop=True)
    try:
        with pytest.raises(WorkspaceUnavailableError) as raised:
            await screen(backend).display_size()
    finally:
        await backend.aclose()

    assert backend.ref is not None
    assert str(raised.value) == (
        f'Could not start the display in Sprite {backend.ref.id}: '
        "the display needs Xvfb, and there's no apt-get to install them with"
    )


async def test_a_desktop_sprite_reattaches_with_its_display(display: FakeDisplay) -> None:
    owner = SpritesSandboxBackend()
    try:
        await owner.get_sandbox()
    finally:
        await owner.aclose()
    assert owner.ref is not None

    attached = SpritesSandbox(desktop=(1440, 900)).backend(owner.ref)
    try:
        await screen(attached).display_size()
    finally:
        await attached.aclose()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert SpritesSandbox().backend(owner.ref).display is None
