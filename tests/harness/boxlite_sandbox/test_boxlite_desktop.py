"""Tests for `BoxLiteSandbox(desktop=True)`, against the fake BoxLite SDK with a scripted X display."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.boxlite_sandbox import BoxLiteSandbox, BoxLiteSandboxBackend
from pydantic_ai_harness.computer_use import ComputerUse

from ..computer_use._fake_x_display import FakeDisplay
from ..computer_use.display_fakes import FAKE_SCREENSHOT, full_display, look_click_then_finish
from .fake_boxlite import FakeBoxLite


@pytest.fixture
def display(fake_boxlite: FakeBoxLite) -> FakeDisplay:
    """Answers the display's commands; the fake box runs every other command on the host."""
    display = FakeDisplay()

    def answer(argv: list[str]) -> tuple[str, str, int] | None:
        command = ' '.join(argv)
        return display(command, None) if display.answers(command) else None

    fake_boxlite.answer = answer
    return display


async def test_a_desktop_box_runs_debian_and_the_computer_tool(
    fake_boxlite: FakeBoxLite, display: FakeDisplay, tmp_path: Path
) -> None:
    sandbox = BoxLiteSandbox(working_dir=str(tmp_path.resolve()), desktop=(1440, 900), allow_internet_access=True)
    agent = Agent(FunctionModel(look_click_then_finish), capabilities=[sandbox, ComputerUse(native=False)])

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert 'fluxbox tint2 xterm' in start and 'setsid fluxbox' in start
    # The default Alpine image has no `apt-get` to install the display programs with.
    (box,) = fake_boxlite.boxes.values()
    assert box.options.image == 'debian:bookworm-slim'


async def test_a_desktop_box_without_network_needs_an_image_with_the_display_programs(
    fake_boxlite: FakeBoxLite, display: FakeDisplay, tmp_path: Path
) -> None:
    with pytest.raises(UserError, match='needs `allow_internet_access=True`'):
        BoxLiteSandboxBackend(desktop=True)

    await BoxLiteSandboxBackend(working_dir=str(tmp_path.resolve()), image='my-desktop', desktop=True).get_sandbox()
    (box,) = fake_boxlite.boxes.values()
    assert box.options.image == 'my-desktop'

    # An existing box already has whatever its display needs.
    BoxLiteSandboxBackend(sandbox=box, desktop=True)


async def test_a_shell_command_finds_the_desktop_running(
    fake_boxlite: FakeBoxLite, display: FakeDisplay, tmp_path: Path
) -> None:
    # A program the shell starts opens on the screen, so a command starts the display.
    backend = BoxLiteSandboxBackend(
        working_dir=str(tmp_path.resolve()), desktop=True, allow_internet_access=True, env={'FOO': 'bar'}
    )
    result = await backend.run('printf %s "$DISPLAY $FOO"', shell=True)

    assert result.stdout == ':1 bar'
    (start,) = display.starts
    assert start.endswith(' sh 1280x800')


async def test_a_desktop_box_reattaches_with_its_display(
    fake_boxlite: FakeBoxLite, display: FakeDisplay, tmp_path: Path
) -> None:
    owner = BoxLiteSandboxBackend(working_dir=str(tmp_path.resolve()))
    await owner.get_sandbox()
    assert owner.ref is not None

    # Attaching installs nothing it can't, so the network setting for new boxes doesn't stop it.
    attached = BoxLiteSandboxBackend(working_dir=str(tmp_path.resolve()), ref=owner.ref, desktop=(1440, 900))
    await full_display(attached).display_size()
    # The capability hands its desktop size to the backend it attaches. Defining it checks the options for a new box.
    sandbox = BoxLiteSandbox(working_dir=str(tmp_path.resolve()), image='my-desktop', desktop=(1440, 900))
    await full_display(sandbox.backend(owner.ref)).display_size()

    assert [start.endswith(' sh 1440x900') for start in display.starts] == [True, True]
    assert BoxLiteSandbox().backend(owner.ref).display is None


@pytest.mark.usefixtures('fake_boxlite')
def test_a_desktop_size_is_checked_when_the_capability_is_created() -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        BoxLiteSandbox(desktop=(0, 800))
