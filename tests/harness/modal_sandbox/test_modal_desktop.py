"""Tests for `ModalSandbox(desktop=True)`, against the fake Modal SDK with a scripted X display."""

from __future__ import annotations

import warnings

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxBackend, ModalSandboxNoToolsWarning

from ..computer_use._fake_x_display import FakeDisplay
from ..computer_use.display_fakes import FAKE_SCREENSHOT, full_display, look_click_then_finish
from .fake_modal import FakeImage, FakeModal


@pytest.fixture
def display(fake_modal: FakeModal) -> FakeDisplay:
    display = FakeDisplay()
    fake_modal.responder = lambda argv, timeout: display(' '.join(argv), timeout)
    return display


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_sandbox(
    fake_modal: FakeModal, display: FakeDisplay
) -> None:
    agent = Agent(
        FunctionModel(look_click_then_finish), capabilities=[ModalSandbox(desktop=True), ComputerUse(native=False)]
    )

    with warnings.catch_warnings():
        # The `computer` tool reaches the sandbox, so the agent has a tool for it.
        warnings.simplefilter('error', ModalSandboxNoToolsWarning)
        result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    (start,) = display.starts
    assert 'apt_get install' in start and 'fluxbox tint2 xterm' in start and 'setsid fluxbox' in start
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)


async def test_the_default_image_has_the_display_programs(fake_modal: FakeModal, display: FakeDisplay) -> None:
    await ModalSandboxBackend(desktop=True).get_sandbox()
    await ModalSandboxBackend().get_sandbox()

    with_desktop, without = [kwargs['image'] for kwargs in fake_modal.create_kwargs]
    assert isinstance(with_desktop, FakeImage) and isinstance(without, FakeImage)
    assert with_desktop.apt_packages == (
        'git',
        'ripgrep',
        'xvfb',
        'xdotool',
        'x11-utils',
        'scrot',
        'util-linux',
        'fluxbox',
        'tint2',
        'xterm',
    )
    assert without.apt_packages == ('git', 'ripgrep')


async def test_a_shell_command_finds_the_desktop_running(fake_modal: FakeModal, display: FakeDisplay) -> None:
    # A program the shell starts opens on the screen, so a command starts the display first. The fake refuses
    # `xwininfo -root` until the display is up.
    result = await ModalSandboxBackend(desktop=True, env={'FOO': 'bar'}).run('xwininfo -root', shell=True)

    assert 'Width: 1280' in result.stdout
    assert len(display.starts) == 1
    command = fake_modal.sandboxes[0].exec_calls[-1]
    assert command.argv[-1] == 'xwininfo -root'
    assert command.env == {'DISPLAY': ':1', 'FOO': 'bar'}


async def test_a_desktop_size_sets_the_display_it_starts(fake_modal: FakeModal, display: FakeDisplay) -> None:
    await full_display(ModalSandboxBackend(desktop=(1440, 900))).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')


async def test_a_desktop_sandbox_reattaches_with_its_display(fake_modal: FakeModal, display: FakeDisplay) -> None:
    owner = ModalSandboxBackend()
    await owner.get_sandbox()
    assert owner.ref is not None

    await full_display(ModalSandbox(desktop=(1440, 900)).backend(owner.ref)).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert ModalSandbox().backend(owner.ref).display is None


def test_a_desktop_size_is_checked_when_the_capability_is_created() -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        ModalSandbox(desktop=(0, 800))
