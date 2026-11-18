"""Tests for `SmolSandbox(desktop=True)`, against the fake smolvm SDK with a scripted X display."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.smol_sandbox import SmolSandbox, SmolSandboxBackend

from .._fake_x_display import FakeDisplay
from ..display_fakes import FAKE_SCREENSHOT, look_click_then_finish, screen
from .fake_smol import FakeSmol


@pytest.fixture
def display(fake_smol: FakeSmol) -> FakeDisplay:
    """Answers the display's commands; the fake VM runs every other command on the host."""
    display = FakeDisplay()

    def answer(argv: list[str]) -> tuple[str, str, int] | None:
        command = ' '.join(argv)
        return display(command, None) if display.answers(command) else None

    fake_smol.answer = answer
    return display


async def test_a_desktop_vm_runs_debian_and_the_computer_tool(
    fake_smol: FakeSmol, display: FakeDisplay, tmp_path: Path
) -> None:
    sandbox = SmolSandbox(working_dir=str(tmp_path.resolve()), desktop=True)
    agent = Agent(FunctionModel(look_click_then_finish), capabilities=[sandbox, ComputerUse(native=False)])

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
    assert len(display.starts) == 1
    # The default Alpine image has no `apt-get` to install the display programs with.
    (vm,) = fake_smol.vms.values()
    assert vm.config.image == 'debian:bookworm-slim'


async def test_a_shell_command_finds_the_desktop_running(
    fake_smol: FakeSmol, display: FakeDisplay, tmp_path: Path
) -> None:
    # The shell is told a program it starts opens on the screen, before anything has looked at the screen.
    backend = SmolSandboxBackend(working_dir=str(tmp_path.resolve()), desktop=True, env={'FOO': 'bar'})
    result = await backend.run('printf %s "$DISPLAY $FOO"', shell=True)

    assert result.stdout == ':1 bar'
    assert len(display.starts) == 1


async def test_a_desktop_vm_reattaches_with_its_display(
    fake_smol: FakeSmol, display: FakeDisplay, tmp_path: Path
) -> None:
    owner = SmolSandboxBackend(working_dir=str(tmp_path.resolve()))
    await owner.get_sandbox()
    assert owner.ref is not None

    attached = SmolSandbox(working_dir=str(tmp_path.resolve()), desktop=(1440, 900)).backend(owner.ref)
    await screen(attached).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert SmolSandbox().backend(owner.ref).display is None
