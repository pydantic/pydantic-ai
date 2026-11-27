"""Tests for `SmolSandbox(desktop=True)`, against the fake smolvm SDK with a scripted X display."""

from __future__ import annotations

from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.smol_sandbox import SmolSandbox, SmolSandboxBackend

from ..computer_use._fake_x_display import FakeDisplay
from ..computer_use.display_fakes import FAKE_SCREENSHOT, full_display, look_click_then_finish
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
    sandbox = SmolSandbox(working_dir=str(tmp_path.resolve()), desktop=(1440, 900), allow_internet_access=True)
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
    (vm,) = fake_smol.vms.values()
    assert vm.config.image == 'debian:bookworm-slim'


async def test_a_desktop_vm_without_network_needs_an_image_with_the_display_programs(
    fake_smol: FakeSmol, display: FakeDisplay, tmp_path: Path
) -> None:
    with pytest.raises(UserError, match='needs `allow_internet_access=True`'):
        SmolSandboxBackend(desktop=True)

    await SmolSandboxBackend(working_dir=str(tmp_path.resolve()), image='my-desktop', desktop=True).get_sandbox()
    (vm,) = fake_smol.vms.values()
    assert vm.config.image == 'my-desktop'


async def test_a_shell_command_finds_the_desktop_running(
    fake_smol: FakeSmol, display: FakeDisplay, tmp_path: Path
) -> None:
    # A program the shell starts opens on the screen, so a command starts the display.
    backend = SmolSandboxBackend(
        working_dir=str(tmp_path.resolve()), desktop=True, allow_internet_access=True, env={'FOO': 'bar'}
    )
    result = await backend.run('printf %s "$DISPLAY $FOO"', shell=True)

    assert result.stdout == ':1 bar'
    assert len(display.starts) == 1


async def test_a_desktop_vm_reattaches_with_its_display(
    fake_smol: FakeSmol, display: FakeDisplay, tmp_path: Path
) -> None:
    owner = SmolSandboxBackend(working_dir=str(tmp_path.resolve()))
    await owner.get_sandbox()
    assert owner.ref is not None

    # Attaching installs nothing it can't, so the network setting for new VMs doesn't stop it.
    sandbox = SmolSandbox(working_dir=str(tmp_path.resolve()), desktop=(1440, 900))
    attached = sandbox.backend(owner.ref)
    await full_display(attached).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert SmolSandbox().backend(owner.ref).display is None


def test_a_desktop_size_is_checked_when_the_capability_is_created() -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        SmolSandbox(desktop=(0, 800))
