"""Tests for `SmolSandbox(desktop=True)`, against the fake smolvm SDK with a scripted X display."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.smol_sandbox import SmolSandbox

from .._fake_x_display import FakeDisplay, computer_returns, look_click_then_finish
from ..display_fakes import FAKE_SCREENSHOT
from .fake_smol import FakeSmol


async def test_a_desktop_vm_runs_debian_and_computer_use(fake_smol: FakeSmol, tmp_path: Path) -> None:
    display = FakeDisplay()
    fake_smol.answer = lambda argv: display(' '.join(argv), None) if display.answers(' '.join(argv)) else None
    sandbox = SmolSandbox(working_dir=str(tmp_path.resolve()), desktop=True)
    agent = Agent(
        FunctionModel(look_click_then_finish), capabilities=[sandbox, ComputerUse(native=False, settle_seconds=0)]
    )

    result = await agent.run('Click.')

    clicked = computer_returns(result.all_messages())[-1]
    assert clicked.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
    assert len(display.starts) == 1
    (vm,) = fake_smol.vms.values()
    assert vm.config.image == 'debian:bookworm-slim'

    # A plain command gets `DISPLAY` too, so a program the agent starts from the shell opens on the desktop.
    await result.workspace.run(['true'])
    command, options = fake_smol.commands[-1]
    assert command[-1] == 'true' and options is not None and options.env == {'DISPLAY': ':1'}

    # A backend attached by the run's ref reaches the same display.
    ref = result.workspace.ref
    assert ref is not None
    attached = sandbox.backend(ref).display
    assert attached is not None
    assert await attached.screenshot() == FAKE_SCREENSHOT
