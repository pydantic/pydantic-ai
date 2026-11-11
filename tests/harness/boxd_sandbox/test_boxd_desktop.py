"""Tests for `BoxdSandbox(desktop=True)`, against the fake boxd SDK with a scripted X display."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox
from pydantic_ai_harness.computer_use import ComputerUse

from .._fake_x_display import FakeDisplay, computer_returns, look_click_then_finish
from ..display_fakes import FAKE_SCREENSHOT
from .fake_boxd import FakeBoxd


async def test_a_desktop_machine_runs_computer_use(host: FakeBoxd) -> None:
    display = FakeDisplay()

    def answer_display_commands(argv: list[str]) -> tuple[str, str, int] | None:
        command = ' '.join(argv[3:])  # After the backend's `env -- DISPLAY=:1` prefix.
        return display(command, None) if display.answers(command) else None

    host.answer = answer_display_commands
    sandbox = BoxdSandbox(desktop=True)
    agent = Agent(FunctionModel(look_click_then_finish), capabilities=[sandbox, ComputerUse(native=False)])

    result = await agent.run('Click.')

    clicked = computer_returns(result.all_messages())[-1]
    assert clicked.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
    assert len(display.starts) == 1

    # A plain command gets `DISPLAY` too, so a program the agent starts from the shell opens on the desktop.
    await result.workspace.run(['true'])
    assert [line for line in host.lines if line.startswith('exec ')][-1] == 'exec env -- DISPLAY=:1 true </dev/null'

    # A backend attached by the run's ref reaches the same display.
    ref = result.workspace.ref
    assert ref is not None
    attached = sandbox.backend(ref).display
    assert attached is not None
    assert await attached.screenshot() == FAKE_SCREENSHOT
