"""Tests for `ModalSandbox(desktop=True)`, against the fake Modal SDK with a scripted X display."""

from __future__ import annotations

import warnings

from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxNoToolsWarning

from .._fake_x_display import FakeDisplay, computer_returns, look_click_then_finish
from ..display_fakes import FAKE_SCREENSHOT
from .fake_modal import FakeImage, FakeModal


async def test_a_desktop_sandbox_has_the_display_programs_and_runs_computer_use(fake_modal: FakeModal) -> None:
    display = FakeDisplay()
    fake_modal.responder = lambda argv, timeout: display(' '.join(argv), timeout)
    sandbox = ModalSandbox(desktop=True)
    agent = Agent(FunctionModel(look_click_then_finish), capabilities=[sandbox, ComputerUse(native=False)])

    with warnings.catch_warnings():
        warnings.simplefilter('error', ModalSandboxNoToolsWarning)
        result = await agent.run('Click.')

    clicked = computer_returns(result.all_messages())[-1]
    assert clicked.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
    assert len(display.starts) == 1
    image = fake_modal.create_kwargs[-1]['image']
    assert isinstance(image, FakeImage)
    assert image.apt_packages == ('git', 'ripgrep', 'xvfb', 'xdotool', 'scrot', 'x11-utils')

    # A plain command gets `DISPLAY` too, so a program the agent starts from the shell opens on the desktop.
    await result.workspace.run(['true'])
    command = fake_modal.sandboxes[0].exec_calls[-1]
    assert command.argv[-1] == 'true' and command.env == {'DISPLAY': ':1'}

    # A backend attached by the run's ref reaches the same display.
    ref = result.workspace.ref
    assert ref is not None
    attached = sandbox.backend(ref).display
    assert attached is not None
    assert await attached.screenshot() == FAKE_SCREENSHOT
