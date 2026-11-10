"""Tests for `ModalSandbox(desktop=True)`, against the fake Modal SDK with a scripted X display."""

from __future__ import annotations

import warnings

from pydantic_ai import Agent, RunContext
from pydantic_ai.computer_use import Click
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxNoToolsWarning

from .._fake_x_display import FakeDisplay
from ..display_fakes import screen
from .fake_modal import FakeImage, FakeModal


async def test_a_desktop_sandbox_has_the_display_programs_and_drives_its_display(fake_modal: FakeModal) -> None:
    display = FakeDisplay()
    fake_modal.responder = lambda argv, timeout: display(' '.join(argv), timeout)
    sandbox = ModalSandbox(desktop=True)
    agent = Agent(TestModel(call_tools=['computer']), capabilities=[sandbox])

    # Named like `ComputerUse`'s tool, which counts as using the sandbox.
    @agent.tool(name='computer')
    async def computer(ctx: RunContext[object]) -> str:
        size = await screen(ctx.workspace.backend).display_size()
        await screen(ctx.workspace.backend).act(Click(x=3, y=4))
        await ctx.workspace.run(['true'])
        assert ctx.workspace.ref is not None and sandbox.backend(ctx.workspace.ref).display is not None
        return 'x'.join(map(str, size))

    with warnings.catch_warnings():
        warnings.simplefilter('error', ModalSandboxNoToolsWarning)
        result = await agent.run('go')

    assert result.output == '{"computer":"1280x800"}'
    image = fake_modal.create_kwargs[-1]['image']
    assert isinstance(image, FakeImage)
    assert image.apt_packages == ('git', 'ripgrep', 'xvfb', 'xdotool', 'scrot', 'x11-utils')
    assert len(display.starts) == 1
    assert display.xdotool[-1].endswith('mousemove 3 4 click --repeat 1 1')
    # A plain command gets `DISPLAY` too, so a program the agent starts from the shell opens on the desktop.
    command = fake_modal.sandboxes[0].exec_calls[-1]
    assert command.argv[-1] == 'true' and command.env == {'DISPLAY': ':1'}
