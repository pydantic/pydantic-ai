"""Tests for `SmolSandbox(desktop=True)`, against the fake smolvm SDK with a scripted X display."""

from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.computer_use import Click
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.smol_sandbox import SmolSandbox

from .._fake_x_display import FakeDisplay
from ..display_fakes import screen
from .fake_smol import FakeSmol


async def test_a_desktop_vm_runs_debian_and_drives_its_display(fake_smol: FakeSmol, tmp_path: Path) -> None:
    display = FakeDisplay()
    fake_smol.answer = lambda argv: display(' '.join(argv), None) if display.answers(' '.join(argv)) else None
    sandbox = SmolSandbox(working_dir=str(tmp_path.resolve()), desktop=True)
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[sandbox])

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        desktop = screen(ctx.workspace.backend)
        size = await desktop.display_size()
        await desktop.act(Click(x=3, y=4))
        await ctx.workspace.run(['true'])
        assert ctx.workspace.ref is not None and sandbox.backend(ctx.workspace.ref).display is not None
        return 'x'.join(map(str, size))

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800"}'
    (vm,) = fake_smol.vms.values()
    assert vm.config.image == 'debian:bookworm-slim'
    assert len(display.starts) == 1
    assert display.xdotool[-1].endswith('mousemove 3 4 click --repeat 1 1')
    # A plain command gets `DISPLAY` too, so a program the agent starts from the shell opens on the desktop.
    command, options = fake_smol.commands[-1]
    assert command[-1] == 'true' and options is not None and options.env == {'DISPLAY': ':1'}
