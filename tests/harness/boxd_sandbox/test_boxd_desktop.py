"""Tests for `BoxdSandbox(desktop=True)`, against the fake boxd SDK with a scripted X display."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.computer_use import Click
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox

from .._fake_x_display import FakeDisplay
from ..display_fakes import screen
from .fake_boxd import FakeBoxd


async def test_a_desktop_machine_drives_its_display(host: FakeBoxd) -> None:
    display = FakeDisplay()

    def answer_display_commands(argv: list[str]) -> tuple[str, str, int] | None:
        command = ' '.join(argv[3:])  # After the backend's `env -- DISPLAY=:1` prefix.
        return display(command, None) if display.answers(command) else None

    host.answer = answer_display_commands
    sandbox = BoxdSandbox(desktop=True)
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
    assert len(display.starts) == 1
    assert display.xdotool[-1].endswith('mousemove 3 4 click --repeat 1 1')
    # A plain command gets `DISPLAY` too, so a program the agent starts from the shell opens on the desktop.
    assert [line for line in host.lines if line.startswith('exec ')][-1] == 'exec env -- DISPLAY=:1 true </dev/null'
