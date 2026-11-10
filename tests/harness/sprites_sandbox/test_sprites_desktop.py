"""Tests for `SpritesSandbox(desktop=True)`, against the fake Sprites SDK with a scripted X display."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.computer_use import Click
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.sprites_sandbox import SpritesSandbox

from .._fake_x_display import FakeDisplay
from ..display_fakes import screen
from .fake_sprites import SpriteTransport


async def test_a_desktop_sprite_drives_its_display(transport: SpriteTransport) -> None:
    display = FakeDisplay()

    def answer_display_commands(argv: list[str]) -> tuple[str, str, int] | None:
        command = ' '.join(argv[3:])  # After the backend's `env -- DISPLAY=:1` prefix.
        return display(command, None) if display.answers(command) else None

    transport.answer = answer_display_commands
    sandbox = SpritesSandbox(desktop=True)
    agent = Agent(TestModel(call_tools=['probe']), capabilities=[sandbox])

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        size = await screen(ctx.workspace.backend).display_size()
        await screen(ctx.workspace.backend).act(Click(x=3, y=4))
        assert ctx.workspace.ref is not None and sandbox.backend(ctx.workspace.ref).display is not None
        # A plain command gets `DISPLAY` too, so a program the agent starts from the shell opens on the desktop.
        shell_display = (await ctx.workspace.run('printf %s "$DISPLAY"', shell=True)).stdout
        return f'{size[0]}x{size[1]} on {shell_display}'

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800 on :1"}'
    assert len(display.starts) == 1
    assert display.xdotool[-1].endswith('mousemove 3 4 click --repeat 1 1')
