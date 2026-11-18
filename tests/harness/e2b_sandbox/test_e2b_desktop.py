"""Tests for `E2BSandbox(desktop=True)`, against the fake E2B SDK with a scripted X display."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.workspaces import WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

from .._fake_x_display import FakeDisplay
from ..display_fakes import FAKE_SCREENSHOT, look_click_then_finish, screen
from .fake_e2b import FakeE2B


@pytest.fixture
def display(fake_e2b: FakeE2B) -> FakeDisplay:
    fake_e2b.responder = display = FakeDisplay()
    return display


@pytest.mark.parametrize(('template', 'created'), [(None, 'desktop'), ('my-desktop', 'my-desktop')])
async def test_a_desktop_sandbox_runs_the_desktop_template_unless_given_one(
    fake_e2b: FakeE2B, display: FakeDisplay, template: str | None, created: str
) -> None:
    await E2BSandboxBackend(template=template, desktop=True).get_sandbox()

    assert [call.template for call in fake_e2b.create_calls] == [created]


async def test_a_shell_command_finds_the_desktop_running(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    # The shell is told a program it starts opens on the screen, before anything has looked at the screen.
    await E2BSandboxBackend(desktop=True, env={'FOO': 'bar'}).run('xterm &', shell=True)

    assert len(display.starts) == 1
    assert fake_e2b.create_calls[0].envs == {'DISPLAY': ':1', 'FOO': 'bar'}
    assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'DISPLAY': ':1', 'FOO': 'bar'}


async def test_a_desktop_size_sets_the_display_it_starts(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    await screen(E2BSandboxBackend(desktop=(1440, 900))).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')


async def test_a_display_that_does_not_start_names_the_sandbox_and_what_it_needs(fake_e2b: FakeE2B) -> None:
    def responder(command: str, timeout: float | None) -> tuple[str, str, int]:
        return ('', 'Xvfb: not found', 1) if 'Xvfb' in command else ('', '', 0)

    fake_e2b.responder = responder

    with pytest.raises(
        WorkspaceUnavailableError,
        match=r'^Could not start the display in E2B sandbox sbx-1: it needs Xvfb, xdotool, xwininfo, scrot and setsid '
        r'\(Xvfb: not found\)$',
    ):
        await screen(E2BSandboxBackend(desktop=True)).display_size()


async def test_a_desktop_sandbox_reattaches_with_its_display(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    ref = WorkspaceRef(provider='e2b', id='sbx-desktop')

    await screen(E2BSandbox(desktop=(1440, 900)).backend(ref)).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert E2BSandbox().backend(ref).display is None


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_sandbox(
    fake_e2b: FakeE2B, display: FakeDisplay
) -> None:
    agent = Agent(
        FunctionModel(look_click_then_finish),
        capabilities=[E2BSandbox(desktop=True), ComputerUse(native=False)],
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert [call.template for call in fake_e2b.create_calls] == ['desktop']
    assert len(display.starts) == 1
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
