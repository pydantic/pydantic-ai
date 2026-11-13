"""Tests for `E2BSandbox(desktop=True)`, against the fake E2B SDK with a scripted X display."""

from __future__ import annotations

import base64

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import WorkspaceRef
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

from ..display_fakes import FAKE_SCREENSHOT, screen
from .fake_e2b import FakeE2B


class FakeDisplay:
    """Answers the start script and `xdotool` like a sandbox whose X server is up or down."""

    def __init__(self) -> None:
        self.up = False
        self.starts: list[str] = []
        self.xdotool: list[str] = []

    def __call__(self, command: str, timeout: float | None) -> tuple[str, str, int]:
        if 'Xvfb' in command:
            self.starts.append(command)
            self.up = True
            return '', '', 0
        elif 'xwininfo -root' in command:
            # The backend reads the size only from a display it started.
            assert self.up, 'xwininfo before the display started'
            return '  Width: 1280\n  Height: 800\n', '', 0
        elif 'scrot' in command:
            return base64.b64encode(FAKE_SCREENSHOT).decode(), '', 0
        elif command.startswith('xdotool ') or ' xdotool ' in command:  # input runs `xdotool` through a script
            self.xdotool.append(command)
        return '', '', 0


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


async def test_commands_see_the_desktop_and_the_env(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    await E2BSandboxBackend(desktop=True, env={'FOO': 'bar'}).run('true', shell=True)

    assert fake_e2b.create_calls[0].envs == {'DISPLAY': ':1', 'FOO': 'bar'}
    assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'DISPLAY': ':1', 'FOO': 'bar'}


async def test_a_desktop_size_sets_the_display_it_starts(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    await screen(E2BSandboxBackend(desktop=(1440, 900))).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')


async def test_a_desktop_sandbox_reattaches_with_its_display(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    ref = WorkspaceRef(provider='e2b', id='sbx-desktop')

    await screen(E2BSandbox(desktop=(1440, 900)).backend(ref)).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert E2BSandbox().backend(ref).display is None


def _look_click_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Takes a screenshot, clicks on what it showed, then answers."""
    steps = [[{'type': 'screenshot'}], [{'type': 'click', 'x': 3, 'y': 4}]]
    done = sum(isinstance(part, ComputerReturnPart) for message in messages for part in message.parts)
    if done == len(steps):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('computer', {'actions': steps[done]})])


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_sandbox(
    fake_e2b: FakeE2B, display: FakeDisplay
) -> None:
    agent = Agent(
        FunctionModel(_look_click_then_finish),
        capabilities=[E2BSandbox(desktop=True), ComputerUse(native=False)],
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert clicked.result == {'performed': 1, 'screenshot_size': (1280, 800)}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert [call.template for call in fake_e2b.create_calls] == ['desktop']
    assert len(display.starts) == 1
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
