"""Tests for `E2BSandbox(desktop=True)`, against the fake E2B SDK with a scripted X display."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, UserError
from pydantic_ai.computer_use import Click, ComputerReturnPart
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef
from pydantic_ai_harness.computer_use import ComputerUse
from pydantic_ai_harness.e2b_sandbox import E2BSandbox, E2BSandboxBackend

from .._fake_x_display import FakeDisplay
from ..display_fakes import FAKE_SCREENSHOT, screen
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


async def test_the_display_starts_on_first_use_and_actions_go_to_it(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    workspace = screen(E2BSandboxBackend(desktop=True))

    assert await workspace.display_size() == (1280, 800)
    await workspace.screenshot()  # an action waits for a screenshot of a display that just started
    await workspace.act(Click(x=3, y=4))

    assert len(display.starts) == 1
    assert display.xdotool[-1].endswith('mousemove 3 4 click --repeat 1 1')
    assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'DISPLAY': ':1'}


async def test_the_live_view_url_opens_the_viewer_with_its_password(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    backend = E2BSandboxBackend(desktop=True)

    url = await backend.live_view_url()

    sandbox_id = fake_e2b.sandboxes[0].sandbox_id
    assert url == f'https://6080-{sandbox_id}.e2b.example/vnc.html?autoconnect=true&resize=scale&password=Pass1234'


async def test_a_viewer_that_does_not_start_is_an_error(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    backend = E2BSandboxBackend(desktop=True)
    await screen(backend).display_size()
    display.up = False  # the start script's own failure, as the fake answers it

    with pytest.raises(WorkspaceError, match=r'Could not start the desktop viewer.*the viewer did not start'):
        await backend.live_view_url()


async def test_commands_see_the_desktop(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    await E2BSandboxBackend(desktop=True).run('true', shell=True)
    assert fake_e2b.sandboxes[0].commands.calls[-1].envs == {'DISPLAY': ':1'}


async def test_a_desktop_size_sets_the_display_it_starts(fake_e2b: FakeE2B, display: FakeDisplay) -> None:
    await screen(E2BSandboxBackend(desktop=(1440, 900))).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')


async def test_a_sandbox_without_a_desktop_has_no_display(fake_e2b: FakeE2B) -> None:
    backend = E2BSandboxBackend()

    assert backend.display is None
    with pytest.raises(UserError, match='has no display: create it with `desktop=True`'):
        await backend.live_view_url()
    assert fake_e2b.create_calls == []


def test_a_desktop_sandbox_reattaches_with_its_display() -> None:
    ref = WorkspaceRef(provider='e2b', id='sbx-desktop')

    assert E2BSandbox(desktop=True).backend(ref).display is not None
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
        capabilities=[E2BSandbox(desktop=True), ComputerUse(native=False, settle_seconds=0)],
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
