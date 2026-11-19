"""Tests for `KubernetesSandbox(desktop=True)`, against the fake cluster with a scripted X display."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.workspaces import WorkspaceUnavailableError
from pydantic_ai_harness import KubernetesSandbox, KubernetesSandboxBackend
from pydantic_ai_harness.computer_use import ComputerUse

from .._fake_x_display import FakeDisplay
from ..display_fakes import FAKE_SCREENSHOT, look_click_then_finish, screen
from .fake_sandboxd import FakeCluster


@pytest.fixture
def display(cluster: FakeCluster) -> FakeDisplay:
    cluster.display = display = FakeDisplay()
    return display


async def test_a_shell_command_finds_the_desktop_running(cluster: FakeCluster, display: FakeDisplay) -> None:
    # A program the agent starts from the shell opens on the screen, before anything has looked at the screen.
    backend = KubernetesSandboxBackend(warm_pool='pool', desktop=True, env={'FOO': 'bar'})

    result = await backend.run('echo "$DISPLAY $FOO"', shell=True)

    assert result.stdout == ':1 bar\n'
    assert len(display.starts) == 1
    await backend.aclose()


async def test_a_desktop_size_sets_the_display_it_starts(cluster: FakeCluster, display: FakeDisplay) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool', desktop=(1440, 900))

    await screen(backend).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    await backend.aclose()


async def test_an_image_without_the_display_programs_is_refused_with_what_it_needs(cluster: FakeCluster) -> None:
    # Pods run as a non-root user, so the start script cannot install the programs and says what is missing.
    def refuse(command: str, timeout: float | None) -> tuple[str, str, int]:
        return '', "the display needs Xvfb xdotool, and there's no root or passwordless sudo to install them with", 1

    cluster.display = refuse
    backend = KubernetesSandboxBackend(warm_pool='pool', desktop=True)

    with pytest.raises(
        WorkspaceUnavailableError,
        match=r'^Could not start the display in Kubernetes sandbox default/\S+: the display needs Xvfb xdotool, and '
        r"there's no root or passwordless sudo to install them with$",
    ):
        await screen(backend).display_size()
    await backend.aclose()


def test_a_bad_desktop_size_fails_where_it_is_written() -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        KubernetesSandbox(warm_pool='desktop-pool', desktop=(0, 800))


async def test_a_desktop_sandbox_reattaches_with_its_display(cluster: FakeCluster, display: FakeDisplay) -> None:
    first = KubernetesSandboxBackend(warm_pool='pool')
    await first.working_dir()
    ref = first.ref
    assert ref is not None
    await first.aclose()

    reattached = KubernetesSandbox(warm_pool='pool', desktop=(1440, 900)).backend(ref)
    await screen(reattached).display_size()
    await reattached.aclose()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert KubernetesSandbox(warm_pool='pool').backend(ref).display is None


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_claim(
    cluster: FakeCluster, display: FakeDisplay
) -> None:
    agent = Agent(
        FunctionModel(look_click_then_finish),
        capabilities=[KubernetesSandbox(warm_pool='pool', desktop=True), ComputerUse(native=False)],
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert len(cluster.created) == 1
    assert len(display.starts) == 1
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
