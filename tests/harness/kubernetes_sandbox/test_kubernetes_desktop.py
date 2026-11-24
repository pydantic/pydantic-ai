"""Tests for `KubernetesSandbox(desktop=True)`, against the fake cluster with a scripted X display."""

from __future__ import annotations

import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.computer_use import ComputerReturnPart
from pydantic_ai.exceptions import UserError
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import KubernetesSandbox, KubernetesSandboxBackend
from pydantic_ai_harness.computer_use import ComputerUse

from ..computer_use._fake_x_display import FakeDisplay
from ..computer_use.display_fakes import FAKE_SCREENSHOT, full_display, look_click_then_finish
from .fake_sandboxd import FakeCluster


@pytest.fixture
def display(cluster: FakeCluster) -> FakeDisplay:
    cluster.display = display = FakeDisplay()
    return display


async def test_a_shell_command_finds_the_desktop_running(cluster: FakeCluster, display: FakeDisplay) -> None:
    # A program the shell starts opens on the screen, so a command starts the display.
    backend = KubernetesSandboxBackend(warm_pool='pool', desktop=True, env={'FOO': 'bar'})

    result = await backend.run('echo "$DISPLAY $FOO"', shell=True)

    assert result.stdout == ':1 bar\n'
    assert len(display.starts) == 1
    await backend.aclose()


async def test_a_desktop_size_sets_the_display_it_starts(cluster: FakeCluster, display: FakeDisplay) -> None:
    backend = KubernetesSandboxBackend(warm_pool='pool', desktop=(1440, 900))

    await full_display(backend).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')
    assert 'fluxbox tint2 xterm' in start and 'setsid fluxbox' in start
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
    await full_display(reattached).display_size()
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


async def test_a_command_and_the_display_start_before_it_are_one_operation(
    cluster: FakeCluster, display: FakeDisplay
) -> None:
    # Outside a run each operation closes its connection when it ends. One connection shows the start and the command
    # are one operation, so an `aclose()` while the display starts waits for the command too.
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage(), run_id='run-1')
    backend = KubernetesSandbox(warm_pool='pool', desktop=True).get_workspace(ctx, ref=None)
    assert isinstance(backend, KubernetesSandboxBackend)

    await backend.run(['true'])

    assert len(display.starts) == 1
    assert [client.closed for client in cluster.clients] == [True]
