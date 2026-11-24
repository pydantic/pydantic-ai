"""Tests for `RenderSandbox(desktop=True)`, against the fake Render SDK with a scripted X display."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

import pytest

from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai import Agent
    from pydantic_ai.computer_use import ComputerReturnPart
    from pydantic_ai.exceptions import UserError
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.workspaces import CommandResult, WorkspaceCommand
    from pydantic_ai_harness.computer_use import ComputerUse
    from pydantic_ai_harness.render_sandbox import RenderSandbox, RenderSandboxBackend

    from ..computer_use._fake_x_display import FakeDisplay
    from ..computer_use.display_fakes import FAKE_SCREENSHOT, full_display, look_click_then_finish
    from .workspace_helpers import WorkspaceHarness

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='render not installed'),
    pytest.mark.skipif(os.name != 'posix', reason='needs a POSIX host'),
]


@pytest.fixture
def display(workspace_harness: WorkspaceHarness) -> FakeDisplay:
    """Answer the display's commands with a scripted X display; every other command runs as before."""
    display = FakeDisplay()
    run = workspace_harness.execute

    async def execute(
        sandbox_id: str, command: WorkspaceCommand, *, working_dir: str, env: dict[str, str]
    ) -> CommandResult:
        line = command if isinstance(command, str) else shlex.join(command)
        if not display.answers(line):
            return await run(sandbox_id, command, working_dir=working_dir, env=env)
        assert env['DISPLAY'] == ':1'
        stdout, stderr, exit_code = display(line, None)
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    workspace_harness.execute = execute
    return display


async def test_a_shell_command_finds_the_desktop_running(
    workspace_harness: WorkspaceHarness, display: FakeDisplay
) -> None:
    # A program the shell starts opens on the screen, so a command starts the display.
    backend = workspace_harness.backend(desktop=True, env={'FOO': 'bar'})

    result = await backend.run('echo "$DISPLAY $FOO"', shell=True)

    assert result.stdout == ':1 bar\n'
    assert len(display.starts) == 1


async def test_a_desktop_size_sets_the_display_it_starts(
    workspace_harness: WorkspaceHarness, display: FakeDisplay
) -> None:
    await full_display(workspace_harness.backend(desktop=(1440, 900))).display_size()

    (start,) = display.starts
    assert start.endswith(' sh 1440x900')


def test_a_new_desktop_sandbox_needs_internet_access_to_install_the_display_programs() -> None:
    with pytest.raises(UserError, match='needs `allow_internet_access=True`'):
        RenderSandboxBackend(desktop=True, allow_internet_access=False)


async def test_a_desktop_sandbox_reattaches_with_its_display(
    workspace_harness: WorkspaceHarness, display: FakeDisplay
) -> None:
    created = workspace_harness.backend()
    await created.working_dir()
    ref = created.ref
    assert ref is not None
    client, working_dir = workspace_harness.client, str(workspace_harness.root)

    # Attaching installs nothing it can't, so the network setting for new sandboxes doesn't stop it.
    desktop = RenderSandboxBackend(
        client=client, ref=ref, working_dir=working_dir, desktop=(1440, 900), allow_internet_access=False
    )
    await full_display(desktop).display_size()
    # The capability hands its desktop size to the backend it attaches.
    await full_display(
        RenderSandbox(client=client, working_dir=working_dir, desktop=(1440, 900)).backend(ref)
    ).display_size()

    assert [start.endswith(' sh 1440x900') for start in display.starts] == [True, True]
    assert RenderSandbox(client=client, working_dir=working_dir).backend(ref).display is None


async def test_a_desktop_capability_runs_the_computer_tool_on_a_new_sandbox(
    workspace_harness: WorkspaceHarness, display: FakeDisplay, tmp_path: Path
) -> None:
    agent = Agent(
        FunctionModel(look_click_then_finish),
        capabilities=[
            RenderSandbox(client=workspace_harness.client, working_dir=str(tmp_path), desktop=True),
            ComputerUse(native=False),
        ],
    )

    result = await agent.run('Click.')

    looked, clicked = [
        part for message in result.all_messages() for part in message.parts if isinstance(part, ComputerReturnPart)
    ]
    assert looked.result == {'performed': 1}
    assert clicked.result == {'performed': 1}
    assert clicked.screenshot is not None and clicked.screenshot.data == FAKE_SCREENSHOT
    assert workspace_harness.created == ['sb-1']
    (start,) = display.starts
    assert 'apt_get install' in start and 'fluxbox tint2 xterm' in start and 'setsid fluxbox' in start
    assert any(command.endswith('mousemove 3 4 click --repeat 1 1') for command in display.xdotool)
