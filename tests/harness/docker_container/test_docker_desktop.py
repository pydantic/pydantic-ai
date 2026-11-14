"""Tests for `DockerContainer(desktop=True)`, against a fake `docker` that runs commands locally."""

from __future__ import annotations

import re
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.computer_use import Click, Point
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceError, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.computer_use import (
    Display,
    SupportsCursorPosition,
    SupportsDisplay,
)
from pydantic_ai_harness.docker_container import DESKTOP_DOCKERFILE, DockerContainer, DockerContainerBackend

from ..display_fakes import FAKE_SCREENSHOT
from ._fake_docker import FakeDocker, install_fake_docker


@pytest.fixture
def docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    return install_fake_docker(tmp_path, monkeypatch)


@pytest.fixture
def container_dir(tmp_path: Path) -> Path:
    path = tmp_path / 'container'
    path.mkdir()
    return path


def _run_image(docker: FakeDocker) -> str:
    (run,) = [call for call in docker.calls if call.startswith('run ')]
    return run.split(' -- ')[1].split(' ')[0]


def _desktop(container_dir: Path, image: str | None = None) -> DockerContainerBackend:
    return DockerContainerBackend(image=image, working_dir=str(container_dir), desktop=True)


def _screen(backend: WorkspaceBackend) -> Display:
    assert isinstance(backend, SupportsDisplay)
    display = backend.display
    assert display is not None
    return display


async def test_the_desktop_image_is_built_once_and_runs_the_container(docker: FakeDocker, container_dir: Path) -> None:
    await _desktop(container_dir).working_dir()

    # Tagged by the Dockerfile's hash, so a changed Dockerfile builds a new image.
    (image,) = docker.images()
    assert re.fullmatch(r'pydantic-ai-desktop:[0-9a-f]{12}', image)
    assert (docker.bin_dir / 'images' / image).read_text() == DESKTOP_DOCKERFILE
    assert _run_image(docker) == image

    await _desktop(container_dir).working_dir()
    assert len([call for call in docker.calls if call.startswith('build ')]) == 1


async def test_a_custom_image_is_not_built(docker: FakeDocker, container_dir: Path) -> None:
    await _desktop(container_dir, image='my-desktop').working_dir()

    assert docker.images() == []
    assert _run_image(docker) == 'my-desktop'


async def test_a_failed_build_says_why(docker: FakeDocker, container_dir: Path) -> None:
    docker.fail_builds()

    with pytest.raises(
        WorkspaceUnavailableError, match='could not build the desktop image: E: Unable to locate package'
    ):
        await _desktop(container_dir).working_dir()
    assert docker.images() == []


async def test_a_cancelled_build_removes_its_context(docker: FakeDocker, container_dir: Path) -> None:
    docker.hold_builds()
    try:
        async with anyio.create_task_group() as tg:
            tg.start_soon(_desktop(container_dir).working_dir)
            # The fake build records its context once it has started, and is cancelled then.
            with anyio.fail_after(10):
                while docker.build_context is None:
                    await anyio.sleep(0.05)
            tg.cancel_scope.cancel()
    finally:
        docker.release_builds()
    context = docker.build_context
    assert context is not None
    assert not context.exists()


@pytest.mark.parametrize(
    'desktop, size',
    [(True, (1280, 800)), ((1440, 900), (1440, 900))],
    ids=['default', 'sized'],
)
async def test_the_display_starts_on_first_use_and_actions_go_to_it(
    docker: FakeDocker, container_dir: Path, desktop: bool | tuple[int, int], size: tuple[int, int]
) -> None:
    docker.remove_display()
    backend = DockerContainerBackend(image='my-desktop', working_dir=str(container_dir), desktop=desktop)
    workspace = _screen(backend)

    assert await workspace.display_size() == size
    await workspace.act(Click(x=3, y=4))

    assert docker.xvfb_starts == 1
    # First letting go of modifiers an earlier action may have left down.
    assert docker.xdotool_calls[-1] == (
        ':1 keyup Shift_L Shift_R Control_L Control_R Alt_L Alt_R Super_L Super_R ISO_Level3_Shift '
        'mousemove 3 4 click --repeat 1 1'
    )


async def test_the_display_waits_for_the_window_manager(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display()
    # `xprop` reports no window manager the first time it is asked, and one the next.
    bin_dir = docker.bin_dir
    (bin_dir / 'xprop').write_text(
        f'#!/bin/sh\necho probe >> {bin_dir}/xprop-probes\n'
        f'[ "$(wc -l < {bin_dir}/xprop-probes)" -ge 2 ] && echo "_NET_SUPPORTING_WM_CHECK(WINDOW): window id # 0x1"\n'
    )
    (bin_dir / 'xprop').chmod(0o755)
    # A window manager keeps running, so the wait doesn't end early: this one runs until it has been seen.
    (bin_dir / 'fluxbox').write_text(
        f'#!/bin/sh\ni=0; until [ "$(cat {bin_dir}/xprop-probes 2>/dev/null | wc -l)" -ge 2 ] || [ $i -ge 100 ]; '
        'do sleep 0.1; i=$((i+1)); done\n'
    )

    await _screen(_desktop(container_dir, image='my-desktop')).display_size()
    assert (bin_dir / 'xprop-probes').read_text().splitlines() == ['probe', 'probe']


async def test_screenshots_and_the_pointer_come_from_the_display(docker: FakeDocker, container_dir: Path) -> None:
    workspace = _screen(_desktop(container_dir, image='my-desktop'))
    assert isinstance(workspace, SupportsCursorPosition)

    assert await workspace.screenshot() == FAKE_SCREENSHOT
    assert await workspace.cursor_position() == Point(x=640, y=400)


async def test_commands_see_the_desktop(docker: FakeDocker, container_dir: Path) -> None:
    result = await _desktop(container_dir, image='my-desktop').run('echo "$DISPLAY"', shell=True)
    assert result.stdout == ':1\n'


async def test_a_display_in_env_does_not_move_the_desktop(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerContainerBackend(
        image='my-desktop', working_dir=str(container_dir), desktop=True, env={'DISPLAY': ':0'}
    )
    await _screen(backend).act(Click(x=3, y=4))

    # Starting the display checks the same X server the actions go to.
    assert {call.split(' ')[0] for call in docker.xdotool_calls} == {':1'}


async def test_a_display_that_went_away_is_started_again(docker: FakeDocker, container_dir: Path) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    workspace = _screen(backend)
    await workspace.display_size()
    assert docker.xvfb_starts == 0  # the display was already up

    docker.remove_display()
    failure = "Display command failed with exit code 1 on display :1: Can't open display: :1"
    with pytest.raises(WorkspaceError) as error:
        await workspace.act(Click(x=1, y=1))
    # Letting go of the button fails on the missing display too.
    assert str(error.value) == (
        f'{failure}. Releasing what it pressed failed too, so a mouse button or key may still be held: {failure}'
    )
    # The next action starts the display again and runs on it.
    await workspace.act(Click(x=1, y=1))

    assert docker.xvfb_starts == 1


async def test_a_lock_left_by_a_stopped_container_does_not_stop_the_display(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.remove_display()
    docker.leave_x_lock(backend.ref.id)

    assert await _screen(backend).display_size() == (1280, 800)
    assert not docker.x_lock_exists(backend.ref.id)


async def test_a_running_x_server_keeps_its_lock(docker: FakeDocker, container_dir: Path) -> None:
    # An image's own X server that refuses the probe, such as a VNC server with authentication on.
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.remove_display()
    docker.leave_x_lock(backend.ref.id, owner='Xvnc')

    with pytest.raises(WorkspaceUnavailableError, match='Server is already active'):
        await _screen(backend).display_size()
    assert docker.x_lock_exists(backend.ref.id)


async def _panel(docker: FakeDocker, backend: DockerContainerBackend) -> Path:
    """Where the start records the panel of a display it started: empty until the panel runs, then its process ID."""
    await backend.working_dir()
    assert backend.ref is not None
    return docker.bin_dir / 'containers' / f'{backend.ref.id}.tmp' / 'panel'


async def test_a_display_we_start_gets_its_panel(docker: FakeDocker, container_dir: Path) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    panel = await _panel(docker, backend)
    docker.remove_display()

    await _screen(backend).display_size()
    assert panel.read_text().strip().isdigit()


async def test_a_start_cancelled_before_the_panel_leaves_it_to_the_next(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    panel = await _panel(docker, backend)
    # As a cancelled start leaves it: the X server runs, and the panel was never started.
    panel.touch()

    await _screen(backend).display_size()
    assert panel.read_text().strip().isdigit()


async def test_an_images_own_display_gets_no_panel(docker: FakeDocker, container_dir: Path) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    panel = await _panel(docker, backend)

    await _screen(backend).display_size()
    assert not panel.exists()


async def test_a_missing_display_program_ends_the_run(docker: FakeDocker, container_dir: Path) -> None:
    # Missing as the shell reports it, not unlinked: a host's own `scrot` further along PATH would answer.
    (docker.bin_dir / 'scrot').write_text('#!/bin/sh\necho "scrot: not found" >&2\nexit 127\n')
    workspace = _screen(_desktop(container_dir, image='my-desktop'))

    with pytest.raises(WorkspaceUnavailableError, match='A display command is missing'):
        await workspace.screenshot()


async def test_an_image_without_a_working_x_server_fails_at_once(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display(can_start=False)
    workspace = _screen(_desktop(container_dir, image='my-desktop'))

    with pytest.raises(WorkspaceUnavailableError, match='it needs Xvfb, xdotool, xwininfo, scrot and setsid') as error:
        await workspace.screenshot()
    assert 'the X server did not start' in str(error.value)
    assert 'Fatal server error' in str(error.value)


def test_a_desktop_needs_no_image() -> None:
    assert DockerContainerBackend(desktop=True).display is not None


def test_a_desktop_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match='must be a positive width and height'):
        DockerContainerBackend(desktop=(0, 0))
    with pytest.raises(ValueError, match='must be a positive width and height'):
        DockerContainer(desktop=(0, 0))


def test_a_desktop_sandbox_reattaches_with_its_display(docker: FakeDocker) -> None:
    ref = WorkspaceRef(provider='docker', id='pydantic-ai-abc')

    assert DockerContainer(desktop=True).backend(ref).display is not None
    assert DockerContainer(image='python:3.13-slim').backend(ref).display is None


async def test_a_desktop_capability_gives_a_new_run_a_display(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display()
    agent = Agent(
        TestModel(call_tools=['probe']), capabilities=[DockerContainer(desktop=True, working_dir=str(container_dir))]
    )

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        return 'x'.join(map(str, await _screen(ctx.workspace.backend).display_size()))

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800"}'  # the size the display was started at
    assert [image.split(':')[0] for image in docker.images()] == ['pydantic-ai-desktop']
