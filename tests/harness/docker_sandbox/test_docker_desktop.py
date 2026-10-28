"""Tests for `DockerSandbox(desktop=True)`, against a fake `docker` that runs commands locally."""

from __future__ import annotations

import re
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import (
    Click,
    Point,
    PressKeys,
    Region,
    Workspace,
    WorkspaceError,
    WorkspaceRef,
    WorkspaceUnavailableError,
    WorkspaceUnsupportedActionError,
)
from pydantic_ai_harness.docker_sandbox import DESKTOP_DOCKERFILE, DockerSandbox, DockerSandboxBackend

from ...workspace_fakes import FAKE_SCREENSHOT
from ._fake_docker import FakeDocker, install_fake_docker

pytestmark = pytest.mark.anyio


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


def _desktop(container_dir: Path, image: str | None = None) -> DockerSandboxBackend:
    return DockerSandboxBackend(image=image, working_dir=str(container_dir), desktop=True)


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


async def test_the_display_starts_on_first_use_and_actions_go_to_it(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display()
    workspace = Workspace(_desktop(container_dir, image='my-desktop'))

    assert await workspace.display_size() == (1280, 800)
    await workspace.act(Click(x=3, y=4))

    assert docker.xvfb_starts == 1
    assert docker.xdotool_calls[-2:] == [':1 getdisplaygeometry', ':1 mousemove 3 4 click --repeat 1 1']


async def test_screenshots_zoom_and_the_pointer_come_from_the_display(docker: FakeDocker, container_dir: Path) -> None:
    workspace = Workspace(_desktop(container_dir, image='my-desktop'))

    assert await workspace.screenshot() == FAKE_SCREENSHOT
    assert await workspace.zoom(Region(left=0, top=0, right=10, bottom=10)) == FAKE_SCREENSHOT
    assert await workspace.cursor_position() == Point(x=640, y=400)


async def test_commands_see_the_desktop(docker: FakeDocker, container_dir: Path) -> None:
    result = await _desktop(container_dir, image='my-desktop').run('echo "$DISPLAY"', shell=True)
    assert result.stdout == ':1\n'


async def test_a_desktop_size_sets_the_display_it_starts(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display()
    backend = DockerSandboxBackend(image='my-desktop', working_dir=str(container_dir), desktop=(1440, 900))
    await Workspace(backend).display_size()

    (start,) = (docker.bin_dir / 'xvfb-starts').read_text().splitlines()
    assert '-screen 0 1440x900x24' in start


async def test_concurrent_first_uses_start_one_display(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display()
    workspace = Workspace(_desktop(container_dir, image='my-desktop'))

    async with anyio.create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(workspace.display_size)

    # Without the lock all three would run the start script; the fake's display would still start once.
    assert len([call for call in docker.calls if 'Xvfb' in call]) == 1


async def test_a_display_that_went_away_is_started_again(docker: FakeDocker, container_dir: Path) -> None:
    workspace = Workspace(_desktop(container_dir, image='my-desktop'))
    await workspace.display_size()
    assert docker.xvfb_starts == 0  # the display was already up

    docker.remove_display()
    with pytest.raises(WorkspaceError, match="Can't open display"):
        await workspace.act(Click(x=1, y=1))
    await workspace.act(Click(x=1, y=1))

    assert docker.xvfb_starts == 1


async def test_a_refused_key_does_not_restart_the_display(docker: FakeDocker, container_dir: Path) -> None:
    workspace = Workspace(_desktop(container_dir))
    await workspace.display_size()
    checks = docker.xdotool_calls.count(':1 getdisplaygeometry')

    with pytest.raises(WorkspaceUnsupportedActionError, match="cannot press the key '!'"):
        await workspace.act(PressKeys(keys=('!',)))
    await workspace.act(Click(x=1, y=1))

    # The start script, which first checks the display with `getdisplaygeometry`, did not run again.
    assert docker.xdotool_calls.count(':1 getdisplaygeometry') == checks


async def test_an_image_without_a_working_x_server_fails_at_once(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display(can_start=False)
    workspace = Workspace(_desktop(container_dir, image='my-desktop'))

    with pytest.raises(WorkspaceError, match='it needs Xvfb, xdotool, xmodmap, scrot and setsid') as error:
        await workspace.screenshot()
    assert 'the X server did not start' in str(error.value)
    assert 'Fatal server error' in str(error.value)
    # It stopped once the X server exited, before the 100 polls that give up on one that never answers.
    assert len(docker.xdotool_calls) < 100


async def test_a_container_without_a_desktop_has_no_display(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerSandboxBackend(image='python:3.13-slim', working_dir=str(container_dir))

    assert Workspace(backend).has_display is False
    with pytest.raises(UserError, match='has no display: create it with `desktop=True`'):
        await backend.screenshot()
    assert docker.calls == []


def test_a_desktop_needs_no_image_and_still_refuses_both_image_and_ref() -> None:
    ref = WorkspaceRef(provider='docker', id='pydantic-ai-abc')
    assert DockerSandboxBackend(desktop=True).display is not None
    with pytest.raises(ValueError, match='pass exactly one of `image`'):
        DockerSandboxBackend(image='x', ref=ref, desktop=True)
    with pytest.raises(ValueError, match='pass exactly one of `image`'):
        DockerSandboxBackend()


def test_a_desktop_sandbox_reattaches_with_its_display(docker: FakeDocker) -> None:
    ref = WorkspaceRef(provider='docker', id='pydantic-ai-abc')

    assert Workspace(DockerSandbox(desktop=True).backend(ref)).has_display is True
    assert Workspace(DockerSandbox(image='python:3.13-slim').backend(ref)).has_display is False


async def test_a_desktop_capability_gives_a_new_run_a_display(docker: FakeDocker, container_dir: Path) -> None:
    agent = Agent(
        TestModel(call_tools=['probe']), capabilities=[DockerSandbox(desktop=True, working_dir=str(container_dir))]
    )

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        return 'x'.join(map(str, await ctx.workspace.display_size()))

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800"}'
    assert [image.split(':')[0] for image in docker.images()] == ['pydantic-ai-desktop']
