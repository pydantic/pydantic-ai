"""Tests for `DockerSandbox(desktop=True)` and `DockerDesktopBackend`, against a fake `docker` that runs commands locally."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import Click, Workspace, WorkspaceError, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.docker_sandbox import (
    DESKTOP_DOCKERFILE,
    DockerDesktopBackend,
    DockerSandbox,
    DockerSandboxBackend,
)
from pydantic_ai_harness.docker_sandbox._backend import _DESKTOP_IMAGE  # pyright: ignore[reportPrivateUsage]

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


async def test_the_desktop_image_is_built_once_and_runs_the_container(docker: FakeDocker, container_dir: Path) -> None:
    await DockerDesktopBackend(working_dir=str(container_dir)).working_dir()

    assert docker.images() == [_DESKTOP_IMAGE]
    assert (docker.bin_dir / 'images' / _DESKTOP_IMAGE).read_text() == DESKTOP_DOCKERFILE
    assert _run_image(docker) == _DESKTOP_IMAGE

    await DockerDesktopBackend(working_dir=str(container_dir)).working_dir()
    assert [call for call in docker.calls if call.startswith('build ')] == [
        next(call for call in docker.calls if call.startswith('build '))
    ]


async def test_a_custom_image_is_not_built(docker: FakeDocker, container_dir: Path) -> None:
    await DockerDesktopBackend(image='my-desktop', working_dir=str(container_dir)).working_dir()

    assert docker.images() == []
    assert _run_image(docker) == 'my-desktop'


async def test_the_display_starts_once_and_actions_go_to_it(docker: FakeDocker, container_dir: Path) -> None:
    workspace = Workspace(DockerDesktopBackend(image='my-desktop', working_dir=str(container_dir)))

    assert await workspace.display_size() == (1280, 800)
    await workspace.act(Click(x=3, y=4))

    assert docker.xdotool_calls == [
        ':1 getdisplaygeometry',  # the start script finds a display already running
        ':1 getdisplaygeometry',
        ':1 getdisplaygeometry',
        ':1 mousemove 3 4 click --repeat 1 1',
    ]


async def test_a_failed_build_says_why(docker: FakeDocker, container_dir: Path) -> None:
    docker.fail_builds()

    with pytest.raises(
        WorkspaceUnavailableError, match='could not build the desktop image: E: Unable to locate package'
    ):
        await DockerDesktopBackend(working_dir=str(container_dir)).working_dir()
    assert docker.images() == []


async def test_an_image_without_a_working_x_server_fails_at_once(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display()
    workspace = Workspace(DockerDesktopBackend(image='my-desktop', working_dir=str(container_dir)))

    with anyio.fail_after(5):
        with pytest.raises(WorkspaceError, match='the image needs Xvfb, xdotool and scrot') as error:
            await workspace.screenshot()
    assert 'the X server did not start' in str(error.value)


def test_only_a_desktop_sandbox_has_a_display(docker: FakeDocker, container_dir: Path) -> None:
    ref = WorkspaceRef(provider='docker', id='pydantic-ai-abc')
    desktop, plain = DockerSandbox(desktop=True), DockerSandbox(image='python:3.13-slim')

    assert isinstance(desktop.backend(ref), DockerDesktopBackend)
    assert type(plain.backend(ref)) is DockerSandboxBackend
    assert Workspace(desktop.backend(ref)).has_display is True
    assert Workspace(plain.backend(ref)).has_display is False


async def test_a_desktop_capability_gives_each_new_run_a_display(docker: FakeDocker, container_dir: Path) -> None:
    agent = Agent(
        TestModel(call_tools=['probe']), capabilities=[DockerSandbox(desktop=True, working_dir=str(container_dir))]
    )

    @agent.tool
    async def probe(ctx: RunContext[object]) -> str:
        return 'x'.join(map(str, await ctx.workspace.display_size()))

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800"}'
    assert docker.images() == [_DESKTOP_IMAGE]
