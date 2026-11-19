"""Tests for `DockerContainer(desktop=True)`, against a fake `docker` that runs commands locally."""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

import anyio
import pytest

from pydantic_ai import Agent, RunContext, UserError
from pydantic_ai.computer_use import Click, Point
from pydantic_ai.models.test import TestModel
from pydantic_ai.workspaces import WorkspaceError, WorkspaceRef, WorkspaceUnavailableError
from pydantic_ai_harness.docker_container import DESKTOP_DOCKERFILE, DockerContainer, DockerContainerBackend

from ..display_fakes import FAKE_SCREENSHOT, screen
from ._fake_docker import FakeDocker, install_fake_docker

pytestmark = pytest.mark.skipif(
    os.name != 'posix', reason='`DockerContainerBackend` runs `docker` as a POSIX subprocess'
)


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


async def test_the_desktop_image_is_built_once_and_runs_the_container(docker: FakeDocker, container_dir: Path) -> None:
    await _desktop(container_dir).working_dir()

    # Tagged by the Dockerfile's hash, so a changed Dockerfile builds a new image.
    (image,) = docker.images()
    assert image == f'pydantic-ai-desktop:{hashlib.sha256(DESKTOP_DOCKERFILE.encode()).hexdigest()[:12]}'
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
    workspace = screen(backend)

    assert await workspace.display_size() == size
    await workspace.act(Click(x=3, y=4))

    assert docker.xvfb_starts == 1
    # Stays up when its last client exits, and listens on neither TCP nor the abstract socket the host shares.
    xvfb_args = (docker.bin_dir / 'xvfb-starts').read_text()
    assert '-nolisten tcp -nolisten local' in xvfb_args and '-noreset' in xvfb_args
    assert docker.xdotool_calls[-1] == ':1 mousemove 3 4 click --repeat 1 1'


async def test_two_starts_at_once_start_one_x_server(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display()
    first = _desktop(container_dir, image='my-desktop')
    await first.working_dir()
    assert first.ref is not None
    second = DockerContainerBackend(ref=first.ref, working_dir=str(container_dir), desktop=True)

    async with anyio.create_task_group() as tg:
        for backend in (first, second):
            tg.start_soon(screen(backend).display_size)

    assert docker.xvfb_starts == 1


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

    await screen(_desktop(container_dir, image='my-desktop')).display_size()
    assert (bin_dir / 'xprop-probes').read_text().splitlines() == ['probe', 'probe']


async def test_screenshots_and_the_pointer_come_from_the_display(docker: FakeDocker, container_dir: Path) -> None:
    workspace = screen(_desktop(container_dir, image='my-desktop'))

    assert await workspace.screenshot() == FAKE_SCREENSHOT
    assert await workspace.cursor_position() == Point(x=640, y=400)


async def test_a_shell_command_finds_the_desktop_running(docker: FakeDocker, container_dir: Path) -> None:
    # The shell is told a program it starts opens on the screen, before anything has looked at the screen.
    docker.remove_display()
    result = await _desktop(container_dir, image='my-desktop').run(
        'echo "$DISPLAY"; xdotool getdisplaygeometry', shell=True
    )

    assert (result.exit_code, result.stdout) == (0, ':1\n1280 800\n')
    assert docker.xvfb_starts == 1


async def test_a_command_env_with_an_invalid_name_is_refused_before_docker_runs(
    docker: FakeDocker, container_dir: Path
) -> None:
    with pytest.raises(ValueError, match="invalid environment variable name: 'A-B'"):
        await _desktop(container_dir, image='my-desktop').run(['true'], env={'A-B': '1'})

    assert docker.calls == []


async def test_a_display_in_env_does_not_move_the_desktop(docker: FakeDocker, container_dir: Path) -> None:
    backend = DockerContainerBackend(
        image='my-desktop', working_dir=str(container_dir), desktop=True, env={'DISPLAY': ':0'}
    )
    await screen(backend).act(Click(x=3, y=4))

    # Starting the display checks the same X server the actions go to.
    assert {call.split(' ')[0] for call in docker.xdotool_calls} == {':1'}


async def test_a_display_that_went_away_is_started_again(docker: FakeDocker, container_dir: Path) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    workspace = screen(backend)
    await workspace.display_size()
    assert docker.xvfb_starts == 1

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

    assert docker.xvfb_starts == 2


async def test_a_lock_left_by_a_stopped_container_does_not_stop_the_display(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.remove_display()
    docker.leave_x_lock(backend.ref.id)

    # The fake Xvfb refuses to start while a lock exists, so the display coming up shows the stale one went.
    assert await screen(backend).display_size() == (1280, 800)


async def test_a_running_x_server_keeps_its_lock(docker: FakeDocker, container_dir: Path) -> None:
    # An image's own X server, such as a VNC server, is used rather than replaced.
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.leave_x_lock(backend.ref.id, owner='Xvnc')

    assert await screen(backend).display_size() == (1280, 800)
    assert docker.xvfb_starts == 0
    assert docker.x_lock_exists(backend.ref.id)


async def test_the_start_opens_the_panel_unless_it_shows(docker: FakeDocker, container_dir: Path) -> None:
    await screen(_desktop(container_dir, image='my-desktop')).display_size()
    assert docker.panel_starts == 1

    await screen(_desktop(container_dir, image='my-desktop')).display_size()
    assert docker.panel_starts == 1

    # As a panel that died, or a start cancelled before it, leaves the display.
    docker.close_panel()
    await screen(_desktop(container_dir, image='my-desktop')).display_size()
    assert docker.panel_starts == 2


async def test_a_panel_that_exits_leaves_a_working_display(docker: FakeDocker, container_dir: Path) -> None:
    # As tint2 does when it can't read its settings, such as under a home that doesn't exist.
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.fail_panel()

    assert await screen(backend).screenshot() == FAKE_SCREENSHOT
    assert docker.panel_starts == 1
    log = docker.bin_dir / 'containers' / f'{backend.ref.id}.tmp' / 'pydantic-ai-desktop.log'
    assert 'tint2: could not open the config' in log.read_text()


async def test_a_missing_display_program_ends_the_run(docker: FakeDocker, container_dir: Path) -> None:
    # Missing as the shell reports it, not unlinked: a host's own `scrot` further along PATH would answer.
    (docker.bin_dir / 'scrot').write_text('#!/bin/sh\necho "scrot: not found" >&2\nexit 127\n')
    workspace = screen(_desktop(container_dir, image='my-desktop'))

    with pytest.raises(WorkspaceUnavailableError, match='A display command is missing'):
        await workspace.screenshot()


async def test_an_image_without_a_working_x_server_fails_at_once(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display(can_start=False)
    workspace = screen(_desktop(container_dir, image='my-desktop'))

    with pytest.raises(WorkspaceUnavailableError, match='the X server did not start') as error:
        await workspace.screenshot()
    assert 'Fatal server error' in str(error.value)


_INSTALL = 'install -y -qq --no-install-recommends xvfb xdotool x11-utils scrot util-linux'

# The fake docker hides a program only from its own bin directory, so a host that has it would skip the install.
_host_without_display_programs = pytest.mark.skipif(
    bool(shutil.which('xdotool') or shutil.which('scrot')), reason='the host has xdotool or scrot on PATH'
)


@_host_without_display_programs
@pytest.mark.parametrize(('uid', 'sudo'), [(0, ''), (1000, 'sudo ')])
async def test_missing_display_programs_are_installed_on_first_use(
    docker: FakeDocker, container_dir: Path, uid: int, sudo: str
) -> None:
    # A user other than root installs with passwordless sudo; root installs without it.
    docker.remove_display()
    docker.uninstall('xdotool', 'scrot')
    docker.run_as(uid, sudo=uid != 0)

    assert await screen(_desktop(container_dir, image='my-desktop')).display_size() == (1280, 800)
    assert docker.apt_get_calls == [f'{sudo}update -qq', f'{sudo}{_INSTALL}']


@_host_without_display_programs
async def test_a_program_missing_beside_a_running_display_is_installed(docker: FakeDocker, container_dir: Path) -> None:
    docker.uninstall('scrot')
    docker.run_as(0)

    assert await screen(_desktop(container_dir, image='my-desktop')).screenshot() == FAKE_SCREENSHOT
    assert docker.apt_get_calls == ['update -qq', _INSTALL]


@_host_without_display_programs
async def test_missing_display_programs_without_root_say_what_is_missing(
    docker: FakeDocker, container_dir: Path
) -> None:
    docker.remove_display()
    docker.uninstall('xdotool', 'scrot')

    with pytest.raises(
        WorkspaceUnavailableError, match="the display needs xdotool scrot, and there's no root or passwordless sudo"
    ):
        await screen(_desktop(container_dir, image='my-desktop')).screenshot()
    assert docker.apt_get_calls == []


async def test_missing_display_programs_without_apt_get_say_what_is_missing(
    docker: FakeDocker, container_dir: Path
) -> None:
    # As in an image that isn't Debian's, such as Alpine.
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    docker.remove_display()
    docker.uninstall('xdotool', 'apt-get')
    docker.hide_host_programs('cat', 'tr', 'dirname', *(['flock'] if shutil.which('flock') else []))
    docker.run_as(0)

    with pytest.raises(
        WorkspaceUnavailableError, match="the display needs xdotool, and there's no apt-get to install them with"
    ):
        await screen(backend).screenshot()


_HOST_TOOLS = ('base64', 'cat', 'cp', 'dirname', 'env', 'grep', 'mkdir', 'mv', 'rm', 'sleep', 'tail', 'touch', 'tr')
"""The host programs the start script and the fakes use, for `hide_host_programs`."""


def _hide_host_programs(docker: FakeDocker) -> None:
    docker.hide_host_programs(*_HOST_TOOLS, *(['flock'] if shutil.which('flock') else []))


async def test_a_running_x_server_needs_no_xvfb(docker: FakeDocker, container_dir: Path) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.leave_x_lock(backend.ref.id, owner='Xvnc')
    docker.uninstall('Xvfb')
    _hide_host_programs(docker)

    assert await screen(backend).display_size() == (1280, 800)
    assert docker.apt_get_calls == []


async def test_an_x_server_that_starts_during_the_install_is_kept(docker: FakeDocker, container_dir: Path) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.remove_display()
    docker.uninstall('xdotool')
    _hide_host_programs(docker)
    docker.run_as(0)
    docker.start_x_during_install()

    assert await screen(backend).display_size() == (1280, 800)
    assert docker.apt_get_calls == ['update -qq', _INSTALL]
    assert docker.xvfb_starts == 0
    assert docker.x_lock_exists(backend.ref.id)


@pytest.mark.parametrize('kind', ['directory', 'named-pipe'])
async def test_a_desktop_log_that_isnt_a_file_still_starts_the_desktop(
    docker: FakeDocker, container_dir: Path, kind: str
) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    # Something other than a file where the log goes. Opening a named pipe would wait forever for a reader.
    log = docker.bin_dir / 'containers' / f'{backend.ref.id}.tmp' / 'pydantic-ai-desktop.log'
    if kind == 'directory':
        log.mkdir()
    else:
        os.mkfifo(log)

    with anyio.fail_after(10):
        assert await screen(backend).screenshot() == FAKE_SCREENSHOT
    assert docker.panel_starts == 1


@_host_without_display_programs
async def test_a_desktop_program_the_package_lists_lack_leaves_the_display_working(
    docker: FakeDocker, container_dir: Path
) -> None:
    # The display's own programs are installed on their own, so the missing panel can't stop them.
    docker.remove_display()
    docker.uninstall('xdotool', 'tint2')
    _hide_host_programs(docker)
    docker.run_as(0)
    docker.lack_packages('tint2')

    assert await screen(_desktop(container_dir, image='my-desktop')).screenshot() == FAKE_SCREENSHOT
    assert docker.apt_get_calls == ['update -qq', _INSTALL, 'install -y -qq --no-install-recommends tint2']
    assert docker.panel_starts == 0


@_host_without_display_programs
async def test_a_running_x_server_gets_no_xvfb_installed(docker: FakeDocker, container_dir: Path) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.leave_x_lock(backend.ref.id, owner='Xvnc')
    docker.uninstall('scrot')
    docker.run_as(0)

    assert await screen(backend).screenshot() == FAKE_SCREENSHOT
    assert docker.apt_get_calls == [
        'update -qq',
        'install -y -qq --no-install-recommends xdotool x11-utils scrot util-linux',
    ]


async def test_without_root_a_missing_desktop_is_skipped(docker: FakeDocker, container_dir: Path) -> None:
    # An image with the display programs but no window manager, panel or terminal, and no way to install them.
    docker.uninstall('fluxbox', 'tint2', 'xterm')
    _hide_host_programs(docker)

    assert await screen(_desktop(container_dir, image='my-desktop')).screenshot() == FAKE_SCREENSHOT
    assert docker.apt_get_calls == []
    assert docker.panel_starts == 0


async def test_the_desktop_settings_are_written_for_the_user_unless_present(
    docker: FakeDocker, container_dir: Path
) -> None:
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    home = docker.bin_dir / 'containers' / f'{backend.ref.id}.tmp' / 'home'
    (home / '.config' / 'tint2').mkdir(parents=True)
    (home / '.config' / 'tint2' / 'tint2rc').write_text('panel_position = top\n')

    await screen(backend).display_size()

    # New windows fill the screen, the panel aside.
    assert (home / '.fluxbox' / 'apps').read_text() == '[app] (class!=Tint2)\n  [Maximized] {yes}\n[end]\n'
    assert (home / '.config' / 'tint2' / 'tint2rc').read_text() == 'panel_position = top\n'


async def test_the_panel_settings_go_where_tint2_reads_them(
    docker: FakeDocker, container_dir: Path, tmp_path: Path
) -> None:
    # tint2 reads its settings under `XDG_CONFIG_HOME` when the image sets it.
    config = tmp_path / 'config'
    docker.set_image_env('XDG_CONFIG_HOME', str(config))

    await screen(_desktop(container_dir, image='my-desktop')).display_size()

    assert 'panel_position = bottom left horizontal' in (config / 'tint2' / 'tint2rc').read_text().splitlines()


@_host_without_display_programs
async def test_a_failed_install_says_why(docker: FakeDocker, container_dir: Path) -> None:
    docker.remove_display()
    docker.uninstall('xdotool')
    docker.run_as(0)
    docker.fail_installs()

    with pytest.raises(WorkspaceUnavailableError, match='could not install xdotool:') as error:
        await screen(_desktop(container_dir, image='my-desktop')).screenshot()
    assert 'Unable to locate package' in str(error.value)
    assert docker.apt_get_calls == ['update -qq', _INSTALL]


async def test_the_socket_directory_is_made_for_a_user_without_root(docker: FakeDocker, container_dir: Path) -> None:
    # Xvfb makes `/tmp/.X11-unix` only when it runs as root.
    backend = _desktop(container_dir, image='my-desktop')
    await backend.working_dir()
    assert backend.ref is not None
    docker.remove_display()

    await screen(backend).display_size()

    socket_dir = docker.bin_dir / 'containers' / f'{backend.ref.id}.tmp' / '.X11-unix'
    assert socket_dir.stat().st_mode & 0o7777 == 0o1777


async def test_a_desktop_refuses_the_host_network(docker: FakeDocker, container_dir: Path) -> None:
    """On the host's network, programs in the container could connect to the host's X server and control its screen."""
    created = screen(
        DockerContainerBackend(
            image='my-desktop', working_dir=str(container_dir), desktop=True, docker_args=['--network', 'host']
        )
    )
    with pytest.raises(UserError, match="a desktop can't use the host's network"):
        await created.screenshot()
    # Without a desktop, nothing in the container talks to an X server.
    await DockerContainerBackend(
        image='python:3.12', working_dir=str(container_dir), docker_args=['--network', 'host']
    ).working_dir()

    # However the container got there: attached, or sharing a host-network container's network.
    docker.add_container('on-host', container_dir, network='host')
    docker.add_container('joined', container_dir, network='container:on-host')
    for name in ['on-host', 'joined']:
        attached = screen(DockerContainerBackend(ref=WorkspaceRef(provider='docker', id=name), desktop=True))
        with pytest.raises(UserError, match="a desktop can't use the host's network"):
            await attached.screenshot()
    assert docker.xdotool_calls == []

    docker.add_container('orphan', container_dir, network='container:gone')
    orphan = screen(DockerContainerBackend(ref=WorkspaceRef(provider='docker', id='orphan'), desktop=True))
    with pytest.raises(WorkspaceUnavailableError, match='could not inspect container gone'):
        await orphan.screenshot()


async def test_a_desktop_refuses_a_podman_namespace_path(docker: FakeDocker, container_dir: Path) -> None:
    """Podman keeps `ns:<path>` as the network mode, and `ns:/proc/1/ns/net` is the host's network."""
    docker.add_container('host-ns', container_dir, network='ns:/proc/1/ns/net')
    docker.add_container('joined', container_dir, network='container:host-ns')
    for name in ['host-ns', 'joined']:
        attached = screen(DockerContainerBackend(ref=WorkspaceRef(provider='docker', id=name), desktop=True))
        with pytest.raises(UserError, match="a desktop can't use the host's network"):
            await attached.screenshot()
    assert docker.xdotool_calls == []


@pytest.mark.parametrize('desktop', [(0, 0), (1280, -1), (1280,)])
def test_a_desktop_size_must_be_a_width_and_height_of_at_least_1(desktop: tuple[int, int]) -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        DockerContainerBackend(desktop=desktop)


def test_a_desktop_size_is_checked_when_the_capability_is_created() -> None:
    with pytest.raises(UserError, match=r'`desktop` must be a `\(width, height\)` of at least 1 pixel each'):
        DockerContainer(desktop=(0, 800))


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
        return 'x'.join(map(str, await screen(ctx.workspace.backend).display_size()))

    result = await agent.run('go')

    assert result.output == '{"probe":"1280x800"}'  # the size the display was started at
    assert [image.split(':')[0] for image in docker.images()] == ['pydantic-ai-desktop']
