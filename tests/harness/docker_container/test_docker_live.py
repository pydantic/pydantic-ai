"""Pydantic AI's workspace backend conformance suite, run against real containers.

Opt in with `PYDANTIC_AI_HARNESS_DOCKER_LIVE=1` on a machine with a Docker daemon:
`PYDANTIC_AI_HARNESS_DOCKER_LIVE=1 uv run pytest -m docker_live tests/harness/docker_container`.
Set `PYDANTIC_AI_HARNESS_DOCKER_EXECUTABLE=podman` to run it against Podman.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence

import anyio
import pytest

from pydantic_ai.computer_use import Click, Drag, HoldKeys, Point, PressKeys, Scroll, TypeText
from pydantic_ai.workspaces import WorkspaceBackend, WorkspaceRef, WorkspaceTimeoutError
from pydantic_ai.workspaces.conformance import WorkspaceBackendSuite
from pydantic_ai_harness.computer_use import Display
from pydantic_ai_harness.docker_container import DockerContainer, DockerContainerBackend

from ..computer_use.display_contract_live import check_display_contract, is_blank
from ..computer_use.display_fakes import full_display

LIVE_IMAGE = 'python:3.13-slim'
EXECUTABLE = os.environ.get('PYDANTIC_AI_HARNESS_DOCKER_EXECUTABLE', 'docker')

_FORK_UNTIL_FULL = """
import os, time
while True:
    try:
        pid = os.fork()
    except OSError:
        time.sleep(300)
    else:
        if pid == 0:
            time.sleep(300)
            os._exit(0)
"""
"""Forks sleeping children until the container's process limit refuses more."""

_ALIVE = "grep -qs '^State:[[:space:]]*[^Z]' {}"
"""Exits 0 when a `/proc/<pid>/status` file names a live process: a killed one is gone or a zombie `--init` reaps."""


def _backend(docker_args: Sequence[str] = ()) -> DockerContainerBackend:
    return DockerContainerBackend(image=LIVE_IMAGE, docker_args=docker_args, executable=EXECUTABLE)


async def _destroy(backend: WorkspaceBackend) -> None:
    assert backend.ref is not None
    await DockerContainer[None](executable=EXECUTABLE).destroy(backend.ref)


@pytest.mark.docker_live
class TestLiveDocker(WorkspaceBackendSuite):
    """Runs against real containers; each test's containers are removed afterwards.

    A timeout's stop finishes before `WorkspaceTimeoutError` is raised, so the tests check once that the
    command is dead rather than waiting for it to die.
    """

    @pytest.fixture
    def can_detect_exit_with_inherited_output_pipes(self) -> bool:
        # Docker's `exec` waits until every copy of the command's output is closed; Podman's returns at exit.
        return EXECUTABLE == 'podman'

    @pytest.fixture
    async def created(self) -> AsyncIterator[list[WorkspaceBackend]]:
        backends: list[WorkspaceBackend] = []
        yield backends
        for backend in backends:
            if backend.ref is not None:
                await _destroy(backend)

    @pytest.fixture
    def fresh_backend(self, created: list[WorkspaceBackend]) -> Callable[[], WorkspaceBackend]:
        def build() -> WorkspaceBackend:
            backend = _backend()
            created.append(backend)
            return backend

        return build

    @pytest.fixture
    def backend(self, fresh_backend: Callable[[], WorkspaceBackend]) -> WorkspaceBackend:
        return fresh_backend()

    @pytest.fixture
    def attach_backend(self) -> Callable[[WorkspaceRef], WorkspaceBackend]:
        return lambda ref: DockerContainerBackend(ref=ref, executable=EXECUTABLE)

    @pytest.fixture
    def destroy_environment(self) -> Callable[[WorkspaceBackend], Awaitable[None]]:
        return _destroy

    @pytest.mark.skipif(EXECUTABLE == 'podman', reason='`podman exec` returns at exit without waiting for the output')
    async def test_a_timeout_stops_background_children_that_hold_the_output(self, backend: WorkspaceBackend) -> None:
        # `docker exec` waits for the output to close, and killing the client leaves the container's processes
        # running: a fake can't reproduce this, because its processes die with the client.
        assert isinstance(backend, DockerContainerBackend)
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run("sh -c 'echo $$ > child.pid; exec sleep 300' &", shell=True, timeout=2)
        pid = (await backend.run(['cat', 'child.pid'])).stdout.strip()
        assert pid.isdigit()

        assert (await backend.run(_ALIVE.format(f'/proc/{pid}/status'), shell=True)).exit_code != 0

    async def test_a_timeout_stops_a_command_that_cleared_tmp(self, backend: WorkspaceBackend) -> None:
        # The PID files live outside `/tmp`, so clearing it doesn't stop the timeout from finding the command.
        assert isinstance(backend, DockerContainerBackend)
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run('rm -rf /tmp/* /tmp/.[!.]*; echo $$ > child.pid; exec sleep 300', shell=True, timeout=2)
        pid = (await backend.run(['cat', 'child.pid'])).stdout.strip()
        assert pid.isdigit()

        assert (await backend.run(_ALIVE.format(f'/proc/{pid}/status'), shell=True)).exit_code != 0

    async def test_a_timeout_leaves_a_process_that_started_its_own_session(self, backend: WorkspaceBackend) -> None:
        assert isinstance(backend, DockerContainerBackend)
        detached = "setsid sh -c 'echo $$ > detached.pid; exec sleep 300' > /dev/null 2>&1"
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(f'{detached} & echo $$ > child.pid; exec sleep 300', shell=True, timeout=2)
        child = (await backend.run(['cat', 'child.pid'])).stdout.strip()
        survivor = (await backend.run(['cat', 'detached.pid'])).stdout.strip()
        assert child.isdigit() and survivor.isdigit()

        assert (await backend.run(_ALIVE.format(f'/proc/{child}/status'), shell=True)).exit_code != 0
        assert (await backend.run(_ALIVE.format(f'/proc/{survivor}/status'), shell=True)).exit_code == 0

    async def test_new_containers_get_a_process_limit_docker_args_can_replace(
        self, created: list[WorkspaceBackend]
    ) -> None:
        # A set value reads back as itself on any host; what `-1` reads back as depends on the host's cgroup setup.
        default = _backend()
        custom = _backend(docker_args=['--pids-limit', '100'])
        created += [default, custom]

        assert (await default.run(['cat', '/sys/fs/cgroup/pids.max'])).stdout.strip() == '2048'
        assert (await custom.run(['cat', '/sys/fs/cgroup/pids.max'])).stdout.strip() == '100'

    async def test_a_timeout_stops_a_command_that_used_up_the_process_limit(
        self, created: list[WorkspaceBackend]
    ) -> None:
        # With no process left to fork, the stop has to manage with shell builtins.
        backend = _backend(docker_args=['--pids-limit', '32'])
        created.append(backend)
        with pytest.raises(WorkspaceTimeoutError):
            await backend.run(['python', '-c', _FORK_UNTIL_FULL], timeout=3)

        alive = await backend.run(
            f'for p in $(grep -lsx python /proc/[0-9]*/comm); do {_ALIVE.format("${p%comm}status")} && exit 0; done; exit 1',
            shell=True,
        )
        assert alive.exit_code != 0


@pytest.mark.docker_live
class TestLiveDockerDesktop:
    """Runs the display against real containers of the desktop image, which the first test builds if this machine
    lacks it. Commands and files are `TestLiveDocker`'s: a desktop container runs them the same way."""

    @pytest.fixture
    async def backend(self) -> AsyncIterator[DockerContainerBackend]:
        backend = DockerContainerBackend(executable=EXECUTABLE, desktop=True)
        yield backend
        if backend.ref is not None:
            await _destroy(backend)

    async def test_display_contract(self, backend: WorkspaceBackend) -> None:
        await check_display_contract(full_display(backend))

    async def test_the_first_screenshot_shows_the_desktop(self, backend: DockerContainerBackend) -> None:
        assert backend.display is not None
        await backend.display.display_size()  # starts the display
        assert not is_blank(await backend.display.screenshot())

    async def test_key_names_arrive_as_their_keysyms(self, backend: DockerContainerBackend) -> None:
        display = await _xev_window(backend)
        # The lock keys come last and twice, so they are off again for the rest.
        for name in [*_KEYSYMS, *_LOCK_KEYS, *_LOCK_KEYS]:
            await display.act(PressKeys(keys=(name,)))
        events = await _xev_events(backend, lambda events: _logged(events, 'KeyRelease', 'Scroll_Lock') == 2)
        presses = [detail for kind, detail, _ in events if kind == 'KeyPress']
        # xdotool presses Num Lock itself to reach a keypad digit or the decimal point.
        expected = [*_KEYSYMS.values(), *_LOCK_KEYS.values(), *_LOCK_KEYS.values()]
        assert [key for key in presses if key != 'Num_Lock'] == [key for key in expected if key != 'Num_Lock']

    async def test_each_button_arrives_as_its_x_button(self, backend: DockerContainerBackend) -> None:
        display = await _xev_window(backend)
        for button in ('left', 'middle', 'right'):
            await display.act(Click(x=400, y=300, button=button))
        events = await _xev_events(backend, lambda events: _logged(events, 'ButtonRelease', '3') == 1)
        presses = [detail for kind, detail, _ in events if kind == 'ButtonPress']
        assert presses == ['1', '2', '3']

    async def test_a_scroll_clicks_the_wheel_once_per_step(self, backend: DockerContainerBackend) -> None:
        display = await _xev_window(backend)
        for direction in ('up', 'down', 'left', 'right'):
            await display.act(Scroll(x=400, y=300, direction=direction, amount=3))
        events = await _xev_events(backend, lambda events: _logged(events, 'ButtonRelease', '7') == 3)
        presses = [detail for kind, detail, _ in events if kind == 'ButtonPress']
        assert presses == [*'444', *'555', *'666', *'777']

    async def test_a_drag_holds_the_button_from_its_start_to_its_end(self, backend: DockerContainerBackend) -> None:
        display = await _xev_window(backend)
        await display.act(Drag(path=(Point(x=200, y=200), Point(x=500, y=250), Point(x=300, y=350))))
        events = await _xev_events(backend, lambda events: _logged(events, 'ButtonRelease', '1') == 1)
        kinds = [kind for kind, _, _ in events]
        assert [(kind, detail) for kind, detail, _ in events if kind.startswith('Button')] == [
            ('ButtonPress', '1'),
            ('ButtonRelease', '1'),
        ]
        moves = events[kinds.index('ButtonPress') + 1 : kinds.index('ButtonRelease')]
        assert moves and all(kind == 'MotionNotify' and state & _BUTTON1 for kind, _, state in moves)

    async def test_a_cancelled_action_lets_go_of_what_it_pressed(self, backend: DockerContainerBackend) -> None:
        display = await _xev_window(backend)
        path = (Point(x=200, y=200), Point(x=600, y=400)) * 20
        # Each is cancelled once `xev` logged its press, however slow the daemon.
        for action, press in [
            (Drag(path=path, modifiers=('shift',)), ('ButtonPress', '1')),
            (HoldKeys(keys=('ctrl', 'a'), seconds=20), ('KeyPress', 'a')),
        ]:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(display.act, action)
                await _xev_events(backend, lambda events: _logged(events, *press) == 1)
                tasks.cancel_scope.cancel()
        # The keys are let go of last, Control_L after `a`.
        logged = await _xev_events(backend, lambda events: _logged(events, 'KeyRelease', 'Control_L') == 1)
        events = [(kind, detail) for kind, detail, _ in logged if kind != 'MotionNotify']
        # real-wait: a stray press after the cancel would reach the real X server later, with no event to wait on
        await anyio.sleep(2)
        # Nothing is pressed after the cancel, and each press has its release.
        assert [(kind, detail) for kind, detail, _ in await _xev_log(backend) if kind != 'MotionNotify'] == events
        pressed = [detail for kind, detail in events if kind.endswith('Press')]
        released = [detail for kind, detail in events if kind.endswith('Release')]
        assert sorted(pressed) == sorted(released)
        assert {'1', 'Shift_L', 'Control_L', 'a'} <= set(pressed)

    async def test_typed_text_reaches_the_focused_window(self, backend: WorkspaceBackend) -> None:
        display = await _focused_typing_window(backend)
        await display.act(TypeText(text='hello, Desktop! @#?\t~'))
        await display.act(PressKeys(keys=('?',)))
        await display.act(PressKeys(keys=('enter',)))
        assert await _typed(backend) == 'hello, Desktop! @#?\t~?\n'


async def _focused_typing_window(backend: WorkspaceBackend) -> Display:
    """Start the display and an xterm that writes what it's typed to `typed.txt`, and focus it."""
    assert isinstance(backend, DockerContainerBackend)
    display = backend.display
    assert display is not None
    await display.display_size()  # starts the display
    await backend.run(
        'setsid xterm -geometry 60x10+100+100 -e sh -c "cat > typed.txt" >/dev/null 2>&1 &',
        shell=True,
        env={'DISPLAY': ':1'},
    )
    await backend.run(
        ['xdotool', 'search', '--sync', '--onlyvisible', '--class', 'xterm'], env={'DISPLAY': ':1'}, timeout=30
    )
    await display.act(Click(x=300, y=180))
    return display


async def _typed(backend: WorkspaceBackend) -> str:
    assert isinstance(backend, DockerContainerBackend)
    typed = await backend.run('until [ -s typed.txt ]; do sleep 0.1; done; cat typed.txt', shell=True, timeout=30)
    return typed.stdout


_KEYSYMS = {
    **{'enter': 'Return', 'escape': 'Escape', 'tab': 'Tab', 'backspace': 'BackSpace', 'delete': 'Delete'},
    **{'insert': 'Insert', 'home': 'Home', 'end': 'End', 'pageup': 'Prior', 'pagedown': 'Next', 'up': 'Up'},
    **{'down': 'Down', 'left': 'Left', 'right': 'Right', 'space': 'space', 'printscreen': 'Print'},
    **{'pause': 'Pause', 'menu': 'Menu', **{f'f{n}': f'F{n}' for n in range(1, 13)}},
    **{'ctrl': 'Control_L', 'shift': 'Shift_L', 'alt': 'Alt_L', 'super': 'Super_L'},
    **{'a': 'a', '7': '7', '`': 'grave'},
}
"""The key names the display takes, with a few unshifted characters, and the keysym a window gets for each."""

_LOCK_KEYS = {'capslock': 'Caps_Lock', 'numlock': 'Num_Lock', 'scrolllock': 'Scroll_Lock'}

_BUTTON1 = 0x100
"""The bit an X event's `state` sets while the left button is down."""

_XEV_LOG = '/tmp/xev.log'

_XEV_EVENT = re.compile(
    r'^(\w+) event, serial.*?state (0x[0-9a-f]+)(?:.*?\(keysym 0x[0-9a-f]+, (\w+)\)|.*?button (\d+))?', re.S | re.M
)


async def _xev_window(backend: DockerContainerBackend) -> Display:
    """Start the display and an `xev` window over its middle that logs the input it gets, with the pointer on it."""
    display = backend.display
    assert display is not None
    await display.display_size()  # starts the display
    await backend.run(
        f'setsid sh -c "stdbuf -oL xev -event keyboard -event button -event mouse -geometry 600x400+100+100 '
        f'>{_XEV_LOG} 2>&1" >/dev/null 2>&1 &',
        shell=True,
        env={'DISPLAY': ':1'},
    )
    await backend.run(
        ['xdotool', 'search', '--sync', '--onlyvisible', '--name', 'Event Tester'], env={'DISPLAY': ':1'}, timeout=30
    )
    await display.act(Click(x=400, y=300))  # focuses it
    await _xev_events(backend, lambda events: _logged(events, 'ButtonRelease', '1') == 1)
    await backend.run(['sh', '-c', f': >{_XEV_LOG}'])
    return display


async def _xev_log(backend: DockerContainerBackend) -> list[tuple[str, str, int]]:
    """Each event `xev` logged: its type, the keysym or button, and its modifier and button state."""
    log = (await backend.run(['cat', _XEV_LOG])).stdout
    events: list[tuple[str, str, int]] = []
    for block in log.split('\n\n'):
        if match := _XEV_EVENT.search(block):
            events.append((match[1], match[3] or match[4] or '', int(match[2], 16)))
    return events


async def _xev_events(
    backend: DockerContainerBackend, done: Callable[[list[tuple[str, str, int]]], bool]
) -> list[tuple[str, str, int]]:
    """The events `xev` logged, read again until `done` holds for them or 10 seconds pass."""
    deadline = time.monotonic() + 10
    # real-wait: `xev` in the real container writes its log with no event to wait on
    while not done(events := await _xev_log(backend)) and time.monotonic() < deadline:
        await anyio.sleep(0.1)
    return events


def _logged(events: list[tuple[str, str, int]], kind: str, detail: str) -> int:
    return sum(event[:2] == (kind, detail) for event in events)
