"""Integration tests that require a real boxd machine.

The fake-backed suites cover the harness-owned logic: deadline handling, protocol translation,
and exception mapping. This live tier admits only what a correctly written fake could not catch:
real process execution over boxd's exec stream, a client-owned deadline over a real process, and
real lifecycle state in boxd's control plane.

Admission rule:
  A test belongs here only when its docstring can name the fake-encoded assumption it
  validates against real boxd behavior.

Gating:
  * `boxd_live` marker separates this tier from fake-backed tests.
  * skipped unless `PYDANTIC_AI_HARNESS_BOXD_LIVE=1` opts in explicitly.
  * also requires a non-empty `BOXD_API_KEY`; without one it skips, or fails when
    `BOXD_REQUIRE_LIVE=1`. The gate is `pytest_runtest_setup` in `conftest.py`.
  * the async `client` fixture opens a boxd client per test and closes it when the test ends.

Run locally:
`PYDANTIC_AI_HARNESS_BOXD_LIVE=1 uv run pytest -m boxd_live tests/harness/boxd_sandbox/`
"""

from __future__ import annotations

import contextlib
import json
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager

import anyio
import boxd
import pytest
from pytest_examples import CodeExample

from pydantic_ai.computer_use import Click, Drag, Point, PressKeys, Scroll, TypeText
from pydantic_ai.workspaces import WorkspaceRef, WorkspaceTimeoutError, WorkspaceUnavailableError
from pydantic_ai_harness.boxd_sandbox import BoxdSandbox, BoxdSandboxBackend
from pydantic_ai_harness.coder import Coder
from pydantic_ai_harness.computer_use import DisplayAction

from .._docs_examples import documented_cleanup, python_blocks, run_block
from .._tool_calls import call_tools
from ..computer_use.display_contract_live import check_display_contract
from ..computer_use.display_fakes import full_display
from .conftest import LIVE_AUTO_SUSPEND_TIMEOUT, delete_live_machine

pytestmark = pytest.mark.boxd_live


def _unique(prefix: str) -> str:
    return f'{prefix}-{uuid.uuid4().hex}'


# What a machine sees of boxd's own services: the HTTP status of a token request to the metadata
# endpoint (`000` when the request never reaches it), and whether boxd installed its env vars and
# integrations (`/etc/profile.d/boxd-env.sh`).
_BOXD_SERVICES = (
    "gw=$(ip route | awk '/default/ {print $3}'); "
    'curl -s -m 3 -o /dev/null -w "%{http_code}\\n" "http://$gw:9002/v1/auth/token"; '
    'test -e /etc/profile.d/boxd-env.sh && echo env-installed || echo no-env'
)


@pytest.fixture
async def client() -> AsyncIterator[boxd.AsyncBoxd]:
    """A caller-owned API client for one test's backends, closed when the test ends."""
    client = boxd.AsyncBoxd()
    try:
        yield client
    finally:
        await client.close()


@asynccontextmanager
async def _owned(
    client: boxd.AsyncBoxd, working_dir: str | None = None, *, desktop: bool | tuple[int, int] = False
) -> AsyncGenerator[BoxdSandboxBackend]:
    """Create a machine and delete it on the way out, even when the test deleted it already."""
    backend = BoxdSandboxBackend(
        client=client, auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT, working_dir=working_dir, desktop=desktop
    )
    try:
        await backend.get_sandbox()
        yield backend
    finally:
        if backend.ref is not None:
            await delete_live_machine(backend.ref, client)


async def _wait_for_status(client: boxd.AsyncBoxd, machine_id: str, status: str) -> None:
    with anyio.fail_after(120):
        while (await client.machines.get(machine_id)).status != status:
            await anyio.sleep(0.5)


async def test_a_desktop_meets_the_display_contract(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that the first computer call starts the machine's desktop, that the
    desktop keeps the size it was set to, and that a screenshot, a zoom and the pointer position come back as the
    display contract expects."""
    async with _owned(client, desktop=(1440, 900)) as backend:
        display = full_display(backend)
        await display.screenshot()
        assert await display.display_size() == (1440, 900)
        await check_display_contract(display)


# A page that logs the `code` of each key Chromium gets (its `key` when it has none), and `=` plus each character
# typed into it. It keeps the function keys, Tab and the menu key from Chromium, where they would move
# the focus off the page or open a dialog.
_KEY_LOG_PAGE = """<textarea autofocus style="width:600px;height:300px"></textarea><script>
window.K = [];
document.addEventListener('keydown', e => {
  K.push(e.code || e.key);
  if (/^F[0-9]+$/.test(e.key) || ['Tab', 'ContextMenu'].includes(e.code)) e.preventDefault();
});
document.addEventListener('input', e => e.data && K.push('=' + e.data));
document.addEventListener('contextmenu', e => e.preventDefault());
</script>"""

# Each key name the driver takes, with the key Chromium gets for it.
_KEY_CODES: dict[str, str] = {
    **{'enter': 'Enter', 'escape': 'Escape', 'tab': 'Tab', 'backspace': 'Backspace', 'delete': 'Delete'},
    **{'insert': 'Insert', 'home': 'Home', 'end': 'End', 'pageup': 'PageUp', 'pagedown': 'PageDown'},
    **{'up': 'ArrowUp', 'down': 'ArrowDown', 'left': 'ArrowLeft', 'right': 'ArrowRight', 'space': 'Space'},
    **{'printscreen': 'PrintScreen', 'pause': 'Pause', 'menu': 'ContextMenu', 'capslock': 'CapsLock'},
    **{'numlock': 'NumLock', 'scrolllock': 'ScrollLock', **{f'f{n}': f'F{n}' for n in range(1, 13)}},
    **{'a': 'KeyA', 'Z': 'KeyZ', '7': 'Digit7', '+': 'Equal', '!': 'Digit1', 'é': '=é', '\n': 'Enter', '\t': 'Tab'},
    **{'super': 'MetaLeft'},
}


async def test_the_desktop_takes_every_key_name(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that the X key name sent for each key presses that key, alone and with
    modifiers held. Chromium (`agent-browser`, which boxd's image includes) reads back each key."""
    async with _owned(client, desktop=True) as backend:
        display = backend.display
        assert display is not None
        await display.screenshot()
        await backend.write_bytes('/tmp/keys.html', _KEY_LOG_PAGE.encode())
        await backend.run(['agent-browser', 'open', 'file:///tmp/keys.html'], timeout=120)
        await display.act(Click(x=300, y=250))

        async def pressed(*keys: str) -> list[str]:
            await display.act(PressKeys(keys=keys))
            # Polls Chromium's key log. A key that never arrives leaves the log empty until the deadline.
            got: list[str] = []
            # real-wait: the real machine answers on its own clock
            with anyio.move_on_after(10):
                while not got:
                    logged = await backend.run(['agent-browser', 'eval', 'JSON.stringify(K.splice(0))'], timeout=60)
                    got = json.loads(json.loads(logged.stdout))
            return got

        missed = {name: got for name, code in _KEY_CODES.items() if code not in (got := await pressed(name))}
        assert missed == {}
        assert await pressed('ctrl', 'shift', '+') == ['ControlLeft', 'ShiftLeft', 'Equal']


# Starts a terminal on the desktop that saves what is typed into it, as an agent would start a program.
_TERMINAL = "setsid xterm -e sh -c 'cat > typed.txt' </dev/null >/dev/null 2>&1 &"


async def test_a_program_started_from_a_command_opens_on_the_desktop_and_takes_any_text(
    client: boxd.AsyncBoxd,
) -> None:
    """Validates the fake-encoded assumptions that commands get the desktop's `DISPLAY`, and that typed text reaches
    the focused window as Unicode, with a newline as Enter, even past boxd's 10,000 characters per call."""
    async with _owned(client, desktop=True) as backend:
        display = backend.display
        assert display is not None
        await display.screenshot()
        await backend.run(['sh', '-c', _TERMINAL], timeout=60)
        # No `--onlyvisible`: without a window manager, xdotool calls xterm's window unviewable even while it takes
        # input.
        await backend.run(['xdotool', 'search', '--sync', '--class', 'xterm'], timeout=30)

        await display.act(Click(x=50, y=50))
        # Over boxd's 10,000 characters per call, in lines short enough for a terminal. A terminal keeps at most 4,095
        # characters of one line.
        text = 'héllo 😀\n' + ('x' * 99 + '\n') * 101
        await display.act(TypeText(text=text))
        await display.act(PressKeys(keys=('ctrl', 'd')))

        with anyio.fail_after(30):  # real-wait: bounds the poll below
            # real-wait: the real terminal writes the file on its own clock
            while (typed := await backend.run(['cat', 'typed.txt'], timeout=30)).stdout.count('\n') < 102:
                await anyio.sleep(0.5)
        assert typed.stdout == text


async def test_the_panel_opens_a_terminal_below_it_that_has_the_keyboard(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that the panel start command, which the fake doesn't run, installs and
    shows the panel on boxd's desktop, and that its Terminal button opens xterm below the panel with the keyboard,
    so text typed straight after the click reaches the shell."""
    async with _owned(client, desktop=True) as backend:
        display = backend.display
        assert display is not None
        await display.screenshot()
        assert (await backend.run(['xdotool', 'search', '--class', 'tint2'], timeout=30)).exit_code == 0

        await display.act(Click(x=60, y=20))
        found = await backend.run(['xdotool', 'search', '--sync', '--class', 'xterm'], timeout=30)
        terminal = found.stdout.split()[0]
        # A new window can take the keyboard only once a computer call has shown it, as the agent's next look does.
        with anyio.fail_after(10):  # real-wait: bounds the poll below
            # real-wait: a real window takes the keyboard only after a later computer call
            while (await backend.run(['xdotool', 'getwindowfocus'], timeout=30)).stdout.strip() != terminal:
                await display.display_size()
                await anyio.sleep(0.2)
        await display.act(TypeText(text='echo typed > /tmp/typed.txt\n'))

        with anyio.fail_after(30):  # real-wait: bounds the poll below
            # real-wait: the real shell writes the file on its own clock
            while (typed := await backend.run(['cat', '/tmp/typed.txt'], timeout=30)).exit_code != 0:
                await anyio.sleep(0.5)
        assert typed.stdout == 'typed\n'
        window = await backend.run(['xwininfo', '-id', terminal], timeout=30)
        assert 'Absolute upper-left Y:  40' in window.stdout


async def test_boxd_takes_scrolls_key_repeats_and_drags_past_its_per_call_limits(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that wheel clicks, key repeats and drag points over boxd's limit of 100
    per call are split into calls that boxd accepts."""
    async with _owned(client, desktop=True) as backend:
        display = backend.display
        assert display is not None
        await display.screenshot()
        await display.act(Scroll(x=50, y=50, direction='down', amount=150))
        await display.act(PressKeys(keys=('shift',), repeat=150))
        await display.act(Drag(path=tuple(Point(x=600 + n, y=400) for n in range(150))))


# A page that logs a drag from the left box that lands on the right one, each key's `code`, and the modifiers held
# around each click and wheel turn. The boxes fill the window's height, so a point 400 pixels down the screen is on
# them below Chromium's toolbar.
_EVENT_LOG_PAGE = """<body style="margin:0">
<div id="source" draggable="true" style="position:absolute;left:0;width:400px;height:100vh"></div>
<div id="target" style="position:absolute;left:600px;width:400px;height:100vh"></div><script>
window.E = [];
source.addEventListener('dragstart', e => { E.push('dragstart'); e.dataTransfer.setData('text/plain', 'x'); });
target.addEventListener('dragover', e => e.preventDefault());
target.addEventListener('drop', e => { e.preventDefault(); E.push('drop'); });
document.addEventListener('keydown', e => E.push(e.code));
document.addEventListener('click', e => E.push(`click shift=${e.shiftKey} meta=${e.metaKey}`));
document.addEventListener('wheel', e => E.push(`wheel ctrl=${e.ctrlKey}`));
</script></body>"""


async def test_drags_key_repeats_and_modifiers_reach_a_browser(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that a browser's drag-and-drop starts and lands both with boxd's `drag`
    and with a drag split over 100 points, that a key repeated 3 times is pressed 3 times, and that boxd holds
    modifiers during a click, a wheel turn and a drag, with `super` as the Meta key."""
    async with _owned(client, desktop=True) as backend:
        display = full_display(backend)
        await display.screenshot()
        await backend.write_bytes('/tmp/events.html', _EVENT_LOG_PAGE.encode())
        await backend.run(['agent-browser', 'open', 'file:///tmp/events.html'], timeout=120)

        async def logged(action: DisplayAction) -> list[str]:
            await display.act(action)
            # real-wait: a stray key press on the real desktop arrives with no event to wait on
            await anyio.sleep(1)  # lets every event arrive, so an extra key press would show
            got = await backend.run(['agent-browser', 'eval', 'JSON.stringify(E.splice(0))'], timeout=60)
            return json.loads(json.loads(got.stdout))

        drag = ['dragstart', 'drop']
        assert await logged(Drag(path=(Point(x=200, y=400), Point(x=800, y=400)))) == drag
        assert await logged(Drag(path=tuple(Point(x=200 + 4 * n, y=400) for n in range(150)))) == drag
        assert await logged(Click(x=500, y=400)) == ['click shift=false meta=false']  # focuses the page
        assert await logged(PressKeys(keys=('a',), repeat=3)) == ['KeyA'] * 3
        assert await logged(Click(x=500, y=400, modifiers=('shift',))) == ['ShiftLeft', 'click shift=true meta=false']
        assert await logged(Click(x=500, y=400, modifiers=('super',))) == ['MetaLeft', 'click shift=false meta=true']
        scroll = Scroll(x=500, y=400, direction='down', amount=1, modifiers=('ctrl',))
        assert await logged(scroll) == ['ControlLeft', 'wheel ctrl=true']
        shift_drag = Drag(path=(Point(x=500, y=400), Point(x=510, y=400)), modifiers=('shift',))
        assert await logged(shift_drag) == ['ShiftLeft', 'click shift=true meta=false']


async def test_creates_an_isolated_machine_and_runs_a_command(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that the exec stream reports real output and exit code on
    separate streams, and that the machine boxd creates is isolated: its metadata endpoint refuses token
    requests, and boxd installs no CLI, env vars, or integrations in it, so code the agent runs holds no
    boxd credentials. `test_a_machine_that_is_not_isolated_is_neither_used_nor_deleted` runs the same
    probe on a machine that is not isolated, where it succeeds."""
    async with _owned(client) as backend:
        result = await backend.run('echo out; echo err 1>&2; exit 3', shell=True, timeout=60)
        assert (result.stdout, result.stderr, result.exit_code) == ('out\n', 'err\n', 3)

        machine = await backend.get_sandbox()
        assert (machine.networking.isolated, machine.networking.networks) == (True, [])
        probe = await backend.run(['sh', '-c', f'command -v boxd || echo no-cli; {_BOXD_SERVICES}'], timeout=60)
        # The endpoint refuses an isolated machine, and boxd may drop the request before it gets there.
        assert probe.stdout.splitlines() in (['no-cli', '403', 'no-env'], ['no-cli', '000', 'no-env'])


async def test_large_output_is_exact(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that the exec stream delivers a command's output whole and in order."""
    async with _owned(client) as backend:
        result = await backend.run(['seq', '1', '200000'], timeout=120)
        assert result.stdout == ''.join(f'{i}\n' for i in range(1, 200001))


# A process that outlives the command starting it, and a check that it still runs.
_TICKER = 'setsid sh -c "while :; do date +%s > tick; sleep 1; done" </dev/null >/dev/null 2>&1 & echo $! > ticker.pid'
_TICKING = 'kill -0 "$(cat ticker.pid)" && echo ticking'


async def test_fork_copies_files_and_running_processes_and_the_copies_diverge(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a fork is a new isolated machine on no network, and shows
    what the fake can't: the copy starts with the machine's files and running processes, and the two
    change independently after."""
    sandbox = BoxdSandbox(client=client)
    async with _owned(client) as source:
        await source.run(['sh', '-c', f'echo before > state.txt; {_TICKER}'], timeout=60)
        assert source.ref is not None
        ref = await sandbox.fork(source.ref)
        try:
            assert ref != source.ref
            copy = BoxdSandboxBackend(client=client, ref=ref)
            machine = await copy.get_sandbox()
            assert (machine.networking.isolated, machine.networking.networks) == (True, [])
            assert (
                await copy.run(['sh', '-c', f'cat state.txt; {_TICKING}'], timeout=60)
            ).stdout == 'before\nticking\n'
            await copy.run(['sh', '-c', 'echo after > state.txt'], timeout=60)
            assert (await source.run(['cat', 'state.txt'], timeout=60)).stdout == 'before\n'
        finally:
            await delete_live_machine(ref, client)


async def test_fork_wakes_a_suspended_machine_so_the_copy_keeps_its_memory(client: boxd.AsyncBoxd) -> None:
    """boxd forks a suspended machine from its disk alone, without its running processes or the writes it
    had not flushed to disk; `fork` wakes the machine first, so the copy has both."""
    sandbox = BoxdSandbox(client=client)
    async with _owned(client) as source:
        await source.run(['sh', '-c', f'echo unflushed > state.txt; {_TICKER}'], timeout=60)
        assert source.ref is not None
        await client.machines.pause(source.ref.id)
        await _wait_for_status(client, source.ref.id, 'suspended')
        ref = await sandbox.fork(source.ref)
        try:
            copy = BoxdSandboxBackend(client=client, ref=ref)
            assert (
                await copy.run(['sh', '-c', f'cat state.txt; {_TICKING}'], timeout=60)
            ).stdout == 'unflushed\nticking\n'
        finally:
            await delete_live_machine(ref, client)


async def test_timeout_raises_with_the_partial_output_and_stops_the_command(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that output printed before a client-side deadline reaches the
    `WorkspaceTimeoutError`, and that cancelling the exec call kills the command's process group."""
    marker = f'/tmp/{_unique("after-deadline")}'
    async with _owned(client) as backend:
        with pytest.raises(WorkspaceTimeoutError) as exc_info:
            await backend.run(f'echo DIAGNOSTIC; (sleep 5; touch {marker}) & sleep 20', shell=True, timeout=3)
        assert exc_info.value.stdout == 'DIAGNOSTIC\n'
        # Outlasts the background `sleep 5`, so a child left running would have written the marker.
        assert (await backend.run(['sleep', '6'], timeout=30)).exit_code == 0
        assert (await backend.run(['test', '-e', marker], timeout=30)).exit_code == 1


async def test_a_new_machine_gets_its_working_dir(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that `mkdir -p` through a unary exec makes `working_dir` on a
    new machine as the non-root `boxd` user, so the first command can start there."""
    directory = f'/home/boxd/{_unique("project")}/nested'
    async with _owned(client, working_dir=directory) as backend:
        result = await backend.run(['sh', '-c', 'pwd -P; stat -c %U .'], timeout=60)
        assert result.stdout.splitlines() == [directory, 'boxd']


async def test_a_stopped_machine_is_started_again_with_its_disk(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions that a command on a stopped machine is refused with
    `ConflictError`, and that attaching starts it again and waits until it accepts commands."""
    path = f'/home/boxd/{_unique("kept")}.txt'
    async with _owned(client) as owner:
        await owner.run(['sh', '-c', 'printf kept > "$1"', 'sh', path], timeout=60)
        assert owner.ref is not None
        await client.machines.stop(owner.ref.id)
        await _wait_for_status(client, owner.ref.id, 'stopped')

        with pytest.raises(WorkspaceUnavailableError, match='was stopped'):
            await owner.run(['true'], timeout=30)
        attached = BoxdSandboxBackend(client=client, ref=owner.ref)
        assert (await attached.run(['cat', path], timeout=60)).stdout == 'kept'


@pytest.mark.parametrize('sleep', ['pause', 'hibernate'])
async def test_a_sleeping_machine_is_woken_by_a_command(client: boxd.AsyncBoxd, sleep: str) -> None:
    """Validates the fake-encoded assumption that a command wakes a suspended or hibernated machine,
    with its running processes, so neither needs an explicit resume."""
    async with _owned(client) as backend:
        assert backend.ref is not None
        machine_id = backend.ref.id
        await backend.run(['sh', '-c', 'nohup sleep 600 >/dev/null 2>&1 & echo $! > /tmp/sleeper'], timeout=60)
        if sleep == 'pause':
            await client.machines.pause(machine_id)
            await _wait_for_status(client, machine_id, 'suspended')
        else:
            await client.machines.hibernate(machine_id)
            await _wait_for_status(client, machine_id, 'hibernated')

        result = await BoxdSandboxBackend(client=client, ref=backend.ref).run(
            ['sh', '-c', 'kill -0 "$(cat /tmp/sleeper)" && echo alive'], timeout=60
        )
        assert result.stdout == 'alive\n'


async def test_a_machine_that_is_not_isolated_is_neither_used_nor_deleted(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a machine record reports `networking.isolated`, so a
    ref from tampered message history cannot point the agent at, or delete, one of the org's other machines.

    The machine's own view is the positive control for the isolation probe above: here the metadata
    endpoint hands out tokens and boxd's env vars and integrations are installed."""
    machine = await client.machines.create(
        _unique('pydantic-ai-e2e-open')[:40], isolated=False, auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT
    )
    try:
        await client.machines.wait_until_ready(machine.id)
        # boxd writes `boxd-env.sh` a moment after `wait_until_ready` returns, so the probe first waits up to
        # 30 seconds for it.
        wait_for_env = "timeout 30 sh -c 'until test -e /etc/profile.d/boxd-env.sh; do sleep 0.1; done'"
        probe = await client.machines.exec(machine.id, f'{wait_for_env}; {_BOXD_SERVICES}')
        assert str(probe.stdout).splitlines() == ['200', 'env-installed']
        ref = WorkspaceRef(provider='boxd', id=machine.id)
        with pytest.raises(WorkspaceUnavailableError, match='is not isolated'):
            await BoxdSandboxBackend(client=client, ref=ref).run(['true'], timeout=30)
        with pytest.raises(ValueError, match='is not isolated'):
            await BoxdSandbox(client=client).destroy(ref)
        with pytest.raises(WorkspaceUnavailableError, match='is not isolated'):
            await BoxdSandbox(client=client).fork(ref)
        assert (await client.machines.get(machine.id)).status != 'destroyed'
    finally:
        await client.machines.delete(machine.id)


async def test_a_snapshot_of_a_machine_on_a_network_is_refused(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a machine created from a snapshot keeps the named networks
    of the machine the snapshot was saved from, even when created isolated, and that the backend refuses it
    before running anything there."""
    snapshot = _unique('pydantic-ai-e2e-net')[:40]
    async with _owned(client) as source:
        assert source.ref is not None
        await client.machines.set_networks(source.ref.id, [snapshot])
        await client.snapshots.create(source.ref.id, snapshot)
        try:
            with anyio.fail_after(120):
                while (await client.snapshots.get(snapshot)).status != 'ready':
                    await anyio.sleep(0.5)
            restored = BoxdSandboxBackend(client=client, snapshot=snapshot)
            try:
                with pytest.raises(WorkspaceUnavailableError, match=f'is on the boxd networks {snapshot}'):
                    await restored.run(['true'], timeout=30)
            finally:
                assert restored.ref is not None
                await delete_live_machine(restored.ref, client)
        finally:
            await client.snapshots.delete(snapshot)


async def test_a_machine_stopped_during_a_command_is_unavailable(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that stopping a machine ends a running command with
    `APIConnectionError`, after which a lookup reports the machine `stopped`."""
    async with _owned(client) as backend:
        assert backend.ref is not None
        machine_id = backend.ref.id

        async def stop_soon() -> None:
            await anyio.sleep(3)
            await client.machines.stop(machine_id)

        async with anyio.create_task_group() as group:
            group.start_soon(stop_soon)
            with pytest.raises(WorkspaceUnavailableError, match='was stopped'):
                await backend.run(['sleep', '60'], timeout=90)


async def test_coder_tools_run_in_the_machine(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a relative path `Coder` writes through boxd's file API
    is where a command started in the working directory finds it."""
    async with _owned(client) as backend:
        results = await call_tools(
            [Coder()],
            [
                ('write_file', {'path': 'hello.py', 'content': "print('hi from boxd')\n"}),
                ('shell', {'command': 'python3 hello.py'}),
            ],
            workspace=backend,
        )
        assert 'hi from boxd' in results[1]


async def test_commands_get_a_usable_environment(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumption that a command sees the image's `PATH` and `HOME`, and what the
    docs say the default image has: the non-root `boxd` user in `/home/boxd`, git, Python, and ripgrep. A
    per-call `env` adds to the environment rather than replacing it."""
    async with _owned(client) as backend:
        result = await backend.run(
            'echo "$PATH"; echo "$HOME"; whoami; pwd -P; git --version; python3 --version; command -v rg',
            shell=True,
            timeout=60,
        )
        path, home, user, cwd, git, python, rg = result.stdout.splitlines()
        assert (home, user, cwd) == ('/home/boxd', 'boxd', '/home/boxd')
        assert rg.endswith('/rg')
        assert path and git.startswith('git version') and python.startswith('Python 3')
        assert await backend.working_dir() == '/home/boxd'

        result = await backend.run('echo "$FOO"; echo "$PATH"', shell=True, env={'FOO': '1'}, timeout=60)
        assert result.stdout.splitlines() == ['1', path]


async def test_secrets_reach_the_machine_as_stand_ins(client: boxd.AsyncBoxd) -> None:
    """Validates the fake-encoded assumptions behind `secrets`: boxd refuses `isolated` and `vcpu` beside a
    spec, so the spec carries them; its `'{n}x{4n}'` size gives the machine `n` vCPUs; a spec combines with
    `from_snapshot`; and the secret's variable holds a `bxds_` stand-in, not the value. The value is a dummy."""
    secret = _unique('PYDANTIC_AI_E2E').replace('-', '_').upper()
    snapshot = _unique('pydantic-ai-e2e-secrets')[:40]
    probe = f'nproc; v=$(printenv {secret}); echo "${{v%%_*}}_"; [ "$v" = dummy-value ] && echo value || echo no-value'
    await client.secrets.set(secret, 'dummy-value', scope='private-unattached', domains=['example.com'])
    machines: list[WorkspaceRef] = []
    try:
        backend = BoxdSandboxBackend(
            client=client, secrets=[secret], vcpu=2, auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT
        )
        machine = await backend.get_sandbox()
        assert backend.ref is not None
        machines.append(backend.ref)
        assert machine.networking.isolated
        assert (await backend.run(['sh', '-c', probe], timeout=60)).stdout.split() == ['2', 'bxds_', 'no-value']

        await client.snapshots.create(backend.ref.id, snapshot)
        with anyio.fail_after(120):
            while (await client.snapshots.get(snapshot)).status != 'ready':
                await anyio.sleep(0.5)
        restored = BoxdSandboxBackend(
            client=client, snapshot=snapshot, secrets=[secret], auto_suspend_timeout=LIVE_AUTO_SUSPEND_TIMEOUT
        )
        machine = await restored.get_sandbox()
        assert restored.ref is not None
        machines.append(restored.ref)
        assert machine.networking.isolated
        assert (await restored.run(['sh', '-c', probe], timeout=60)).stdout.split()[1:] == ['bxds_', 'no-value']
    finally:
        for ref in machines:
            await delete_live_machine(ref, client)
        with contextlib.suppress(boxd.NotFoundError):
            await client.snapshots.delete(snapshot)
        await client.secrets.delete(secret, scope='private-unattached')


# The README's Python blocks are the same as this page's.
_DOCS_BLOCKS = python_blocks('docs/harness/boxd-sandbox.md')


@pytest.mark.parametrize('example', [pytest.param(block, id=f'line {block.start_line}') for block in _DOCS_BLOCKS])
def test_docs_example(example: CodeExample) -> None:
    """Every example on the docs page runs as written, and its agent's tools do their work in a real machine.

    The fake stands in for boxd, so only this shows the page's code, its default settings, and its
    cleanup work against the real service. A follow-up run, from the message history or a stored ref,
    works in the first run's machine.
    """
    _, runs = run_block(example, cleanup=documented_cleanup(_DOCS_BLOCKS, 'delete_machine'))
    assert all(run.used_sandbox for run in runs), runs
    # Follow-up runs work in the first run's machine, or in a copy of it the block forked.
    assert len({run.ref for run in runs}) <= 1 + example.source.count('.fork('), runs
