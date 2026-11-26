"""Tests for the desktop display's input and capture: `xdotool` and `scrot` commands, run through a backend's commands.

`ScriptedDesktop` is a desktop `DockerContainerBackend` whose display commands answer from a script instead of a
container.
"""

from __future__ import annotations

import base64
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import Path

import anyio
import pytest

from pydantic_ai.computer_use import (
    Click,
    Drag,
    HoldKeys,
    MouseDown,
    MouseMove,
    MouseUp,
    Point,
    PressKeys,
    Region,
    Scroll,
    TypeText,
)
from pydantic_ai.workspaces import (
    CommandResult,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceTimeoutError,
    WorkspaceUnavailableError,
)
from pydantic_ai_harness.computer_use import Display, DisplayAction, SupportsZoom, UnsupportedActionError, _x11
from pydantic_ai_harness.docker_container import DockerContainerBackend

from ._fake_x_display import FakeDisplay
from .display_contract_live import is_blank
from .display_fakes import FAKE_SCREENSHOT, png

pytestmark = pytest.mark.skipif(
    os.name != 'posix', reason='`DockerContainerBackend` runs `docker` as a POSIX subprocess'
)

INPUT = 'echo $$ > /tmp/.pydantic-ai-input-1; exec "$@"'
"""How an input command runs: recording its process id, so a cancel can stop it."""

STOP_INPUT_PREFIX = 'p=$(cat /tmp/.pydantic-ai-input-1'
"""How the script that stops a failed or cancelled input command starts."""


def stops_then(*releases: list[str]) -> list[list[str]]:
    """After a failed or cancelled input, the stop of its command, then each release on its own."""
    return [['sh', '-c', STOP_INPUT_PREFIX], *releases]


def recovery(argvs: list[WorkspaceCommand]) -> list[WorkspaceCommand]:
    """`argvs` after the input command, with the stop script cut to its start for comparing."""
    return [
        ['sh', '-c', STOP_INPUT_PREFIX] if argv[:2] == ['sh', '-c'] and argv[2].startswith(STOP_INPUT_PREFIX) else argv
        for argv in argvs[1:]
    ]


PNG_OUTPUT = CommandResult(exit_code=0, stdout=base64.encodebytes(FAKE_SCREENSHOT).decode(), stderr='')
"""A capture's output: the screenshot in wrapped base64 lines, as `base64` prints it."""


class ScriptedDesktop(DockerContainerBackend):
    """Records each display command and answers with the next scripted result (exit 0 and no output by default).

    The display runs its commands through `_run_argv`, which doesn't start the display first, so that is what this
    scripts.

    Starting the display always succeeds and is only counted. With `block_first`, the first command blocks until
    cancelled.
    """

    def __init__(
        self, *results: CommandResult | Exception, block_first: bool = False, desktop: bool | tuple[int, int] = True
    ) -> None:
        super().__init__(image='my-desktop', desktop=desktop)
        self.results = list(results)
        self.calls: list[tuple[WorkspaceCommand, Mapping[str, str]]] = []
        self.timeouts: list[float | None] = []
        self.starts = 0
        self.start_argvs: list[list[str]] = []
        self.block_first = block_first
        self.started = anyio.Event()
        assert self.display is not None
        self.screen: Display = self.display

    async def _run_argv(self, argv: list[str], *, env: Mapping[str, str], timeout: float | None) -> CommandResult:
        await anyio.lowlevel.checkpoint()  # as a real command does, so a deadline can cut it short
        if 'Xvfb' in str(argv):
            self.starts += 1
            self.start_argvs.append(argv)
            await anyio.sleep(0)  # a start takes a moment, so concurrent first uses overlap
            return CommandResult(exit_code=0, stdout='', stderr='')
        self.calls.append((argv, env))
        self.timeouts.append(timeout)
        if self.block_first and len(self.calls) == 1:
            self.started.set()
            await anyio.sleep_forever()
        result = self.results.pop(0) if self.results else CommandResult(exit_code=0, stdout='', stderr='')
        if isinstance(result, Exception):
            raise result
        return result

    @property
    def argvs(self) -> list[WorkspaceCommand]:
        return [command for command, _ in self.calls]


@pytest.mark.parametrize(
    'action, argv',
    [
        (Click(x=10, y=20), ['xdotool', 'mousemove', '10', '20', 'click', '--repeat', '1', '1']),
        (Click(count=2, button='right'), ['xdotool', 'click', '--repeat', '2', '3']),
        (
            Click(x=1, y=2, button='middle', modifiers=('ctrl', 'shift')),
            ['xdotool', 'keydown', 'ctrl', 'shift', 'mousemove', '1', '2', 'click', '--repeat', '1', '2']
            + ['keyup', 'shift', 'ctrl'],
        ),
        (MouseMove(x=5, y=6), ['xdotool', 'mousemove', '5', '6']),
        (
            Drag(path=(Point(x=0, y=0), Point(x=500, y=0), Point(x=500, y=10))),
            # Each leg is moved along in steps of at most 200 pixels, and at least two.
            # It ends 1 pixel past the end and back, so a browser's drag-and-drop takes the drop.
            ['xdotool', 'mousemove', '0', '0', 'mousedown', '1']
            + ['sleep', '0.05', 'mousemove', '166', '0', 'sleep', '0.05', 'mousemove', '333', '0']
            + ['sleep', '0.05', 'mousemove', '500', '0', 'sleep', '0.05', 'mousemove', '500', '5']
            + ['sleep', '0.05', 'mousemove', '500', '10', 'sleep', '0.05', 'mousemove', '501', '10']
            + ['sleep', '0.05', 'mousemove', '500', '10', 'sleep', '0.05', 'mouseup', '1'],
        ),
        (MouseDown(button='middle'), ['xdotool', 'mousedown', '2']),
        (MouseUp(), ['xdotool', 'mouseup', '1']),
        (
            Scroll(x=7, y=8, direction='down', amount=3),
            ['xdotool', 'mousemove', '7', '8', 'click', '--repeat', '3', '--delay', '20', '5'],
        ),
        (Scroll(direction='up', amount=2), ['xdotool', 'click', '--repeat', '2', '--delay', '20', '4']),
        (
            TypeText(text='-n "hi"'),
            ['xdotool', 'key', '--', 'U002D', 'U006E', 'U0020', 'U0022', 'U0068', 'U0069', 'U0022'],
        ),
        (PressKeys(keys=('ctrl', 's'), repeat=3), ['xdotool', 'key', '--repeat', '3', '--', 'ctrl+s']),
        (PressKeys(keys=('pagedown', 'f5', 'up')), ['xdotool', 'key', '--repeat', '1', '--', 'Next+F5+Up']),
        (
            PressKeys(keys=('printscreen', 'capslock', 'menu')),
            ['xdotool', 'key', '--repeat', '1', '--', 'Print+Caps_Lock+Menu'],
        ),
        (PressKeys(keys=('ctrl', '+')), ['xdotool', 'key', '--repeat', '1', '--', 'ctrl+plus']),
        # A character key presses what the layout needs to type it: xdotool adds shift for `?`.
        (PressKeys(keys=('ctrl', '?')), ['xdotool', 'key', '--repeat', '1', '--', 'ctrl+U003F']),
        (PressKeys(keys=('@',)), ['xdotool', 'key', '--repeat', '1', '--', 'U0040']),
        (HoldKeys(keys=('shift',), seconds=0.5), ['xdotool', 'keydown', 'shift', 'sleep', '0.5', 'keyup', 'shift']),
        # Enter and Tab for newlines and tabs in typed text.
        (
            TypeText(text='l\n\tx\n'),
            ['xdotool', 'key', '--', 'U006C', 'Return', 'Tab', 'U0078', 'Return'],
        ),
    ],
)
async def test_x11_computer_runs_one_xdotool_command_per_action(action: DisplayAction, argv: list[str]) -> None:
    desktop = ScriptedDesktop()
    await desktop.screen.act(action)
    assert desktop.calls == [(['sh', '-c', INPUT, 'sh', *argv], {'DISPLAY': ':1'})]


async def test_x11_computer_splits_long_typing_and_drags_into_commands_under_the_size_limit() -> None:
    """Each command's `xdotool` arguments stay under 100,000 bytes, well below Linux's 128 KiB limit on one argument."""
    desktop = ScriptedDesktop()

    await desktop.screen.act(TypeText(text='a' * 20_000))
    typed = desktop.argvs
    assert len(typed) == 2
    assert all(len(shlex.join(argv[5:])) < 100_000 for argv in typed)
    assert [argv[4:7] for argv in typed] == [['xdotool', 'key', '--']] * 2
    assert sum(argv.count('U0061') for argv in typed) == 20_000

    # Held keys go down in the first command and come up in the last, so they stay held in between.
    await desktop.screen.act(Drag(path=tuple(Point(x=n % 2, y=0) for n in range(2_500)), modifiers=('alt',)))
    dragged = desktop.argvs[2:]
    assert len(dragged) == 2
    assert all(len(shlex.join(argv[5:])) < 100_000 for argv in dragged)
    assert dragged[0][4:7] == ['xdotool', 'keydown', 'alt']
    assert dragged[-1][-4:] == ['mouseup', '1', 'keyup', 'alt']


@pytest.mark.parametrize(
    'action, timeout',
    [
        (HoldKeys(keys=('shift',), seconds=200), 350),
        (Scroll(direction='down', amount=1_000), 170),
        (PressKeys(keys=('a',), repeat=1_000), 162),
        (PressKeys(keys=('ctrl', 'shift', 'a'), repeat=1_000), 186),
        (TypeText(text='a' * 1_000), 162),
        # 0.05 s before each of the 2 moves, the 2 moves past the end and back, and letting go.
        (Drag(path=(Point(x=0, y=0), Point(x=10, y=0))), 150.25),
        (Click(x=1, y=1), 150.1),
        # xdotool waits 0.1 s between the clicks of a repeated click.
        (Click(count=3), 150.3),
    ],
)
async def test_x11_computer_allows_for_the_pauses_an_action_asks_for(action: DisplayAction, timeout: float) -> None:
    desktop = ScriptedDesktop()
    await desktop.screen.act(action)
    assert desktop.timeouts == [pytest.approx(timeout)]


@pytest.mark.parametrize(
    'action',
    [TypeText(text='café'), PressKeys(keys=('ctrl', 'é')), HoldKeys(keys=('日',), seconds=1)],
    ids=['type', 'press', 'hold'],
)
async def test_x11_computer_refuses_characters_outside_ascii(action: DisplayAction) -> None:
    # `xdotool` types them by remapping a key per keystroke, which Chrome reads too late; nothing is sent.
    desktop = ScriptedDesktop()
    with pytest.raises(UnsupportedActionError, match='can only type ASCII characters'):
        await desktop.screen.act(action)
    assert desktop.calls == []


@pytest.mark.parametrize(
    'action',
    [Click(count=2**31), Scroll(direction='down', amount=2**31), PressKeys(keys=('a',), repeat=10**400)],
    ids=['click', 'scroll', 'press_keys'],
)
async def test_x11_computer_refuses_a_count_xdotool_cannot_take(action: DisplayAction) -> None:
    desktop = ScriptedDesktop()
    with pytest.raises(UnsupportedActionError, match='at most 2147483647 times'):
        await desktop.screen.act(action)
    assert desktop.calls == []


@pytest.mark.parametrize(
    'action, releases',
    [
        pytest.param(
            Click(x=1, y=2, modifiers=('ctrl', 'shift')),
            stops_then(['xdotool', 'mouseup', '1'], ['xdotool', 'keyup', 'shift', 'ctrl']),
            id='click-with-modifiers',
        ),
        pytest.param(
            Drag(path=(Point(x=1, y=2), Point(x=3, y=4)), modifiers=('alt',)),
            stops_then(['xdotool', 'mouseup', '1'], ['xdotool', 'keyup', 'alt']),
            id='drag-with-modifier',
        ),
        pytest.param(PressKeys(keys=('ctrl', 's'), repeat=3), stops_then(['xdotool', 'keyup', 's', 'ctrl']), id='keys'),
        # Typing can stop between a key's press and release; X would then repeat the key without end.
        pytest.param(
            TypeText(text='Yay'),
            stops_then(['xdotool', 'keyup', 'Shift_L', 'U0059', 'U0061', 'U0079']),
            id='type',
        ),
        # A click can stop between pressing and releasing its button.
        pytest.param(Click(x=1, y=2, button='right'), stops_then(['xdotool', 'mouseup', '3']), id='click'),
        pytest.param(MouseMove(x=1, y=2), stops_then(), id='mouse_move'),
    ],
)
async def test_x11_computer_releases_what_a_failed_action_held(
    action: DisplayAction, releases: list[list[str]]
) -> None:
    desktop = ScriptedDesktop(CommandResult(exit_code=1, stdout='', stderr='BadValue'))
    with pytest.raises(WorkspaceError, match='BadValue'):
        await desktop.screen.act(action)
    assert recovery(desktop.argvs) == releases


async def test_x11_computer_tries_every_release_when_one_fails_and_says_what_may_still_be_held() -> None:
    failed = CommandResult(exit_code=1, stdout='', stderr='BadValue')
    # The action fails, then stopping it, then letting go of the button.
    desktop = ScriptedDesktop(failed, failed, failed)
    failure = 'Display command failed with exit code 1 on display :1: BadValue'
    with pytest.raises(WorkspaceError) as error:
        await desktop.screen.act(Click(x=1, y=2, modifiers=('ctrl',)))
    assert str(error.value) == (
        f'{failure}. Releasing what it pressed failed too, so a mouse button or key may still be held: '
        f'{failure}; {failure}'
    )
    assert recovery(desktop.argvs) == stops_then(['xdotool', 'mouseup', '1'], ['xdotool', 'keyup', 'ctrl'])


async def test_x11_computer_says_what_may_still_be_held_when_releasing_runs_out_of_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    move_on_after = anyio.move_on_after

    def no_time_left(delay: float, shield: bool) -> anyio.CancelScope:
        return move_on_after(0, shield=shield)

    # As if stopping the failed command had used up all the time for releasing.
    monkeypatch.setattr(anyio, 'move_on_after', no_time_left)
    desktop = ScriptedDesktop(CommandResult(exit_code=1, stdout='', stderr='BadValue'))
    with pytest.raises(WorkspaceError, match='may still be held: releasing took longer than 5 seconds'):
        await desktop.screen.act(Click(x=1, y=2))


async def test_x11_computer_raises_a_lost_environment_while_releasing() -> None:
    failed = CommandResult(exit_code=1, stdout='', stderr='BadValue')
    desktop = ScriptedDesktop(
        failed, CommandResult(exit_code=0, stdout='', stderr=''), WorkspaceUnavailableError('gone')
    )
    with pytest.raises(WorkspaceUnavailableError, match='gone'):
        await desktop.screen.act(Click(x=1, y=2))


@pytest.mark.parametrize(
    'action, release',
    [
        (HoldKeys(keys=('shift',), seconds=5), ['xdotool', 'keyup', 'shift']),
        # Each typed key is let go once, however often it was typed.
        (TypeText(text='aa!'), ['xdotool', 'keyup', 'Shift_L', 'U0061', 'U0021']),
        # A scroll, like a click, can stop between pressing and releasing its wheel button.
        (Scroll(direction='down', amount=3), ['xdotool', 'mouseup', '5']),
    ],
)
async def test_x11_computer_releases_what_it_pressed_when_cancelled(action: DisplayAction, release: list[str]) -> None:
    desktop = ScriptedDesktop(block_first=True)
    async with anyio.create_task_group() as tg:
        tg.start_soon(desktop.screen.act, action)
        await desktop.started.wait()
        tg.cancel_scope.cancel()
    # The cancelled command is stopped first, so it can't press again what was released.
    assert recovery(desktop.argvs) == stops_then(release)


async def test_x11_computer_passes_a_zoom_region_as_an_argument() -> None:
    desktop = ScriptedDesktop(PNG_OUTPUT)
    assert isinstance(desktop.screen, SupportsZoom)
    assert await desktop.screen.zoom(Region(left=10, top=20, right=110, bottom=70)) == FAKE_SCREENSHOT
    # The region goes in as an argument, never into the script.
    assert desktop.argvs[0][3:] == ['sh', '-a', '10,20,100,50']


@pytest.mark.parametrize(
    'result, call, message',
    [
        (
            CommandResult(exit_code=1, stdout='', stderr="Can't open display: :1\n"),
            'display_size',
            "exit code 1 on display :1: Can't open display: :1",
        ),
        (CommandResult(exit_code=0, stdout='garbage\n', stderr=''), 'display_size', "output: 'garbage'"),
        (CommandResult(exit_code=0, stdout=' \n' * 3, stderr=''), 'display_size', "output: ''"),
        (
            CommandResult(exit_code=0, stdout='  Width: 65536\n  Height: 800\n', stderr=''),
            'display_size',
            'impossible size: 65536x800',
        ),
        (CommandResult(exit_code=0, stdout='garbage\n', stderr=''), 'cursor_position', 'getmouselocation'),
        (CommandResult(exit_code=0, stdout='bm90IGEgcG5n', stderr=''), 'screenshot', 'did not return a PNG'),
        (CommandResult(exit_code=0, stdout='abc', stderr=''), 'screenshot', 'did not return a PNG'),
    ],
    ids=[
        'display-size-fails',
        'display-size-garbage',
        'display-size-blank',
        'display-size-impossible',
        'pointer-garbage',
        'not-a-png',
        'not-base64',
    ],
)
async def test_x11_computer_failures_raise_workspace_error(result: CommandResult, call: str, message: str) -> None:
    # A failed look starts the display again and looks once more, so it fails twice.
    desktop = ScriptedDesktop(result, result)
    with pytest.raises(WorkspaceError, match=message):
        await getattr(desktop.screen, call)()
    assert desktop.starts == 2


async def test_a_failed_look_starts_the_display_again_and_looks_again() -> None:
    desktop = ScriptedDesktop(CommandResult(exit_code=1, stdout='', stderr='BadWindow'), PNG_OUTPUT)
    assert await desktop.screen.screenshot() == FAKE_SCREENSHOT
    assert desktop.starts == 2


@pytest.mark.parametrize(
    'error', [WorkspaceUnavailableError('the sandbox is gone'), WorkspaceTimeoutError('the display is stuck')]
)
async def test_an_unavailable_or_stuck_environment_is_not_retried(error: WorkspaceError) -> None:
    desktop = ScriptedDesktop(error)
    with pytest.raises(type(error)):
        await desktop.screen.screenshot()
    assert desktop.starts == 1


async def test_concurrent_first_uses_start_one_display() -> None:
    desktop = ScriptedDesktop()
    async with anyio.create_task_group() as tg:
        for _ in range(3):
            tg.start_soon(desktop.screen.act, MouseUp())
    assert desktop.starts == 1


async def test_x11_computer_refuses_an_unknown_action() -> None:
    with pytest.raises(UnsupportedActionError):
        await ScriptedDesktop().screen.act(object())  # pyright: ignore[reportArgumentType]


def test_a_one_colour_screenshot_is_blank() -> None:
    assert is_blank(png(4, 3))
    assert not is_blank(png(4, 3, red_box=(1, 1, 2, 2)))


def test_a_sentence_that_lists_what_a_custom_image_needs_names_every_program_the_display_runs() -> None:
    # Such a sentence, as on the E2B pages, is a build list: a program it leaves out fails on first use.
    root = Path(__file__).parents[3]
    pages = [*root.glob('docs/harness/*.md'), *root.glob('src/pydantic_ai_harness/pydantic_ai_harness/*/README.md')]
    wanted = [*_x11._DISPLAY_PROGRAMS.split(), 'Xvfb']  # pyright: ignore[reportPrivateUsage]
    sentences = [
        (page.name, sentence)
        for page in pages
        for sentence in re.split(r'(?<=\.) ', page.read_text())
        if '`xdotool`' in sentence and '`setsid`' in sentence
    ]
    assert sentences
    assert [
        (name, [p for p in wanted if p not in sentence])
        for name, sentence in sentences
        if any(p not in sentence for p in wanted)
    ] == []


async def test_an_x11_display_starts_xvfb_at_its_size() -> None:
    desktop = ScriptedDesktop(desktop=(8, 6))
    await desktop.screen.act(MouseUp())
    (argv,) = desktop.start_argvs
    assert 'Xvfb' in argv[2] and argv[-1] == '8x6'


async def test_a_display_marked_stopped_is_started_again_on_its_next_use() -> None:
    # A backend marks its display stopped when the sandbox restarted, which stops the X server, as Daytona's and boxd's do.
    sandbox = FakeDisplay()

    async def run(argv: list[str], /, *, env: Mapping[str, str], timeout: float) -> CommandResult:
        command = shlex.join(argv)
        assert sandbox.answers(command)
        stdout, stderr, exit_code = sandbox(command, timeout)
        return CommandResult(exit_code=exit_code, stdout=stdout, stderr=stderr)

    display = _x11.x11_display(run, (1280, 800), str)
    assert display is not None
    assert await display.display_size() == (1280, 800)
    await display.act(Click(x=1, y=2))
    display.mark_stopped()
    assert await display.screenshot() == FAKE_SCREENSHOT

    assert len(sandbox.starts) == 2
    # A fake that sends the display every command gets an empty success for one that isn't the display's.
    assert not sandbox.answers('mkdir -p /work')
    assert sandbox('mkdir -p /work', None) == ('', '', 0)
    (click,) = sandbox.xdotool
    assert click.endswith(' xdotool mousemove 1 2 click --repeat 1 1')
