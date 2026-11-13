"""Tests for the desktop display's input and capture: `xdotool` and `scrot` commands, run through a backend's commands.

`ScriptedDesktop` is a desktop `DockerContainerBackend` whose commands answer from a script instead of a container.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping

import anyio
import pytest

from pydantic_ai.computer_use import (
    Click,
    Drag,
    HoldKeys,
    MouseDown,
    MouseUp,
    MoveMouse,
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
from pydantic_ai_harness.computer_use import Display, DisplayAction, SupportsZoom, UnsupportedActionError
from pydantic_ai_harness.docker_container import DockerContainerBackend

from .display_fakes import FAKE_SCREENSHOT

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


RELEASE_MODIFIERS = [
    *['keyup', 'Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R', 'Super_L', 'Super_R'],
    'ISO_Level3_Shift',
]
"""What every acting `xdotool` chain starts with: letting go of modifiers an earlier action may have left down."""

PNG_OUTPUT = CommandResult(exit_code=0, stdout=base64.encodebytes(FAKE_SCREENSHOT).decode(), stderr='')
"""A capture's output: the screenshot in wrapped base64 lines, as `base64` prints it."""


class ScriptedDesktop(DockerContainerBackend):
    """Records each display command and answers with the next scripted result (exit 0 and no output by default).

    Starting the display always succeeds and is only counted. With `block_first`, the first command blocks until
    cancelled.
    """

    def __init__(self, *results: CommandResult | Exception, block_first: bool = False) -> None:
        super().__init__(image='my-desktop', desktop=True)
        self.results = list(results)
        self.calls: list[tuple[WorkspaceCommand, bool, Mapping[str, str] | None]] = []
        self.timeouts: list[float | None] = []
        self.starts = 0
        self.block_first = block_first
        self.started = anyio.Event()
        assert self.display is not None
        self.screen: Display = self.display

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        await anyio.lowlevel.checkpoint()  # as a real command does, so a deadline can cut it short
        if 'Xvfb' in str(command):
            self.starts += 1
            await anyio.sleep(0)  # a start takes a moment, so concurrent first uses overlap
            return CommandResult(exit_code=0, stdout='', stderr='')
        self.calls.append((command, shell, env))
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
        return [command for command, _, _ in self.calls]


@pytest.mark.parametrize(
    'action, argv',
    [
        (Click(x=10, y=20), ['xdotool', 'mousemove', '10', '20', 'click', '--repeat', '1', '1']),
        (Click(count=2, button='right'), ['xdotool', 'click', '--repeat', '2', '3']),
        (
            Click(x=1, y=2, button='back', modifiers=('ctrl', 'shift')),
            ['xdotool', 'keydown', 'ctrl', 'shift', 'mousemove', '1', '2', 'click', '--repeat', '1', '8']
            + ['keyup', 'shift', 'ctrl'],
        ),
        (MoveMouse(x=5, y=6), ['xdotool', 'mousemove', '5', '6']),
        (
            Drag(path=(Point(x=0, y=0), Point(x=500, y=0), Point(x=500, y=10))),
            # Each leg is moved along in steps of at most 200 pixels, and at least two.
            ['xdotool', 'mousemove', '0', '0', 'mousedown', '1', 'sleep', '0.1']
            + ['sleep', '0.05', 'mousemove', '166', '0', 'sleep', '0.05', 'mousemove', '333', '0']
            + ['sleep', '0.05', 'mousemove', '500', '0', 'sleep', '0.05', 'mousemove', '500', '5']
            + ['sleep', '0.05', 'mousemove', '500', '10', 'sleep', '0.1', 'mouseup', '1'],
        ),
        (MouseDown(button='middle'), ['xdotool', 'mousedown', '2']),
        (MouseUp(), ['xdotool', 'mouseup', '1']),
        (
            Scroll(x=7, y=8, direction='down'),
            ['xdotool', 'mousemove', '7', '8', 'click', '--repeat', '3', '--delay', '20', '5'],
        ),
        (Scroll(direction='up', amount=2), ['xdotool', 'click', '--repeat', '2', '--delay', '20', '4']),
        (
            TypeText(text='-n "hi"'),
            ['xdotool', 'key', '--delay', '12', '--', 'U002D', 'U006E', 'U0020', 'U0022', 'U0068', 'U0069', 'U0022'],
        ),
        (PressKeys(keys=('ctrl', 's'), repeat=3), ['xdotool', 'key', '--repeat', '3', '--', 'ctrl+s']),
        (PressKeys(keys=('PageDown', 'F5', 'ArrowUp')), ['xdotool', 'key', '--repeat', '1', '--', 'Next+F5+Up']),
        (PressKeys(keys=('shift', 'XF86AudioMute')), ['xdotool', 'key', '--repeat', '1', '--', 'shift+XF86AudioMute']),
        (
            PressKeys(keys=('printscreen', 'capslock', 'numpad_enter')),
            ['xdotool', 'key', '--repeat', '1', '--', 'Print+Caps_Lock+KP_Enter'],
        ),
        (PressKeys(keys=('ctrl', '+')), ['xdotool', 'key', '--repeat', '1', '--', 'ctrl+plus']),
        # A character key presses what the layout needs to type it: xdotool adds shift for `?`.
        (PressKeys(keys=('ctrl', '?')), ['xdotool', 'key', '--repeat', '1', '--', 'ctrl+U003F']),
        (PressKeys(keys=('@',)), ['xdotool', 'key', '--repeat', '1', '--', 'U0040']),
        (HoldKeys(keys=('shift',), seconds=0.5), ['xdotool', 'keydown', 'shift', 'sleep', '0.5', 'keyup', 'shift']),
        # Enter and Tab for newlines and tabs in typed text.
        (
            TypeText(text='l\n\tx\n'),
            ['xdotool', 'key', '--delay', '12', '--', 'U006C', 'Return', 'Tab', 'U0078', 'Return'],
        ),
    ],
)
async def test_x11_computer_runs_one_xdotool_command_per_action(action: DisplayAction, argv: list[str]) -> None:
    desktop = ScriptedDesktop()
    await desktop.screen.act(action)
    assert desktop.calls == [
        (['sh', '-c', INPUT, 'sh', 'xdotool', *RELEASE_MODIFIERS, *argv[1:]], False, {'DISPLAY': ':1'})
    ]


def fits_the_providers(argv: WorkspaceCommand) -> bool:
    """Whether Sprites and Modal take `argv`, as measured: Sprites puts each argument in its exec URL after `&cmd=` and
    ran a command of one 64,984-byte argument; Modal takes arguments adding up to 65,536 bytes."""
    return sum(len(arg) + len('&cmd=') for arg in argv) <= 64_984


async def test_x11_computer_types_long_text_in_commands_the_providers_take() -> None:
    desktop = ScriptedDesktop()
    await desktop.screen.act(TypeText(text='a' * 10_909))
    typed = [argv[argv.index('--') + 1 :] for argv in desktop.argvs]
    assert [len(keysyms) for keysyms in typed] == [5_454, 5_454, 1]
    assert {keysym for keysyms in typed for keysym in keysyms} == {'U0061'}
    assert all(fits_the_providers(argv) for argv in desktop.argvs)


LONG_DRAG = Drag(path=tuple(Point(x=100 + i % 2 * 300, y=100 + i) for i in range(1_500)), modifiers=('alt',))
"""A drag too long for one command: on Sprites, one command for it was refused (HTTP 414)."""


async def test_x11_computer_splits_a_long_drag_holding_the_button_across_commands() -> None:
    desktop = ScriptedDesktop()
    await desktop.screen.act(LONG_DRAG)
    chains = [argv[argv.index('xdotool') + 1 :] for argv in desktop.argvs]
    assert len(chains) > 1
    assert all(fits_the_providers(argv) for argv in desktop.argvs)
    # Pressed in the first, let go in the last: X keeps the button and the key down between commands.
    pressed = ['keydown', 'alt', 'mousemove', '100', '100', 'mousedown', '1']
    assert chains[0][: len(RELEASE_MODIFIERS) + len(pressed)] == [*RELEASE_MODIFIERS, *pressed]
    assert chains[-1][-6:] == ['sleep', '0.1', 'mouseup', '1', 'keyup', 'alt']
    held = [arg for chain in chains for arg in chain if arg in ('mousedown', 'mouseup', 'keydown', 'keyup')]
    assert held == ['keyup', 'keydown', 'mousedown', 'mouseup', 'keyup']
    # Every step of every leg, ending on the last point: each 300-pixel leg is two steps.
    moves = [chain[i + 1 : i + 3] for chain in chains for i, arg in enumerate(chain) if arg == 'mousemove']
    assert len(moves) == 1 + 2 * 1_499
    assert moves[-1] == ['400', '1599']


async def test_x11_computer_lets_go_when_a_later_command_of_a_long_drag_fails() -> None:
    desktop = ScriptedDesktop(
        CommandResult(exit_code=0, stdout='', stderr=''), CommandResult(exit_code=1, stdout='', stderr='BadValue')
    )
    with pytest.raises(WorkspaceError, match='BadValue'):
        await desktop.screen.act(LONG_DRAG)
    assert recovery(desktop.argvs[1:]) == stops_then(['xdotool', 'mouseup', '1'], ['xdotool', 'keyup', 'alt'])


@pytest.mark.parametrize(
    'action, timeout',
    [
        (HoldKeys(keys=('shift',), seconds=200), 350),
        (Scroll(direction='down', amount=1_000), 182),
        (PressKeys(keys=('a',), repeat=1_000), 162),
        (PressKeys(keys=('ctrl', 'shift', 'a'), repeat=1_000), 186),
        (TypeText(text='a' * 1_000), 162),
        # 0.1 s before the first move and after the last, and 0.05 s before each of the 2 moves.
        (Drag(path=(Point(x=0, y=0), Point(x=10, y=0))), 150.3),
        (Click(x=1, y=1), 150),
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
    [Scroll(direction='down', amount=2**31), PressKeys(keys=('a',), repeat=10**400)],
    ids=['scroll', 'keypress'],
)
async def test_x11_computer_refuses_a_count_xdotool_cannot_take(action: DisplayAction) -> None:
    desktop = ScriptedDesktop()
    with pytest.raises(UnsupportedActionError, match='at most 2147483647 times'):
        await desktop.screen.act(action)
    assert desktop.calls == []


async def test_x11_computer_refuses_key_names_xdotool_would_read_as_commands() -> None:
    # `xdotool key exec sh -c ...` runs the program: `exec` ends the key names and starts a command. Validation
    # already refuses these names; the display refuses them too, for an action built without it.
    unvalidated = PressKeys(keys=('a',))
    object.__setattr__(unvalidated, 'keys', ('Exec',))
    desktop = ScriptedDesktop()
    with pytest.raises(UnsupportedActionError, match='cannot press the key'):
        await desktop.screen.act(unvalidated)
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
            stops_then(['xdotool', 'keyup', 'Shift_L', 'ISO_Level3_Shift', 'U0059', 'U0061', 'U0079']),
            id='type',
        ),
        # A click can stop between pressing and releasing its button.
        pytest.param(Click(x=1, y=2, button='right'), stops_then(['xdotool', 'mouseup', '3']), id='click'),
        pytest.param(MoveMouse(x=1, y=2), stops_then(), id='move'),
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
    with pytest.raises(WorkspaceError, match='may still be held: releasing took longer than 10 seconds'):
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
        (TypeText(text='aa!'), ['xdotool', 'keyup', 'Shift_L', 'ISO_Level3_Shift', 'U0061', 'U0021']),
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
        # A line's leading whitespace can't run into the next line, so this blank output is read in time linear in its
        # length, not its square.
        (CommandResult(exit_code=0, stdout=' \n' * 100_000, stderr=''), 'display_size', "output: ''"),
        # An X screen is at most 65535 pixels wide, so a larger size can't make a drag move through millions of points.
        (
            CommandResult(exit_code=0, stdout='Width: 1000000\nHeight: 800\n', stderr=''),
            'display_size',
            'Unexpected display size output',
        ),
        (CommandResult(exit_code=0, stdout='garbage\n', stderr=''), 'cursor_position', 'getmouselocation'),
        (CommandResult(exit_code=0, stdout='bm90IGEgcG5n', stderr=''), 'screenshot', 'did not return a PNG'),
        (CommandResult(exit_code=0, stdout='abc', stderr=''), 'screenshot', 'did not return a PNG'),
    ],
    ids=[
        'display-size-fails',
        'display-size-garbage',
        'display-size-blank',
        'display-size-too-wide',
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
