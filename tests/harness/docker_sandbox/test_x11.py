"""Tests for `X11Computer`, which drives the desktop container's X server with `xdotool` and `scrot`."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import anyio
import pytest

from pydantic_ai.workspaces import (
    Click,
    CommandResult,
    ComputerAction,
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
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceUnsupportedActionError,
)
from pydantic_ai_harness.docker_sandbox._x11 import (
    _PRESS_MAPPED,  # pyright: ignore[reportPrivateUsage]
    X11Computer,
    _keysym,  # pyright: ignore[reportPrivateUsage]
)

from ...workspace_fakes import FAKE_SCREENSHOT

pytestmark = pytest.mark.anyio


class ScriptedCommands:
    """Records each command and answers with the next scripted result (exit 0 and no output by default)."""

    def __init__(self, *results: CommandResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[WorkspaceCommand, bool, Mapping[str, str] | None]] = []
        self.timeouts: list[float | None] = []

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        self.calls.append((command, shell, env))
        self.timeouts.append(timeout)
        return self.results.pop(0) if self.results else CommandResult(exit_code=0, stdout='', stderr='')

    @property
    def argvs(self) -> list[WorkspaceCommand]:
        return [command for command, _, _ in self.calls]


class BlockingCommands(ScriptedCommands):
    """Blocks the first command until cancelled, then records the rest."""

    def __init__(self) -> None:
        super().__init__()
        self.started = anyio.Event()

    async def run(self, command: WorkspaceCommand, **kwargs: Any) -> CommandResult:
        if not self.calls:
            self.calls.append((command, False, None))
            self.started.set()
            await anyio.sleep_forever()
        return await super().run(command, **kwargs)


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
    ],
)
async def test_x11_computer_runs_one_xdotool_command_per_action(action: ComputerAction, argv: list[str]) -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(action)
    assert commands.calls == [(argv, False, {'DISPLAY': ':1', 'LC_ALL': 'C.UTF-8'})]


async def test_x11_computer_presses_enter_and_tab_for_newlines_and_tabs_in_typed_text() -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(TypeText(text='l\n\tx\n'))
    assert commands.argvs == [['xdotool', 'key', '--delay', '12', '--', 'U006C', 'Return', 'Tab', 'U0078', 'Return']]


async def test_x11_computer_types_mixed_text_in_one_command_mapping_each_character_outside_ascii_once() -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(TypeText(text='Zoë 日本日 é'))
    ((sh, _, script, _, *args),) = commands.argvs
    assert (sh, script) == ('sh', _PRESS_MAPPED)
    # The characters to map, then `xdotool key`'s arguments: the whole text, ASCII included.
    assert args == ['U00EB', 'U65E5', 'U672C', 'U00E9', '--', '--delay', '12', '--'] + [
        'U005A',
        'U006F',
        'U00EB',
        'U0020',
        'U65E5',
        'U672C',
        'U65E5',
        'U0020',
        'U00E9',
    ]


async def test_x11_computer_maps_at_most_50_characters_a_command() -> None:
    commands = ScriptedCommands()
    text = ''.join(chr(0x4E00 + n) for n in range(51))
    await X11Computer(commands).act(TypeText(text=text))
    first, second = commands.argvs
    assert first[4 : first.index('--')] == [f'U{0x4E00 + n:04X}' for n in range(50)]
    assert second[4:] == ['U4E32', '--', '--delay', '12', '--', 'U4E32']


async def test_x11_computer_maps_a_character_key_outside_ascii_before_pressing_it() -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(PressKeys(keys=('ctrl', 'é')))
    assert commands.argvs[0][4:] == ['U00E9', '--', '--repeat', '1', '--', 'ctrl+U00E9']


@pytest.mark.parametrize(
    'key',
    [
        # xdotool matches command names in any case.
        'Exec',
        # A command only newer xdotool releases have.
        'windowquit',
        '--help',
    ],
)
def test_x11_computer_refuses_key_names_xdotool_would_read_as_commands(key: str) -> None:
    # `xdotool key exec sh -c ...` runs the program: `exec` ends the key names and starts a command. Validation
    # already refuses these names; the display refuses them too, for an action built without it.
    with pytest.raises(WorkspaceUnsupportedActionError, match='cannot press the key'):
        _keysym(key)


@pytest.mark.parametrize(
    'action, releases',
    [
        (Click(x=1, y=2, modifiers=('ctrl', 'shift')), [['xdotool', 'keyup', 'shift', 'ctrl']]),
        (Drag(path=(Point(x=1, y=2), Point(x=3, y=4))), [['xdotool', 'mouseup', '1']]),
        (
            Drag(path=(Point(x=1, y=2), Point(x=3, y=4)), modifiers=('alt',)),
            [['xdotool', 'mouseup', '1', 'keyup', 'alt']],
        ),
        (PressKeys(keys=('ctrl', 's'), repeat=3), [['xdotool', 'keyup', 's', 'ctrl']]),
        # Typing can stop between a key's press and release; X would then repeat the key without end.
        (TypeText(text='Yay'), [['xdotool', 'keyup', 'Shift_L', 'ISO_Level3_Shift', 'U0059', 'U0061', 'U0079']]),
        (Click(x=1, y=2), []),
    ],
)
async def test_x11_computer_releases_what_a_failed_action_held(
    action: ComputerAction, releases: list[list[str]]
) -> None:
    commands = ScriptedCommands(CommandResult(exit_code=1, stdout='', stderr='BadValue'))
    with pytest.raises(WorkspaceError, match='BadValue'):
        await X11Computer(commands).act(action)
    assert commands.argvs[1:] == releases


async def test_x11_computer_releases_held_keys_when_cancelled() -> None:
    commands = BlockingCommands()
    async with anyio.create_task_group() as tg:
        tg.start_soon(X11Computer(commands).act, HoldKeys(keys=('shift',), seconds=5))
        await commands.started.wait()
        tg.cancel_scope.cancel()
    assert commands.argvs == [
        ['xdotool', 'keydown', 'shift', 'sleep', '5.0', 'keyup', 'shift'],
        ['xdotool', 'keyup', 'shift'],
    ]


async def test_x11_computer_releases_typed_keys_when_cancelled() -> None:
    commands = BlockingCommands()
    async with anyio.create_task_group() as tg:
        tg.start_soon(X11Computer(commands).act, TypeText(text='é'))
        await commands.started.wait()
        tg.cancel_scope.cancel()
    assert commands.argvs[1:] == [['xdotool', 'keyup', 'Shift_L', 'ISO_Level3_Shift', 'U00E9']]


async def test_x11_computer_captures_a_scaled_screenshot() -> None:
    commands = ScriptedCommands(
        CommandResult(exit_code=0, stdout=base64.b64encode(FAKE_SCREENSHOT).decode(), stderr='')
    )
    assert await X11Computer(commands).scaled_screenshot(640, 400) == FAKE_SCREENSHOT
    ((sh, _, script, _, size),) = commands.argvs
    assert (sh, size) == ('sh', '640x400')
    assert 'scrot -t "$1"' in script


async def test_x11_computer_reads_the_display_size_and_screenshot_on_its_display() -> None:
    encoded = base64.encodebytes(FAKE_SCREENSHOT).decode()  # wrapped lines, as `base64` prints
    commands = ScriptedCommands(
        CommandResult(exit_code=0, stdout='1280 800\n', stderr=''),
        CommandResult(exit_code=0, stdout=encoded, stderr=''),
    )
    computer = X11Computer(commands)
    assert await computer.display_size() == (1280, 800)
    assert await computer.screenshot() == FAKE_SCREENSHOT
    (geometry, _, env), (capture, _, _) = commands.calls
    assert geometry == ['xdotool', 'getdisplaygeometry']
    assert env == {'DISPLAY': ':1', 'LC_ALL': 'C.UTF-8'}
    assert capture[:2] == ['sh', '-c']
    assert commands.timeouts == [150, 150]


@pytest.mark.parametrize(
    'result, call, message',
    [
        (
            CommandResult(exit_code=1, stdout='', stderr="Can't open display: :1\n"),
            'display_size',
            "exit code 1 on display :1: Can't open display: :1",
        ),
        (CommandResult(exit_code=0, stdout='garbage\n', stderr=''), 'display_size', "output: 'garbage'"),
        (CommandResult(exit_code=127, stdout='', stderr='sh: scrot: not found'), 'screenshot', 'scrot: not found'),
    ],
)
async def test_x11_computer_failures_raise_workspace_error(result: CommandResult, call: str, message: str) -> None:
    computer = X11Computer(ScriptedCommands(result))
    with pytest.raises(WorkspaceError, match=message):
        await getattr(computer, call)()


@pytest.mark.parametrize(
    'output',
    [
        pytest.param('bm90IGEgcG5n', id='not-a-png'),
        pytest.param('abc', id='not-base64'),
    ],
)
async def test_x11_computer_refuses_a_screenshot_that_is_not_a_png(output: str) -> None:
    computer = X11Computer(ScriptedCommands(CommandResult(exit_code=0, stdout=output, stderr='')))
    with pytest.raises(WorkspaceError, match='did not print a base64-encoded PNG'):
        await computer.screenshot()


async def test_x11_computer_refuses_a_key_name_the_display_does_not_know() -> None:
    warning = "(symbol) No such key name 'CAPSLOCK'. Ignoring it."
    commands = ScriptedCommands(CommandResult(exit_code=0, stdout='', stderr=warning))
    with pytest.raises(WorkspaceUnsupportedActionError, match="No such key name 'CAPSLOCK'"):
        await X11Computer(commands).act(PressKeys(keys=('CAPSLOCK',)))


async def test_x11_computer_refuses_an_unknown_action() -> None:
    with pytest.raises(WorkspaceUnsupportedActionError):
        await X11Computer(ScriptedCommands()).act(object())  # pyright: ignore[reportArgumentType]


async def test_x11_computer_zooms_on_a_region_of_its_display() -> None:
    encoded = base64.encodebytes(FAKE_SCREENSHOT).decode()
    commands = ScriptedCommands(CommandResult(exit_code=0, stdout=encoded, stderr=''))
    assert await X11Computer(commands).zoom(Region(left=10, top=20, right=110, bottom=70)) == FAKE_SCREENSHOT
    ((command, _, env),) = commands.calls
    # The region goes in as an argument, never into the script.
    assert (command[:2], command[-1], env) == (['sh', '-c'], '10,20,100,50', {'DISPLAY': ':1', 'LC_ALL': 'C.UTF-8'})


@pytest.mark.parametrize(
    'output, expected',
    [
        pytest.param('X=12\nY=34\nSCREEN=0\nWINDOW=65\n', Point(x=12, y=34), id='position'),
        pytest.param('garbage\n', None, id='unreadable'),
    ],
)
async def test_x11_computer_reads_the_pointer_position(output: str, expected: Point | None) -> None:
    computer = X11Computer(ScriptedCommands(CommandResult(exit_code=0, stdout=output, stderr='')))
    if expected is None:
        with pytest.raises(WorkspaceError, match='getmouselocation'):
            await computer.cursor_position()
    else:
        assert await computer.cursor_position() == expected
