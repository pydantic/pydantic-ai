"""Tests for `X11Computer`, which drives an X server with `xdotool` and `scrot`."""

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
from pydantic_ai_harness._x11 import X11Computer

from ..workspace_fakes import FAKE_SCREENSHOT

pytestmark = pytest.mark.anyio

RELEASE_MODIFIERS = [
    *['keyup', 'Shift_L', 'Shift_R', 'Control_L', 'Control_R', 'Alt_L', 'Alt_R', 'Super_L', 'Super_R'],
    'ISO_Level3_Shift',
]
"""What every acting `xdotool` chain starts with: letting go of modifiers an earlier action may have left down."""


class ScriptedCommands:
    """Records each command and answers with the next scripted result (exit 0 and no output by default)."""

    def __init__(self, *results: CommandResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[WorkspaceCommand, bool, Mapping[str, str] | None]] = []

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        self.calls.append((command, shell, env))
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
        # Enter and Tab for newlines and tabs in typed text.
        (
            TypeText(text='l\n\tx\n'),
            ['xdotool', 'key', '--delay', '12', '--', 'U006C', 'Return', 'Tab', 'U0078', 'Return'],
        ),
    ],
)
async def test_x11_computer_runs_one_xdotool_command_per_action(action: ComputerAction, argv: list[str]) -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(action)
    assert commands.calls == [
        (['xdotool', *RELEASE_MODIFIERS, *argv[1:]], False, {'DISPLAY': ':1', 'LC_ALL': 'C.UTF-8'})
    ]


async def test_x11_computer_types_mixed_text_in_one_command_mapping_each_character_outside_ascii_once() -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(TypeText(text='Zoë 日本日 é'))
    ((*_, args),) = [(argv[:4], argv[4:]) for argv in commands.argvs]  # after `sh -c <script> sh`
    # The characters to map, then `xdotool`'s arguments: the whole text, ASCII included.
    assert args == ['U00EB', 'U65E5', 'U672C', 'U00E9', '--', *RELEASE_MODIFIERS, 'key', '--delay', '12', '--'] + [
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
    assert second[4:] == ['U4E32', '--', *RELEASE_MODIFIERS, 'key', '--delay', '12', '--', 'U4E32']


async def test_x11_computer_maps_a_character_key_outside_ascii_before_pressing_it() -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(PressKeys(keys=('ctrl', 'é')))
    assert commands.argvs[0][4:] == ['U00E9', '--', *RELEASE_MODIFIERS, 'key', '--repeat', '1', '--', 'ctrl+U00E9']


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
async def test_x11_computer_refuses_key_names_xdotool_would_read_as_commands(key: str) -> None:
    # `xdotool key exec sh -c ...` runs the program: `exec` ends the key names and starts a command. Validation
    # already refuses these names; the display refuses them too, for an action built without it.
    unvalidated = PressKeys(keys=('a',))
    object.__setattr__(unvalidated, 'keys', (key,))
    commands = ScriptedCommands()
    with pytest.raises(WorkspaceUnsupportedActionError, match='cannot press the key'):
        await X11Computer(commands).act(unvalidated)
    assert commands.calls == []


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


@pytest.mark.parametrize(
    'action, release',
    [
        (HoldKeys(keys=('shift',), seconds=5), ['xdotool', 'keyup', 'shift']),
        (TypeText(text='é'), ['xdotool', 'keyup', 'Shift_L', 'ISO_Level3_Shift', 'U00E9']),
    ],
)
async def test_x11_computer_releases_what_it_pressed_when_cancelled(action: ComputerAction, release: list[str]) -> None:
    commands = BlockingCommands()
    async with anyio.create_task_group() as tg:
        tg.start_soon(X11Computer(commands).act, action)
        await commands.started.wait()
        tg.cancel_scope.cancel()
    assert commands.argvs[1:] == [release]


async def test_x11_computer_looks_return_the_png_and_pass_their_size_or_region_as_arguments() -> None:
    encoded = base64.encodebytes(FAKE_SCREENSHOT).decode()  # wrapped lines, as `base64` prints
    png = CommandResult(exit_code=0, stdout=encoded, stderr='')
    commands = ScriptedCommands(CommandResult(exit_code=0, stdout='1280 800\n', stderr=''), png, png, png)
    computer = X11Computer(commands)

    assert await computer.display_size() == (1280, 800)
    assert await computer.screenshot() == FAKE_SCREENSHOT
    assert await computer.scaled_screenshot(640, 400) == FAKE_SCREENSHOT
    assert await computer.zoom(Region(left=10, top=20, right=110, bottom=70)) == FAKE_SCREENSHOT
    # The size and region go in as arguments, never into the script.
    assert [command[-1] for command in commands.argvs[2:]] == ['640x400', '10,20,100,50']
    assert all(env == {'DISPLAY': ':1', 'LC_ALL': 'C.UTF-8'} for _, _, env in commands.calls)


async def test_x11_display_size_is_the_root_window_spanning_every_monitor() -> None:
    # Two 1920x1080 monitors side by side: a screenshot is 3840x1080, and so must the reported size be.
    xwininfo = (
        'xwininfo: Window id: 0x3c5 (the root window) (has no name)\n\n  Width: 3840\n  Height: 1080\n  Depth: 24\n'
    )
    commands = ScriptedCommands(CommandResult(exit_code=0, stdout=xwininfo, stderr=''))
    assert await X11Computer(commands).display_size() == (3840, 1080)


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
        (CommandResult(exit_code=0, stdout='garbage\n', stderr=''), 'cursor_position', 'getmouselocation'),
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
    with pytest.raises(WorkspaceError, match='did not return a PNG screenshot'):
        await computer.screenshot()


async def test_x11_computer_refuses_a_key_name_the_display_does_not_know() -> None:
    warning = "(symbol) No such key name 'CAPSLOCK'. Ignoring it."
    commands = ScriptedCommands(CommandResult(exit_code=0, stdout='', stderr=warning))
    with pytest.raises(WorkspaceUnsupportedActionError, match="No such key name 'CAPSLOCK'"):
        await X11Computer(commands).act(PressKeys(keys=('CAPSLOCK',)))


async def test_x11_computer_refuses_an_unknown_action() -> None:
    with pytest.raises(WorkspaceUnsupportedActionError):
        await X11Computer(ScriptedCommands()).act(object())  # pyright: ignore[reportArgumentType]


async def test_x11_computer_reads_the_pointer_position() -> None:
    output = 'X=12\nY=34\nSCREEN=0\nWINDOW=65\n'
    computer = X11Computer(ScriptedCommands(CommandResult(exit_code=0, stdout=output, stderr='')))
    assert await computer.cursor_position() == Point(x=12, y=34)
