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
    UnsupportedComputerActionError,
    WorkspaceCommand,
    WorkspaceError,
)
from pydantic_ai_harness._x11 import X11Computer

from ..workspace_fakes import FAKE_SCREENSHOT

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
            Drag(path=(Point(x=1, y=2), Point(x=3, y=4), Point(x=5, y=6))),
            ['xdotool', 'mousemove', '1', '2', 'mousedown', '1', 'mousemove', '3', '4', 'mousemove', '5', '6']
            + ['mouseup', '1'],
        ),
        (MouseDown(button='middle'), ['xdotool', 'mousedown', '2']),
        (MouseUp(), ['xdotool', 'mouseup', '1']),
        (Scroll(x=7, y=8, direction='down'), ['xdotool', 'mousemove', '7', '8', 'click', '--repeat', '3', '5']),
        (Scroll(direction='up', amount=2), ['xdotool', 'click', '--repeat', '2', '4']),
        (TypeText(text='-n "hi"\n'), ['xdotool', 'type', '--', '-n "hi"\n']),
        (PressKeys(keys=('ctrl', 's'), repeat=3), ['xdotool', 'key', '--repeat', '3', '--', 'ctrl+s']),
        (PressKeys(keys=('PageDown', 'F5', 'ArrowUp')), ['xdotool', 'key', '--repeat', '1', '--', 'Next+F5+Up']),
        (PressKeys(keys=('shift', 'XF86AudioMute')), ['xdotool', 'key', '--repeat', '1', '--', 'shift+XF86AudioMute']),
        (PressKeys(keys=('ctrl', '+')), ['xdotool', 'key', '--repeat', '1', '--', 'ctrl+plus']),
        (HoldKeys(keys=('shift',), seconds=0.5), ['xdotool', 'keydown', 'shift', 'sleep', '0.5', 'keyup', 'shift']),
    ],
)
async def test_x11_computer_runs_one_xdotool_command_per_action(action: ComputerAction, argv: list[str]) -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(action)
    assert commands.calls == [(argv, False, {'DISPLAY': ':1'})]


@pytest.mark.parametrize(
    'action',
    [
        # xdotool matches command names in any case.
        HoldKeys(keys=('Exec',), seconds=1),
        # A command only newer xdotool releases have.
        PressKeys(keys=('windowquit',)),
        HoldKeys(keys=('--help',), seconds=1),
    ],
)
async def test_x11_computer_refuses_key_names_xdotool_would_read_as_commands(action: ComputerAction) -> None:
    # `xdotool key exec sh -c ...` runs the program: `exec` ends the key names and starts a command.
    commands = ScriptedCommands()
    with pytest.raises(UnsupportedComputerActionError, match='cannot press the key'):
        await X11Computer(commands).act(action)
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


async def test_x11_computer_reads_the_display_size_and_screenshot_on_its_display() -> None:
    encoded = base64.encodebytes(FAKE_SCREENSHOT).decode()  # wrapped lines, as `base64` prints
    commands = ScriptedCommands(
        CommandResult(exit_code=0, stdout='1280 800\n', stderr=''),
        CommandResult(exit_code=0, stdout=encoded, stderr=''),
    )
    computer = X11Computer(commands)
    assert await computer.display_size() == (1280, 800)
    assert await computer.screenshot() == FAKE_SCREENSHOT
    (geometry, _, env), (_, shell, _) = commands.calls
    assert geometry == ['xdotool', 'getdisplaygeometry']
    assert env == {'DISPLAY': ':1'}
    assert shell is True
    assert commands.timeouts == [60, 60]


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
    with pytest.raises(UnsupportedComputerActionError, match="No such key name 'CAPSLOCK'"):
        await X11Computer(commands).act(PressKeys(keys=('CAPSLOCK',)))


async def test_x11_computer_refuses_an_unknown_action() -> None:
    with pytest.raises(UnsupportedComputerActionError):
        await X11Computer(ScriptedCommands()).act(object())  # pyright: ignore[reportArgumentType]


async def test_x11_computer_zooms_on_a_region_of_its_display() -> None:
    encoded = base64.encodebytes(FAKE_SCREENSHOT).decode()
    commands = ScriptedCommands(CommandResult(exit_code=0, stdout=encoded, stderr=''))
    assert await X11Computer(commands).zoom(Region(left=10, top=20, right=110, bottom=70)) == FAKE_SCREENSHOT
    ((command, _, env),) = commands.calls
    # The region goes in as an argument, never into the script.
    assert (command[:2], command[-1], env) == (['sh', '-c'], '10,20,100,50', {'DISPLAY': ':1'})


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


async def test_x11_computer_refuses_a_button_it_does_not_know() -> None:
    # As a button added to `MouseButton` after this display was written.
    action = MouseDown()
    object.__setattr__(action, 'button', 'thumb')
    with pytest.raises(UnsupportedComputerActionError, match="cannot use 'thumb'"):
        await X11Computer(ScriptedCommands()).act(action)
