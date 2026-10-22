"""Tests for display control: the actions, `X11Computer`, and the `Workspace` display methods."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any

import anyio
import pytest
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import UserError
from pydantic_ai.workspaces import (
    Click,
    CommandResult,
    ComputerAction,
    Drag,
    HoldKey,
    KeyPress,
    MouseDown,
    MouseUp,
    MoveMouse,
    Point,
    ReadOnlyWorkspace,
    Scroll,
    TypeText,
    UnavailableWorkspace,
    UnsupportedComputerActionError,
    Workspace,
    WorkspaceCommand,
    WorkspaceError,
    WorkspaceReadOnlyError,
    WorkspaceUnavailableError,
    X11Computer,
)

from .workspace_fakes import FAKE_SCREENSHOT, FakeWorkspace, InMemoryProvider

pytestmark = pytest.mark.anyio


class ScriptedCommands:
    """Records each command and answers with the next scripted result (exit 0 and no output by default)."""

    def __init__(self, *results: CommandResult) -> None:
        self.results = list(results)
        self.calls: list[tuple[WorkspaceCommand, bool, Mapping[str, str] | None]] = []
        self.timeouts: list[tuple[WorkspaceCommand, bool, float | None]] = []

    async def run(
        self,
        command: WorkspaceCommand,
        *,
        shell: bool = False,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        self.calls.append((command, shell, env))
        self.timeouts.append((command, shell, timeout))
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
        (
            Scroll(direction='right', amount=1, modifiers=('shift',)),
            ['xdotool', 'keydown', 'shift', 'click', '--repeat', '1', '7', 'keyup', 'shift'],
        ),
        (Scroll(direction='left', amount=4), ['xdotool', 'click', '--repeat', '4', '6']),
        (TypeText(text='-n "hi"\n'), ['xdotool', 'type', '--', '-n "hi"\n']),
        (KeyPress(keys=('ctrl', 's')), ['xdotool', 'key', '--repeat', '1', '--', 'ctrl+s']),
        (KeyPress(keys=('Return',), repeat=3), ['xdotool', 'key', '--repeat', '3', '--', 'Return']),
        (KeyPress(keys=('PageDown', 'F5', 'ArrowUp')), ['xdotool', 'key', '--repeat', '1', '--', 'Next+F5+Up']),
        (KeyPress(keys=('KP_Enter',)), ['xdotool', 'key', '--repeat', '1', '--', 'KP_Enter']),
        (KeyPress(keys=('shift', 'XF86AudioMute')), ['xdotool', 'key', '--repeat', '1', '--', 'shift+XF86AudioMute']),
        (KeyPress(keys=('ctrl', '+')), ['xdotool', 'key', '--repeat', '1', '--', 'ctrl+plus']),
        (HoldKey(keys=('shift',), seconds=0.5), ['xdotool', 'keydown', 'shift', 'sleep', '0.5', 'keyup', 'shift']),
    ],
)
async def test_x11_computer_runs_one_xdotool_command_per_action(action: ComputerAction, argv: list[str]) -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(action)
    assert commands.calls == [(argv, False, {'DISPLAY': ':1'})]


@pytest.mark.parametrize(
    'action',
    [
        KeyPress(keys=('exec',)),
        KeyPress(keys=('ctrl', 'a b')),
        Click(modifiers=('ctrl', 'exec')),
        HoldKey(keys=('--help',), seconds=1),
        KeyPress(keys=('é',)),
    ],
)
async def test_x11_computer_refuses_key_names_xdotool_would_read_as_commands(action: ComputerAction) -> None:
    # `xdotool key exec sh -c ...` runs the program: `exec` ends the key names and starts a command.
    commands = ScriptedCommands()
    with pytest.raises(UnsupportedComputerActionError, match='cannot press the key'):
        await X11Computer(commands).act(action)
    assert commands.calls == []


@pytest.mark.parametrize(
    'action, release',
    [
        (Click(x=1, y=2, modifiers=('ctrl', 'shift')), ['xdotool', 'keyup', 'shift', 'ctrl']),
        (Drag(path=(Point(x=1, y=2), Point(x=3, y=4))), ['xdotool', 'mouseup', '1']),
        (
            Drag(path=(Point(x=1, y=2), Point(x=3, y=4)), modifiers=('alt',)),
            ['xdotool', 'mouseup', '1', 'keyup', 'alt'],
        ),
    ],
)
async def test_x11_computer_releases_what_a_failed_action_held(action: ComputerAction, release: list[str]) -> None:
    commands = ScriptedCommands(CommandResult(exit_code=1, stdout='', stderr='BadValue'))
    with pytest.raises(WorkspaceError, match='BadValue'):
        await X11Computer(commands).act(action)
    assert commands.argvs[1:] == [release]


async def test_x11_computer_releases_held_keys_when_cancelled() -> None:
    commands = BlockingCommands()
    async with anyio.create_task_group() as tg:
        tg.start_soon(X11Computer(commands).act, HoldKey(keys=('shift',), seconds=5))
        await commands.started.wait()
        tg.cancel_scope.cancel()
    assert commands.argvs == [
        ['xdotool', 'keydown', 'shift', 'sleep', '5.0', 'keyup', 'shift'],
        ['xdotool', 'keyup', 'shift'],
    ]


async def test_x11_computer_does_not_release_after_an_action_without_held_input() -> None:
    commands = ScriptedCommands(CommandResult(exit_code=1, stdout='', stderr='BadValue'))
    with pytest.raises(WorkspaceError):
        await X11Computer(commands).act(Click(x=1, y=2))
    assert len(commands.calls) == 1


async def test_x11_computer_reads_the_display_size_and_screenshot_on_its_display() -> None:
    encoded = base64.encodebytes(FAKE_SCREENSHOT).decode()  # wrapped lines, as `base64` prints
    commands = ScriptedCommands(
        CommandResult(exit_code=0, stdout='1280 800\n', stderr=''),
        CommandResult(exit_code=0, stdout=encoded, stderr=''),
    )
    computer = X11Computer(commands, display=':99')
    assert await computer.display_size() == (1280, 800)
    assert await computer.screenshot() == FAKE_SCREENSHOT
    (geometry, _, env), (screenshot, shell, _) = commands.calls
    assert geometry == ['xdotool', 'getdisplaygeometry']
    assert env == {'DISPLAY': ':99'}
    assert shell is True
    assert screenshot == (
        'd=$(mktemp -d) && scrot "$d/screen.png" && base64 "$d/screen.png"; status=$?; rm -rf "$d"; exit $status'
    )


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
        pytest.param('abc', id='bad-padding'),
        pytest.param('iVBO*w0KGgo=', id='not-base64'),
        pytest.param('iVBORw0KGgo\u00e9', id='not-ascii'),
        pytest.param(base64.b64encode(FAKE_SCREENSHOT[:-12]).decode(), id='truncated'),
    ],
)
async def test_x11_computer_refuses_a_screenshot_that_is_not_a_complete_png(output: str) -> None:
    computer = X11Computer(ScriptedCommands(CommandResult(exit_code=0, stdout=output, stderr='')))
    with pytest.raises(WorkspaceError, match='did not print a complete base64-encoded PNG'):
        await computer.screenshot()


async def test_x11_computer_refuses_an_unknown_action() -> None:
    with pytest.raises(UnsupportedComputerActionError):
        await X11Computer(ScriptedCommands()).act(object())  # pyright: ignore[reportArgumentType]


def test_actions_normalize_key_names_and_validate_from_json() -> None:
    adapter = TypeAdapter(list[ComputerAction])
    actions = adapter.validate_python(
        [
            {'type': 'keypress', 'keys': ['CTRL', 'Return']},
            {'type': 'click', 'x': 1, 'y': 2, 'modifiers': ['Control', 'cmd', 'Option'], 'scale': 0.5},
            {'type': 'hold_key', 'keys': ['ESC'], 'seconds': 1},
            {'type': 'keypress', 'keys': ['ctrl+shift+T']},
            {'type': 'keypress', 'keys': ['+']},
        ]
    )
    assert actions == [
        KeyPress(keys=('ctrl', 'enter')),
        Click(x=1, y=2, modifiers=('ctrl', 'super', 'alt')),
        HoldKey(keys=('escape',), seconds=1),
        KeyPress(keys=('ctrl', 'shift', 't')),
        KeyPress(keys=('+',)),
    ]
    assert adapter.validate_json(adapter.dump_json(actions)) == actions
    invalid_actions: list[dict[str, object]] = [
        {'type': 'click', 'count': 4},
        {'type': 'drag', 'path': [{'x': 1, 'y': 2}]},
        {'type': 'keypress', 'keys': []},
        {'type': 'click', 'x': 10},
        {'type': 'scroll', 'y': 10, 'direction': 'down'},
        {'type': 'scroll', 'x': 1, 'y': 2},
        {'type': 'teleport'},
        {'x': 1, 'y': 2},
    ]
    for invalid in invalid_actions:
        with pytest.raises(ValidationError):
            adapter.validate_python([invalid])


@pytest.mark.parametrize(
    'build',
    [
        lambda: Click(count=0),
        lambda: Click(x=-1, y=0),
        lambda: Drag(path=()),
        lambda: KeyPress(keys=()),
        lambda: HoldKey(keys=('shift',), seconds=float('inf')),
        lambda: HoldKey(keys=('shift',), seconds=11),
        lambda: KeyPress(keys=('a',), repeat=51),
        lambda: KeyPress(keys=('a',) * 9),
        lambda: Scroll(direction='down', amount=51),
        lambda: TypeText(text='x' * 2_001),
        lambda: Drag(path=(Point(x=0, y=0),) * 51),
    ],
)
def test_actions_are_validated_when_built_in_python(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()


async def test_workspace_display_methods_reach_a_backend_with_a_display() -> None:
    provider = InMemoryProvider()
    workspace = Workspace(provider.backend(None))
    assert await workspace.display_size() == (1280, 800)
    assert await workspace.screenshot() == FAKE_SCREENSHOT
    await workspace.act(TypeText(text='hi'))
    assert provider.actions == [TypeText(text='hi')]


async def test_workspace_without_a_display_says_how_to_get_one() -> None:
    workspace = Workspace(FakeWorkspace('no-display'))
    for call in (workspace.display_size(), workspace.screenshot(), workspace.act(MouseUp())):
        with pytest.raises(UserError, match='has no display to control'):
            await call


async def test_unavailable_workspace_reports_its_reason_for_display_calls() -> None:
    workspace = Workspace(UnavailableWorkspace('no workspace here'))
    for call in (workspace.display_size(), workspace.screenshot(), workspace.act(MouseUp())):
        with pytest.raises(WorkspaceUnavailableError, match='no workspace here'):
            await call


async def test_read_only_workspace_allows_screenshots_and_refuses_actions() -> None:
    provider = InMemoryProvider()
    workspace = ReadOnlyWorkspace(Workspace(provider.backend(None)))
    assert await workspace.display_size() == (1280, 800)
    assert await workspace.screenshot() == FAKE_SCREENSHOT
    with pytest.raises(WorkspaceReadOnlyError, match='acting on the display'):
        await workspace.act(Click())
    assert provider.actions == []


def test_has_display_follows_wrappers_without_io() -> None:
    provider = InMemoryProvider()
    assert Workspace(provider.backend(None)).has_display is True
    assert ReadOnlyWorkspace(Workspace(provider.backend(None))).has_display is True
    assert ReadOnlyWorkspace(Workspace(FakeWorkspace('no-display'))).has_display is False
    assert Workspace(UnavailableWorkspace('no workspace here')).has_display is False
    assert provider.log == []


def test_the_schema_tells_the_model_what_each_field_means() -> None:
    schema = TypeAdapter(ComputerAction).json_schema()['$defs']
    assert all(definition['required'][0] == 'type' for name, definition in schema.items() if name != 'Point')
    assert schema['Scroll']['properties']['direction']['description'] == (
        'Which way the content moves into view: `down` shows what is below.'
    )
    assert schema['KeyPress']['properties']['keys']['description'].startswith('One key per entry')


async def test_x11_computer_bounds_every_command() -> None:
    commands = ScriptedCommands()
    await X11Computer(commands).act(TypeText(text='hi'))
    ((_, _, timeout),) = commands.timeouts
    assert timeout == 60
