"""Tests for display control: the actions and the `Workspace` display methods."""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter, ValidationError

from pydantic_ai import UserError
from pydantic_ai.computer_use import ComputerToolAction
from pydantic_ai.workspaces import (
    Click,
    ComputerAction,
    Display,
    HoldKeys,
    MouseUp,
    Point,
    PressKeys,
    ReadOnlyWorkspace,
    Region,
    Scroll,
    SupportsDisplay,
    SupportsZoom,
    TypeText,
    UnavailableWorkspace,
    Workspace,
    WorkspaceReadOnlyError,
    WorkspaceUnavailableError,
    WorkspaceUnsupportedActionError,
)

from .workspace_fakes import FAKE_SCREENSHOT, DisplayOnlyBackend, FakeWorkspace, InMemoryProvider

pytestmark = pytest.mark.anyio


def test_keys_are_a_closed_set_with_x_keysyms_as_aliases() -> None:
    keys = PressKeys(keys=('KP_Enter', 'XF86AudioMute', 'Caps_Lock', 'Print', 'Super_L', 'ctrl+Page_Down')).keys
    assert keys == ('numpad_enter', 'volumemute', 'capslock', 'printscreen', 'super', 'ctrl', 'pagedown')
    with pytest.raises(
        ValidationError, match=r"'Hyper_L' is not a key: use a single character or one of ctrl, .*, f1 to f24"
    ):
        PressKeys(keys=('Hyper_L',))


def test_punctuation_keysyms_and_browser_key_names_are_aliases() -> None:
    keys = PressKeys(keys=('ctrl+plus', 'minus', 'bracketleft', 'question', 'Meta_L', 'NumpadAdd', 'NumpadEnter')).keys
    assert keys == ('ctrl', '+', '-', '[', '?', 'super', 'numpad_add', 'numpad_enter')
    assert PressKeys(keys=('AudioVolumeUp', 'MediaPlayPause', 'ShiftLeft')).keys == ('volumeup', 'mediaplay', 'shift')


def test_fields_an_action_does_not_have_are_refused() -> None:
    # Dropping one would act differently than asked: a `mouse_down` with a position would press at the pointer.
    adapter = TypeAdapter(list[ComputerToolAction])
    for invalid in (
        {'type': 'mouse_down', 'x': 400, 'y': 300},
        {'type': 'wait', 'ms': 200},
        {'type': 'drag', 'path': [{'x': 0, 'y': 0}, {'x': 1, 'y': 1, 'pressure': 0.5}]},
        {'type': 'zoom', 'region': {'left': 0, 'top': 0, 'right': 1, 'bottom': 1, 'scale': 2}},
    ):
        with pytest.raises(ValidationError, match='Unexpected keyword argument'):
            adapter.validate_python([invalid])


def test_actions_normalize_key_names_and_validate_from_json() -> None:
    adapter = TypeAdapter(list[ComputerAction])
    actions = adapter.validate_python(
        [
            {'type': 'keypress', 'keys': ['CTRL', 'Return']},
            {'type': 'click', 'x': 1, 'y': 2, 'modifiers': ['Control', 'cmd', 'Option']},
            {'type': 'hold_key', 'keys': ['ESC'], 'seconds': 1},
            {'type': 'keypress', 'keys': ['ctrl+shift+T']},
            {'type': 'keypress', 'keys': ['+']},
            {'type': 'keypress', 'keys': ['ctrl+shift++']},
        ]
    )
    assert actions == [
        PressKeys(keys=('ctrl', 'enter')),
        Click(x=1, y=2, modifiers=('ctrl', 'super', 'alt')),
        HoldKeys(keys=('escape',), seconds=1),
        PressKeys(keys=('ctrl', 'shift', 't')),
        PressKeys(keys=('+',)),
        PressKeys(keys=('ctrl', 'shift', '+')),
    ]
    assert adapter.validate_json(adapter.dump_json(actions)) == actions
    invalid_actions: list[dict[str, object]] = [
        {'type': 'click', 'x': 10},
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
        lambda: Scroll(direction='down', amount=51),
        lambda: Region(left=5, top=0, right=5, bottom=10),
        lambda: TypeText(text='a\x00b'),
        lambda: PressKeys(keys=('a+b+c+d+e+f+g+h+i',)),
        lambda: HoldKeys(keys=('',), seconds=1),
        # Modifiers are checked before anything runs, so an unknown one can't be skipped mid-action.
        lambda: Click(modifiers=('ctrl', 'exec')),
    ],
)
def test_actions_are_validated_when_built_in_python(build: Any) -> None:
    with pytest.raises(ValidationError):
        build()


def test_typed_text_refuses_control_characters_other_than_newline_and_tab() -> None:
    assert TypeText(text='line 1\n\tline 2').text == 'line 1\n\tline 2'
    # Each would press a key other than Enter or Tab (`\r` a second Enter, `\b` Backspace), or has no key at all.
    for text in ('a\r\nb', 'a\x08b', 'a\x1bb', 'a\x7fb', 'a\x85b', 'a\x9fb'):
        with pytest.raises(ValidationError, match=r'control character U\+00[0-9A-F]{2}: only `\\n` \(Enter\)'):
            TypeText(text=text)


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
    region = Region(left=0, top=0, right=4, bottom=4)
    calls = (
        workspace.display_size(),
        workspace.screenshot(),
        workspace.act(MouseUp()),
        workspace.zoom(region),
        workspace.cursor_position(),
    )
    for call in calls:
        with pytest.raises(WorkspaceUnavailableError, match='no workspace here'):
            await call


async def test_read_only_workspace_allows_screenshots_and_refuses_actions() -> None:
    provider = InMemoryProvider()
    workspace = ReadOnlyWorkspace(Workspace(provider.backend(None)))
    assert await workspace.display_size() == (1280, 800)
    assert await workspace.screenshot() == FAKE_SCREENSHOT
    assert await workspace.cursor_position() == Point(x=640, y=400)
    assert len(await workspace.zoom(Region(left=0, top=0, right=4, bottom=4))) > 0
    with pytest.raises(WorkspaceReadOnlyError, match='acting on the display'):
        await workspace.act(Click())
    assert provider.actions == []


async def test_a_display_without_zoom_or_cursor_position_refuses_them() -> None:
    provider = InMemoryProvider()
    workspace = Workspace(DisplayOnlyBackend(provider.backend(None)))
    assert await workspace.display_size() == (1280, 800)
    assert await workspace.screenshot() == FAKE_SCREENSHOT
    await workspace.act(MouseUp())
    assert provider.actions == [MouseUp()]
    with pytest.raises(WorkspaceUnsupportedActionError, match='cannot zoom'):
        await workspace.zoom(Region(left=0, top=0, right=4, bottom=4))
    with pytest.raises(WorkspaceUnsupportedActionError, match='cannot report the pointer position'):
        await workspace.cursor_position()


async def test_wrappers_keep_their_policy_and_the_backend_display() -> None:
    provider = InMemoryProvider()
    backend = DisplayOnlyBackend(provider.backend(None))
    read_only = Workspace(ReadOnlyWorkspace(Workspace(backend)))
    assert read_only.has_display is True
    assert await read_only.screenshot() == FAKE_SCREENSHOT
    with pytest.raises(WorkspaceReadOnlyError):
        await read_only.act(MouseUp())
    assert provider.actions == []
    # Capability and identity are the backend's: a workspace never poses as a display that can zoom.
    assert not isinstance(read_only, SupportsDisplay)
    assert read_only.backend is backend and not isinstance(backend.display, SupportsZoom)
    assert ReadOnlyWorkspace(Workspace(FakeWorkspace('no-display'))).has_display is False
    assert Workspace(UnavailableWorkspace('no workspace here')).has_display is False


async def test_a_backend_can_take_its_display_from_another_object() -> None:
    provider = InMemoryProvider()
    screen = provider.backend(None)

    class LocalFilesWithAScreen(FakeWorkspace):
        @property
        def display(self) -> Display:
            return screen

    workspace = Workspace(LocalFilesWithAScreen('local'))
    await workspace.act(TypeText(text='hi'))
    assert await workspace.cursor_position() == Point(x=640, y=400)
    assert provider.actions == [TypeText(text='hi')]


async def test_an_attribute_named_display_that_is_not_a_display_does_not_count() -> None:
    class XServerBackend(FakeWorkspace):
        display = ':1'

    workspace = Workspace(XServerBackend('x-server'))
    assert workspace.has_display is False
    with pytest.raises(UserError, match='has no display to control'):
        await workspace.screenshot()


def test_the_computer_tool_schema_requires_type_first_and_only_declared_fields() -> None:
    schema = TypeAdapter(list[ComputerToolAction]).json_schema()['$defs']
    assert sorted(schema) == snapshot(
        [
            'Click',
            'CursorPosition',
            'Drag',
            'HoldKeys',
            'MouseDown',
            'MouseUp',
            'MoveMouse',
            'Point',
            'PressKeys',
            'Region',
            'Screenshot',
            'Scroll',
            'TypeText',
            'Wait',
            'Zoom',
        ]
    )
    assert all(
        definition['required'][0] == 'type' for name, definition in schema.items() if name not in ('Point', 'Region')
    )
    # Every required name must exist: Gemini rejects a schema that requires an undeclared property.
    assert all(set(definition['required']) <= set(definition['properties']) for definition in schema.values())
