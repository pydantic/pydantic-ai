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
    HoldKeys,
    MouseUp,
    Point,
    PressKeys,
    ReadOnlyWorkspace,
    Region,
    Scroll,
    SupportsDisplay,
    TypeText,
    UnavailableWorkspace,
    UnsupportedComputerActionError,
    Workspace,
    WorkspaceBackend,
    WorkspaceReadOnlyError,
    WorkspaceRef,
    WorkspaceUnavailableError,
)

from .workspace_fakes import FAKE_SCREENSHOT, FakeWorkspace, InMemoryProvider, ProviderBackend

pytestmark = pytest.mark.anyio


class _DisplayOnly(WorkspaceBackend, SupportsDisplay):
    """A backend with a display but neither zoom nor a pointer position."""

    def __init__(self, inner: ProviderBackend) -> None:
        self._inner = inner

    @property
    def ref(self) -> WorkspaceRef | None:
        return None

    async def working_dir(self) -> str:
        return '/'

    @property
    def has_display(self) -> bool:
        return True

    async def display_size(self) -> tuple[int, int]:
        return await self._inner.display_size()

    async def screenshot(self) -> bytes:
        return await self._inner.screenshot()

    async def act(self, action: ComputerAction) -> None:
        await self._inner.act(action)


def test_actions_normalize_key_names_and_validate_from_json() -> None:
    adapter = TypeAdapter(list[ComputerAction])
    actions = adapter.validate_python(
        [
            {'type': 'keypress', 'keys': ['CTRL', 'Return']},
            # A field the action doesn't have, such as one a provider adds, is ignored.
            {'type': 'click', 'x': 1, 'y': 2, 'modifiers': ['Control', 'cmd', 'Option'], 'scale': 0.5},
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
    assert await workspace.cursor_position() == Point(x=640, y=400)
    assert len(await workspace.zoom(Region(left=0, top=0, right=4, bottom=4))) > 0
    with pytest.raises(WorkspaceReadOnlyError, match='acting on the display'):
        await workspace.act(Click())
    assert provider.actions == []


async def test_a_display_without_zoom_or_cursor_position_refuses_them() -> None:
    workspace = Workspace(_DisplayOnly(InMemoryProvider().backend(None)))
    with pytest.raises(UnsupportedComputerActionError, match='cannot zoom'):
        await workspace.zoom(Region(left=0, top=0, right=4, bottom=4))
    with pytest.raises(UnsupportedComputerActionError, match='cannot report the pointer position'):
        await workspace.cursor_position()


def test_has_display_follows_wrappers_without_io() -> None:
    provider = InMemoryProvider()
    assert Workspace(provider.backend(None)).has_display is True
    assert ReadOnlyWorkspace(Workspace(provider.backend(None))).has_display is True
    assert ReadOnlyWorkspace(Workspace(FakeWorkspace('no-display'))).has_display is False
    assert Workspace(UnavailableWorkspace('no workspace here')).has_display is False


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
