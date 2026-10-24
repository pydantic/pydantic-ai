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
    PressKeys,
    ReadOnlyWorkspace,
    Scroll,
    TypeText,
    UnavailableWorkspace,
    Workspace,
    WorkspaceReadOnlyError,
    WorkspaceUnavailableError,
)

from .workspace_fakes import FAKE_SCREENSHOT, FakeWorkspace, InMemoryProvider

pytestmark = pytest.mark.anyio


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
    with pytest.raises(WorkspaceReadOnlyError, match='acting on the display'):
        await workspace.act(Click())
    assert provider.actions == []


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
            'Drag',
            'HoldKeys',
            'MouseDown',
            'MouseUp',
            'MoveMouse',
            'Point',
            'PressKeys',
            'Screenshot',
            'Scroll',
            'TypeText',
            'Wait',
        ]
    )
    assert all(definition['required'][0] == 'type' for name, definition in schema.items() if name != 'Point')
    # Every required name must exist: Gemini rejects a schema that requires an undeclared property.
    assert all(set(definition['required']) <= set(definition['properties']) for definition in schema.values())
