"""Tests for the `computer` tool's actions.

Not VCR tests: validating and serializing an action happens before any provider is involved, so a recording adds nothing.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import fields
from typing import get_args

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter, ValidationError

from pydantic_ai.computer_use import (
    Click,
    ComputerToolAction,
    CursorPosition,
    Drag,
    HoldKeys,
    MouseDown,
    MouseMove,
    MouseUp,
    Point,
    PressKeys,
    Region,
    Screenshot,
    Scroll,
    TypeText,
    Wait,
    Zoom,
)


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


_EVERY_ACTION: list[tuple[dict[str, object], ComputerToolAction]] = [
    (
        {'type': 'click', 'x': 1, 'y': 2, 'modifiers': ['CTRL', 'Super', 'alt']},
        Click(x=1, y=2, modifiers=('ctrl', 'super', 'alt')),
    ),
    ({'type': 'mouse_move', 'x': 3, 'y': 4}, MouseMove(x=3, y=4)),
    (
        {'type': 'drag', 'path': [{'x': 0, 'y': 0}, {'x': 5, 'y': 6}]},
        Drag(path=(Point(x=0, y=0), Point(x=5, y=6))),
    ),
    ({'type': 'mouse_down', 'button': 'right'}, MouseDown(button='right')),
    ({'type': 'mouse_up'}, MouseUp()),
    (
        {'type': 'scroll', 'direction': 'down', 'amount': 3, 'x': 10, 'y': 20},
        Scroll(direction='down', amount=3, x=10, y=20),
    ),
    ({'type': 'type_text', 'text': 'hello'}, TypeText(text='hello')),
    ({'type': 'press_keys', 'keys': ['CTRL', 'Enter', 'T']}, PressKeys(keys=('ctrl', 'enter', 't'))),
    ({'type': 'hold_keys', 'keys': ['Escape'], 'seconds': 1}, HoldKeys(keys=('escape',), seconds=1)),
    ({'type': 'wait', 'seconds': 0.5}, Wait(seconds=0.5)),
    ({'type': 'screenshot'}, Screenshot()),
    (
        {'type': 'zoom', 'region': {'left': 0, 'top': 0, 'right': 5, 'bottom': 5}},
        Zoom(region=Region(left=0, top=0, right=5, bottom=5)),
    ),
    ({'type': 'cursor_position'}, CursorPosition()),
]


@pytest.mark.parametrize('given, expected', [pytest.param(*case, id=case[1].type) for case in _EVERY_ACTION])
def test_each_action_reads_its_fields_and_round_trips_through_json(
    given: dict[str, object], expected: ComputerToolAction
) -> None:
    adapter = TypeAdapter(list[ComputerToolAction])
    actions = adapter.validate_python([given])
    assert actions == [expected]
    assert adapter.validate_json(adapter.dump_json(actions)) == actions


@pytest.mark.parametrize(
    'invalid',
    [
        pytest.param({'type': 'click', 'x': 10}, id='x-without-y'),
        pytest.param({'type': 'scroll', 'x': 1, 'y': 2}, id='scroll-without-direction'),
        pytest.param({'type': 'teleport'}, id='unknown-type'),
        pytest.param({'x': 1, 'y': 2}, id='no-type'),
    ],
)
def test_actions_of_a_bad_shape_are_refused(invalid: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TypeAdapter(list[ComputerToolAction]).validate_python([invalid])


@pytest.mark.parametrize(
    'build, message',
    [
        pytest.param(lambda: Scroll(direction='down', amount=0), 'greater than or equal to 1', id='scroll-of-zero'),
        pytest.param(
            lambda: Region(left=5, top=0, right=5, bottom=10),
            'a region needs `right` greater than `left` and `bottom` greater than `top`',
            id='empty-region',
        ),
        pytest.param(lambda: PressKeys(keys=()), 'at least 1 item', id='no-keys'),
        # A provider's own spelling is the model adapter's to convert, so it doesn't reach an action.
        pytest.param(
            lambda: PressKeys(keys=('Page_Down',)),
            "Input should be 'ctrl', 'shift', 'alt', 'super', 'enter'",
            id='unknown-key',
        ),
        pytest.param(
            lambda: PressKeys(keys=('ctrl+s',)),
            "Input should be 'ctrl', 'shift', 'alt', 'super', 'enter'",
            id='combination',
        ),
        pytest.param(
            lambda: HoldKeys(keys=('',), seconds=1), 'String should have at least 1 character', id='empty-key'
        ),
        # Modifiers are checked before anything runs, so a key that isn't one can't be skipped mid-action.
        pytest.param(
            lambda: TypeAdapter(Click).validate_python({'modifiers': ['ctrl', 'a']}),
            "Input should be 'ctrl', 'shift', 'alt' or 'super'",
            id='not-a-modifier',
        ),
    ],
)
def test_actions_are_validated_when_built_in_python(build: Callable[[], object], message: str) -> None:
    with pytest.raises(ValidationError, match=re.escape(message)):
        build()


def test_each_action_type_is_its_class_name_in_snake_case() -> None:
    (actions, _) = get_args(ComputerToolAction)
    for action in get_args(actions):
        (type_field,) = (field for field in fields(action) if field.name == 'type')
        assert type_field.default == re.sub(r'(?<!^)(?=[A-Z])', '_', action.__name__).lower()


def test_the_computer_tool_schema_requires_type_first_and_only_names_it_declares() -> None:
    schema = TypeAdapter(list[ComputerToolAction]).json_schema()['$defs']
    assert sorted(schema) == snapshot(
        [
            'Click',
            'CursorPosition',
            'Drag',
            'HoldKeys',
            'MouseDown',
            'MouseMove',
            'MouseUp',
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


@pytest.mark.parametrize('seconds', [float('inf'), float('nan')])
def test_a_duration_must_be_finite(seconds: float) -> None:
    # JSON can't hold it, so a saved history would lose the action.
    with pytest.raises(ValidationError, match='finite number'):
        Wait(seconds=seconds)
    with pytest.raises(ValidationError, match='finite number'):
        HoldKeys(keys=('shift',), seconds=seconds)
