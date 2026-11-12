"""Tests for the `computer` tool's actions."""

from __future__ import annotations

from typing import Any

import pytest
from inline_snapshot import snapshot
from pydantic import TypeAdapter, ValidationError

from pydantic_ai.computer_use import (
    Click,
    ComputerToolAction,
    HoldKeys,
    PressKeys,
    Region,
    Scroll,
    TypeText,
)


@pytest.mark.parametrize(
    'given, expected',
    [
        ('KP_Enter', ('numpad_enter',)),
        ('XF86AudioMute', ('volumemute',)),
        ('Caps_Lock', ('capslock',)),
        ('Print', ('printscreen',)),
        ('Super_L', ('super',)),
        ('Meta_L', ('super',)),
        ('ctrl+Page_Down', ('ctrl', 'pagedown')),
        ('ctrl+plus', ('ctrl', '+')),
        ('minus', ('-',)),
        ('bracketleft', ('[',)),
        ('question', ('?',)),
        ('NumpadAdd', ('numpad_add',)),
        ('NumpadEnter', ('numpad_enter',)),
        ('AudioVolumeUp', ('volumeup',)),
        ('MediaPlayPause', ('mediaplay',)),
        ('ShiftLeft', ('shift',)),
        ('CTRL', ('ctrl',)),
        ('Return', ('enter',)),
        ('ESC', ('escape',)),
        ('ctrl+shift+T', ('ctrl', 'shift', 't')),
        ('+', ('+',)),
        ('ctrl+shift++', ('ctrl', 'shift', '+')),
    ],
)
def test_key_names_normalize_to_one_spelling(given: str, expected: tuple[str, ...]) -> None:
    assert PressKeys(keys=(given,)).keys == expected


@pytest.mark.parametrize('key', ['Control_R', 'Shift_R', 'Alt_R', 'Super_R', 'Meta_R', 'ControlRight', 'AltRight'])
def test_right_hand_modifiers_are_refused(key: str) -> None:
    # Kept free to name keys of their own later, such as AltGr, rather than meaning the left-hand modifier forever.
    with pytest.raises(ValidationError, match='is not a key'):
        PressKeys(keys=(key,))


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


def test_actions_round_trip_through_json_and_refuse_bad_shapes() -> None:
    adapter = TypeAdapter(list[ComputerToolAction])
    actions = adapter.validate_python(
        [
            {'type': 'keypress', 'keys': ['CTRL', 'Return']},
            {'type': 'click', 'x': 1, 'y': 2, 'modifiers': ['Control', 'cmd', 'Option']},
            {'type': 'hold_key', 'keys': ['ESC'], 'seconds': 1},
        ]
    )
    assert actions == [
        PressKeys(keys=('ctrl', 'enter')),
        Click(x=1, y=2, modifiers=('ctrl', 'super', 'alt')),
        HoldKeys(keys=('escape',), seconds=1),
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
    with pytest.raises(
        ValidationError, match=r"'Hyper_L' is not a key: use a single character or one of ctrl, .*, f1 to f24"
    ):
        PressKeys(keys=('Hyper_L',))


@pytest.mark.parametrize(
    'build',
    [
        lambda: Scroll(direction='down', amount=0),
        lambda: Region(left=5, top=0, right=5, bottom=10),
        lambda: TypeText(text='a\x00b'),
        lambda: PressKeys(keys=()),
        lambda: HoldKeys(keys=('',), seconds=1),
        # Splitting a combination can leave an empty key between two `+`.
        lambda: PressKeys(keys=('ctrl+++',)),
        # Modifiers are checked before anything runs, so an unknown one can't be skipped mid-action.
        lambda: Click(modifiers=('ctrl', 'exec')),
        # A known key that isn't a modifier is refused too.
        lambda: Click(modifiers=('ctrl', 'a')),
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
