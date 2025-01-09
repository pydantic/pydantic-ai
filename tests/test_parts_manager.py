from __future__ import annotations as _annotations

import pytest
from inline_snapshot import snapshot

from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    ArgsJson,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)


@pytest.mark.parametrize('vendor_part_id', [None, 'content'])
def test_handle_text_deltas(vendor_part_id: str | None):
    manager = ModelResponsePartsManager()
    assert manager.get_parts() == []

    event = manager.handle_text_delta(vendor_part_id=vendor_part_id, content='hello ')
    assert event == snapshot(
        PartStartEvent(index=0, part=TextPart(content='hello ', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='hello ', part_kind='text')])

    event = manager.handle_text_delta(vendor_part_id=vendor_part_id, content='world')
    assert event == snapshot(
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='world', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot([TextPart(content='hello world', part_kind='text')])


def test_handle_dovetailed_text_deltas():
    manager = ModelResponsePartsManager()

    event = manager.handle_text_delta(vendor_part_id='first', content='hello ')
    assert event == snapshot(
        PartStartEvent(index=0, part=TextPart(content='hello ', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='hello ', part_kind='text')])

    event = manager.handle_text_delta(vendor_part_id='second', content='goodbye ')
    assert event == snapshot(
        PartStartEvent(index=1, part=TextPart(content='goodbye ', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='hello ', part_kind='text'), TextPart(content='goodbye ', part_kind='text')]
    )

    event = manager.handle_text_delta(vendor_part_id='first', content='world')
    assert event == snapshot(
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='world', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='hello world', part_kind='text'), TextPart(content='goodbye ', part_kind='text')]
    )

    event = manager.handle_text_delta(vendor_part_id='second', content='Samuel')
    assert event == snapshot(
        PartDeltaEvent(
            index=1, delta=TextPartDelta(content_delta='Samuel', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='hello world', part_kind='text'), TextPart(content='goodbye Samuel', part_kind='text')]
    )


def test_handle_tool_call_deltas():
    manager = ModelResponsePartsManager()

    event = manager.handle_tool_call_delta(
        vendor_part_id='first', tool_name='tool1', args='{"arg1":', tool_call_id=None
    )
    assert event == snapshot(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                tool_name='tool1', args=ArgsJson(args_json='{"arg1":'), tool_call_id=None, part_kind='tool-call'
            ),
            event_kind='part_start',
        )
    )
    assert manager.get_parts() == snapshot(
        [ToolCallPart(tool_name='tool1', args=ArgsJson(args_json='{"arg1":'), tool_call_id=None, part_kind='tool-call')]
    )

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name=None, args='"value1"}', tool_call_id=None)
    assert event == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_name_delta=None, args_delta='"value1"}', tool_call_id=None, part_delta_kind='tool_call'
            ),
            event_kind='part_delta',
        )
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args=ArgsJson(args_json='{"arg1":"value1"}'),
                tool_call_id=None,
                part_kind='tool-call',
            )
        ]
    )


@pytest.mark.parametrize('vendor_part_id', [None, 'content'])
def test_handle_mixed_deltas_without_text_part_id(vendor_part_id: str | None):
    manager = ModelResponsePartsManager()

    event = manager.handle_text_delta(vendor_part_id=vendor_part_id, content='hello ')
    assert event == snapshot(
        PartStartEvent(index=0, part=TextPart(content='hello ', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='hello ', part_kind='text')])

    event = manager.handle_tool_call_delta(
        vendor_part_id='first_tool_call', tool_name='tool1', args='{"arg1":', tool_call_id='abc'
    )
    assert event == snapshot(
        PartStartEvent(
            index=1,
            part=ToolCallPart(
                tool_name='tool1', args=ArgsJson(args_json='{"arg1":'), tool_call_id='abc', part_kind='tool-call'
            ),
            event_kind='part_start',
        )
    )

    event = manager.handle_text_delta(vendor_part_id=vendor_part_id, content='world')
    if vendor_part_id is None:
        assert event == snapshot(
            PartStartEvent(
                index=2,
                part=TextPart(content='world', part_kind='text'),
                event_kind='part_start',
            )
        )
        assert manager.get_parts() == snapshot(
            [
                TextPart(content='hello ', part_kind='text'),
                ToolCallPart(
                    tool_name='tool1', args=ArgsJson(args_json='{"arg1":'), tool_call_id='abc', part_kind='tool-call'
                ),
                TextPart(content='world', part_kind='text'),
            ]
        )
    else:
        assert event == snapshot(
            PartDeltaEvent(
                index=0, delta=TextPartDelta(content_delta='world', part_delta_kind='text'), event_kind='part_delta'
            )
        )
        assert manager.get_parts() == snapshot(
            [
                TextPart(content='hello world', part_kind='text'),
                ToolCallPart(
                    tool_name='tool1', args=ArgsJson(args_json='{"arg1":'), tool_call_id='abc', part_kind='tool-call'
                ),
            ]
        )
