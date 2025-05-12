from __future__ import annotations as _annotations

import re
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import UnexpectedModelBehavior
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.messages import (
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

from .conftest import IsStr


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

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name='tool', args=None, tool_call_id=None)
    # Not enough information to produce a part, so no event and no part
    assert event == snapshot(None)
    assert manager.get_parts() == snapshot([])

    # Now that we have args, we can produce a part:
    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name=None, args='{"arg1":', tool_call_id=None)
    assert event == snapshot(
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name='tool', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call'),
            event_kind='part_start',
        )
    )
    assert manager.get_parts() == snapshot(
        [ToolCallPart(tool_name='tool', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call')]
    )

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name='1', args=None, tool_call_id=None)
    assert event == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_name_delta='1', args_delta=None, tool_call_id=IsStr(), part_delta_kind='tool_call'
            ),
            event_kind='part_delta',
        )
    )
    assert manager.get_parts() == snapshot(
        [ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call')]
    )

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name=None, args='"value1"}', tool_call_id=None)
    assert event == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_name_delta=None, args_delta='"value1"}', tool_call_id=IsStr(), part_delta_kind='tool_call'
            ),
            event_kind='part_delta',
        )
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":"value1"}',
                tool_call_id=IsStr(),
                part_kind='tool-call',
            )
        ]
    )


def test_handle_tool_call_deltas_without_vendor_id():
    # Note, tool_name should not be specified in subsequent deltas when the vendor_part_id is None
    manager = ModelResponsePartsManager()
    manager.handle_tool_call_delta(vendor_part_id=None, tool_name='tool1', args='{"arg1":', tool_call_id=None)
    manager.handle_tool_call_delta(vendor_part_id=None, tool_name=None, args='"value1"}', tool_call_id=None)
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":"value1"}',
                tool_call_id=IsStr(),
                part_kind='tool-call',
            )
        ]
    )

    # This test is included just to document/demonstrate what happens if you do repeat the tool name
    manager = ModelResponsePartsManager()
    manager.handle_tool_call_delta(vendor_part_id=None, tool_name='tool2', args='{"arg1":', tool_call_id=None)
    manager.handle_tool_call_delta(vendor_part_id=None, tool_name='tool2', args='"value1"}', tool_call_id=None)
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(tool_name='tool2', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call'),
            ToolCallPart(tool_name='tool2', args='"value1"}', tool_call_id=IsStr(), part_kind='tool-call'),
        ]
    )


def test_handle_tool_call_part():
    manager = ModelResponsePartsManager()

    # Basic use of this API
    event = manager.handle_tool_call_part(vendor_part_id='first', tool_name='tool1', args='{"arg1":', tool_call_id=None)
    assert event == snapshot(
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call'),
            event_kind='part_start',
        )
    )

    # Add a delta
    manager.handle_tool_call_delta(vendor_part_id='second', tool_name='tool1', args=None, tool_call_id=None)
    assert manager.get_parts() == snapshot(
        [ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call')]
    )

    # Override it with handle_tool_call_part
    manager.handle_tool_call_part(vendor_part_id='second', tool_name='tool1', args='{}', tool_call_id=None)
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call'),
            ToolCallPart(tool_name='tool1', args='{}', tool_call_id=IsStr(), part_kind='tool-call'),
        ]
    )

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name=None, args='"value1"}', tool_call_id=None)
    assert event == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_name_delta=None, args_delta='"value1"}', tool_call_id=IsStr(), part_delta_kind='tool_call'
            ),
            event_kind='part_delta',
        )
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":"value1"}',
                tool_call_id=IsStr(),
                part_kind='tool-call',
            ),
            ToolCallPart(tool_name='tool1', args='{}', tool_call_id=IsStr(), part_kind='tool-call'),
        ]
    )

    # Finally, demonstrate behavior when no vendor_part_id is provided:
    event = manager.handle_tool_call_part(vendor_part_id=None, tool_name='tool1', args='{}', tool_call_id=None)
    assert event == snapshot(
        PartStartEvent(
            index=2,
            part=ToolCallPart(tool_name='tool1', args='{}', tool_call_id=IsStr(), part_kind='tool-call'),
            event_kind='part_start',
        )
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":"value1"}',
                tool_call_id=IsStr(),
                part_kind='tool-call',
            ),
            ToolCallPart(tool_name='tool1', args='{}', tool_call_id=IsStr(), part_kind='tool-call'),
            ToolCallPart(tool_name='tool1', args='{}', tool_call_id=IsStr(), part_kind='tool-call'),
        ]
    )


@pytest.mark.parametrize('text_vendor_part_id', [None, 'content'])
@pytest.mark.parametrize('tool_vendor_part_id', [None, 'tool'])
def test_handle_mixed_deltas_without_text_part_id(text_vendor_part_id: str | None, tool_vendor_part_id: str | None):
    manager = ModelResponsePartsManager()

    event = manager.handle_text_delta(vendor_part_id=text_vendor_part_id, content='hello ')
    assert event == snapshot(
        PartStartEvent(index=0, part=TextPart(content='hello ', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='hello ', part_kind='text')])

    event = manager.handle_tool_call_delta(
        vendor_part_id=tool_vendor_part_id, tool_name='tool1', args='{"arg1":', tool_call_id='abc'
    )
    assert event == snapshot(
        PartStartEvent(
            index=1,
            part=ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id='abc', part_kind='tool-call'),
            event_kind='part_start',
        )
    )

    event = manager.handle_text_delta(vendor_part_id=text_vendor_part_id, content='world')
    if text_vendor_part_id is None:
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
                ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id='abc', part_kind='tool-call'),
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
                ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id='abc', part_kind='tool-call'),
            ]
        )


def test_cannot_convert_from_text_to_tool_call():
    manager = ModelResponsePartsManager()
    manager.handle_text_delta(vendor_part_id=1, content='hello')
    with pytest.raises(
        UnexpectedModelBehavior, match=re.escape('Cannot apply a tool call delta to existing_part=TextPart(')
    ):
        manager.handle_tool_call_delta(vendor_part_id=1, tool_name='tool1', args='{"arg1":', tool_call_id=None)


def test_cannot_convert_from_tool_call_to_text():
    manager = ModelResponsePartsManager()
    manager.handle_tool_call_delta(vendor_part_id=1, tool_name='tool1', args='{"arg1":', tool_call_id=None)
    with pytest.raises(
        UnexpectedModelBehavior, match=re.escape('Cannot apply a text delta to existing_part=ToolCallPart(')
    ):
        manager.handle_text_delta(vendor_part_id=1, content='hello')


def test_tool_call_id_delta():
    manager = ModelResponsePartsManager()
    manager.handle_tool_call_delta(vendor_part_id=1, tool_name='tool1', args='{"arg1":', tool_call_id=None)
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":',
                tool_call_id=IsStr(),
                part_kind='tool-call',
            )
        ]
    )

    manager.handle_tool_call_delta(vendor_part_id=1, tool_name=None, args='"value1"}', tool_call_id='id2')
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":"value1"}',
                tool_call_id='id2',
                part_kind='tool-call',
            )
        ]
    )


@pytest.mark.parametrize('apply_to_delta', [True, False])
def test_tool_call_id_delta_failure(apply_to_delta: bool):
    tool_name = 'tool1'
    manager = ModelResponsePartsManager()
    manager.handle_tool_call_delta(
        vendor_part_id=1, tool_name=None if apply_to_delta else tool_name, args='{"arg1":', tool_call_id='id1'
    )
    assert (
        manager.get_parts() == []
        if apply_to_delta
        else [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":',
                tool_call_id='id1',
                part_kind='tool-call',
            )
        ]
    )


@pytest.mark.parametrize(
    'args1,args2,result',
    [
        ('{"arg1":', '"value1"}', '{"arg1":"value1"}'),
        ('{"a":1}', {}, UnexpectedModelBehavior('Cannot apply dict deltas to non-dict tool arguments ')),
        ({}, '{"b":2}', UnexpectedModelBehavior('Cannot apply JSON deltas to non-JSON tool arguments ')),
        ({'a': 1}, {'b': 2}, {'a': 1, 'b': 2}),
    ],
)
@pytest.mark.parametrize('apply_to_delta', [False, True])
def test_apply_tool_delta_variants(
    args1: str | dict[str, Any],
    args2: str | dict[str, Any],
    result: str | dict[str, Any] | UnexpectedModelBehavior,
    apply_to_delta: bool,
):
    tool_name = 'tool1'

    manager = ModelResponsePartsManager()
    manager.handle_tool_call_delta(
        vendor_part_id=1, tool_name=None if apply_to_delta else tool_name, args=args1, tool_call_id=None
    )

    if isinstance(result, UnexpectedModelBehavior):
        with pytest.raises(UnexpectedModelBehavior, match=re.escape(str(result))):
            manager.handle_tool_call_delta(vendor_part_id=1, tool_name=None, args=args2, tool_call_id=None)
    else:
        manager.handle_tool_call_delta(vendor_part_id=1, tool_name=None, args=args2, tool_call_id=None)
        if apply_to_delta:
            assert len(manager.get_parts()) == 0  # Ensure there are only deltas being managed
            manager.handle_tool_call_delta(vendor_part_id=1, tool_name=tool_name, args=None, tool_call_id=None)
        tool_call_part = manager.get_parts()[0]
        assert isinstance(tool_call_part, ToolCallPart)
        assert tool_call_part.args == result
