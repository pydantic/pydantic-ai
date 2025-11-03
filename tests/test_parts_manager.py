from __future__ import annotations as _annotations

import re
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import (
    BuiltinToolCallPart,
    PartDeltaEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
    UnexpectedModelBehavior,
)
from pydantic_ai._parts_manager import ModelResponsePartsManager

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


def test_handle_text_deltas_with_think_tags():
    manager = ModelResponsePartsManager()
    thinking_tags = ('<think>', '</think>')

    event = manager.handle_text_delta(vendor_part_id='content', content='pre-', thinking_tags=thinking_tags)
    assert event == snapshot(
        PartStartEvent(index=0, part=TextPart(content='pre-', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='pre-', part_kind='text')])

    event = manager.handle_text_delta(vendor_part_id='content', content='thinking', thinking_tags=thinking_tags)
    assert event == snapshot(
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='thinking', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot([TextPart(content='pre-thinking', part_kind='text')])

    event = manager.handle_text_delta(vendor_part_id='content', content='<think>', thinking_tags=thinking_tags)
    assert event == snapshot(
        PartStartEvent(index=1, part=ThinkingPart(content='', part_kind='thinking'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='pre-thinking', part_kind='text'), ThinkingPart(content='', part_kind='thinking')]
    )

    event = manager.handle_text_delta(vendor_part_id='content', content='thinking', thinking_tags=thinking_tags)
    assert event == snapshot(
        PartDeltaEvent(
            index=1,
            delta=ThinkingPartDelta(content_delta='thinking', part_delta_kind='thinking'),
            event_kind='part_delta',
        )
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='pre-thinking', part_kind='text'), ThinkingPart(content='thinking', part_kind='thinking')]
    )

    event = manager.handle_text_delta(vendor_part_id='content', content=' more', thinking_tags=thinking_tags)
    assert event == snapshot(
        PartDeltaEvent(
            index=1, delta=ThinkingPartDelta(content_delta=' more', part_delta_kind='thinking'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot(
        [
            TextPart(content='pre-thinking', part_kind='text'),
            ThinkingPart(content='thinking more', part_kind='thinking'),
        ]
    )

    event = manager.handle_text_delta(vendor_part_id='content', content='</think>', thinking_tags=thinking_tags)
    assert event is None

    event = manager.handle_text_delta(vendor_part_id='content', content='post-', thinking_tags=thinking_tags)
    assert event == snapshot(
        PartStartEvent(index=2, part=TextPart(content='post-', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot(
        [
            TextPart(content='pre-thinking', part_kind='text'),
            ThinkingPart(content='thinking more', part_kind='thinking'),
            TextPart(content='post-', part_kind='text'),
        ]
    )

    event = manager.handle_text_delta(vendor_part_id='content', content='thinking', thinking_tags=thinking_tags)
    assert event == snapshot(
        PartDeltaEvent(
            index=2, delta=TextPartDelta(content_delta='thinking', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot(
        [
            TextPart(content='pre-thinking', part_kind='text'),
            ThinkingPart(content='thinking more', part_kind='thinking'),
            TextPart(content='post-thinking', part_kind='text'),
        ]
    )


def test_handle_tool_call_deltas():
    manager = ModelResponsePartsManager()

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name=None, args='{"arg1":', tool_call_id=None)
    # Not enough information to produce a part, so no event and no part
    assert event == snapshot(None)
    assert manager.get_parts() == snapshot([])

    # Now that we have a tool name, we can produce a part:
    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name='tool', args=None, tool_call_id='call')
    assert event == snapshot(
        PartStartEvent(
            index=0,
            part=ToolCallPart(tool_name='tool', args='{"arg1":', tool_call_id='call', part_kind='tool-call'),
            event_kind='part_start',
        )
    )
    assert manager.get_parts() == snapshot(
        [ToolCallPart(tool_name='tool', args='{"arg1":', tool_call_id='call', part_kind='tool-call')]
    )

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name='1', args=None, tool_call_id=None)
    assert event == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_name_delta='1', args_delta=None, tool_call_id='call', part_delta_kind='tool_call'
            ),
            event_kind='part_delta',
        )
    )
    assert manager.get_parts() == snapshot(
        [ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id='call', part_kind='tool-call')]
    )

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name=None, args='"value1"}', tool_call_id=None)
    assert event == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_name_delta=None, args_delta='"value1"}', tool_call_id='call', part_delta_kind='tool_call'
            ),
            event_kind='part_delta',
        )
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":"value1"}',
                tool_call_id='call',
                part_kind='tool-call',
            )
        ]
    )


def test_handle_tool_call_deltas_without_args():
    manager = ModelResponsePartsManager()

    # Test None args followed by a string
    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name='tool', args=None, tool_call_id=None)
    assert event == snapshot(
        PartStartEvent(index=0, part=ToolCallPart(tool_name='tool', args=None, tool_call_id=IsStr()))
    )
    assert manager.get_parts() == snapshot([ToolCallPart(tool_name='tool', tool_call_id=IsStr())])

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name=None, args='{"arg1":', tool_call_id=None)
    assert event == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(args_delta='{"arg1":', tool_call_id=IsStr()),
        )
    )
    assert manager.get_parts() == snapshot([ToolCallPart(tool_name='tool', args='{"arg1":', tool_call_id=IsStr())])

    # Test None args followed by a dict
    event = manager.handle_tool_call_delta(vendor_part_id='second', tool_name='tool', args=None, tool_call_id=None)
    assert event == snapshot(
        PartStartEvent(index=1, part=ToolCallPart(tool_name='tool', args=None, tool_call_id=IsStr()))
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(tool_name='tool', args='{"arg1":', tool_call_id=IsStr()),
            ToolCallPart(tool_name='tool', args=None, tool_call_id=IsStr()),
        ]
    )

    event = manager.handle_tool_call_delta(
        vendor_part_id='second', tool_name=None, args={'arg1': 'value1'}, tool_call_id=None
    )
    assert event == snapshot(
        PartDeltaEvent(
            index=1,
            delta=ToolCallPartDelta(args_delta={'arg1': 'value1'}, tool_call_id=IsStr()),
        )
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(tool_name='tool', args='{"arg1":', tool_call_id=IsStr()),
            ToolCallPart(tool_name='tool', args={'arg1': 'value1'}, tool_call_id=IsStr()),
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
        [
            ToolCallPart(tool_name='tool1', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call'),
            ToolCallPart(tool_name='tool1', tool_call_id=IsStr()),
        ]
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


def test_handle_thinking_delta_no_vendor_id_with_existing_thinking_part():
    manager = ModelResponsePartsManager()

    # Add a thinking part first
    event = manager.handle_thinking_delta(vendor_part_id='first', content='initial thought', signature=None)
    assert isinstance(event, PartStartEvent)
    assert event.index == 0

    # Now add another thinking delta with no vendor_part_id - should update the latest thinking part
    event = manager.handle_thinking_delta(vendor_part_id=None, content=' more', signature=None)
    assert isinstance(event, PartDeltaEvent)
    assert event.index == 0

    parts = manager.get_parts()
    assert parts == snapshot([ThinkingPart(content='initial thought more')])


def test_handle_thinking_delta_wrong_part_type():
    manager = ModelResponsePartsManager()

    # Add a text part first
    manager.handle_text_delta(vendor_part_id='text', content='hello')

    # Try to apply thinking delta to the text part - should raise error
    with pytest.raises(UnexpectedModelBehavior, match=r'Cannot apply a thinking delta to existing_part='):
        manager.handle_thinking_delta(vendor_part_id='text', content='thinking', signature=None)


def test_handle_thinking_delta_new_part_with_vendor_id():
    manager = ModelResponsePartsManager()

    event = manager.handle_thinking_delta(vendor_part_id='thinking', content='new thought', signature=None)
    assert isinstance(event, PartStartEvent)
    assert event.index == 0

    parts = manager.get_parts()
    assert parts == snapshot([ThinkingPart(content='new thought')])


def test_handle_thinking_delta_no_content():
    manager = ModelResponsePartsManager()

    with pytest.raises(UnexpectedModelBehavior, match='Cannot create a ThinkingPart with no content'):
        manager.handle_thinking_delta(vendor_part_id=None, content=None, signature=None)


def test_handle_thinking_delta_no_content_or_signature():
    manager = ModelResponsePartsManager()

    # Add a thinking part first
    manager.handle_thinking_delta(vendor_part_id='thinking', content='initial', signature=None)

    # Try to update with no content or signature - should raise error
    with pytest.raises(UnexpectedModelBehavior, match='Cannot update a ThinkingPart with no content or signature'):
        manager.handle_thinking_delta(vendor_part_id='thinking', content=None, signature=None)


def test_handle_part():
    manager = ModelResponsePartsManager()

    part = BuiltinToolCallPart(tool_name='tool1', args='{"arg1": ')

    event = manager.handle_part(vendor_part_id='builtin', part=part)
    assert event == snapshot(PartStartEvent(index=0, part=part))
    assert manager.get_parts() == snapshot([part])

    # Add a delta
    event = manager.handle_tool_call_delta(vendor_part_id='builtin', args='"value1"}')
    assert event == snapshot(
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='"value1"}', tool_call_id=part.tool_call_id))
    )
    assert manager.get_parts() == snapshot(
        [BuiltinToolCallPart(tool_name='tool1', args='{"arg1": "value1"}', tool_call_id=part.tool_call_id)]
    )

    # Override it with handle_part
    part2 = BuiltinToolCallPart(tool_name='tool1', args='{"arg2": ')
    event = manager.handle_part(vendor_part_id='builtin', part=part2)
    assert event == snapshot(PartStartEvent(index=0, part=part2))
    assert manager.get_parts() == snapshot([part2])

    # Finally, demonstrate behavior when no vendor_part_id is provided:
    part3 = BuiltinToolCallPart(tool_name='tool1', args='{"arg3": ')
    event = manager.handle_part(vendor_part_id=None, part=part3)
    assert event == snapshot(PartStartEvent(index=1, part=part3))
    assert manager.get_parts() == snapshot([part2, part3])
