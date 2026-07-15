from __future__ import annotations as _annotations

import re
from typing import Any

import pytest
from pytest_mock import MockerFixture

from pydantic_ai import (
    NativeToolCallPart,
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
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart
from pydantic_ai._parts_manager import ModelResponsePartsManager
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.tools import ToolDefinition

from ._inline_snapshot import snapshot
from .conftest import IsStr


@pytest.mark.parametrize('vendor_part_id', [None, 'content'])
def test_handle_text_deltas(vendor_part_id: str | None):
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    assert manager.get_parts() == []

    event = next(manager.handle_text_delta(vendor_part_id=vendor_part_id, content='hello '))
    assert event == snapshot(
        PartStartEvent(index=0, part=TextPart(content='hello ', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='hello ', part_kind='text')])

    event = next(manager.handle_text_delta(vendor_part_id=vendor_part_id, content='world'))
    assert event == snapshot(
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='world', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot([TextPart(content='hello world', part_kind='text')])


def test_handle_dovetailed_text_deltas():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    event = next(manager.handle_text_delta(vendor_part_id='first', content='hello '))
    assert event == snapshot(
        PartStartEvent(index=0, part=TextPart(content='hello ', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='hello ', part_kind='text')])

    event = next(manager.handle_text_delta(vendor_part_id='second', content='goodbye '))
    assert event == snapshot(
        PartStartEvent(index=1, part=TextPart(content='goodbye ', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='hello ', part_kind='text'), TextPart(content='goodbye ', part_kind='text')]
    )

    event = next(manager.handle_text_delta(vendor_part_id='first', content='world'))
    assert event == snapshot(
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='world', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='hello world', part_kind='text'), TextPart(content='goodbye ', part_kind='text')]
    )

    event = next(manager.handle_text_delta(vendor_part_id='second', content='Samuel'))
    assert event == snapshot(
        PartDeltaEvent(
            index=1, delta=TextPartDelta(content_delta='Samuel', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='hello world', part_kind='text'), TextPart(content='goodbye Samuel', part_kind='text')]
    )


def test_string_deltas_materialize_on_reads_and_replacement():
    """Internal buffer lifecycle and replacement are not observable in provider cassettes."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    next(
        manager.handle_text_delta(
            vendor_part_id='text', content='text', provider_name='provider', provider_details={'first': 1}
        )
    )
    next(manager.handle_thinking_delta(vendor_part_id='thinking', content='thinking'))
    manager.handle_tool_call_delta(vendor_part_id='tool', tool_name='tool', args='{"value":', tool_call_id='call')

    for suffix in ('-one', '-two', '-three'):
        next(
            manager.handle_text_delta(
                vendor_part_id='text',
                content=suffix,
                provider_name='provider',
                provider_details={suffix: True},
            )
        )
        next(manager.handle_thinking_delta(vendor_part_id='thinking', content=suffix))
        manager.handle_tool_call_delta(vendor_part_id='tool', args=suffix)

    assert manager.get_part_by_vendor_id('text') == TextPart(
        'text-one-two-three',
        provider_name='provider',
        provider_details={'first': 1, '-one': True, '-two': True, '-three': True},
    )
    snapshot_parts = manager.get_parts()
    assert snapshot_parts == [
        TextPart(
            'text-one-two-three',
            provider_name='provider',
            provider_details={'first': 1, '-one': True, '-two': True, '-three': True},
        ),
        ThinkingPart('thinking-one-two-three'),
        ToolCallPart('tool', '{"value":-one-two-three', 'call'),
    ]

    next(manager.handle_text_delta(vendor_part_id='text', content='-discarded'))
    assert snapshot_parts[0] == TextPart(
        'text-one-two-three',
        provider_name='provider',
        provider_details={'first': 1, '-one': True, '-two': True, '-three': True},
    )
    replacement = TextPart('replacement')
    assert manager.handle_part(vendor_part_id='text', part=replacement) == PartStartEvent(index=0, part=replacement)
    assert manager.get_parts() == [
        replacement,
        ThinkingPart('thinking-one-two-three'),
        ToolCallPart('tool', '{"value":-one-two-three', 'call'),
    ]


def _emit_string_part_delta(manager: ModelResponsePartsManager, part_kind: str, content: str, **kwargs: Any) -> Any:
    if part_kind == 'text':
        return next(manager.handle_text_delta(vendor_part_id='part', content=content, **kwargs))
    if part_kind == 'thinking':
        return next(manager.handle_thinking_delta(vendor_part_id='part', content=content, **kwargs))
    return next(
        manager.handle_text_delta(
            vendor_part_id='part', content=content, thinking_tags=('<think>', '</think>'), **kwargs
        )
    )


@pytest.mark.parametrize('part_kind', ['text', 'thinking', 'embedded-thinking'])
def test_content_only_delta_matches_provider_details_normalization(part_kind: str):
    """Empty metadata normalization is not observable in provider cassettes."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    embedded = part_kind == 'embedded-thinking'
    initial_content = '<think>' if embedded else 'a'
    initial_part: TextPart | ThinkingPart = (
        TextPart('a', provider_details={})
        if part_kind == 'text'
        else ThinkingPart('' if embedded else 'a', provider_details={})
    )
    delta_type = TextPartDelta if part_kind == 'text' else ThinkingPartDelta
    start_event = _emit_string_part_delta(manager, part_kind, initial_content, provider_details={})
    if embedded:
        assert _emit_string_part_delta(manager, part_kind, '') == PartDeltaEvent(
            index=0, delta=ThinkingPartDelta(content_delta='')
        )

    normalized_delta = delta_type(content_delta='b')
    pure_content_delta = delta_type(content_delta='c')
    metadata_delta = delta_type(content_delta='d', provider_name='provider', provider_details={'metadata': True})
    expected_part = normalized_delta.apply(initial_part)
    expected_final_part = metadata_delta.apply(pure_content_delta.apply(expected_part))

    assert start_event == PartStartEvent(index=0, part=initial_part)
    assert _emit_string_part_delta(manager, part_kind, 'b') == PartDeltaEvent(index=0, delta=normalized_delta)
    assert expected_part.provider_details is None
    assert manager.get_parts() == [expected_part]
    assert manager.get_part_by_vendor_id('part') == expected_part
    assert _emit_string_part_delta(manager, part_kind, 'c') == PartDeltaEvent(index=0, delta=pure_content_delta)
    assert _emit_string_part_delta(
        manager, part_kind, 'd', provider_name='provider', provider_details={'metadata': True}
    ) == PartDeltaEvent(index=0, delta=metadata_delta)
    assert manager.get_parts() == [expected_final_part]


@pytest.mark.parametrize('part_kind', ['text', 'thinking', 'embedded-thinking'])
def test_content_delta_keeps_previous_provider_details_snapshot_isolated(part_kind: str):
    """Mutable provider metadata aliasing is not observable in provider cassettes."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    initial_content = '<think>' if part_kind == 'embedded-thinking' else 'a'
    start_event = _emit_string_part_delta(manager, part_kind, initial_content, provider_details={'stable': 1})
    _emit_string_part_delta(manager, part_kind, 'b')

    assert isinstance(start_event, PartStartEvent)
    assert start_event.part.provider_details is not None
    start_event.part.provider_details['stable'] = 99
    assert manager.get_parts()[0].provider_details == {'stable': 1}


def test_thinking_content_with_callable_metadata_stays_buffered():
    """The combined callable/content path is not produced by current provider cassettes."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    next(manager.handle_thinking_delta(vendor_part_id='part', content='a', provider_details={'count': 0}))

    def increment(details: dict[str, Any] | None) -> dict[str, Any]:
        return {'count': (details or {}).get('count', 0) + 1}

    for _ in range(2_048):
        next(manager.handle_thinking_delta(vendor_part_id='part', content='b', provider_details=increment))

    assert manager.get_parts() == [ThinkingPart('a' + 'b' * 2_048, provider_details={'count': 2_048})]


@pytest.mark.parametrize('size', [8_192, 16_384])
def test_incomplete_tool_call_string_arguments_are_buffered(size: int):
    """Cover buffered argument assembly that provider cassettes cannot express."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    for _ in range(size):
        assert manager.handle_tool_call_delta(vendor_part_id='tool', args='x') is None
        assert manager.get_parts() == []

    assert (
        manager.handle_tool_call_delta(
            vendor_part_id='tool',
            tool_call_id='call',
            provider_name='provider',
            provider_details={'first': True},
        )
        is None
    )

    event = manager.handle_tool_call_delta(
        vendor_part_id='tool', tool_name='tool', provider_name='provider', provider_details={'second': True}
    )
    assert isinstance(event, PartStartEvent)
    assert event.part == ToolCallPart(
        'tool',
        'x' * size,
        'call',
        provider_name='provider',
        provider_details={'first': True, 'second': True},
    )
    assert manager.get_parts() == [event.part]


def test_tool_call_provider_details_snapshot_isolated_from_buffered_arguments():
    """Ensure a prior public event snapshot cannot mutate buffered manager state."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    start_event = manager.handle_tool_call_delta(
        vendor_part_id='tool',
        tool_name='tool',
        args='{"value":',
        tool_call_id='call',
        provider_name='provider',
        provider_details={'stable': 1},
    )
    assert isinstance(start_event, PartStartEvent)

    manager.handle_tool_call_delta(vendor_part_id='tool', args='true}')
    assert start_event.part.provider_details is not None
    start_event.part.provider_details['stable'] = 99
    assert manager.get_parts()[0].provider_details == {'stable': 1}


def test_incomplete_tool_call_buffered_updates_preserve_state():
    """Cover incomplete buffered state transitions that provider cassettes cannot isolate."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    manager.handle_tool_call_delta(vendor_part_id='tool', args='{"value":', provider_details={'stable': 1})
    previous_part = manager.get_part_by_vendor_id('tool')
    assert isinstance(previous_part, ToolCallPartDelta)

    manager.handle_tool_call_delta(vendor_part_id='tool', args='true')
    manager.handle_tool_call_delta(vendor_part_id='tool', args='}', tool_call_id='call', provider_name='provider')
    assert previous_part.provider_details is not None
    previous_part.provider_details['stable'] = 99

    event = manager.handle_tool_call_delta(vendor_part_id='tool', tool_name='tool', args='')
    assert event == PartStartEvent(
        index=0,
        part=ToolCallPart(
            'tool',
            '{"value":true}',
            'call',
            provider_name='provider',
            provider_details={'stable': 1},
        ),
    )

    empty_manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    empty_manager.handle_tool_call_delta(vendor_part_id='tool', tool_call_id='call')
    empty_manager.handle_tool_call_delta(vendor_part_id='tool', args='')
    assert empty_manager.get_part_by_vendor_id('tool') == ToolCallPartDelta(args_delta='', tool_call_id='call')


def test_tool_call_promotes_after_buffered_arguments_are_materialized():
    """Cover typed promotion after arguments become complete across fragments."""
    manager = ModelResponsePartsManager(
        model_request_parameters=ModelRequestParameters(
            function_tools=[ToolDefinition(name='load_capability', tool_kind='capability-load')]
        )
    )
    start_event = manager.handle_tool_call_delta(
        vendor_part_id='tool',
        tool_name='load_capability',
        args='{"id":',
        tool_call_id='call',
    )
    assert isinstance(start_event, PartStartEvent)
    assert isinstance(start_event.part, LoadCapabilityCallPart)
    assert start_event.part.typed_args is None

    event = manager.handle_tool_call_delta(vendor_part_id='tool', args='"capability"}')
    assert isinstance(event, PartDeltaEvent)

    part = manager.get_part_by_vendor_id('tool')
    assert isinstance(part, LoadCapabilityCallPart)
    assert part.capability_id == 'capability'


def test_tool_call_buffer_changes_are_atomic_when_typed_promotion_fails(mocker: MockerFixture):
    """Injected internal promotion failures cannot be produced by a provider cassette."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    manager.handle_tool_call_delta(vendor_part_id='tool', tool_name='tool', args='initial', tool_call_id='call')
    manager.handle_tool_call_delta(vendor_part_id='tool', args=' buffered')

    promotion = mocker.patch.object(manager, '_typed_call_part', side_effect=RuntimeError('typed promotion failed'))
    with pytest.raises(RuntimeError, match='typed promotion failed'):
        manager.handle_tool_call_delta(vendor_part_id='tool', args=' discarded')
    mocker.stop(promotion)

    assert manager.get_parts() == [ToolCallPart('tool', 'initial buffered', 'call')]
    event = manager.handle_tool_call_delta(vendor_part_id='tool', args=' accepted')
    assert isinstance(event, PartDeltaEvent)
    assert manager.get_parts() == [ToolCallPart('tool', 'initial buffered accepted', 'call')]


def test_equality_and_repr_materialize_string_buffers():
    """Manager equality and repr are internal state contracts outside provider responses."""

    def build_manager(*, buffered: bool) -> ModelResponsePartsManager:
        manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
        next(manager.handle_text_delta(vendor_part_id='text', content='a' if buffered else 'ab'))
        manager.handle_tool_call_delta(vendor_part_id='tool', args='{' if buffered else '{"value": true}')
        if buffered:
            next(manager.handle_text_delta(vendor_part_id='text', content='b'))
            manager.handle_tool_call_delta(vendor_part_id='tool', args='"value": true}')
        return manager

    assert repr(build_manager(buffered=True)) == repr(build_manager(buffered=False))
    assert build_manager(buffered=True) == build_manager(buffered=False)
    assert build_manager(buffered=True) != object()


def test_thinking_delta_callback_failure_is_atomic():
    """Injected provider-details callback failures are not representable in a cassette."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    next(
        manager.handle_thinking_delta(
            vendor_part_id='thinking',
            content='initial',
            provider_details={'keep': True},
        )
    )
    next(manager.handle_thinking_delta(vendor_part_id='thinking', content=' pending'))

    def fail_provider_details(_details: dict[str, Any] | None) -> dict[str, Any]:
        raise RuntimeError('provider details failed')

    with pytest.raises(RuntimeError, match='provider details failed'):
        next(
            manager.handle_thinking_delta(
                vendor_part_id='thinking',
                content=' leaked',
                provider_details=fail_provider_details,
            )
        )

    assert manager.get_parts() == [ThinkingPart('initial pending', provider_details={'keep': True})]
    next(manager.handle_thinking_delta(vendor_part_id='thinking', content=' accepted'))
    assert manager.get_parts() == [ThinkingPart('initial pending accepted', provider_details={'keep': True})]


@pytest.mark.parametrize('read_method', ['get_parts', 'repr'])
def test_thinking_delta_callback_can_read_materialized_parts(read_method: str):
    """Reentrant reads inside a private callback cannot be exercised by provider cassettes."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    next(
        manager.handle_thinking_delta(
            vendor_part_id='thinking',
            content='initial',
            provider_details={'count': 1},
        )
    )
    next(manager.handle_thinking_delta(vendor_part_id='thinking', content=' pending'))

    observed_reprs: list[str] = []

    def update_provider_details(details: dict[str, Any] | None) -> dict[str, Any]:
        if read_method == 'get_parts':
            observed_reprs.append(repr(manager.get_parts()))
        else:
            observed_reprs.append(repr(manager))
        return {**(details or {}), 'count': (details or {}).get('count', 0) + 1}

    next(
        manager.handle_thinking_delta(
            vendor_part_id='thinking',
            content=' current',
            provider_details=update_provider_details,
        )
    )

    assert len(observed_reprs) == 1
    assert "ThinkingPart(content='initial pending', provider_details={'count': 1})" in observed_reprs[0]
    assert manager.get_parts() == [ThinkingPart('initial pending current', provider_details={'count': 2})]


@pytest.mark.parametrize(
    ('mutation', 'initial_chunks', 'nested_part', 'expected_content'),
    [
        pytest.param(
            'append',
            ('a', 'b'),
            ThinkingPart('abd', provider_details={'count': 1}),
            'abc',
            id='append',
        ),
        pytest.param('replacement', ('a',), ThinkingPart('replacement'), 'ac', id='replacement'),
    ],
)
def test_thinking_delta_nested_callback_mutation_matches_unbuffered_apply(
    mutation: str,
    initial_chunks: tuple[str, ...],
    nested_part: ThinkingPart,
    expected_content: str,
):
    """Nested callback writes are an internal ordering contract outside provider cassettes."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    next(
        manager.handle_thinking_delta(
            vendor_part_id='thinking', content=initial_chunks[0], provider_details={'count': 1}
        )
    )
    for chunk in initial_chunks[1:]:
        next(manager.handle_thinking_delta(vendor_part_id='thinking', content=chunk))

    def update_provider_details(details: dict[str, Any] | None) -> dict[str, Any]:
        if mutation == 'append':
            assert next(manager.handle_thinking_delta(vendor_part_id='thinking', content='d')) == PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta='d')
            )
        else:
            assert manager.handle_part(vendor_part_id='thinking', part=nested_part) == PartStartEvent(
                index=0, part=nested_part
            )
        assert manager.get_parts() == [nested_part]
        assert repr(nested_part) in repr(manager)
        return {**(details or {}), 'count': 2}

    event = next(
        manager.handle_thinking_delta(vendor_part_id='thinking', content='c', provider_details=update_provider_details)
    )
    assert event == PartDeltaEvent(
        index=0, delta=ThinkingPartDelta(content_delta='c', provider_details=update_provider_details)
    )

    original_part = ThinkingPart(''.join(initial_chunks), provider_details={'count': 1})
    expected_part = ThinkingPartDelta(content_delta='c', provider_details={'count': 2}).apply(original_part)
    assert expected_part == ThinkingPart(expected_content, provider_details={'count': 2})
    assert manager.get_parts() == [expected_part]


def test_handle_text_deltas_with_think_tags():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    thinking_tags = ('<think>', '</think>')

    event = next(manager.handle_text_delta(vendor_part_id='content', content='pre-', thinking_tags=thinking_tags))
    assert event == snapshot(
        PartStartEvent(index=0, part=TextPart(content='pre-', part_kind='text'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot([TextPart(content='pre-', part_kind='text')])

    event = next(manager.handle_text_delta(vendor_part_id='content', content='thinking', thinking_tags=thinking_tags))
    assert event == snapshot(
        PartDeltaEvent(
            index=0, delta=TextPartDelta(content_delta='thinking', part_delta_kind='text'), event_kind='part_delta'
        )
    )
    assert manager.get_parts() == snapshot([TextPart(content='pre-thinking', part_kind='text')])

    event = next(manager.handle_text_delta(vendor_part_id='content', content='<think>', thinking_tags=thinking_tags))
    assert event == snapshot(
        PartStartEvent(index=1, part=ThinkingPart(content='', part_kind='thinking'), event_kind='part_start')
    )
    assert manager.get_parts() == snapshot(
        [TextPart(content='pre-thinking', part_kind='text'), ThinkingPart(content='', part_kind='thinking')]
    )

    event = next(manager.handle_text_delta(vendor_part_id='content', content='thinking', thinking_tags=thinking_tags))
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

    event = next(manager.handle_text_delta(vendor_part_id='content', content=' more', thinking_tags=thinking_tags))
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

    events = list(manager.handle_text_delta(vendor_part_id='content', content='</think>', thinking_tags=thinking_tags))
    assert events == []

    event = next(manager.handle_text_delta(vendor_part_id='content', content='post-', thinking_tags=thinking_tags))
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

    event = next(manager.handle_text_delta(vendor_part_id='content', content='thinking', thinking_tags=thinking_tags))
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
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    event = manager.handle_tool_call_delta(vendor_part_id='first', tool_name=None, args='{"arg1":', tool_call_id=None)
    # Not enough information to produce a part, so no event and no part
    assert event == snapshot(None)
    assert manager.get_parts() == snapshot([])

    # Now that we have a tool name, we can produce a part:
    event = manager.handle_tool_call_delta(
        vendor_part_id='first',
        tool_name='tool',
        args=None,
        tool_call_id='call',
        provider_name='test_provider',
        provider_details={'foo': 'bar'},
    )
    assert event == snapshot(
        PartStartEvent(
            index=0,
            part=ToolCallPart(
                tool_name='tool',
                args='{"arg1":',
                tool_call_id='call',
                provider_name='test_provider',
                part_kind='tool-call',
                provider_details={'foo': 'bar'},
            ),
            event_kind='part_start',
        )
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool',
                args='{"arg1":',
                tool_call_id='call',
                provider_name='test_provider',
                part_kind='tool-call',
                provider_details={'foo': 'bar'},
            ),
        ]
    )

    event = manager.handle_tool_call_delta(
        vendor_part_id='first',
        tool_name='1',
        args=None,
        tool_call_id=None,
        provider_name='updated_provider',
        provider_details={'baz': 'qux'},
    )
    assert event == snapshot(
        PartDeltaEvent(
            index=0,
            delta=ToolCallPartDelta(
                tool_name_delta='1',
                args_delta=None,
                provider_name='updated_provider',
                tool_call_id='call',
                part_delta_kind='tool_call',
                provider_details={'baz': 'qux'},
            ),
            event_kind='part_delta',
        )
    )
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(
                tool_name='tool1',
                args='{"arg1":',
                tool_call_id='call',
                provider_name='updated_provider',
                part_kind='tool-call',
                provider_details={'foo': 'bar', 'baz': 'qux'},
            ),
        ]
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
                provider_name='updated_provider',
                provider_details={'foo': 'bar', 'baz': 'qux'},
                part_kind='tool-call',
            )
        ]
    )


def test_handle_tool_call_deltas_without_args():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

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
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
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
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    manager.handle_tool_call_delta(vendor_part_id=None, tool_name='tool2', args='{"arg1":', tool_call_id=None)
    manager.handle_tool_call_delta(vendor_part_id=None, tool_name='tool2', args='"value1"}', tool_call_id=None)
    assert manager.get_parts() == snapshot(
        [
            ToolCallPart(tool_name='tool2', args='{"arg1":', tool_call_id=IsStr(), part_kind='tool-call'),
            ToolCallPart(tool_name='tool2', args='"value1"}', tool_call_id=IsStr(), part_kind='tool-call'),
        ]
    )


def test_handle_tool_call_part():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

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
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    event = next(manager.handle_text_delta(vendor_part_id=text_vendor_part_id, content='hello '))
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

    event = next(manager.handle_text_delta(vendor_part_id=text_vendor_part_id, content='world'))
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
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    list(manager.handle_text_delta(vendor_part_id=1, content='hello'))
    with pytest.raises(
        UnexpectedModelBehavior, match=re.escape('Cannot apply a tool call delta to existing_part=TextPart(')
    ):
        manager.handle_tool_call_delta(vendor_part_id=1, tool_name='tool1', args='{"arg1":', tool_call_id=None)


def test_cannot_convert_from_tool_call_to_text():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
    manager.handle_tool_call_delta(vendor_part_id=1, tool_name='tool1', args='{"arg1":', tool_call_id=None)
    with pytest.raises(
        UnexpectedModelBehavior, match=re.escape('Cannot apply a text delta to existing_part=ToolCallPart(')
    ):
        list(manager.handle_text_delta(vendor_part_id=1, content='hello'))


def test_tool_call_id_delta():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
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
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
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

    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())
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
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    # Add a thinking part first
    event = next(manager.handle_thinking_delta(vendor_part_id='first', content='initial thought', signature=None))
    assert isinstance(event, PartStartEvent)
    assert event.index == 0

    # Now add another thinking delta with no vendor_part_id - should update the latest thinking part
    event = next(manager.handle_thinking_delta(vendor_part_id=None, content=' more', signature=None))
    assert isinstance(event, PartDeltaEvent)
    assert event.index == 0

    parts = manager.get_parts()
    assert parts == snapshot([ThinkingPart(content='initial thought more')])


def test_handle_thinking_delta_wrong_part_type():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    # Add a text part first
    list(manager.handle_text_delta(vendor_part_id='text', content='hello'))

    # Try to apply thinking delta to the text part - should raise error
    with pytest.raises(UnexpectedModelBehavior, match=r'Cannot apply a thinking delta to existing_part='):
        list(manager.handle_thinking_delta(vendor_part_id='text', content='thinking', signature=None))


def test_handle_thinking_delta_new_part_with_vendor_id():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    event = next(manager.handle_thinking_delta(vendor_part_id='thinking', content='new thought', signature=None))
    assert isinstance(event, PartStartEvent)
    assert event.index == 0

    parts = manager.get_parts()
    assert parts == snapshot([ThinkingPart(content='new thought')])


def test_handle_thinking_delta_no_content():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    with pytest.raises(UnexpectedModelBehavior, match='Cannot create a ThinkingPart with no content'):
        list(manager.handle_thinking_delta(vendor_part_id=None, content=None, signature=None))


def test_handle_thinking_delta_no_content_or_signature():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    # Add a thinking part first
    list(manager.handle_thinking_delta(vendor_part_id='thinking', content='initial', signature=None))

    # Updating with no content, signature, or provider_details emits no event
    events = list(manager.handle_thinking_delta(vendor_part_id='thinking', content=None, signature=None))
    assert events == []


def test_handle_thinking_delta_provider_details_callback():
    """Test that provider_details can be a callback function."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    # Create initial part with provider_details
    list(manager.handle_thinking_delta(vendor_part_id='t', content='initial', provider_details={'count': 1}))

    # Update using callback to modify provider_details
    def update_details(existing: dict[str, Any] | None) -> dict[str, Any]:
        details = dict(existing or {})
        details['count'] = details.get('count', 0) + 1
        return details

    list(manager.handle_thinking_delta(vendor_part_id='t', content=' more', provider_details=update_details))

    assert manager.get_parts() == snapshot([ThinkingPart(content='initial more', provider_details={'count': 2})])


def test_handle_thinking_delta_provider_details_callback_from_none():
    """Test callback when existing provider_details is None."""
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    # Create initial part without provider_details
    list(manager.handle_thinking_delta(vendor_part_id='t', content='initial'))

    # Update using callback that handles None
    def add_details(existing: dict[str, Any] | None) -> dict[str, Any]:
        details = dict(existing or {})
        details['new_key'] = 'new_value'
        return details

    list(manager.handle_thinking_delta(vendor_part_id='t', content=' more', provider_details=add_details))

    assert manager.get_parts() == snapshot(
        [ThinkingPart(content='initial more', provider_details={'new_key': 'new_value'})]
    )


def test_handle_part():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    part = NativeToolCallPart(tool_name='tool1', args='{"arg1": ')

    event = manager.handle_part(vendor_part_id='builtin', part=part)
    assert event == snapshot(PartStartEvent(index=0, part=part))
    assert manager.get_parts() == snapshot([part])

    # Add a delta
    event = manager.handle_tool_call_delta(vendor_part_id='builtin', args='"value1"}')
    assert event == snapshot(
        PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='"value1"}', tool_call_id=part.tool_call_id))
    )
    assert manager.get_parts() == snapshot(
        [NativeToolCallPart(tool_name='tool1', args='{"arg1": "value1"}', tool_call_id=part.tool_call_id)]
    )

    # Override it with handle_part
    part2 = NativeToolCallPart(tool_name='tool1', args='{"arg2": ')
    event = manager.handle_part(vendor_part_id='builtin', part=part2)
    assert event == snapshot(PartStartEvent(index=0, part=part2))
    assert manager.get_parts() == snapshot([part2])

    # Finally, demonstrate behavior when no vendor_part_id is provided:
    part3 = NativeToolCallPart(tool_name='tool1', args='{"arg3": ')
    event = manager.handle_part(vendor_part_id=None, part=part3)
    assert event == snapshot(PartStartEvent(index=1, part=part3))
    assert manager.get_parts() == snapshot([part2, part3])


def test_get_part_by_vendor_id():
    manager = ModelResponsePartsManager(model_request_parameters=ModelRequestParameters())

    event = next(manager.handle_text_delta(vendor_part_id='content', content='hello'))
    assert isinstance(event, PartStartEvent)

    part = manager.get_part_by_vendor_id('content')
    assert part == snapshot(TextPart(content='hello', part_kind='text'))

    assert manager.get_part_by_vendor_id('missing') is None
