"""Tool kinds registered by defining typed tool parts, and how parts with unregistered kinds behave."""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Literal

import pytest
from typing_extensions import TypedDict

from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnPart,
    parse_tool_kind,
)
from pydantic_ai.tools import ToolDefinition


class LookupArgs(TypedDict):
    sku: str


class LookupResult(TypedDict):
    in_stock: bool


@dataclass(repr=False)
class LookupCallPart(ToolCallPart):
    _: KW_ONLY

    args: str | LookupArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    tool_kind: Literal['test-lookup'] = 'test-lookup'  # pyright: ignore[reportIncompatibleVariableOverride]


@dataclass(repr=False)
class LookupReturnPart(ToolReturnPart):
    _: KW_ONLY

    content: LookupResult  # pyright: ignore[reportIncompatibleVariableOverride]
    tool_kind: Literal['test-lookup'] = 'test-lookup'  # pyright: ignore[reportIncompatibleVariableOverride]


@dataclass(repr=False)
class NativeLookupCallPart(NativeToolCallPart):
    _: KW_ONLY

    args: str | LookupArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]
    tool_kind: Literal['test-lookup'] = 'test-lookup'  # pyright: ignore[reportIncompatibleVariableOverride]


@dataclass(repr=False)
class _Intermediate(ToolCallPart):
    """Declares no `tool_kind` default, so it registers nothing."""


def _round_trip(messages: list[ModelMessage]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))


def test_a_registered_kind_is_promoted_and_survives_a_round_trip() -> None:
    response = ModelResponse(
        parts=[
            ToolCallPart('lookup_v2', {'sku': 'A-1'}, tool_call_id='c1', tool_kind='test-lookup'),
            NativeToolCallPart('lookup', {'sku': 'A-2'}, tool_call_id='c2', tool_kind='test-lookup'),
        ]
    )
    request = ModelRequest(
        parts=[ToolReturnPart('lookup_v2', {'in_stock': True}, tool_call_id='c1', tool_kind='test-lookup')]
    )

    for messages in ([response, request], _round_trip([response, request])):
        call, native_call = messages[0].parts
        assert isinstance(call, LookupCallPart) and call.tool_name == 'lookup_v2'
        assert isinstance(native_call, NativeLookupCallPart)
        (tool_return,) = messages[1].parts
        assert isinstance(tool_return, LookupReturnPart) and tool_return.content == {'in_stock': True}


def test_json_string_content_is_parsed_before_promotion() -> None:
    part = ToolReturnPart.narrow_type(
        ToolReturnPart('lookup', '{"in_stock": false}', tool_call_id='c1', tool_kind='test-lookup')
    )
    assert isinstance(part, LookupReturnPart) and part.content == {'in_stock': False}


def test_an_unregistered_kind_stays_a_base_part_and_keeps_its_kind() -> None:
    response = ModelResponse(parts=[ToolCallPart('old_tool', {'x': 1}, tool_call_id='c1', tool_kind='test-removed')])
    request = ModelRequest(parts=[ToolReturnPart('old_tool', 'done', tool_call_id='c1', tool_kind='test-removed')])

    loaded = _round_trip([response, request])

    call = loaded[0].parts[0]
    assert type(call) is ToolCallPart and call.tool_kind == 'test-removed'
    tool_return = loaded[1].parts[0]
    assert type(tool_return) is ToolReturnPart and tool_return.tool_kind == 'test-removed'


def test_data_that_does_not_fit_keeps_a_registered_kind_on_a_base_part() -> None:
    response = ModelResponse(parts=[ToolCallPart('lookup', {'wrong': 1}, tool_call_id='c1', tool_kind='test-lookup')])
    call = _round_trip([response])[0].parts[0]
    assert type(call) is ToolCallPart and call.tool_kind == 'test-lookup'


@pytest.mark.parametrize('outcome', ['failed', 'denied', 'interrupted'])
def test_a_return_that_is_not_a_success_is_not_promoted(
    outcome: Literal['failed', 'denied', 'interrupted'],
) -> None:
    custom = ToolReturnPart('lookup', 'boom', tool_call_id='c1', tool_kind='test-lookup', outcome=outcome)
    core = ToolReturnPart('search_tools', 'boom', tool_call_id='c2', tool_kind='tool-search', outcome=outcome)

    loaded = _round_trip([ModelRequest(parts=[custom, core])])[0].parts

    assert [type(part) for part in loaded] == [ToolReturnPart, ToolReturnPart]
    assert [part.tool_kind for part in loaded if isinstance(part, ToolReturnPart)] == ['test-lookup', 'tool-search']
    assert not isinstance(ToolReturnPart.narrow_type(core), ToolSearchReturnPart)


def test_a_kind_registered_twice_is_an_error() -> None:
    with pytest.raises(UserError, match="Tool kind 'test-lookup' is already registered for ToolCallPart"):

        @dataclass(repr=False)
        class OtherLookupCallPart(ToolCallPart):  # pyright: ignore[reportUnusedClass]
            _: KW_ONLY

            tool_kind: Literal['test-lookup'] = 'test-lookup'  # pyright: ignore[reportIncompatibleVariableOverride]


def test_redefining_the_same_class_replaces_its_registration() -> None:
    def define() -> type[ToolCallPart]:
        @dataclass(repr=False)
        class RedefinedCallPart(ToolCallPart):
            _: KW_ONLY

            tool_kind: Literal['test-redefined'] = 'test-redefined'  # pyright: ignore[reportIncompatibleVariableOverride]

        return RedefinedCallPart

    define()
    latest = define()
    promoted = ToolCallPart.narrow_type(ToolCallPart('t', {}, tool_kind='test-redefined'))
    assert type(promoted) is latest


def test_a_typed_part_may_not_add_fields() -> None:
    @dataclass(repr=False)
    class WithExtraField(ToolReturnPart):  # pyright: ignore[reportUnusedClass]
        _: KW_ONLY

        extra: int = 0
        tool_kind: Literal['test-extra-field'] = 'test-extra-field'  # pyright: ignore[reportIncompatibleVariableOverride]

    with pytest.raises(UserError, match='WithExtraField adds the field\\(s\\) extra to ToolReturnPart'):
        ModelRequest(parts=[ToolReturnPart('t', {}, tool_kind='test-extra-field')])


def test_tool_definitions_only_accept_registered_kinds() -> None:
    assert ToolDefinition(name='lookup_v2', tool_kind='test-lookup').tool_kind == 'test-lookup'
    with pytest.raises(UserError, match="declares `tool_kind='test-unknown'`, which no typed tool part has registered"):
        ToolDefinition(name='lookup', tool_kind='test-unknown')


def test_parse_tool_kind_accepts_registered_kinds_only() -> None:
    assert parse_tool_kind('test-lookup') == 'test-lookup'
    assert parse_tool_kind('tool-search') == 'tool-search'
    assert parse_tool_kind('test-unknown') is None


def test_an_intermediate_subclass_registers_nothing() -> None:
    part = ToolCallPart('t', {}, tool_kind='test-unknown')
    assert ToolCallPart.narrow_type(part) is part
    assert not isinstance(part, _Intermediate)
