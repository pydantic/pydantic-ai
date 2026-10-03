"""Tool kinds registered by defining typed tool parts, and how parts with unregistered kinds behave."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import KW_ONLY, dataclass, replace
from typing import ClassVar, Literal

import pytest
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai._event_registry import set_replay_isolation_guard
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchReturnPart,
    parse_tool_kind,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import ToolDefinition


class LookupArgs(TypedDict):
    sku: str


class LookupResult(TypedDict):
    in_stock: bool


@dataclass(repr=False)
class LookupCallPart(ToolCallPart, namespace='test', tool_kind='lookup'):
    _: KW_ONLY

    args: str | LookupArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    label: ClassVar[str] = 'a ClassVar is not a stored field'


@dataclass(repr=False)
class LookupReturnPart(ToolReturnPart, namespace='test', tool_kind='lookup'):
    _: KW_ONLY

    content: LookupResult


@dataclass(repr=False)
class NativeLookupCallPart(NativeToolCallPart, namespace='test', tool_kind='lookup'):
    _: KW_ONLY

    args: str | LookupArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]


@dataclass(repr=False)
class _Intermediate(ToolCallPart):
    """Declares no kind, so it registers nothing."""


def _round_trip(messages: list[ModelMessage]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))


def test_the_kind_is_namespaced_and_becomes_the_default() -> None:
    assert LookupCallPart('lookup_v2', args={'sku': 'A-1'}).tool_kind == 'test.lookup'
    assert LookupReturnPart('lookup_v2', content={'in_stock': True}).tool_kind == 'test.lookup'


def test_a_registered_kind_is_promoted_and_survives_a_round_trip() -> None:
    response = ModelResponse(
        parts=[
            TextPart('checking'),
            ToolCallPart('lookup_v2', {'sku': 'A-1'}, tool_call_id='c1', tool_kind='test.lookup'),
            NativeToolCallPart('lookup', {'sku': 'A-2'}, tool_call_id='c2', tool_kind='test.lookup'),
        ]
    )
    request = ModelRequest(
        parts=[ToolReturnPart('lookup_v2', {'in_stock': True}, tool_call_id='c1', tool_kind='test.lookup')]
    )

    for messages in ([response, request], _round_trip([response, request])):
        text, call, native_call = messages[0].parts
        assert isinstance(text, TextPart)
        assert isinstance(call, LookupCallPart) and call.tool_name == 'lookup_v2'
        assert isinstance(native_call, NativeLookupCallPart)
        (tool_return,) = messages[1].parts
        assert isinstance(tool_return, LookupReturnPart) and tool_return.content == {'in_stock': True}


def test_json_string_content_is_parsed_before_promotion() -> None:
    part = ToolReturnPart.narrow_type(
        ToolReturnPart('lookup', '{"in_stock": false}', tool_call_id='c1', tool_kind='test.lookup')
    )
    assert isinstance(part, LookupReturnPart) and part.content == {'in_stock': False}


def test_an_unregistered_kind_stays_a_base_part_and_keeps_its_kind() -> None:
    response = ModelResponse(parts=[ToolCallPart('old_tool', {'x': 1}, tool_call_id='c1', tool_kind='test.removed')])
    request = ModelRequest(parts=[ToolReturnPart('old_tool', 'done', tool_call_id='c1', tool_kind='test.removed')])

    loaded = _round_trip([response, request])

    call = loaded[0].parts[0]
    assert type(call) is ToolCallPart and call.tool_kind == 'test.removed'
    tool_return = loaded[1].parts[0]
    assert type(tool_return) is ToolReturnPart and tool_return.tool_kind == 'test.removed'


def test_data_that_does_not_fit_keeps_a_registered_kind_on_a_base_part() -> None:
    response = ModelResponse(parts=[ToolCallPart('lookup', {'wrong': 1}, tool_call_id='c1', tool_kind='test.lookup')])
    call = _round_trip([response])[0].parts[0]
    assert type(call) is ToolCallPart and call.tool_kind == 'test.lookup'


@pytest.mark.parametrize('outcome', ['failed', 'denied', 'interrupted'])
def test_a_return_that_is_not_a_success_is_not_promoted(
    outcome: Literal['failed', 'denied', 'interrupted'],
) -> None:
    custom = ToolReturnPart('lookup', 'boom', tool_call_id='c1', tool_kind='test.lookup', outcome=outcome)
    core = ToolReturnPart('search_tools', 'boom', tool_call_id='c2', tool_kind='tool-search', outcome=outcome)

    loaded = _round_trip([ModelRequest(parts=[custom, core])])[0].parts

    assert [type(part) for part in loaded] == [ToolReturnPart, ToolReturnPart]
    assert [part.tool_kind for part in loaded if isinstance(part, ToolReturnPart)] == ['test.lookup', 'tool-search']
    assert not isinstance(ToolReturnPart.narrow_type(core), ToolSearchReturnPart)


def test_a_kind_needs_a_namespace() -> None:
    with pytest.raises(UserError, match='needs a namespace'):

        @dataclass(repr=False)
        class Unnamespaced(ToolCallPart, tool_kind='lookup'):
            pass

    with pytest.raises(UserError, match='has an empty `tool_kind`'):

        @dataclass(repr=False)
        class EmptyKind(ToolCallPart, namespace='test', tool_kind=''):
            pass


def test_a_kind_must_be_declared_with_class_arguments() -> None:
    with pytest.raises(UserError, match='must declare its kind with class arguments'):

        @dataclass(repr=False)
        class BodyDefault(ToolCallPart):
            _: KW_ONLY

            tool_kind: Literal['test.body'] = 'test.body'  # pyright: ignore[reportIncompatibleVariableOverride]

    with pytest.raises(UserError, match='must declare its kind with class arguments'):

        @dataclass(repr=False)
        class NamespaceOnly(ToolCallPart, namespace='test'):
            pass


def test_a_kind_registered_twice_is_an_error() -> None:
    with pytest.raises(UserError, match=r"Tool kind 'test\.lookup' is already registered for ToolCallPart"):

        @dataclass(repr=False)
        class OtherLookupCallPart(ToolCallPart, namespace='test', tool_kind='lookup'):
            pass


def test_redefining_the_same_class_replaces_its_registration() -> None:
    def define() -> type[ToolCallPart]:
        @dataclass(repr=False)
        class RedefinedCallPart(ToolCallPart, namespace='test', tool_kind='redefined'):
            pass

        return RedefinedCallPart

    define()
    latest = define()
    promoted = ToolCallPart.narrow_type(ToolCallPart('t', {}, tool_kind='test.redefined'))
    assert type(promoted) is latest


def test_a_re_executed_copy_keeps_the_host_registration() -> None:
    """A Temporal workflow sandbox re-executes app modules; its copy must not displace the host's class."""

    def define() -> type[ToolCallPart]:
        @dataclass(repr=False)
        class SandboxedCallPart(ToolCallPart, namespace='test', tool_kind='sandboxed'):
            pass

        return SandboxedCallPart

    host_cls = define()
    set_replay_isolation_guard(lambda: True)
    try:
        copy_cls = define()
        assert copy_cls is not host_cls
        assert type(ToolCallPart.narrow_type(ToolCallPart('t', {}, tool_kind='test.sandboxed'))) is host_cls
    finally:
        set_replay_isolation_guard(lambda: False)


def test_a_typed_part_may_not_add_fields() -> None:
    with pytest.raises(UserError, match='WithExtraField adds the field\\(s\\) extra to ToolReturnPart'):

        @dataclass(repr=False)
        class WithExtraField(ToolReturnPart, namespace='test', tool_kind='extra-field'):
            _: KW_ONLY

            extra: int = 0


def test_a_field_inherited_from_an_intermediate_dataclass_is_rejected() -> None:
    @dataclass(repr=False)
    class WithField(ToolCallPart):
        _: KW_ONLY

        extra: int = 0

    with pytest.raises(UserError, match='InheritsField adds the field\\(s\\) extra to ToolCallPart'):

        @dataclass(repr=False)
        class InheritsField(WithField, namespace='test', tool_kind='inherits-field'):
            pass


def test_a_payload_type_defined_in_a_function_is_reported() -> None:
    class LocalArgs(TypedDict):
        name: str

    @dataclass(repr=False)
    class LocalCallPart(ToolCallPart, namespace='test', tool_kind='local-args'):
        _: KW_ONLY

        args: str | LocalArgs | None = None  # pyright: ignore[reportIncompatibleVariableOverride]

    with pytest.raises(UserError, match='Define the types it names at module level'):
        ModelResponse(parts=[ToolCallPart('t', {'name': 'x'}, tool_kind='test.local-args')])


def test_tool_definitions_only_accept_registered_kinds() -> None:
    assert ToolDefinition(name='lookup_v2', tool_kind='test.lookup').tool_kind == 'test.lookup'
    with pytest.raises(
        UserError, match=r"declares `tool_kind='test\.unknown'`, which no typed tool part has registered"
    ):
        ToolDefinition(name='lookup', tool_kind='test.unknown')


def test_parse_tool_kind_accepts_registered_kinds_only() -> None:
    assert parse_tool_kind('test.lookup') == 'test.lookup'
    assert parse_tool_kind('tool-search') == 'tool-search'
    assert parse_tool_kind('test.unknown') is None


def test_an_intermediate_subclass_registers_nothing() -> None:
    part = ToolCallPart('t', {}, tool_kind='test.unknown')
    assert ToolCallPart.narrow_type(part) is part
    assert not isinstance(part, _Intermediate)


def test_a_slotted_typed_part_registers_the_recreated_class() -> None:
    # `@dataclass(slots=True)` builds a new class, re-running registration without the class arguments.
    @dataclass(repr=False, slots=True)
    class SlottedCallPart(ToolCallPart, namespace='test', tool_kind='slotted'):
        pass

    promoted = ToolCallPart.narrow_type(ToolCallPart('t', {}, tool_kind='test.slotted'))
    assert type(promoted) is SlottedCallPart
    assert SlottedCallPart('t').tool_kind == 'test.slotted'


async def test_an_agent_tool_with_a_registered_kind_produces_typed_parts() -> None:
    async def respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
        if len(messages) == 1:
            yield {0: DeltaToolCall(name='lookup', json_args='{"sku": "A-1"}', tool_call_id='c1')}
        else:
            yield 'done'

    async def mark_kind(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, tool_kind='test.lookup')

    def lookup(sku: str) -> LookupResult:
        return {'in_stock': sku == 'A-1'}

    events: list[AgentStreamEvent] = []

    async def collect(ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    agent = Agent(FunctionModel(stream_function=respond), tools=[Tool(lookup, prepare=mark_kind)])
    result = await agent.run('Is A-1 in stock?', event_stream_handler=collect)

    call = result.all_messages()[1].parts[0]
    assert isinstance(call, LookupCallPart) and call.args_as_dict() == {'sku': 'A-1'}
    tool_return = result.all_messages()[2].parts[0]
    assert isinstance(tool_return, LookupReturnPart) and tool_return.content == {'in_stock': True}
    assert [type(event.part) for event in events if isinstance(event, FunctionToolCallEvent)] == [LookupCallPart]
    assert [type(event.part) for event in events if isinstance(event, FunctionToolResultEvent)] == [LookupReturnPart]
