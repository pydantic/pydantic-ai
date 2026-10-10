"""Tool kinds registered by defining typed tool parts, and how parts with unregistered kinds behave."""

from __future__ import annotations

import pickle
import sys
import textwrap
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import KW_ONLY, dataclass, replace
from typing import Any, ClassVar, Literal, NotRequired

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext, Tool, TypedArgs, TypedContent
from pydantic_ai._event_registry import set_replay_isolation_guard
from pydantic_ai.capabilities import Capability
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
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    narrow_message_parts,
    parse_tool_kind,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset


class LookupArgs(TypedDict):
    sku: str
    quantity: NotRequired[int]


class LookupResult(TypedDict):
    in_stock: bool


class LookupCallPart(ToolCallPart, namespace='test', tool_kind='lookup'):
    typed_args = TypedArgs(LookupArgs)

    label: ClassVar[str] = 'a ClassVar is not a stored field'

    @property
    def sku(self) -> str | None:
        typed = self.typed_args
        return typed['sku'] if typed is not None else None


class LookupReturnPart(ToolReturnPart, namespace='test', tool_kind='lookup'):
    typed_content = TypedContent(LookupResult)


class NativeLookupCallPart(NativeToolCallPart, namespace='test', tool_kind='lookup'):
    typed_args = TypedArgs(LookupArgs)


class AuditCallPart(ToolCallPart, namespace='test', tool_kind='audit'):
    """A kind that declares no arguments shape, so any arguments fit."""


class AuditReturnPart(ToolReturnPart, namespace='test', tool_kind='audit'):
    """Declares no content shape, so any successful content fits."""


@dataclass(repr=False)
class _Intermediate(ToolCallPart):
    """Declares no kind, so it registers nothing."""


def _round_trip(messages: list[ModelMessage]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))


def test_the_kind_is_namespaced_and_becomes_the_default() -> None:
    assert LookupCallPart('lookup_v2', {'sku': 'A-1'}).tool_kind == 'test.lookup'
    assert LookupReturnPart('lookup_v2', {'in_stock': True}).tool_kind == 'test.lookup'
    # As with capability events, the class carries its kind as the field's default.
    assert LookupCallPart.tool_kind == 'test.lookup'
    assert LookupCallPart.label == 'a ClassVar is not a stored field'


def test_a_typed_part_needs_no_decorator_and_keeps_the_base_signature() -> None:
    part = LookupCallPart('lookup', '{"sku": "A-1"}', 'c1')
    assert (part.tool_name, part.args, part.tool_call_id) == ('lookup', '{"sku": "A-1"}', 'c1')
    assert repr(part) == "LookupCallPart(tool_name='lookup', args='{\"sku\": \"A-1\"}', tool_call_id='c1')"
    assert part == LookupCallPart('lookup', '{"sku": "A-1"}', 'c1')
    assert replace(part, args={'sku': 'B-2'}).sku == 'B-2'


def test_a_decorated_typed_part_keeps_its_kind_keyword_only() -> None:
    @dataclass(repr=False)
    class DecoratedCallPart(ToolCallPart, namespace='test', tool_kind='decorated'):
        typed_args = TypedArgs(LookupArgs)

    part = DecoratedCallPart('lookup', {'sku': 'A-1'}, 'c1')
    assert part.tool_kind == 'test.decorated'
    with pytest.raises(TypeError):
        DecoratedCallPart('lookup', {'sku': 'A-1'}, 'c1', 'test.other')  # pyright: ignore[reportCallIssue]


def test_typed_args_validates_complete_arguments_only() -> None:
    assert LookupCallPart('lookup', {'sku': 'A-1', 'quantity': '2'}).typed_args == {'sku': 'A-1', 'quantity': 2}
    assert LookupCallPart('lookup', '{"sku": "A-1"}').typed_args == {'sku': 'A-1'}
    # Arguments still streaming in, or not (yet) matching the declared shape.
    assert LookupCallPart('lookup', '{"sku": "A').typed_args is None
    assert LookupCallPart('lookup', {'quantity': 2}).typed_args is None
    assert LookupCallPart('lookup', '[1]').typed_args is None
    assert LookupCallPart('lookup').typed_args is None
    assert LookupCallPart.typed_args.args_type is LookupArgs


def test_typed_args_on_a_base_part_is_the_arguments_dictionary() -> None:
    assert ToolCallPart('t', '{"a": 1}').typed_args == {'a': 1}
    assert ToolCallPart('t', {'a': 1}).typed_args == {'a': 1}
    assert ToolCallPart('t').typed_args == {}
    assert ToolCallPart('t', '{"a"').typed_args is None
    assert AuditCallPart('audit', {'anything': True}).typed_args == {'anything': True}


def test_typed_content_validates_successful_content_only() -> None:
    assert LookupReturnPart('lookup', {'in_stock': True}).typed_content == {'in_stock': True}
    assert LookupReturnPart('lookup', '{"in_stock": false}').typed_content == {'in_stock': False}
    assert LookupReturnPart('lookup', {'unexpected': 1}).typed_content is None
    assert LookupReturnPart('lookup', 'not json').typed_content is None
    assert LookupReturnPart('lookup', {'in_stock': True}, outcome='failed').typed_content is None
    assert ToolReturnPart('t', 'plain').typed_content == 'plain'
    assert ToolReturnPart('t', 'boom', outcome='denied').typed_content is None
    assert LookupReturnPart.typed_content.content_type is LookupResult


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
        assert isinstance(call, LookupCallPart) and call.tool_name == 'lookup_v2' and call.sku == 'A-1'
        assert isinstance(native_call, NativeLookupCallPart)
        (tool_return,) = messages[1].parts
        assert isinstance(tool_return, LookupReturnPart) and tool_return.typed_content == {'in_stock': True}


def test_core_kinds_are_promoted_on_construction_too() -> None:
    response = ModelResponse(
        parts=[ToolCallPart('search_tools', {'queries': ['weather']}, tool_call_id='c1', tool_kind='tool-search')]
    )
    request = ModelRequest(
        parts=[
            ToolReturnPart(
                'search_tools',
                {'discovered_tools': [{'name': 'get_weather'}]},
                tool_call_id='c1',
                tool_kind='tool-search',
            )
        ]
    )

    (call,) = response.parts
    assert isinstance(call, ToolSearchCallPart) and call.queries == ['weather']
    (tool_return,) = request.parts
    assert isinstance(tool_return, ToolSearchReturnPart) and tool_return.discovered_tools == [{'name': 'get_weather'}]


def test_a_core_kind_whose_data_does_not_fit_is_stripped_on_construction() -> None:
    """The deserialization union would route the part to its typed class, so the unsubstantiated kind goes."""
    response = ModelResponse(
        parts=[ToolCallPart('search_tools', {'wrong': 1}, tool_call_id='c1', tool_kind='tool-search')]
    )
    (call,) = response.parts
    assert type(call) is ToolCallPart and call.tool_kind is None
    assert _round_trip([response]) == [response]


def test_a_typed_part_passes_through_construction_untouched() -> None:
    call = LookupCallPart('lookup', {'sku': 'A-1'})
    parts: list[Any] = [TextPart('checking'), call]
    response = ModelResponse(parts=parts)
    assert response.parts is parts
    assert response.parts[1] is call


def test_promotion_updates_a_parts_list_in_place_and_keeps_args_as_sent() -> None:
    args = {'sku': 'A-1', 'note': 'not part of LookupArgs'}
    parts: list[Any] = [ToolCallPart('lookup', args, tool_kind='test.lookup')]
    response = ModelResponse(parts=parts)
    assert response.parts is parts
    assert isinstance(parts[0], LookupCallPart) and parts[0].args is args

    from_tuple = ModelResponse(parts=(ToolCallPart('lookup', args, tool_kind='test.lookup'),))
    assert isinstance(from_tuple.parts[0], LookupCallPart)


def test_streaming_arguments_are_promoted_and_read_once_complete() -> None:
    response = ModelResponse(parts=[ToolCallPart('lookup', '{"sku": "A', tool_kind='test.lookup')])
    (call,) = response.parts
    assert isinstance(call, LookupCallPart) and call.typed_args is None
    call.args = '{"sku": "A-1"}'
    assert call.sku == 'A-1'


def test_a_speech_part_still_needs_the_right_speaker() -> None:
    with pytest.raises(ValueError, match=r"`SpeechPart` in `ModelResponse\.parts` must have `speaker='assistant'`"):
        ModelResponse(
            parts=[ToolCallPart('lookup', {'sku': 'A-1'}, tool_kind='test.lookup'), SpeechPart(speaker='user')]
        )
    with pytest.raises(ValueError, match=r"`SpeechPart` in `ModelRequest\.parts` must have `speaker='user'`"):
        ModelRequest(parts=[SpeechPart(speaker='assistant')])


def test_narrow_message_parts_promotes_a_kind_set_after_the_message_was_built() -> None:
    request = ModelRequest(parts=[ToolReturnPart('lookup', {'in_stock': True}, tool_call_id='c1')])
    response = ModelResponse(parts=[ToolCallPart('lookup', {'sku': 'A-1'}, tool_call_id='c1')])
    request.parts[0].tool_kind = 'test.lookup'  # pyright: ignore[reportAttributeAccessIssue]
    response.parts[0].tool_kind = 'test.lookup'  # pyright: ignore[reportAttributeAccessIssue]

    narrowed_response, narrowed_request = narrow_message_parts([response, request])

    assert isinstance(narrowed_response.parts[0], LookupCallPart)
    assert isinstance(narrowed_request.parts[0], LookupReturnPart)


def test_json_string_content_is_parsed_before_promotion() -> None:
    part = ToolReturnPart.narrow_type(
        ToolReturnPart('lookup', '{"in_stock": false}', tool_call_id='c1', tool_kind='test.lookup')
    )
    assert isinstance(part, LookupReturnPart) and part.content == {'in_stock': False}


def test_a_kind_without_a_declared_shape_is_promoted_as_is() -> None:
    response = ModelResponse(parts=[ToolCallPart('audit', '{"anything": true}', tool_kind='test.audit')])
    (call,) = response.parts
    assert isinstance(call, AuditCallPart) and call.args == '{"anything": true}'
    request = ModelRequest(parts=[ToolReturnPart('audit', '{"logged": true}', tool_kind='test.audit')])
    (tool_return,) = request.parts
    assert isinstance(tool_return, AuditReturnPart) and tool_return.content == '{"logged": true}'


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
    request = ModelRequest(parts=[ToolReturnPart('lookup', {'wrong': 1}, tool_call_id='c1', tool_kind='test.lookup')])
    for messages in ([response, request], _round_trip([response, request])):
        call = messages[0].parts[0]
        assert type(call) is ToolCallPart and call.tool_kind == 'test.lookup'
        tool_return = messages[1].parts[0]
        assert type(tool_return) is ToolReturnPart and tool_return.tool_kind == 'test.lookup'


@pytest.mark.parametrize('outcome', ['failed', 'denied', 'interrupted'])
def test_a_return_that_is_not_a_success_is_not_promoted(
    outcome: Literal['failed', 'denied', 'interrupted'],
) -> None:
    custom = ToolReturnPart('lookup', 'boom', tool_call_id='c1', tool_kind='test.lookup', outcome=outcome)
    core = ToolReturnPart('search_tools', 'boom', tool_call_id='c2', tool_kind='tool-search', outcome=outcome)

    request = ModelRequest(parts=[custom, core])
    for parts in (request.parts, _round_trip([request])[0].parts):
        assert [type(part) for part in parts] == [ToolReturnPart, ToolReturnPart]
        assert [part.tool_kind for part in parts if isinstance(part, ToolReturnPart)] == ['test.lookup', 'tool-search']
    assert not isinstance(ToolReturnPart.narrow_type(core), ToolSearchReturnPart)


def test_existing_histories_and_pickles_still_load() -> None:
    """Histories and pickles recorded before typed parts declared their shape with descriptors."""
    stored = (
        b'[{"parts":[{"tool_name":"search_tools","args":{"queries":["weather"]},"tool_call_id":"c1",'
        b'"tool_kind":"tool-search","id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"},'
        b'{"tool_name":"run_code","args":"{\\"code\\": \\"1\\"}","tool_call_id":"c2","tool_kind":"other.removed",'
        b'"id":null,"provider_name":null,"provider_details":null,"part_kind":"tool-call"}],'
        b'"usage":{},"model_name":null,"timestamp":"2026-01-01T00:00:00Z","kind":"response"}]'
    )
    (response,) = ModelMessagesTypeAdapter.validate_json(stored)
    search, removed = response.parts
    assert isinstance(search, ToolSearchCallPart) and search.typed_args == {'queries': ['weather']}
    assert type(removed) is ToolCallPart and removed.tool_kind == 'other.removed'

    messages: list[ModelMessage] = [
        response,
        ModelResponse(parts=[ToolCallPart('lookup', {'sku': 'A-1'}, tool_kind='test.lookup')]),
    ]
    assert pickle.loads(pickle.dumps(messages)) == messages
    tool_def = ToolDefinition(name='lookup', tool_kind=LookupCallPart)
    assert pickle.loads(pickle.dumps(tool_def)) == tool_def


def test_a_kind_needs_a_namespace() -> None:
    with pytest.raises(UserError, match='needs a namespace'):

        class Unnamespaced(ToolCallPart, tool_kind='lookup'):  # pyright: ignore[reportUnusedClass]
            pass

    with pytest.raises(UserError, match='has an empty `tool_kind`'):

        class EmptyKind(ToolCallPart, namespace='test', tool_kind=''):  # pyright: ignore[reportUnusedClass]
            pass


def test_a_kind_must_be_declared_with_class_arguments() -> None:
    with pytest.raises(UserError, match='must declare its kind with class arguments'):

        @dataclass(repr=False)
        class BodyDefault(ToolCallPart):
            _: KW_ONLY

            tool_kind: Literal['test.body'] = 'test.body'  # pyright: ignore[reportIncompatibleVariableOverride]

    with pytest.raises(UserError, match='must declare its kind with class arguments'):

        class NamespaceOnly(ToolCallPart, namespace='test'):  # pyright: ignore[reportUnusedClass]
            pass


def test_a_kind_registered_twice_is_an_error() -> None:
    with pytest.raises(UserError, match=r"Tool kind 'test\.lookup' is already registered for ToolCallPart"):

        class OtherLookupCallPart(ToolCallPart, namespace='test', tool_kind='lookup'):  # pyright: ignore[reportUnusedClass]
            pass


def test_redefining_the_same_class_replaces_its_registration() -> None:
    def define() -> type[ToolCallPart]:
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

        class WithExtraField(ToolReturnPart, namespace='test', tool_kind='extra-field'):  # pyright: ignore[reportUnusedClass]
            extra: int = 0


def test_a_field_inherited_from_an_intermediate_dataclass_is_rejected() -> None:
    @dataclass(repr=False)
    class WithField(ToolCallPart):
        _: KW_ONLY

        extra: int = 0

    with pytest.raises(UserError, match='InheritsField adds the field\\(s\\) extra to ToolCallPart'):

        class InheritsField(WithField, namespace='test', tool_kind='inherits-field'):  # pyright: ignore[reportUnusedClass]
            pass


def test_a_shape_defined_in_a_function_works() -> None:
    class LocalArgs(TypedDict):
        name: str

    class LocalCallPart(ToolCallPart, namespace='test', tool_kind='local-args'):
        typed_args = TypedArgs(LocalArgs)

    (call,) = ModelResponse(parts=[ToolCallPart('t', {'name': 'x'}, tool_kind='test.local-args')]).parts
    assert isinstance(call, LocalCallPart) and call.typed_args == {'name': 'x'}


def test_a_typed_part_with_lazy_annotations() -> None:
    """On Python 3.14+ a module without `from __future__ import annotations` keeps annotations lazy (PEP 649).

    The class is built from source, as this module's annotations are strings; before 3.14 they're eager.
    """
    namespace: dict[str, Any] = {'ToolCallPart': ToolCallPart, 'ClassVar': ClassVar}
    source = textwrap.dedent(
        """
        class Label:
            pass

        class LazyCallPart(ToolCallPart, namespace='test', tool_kind='lazy'):
            label: ClassVar[Label]
        """
    )
    # `dont_inherit` keeps this module's `from __future__ import annotations` out of the compiled source.
    exec(compile(source, '<lazy>', 'exec', dont_inherit=True), namespace)
    lazy_cls = namespace['LazyCallPart']
    assert lazy_cls('t').tool_kind == 'test.lazy'
    if sys.version_info >= (3, 14):
        assert set(lazy_cls.__annotations__) == {'label', '_', 'tool_kind'}


def test_a_stored_tool_definition_with_an_unregistered_kind_loads() -> None:
    adapter = TypeAdapter(ToolDefinition)
    stored = adapter.dump_python(ToolDefinition(name='lookup', tool_kind='test.unknown'))
    assert stored['tool_kind'] == 'test.unknown'
    assert adapter.validate_python(stored).tool_kind == 'test.unknown'
    assert adapter.json_schema()['properties']['tool_kind'] == {
        'anyOf': [{'type': 'string'}, {'type': 'null'}],
        'default': None,
        'title': 'Tool Kind',
    }


def test_a_tool_definition_takes_the_typed_part_class() -> None:
    tool_def = ToolDefinition(name='lookup', tool_kind=LookupCallPart)
    assert tool_def.tool_kind == 'test.lookup'
    assert replace(tool_def, tool_kind=LookupReturnPart).tool_kind == 'test.lookup'
    assert replace(tool_def, tool_kind=None).tool_kind is None
    assert ToolDefinition(name='search_tools', tool_kind=ToolSearchCallPart).tool_kind == 'tool-search'
    assert ToolDefinition(name='lookup').tool_kind is None
    with pytest.raises(UserError, match='`ToolCallPart` registers no tool kind'):
        ToolDefinition(name='lookup', tool_kind=ToolCallPart)


async def test_a_run_refuses_a_tool_with_an_unregistered_kind() -> None:
    async def mark_kind(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, tool_kind='test.unknown')

    def lookup(sku: str) -> str:
        return sku  # pragma: no cover

    agent = Agent(TestModel(), tools=[Tool(lookup, prepare=mark_kind)])
    with pytest.raises(
        UserError, match=r"declares `tool_kind='test\.unknown'`, which no typed tool part has registered"
    ):
        await agent.run('go')


def _missing_sku(code: str) -> str:
    return code  # pragma: no cover


def _optional_sku(sku: str = '') -> str:
    return sku  # pragma: no cover


def _numeric_sku(sku: int) -> str:
    return str(sku)  # pragma: no cover


def _string_quantity(sku: str | None, quantity: str) -> str:
    return f'{sku} {quantity}'  # pragma: no cover


@pytest.mark.parametrize(
    'function,problem',
    [
        (_missing_sku, "it has no 'sku' parameter"),
        (_optional_sku, "its 'sku' parameter is not required"),
        (_numeric_sku, "its 'sku' parameter is of type integer, not string"),
        (_string_quantity, "its 'quantity' parameter is of type string, not integer"),
    ],
)
async def test_a_run_refuses_a_tool_whose_parameters_do_not_fit_its_kind(function: Any, problem: str) -> None:
    agent = Agent(TestModel(), tools=[Tool(function, name='lookup', tool_kind=LookupCallPart)])
    with pytest.raises(UserError) as exc_info:
        await agent.run('go')
    message = str(exc_info.value)
    assert message.startswith(
        "Tool 'lookup' declares `tool_kind='test.lookup'`, but its parameters don't fit the arguments "
        '`LookupCallPart` declares (LookupArgs): '
    )
    assert problem in message


async def test_a_numeric_parameter_fits_an_integer_field() -> None:
    def lookup(sku: str, quantity: float = 1) -> LookupResult:
        return _lookup(sku)

    agent = Agent(TestModel(), tools=[Tool(lookup, tool_kind=LookupCallPart)])
    result = await agent.run('go')
    assert isinstance(result.all_messages()[1].parts[0], LookupCallPart)


async def test_a_kind_without_a_declared_shape_accepts_any_parameters() -> None:
    def audit(entry: int) -> str:
        return str(entry)

    agent = Agent(TestModel(), tools=[Tool(audit, tool_kind=AuditCallPart)])
    result = await agent.run('go')
    assert isinstance(result.all_messages()[1].parts[0], AuditCallPart)


async def test_a_kind_registered_only_by_a_return_part_is_not_checked() -> None:
    class ReportReturnPart(ToolReturnPart, namespace='test', tool_kind='report'):
        typed_content = TypedContent(LookupResult)

    def report(sku: str) -> LookupResult:
        return {'in_stock': sku == 'a'}

    agent = Agent(TestModel(), tools=[Tool(report, tool_kind=ReportReturnPart)])
    result = await agent.run('go')
    tool_return = result.all_messages()[2].parts[0]
    assert isinstance(tool_return, ReportReturnPart) and tool_return.typed_content == {'in_stock': True}


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


def _lookup(sku: str) -> LookupResult:
    return {'in_stock': sku == 'A-1'}


async def _respond(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
    if len(messages) == 1:
        yield {0: DeltaToolCall(name='lookup', json_args='{"sku": "A-1"}', tool_call_id='c1')}
    else:
        yield 'done'


async def _assert_typed_lookup_parts(agent: Agent[Any, str]) -> None:
    events: list[AgentStreamEvent] = []

    async def collect(ctx: RunContext[Any], stream: AsyncIterable[AgentStreamEvent]) -> None:
        async for event in stream:
            events.append(event)

    result = await agent.run('Is A-1 in stock?', event_stream_handler=collect)

    call = result.all_messages()[1].parts[0]
    assert isinstance(call, LookupCallPart) and call.sku == 'A-1'
    tool_return = result.all_messages()[2].parts[0]
    assert isinstance(tool_return, LookupReturnPart) and tool_return.typed_content == {'in_stock': True}
    assert [type(event.part) for event in events if isinstance(event, FunctionToolCallEvent)] == [LookupCallPart]
    assert [type(event.part) for event in events if isinstance(event, FunctionToolResultEvent)] == [LookupReturnPart]


async def test_a_tool_declares_its_kind_with_the_part_class() -> None:
    await _assert_typed_lookup_parts(
        Agent(FunctionModel(stream_function=_respond), tools=[Tool(_lookup, name='lookup', tool_kind=LookupCallPart)])
    )


async def test_agent_decorators_declare_a_kind() -> None:
    agent = Agent(FunctionModel(stream_function=_respond))

    @agent.tool_plain(tool_kind=LookupCallPart)
    def lookup(sku: str) -> LookupResult:
        return _lookup(sku)

    await _assert_typed_lookup_parts(agent)

    @agent.tool(tool_kind='test.lookup')
    def lookup_with_ctx(ctx: RunContext[object], sku: str) -> LookupResult:
        return _lookup(sku)  # pragma: no cover

    assert agent._function_toolset.tools['lookup_with_ctx'].tool_kind == 'test.lookup'  # pyright: ignore[reportPrivateUsage]


async def test_toolset_and_capability_decorators_declare_a_kind() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(tool_kind=LookupCallPart)
    def lookup(sku: str) -> LookupResult:
        return _lookup(sku)

    @toolset.tool(tool_kind=LookupCallPart)
    def lookup_with_ctx(ctx: RunContext[None], sku: str) -> LookupResult:
        return _lookup(sku)  # pragma: no cover

    assert toolset.tools['lookup_with_ctx'].tool_kind == 'test.lookup'
    await _assert_typed_lookup_parts(Agent(FunctionModel(stream_function=_respond), toolsets=[toolset]))

    capability: Capability[None] = Capability()

    @capability.tool_plain(tool_kind=LookupCallPart)
    def lookup_from_capability(sku: str) -> LookupResult:
        return _lookup(sku)  # pragma: no cover

    @capability.tool(tool_kind=LookupCallPart)
    def lookup_from_capability_with_ctx(ctx: RunContext[None], sku: str) -> LookupResult:
        return _lookup(sku)  # pragma: no cover

    capability_tools = capability._function_toolset.tools  # pyright: ignore[reportPrivateUsage]
    assert {tool.tool_kind for tool in capability_tools.values()} == {'test.lookup'}


async def test_an_agent_tool_with_a_kind_set_in_prepare_produces_typed_parts() -> None:
    async def mark_kind(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, tool_kind=LookupCallPart)

    await _assert_typed_lookup_parts(
        Agent(FunctionModel(stream_function=_respond), tools=[Tool(_lookup, name='lookup', prepare=mark_kind)])
    )
