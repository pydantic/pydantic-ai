"""Tool kinds, the typed tool parts registered for them, and how parts with undefined kinds behave."""

from __future__ import annotations

import pickle
import sys
import textwrap
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import KW_ONLY, dataclass, replace
from typing import Any, ClassVar, Generic, Literal, NotRequired, TypeVar

import pytest
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext, Tool, ToolKind, tools
from pydantic_ai._event_registry import set_replay_isolation_guard
from pydantic_ai._tool_search import TOOL_SEARCH_KIND
from pydantic_ai._warnings import PydanticAIDeprecationWarning
from pydantic_ai.capabilities import (
    AbstractCapability,
    Capability,
    HandleDeferredToolCalls,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import ToolFailed, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ImageUrl,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    SpeechPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    ToolSearchCallPart,
    ToolSearchReturnPart,
    narrow_message_parts,
    parse_tool_kind,
    sanitize_messages,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tool_manager import ToolManager
from pydantic_ai.tools import DeferredToolRequests, DeferredToolResults, ToolDefinition, ToolDenied
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage


class LookupArgs(TypedDict):
    sku: str
    quantity: NotRequired[int]


class LookupResult(TypedDict):
    in_stock: bool


class ImageResult(TypedDict):
    image: ImageUrl


lookup_kind = ToolKind('test', 'lookup', args=LookupArgs, content=LookupResult)
audit_kind = ToolKind('test', 'audit')
image_kind = ToolKind('test', 'image', content=ImageResult)


class LookupCallPart(ToolCallPart[LookupArgs], kind=lookup_kind):
    label: ClassVar[str] = 'a ClassVar is not a stored field'

    @property
    def sku(self) -> str | None:
        typed = self.typed_args
        return typed['sku'] if typed is not None else None


class LookupReturnPart(ToolReturnPart[LookupResult], kind=lookup_kind):
    pass


class NativeLookupCallPart(NativeToolCallPart[LookupArgs], kind=lookup_kind):
    pass


class AuditCallPart(ToolCallPart, kind=audit_kind):
    """A kind that declares no arguments shape, so any arguments fit."""


class AuditReturnPart(ToolReturnPart, kind=audit_kind):
    """Declares no content shape, so any successful content fits."""


class ImageReturnPart(ToolReturnPart[ImageResult], kind=image_kind):
    pass


@dataclass(repr=False)
class _Intermediate(ToolCallPart):
    """Declares no kind, so it registers nothing."""


@dataclass(repr=False)
class _NativeIntermediateCall(NativeToolCallPart):
    """Declares no kind, so it registers nothing."""


@dataclass(repr=False)
class _NativeIntermediateReturn(NativeToolReturnPart):
    """Declares no kind, so it registers nothing."""


def _round_trip(messages: list[ModelMessage]) -> list[ModelMessage]:
    return ModelMessagesTypeAdapter.validate_json(ModelMessagesTypeAdapter.dump_json(messages))


def test_a_kind_is_namespaced_and_carries_its_shapes() -> None:
    assert (lookup_kind.namespace, lookup_kind.name, lookup_kind.id) == ('test', 'lookup', 'test.lookup')
    assert (lookup_kind.args, lookup_kind.content) == (LookupArgs, LookupResult)
    assert (audit_kind.args, audit_kind.content) == (None, None)
    assert repr(lookup_kind) == "ToolKind('test', 'lookup', args=LookupArgs, content=LookupResult)"
    assert repr(audit_kind) == "ToolKind('test', 'audit')"
    assert (TOOL_SEARCH_KIND.namespace, TOOL_SEARCH_KIND.id) == (None, 'tool-search')
    assert repr(TOOL_SEARCH_KIND) == "ToolKind('tool-search', args=ToolSearchArgs, content=ToolSearchReturnContent)"
    assert repr(ToolKind('test', 'list-args', args=list[int])) == "ToolKind('test', 'list-args', args=list[int])"


def test_the_kind_becomes_the_default() -> None:
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
    class DecoratedCallPart(ToolCallPart[LookupArgs], kind=ToolKind('test', 'decorated', args=LookupArgs)):
        pass

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
    assert AuditReturnPart('audit', 'plain').typed_content == 'plain'


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
    return_part = ToolReturnPart('lookup', {'in_stock': True}, tool_call_id='c1')
    call_part = ToolCallPart('lookup', {'sku': 'A-1'}, tool_call_id='c1')
    request = ModelRequest(parts=[return_part])
    response = ModelResponse(parts=[call_part])
    return_part.tool_kind = 'test.lookup'
    call_part.tool_kind = 'test.lookup'

    narrowed_response, narrowed_request = narrow_message_parts([response, request])

    assert isinstance(narrowed_response.parts[0], LookupCallPart)
    assert isinstance(narrowed_request.parts[0], LookupReturnPart)


def test_json_string_content_is_validated_but_kept_as_sent() -> None:
    part = ToolReturnPart.narrow_type(
        ToolReturnPart('lookup', '{"in_stock": false}', tool_call_id='c1', tool_kind='test.lookup')
    )
    assert isinstance(part, LookupReturnPart)
    assert part.content == '{"in_stock": false}'
    assert part.typed_content == {'in_stock': False}


def test_a_json_string_content_history_round_trips_byte_for_byte() -> None:
    """Promotion doesn't rewrite the content, so what's dumped (and sent back to the model) doesn't change."""
    request = ModelRequest(
        parts=[ToolReturnPart('lookup', '{"in_stock":  true}', tool_call_id='c1', tool_kind='test.lookup')]
    )
    assert isinstance(request.parts[0], LookupReturnPart)
    dumped = ModelMessagesTypeAdapter.dump_json([request])
    reloaded = ModelMessagesTypeAdapter.validate_json(dumped)
    assert isinstance(reloaded[0].parts[0], LookupReturnPart)
    assert ModelMessagesTypeAdapter.dump_json(reloaded) == dumped


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
    tool_def = ToolDefinition(name='lookup', tool_kind=lookup_kind)
    assert pickle.loads(pickle.dumps(tool_def)) == tool_def


def test_a_kind_needs_a_namespace_and_a_name() -> None:
    with pytest.raises(UserError, match='needs a namespace'):
        ToolKind('', 'lookup')
    with pytest.raises(UserError, match='needs a namespace'):
        ToolKind('test..nested', 'lookup')
    with pytest.raises(UserError, match="Tool kind in namespace 'test' has an empty name"):
        ToolKind('test', '')


def test_a_kind_defined_again_must_keep_its_shapes() -> None:
    assert ToolKind('test', 'lookup', args=LookupArgs, content=LookupResult).id == 'test.lookup'
    with pytest.raises(UserError, match=r"Tool kind 'test\.lookup' is already defined as ToolKind\('test', 'lookup', "):
        ToolKind('test', 'lookup', args=ImageResult)


def test_a_kind_set_in_the_class_body_is_deprecated_and_registers_nothing() -> None:
    """Before `ToolKind`, a subclass could pin `tool_kind` in its body; it keeps working as a plain subclass."""
    with pytest.warns(PydanticAIDeprecationWarning, match='Declare its kind with the `kind` class argument'):

        @dataclass(repr=False)
        class BodyDefault(ToolCallPart):
            _: KW_ONLY

            tool_kind: Literal['test.body'] = 'test.body'  # pyright: ignore[reportIncompatibleVariableOverride]

    assert BodyDefault('t').tool_kind == 'test.body'
    part = ToolCallPart('t', {}, tool_kind='test.body')
    assert ToolCallPart.narrow_type(part) is part


def test_a_kind_registered_twice_is_an_error() -> None:
    with pytest.raises(UserError, match=r"Tool kind 'test\.lookup' already has a typed ToolCallPart subclass"):

        class OtherLookupCallPart(ToolCallPart[LookupArgs], kind=lookup_kind):  # pyright: ignore[reportUnusedClass]
            pass


_T = TypeVar('_T')


def test_the_parameterized_base_must_match_the_kind() -> None:
    with pytest.raises(
        UserError,
        match=r"MismatchedCallPart parameterizes ToolCallPart with ImageResult, but its kind 'test\.lookup' declares `args=LookupArgs`",
    ):

        class MismatchedCallPart(ToolCallPart[ImageResult], kind=lookup_kind):  # pyright: ignore[reportUnusedClass,reportGeneralTypeIssues,reportArgumentType]
            pass

    with pytest.raises(
        UserError,
        match=r"ShapelessReturnPart parameterizes ToolReturnPart with LookupResult, but its kind 'test\.audit' declares `content=None`",
    ):

        class ShapelessReturnPart(ToolReturnPart[LookupResult], kind=audit_kind):  # pyright: ignore[reportUnusedClass]
            pass

    class AnyCallPart(ToolCallPart[Any], kind=ToolKind('test', 'any-shape')):
        pass

    class GenericCallPart(ToolCallPart[_T], Generic[_T]):
        pass

    class ViaGenericCallPart(GenericCallPart[LookupArgs], kind=ToolKind('test', 'via-generic', args=LookupArgs)):
        pass

    class OpenCallPart(ToolCallPart[_T], Generic[_T], kind=ToolKind('test', 'open', args=LookupArgs)):
        pass

    assert AnyCallPart('t').tool_kind == 'test.any-shape'
    assert ViaGenericCallPart('t', {'sku': 'A-1'}).typed_args == {'sku': 'A-1'}
    assert OpenCallPart('t').tool_kind == 'test.open'


def test_redefining_the_same_class_replaces_its_registration() -> None:
    def define() -> type[ToolCallPart]:
        class RedefinedCallPart(ToolCallPart, kind=ToolKind('test', 'redefined')):
            pass

        return RedefinedCallPart

    define()
    latest = define()
    promoted = ToolCallPart.narrow_type(ToolCallPart('t', {}, tool_kind='test.redefined'))
    assert type(promoted) is latest


def test_a_re_executed_copy_keeps_the_host_registration() -> None:
    """A Temporal workflow sandbox re-executes app modules; its copy must not displace the host's class."""

    def define() -> type[ToolCallPart]:
        class SandboxedCallPart(ToolCallPart, kind=ToolKind('test', 'sandboxed')):
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

        class WithExtraField(ToolReturnPart, kind=ToolKind('test', 'extra-field')):  # pyright: ignore[reportUnusedClass]
            extra: int = 0


def test_a_field_inherited_from_an_intermediate_dataclass_is_rejected() -> None:
    @dataclass(repr=False)
    class WithField(ToolCallPart):
        _: KW_ONLY

        extra: int = 0

    with pytest.raises(UserError, match='InheritsField adds the field\\(s\\) extra to ToolCallPart'):

        class InheritsField(WithField, kind=ToolKind('test', 'inherits-field')):  # pyright: ignore[reportUnusedClass]
            pass


def test_a_shape_defined_in_a_function_works() -> None:
    class LocalArgs(TypedDict):
        name: str

    class LocalCallPart(ToolCallPart[LocalArgs], kind=ToolKind('test', 'local-args', args=LocalArgs)):
        pass

    (call,) = ModelResponse(parts=[ToolCallPart('t', {'name': 'x'}, tool_kind='test.local-args')]).parts
    assert isinstance(call, LocalCallPart) and call.typed_args == {'name': 'x'}


def test_a_typed_part_with_lazy_annotations() -> None:
    """On Python 3.14+ a module without `from __future__ import annotations` keeps annotations lazy (PEP 649).

    The class is built from source, as this module's annotations are strings; before 3.14 they're eager.
    """
    namespace: dict[str, Any] = {
        'ToolCallPart': ToolCallPart,
        'ClassVar': ClassVar,
        'lazy_kind': ToolKind('test', 'lazy'),
    }
    source = textwrap.dedent(
        """
        class Label:
            pass

        class LazyCallPart(ToolCallPart, kind=lazy_kind):
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


def test_a_tool_definition_takes_the_kind_or_its_id() -> None:
    tool_def = ToolDefinition(name='lookup', tool_kind=lookup_kind)
    assert tool_def.tool_kind == 'test.lookup'
    assert replace(tool_def, tool_kind='test.audit').tool_kind == 'test.audit'
    assert replace(tool_def, tool_kind=None).tool_kind is None
    assert ToolDefinition(name='lookup').tool_kind is None


async def test_a_run_refuses_a_tool_with_an_unregistered_kind() -> None:
    async def mark_kind(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, tool_kind='test.unknown')

    def lookup(sku: str) -> str:
        return sku  # pragma: no cover

    agent = Agent(TestModel(), tools=[Tool(lookup, prepare=mark_kind)])
    with pytest.raises(UserError, match=r"declares `tool_kind='test\.unknown'`, which is not a defined `ToolKind`"):
        await agent.run('go')


class _OtherArgs(TypedDict):
    code: str


def _missing_sku(code: str) -> LookupResult:
    return _lookup(code)  # pragma: no cover


def _optional_sku(sku: str = '') -> LookupResult:
    return _lookup(sku)  # pragma: no cover


def _numeric_sku(sku: int) -> LookupResult:
    return _lookup(str(sku))  # pragma: no cover


def _string_quantity(sku: str | None, quantity: str) -> LookupResult:
    return _lookup(f'{sku} {quantity}')  # pragma: no cover


def _other_args(args: _OtherArgs) -> LookupResult:
    return _lookup(args['code'])  # pragma: no cover


@pytest.mark.parametrize(
    'function,problem',
    [
        (_missing_sku, "it has no 'sku' parameter"),
        (_optional_sku, "its 'sku' parameter is not required"),
        (_numeric_sku, "its 'sku' parameter is of type integer, not string"),
        (_string_quantity, "its 'quantity' parameter is of type string, not integer"),
        (_other_args, "it has no 'sku' parameter"),
    ],
)
async def test_a_run_refuses_a_tool_whose_parameters_do_not_fit_its_kind(function: Any, problem: str) -> None:
    agent = Agent(TestModel(), tools=[Tool(function, name='lookup', tool_kind=lookup_kind)])
    with pytest.raises(UserError) as exc_info:
        await agent.run('go')
    message = str(exc_info.value)
    assert message.startswith(
        "Tool 'lookup' declares `tool_kind='test.lookup'`, but its parameters don't fit the arguments "
        'the kind declares (LookupArgs): '
    )
    assert problem in message


async def test_a_numeric_parameter_fits_an_integer_field() -> None:
    def lookup(sku: str, quantity: float = 1) -> LookupResult:
        return _lookup(sku)

    agent = Agent(TestModel(), tools=[Tool(lookup, tool_kind=lookup_kind)])
    result = await agent.run('go')
    assert isinstance(result.all_messages()[1].parts[0], LookupCallPart)


async def test_a_kind_without_a_declared_shape_accepts_any_parameters() -> None:
    def audit(entry: int) -> str:
        return str(entry)

    agent = Agent(TestModel(), tools=[Tool(audit, tool_kind=audit_kind)])
    result = await agent.run('go')
    assert isinstance(result.all_messages()[1].parts[0], AuditCallPart)


@dataclass
class _RecordExecutedCalls(AbstractCapability[None]):
    calls: list[ToolCallPart]

    async def wrap_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        self.calls.append(call)
        return await handler(args)


async def test_a_call_built_without_its_kind_reaches_hooks_typed() -> None:
    """A dispatcher (like code mode's sandbox) that builds a bare call still hands hooks the typed part."""
    executed: list[ToolCallPart] = []
    toolset = FunctionToolset[None]([Tool(_lookup, name='lookup', tool_kind=lookup_kind)])
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    manager = await ToolManager[None](toolset=toolset, root_capability=_RecordExecutedCalls(executed)).for_run_step(ctx)

    result = await manager.handle_call(ToolCallPart('lookup', {'sku': 'A-1'}, tool_call_id='c1'))

    assert result == {'in_stock': True}
    (call,) = executed
    assert isinstance(call, LookupCallPart) and call.sku == 'A-1'


async def test_a_kind_without_an_arguments_shape_checks_only_the_return_type() -> None:
    report_kind = ToolKind('test', 'report', content=LookupResult)

    class ReportReturnPart(ToolReturnPart[LookupResult], kind=report_kind):
        pass

    def report(sku: str) -> LookupResult:
        return {'in_stock': sku == 'a'}

    agent = Agent(TestModel(), tools=[Tool(report, tool_kind=report_kind)])
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

    native_call = NativeToolCallPart('t', {}, tool_kind='test.unknown')
    assert NativeToolCallPart.narrow_type(native_call) is native_call
    assert not isinstance(native_call, _NativeIntermediateCall)

    native_return = NativeToolReturnPart('t', {}, tool_kind='test.unknown')
    assert NativeToolReturnPart.narrow_type(native_return) is native_return
    assert not isinstance(native_return, _NativeIntermediateReturn)


def test_a_slotted_typed_part_registers_the_recreated_class() -> None:
    # `@dataclass(slots=True)` builds a new class, re-running registration without the class arguments.
    @dataclass(repr=False, slots=True)
    class SlottedCallPart(ToolCallPart, kind=ToolKind('test', 'slotted')):
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


async def _tool_kinds_sent(agent: Agent[Any, str]) -> dict[str, str | None]:
    """The `tool_kind` of each function tool the model is sent on a run of `agent`."""
    kinds: dict[str, str | None] = {}

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        kinds.update({tool.name: tool.tool_kind for tool in info.function_tools})
        return ModelResponse(parts=[TextPart('done')])

    await agent.run('go', model=FunctionModel(record))
    return kinds


async def test_a_tool_declares_its_kind_with_the_part_class() -> None:
    await _assert_typed_lookup_parts(
        Agent(FunctionModel(stream_function=_respond), tools=[Tool(_lookup, name='lookup', tool_kind=lookup_kind)])
    )


async def test_agent_decorators_declare_a_kind() -> None:
    agent = Agent(FunctionModel(stream_function=_respond))

    @agent.tool_plain(tool_kind=lookup_kind)
    def lookup(sku: str) -> LookupResult:
        return _lookup(sku)

    await _assert_typed_lookup_parts(agent)

    @agent.tool(tool_kind='test.lookup')
    def lookup_with_ctx(ctx: RunContext[object], sku: str) -> LookupResult:
        return _lookup(sku)  # pragma: no cover

    assert await _tool_kinds_sent(agent) == {'lookup': 'test.lookup', 'lookup_with_ctx': 'test.lookup'}


async def test_toolset_and_capability_decorators_declare_a_kind() -> None:
    toolset = FunctionToolset[None]()

    @toolset.tool_plain(tool_kind=lookup_kind)
    def lookup(sku: str) -> LookupResult:
        return _lookup(sku)

    @toolset.tool(tool_kind=lookup_kind)
    def lookup_with_ctx(ctx: RunContext[None], sku: str) -> LookupResult:
        return _lookup(sku)  # pragma: no cover

    assert toolset.tools['lookup_with_ctx'].tool_kind == 'test.lookup'
    await _assert_typed_lookup_parts(Agent(FunctionModel(stream_function=_respond), toolsets=[toolset]))

    capability: Capability[None] = Capability()

    @capability.tool_plain(tool_kind=lookup_kind)
    def lookup_from_capability(sku: str) -> LookupResult:
        return _lookup(sku)  # pragma: no cover

    @capability.tool(tool_kind=lookup_kind)
    def lookup_from_capability_with_ctx(ctx: RunContext[None], sku: str) -> LookupResult:
        return _lookup(sku)  # pragma: no cover

    assert await _tool_kinds_sent(Agent(capabilities=[capability])) == {
        'lookup_from_capability': 'test.lookup',
        'lookup_from_capability_with_ctx': 'test.lookup',
    }


async def test_an_agent_tool_with_a_kind_set_in_prepare_produces_typed_parts() -> None:
    async def mark_kind(ctx: RunContext[object], tool_def: ToolDefinition) -> ToolDefinition:
        return replace(tool_def, tool_kind=lookup_kind)

    await _assert_typed_lookup_parts(
        Agent(FunctionModel(stream_function=_respond), tools=[Tool(_lookup, name='lookup', prepare=mark_kind)])
    )


def _sanitized(part: ToolReturnPart | NativeToolReturnPart) -> ToolReturnPart | NativeToolReturnPart:
    """Sanitize `part` as client-submitted history, answering a call the history also contains."""
    if isinstance(part, NativeToolReturnPart):
        messages: list[ModelMessage] = [
            ModelResponse(parts=[NativeToolCallPart(part.tool_name, {}, part.tool_call_id), part])
        ]
    else:
        messages = [
            ModelResponse(parts=[ToolCallPart(part.tool_name, {}, part.tool_call_id)]),
            ModelRequest(parts=[part]),
        ]
    sanitized = sanitize_messages(narrow_message_parts(messages))[-1].parts[-1]
    assert isinstance(sanitized, ToolReturnPart | NativeToolReturnPart)
    return sanitized


def test_sanitizing_a_return_does_not_trust_its_claimed_kind() -> None:
    local = ImageUrl('http://169.254.169.254/latest', force_download='allow-local')
    reset = ImageUrl('http://169.254.169.254/latest')
    with pytest.warns(UserWarning, match='force_download'):
        failed = _sanitized(ToolReturnPart('search', [local], 'c1', outcome='failed', tool_kind='tool-search'))
    assert (type(failed), failed.tool_kind, failed.content) == (ToolReturnPart, 'tool-search', [reset])

    with pytest.warns(UserWarning, match='force_download'):
        native = _sanitized(NativeToolReturnPart('search', [local], 'c1', outcome='failed', tool_kind='tool-search'))
    assert (type(native), native.tool_kind, native.content) == (NativeToolReturnPart, 'tool-search', [reset])

    # A kind that declares no content shape is promoted with any content, files included.
    with pytest.warns(UserWarning, match='force_download'):
        audit = _sanitized(ToolReturnPart('audit', [local], 'c1', tool_kind='test.audit'))
    assert (type(audit), audit.content) == (AuditReturnPart, [reset])


def test_a_typed_return_whose_sanitized_content_no_longer_fits_falls_back_to_the_base_part() -> None:
    lookup = _sanitized(ToolReturnPart('lookup', {'in_stock': True}, 'c1', tool_kind='test.lookup'))
    assert (type(lookup), lookup.content) == (LookupReturnPart, {'in_stock': True})

    with pytest.warns(UserWarning, match='force_download'):
        reset = _sanitized(
            ToolReturnPart(
                'image',
                {'image': ImageUrl('https://example.com/a.png', force_download=True)},
                'c1',
                tool_kind='test.image',
            )
        )
    assert (type(reset), reset.content) == (ImageReturnPart, {'image': ImageUrl('https://example.com/a.png')})

    with pytest.warns(UserWarning, match='s3'):
        dropped = _sanitized(
            ToolReturnPart('image', {'image': ImageUrl('s3://bucket/a.png')}, 'c1', tool_kind='test.image')
        )
    assert (type(dropped), dropped.tool_kind, dropped.content) == (ToolReturnPart, 'test.image', {})


class _UnstockedResult(TypedDict):
    stock: int


class _MaybeStockedResult(TypedDict):
    in_stock: NotRequired[bool]


class _TextStockResult(TypedDict):
    in_stock: str


def _returns_text(sku: str) -> str:
    return sku  # pragma: no cover


def _returns_unstocked(sku: str) -> _UnstockedResult:
    return {'stock': 0}  # pragma: no cover


def _returns_maybe_stocked(sku: str) -> _MaybeStockedResult:
    return {}  # pragma: no cover


def _returns_text_stock(sku: str) -> _TextStockResult:
    return {'in_stock': 'yes'}  # pragma: no cover


@pytest.mark.parametrize(
    'function,problem',
    [
        (_returns_text, 'it returns string, not object'),
        (_returns_unstocked, "its return value has no 'in_stock' field"),
        (_returns_maybe_stocked, "its return value's 'in_stock' field is not required"),
        (_returns_text_stock, "its return value's 'in_stock' field is of type string, not boolean"),
    ],
)
async def test_a_run_refuses_a_tool_whose_return_type_does_not_fit_its_kind(function: Any, problem: str) -> None:
    agent = Agent(TestModel(), tools=[Tool(function, name='lookup', tool_kind=lookup_kind)])
    with pytest.raises(UserError) as exc_info:
        await agent.run('go')
    message = str(exc_info.value)
    assert message.startswith(
        "Tool 'lookup' declares `tool_kind='test.lookup'`, but its return type doesn't fit the content "
        'the kind declares (LookupResult): '
    )
    assert problem in message


class _StockTree(TypedDict):
    in_stock: bool
    parts: list[_StockTree]


def _returns_untyped(sku: str):
    return _lookup(sku)


def _returns_optional(sku: str) -> LookupResult | None:
    return _lookup(sku)


def _returns_tool_return(sku: str) -> ToolReturn[LookupResult]:
    return ToolReturn(_lookup(sku))


def _returns_tree(sku: str) -> _StockTree:
    return {'in_stock': sku == 'A-1', 'parts': []}


@pytest.mark.parametrize('function', [_returns_untyped, _returns_optional, _returns_tool_return, _returns_tree])
async def test_return_types_that_fit_or_cannot_be_checked_are_accepted(function: Any) -> None:
    agent = Agent(TestModel(), tools=[Tool(function, name='lookup', tool_kind=lookup_kind)])
    result = await agent.run('go')
    assert isinstance(result.all_messages()[2].parts[0], LookupReturnPart)


async def test_a_tool_can_take_the_kinds_arguments_as_one_parameter() -> None:
    received: list[LookupArgs] = []
    tool_defs: list[ToolDefinition] = []
    agent = Agent(FunctionModel(stream_function=_respond))

    @agent.tool_plain(tool_kind=lookup_kind)
    def lookup(args: LookupArgs) -> LookupResult:
        """Check whether an item is in stock."""
        received.append(args)
        return _lookup(args['sku'])

    await _assert_typed_lookup_parts(agent)
    assert received == [{'sku': 'A-1'}]

    def record(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool_defs.extend(info.function_tools)
        return ModelResponse(parts=[TextPart('done')])

    await agent.run('go', model=FunctionModel(record))
    (tool_def,) = tool_defs
    assert tool_def.description == 'Check whether an item is in stock.'
    assert tool_def.parameters_json_schema == {
        'properties': {'sku': {'type': 'string'}, 'quantity': {'type': 'integer'}},
        'required': ['sku'],
        'title': 'LookupArgs',
        'type': 'object',
    }


async def test_a_context_tool_can_take_the_kinds_arguments_as_one_parameter() -> None:
    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(parts=[ToolCallPart('lookup', {'sku': 'A-1'}, tool_call_id='c1')])
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent(FunctionModel(respond), deps_type=str)

    @agent.tool(tool_kind=lookup_kind)
    async def lookup(ctx: RunContext[str], args: LookupArgs) -> LookupResult:
        return _lookup(ctx.deps if args['sku'] == 'A-1' else args['sku'])

    result = await agent.run('Is A-1 in stock?', deps='A-1')
    call = result.all_messages()[1].parts[0]
    assert isinstance(call, LookupCallPart) and call.typed_args == {'sku': 'A-1'}
    tool_return = result.all_messages()[2].parts[0]
    assert isinstance(tool_return, LookupReturnPart) and tool_return.typed_content == {'in_stock': True}


def test_the_old_tool_definition_kind_alias_is_deprecated() -> None:
    with pytest.warns(PydanticAIDeprecationWarning, match='renamed to `ToolDefinitionKind`'):
        assert tools.ToolKind is tools.ToolDefinitionKind
    with pytest.raises(AttributeError, match="has no attribute 'Missing'"):
        tools.Missing


def test_an_explicit_kind_is_kept_when_the_data_does_not_fit() -> None:
    call = ToolCallPart.narrow_type(ToolCallPart('lookup', {'sku': 1.5}, tool_call_id='c1'), tool_kind='test.lookup')
    assert type(call) is ToolCallPart and call.tool_kind == 'test.lookup'
    tool_return = ToolReturnPart.narrow_type(
        ToolReturnPart('lookup', 'oops', tool_call_id='c1'), tool_kind='test.lookup'
    )
    assert type(tool_return) is ToolReturnPart and tool_return.tool_kind == 'test.lookup'
    unregistered = ToolCallPart.narrow_type(ToolCallPart('t', {}, tool_call_id='c2'), tool_kind='test.unregistered')
    assert unregistered.tool_kind == 'test.unregistered'
    failed = ToolReturnPart.narrow_type(
        ToolReturnPart('lookup', {'in_stock': True}, tool_call_id='c3', outcome='failed'), tool_kind='test.lookup'
    )
    assert type(failed) is ToolReturnPart and failed.tool_kind == 'test.lookup'
    # One of core's kinds is still stripped, as its typed class is in the deserialization union.
    search = ToolCallPart.narrow_type(
        ToolCallPart('search_tools', {'wrong': 1}, tool_call_id='c4'), tool_kind='tool-search'
    )
    assert type(search) is ToolCallPart and search.tool_kind is None


async def test_failed_denied_and_non_fitting_calls_keep_the_tool_kind() -> None:
    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart('failing', {'sku': 'A-1'}, tool_call_id='f1'),
                    ToolCallPart('gated', {'sku': 'A-1'}, tool_call_id='d1'),
                    ToolCallPart('failing', {'sku': 5.5}, tool_call_id='b1'),
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    async def deny_all(ctx: RunContext[None], requests: DeferredToolRequests) -> DeferredToolResults:
        return DeferredToolResults(approvals={call.tool_call_id: ToolDenied('no') for call in requests.approvals})

    agent = Agent(
        FunctionModel(model_fn), deps_type=type(None), capabilities=[HandleDeferredToolCalls(handler=deny_all)]
    )

    @agent.tool_plain(tool_kind=lookup_kind)
    def failing(args: LookupArgs) -> LookupResult:
        raise ToolFailed('Inventory offline')

    @agent.tool_plain(tool_kind=lookup_kind, requires_approval=True)
    def gated(args: LookupArgs) -> LookupResult:
        return _lookup(args['sku'])  # pragma: no cover

    result = await agent.run('go')
    calls = [part for part in result.all_messages()[1].parts if isinstance(part, ToolCallPart)]
    assert [(type(call), call.tool_kind) for call in calls] == [
        (LookupCallPart, 'test.lookup'),
        (LookupCallPart, 'test.lookup'),
        (ToolCallPart, 'test.lookup'),
    ]
    returns = {
        part.tool_call_id: (type(part), part.tool_kind, part.outcome)
        for part in result.all_messages()[2].parts
        if isinstance(part, ToolReturnPart)
    }
    assert returns == {
        'f1': (ToolReturnPart, 'test.lookup', 'failed'),
        'd1': (ToolReturnPart, 'test.lookup', 'denied'),
        'b1': (ToolReturnPart, 'test.lookup', 'retried'),
    }
