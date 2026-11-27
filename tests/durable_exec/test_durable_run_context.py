"""`DurableRunContext` rebuilt from a JSON payload, as an engine with out-of-process units sees it."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal
from uuid import UUID, uuid4

import pytest
from pydantic import (
    AfterValidator,
    AliasChoices,
    AliasPath,
    AnyUrl,
    BaseModel,
    Discriminator,
    Field,
    Json,
    PrivateAttr,
    SecretStr,
    Tag,
    TypeAdapter,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic.plugin import (
    SchemaKind,
    SchemaTypePath,
    ValidateJsonHandlerProtocol,
    ValidatePythonHandlerProtocol,
    ValidateStringsHandlerProtocol,
)
from pydantic_core import CoreConfig, CoreSchema, to_jsonable_python
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import Hooks, RawToolArgs, ValidatedToolArgs
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    DurabilityEngineSpec,
    DurableRunContext,
    ToolsetCallToolParams,
    raw_tool_args_for_unit,
    revalidate_function_call,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CustomEvent, ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage


@dataclass
class Progress(CustomEvent, name='durable_run_context_progress'):
    done: int = 0


SPEC = DurabilityEngineSpec(engine_name='Example', durable_unit_noun='job', durable_container_noun='pipeline')


def _round_trip(ctx: RunContext[Any]) -> dict[str, Any]:
    """Encode the serialized context the way a JSON-journal engine would, then decode it."""
    serialized = DurableRunContext.serialize_run_context(ctx)
    return JSON_CODEC.load(dict[str, Any], JSON_CODEC.dump(dict[str, Any], serialized))


async def test_restore_rebuilds_carried_fields_and_guards_the_rest():
    payloads: list[dict[str, Any]] = []
    agent = Agent(TestModel(), name='example', deps_type=str)

    @agent.tool
    def capture(ctx: RunContext[str]) -> str:
        payloads.append(_round_trip(ctx))
        return 'ok'

    await agent.run('hello', deps='worker deps')
    [payload] = payloads

    ctx = DurableRunContext.restore(payload, deps='worker deps', agent=agent, engine_spec=SPEC)

    assert ctx.deps == 'worker deps'
    assert ctx.tool_name == 'capture'
    assert isinstance(ctx.usage, RunUsage)
    assert 'capture' in ctx.available_tool_names
    assert ctx.agent is agent
    assert ctx.root_capability is agent.root_capability
    assert ctx.tool_manager is None

    with pytest.raises(UserError, match="'messages' is not available on 'DurableRunContext' inside a durable unit"):
        ctx.messages
    with pytest.raises(UserError, match=r'`ctx.enqueue\(\)` is not supported inside a durable job'):
        ctx.enqueue('later')
    with pytest.raises(UserError, match='Emitting events from a tool or event stream handler is not supported'):
        await ctx.emit(Progress())


async def test_subclass_carries_extra_fields():
    class PromptRunContext(DurableRunContext[Any]):
        @classmethod
        def serialize_run_context(cls, ctx: RunContext[Any]) -> dict[str, Any]:
            return {**super().serialize_run_context(ctx), 'prompt': ctx.prompt}

    payloads: list[dict[str, Any]] = []
    agent = Agent(TestModel(), name='example', deps_type=type(None))

    @agent.tool
    def capture(ctx: RunContext[None]) -> str:
        payloads.append(PromptRunContext.serialize_run_context(ctx))
        return 'ok'

    await agent.run('hello')
    ctx = PromptRunContext.restore(payloads[0], deps=None, agent=None, engine_spec=SPEC)

    assert ctx.prompt == 'hello'
    assert ctx.agent is None


async def test_validation_context_is_rebuilt_from_the_agent_on_first_read():
    calls: list[str | None] = []

    def build_validation_context(ctx: RunContext[str]) -> dict[str, str]:
        calls.append(ctx.tool_name)
        return {'tenant': ctx.deps}

    payloads: list[dict[str, Any]] = []
    agent = Agent(TestModel(), name='example', deps_type=str, validation_context=build_validation_context)

    @agent.tool
    def capture(ctx: RunContext[str]) -> str:
        payloads.append(_round_trip(ctx))
        return 'ok'

    await agent.run('hello', deps='acme')
    calls.clear()

    ctx = DurableRunContext.restore(payloads[0], deps='acme', agent=agent, engine_spec=SPEC)
    assert calls == []
    assert ctx.validation_context == {'tenant': 'acme'}
    assert ctx.validation_context == {'tenant': 'acme'}
    assert calls == ['capture']

    without_agent = DurableRunContext.restore(payloads[0], deps='acme', agent=None, engine_spec=SPEC)
    with pytest.raises(UserError, match="'validation_context' is not available"):
        without_agent.validation_context


async def test_revalidate_function_call_rebuilds_the_tool_and_validated_arguments():
    payloads: list[dict[str, Any]] = []
    toolset = FunctionToolset[None]()

    @toolset.tool
    def add(ctx: RunContext[None], a: int, b: int) -> int:
        payloads.append(_round_trip(ctx))
        return a + b

    agent = Agent(TestModel(), name='example', deps_type=type(None), toolsets=[toolset])
    await agent.run('hello')
    ctx = DurableRunContext.restore(payloads[0], deps=None, agent=agent, engine_spec=SPEC)

    params = await revalidate_function_call(
        toolset, ToolsetCallToolParams('add', tool_args={'a': '1', 'b': 2}, ctx=ctx, tool=None), engine_spec=SPEC
    )
    assert params.tool is not None
    assert params.tool_args == {'a': 1, 'b': 2}

    with pytest.raises(UserError, match=r"Tool 'gone' not found in toolset .* not supported with Example"):
        await revalidate_function_call(
            toolset, ToolsetCallToolParams('gone', tool_args={}, ctx=ctx, tool=None), engine_spec=SPEC
        )


class Job(BaseModel):
    data: Any = None
    dry_run: bool = False


async def test_revalidate_function_call_reads_a_single_model_argument_as_it_was_validated():
    """Arguments for a model with a field named after the tool's parameter come back as they were validated.

    Validated, they're `{'data': Job(...)}`; transported, `{'data': {'data': ..., 'dry_run': True}}`, which also
    reads as a `Job` whose `data` field holds everything else. A tool whose validator isn't the
    toolset's own, which no engine produces, validates its arguments as given.
    """
    toolset = FunctionToolset[None]()

    @toolset.tool_plain
    def run_job(data: Job) -> str:  # pragma: no cover
        return 'done'

    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    tool = (await toolset.get_tools(ctx))['run_job']
    validated = tool.args_validator.validate_python({'data': {'path': '/reports'}, 'dry_run': True})
    transported = JSON_CODEC.load(dict[str, Any], JSON_CODEC.dump(dict[str, Any], validated))

    params = await revalidate_function_call(
        toolset, ToolsetCallToolParams('run_job', tool_args=transported, ctx=ctx, tool=tool), engine_spec=SPEC
    )
    assert params.tool_args == {'data': Job(data={'path': '/reports'}, dry_run=True)}

    foreign = replace(tool, args_validator=TypeAdapter(dict[str, Any]).validator)
    params = await revalidate_function_call(
        toolset, ToolsetCallToolParams('run_job', tool_args=transported, ctx=ctx, tool=foreign), engine_spec=SPEC
    )
    assert params.tool_args == transported


class Venue(TypedDict):
    room: Annotated[str, Field(alias='roomName')]


@dataclass
class Seat:
    row: Annotated[int, Field(alias='rowNumber')]


class Place(BaseModel):
    city: str = Field(alias='cityName')


class Revision(BaseModel):
    draft: int = Field(0, alias='draftNumber', exclude=True)
    number: int = Field(alias='draft')


class Card(BaseModel):
    id: str = Field(alias='cardId')


class Bank(BaseModel):
    id: str = Field(alias='bankId')


class Node(BaseModel):
    name: str
    children: list[Node] = []


class Lodging(BaseModel):
    city: str = Field(serialization_alias='cityOut')


class Room(BaseModel):
    room: str = Field(serialization_alias='roomOut', validation_alias=AliasChoices('room', 'roomOut'))
    floor: int = Field(validation_alias=AliasChoices('floorIn', 'floorOut'), serialization_alias='floorOut')


def alias(city: Annotated[str, Field(alias='cityName')]) -> None:
    pass


def alias_choices(city: Annotated[str, Field(validation_alias=AliasChoices('cityName', 'town'))]) -> None:
    pass


def alias_path(city: Annotated[str, Field(validation_alias=AliasPath('place', 'city'))]) -> None:
    pass


def alias_default(city: Annotated[str, Field(alias='cityName')] = 'Paris') -> None:
    pass


def alias_kwargs(city: Annotated[str, Field(alias='cityName')], **counts: int) -> None:
    pass


def typed_dict(venue: Venue, guests: int) -> None:
    pass


def single_typed_dict(venue: Venue) -> None:
    pass


def stdlib_dataclass(seat: Seat, guests: int) -> None:
    pass


def model(place: Place, guests: int) -> None:
    pass


def excluded_field(revision: Revision, guests: int) -> None:
    pass


def alias_excluded_field(revision: Revision, guests: Annotated[int, Field(alias='guestCount')]) -> None:
    pass


def alias_union(payment: Card | Bank, guests: Annotated[int, Field(alias='guestCount')]) -> None:
    pass


def recursive_model(tree: Node, depth: int) -> None:
    pass


def alias_recursive_model(tree: Node, depth: Annotated[int, Field(alias='maxDepth')]) -> None:
    pass


def serialization_alias(lodging: Lodging, guests: int) -> None:
    pass


def serialization_alias_choices(room: Room, guests: int) -> None:
    pass


@dataclass(frozen=True)
class RevalidationCase:
    id: str
    function: Callable[..., None]
    model_args: dict[str, Any]
    wire: Literal['by_name', 'by_alias']
    """How nested models are written: by Python name through `JSON_CODEC`, by alias in built-in engine tool calls."""
    shipped: dict[str, Any]
    rejected: str | None = None
    """Why revalidation rejects the shipped arguments: a nested value ships under a key its validation doesn't read."""


NESTED_BY_NAME = 'nested models shipped by Python name: https://github.com/pydantic/pydantic-ai/issues/13568'
NO_ALIASES = 'TypedDict and dataclass values ship by Python name: https://github.com/pydantic/pydantic-ai/issues/13440'
SERIALIZATION_ONLY = 'serialization alias not read by validation: https://github.com/pydantic/pydantic-ai/issues/13567'

REVALIDATION_CASES = [
    RevalidationCase('alias', alias, {'cityName': 'Paris'}, 'by_name', {'city': 'Paris'}),
    RevalidationCase('alias-choices', alias_choices, {'town': 'Paris'}, 'by_name', {'city': 'Paris'}),
    RevalidationCase('alias-path', alias_path, {'place': {'city': 'Paris'}}, 'by_name', {'city': 'Paris'}),
    RevalidationCase('alias-default', alias_default, {}, 'by_name', {'city': 'Paris'}),
    RevalidationCase(
        'alias-kwargs', alias_kwargs, {'cityName': 'Paris', 'nights': 2}, 'by_name', {'city': 'Paris', 'nights': 2}
    ),
    RevalidationCase(
        'typed-dict',
        typed_dict,
        {'venue': {'roomName': 'A1'}, 'guests': 2},
        'by_alias',
        {'venue': {'room': 'A1'}, 'guests': 2},
        rejected=NO_ALIASES,
    ),
    RevalidationCase(
        'single-typed-dict',
        single_typed_dict,
        {'roomName': 'A1'},
        'by_alias',
        {'venue': {'room': 'A1'}},
        rejected=NO_ALIASES,
    ),
    RevalidationCase(
        'stdlib-dataclass',
        stdlib_dataclass,
        {'seat': {'rowNumber': 3}, 'guests': 2},
        'by_alias',
        {'seat': {'row': 3}, 'guests': 2},
        rejected=NO_ALIASES,
    ),
    RevalidationCase(
        'model-by-name',
        model,
        {'place': {'cityName': 'Paris'}, 'guests': 2},
        'by_name',
        {'place': {'city': 'Paris'}, 'guests': 2},
        rejected=NESTED_BY_NAME,
    ),
    RevalidationCase(
        'model-by-alias',
        model,
        {'place': {'cityName': 'Paris'}, 'guests': 2},
        'by_alias',
        {'place': {'cityName': 'Paris'}, 'guests': 2},
    ),
    RevalidationCase(
        'excluded-field',
        excluded_field,
        {'revision': {'draft': 7}, 'guests': 2},
        'by_alias',
        {'revision': {'draft': 7}, 'guests': 2},
    ),
    RevalidationCase(
        'alias-excluded-field',
        alias_excluded_field,
        {'revision': {'draft': 7}, 'guestCount': 2},
        'by_alias',
        {'revision': {'draft': 7}, 'guests': 2},
    ),
    RevalidationCase(
        'alias-union-by-alias',
        alias_union,
        {'payment': {'bankId': 'B1'}, 'guestCount': 2},
        'by_alias',
        {'payment': {'bankId': 'B1'}, 'guests': 2},
    ),
    RevalidationCase(
        'alias-union-by-name',
        alias_union,
        {'payment': {'bankId': 'B1'}, 'guestCount': 2},
        'by_name',
        {'payment': {'id': 'B1'}, 'guests': 2},
        rejected=NESTED_BY_NAME,
    ),
    RevalidationCase(
        'recursive-model',
        recursive_model,
        {'tree': {'name': 'root', 'children': [{'name': 'leaf'}]}, 'depth': 2},
        'by_alias',
        {'tree': {'name': 'root', 'children': [{'name': 'leaf', 'children': []}]}, 'depth': 2},
    ),
    RevalidationCase(
        'alias-recursive-model',
        alias_recursive_model,
        {'tree': {'name': 'root', 'children': [{'name': 'leaf'}]}, 'maxDepth': 2},
        'by_alias',
        {'tree': {'name': 'root', 'children': [{'name': 'leaf', 'children': []}]}, 'depth': 2},
    ),
    RevalidationCase(
        'serialization-alias',
        serialization_alias,
        {'lodging': {'city': 'Paris'}, 'guests': 2},
        'by_alias',
        {'lodging': {'cityOut': 'Paris'}, 'guests': 2},
        rejected=SERIALIZATION_ONLY,
    ),
    RevalidationCase(
        'serialization-alias-choices',
        serialization_alias_choices,
        {'room': {'room': 'A1', 'floorIn': 3}, 'guests': 2},
        'by_alias',
        {'room': {'roomOut': 'A1', 'floorOut': 3}, 'guests': 2},
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in REVALIDATION_CASES])
async def test_revalidate_function_call_reads_aliased_arguments_as_they_were_validated(case: RevalidationCase):
    """Arguments keyed by Python name come back as they were validated, even where a parameter's schema reads an alias.

    Validated arguments are keyed by Python name. Each engine ships them as its serializer writes them, and
    the model-facing validator reads aliases only. When that reading fails, only the parameters are read by
    Python name: nested values keep reading aliases, so they come back as the tool's hooks saw them. Read by
    name, `Revision.draft` in `alias-excluded-field` would get the value of `number`, which ships under the key
    `draft`, and the payment in `alias-union-by-name` would become a `Card`. Not a VCR test: no model request
    is involved.
    """
    toolset = FunctionToolset[None]([case.function])
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    name = case.function.__name__
    tool = (await toolset.get_tools(ctx))[name]
    validated = tool.args_validator.validate_python(case.model_args)
    if case.wire == 'by_name':
        assert JSON_CODEC.load(dict[str, Any], JSON_CODEC.dump(dict[str, Any], validated)) == case.shipped
    else:
        assert to_jsonable_python(validated, by_alias=True) == case.shipped

    params = ToolsetCallToolParams(name, tool_args=case.shipped, ctx=ctx, tool=tool)
    if case.rejected is None:
        assert (await revalidate_function_call(toolset, params, engine_spec=SPEC)).tool_args == validated
    else:
        with pytest.raises(ValidationError, match='Field required'):
            await revalidate_function_call(toolset, params, engine_spec=SPEC)


@pytest.mark.parametrize(
    'raw_tool_args',
    [{'venue': {'roomName': 'A1'}, 'guests': 2}, '{"venue": {"roomName": "A1"}, "guests": 2}'],
    ids=['object', 'json-string'],
)
async def test_revalidate_function_call_validates_the_carried_model_arguments(raw_tool_args: str | dict[str, Any]):
    """With `raw_tool_args` carried, the unit validates the model's arguments instead of `tool_args`.

    `tool_args` ships the `TypedDict` by Python name, which its schema doesn't read; the model's arguments read
    as the run validated them. Not a VCR test: no model request is involved.
    """
    toolset = FunctionToolset[None]([typed_dict])
    ctx = RunContext[None](deps=None, model=TestModel(), usage=RunUsage())
    tool = (await toolset.get_tools(ctx))['typed_dict']
    params = ToolsetCallToolParams(
        'typed_dict', tool_args={'venue': {'room': 'A1'}, 'guests': 2}, ctx=ctx, tool=tool, raw_tool_args=raw_tool_args
    )
    assert (await revalidate_function_call(toolset, params, engine_spec=SPEC)).tool_args == {
        'venue': {'room': 'A1'},
        'guests': 2,
    }


@dataclass(frozen=True)
class RawArgsCase:
    id: str
    args: str | dict[str, Any] | None
    sent: bool
    """Whether the call's own arguments are returned for the unit to validate."""
    changed_by_hook: bool = False
    dump_reproduces: bool = False
    """Whether the unit's load of the dumped arguments validates back to them."""


RAW_ARGS_CASES = [
    RawArgsCase('json-string', '{"pair": [1.5, "x"], "labels": {"1": 2}}', sent=True),
    RawArgsCase('json-object', {'pair': [1.5, 'x'], 'labels': {'1': 2}}, sent=True),
    RawArgsCase('dump-reproduces', {'pair': [1.5, 'x']}, sent=False, dump_reproduces=True),
    RawArgsCase('no-arguments', None, sent=False),
    RawArgsCase('tuple', {'pair': (1.5, 'x')}, sent=False),
    RawArgsCase('infinite-float', {'pair': [math.inf, 'x']}, sent=False),
    RawArgsCase('integer-key', {'labels': {1: 2}}, sent=False),
    RawArgsCase('changed-by-hook', {'pair': [1.5, 'x']}, sent=False, changed_by_hook=True),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in RAW_ARGS_CASES])
async def test_raw_tool_args_for_unit_sends_the_model_arguments_only_when_needed(case: RawArgsCase):
    """Which function tool calls `raw_tool_args_for_unit` lets a durable unit validate from the model's arguments.

    These run in-process rather than through an engine: every engine's model response crosses JSON before its tools
    run, so a value JSON would change only reaches a unit in arguments built workflow-side, like an approval's
    `override_args`. The engines' own suites cover the arguments a model sends.
    """
    calls: list[tuple[RunContext[None], dict[str, Any]]] = []
    toolset = FunctionToolset[None]()

    @toolset.tool
    def record(ctx: RunContext[None], pair: tuple[float, str] = (0.0, ''), labels: dict[int, int] | None = None) -> str:
        calls.append((ctx, {'pair': pair, 'labels': labels}))
        return 'recorded'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('record', case.args)])

    await Agent(FunctionModel(respond), deps_type=type(None), toolsets=[toolset]).run('go')
    [(ctx, tool_args)] = calls
    if case.changed_by_hook:
        tool_args = {**tool_args, 'pair': (9.0, 'y')}
    tool = (await toolset.get_tools(ctx))['record']

    # A dump the unit reads back as the defaults stands in for one that loses the arguments.
    loaded_tool_args: dict[str, Any] = {}
    if case.dump_reproduces:
        loaded_tool_args = TypeAdapter(dict[str, Any]).dump_python(tool_args, mode='json')

    params = ToolsetCallToolParams('record', tool_args=tool_args, ctx=ctx, tool=tool)
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=loaded_tool_args) == (
        case.args if case.sent else None
    )


class Filters(BaseModel):
    max_price: float


async def test_raw_tool_args_for_unit_keeps_the_dump_when_a_check_raises():
    """Validating a dump without a single model argument's key, as a hook could leave it, raises `KeyError`.

    The check leaves the call on today's path, so the unit raises it there instead of the workflow. Not a VCR test:
    it pins the helper's own fallback in-process, with no model request.
    """
    calls: list[RunContext[None]] = []
    toolset = FunctionToolset[None]()

    @toolset.tool
    def lookup(ctx: RunContext[None], filters: Filters) -> str:
        calls.append(ctx)
        return 'found'

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart('lookup', {'max_price': 10})])

    await Agent(FunctionModel(respond), deps_type=type(None), toolsets=[toolset]).run('go')
    [ctx] = calls
    tool = (await toolset.get_tools(ctx))['lookup']

    params = ToolsetCallToolParams('lookup', tool_args={}, ctx=ctx, tool=tool)
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args={}) is None


async def _validated_call(
    tool: Tool[None],
    args: dict[str, Any],
    change: Callable[[ValidatedToolArgs], ValidatedToolArgs] | None = None,
    *,
    repair: Callable[[RawToolArgs], RawToolArgs] | None = None,
) -> tuple[FunctionToolset[None], ToolsetCallToolParams]:
    """Run `tool` once, and return the parameters a durable engine's transport gets for the call.

    `change` stands in for a hook that changes the validated arguments, `repair` for one that changes the model's
    arguments before validation.
    """
    calls: list[tuple[RunContext[None], ValidatedToolArgs]] = []
    hooks = Hooks[None]()

    if repair is not None:

        @hooks.on.before_tool_validate(tools=[tool.name])
        async def fix(
            ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs
        ) -> RawToolArgs:
            return repair(args)

    @hooks.on.after_tool_validate(tools=[tool.name])
    async def capture(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        changed = change(args) if change is not None else args
        calls.append((ctx, changed))
        return changed

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) > 1:
            return ModelResponse(parts=[TextPart('done')])
        return ModelResponse(parts=[ToolCallPart(tool.name, copy.deepcopy(args))])

    toolset = FunctionToolset[None]([tool])
    await Agent(FunctionModel(respond), deps_type=type(None), toolsets=[toolset], capabilities=[hooks]).run('go')
    [(ctx, tool_args)] = calls
    toolset_tool = (await toolset.get_tools(ctx))[tool.name]
    return toolset, ToolsetCallToolParams(tool.name, tool_args=tool_args, ctx=ctx, tool=toolset_tool)


def _dumped_by_name(tool_args: dict[str, Any]) -> dict[str, Any]:
    """A dump that writes the `Venue` by Python name, which its schema doesn't read, as engines did before #13929."""
    return TypeAdapter(dict[str, Any]).dump_python(tool_args, mode='json')


VENUE = {'roomName': 'A1'}


class Lowered(BaseModel):
    email: str

    @field_validator('email')
    @classmethod
    def lower(cls, email: str) -> str:
        return email.lower()


class Folded(BaseModel):
    price: int
    qty: int = 1

    @model_validator(mode='before')
    @classmethod
    def fold_quantity(cls, data: dict[str, int]) -> dict[str, int]:
        data['price'] = data.pop('price') * data.pop('qty')
        return data


def _shout(word: str) -> str:
    return word.upper()


class Keyed(BaseModel):
    key: UUID = Field(default_factory=uuid4)


class PostInit(BaseModel):
    n: int

    def model_post_init(self, context: Any, /) -> None:
        pass


@dataclass
class Ticket:
    n: int

    def __post_init__(self) -> None:
        pass


class CustomInit(BaseModel):
    n: int

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)


class Cat(BaseModel):
    kind: Literal['cat'] = 'cat'


class Dog(BaseModel):
    kind: Literal['dog'] = 'dog'


def _pet_kind(value: Cat | Dog | dict[str, str]) -> str:
    return value.get('kind', 'cat') if isinstance(value, dict) else value.kind


class Shade(Enum):
    RED = 'red'

    @classmethod
    def _missing_(cls, value: object) -> Shade:
        return cls.RED


class Hue(Enum):
    RED = 'red'


class Point(BaseModel):
    x: int


class Defaulted(BaseModel):
    # Pydantic deep-copies this unhashable default for each instance.
    points: list[Point] = [Point(x=1)]


class Tags(BaseModel):
    tags: list[str] = Field(default_factory=list)


class Weights(BaseModel):
    # Pydantic deep-copies this unhashable default, which holds only builtin data.
    weights: dict[str, list[int]] = {'a': [1]}


class Cached(BaseModel):
    n: int
    _cache: int = PrivateAttr(default=0)


def no_user_code(
    venue: Venue,
    token: SecretStr,
    nums: Json[list[int]],
    tags: Tags,
    weights: Weights,
    cached: Cached,
    hue: Hue,
    pet: Annotated[Cat | Dog, Field(discriminator='kind')],
) -> None:
    pass


def field_validator_runs(venue: Venue, user: Lowered) -> None:
    pass


def before_validator_runs(venue: Venue, filters: Folded) -> None:
    pass


def annotated_validator_runs(venue: Venue, word: Annotated[str, AfterValidator(_shout)]) -> None:
    pass


def custom_default_factory(venue: Venue, keyed: Keyed) -> None:
    pass


def model_post_init_runs(venue: Venue, p: PostInit) -> None:
    pass


def dataclass_post_init_runs(venue: Venue, t: Ticket) -> None:
    pass


def custom_init_runs(venue: Venue, c: CustomInit) -> None:
    pass


def callable_discriminator(
    venue: Venue, pet: Annotated[Annotated[Cat, Tag('cat')] | Annotated[Dog, Tag('dog')], Discriminator(_pet_kind)]
) -> None:
    pass


def enum_missing(venue: Venue, shade: Shade) -> None:
    pass


def copied_default(venue: Venue, d: Defaulted) -> None:
    pass


@dataclass
class Clamp:
    # `eq=True` without `frozen` makes instances unhashable.
    high: int

    def __call__(self, n: int) -> int:
        return min(n, self.high)


def unhashable_validator(venue: Venue, n: Annotated[int, AfterValidator(Clamp(10))]) -> None:
    pass


class ListCheck(list[int]):
    def __call__(self, n: int) -> int:
        return n


class DictCheck(dict[str, int]):
    def __call__(self, n: int) -> int:
        return n


def list_subclass_validator(venue: Venue, n: Annotated[int, AfterValidator(ListCheck())]) -> None:
    pass


def dict_subclass_validator(venue: Venue, n: Annotated[int, AfterValidator(DictCheck())]) -> None:
    pass


def schema_only(venue: object) -> None:
    pass


@dataclass(frozen=True)
class GateCase:
    tool: Tool[None]
    args: dict[str, Any]
    sent: bool


GATE_CASES = [
    GateCase(
        Tool(no_user_code),
        {
            'venue': VENUE,
            'token': 'hunter2',
            'nums': '[1, 2]',
            'tags': {},
            'weights': {},
            'cached': {'n': 1},
            'hue': 'red',
            'pet': {'kind': 'dog'},
        },
        sent=True,
    ),
    GateCase(Tool(field_validator_runs), {'venue': VENUE, 'user': {'email': 'A@B.C'}}, sent=False),
    GateCase(Tool(before_validator_runs), {'venue': VENUE, 'filters': {'price': 3, 'qty': 2}}, sent=False),
    GateCase(Tool(annotated_validator_runs), {'venue': VENUE, 'word': 'hi'}, sent=False),
    GateCase(Tool(custom_default_factory), {'venue': VENUE, 'keyed': {}}, sent=False),
    GateCase(Tool(model_post_init_runs), {'venue': VENUE, 'p': {'n': 1}}, sent=False),
    GateCase(Tool(dataclass_post_init_runs), {'venue': VENUE, 't': {'n': 1}}, sent=False),
    GateCase(Tool(custom_init_runs), {'venue': VENUE, 'c': {'n': 1}}, sent=False),
    GateCase(Tool(callable_discriminator), {'venue': VENUE, 'pet': {'kind': 'dog'}}, sent=False),
    GateCase(Tool(enum_missing), {'venue': VENUE, 'shade': 'crimson'}, sent=False),
    GateCase(Tool(copied_default), {'venue': VENUE, 'd': {}}, sent=False),
    GateCase(Tool(unhashable_validator), {'venue': VENUE, 'n': 3}, sent=False),
    GateCase(Tool(list_subclass_validator), {'venue': VENUE, 'n': 3}, sent=False),
    GateCase(Tool(dict_subclass_validator), {'venue': VENUE, 'n': 3}, sent=False),
    GateCase(
        Tool.from_schema(
            schema_only,
            name='from_schema',
            description=None,
            json_schema={'type': 'object', 'properties': {'venue': {'type': 'object'}}},
        ),
        {'venue': VENUE},
        sent=False,
    ),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.tool.name) for case in GATE_CASES])
async def test_raw_tool_args_for_unit_sends_the_model_arguments_only_for_tools_without_user_validation_code(
    case: GateCase,
):
    """A tool whose argument validation can run user code always crosses as its dump.

    Every case's dump loses the `Venue` field, so only the tool decides. The checks validate again in the workflow,
    and for these tools that would run their validator functions, default factories, post-init hooks, custom
    `__init__`, discriminator or enum `_missing_` an extra time, or copy a default with its `__deepcopy__`. Pydantic's
    own validators, a builtin `default_factory`, private attributes and a field discriminator run no user code.

    Not a VCR test: the helper's decision is internal to a durable engine's transport, and no model request is
    involved.
    """
    toolset, params = await _validated_call(case.tool, case.args)
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=_dumped_by_name(params.tool_args)) == (
        case.args if case.sent else None
    )


@dataclass
class Account:
    name: str
    tenant_id: int = field(compare=False)


class Holder(BaseModel):
    name: str
    tenant_id: int = Field(0, exclude=True)
    _note: str = PrivateAttr(default='')


def changeable(
    venue: Venue, account: Account, holder: Holder, x: float, d: Decimal, n: int | float, labels: dict[str, int]
) -> None:
    pass


def _set_tenant(args: ValidatedToolArgs) -> ValidatedToolArgs:
    args['account'].tenant_id = 1
    return args


def _set_note(args: ValidatedToolArgs) -> ValidatedToolArgs:
    args['holder']._note = 'hook'
    return args


@dataclass(frozen=True)
class ChangeCase:
    id: str
    change: Callable[[ValidatedToolArgs], ValidatedToolArgs]
    sent: bool


CHANGE_CASES = [
    ChangeCase('unchanged', lambda args: args, sent=True),
    ChangeCase('copied', lambda args: {**args, 'holder': args['holder'].model_copy()}, sent=True),
    ChangeCase('compare-false-field', _set_tenant, sent=False),
    ChangeCase(
        'excluded-field',
        lambda args: {**args, 'holder': args['holder'].model_copy(update={'tenant_id': 1})},
        sent=False,
    ),
    ChangeCase('private-attribute', _set_note, sent=False),
    ChangeCase('negative-zero', lambda args: {**args, 'x': -0.0}, sent=False),
    ChangeCase('decimal-exponent', lambda args: {**args, 'd': Decimal('1.00')}, sent=False),
    ChangeCase('int-to-float', lambda args: {**args, 'n': 1.0}, sent=False),
    ChangeCase('reordered', lambda args: {**args, 'labels': {'b': 2, 'a': 1}}, sent=False),
]


@pytest.mark.parametrize('case', [pytest.param(case, id=case.id) for case in CHANGE_CASES])
async def test_raw_tool_args_for_unit_keeps_arguments_a_hook_changed_on_the_dump(case: ChangeCase):
    """Arguments a hook changed cross as their dump, which carries the change; equal copies don't count as a change.

    Each change is one that `==` or the arguments' JSON form misses: a `compare=False` field, an excluded field, a
    private attribute, `-0.0`, a `Decimal` exponent, `1.0` for `1` and the order of a dict's keys.

    Not a VCR test: the helper's decision is internal to a durable engine's transport, and no model request is
    involved.
    """
    args = {
        'venue': VENUE,
        'account': {'name': 'a', 'tenant_id': 2},
        'holder': {'name': 'h', 'tenant_id': 2},
        'x': 0.0,
        'd': '1.0',
        'n': 1,
        'labels': {'a': 1, 'b': 2},
    }
    toolset, params = await _validated_call(Tool(changeable), args, case.change)
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=_dumped_by_name(params.tool_args)) == (
        args if case.sent else None
    )


def repaired(venue: Venue, n: int) -> None:
    pass


async def test_raw_tool_args_for_unit_keeps_the_dump_when_a_hook_repaired_the_model_arguments():
    """Arguments a `before_tool_validate` hook repaired cross as their dump: the model's own don't validate.

    Not a VCR test: the helper's decision is internal to a durable engine's transport, and no model request is
    involved.
    """
    toolset, params = await _validated_call(
        Tool(repaired),
        {'venue': VENUE, 'n': 'three'},
        repair=lambda args: {**args, 'n': 3} if isinstance(args, dict) else args,
    )
    assert params.tool_args['n'] == 3
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=_dumped_by_name(params.tool_args)) is None


def linked(venue: Venue, url: AnyUrl) -> None:
    pass


async def test_raw_tool_args_for_unit_keeps_the_dump_for_a_type_it_cannot_compare():
    """An argument of a type the comparison doesn't know, such as a URL, keeps the call on its dump.

    Not a VCR test: the helper's decision is internal to a durable engine's transport, and no model request is
    involved.
    """
    toolset, params = await _validated_call(Tool(linked), {'venue': VENUE, 'url': 'https://example.com/'})
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=_dumped_by_name(params.tool_args)) is None


class CountingHandler(ValidatePythonHandlerProtocol, ValidateJsonHandlerProtocol, ValidateStringsHandlerProtocol):
    # The protocols give `on_success`, `on_error` and `on_exception` bodies that do nothing.
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def on_enter(self, *args: object, **kwargs: object) -> None:
        self.events.append('validate')


@dataclass
class CountingPlugin:
    events: list[str] = field(default_factory=list[str])

    def new_schema_validator(
        self,
        schema: CoreSchema,
        schema_type: object,
        schema_type_path: SchemaTypePath,
        schema_kind: SchemaKind,
        config: CoreConfig | None,
        plugin_settings: dict[str, object],
    ) -> tuple[ValidatePythonHandlerProtocol, ValidateJsonHandlerProtocol, ValidateStringsHandlerProtocol]:
        handler = CountingHandler(self.events)
        return handler, handler, handler


def plugin_observed(venue: Venue, n: int) -> None:
    pass


async def test_raw_tool_args_for_unit_runs_no_validation_a_pydantic_plugin_observes(monkeypatch: pytest.MonkeyPatch):
    """Deciding what to send validates with pydantic-core alone, so an installed Pydantic plugin sees nothing.

    Not a VCR test: the claim is about work the helper doesn't show a plugin, and no model request is involved.
    """
    plugin = CountingPlugin()
    monkeypatch.setattr('pydantic.plugin._loader._plugins', {'counting': plugin})
    args = {'venue': VENUE, 'n': 1}
    toolset, params = await _validated_call(Tool(plugin_observed), args)
    loaded = _dumped_by_name(params.tool_args)
    assert plugin.events
    seen = len(plugin.events)
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=loaded) == args
    assert len(plugin.events) == seen


class Sparse(BaseModel):
    a: int
    b: int = 0


def sparse(m: Sparse) -> None:
    pass


async def test_raw_tool_args_for_unit_keeps_the_dump_when_only_the_fields_set_differ():
    """A model argument that leaves out a defaulted field crosses as its dump, as on `main`.

    The dump writes every field, so the unit's model has them all in `model_fields_set`, unlike the run's. Its
    values are the same, and comparing `model_fields_set` would send the model's arguments for all such calls.

    Not a VCR test: the helper's decision is internal to a durable engine's transport, and no model request is
    involved.
    """
    toolset, params = await _validated_call(Tool(sparse), {'m': {'a': 1}})
    loaded = TypeAdapter(dict[str, Any]).dump_python(params.tool_args, mode='json', by_alias=True)
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=loaded) is None


@dataclass
class Writes:
    values: list[int] = field(default_factory=list[int])


class Noted(BaseModel):
    n: int

    @field_validator('n')
    @classmethod
    def note(cls, n: int, info: ValidationInfo) -> int:
        if isinstance(info.context, Writes):
            info.context.values.append(n)
        return n


def noted(venue: Venue, q: Noted) -> None:
    pass


async def test_raw_tool_args_for_unit_runs_no_validator_again_with_the_run_validation_context():
    """Deciding what to send runs none of the tool's validators, so a validator that writes to the context writes nothing.

    Not a VCR test: the claim is about work the helper doesn't do in the workflow, and no model request is involved.
    """
    toolset, params = await _validated_call(Tool(noted), {'venue': VENUE, 'q': {'n': 1}})
    writes = Writes()
    Noted.model_validate({'n': 1}, context=writes)
    assert writes.values == [1]
    params = replace(params, ctx=replace(params.ctx, validation_context=writes))
    assert raw_tool_args_for_unit(toolset, params, loaded_tool_args=_dumped_by_name(params.tool_args)) is None
    assert writes.values == [1]


async def test_validation_context_rebuilt_on_a_shallow_copy_leaves_the_original_rebuildable():
    payloads: list[dict[str, Any]] = []
    agent = Agent(TestModel(), name='example', deps_type=str, validation_context=lambda ctx: {'tenant': ctx.deps})

    @agent.tool
    def capture(ctx: RunContext[str]) -> str:
        payloads.append(_round_trip(ctx))
        return 'ok'

    await agent.run('hello', deps='acme')
    original = DurableRunContext.restore(payloads[0], deps='acme', agent=agent, engine_spec=SPEC)
    shallow = copy.copy(original)

    assert shallow.validation_context == {'tenant': 'acme'}
    assert original.validation_context == {'tenant': 'acme'}
