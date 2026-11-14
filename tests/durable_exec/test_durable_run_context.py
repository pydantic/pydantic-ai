"""`DurableRunContext` rebuilt from a JSON payload, as an engine with out-of-process units sees it."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Annotated, Any, Literal

import pytest
from pydantic import AliasChoices, AliasPath, BaseModel, Field, TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec import (
    JSON_CODEC,
    DurabilityEngineSpec,
    DurableRunContext,
    ToolsetCallToolParams,
    revalidate_function_call,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import CustomEvent
from pydantic_ai.models.test import TestModel
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


@dataclass(frozen=True)
class RevalidationCase:
    id: str
    function: Callable[..., None]
    model_args: dict[str, Any]
    wire: Literal['by_name', 'by_alias']
    """How the engine writes nested models: Render and Azure Durable Task by Python name, Temporal by alias."""
    shipped: dict[str, Any]
    rejected: str | None = None
    """Why revalidation rejects the shipped arguments: a nested value arrives keyed by Python name, read only by alias."""


NESTED_BY_NAME = 'nested models shipped by Python name: https://github.com/pydantic/pydantic-ai/issues/13431'
NO_ALIASES = 'TypedDict and dataclass values ship by Python name: https://github.com/pydantic/pydantic-ai/issues/13440'

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
