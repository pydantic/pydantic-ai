from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass
from datetime import date
from typing import Annotated

import pytest
from pydantic import BaseModel, ConfigDict, Field, Json, SecretStr, computed_field, model_validator, with_config
from pydantic.dataclasses import dataclass as pydantic_dataclass
from render.workflows import TaskContext, Workflows
from typing_extensions import TypedDict

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.capabilities import Hooks, RawToolArgs, ValidatedToolArgs
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolKind, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext, run_agent_in_task


@dataclass
class Deps:
    tenant: str


class TypedPayload(BaseModel):
    due: date
    labels: list[str]


class TypedDeps(TypedDict):
    tenant: str


class CountDeps(TypedDict):
    retries: int


def typed_argument_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('typed arguments arrived')])
    return ModelResponse(
        parts=[
            ToolCallPart(
                'typed_lookup',
                {
                    'payload': {'due': '2026-09-15', 'labels': ['render', 'pydantic']},
                    'dates': ['2026-09-16', '2026-09-17'],
                },
                tool_call_id='typed-lookup',
            )
        ]
    )


def invalid_then_valid_typed_argument_model(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> ModelResponse:
    del info
    parts = [part for message in messages for part in message.parts]
    returns = [part for part in parts if isinstance(part, ToolReturnPart)]
    if any(part.outcome != 'retried' for part in returns):
        return ModelResponse(parts=[TextPart('typed arguments recovered')])
    due = '2026-09-15' if returns else 'not-a-date'
    return ModelResponse(
        parts=[
            ToolCallPart(
                'typed_lookup',
                {'payload': {'due': due, 'labels': ['render']}, 'dates': ['2026-09-16']},
                tool_call_id='typed-lookup',
            )
        ]
    )


async def test_model_and_function_transports_round_trip_public_run_context() -> None:
    seen: list[tuple[Deps, str | None, RunUsage, UsageLimits | None, object]] = []
    model = TestModel(call_tools=['lookup'])
    runtime = RenderWorkflows[Deps](Workflows(), deps_type=Deps)
    agent = Agent[Deps, str](
        model,
        name='transport-agent',
        deps_type=Deps,
        capabilities=[runtime],
    )

    @agent.tool
    async def lookup(ctx: RunContext[Deps], value: str) -> str:
        seen.append((ctx.deps, ctx.run_id, ctx.usage, ctx.usage_limits, ctx.model))
        return f'{ctx.deps.tenant}:{value}'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        result = await agent.run(
            'look up status',
            deps=Deps('acme'),
            usage=RunUsage(requests=2, input_tokens=10),
            usage_limits=UsageLimits(request_limit=8),
        )
        return result.output

    run_agent = runtime.task(run_agent_impl)
    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)

    assert seen
    deps, run_id, usage, usage_limits, child_model = seen[0]
    assert deps == Deps('acme')
    assert run_id is not None
    assert usage.requests >= 3
    assert usage_limits == UsageLimits(request_limit=8)
    assert child_model is model
    assert 'transport-agent__model.request' in context.task_names
    assert 'transport-agent__function_toolset__<agent>.call_tool' in context.task_names


async def test_function_arguments_are_revalidated_after_the_json_boundary() -> None:
    seen: list[tuple[TypedPayload, list[date]]] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(typed_argument_model),
        name='typed-arguments',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool_plain
    async def typed_lookup(payload: TypedPayload, dates: list[date]) -> str:
        seen.append((payload, dates))
        return 'ok'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('use typed arguments')).output

    run_agent = runtime.task(run_agent_impl)
    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)

    assert await pending == 'typed arguments arrived'
    assert seen == [
        (
            TypedPayload(due=date(2026, 9, 15), labels=['render', 'pydantic']),
            [date(2026, 9, 16), date(2026, 9, 17)],
        )
    ]
    assert 'typed-arguments__function_toolset__<agent>.call_tool' in context.task_names


def aliased_argument_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('aliased arguments arrived')])
    args = {'guestName': 'Ada', 'nights': 2}
    return ModelResponse(parts=[ToolCallPart('book', args, tool_call_id='book')])


async def test_aliased_function_arguments_are_revalidated_after_the_json_boundary() -> None:
    """Render's JSON codec ships validated arguments by Python name, while the tool's schema reads `guestName`."""
    seen: list[tuple[str, int]] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(aliased_argument_model),
        name='aliased-arguments',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool_plain
    async def book(guest: Annotated[str, Field(alias='guestName')], nights: int) -> str:
        seen.append((guest, nights))
        return 'ok'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('book Paris for Ada')).output

    run_agent = runtime.task(run_agent_impl)
    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)

    assert await pending == 'aliased arguments arrived'
    assert seen == [('Ada', 2)]
    assert 'aliased-arguments__function_toolset__<agent>.call_tool' in context.task_names


class Venue(TypedDict):
    room: Annotated[str, Field(alias='roomName')]


@dataclass
class Place:
    city: Annotated[str, Field(alias='cityName')]


class SerializedAs(BaseModel):
    room: str = Field(serialization_alias='roomOut')


class Counted(BaseModel):
    model_config = ConfigDict(extra='forbid')
    qty: int

    @computed_field
    @property
    def doubled(self) -> int:
        return self.qty * 2


def model_sent_arguments_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    args = {
        'venue': {'roomName': 'A1'},
        'place': {'cityName': 'Paris'},
        'token': 'hunter2',
        'nums': '[1, 2]',
        'out': {'room': 'A1'},
        'counted': {'qty': 2},
    }
    return ModelResponse(parts=[ToolCallPart('book', args, tool_call_id='book')])


async def test_function_arguments_are_validated_from_the_model_arguments_in_the_task() -> None:
    """The task validates the arguments the model sent, as the run did, rather than a JSON dump of the validated ones.

    The dump writes a `TypedDict` and a stdlib dataclass by field name while their schema reads the alias, masks
    the `SecretStr`, writes the parsed `Json` value while the schema expects a string, and writes a computed field
    that a model forbidding extra fields rejects.
    """
    seen: list[tuple[Venue, Place, str, list[int], SerializedAs, Counted]] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(model_sent_arguments_model), name='model-arguments', deps_type=type(None), capabilities=[runtime]
    )

    @agent.tool_plain
    async def book(
        venue: Venue, place: Place, token: SecretStr, nums: Json[list[int]], out: SerializedAs, counted: Counted
    ) -> str:
        seen.append((venue, place, token.get_secret_value(), nums, out, counted))
        return 'ok'

    assert await run_agent_in_task(agent, runtime, RecordingTaskContext()) == 'done'
    assert seen == [({'room': 'A1'}, Place(city='Paris'), 'hunter2', [1, 2], SerializedAs(room='A1'), Counted(qty=2))]


class Folded(BaseModel):
    price: int
    qty: int = 1

    @model_validator(mode='before')
    @classmethod
    def fold_quantity(cls, data: dict[str, int]) -> dict[str, int]:
        # Mutates its input: run twice on one dict, the second run finds no `qty`.
        data['price'] = data.pop('price') * data.pop('qty')
        return data


class PriceOptions(TypedDict, total=False):
    max_price: Annotated[float, Field(alias='maxPrice')]


class Order(BaseModel):
    qty: int


def hook_arguments_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(
        parts=[
            ToolCallPart('order', {'order': {'qty': 1}, 'note': 'model'}, tool_call_id='order'),
            ToolCallPart('stay', '{"nights": "two"}', tool_call_id='stay'),
        ]
    )


async def test_arguments_changed_by_hooks_reach_the_task_as_changed() -> None:
    """Arguments a hook replaced, mutated or repaired don't validate back from the model's, so the task gets them dumped."""
    seen: list[object] = []
    hooks = Hooks[None]()

    @hooks.on.before_tool_validate(tools=['stay'])
    async def repair(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: RawToolArgs
    ) -> RawToolArgs:
        return {'nights': 2}

    @hooks.on.after_tool_validate(tools=['order'])
    async def replace_note(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        return {**args, 'note': 'hook'}

    @hooks.on.before_tool_execute(tools=['order'])
    async def mutate_order(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        args['order'].qty = 99
        return args

    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(hook_arguments_model), name='hook-arguments', deps_type=type(None), capabilities=[hooks, runtime]
    )

    @agent.tool_plain
    async def order(order: Order, note: str) -> str:
        seen.append((order, note))
        return 'ok'

    @agent.tool_plain
    async def stay(nights: int) -> str:
        seen.append(nights)
        return 'ok'

    assert await run_agent_in_task(agent, runtime, RecordingTaskContext()) == 'done'
    assert seen == [(Order(qty=99), 'hook'), 2]


@dataclass
class Account:
    name: str
    tenant_id: int = dataclasses.field(compare=False)


def pinned_tenant_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    args = {'account': {'name': 'acme', 'tenant_id': 2}, 'token': 'hunter2'}
    return ModelResponse(parts=[ToolCallPart('act', args, tool_call_id='act')])


async def test_field_a_hook_pinned_reaches_the_task_as_pinned_where_eq_ignores_it() -> None:
    """A field a hook pinned reaches the task as pinned, even where `==` ignores it.

    `token` makes the dump lossy, so the model's arguments would be sent if they validated to the hook's; they do
    under `==`, which skips `tenant_id`. Their JSON form doesn't, so the call keeps the dump, as on main, and the
    `SecretStr` arrives masked.
    """
    seen: list[tuple[int, str]] = []
    hooks = Hooks[None]()

    @hooks.on.after_tool_validate(tools=['act'])
    async def pin_tenant(
        ctx: RunContext[None], *, call: ToolCallPart, tool_def: ToolDefinition, args: ValidatedToolArgs
    ) -> ValidatedToolArgs:
        return {**args, 'account': dataclasses.replace(args['account'], tenant_id=1)}

    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(pinned_tenant_model), name='pinned-tenant', deps_type=type(None), capabilities=[hooks, runtime]
    )

    @agent.tool_plain
    async def act(account: Account, token: SecretStr) -> str:
        seen.append((account.tenant_id, token.get_secret_value()))
        return 'ok'

    assert await run_agent_in_task(agent, runtime, RecordingTaskContext()) == 'done'
    assert seen == [(1, '**********')]


async def test_invalid_typed_arguments_follow_normal_model_retry_control_flow() -> None:
    seen: list[TypedPayload] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(invalid_then_valid_typed_argument_model),
        name='invalid-typed-arguments',
        deps_type=type(None),
        retries=1,
        capabilities=[runtime],
    )

    @agent.tool_plain
    async def typed_lookup(payload: TypedPayload, dates: list[date]) -> str:
        assert dates == [date(2026, 9, 16)]
        seen.append(payload)
        return 'ok'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('retry invalid typed arguments')).output

    run_agent = runtime.task(run_agent_impl)
    context = RecordingTaskContext()
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)

    assert await pending == 'typed arguments recovered'
    assert seen == [TypedPayload(due=date(2026, 9, 15), labels=['render'])]
    call_task = 'invalid-typed-arguments__function_toolset__<agent>.call_tool'
    # Pydantic rejects the malformed date before dispatching the tool call. The
    # corrected retry is the only invocation that crosses the task boundary.
    assert context.task_names.count(call_task) == 1


async def test_parameterized_mapping_dependencies_round_trip() -> None:
    seen: list[dict[str, str]] = []
    deps_type = dict[str, str]
    runtime = RenderWorkflows[dict[str, str]](Workflows(), deps_type=deps_type)
    agent = Agent[dict[str, str], str](
        TestModel(call_tools=['lookup']),
        name='mapping-deps',
        deps_type=deps_type,
        capabilities=[runtime],
    )

    @agent.tool
    async def lookup(ctx: RunContext[dict[str, str]], value: str) -> str:
        seen.append(ctx.deps)
        return f'{ctx.deps["tenant"]}:{value}'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('lookup', deps={'tenant': 'acme'})).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)
    assert seen == [{'tenant': 'acme'}]


async def test_typed_dict_dependencies_round_trip() -> None:
    seen: list[TypedDeps] = []
    runtime = RenderWorkflows[TypedDeps](Workflows(), deps_type=TypedDeps)
    agent = Agent[TypedDeps, str](
        TestModel(call_tools=['lookup']),
        name='typed-dict-deps',
        deps_type=TypedDeps,
        capabilities=[runtime],
    )

    @agent.tool
    async def lookup(ctx: RunContext[TypedDeps], value: str) -> str:
        seen.append(ctx.deps)
        return f'{ctx.deps["tenant"]}:{value}'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('lookup', deps=TypedDeps(tenant='acme'))).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)
    assert isinstance(await pending, str)
    assert seen == [TypedDeps(tenant='acme')]


async def test_invalid_typed_dict_dependencies_fail_codec_validation() -> None:
    runtime = RenderWorkflows[CountDeps](Workflows(), deps_type=CountDeps)
    agent = Agent[CountDeps, str](
        TestModel(),
        name='invalid-typed-dict-deps',
        deps_type=CountDeps,
        capabilities=[runtime],
    )

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        invalid_deps = {'retries': 'not-an-integer'}
        return (await agent.run('inspect', deps=invalid_deps)).output  # type: ignore[arg-type]

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)
    with (
        pytest.warns(UserWarning, match='Expected `int`'),
        pytest.raises(UserError, match='Render operation invalid request'),
    ):
        await pending


async def test_child_task_model_is_guarded_when_the_worker_registers_no_instance() -> None:
    # A model-name string is deliberately not resolved when the agent is bound, so nothing is
    # registered under `default` and a child task has no instance to attach to its run context.
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str]('test', name='string-model-agent', deps_type=type(None), capabilities=[runtime])

    @agent.tool
    async def inspect_model(ctx: RunContext[None]) -> str:
        return str(ctx.model)

    with pytest.raises(UserError, match="'model' is not available on 'RenderRunContext'"):
        await run_agent_in_task(agent, runtime, RecordingTaskContext())


async def test_registered_model_id_resolves_to_its_instance_in_the_child_task() -> None:
    alternate = TestModel(custom_output_text='from the alternate model')
    runtime = RenderWorkflows[None](Workflows(), models={'alternate': alternate})
    agent = Agent[None, str](
        TestModel(call_tools=['inspect_model']),
        name='registry-agent',
        deps_type=type(None),
        capabilities=[runtime],
    )
    seen: list[object] = []

    @agent.tool
    async def inspect_model(ctx: RunContext[None]) -> str:
        seen.append(ctx.model)
        return 'ok'

    output = await run_agent_in_task(agent, runtime, RecordingTaskContext(), model='alternate')

    assert output == 'from the alternate model'
    assert seen == [alternate]


async def test_unregistered_model_instance_is_rejected_before_dispatch() -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](TestModel(), name='instance-agent', deps_type=type(None), capabilities=[runtime])
    context = RecordingTaskContext()

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('inspect', model=TestModel(custom_output_text='unregistered'))).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(context)
    assert inspect.isawaitable(pending)

    # An instance cannot cross the task boundary, so the run fails before any task is started
    # rather than rebuilding a different model from its name on the worker.
    with pytest.raises(UserError, match='was not registered with `RenderWorkflows`'):
        await pending
    assert context.task_names == []


async def test_worker_context_keeps_availability_validation_and_enqueue_guards() -> None:
    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](
        TestModel(call_tools=['inspect_context']),
        name='worker-context-contract',
        deps_type=type(None),
        validation_context=lambda ctx: {'worker_run_id': ctx.run_id},
        capabilities=[Hooks(id='visible'), runtime],
    )

    @agent.tool
    def inspect_context(ctx: RunContext[None]) -> str:
        assert ctx.available_tool_names == {'inspect_context'}
        assert 'visible' in ctx.active_capability_ids
        assert ctx.is_tool_available('inspect_context')
        assert ctx.validation_context == {'worker_run_id': ctx.run_id}
        with pytest.raises(UserError, match='not supported inside a durable task'):
            ctx.enqueue('cannot be transferred to the parent')
        with pytest.raises(UserError, match='not supported inside a durable task'):
            ctx.cancel()
        with pytest.raises(UserError, match="'messages' is not available"):
            _ = ctx.messages
        return 'context checked'

    context = RecordingTaskContext()
    assert 'context checked' in await run_agent_in_task(agent, runtime, context)
    assert 'worker-context-contract__function_toolset__<agent>.call_tool' in context.task_names


async def test_model_compaction_preserves_settings_and_rebuilds_the_worker_model() -> None:
    from pydantic_ai import ModelRequestContext
    from pydantic_ai.models import ModelRequestParameters

    class CompactModel(TestModel):
        async def compact_messages(
            self, request_context: ModelRequestContext, *, instructions: str | None = None
        ) -> ModelResponse:
            assert request_context.model is self
            assert request_context.model_settings == {'temperature': 0.2}
            assert instructions == 'Keep decisions'
            return ModelResponse(parts=[TextPart('compacted')])

    model = CompactModel()
    agent = Agent(model, name='compact', capabilities=[RenderWorkflows(Workflows())])
    runtime = RenderWorkflows.from_agent(agent)
    assert runtime is not None
    request = ModelRequestContext(
        model=model,
        messages=[],
        model_settings={'temperature': 0.2},
        model_request_parameters=ModelRequestParameters(),
    )

    async def handler(context: ModelRequestContext) -> ModelResponse:
        return await context.model.compact_messages(context, instructions='Keep decisions')

    tasks = RecordingTaskContext()
    with runtime.activate(tasks):
        result = await runtime.wrap_model_request(
            RunContext(deps=None, model=model, usage=RunUsage()), request_context=request, handler=handler
        )
    assert result.parts == [TextPart('compacted')]
    assert tasks.task_names == ['compact__model.compact_messages']


async def test_worker_context_names_tools_by_kind_on_copies() -> None:
    lookup_kind = ToolKind('test_render_transports', 'lookup')

    def lookup(query: str) -> str:
        return query  # pragma: no cover - only listed

    def inspect_kinds(ctx: RunContext[None]) -> str:
        assert ctx.tool_name_for_kind(lookup_kind) == 'lookup'
        assert dataclasses.replace(ctx).tool_name_for_kind(lookup_kind) == 'lookup'
        return 'kinds checked'

    runtime = RenderWorkflows[None](Workflows())
    agent = Agent[None, str](
        TestModel(call_tools=['inspect_kinds']),
        name='worker-tool-kinds',
        deps_type=type(None),
        toolsets=[FunctionToolset([Tool(lookup, tool_kind=lookup_kind), inspect_kinds], id='kinds')],
        capabilities=[runtime],
    )

    context = RecordingTaskContext()
    assert 'kinds checked' in await run_agent_in_task(agent, runtime, context)


def folding_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(
        parts=[
            ToolCallPart('fold', {'filters': {'price': 3, 'qty': 2}}, tool_call_id='fold'),
            ToolCallPart(
                'fold_with_options', {'filters': {'price': 3, 'qty': 2}, 'opts': {'maxPrice': 10}}, tool_call_id='both'
            ),
        ]
    )


async def test_validator_that_mutates_its_input_runs_once_in_the_task() -> None:
    """Checking which arguments to send runs the tool's validators on copies, so the task validates each input once.

    `fold` crosses as its dump, which the task validates once. The run has already folded `fold_with_options`'s own
    arguments in place, so validating them again raises in the run; that call keeps the dump too, whose
    `PriceOptions` field arrives without its aliased key.
    """
    seen: list[object] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(folding_model), name='folding-arguments', deps_type=type(None), capabilities=[runtime]
    )

    @agent.tool_plain
    async def fold(filters: Folded) -> str:
        seen.append(filters)
        return 'ok'

    @agent.tool_plain
    async def fold_with_options(filters: Folded, opts: PriceOptions) -> str:
        seen.append((filters, opts))
        return 'ok'

    assert await run_agent_in_task(agent, runtime, RecordingTaskContext()) == 'done'
    # The model's arguments were folded in place, so this call keeps the dump and loses `opts` as on main (#13440).
    assert seen == [Folded(price=6, qty=1), (Folded(price=6, qty=1), {})]


class Room(BaseModel):
    name: str = Field(alias='roomName')
    max_price: float | None = Field(None, alias='maxPrice')


@pydantic_dataclass
class Guest:
    name: Annotated[str, Field(alias='guestName')]


def nested_aliased_argument_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
        return ModelResponse(parts=[TextPart('nested aliased arguments arrived')])
    args = {'room': {'roomName': 'A1', 'maxPrice': 10}, 'guest': {'guestName': 'Ada'}}
    return ModelResponse(parts=[ToolCallPart('book', args, tool_call_id='book')])


async def test_nested_aliased_function_arguments_cross_the_json_boundary_by_alias() -> None:
    """By Python name, the required `roomName` would fail revalidation and `maxPrice` would revalidate to its default.

    Not a VCR test: the claim is about the arguments crossing into the chained task run, and no provider request
    is involved.
    """
    seen: list[tuple[str, float | None, str]] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(nested_aliased_argument_model),
        name='nested-aliased-arguments',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool_plain
    async def book(room: Room, guest: Guest) -> str:
        seen.append((room.name, room.max_price, guest.name))
        return 'ok'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('book A1 for Ada')).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)

    assert await pending == 'nested aliased arguments arrived'
    assert seen == [('A1', 10.0, 'Ada')]


@with_config(ConfigDict(validate_by_name=True))
class NamedVenue(TypedDict):
    room: Annotated[str, Field(alias='roomName')]


def typed_dict_aliased_argument_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    returns = [part for message in messages for part in message.parts if isinstance(part, ToolReturnPart)]
    if returns:
        return ModelResponse(parts=[TextPart(', '.join(f'{part.tool_name}: {part.outcome}' for part in returns))])
    args = {'venue': {'roomName': 'A1'}, 'guests': 2}
    return ModelResponse(
        parts=[
            ToolCallPart('book_venue', args, tool_call_id='venue'),
            ToolCallPart('book_named_venue', args, tool_call_id='named-venue'),
        ]
    )


async def test_typed_dict_aliased_fields_need_validate_by_name_to_cross_the_json_boundary() -> None:
    """A validated `TypedDict` parameter is a plain dict keyed by Python name, so it crosses the boundary that way.

    The serializer for the arguments sees no field types, so it can't write the `TypedDict`'s aliases. After the
    boundary, the tool's schema reads `roomName`, so the argument revalidates only when the `TypedDict` also
    validates by name.

    Not a VCR test: the claim is about the arguments crossing into the chained task run, and no provider request
    is involved.
    """
    seen: list[str] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(typed_dict_aliased_argument_model),
        name='typed-dict-aliased-arguments',
        deps_type=type(None),
        capabilities=[runtime],
    )

    # Revalidation rejects the by-name `venue`, so this tool never runs.
    @agent.tool_plain
    async def book_venue(venue: Venue, guests: int) -> str:  # pragma: no cover
        return venue['room']

    @agent.tool_plain
    async def book_named_venue(venue: NamedVenue, guests: int) -> str:
        seen.append(venue['room'])
        return 'ok'

    async def run_agent_impl(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('book A1')).output

    run_agent = runtime.task(run_agent_impl)
    pending = run_agent.func(RecordingTaskContext())
    assert inspect.isawaitable(pending)

    assert await pending == 'book_venue: retried, book_named_venue: success'
    assert seen == ['A1']
