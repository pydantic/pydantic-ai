from __future__ import annotations

import copy
import uuid
from collections.abc import Awaitable, Callable, Generator, Mapping
from decimal import Decimal
from typing import Any, ClassVar, cast

import pytest
from dbos import DBOS, DBOSConfig, SetWorkflowID
from prefect import flow
from prefect.context import TaskRunContext
from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]

from pydantic_ai import Agent
from pydantic_ai.capabilities import AbstractCapability, WrapperCapability, durable_operation
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityOperationParams,
    ModelRequestContextProjection,
    _CapabilityOperationResult,  # pyright: ignore[reportPrivateUsage]
    _model_request_schema,  # pyright: ignore[reportPrivateUsage]
    call_declaration,
    collect_capability_operations,
    recover_capability,
    tier_one_durable_operation,
)
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._operation import ToolsetKind
from pydantic_ai.durable_exec._toolset import Lifecycle
from pydantic_ai.durable_exec.dbos import DBOSDurability
from pydantic_ai.durable_exec.prefect import PrefectDurability
from pydantic_ai.durable_exec.temporal import TemporalDurability
from pydantic_ai.durable_exec.temporal._transports import _CapabilityOperationParams, _CapabilityOperationTransport
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage, _delta  # pyright: ignore[reportPrivateUsage]

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def blockbuster_enabled() -> bool:
    return False


class RecordingDurability(BaseDurabilityCapability[Any]):
    engine_name = 'recording'
    _durable_unit_noun = 'unit'
    _durable_container_noun = 'journal'
    _codec: ClassVar = JSON_CODEC
    _unsupported_runtime_toolset_kinds: ClassVar = frozenset()
    _wrapped_toolset_kinds: ClassVar = frozenset({'function', 'mcp', 'dynamic'})
    _toolset_lifecycles: ClassVar[Mapping[ToolsetKind, Lifecycle]] = {
        'function': 'enter-always',
        'mcp': 'enter-always',
        'dynamic': 'enter-never',
    }

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    @property
    def in_durable_context(self) -> bool:
        return True

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        self.calls.append((name, inputs))
        return await fn()


class ReplayingDurability(RecordingDurability):
    def __init__(self) -> None:
        super().__init__()
        self.recorded_results: dict[str, Any] = {}

    async def run_durable_unit(
        self, name: str, fn: Callable[[], Awaitable[Any]], *, inputs: tuple[Any, ...], config: Any
    ) -> Any:
        self.calls.append((name, inputs))
        if '__capability__' not in name:
            return await fn()
        if name not in self.recorded_results:
            self.recorded_results[name] = await fn()
        return self.recorded_results[name]


class Operations(AbstractCapability[Any]):
    id = 'operations'

    def __init__(self) -> None:
        self.calls: list[tuple[RunContext[Any], object]] = []
        self.result: int | None = None
        self.arguments: tuple[tuple[Any, ...], dict[str, Any]] = ((), {})

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.result = await self._calculate(ctx, *self.arguments[0], **self.arguments[1])

    @durable_operation
    async def _calculate(
        self,
        ctx: RunContext[Any],
        value: int = 2,
        *extra: int,
        scale: int = 1,
        **offsets: int,
    ) -> int:
        marker = object()
        self.calls.append((ctx, marker))
        return (value + sum(extra) + sum(offsets.values())) * scale


class DurableBeforeModelRequest(AbstractCapability[Any]):
    id = 'before_model'

    @durable_operation
    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context.messages = [ModelRequest(parts=[UserPromptPart('replaced')])]
        return request_context


class RenamedContextBeforeModelRequest(AbstractCapability[Any]):
    id = 'renamed_before_model'

    @durable_operation
    async def before_model_request(  # pyright: ignore[reportIncompatibleMethodOverride]
        self, run_context: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return request_context


class ContextPositions(AbstractCapability[Any]):
    id = 'context_positions'

    def __init__(self) -> None:
        self.results: list[str] = []

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.results = [
            await self.ctx_first(ctx, 'first'),
            await self.ctx_last('last', ctx),
            await self.ctx_keyword_only('keyword', ctx=ctx),
            await self.no_ctx('none'),
            await self._summarize(['one', 'two'], ctx, previous_summary='previous'),
        ]

    @durable_operation
    async def ctx_first(self, ctx: RunContext[Any], value: str) -> str:
        return f'{value}:{ctx.model.model_name}'

    @durable_operation
    async def ctx_last(self, value: str, ctx: RunContext[Any]) -> str:
        return f'{value}:{ctx.model.model_name}'

    @durable_operation
    async def ctx_keyword_only(self, value: str, *, ctx: RunContext[Any]) -> str:
        return f'{value}:{ctx.model.model_name}'

    @durable_operation
    async def no_ctx(self, value: str) -> str:
        return value

    @durable_operation
    async def _summarize(
        self, messages: list[str], ctx: RunContext[Any], *, previous_summary: str | None = None
    ) -> str:
        return f'{previous_summary}:{len(messages)}:{ctx.model.model_name}'


class PerRunOperation(AbstractCapability[Any]):
    id = 'per_run_operation'

    def __init__(self, replacements: list[PerRunOperation]) -> None:
        self.replacements = replacements
        self.calls = 0

    async def for_run(self, ctx: RunContext[Any]) -> AbstractCapability[Any]:
        replacement = PerRunOperation(self.replacements)
        self.replacements.append(replacement)
        return replacement

    async def before_run(self, ctx: RunContext[Any]) -> None:
        await self.operation(ctx)

    @durable_operation
    async def operation(self, ctx: RunContext[Any]) -> None:
        self.calls += 1


class ModelReadingOperation(AbstractCapability[Any]):
    id = 'model_reader'

    def __init__(self, expected: TestModel) -> None:
        self.expected = expected
        self.result = False

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.result = await self.read_model(ctx)

    @durable_operation
    async def read_model(self, ctx: RunContext[Any]) -> bool:
        return ctx.model is self.expected


class UsageOperation(AbstractCapability[Any]):
    id = 'usage_operation'

    def __init__(self) -> None:
        self.calls = 0

    async def before_run(self, ctx: RunContext[Any]) -> None:
        await self.record_nested_usage(ctx)

    @durable_operation
    async def record_nested_usage(self, ctx: RunContext[Any]) -> None:
        self.calls += 1
        await Agent(TestModel(custom_output_text='summary')).run('summarize', usage=ctx.usage)
        ctx.usage.tool_calls += 2
        ctx.usage.details['summary_tokens'] = ctx.usage.details.get('summary_tokens', 0) + 3
        ctx.usage.cost = (ctx.usage.cost or 0) + Decimal('0.25')
        ctx.usage.__dict__['custom_units'] = ctx.usage.__dict__.get('custom_units', 0) + 7


async def test_non_durable_call_is_direct_and_preserves_identity() -> None:
    capability = Operations()
    agent = Agent(TestModel(), capabilities=[capability])

    await agent.run('test')

    assert capability.result == 2
    assert capability.calls[0][0].agent is agent


async def test_run_context_durable_operation_is_direct_without_durability() -> None:
    capability = Operations()
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())

    async def handler(value: int) -> int:
        return value * 2

    operation = ctx.durable_operation(capability, 'any-name', handler)

    assert await operation(3) == 6


async def test_run_context_durable_operation_rejects_unknown_bound_name() -> None:
    class Dynamic(AbstractCapability[Any]):
        id = 'dynamic'

        async def alpha(self) -> str:
            return 'alpha'

        async def zeta(self) -> str:
            return 'zeta'

        def get_durable_operations(self) -> dict[str, Callable[..., Awaitable[Any]]]:
            return {'zeta': self.zeta, 'alpha': self.alpha}

    capability = Dynamic()
    model = TestModel()
    agent = Agent(model, name='unknown_operation', capabilities=[capability, RecordingDurability()])
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), agent=agent)
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    durability._prepare_run_context(ctx)  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(UserError) as exc_info:
        ctx.durable_operation(capability, 'alpah', capability.alpha)

    assert str(exc_info.value) == (
        "Unknown durable operation 'alpah' for capability 'dynamic'. "
        "Known durable operations for this capability: 'alpha', 'zeta'. "
        'Check the operation name passed to `RunContext.durable_operation()`.'
    )


async def test_zero_declared_operations_rejects_unknown_bound_name() -> None:
    capability = AbstractCapability[Any]()
    capability.id = 'empty'
    model = TestModel()
    agent = Agent(model, name='empty_operations', capabilities=[capability, RecordingDurability()])
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), agent=agent)
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    durability._prepare_run_context(ctx)  # pyright: ignore[reportPrivateUsage]

    async def handler() -> None:
        pass

    with pytest.raises(UserError) as exc_info:
        ctx.durable_operation(capability, 'missing', handler)

    assert str(exc_info.value) == (
        "Unknown durable operation 'missing' for capability 'empty'. "
        'This capability declares no durable operations. Implement `get_durable_operations()` or mark a method with '
        '`@durable_operation`.'
    )


async def test_run_context_durable_operation_dispatches_bound_name() -> None:
    class Dynamic(AbstractCapability[Any]):
        id = 'dynamic'

        async def operation(self) -> str:
            return 'dispatched'

        def get_durable_operations(self) -> dict[str, Callable[..., Awaitable[Any]]]:
            return {'operation': self.operation}

    capability = Dynamic()
    model = TestModel()
    agent = Agent(model, name='known_operation', capabilities=[capability, RecordingDurability()])
    ctx = RunContext(deps=None, model=model, usage=RunUsage(), agent=agent)
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    durability._prepare_run_context(ctx)  # pyright: ignore[reportPrivateUsage]

    operation = ctx.durable_operation(capability, 'operation', capability.operation)

    assert await operation() == 'dispatched'
    assert any(name == 'known_operation__capability__dynamic.operation' for name, _ in durability.calls)


async def test_for_run_replacement_dispatches_on_run_instance() -> None:
    replacements: list[PerRunOperation] = []
    agent = Agent(
        TestModel(),
        name='for_run_operation',
        capabilities=[PerRunOperation(replacements), RecordingDurability()],
    )

    await agent.run('test')

    assert len(replacements) == 1
    assert replacements[0].calls == 1
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    assert any(name == 'for_run_operation__capability__per_run_operation.operation' for name, _ in durability.calls)


async def test_shared_capability_dispatch_is_scoped_to_each_agent() -> None:
    capability = Operations()
    first_agent = Agent(TestModel(), name='first_agent', capabilities=[capability, RecordingDurability()])
    second_agent = Agent(TestModel(), name='second_agent', capabilities=[capability, RecordingDurability()])

    await first_agent.run('test')
    await second_agent.run('test')

    first_durability = RecordingDurability.from_agent(first_agent)
    second_durability = RecordingDurability.from_agent(second_agent)
    assert first_durability is not None and second_durability is not None
    assert any(name == 'first_agent__capability__operations.calculate' for name, _ in first_durability.calls)
    assert any(name == 'second_agent__capability__operations.calculate' for name, _ in second_durability.calls)


async def test_wrapped_durability_dispatches_capability_operation() -> None:
    capability = Operations()
    agent = Agent(
        TestModel(),
        name='wrapped_durability',
        capabilities=[capability, WrapperCapability(wrapped=RecordingDurability())],
    )

    await agent.run('test')

    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    assert any(name == 'wrapped_durability__capability__operations.calculate' for name, _ in durability.calls)


def test_wrapped_temporal_durability_registers_capability_operation() -> None:
    agent = Agent(
        TestModel(),
        name='wrapped_temporal',
        capabilities=[Operations(), WrapperCapability(wrapped=TemporalDurability())],
    )

    durability = TemporalDurability.from_agent(agent)

    assert durability is not None
    activity_names = {
        ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        for activity in durability.temporal_activities
    }
    assert 'agent__wrapped_temporal__capability__operations__calculate' in activity_names


async def test_no_context_operation_is_direct_outside_a_run() -> None:
    assert await ContextPositions().no_ctx('outside') == 'outside'


async def test_capability_operation_cache_identity_includes_context_and_model() -> None:
    capability = Operations()
    capability.arguments = ((3, 4), {'scale': 2, 'bonus': 5})
    agent = Agent(TestModel(), name='binding', capabilities=[capability, RecordingDurability()])

    await agent.run('test')

    assert capability.result == 24
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    [(name, inputs)] = [call for call in durability.calls if '__capability__' in call[0]]
    assert name == 'binding__capability__operations.calculate'
    assert inputs[:2] == (None, {'value': 3, 'extra': [4], 'scale': 2, 'bonus': 5})
    assert isinstance(inputs[2], RunContext)


async def test_recorded_usage_delta_is_applied_once_per_replayed_run() -> None:
    capability = UsageOperation()
    agent = Agent(TestModel(), name='replayed_usage', capabilities=[capability, ReplayingDurability()])

    results = [await agent.run('test'), await agent.run('test')]

    for result in results:
        usage = result.usage
        assert (
            usage.requests,
            usage.tool_calls,
            usage.details,
            usage.cost,
            cast(int, usage.__dict__['custom_units']),
        ) == (2, 2, {'summary_tokens': 3}, Decimal('0.25'), 7)
    assert capability.calls == 1


def test_decorated_capability_requires_explicit_stable_id() -> None:
    class MissingId(AbstractCapability[Any]):
        @durable_operation
        async def operation(self, ctx: RunContext[Any]) -> None:
            pass

    with pytest.raises(UserError, match='needs an explicit `id` because persisted operation identity'):
        Agent(TestModel(), name='missing_id', capabilities=[MissingId(), RecordingDurability()])


def test_duplicate_operation_names_fail_during_agent_construction() -> None:
    class Duplicate(AbstractCapability[Any]):
        id = 'duplicate'

        @durable_operation(name='same')
        async def first(self, ctx: RunContext[Any]) -> None:
            pass

        @durable_operation(name='same')
        async def second(self, ctx: RunContext[Any]) -> None:
            pass

    with pytest.raises(UserError, match="Duplicate durable operation name 'same'"):
        Agent(TestModel(), name='duplicate', capabilities=[Duplicate(), RecordingDurability()])


@pytest.mark.parametrize(
    'hook',
    [
        'get_toolset',
        'get_wrapper_toolset',
        'wrap_run',
        'wrap_node_run',
        'wrap_model_request',
        'wrap_tool_validate',
        'wrap_tool_execute',
        'wrap_output_validate',
        'wrap_output_process',
        'wrap_run_event_stream',
    ],
)
def test_never_durable_hooks_fail_at_bind(hook: str) -> None:
    if hook in ('get_toolset', 'get_wrapper_toolset'):

        def sync_invalid(self: AbstractCapability[Any], *args: Any, **kwargs: Any) -> None:
            return None

        invalid: Any = sync_invalid
    else:

        async def async_invalid(self: AbstractCapability[Any], ctx: RunContext[Any], *args: Any, **kwargs: Any) -> None:
            pass

        invalid = async_invalid

    invalid.__name__ = hook

    decorated = durable_operation(invalid)
    capability_type = type('Invalid', (AbstractCapability,), {'id': 'invalid', hook: decorated})
    with pytest.raises(UserError, match=f'`{hook}`'):
        Agent(TestModel(), name='invalid', capabilities=[capability_type(), RecordingDurability()])


def test_tier_one_override_is_automatically_registered() -> None:
    class TierOneBase(AbstractCapability[Any]):
        @tier_one_durable_operation
        async def provision(self, ctx: RunContext[Any]) -> str:
            return 'base'

    class TierOne(TierOneBase):
        id = 'tier_one'

        async def provision(self, ctx: RunContext[Any]) -> str:
            return 'override'

    agent = Agent(TestModel(), name='tier_one', capabilities=[TierOne(), RecordingDurability()])
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    assert ('tier_one', 'provision') in durability._bound_capability_operations  # pyright: ignore[reportPrivateUsage]


async def test_inherited_tier_one_hook_is_not_registered_or_dispatched() -> None:
    class TierOneBase(AbstractCapability[Any]):
        def __init__(self) -> None:
            self.provisioned = False

        @tier_one_durable_operation
        async def provision(self, ctx: RunContext[Any]) -> None:
            self.provisioned = True

    class TierOne(TierOneBase):
        id = 'tier_one'

        async def before_run(self, ctx: RunContext[Any]) -> None:
            await self.provision(ctx)

    capability = TierOne()
    assert collect_capability_operations(capability) == {}

    agent = Agent(TestModel(), name='tier_one', capabilities=[capability, RecordingDurability()])
    await agent.run('test')

    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    assert capability.provisioned
    assert not any('__capability__' in name for name, _ in durability.calls)


def test_temporal_registration_has_stable_name_and_types() -> None:
    agent = Agent(TestModel(), name='temporal_operations', capabilities=[Operations(), TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    registration = next(
        activity
        for activity in durability.temporal_activities
        if ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        == 'agent__temporal_operations__capability__operations__calculate'
    )
    definition = ActivityDefinition.must_from_callable(registration)  # pyright: ignore[reportUnknownMemberType]
    assert definition.arg_types is not None
    assert definition.arg_types[0] is _CapabilityOperationParams
    assert definition.ret_type == _CapabilityOperationResult[int]


def test_unannotated_parameter_is_rejected_at_bind() -> None:
    class Unannotated(AbstractCapability[Any]):
        id = 'unannotated'

        @durable_operation  # pyright: ignore[reportUnknownArgumentType]
        async def operation(
            self,
            ctx: RunContext[Any],
            value,  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        ) -> str:
            return str(value)  # pyright: ignore[reportUnknownArgumentType]

    with pytest.raises(UserError, match="Parameter 'value' must have a type annotation"):
        Agent(TestModel(), name='unannotated', capabilities=[Unannotated(), RecordingDurability()])


async def test_decorated_model_request_hook_round_trips_mutation() -> None:
    agent = Agent(
        TestModel(call_tools=[]),
        name='before_model',
        capabilities=[DurableBeforeModelRequest(), RecordingDurability()],
    )

    result = await agent.run('original')

    requests = [message for message in result.all_messages() if isinstance(message, ModelRequest)]
    assert isinstance(requests[-1].parts[0], UserPromptPart)
    assert requests[-1].parts[0].content == 'replaced'
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    assert any(name == 'before_model__capability__before_model.before_model_request' for name, _ in durability.calls)


async def test_dynamic_hook_dispatches_through_public_run_context_lookup() -> None:
    class Dynamic(AbstractCapability[Any]):
        id = 'dynamic'

        def __init__(self) -> None:
            self.in_task = False

        async def before_run(self, ctx: RunContext[Any]) -> None:
            operation = ctx.durable_operation(self, 'operation', self.operation)
            self.in_task = await operation(1)

        async def operation(self, value: int) -> bool:
            return value == 1 and TaskRunContext.get() is not None

        def get_durable_operations(self) -> dict[str, Callable[..., Awaitable[Any]]]:
            return {'operation': self.operation}

    capability = Dynamic()
    agent = Agent(TestModel(), name='dynamic', capabilities=[capability, PrefectDurability()])

    @flow
    async def run() -> None:
        await agent.run('test')

    await run()
    assert capability.in_task


def test_sync_non_hook_operation_is_rejected_by_decorator() -> None:
    def operation() -> None:
        pass

    with pytest.raises(TypeError, match='can only decorate async methods'):
        durable_operation(operation)  # pyright: ignore[reportArgumentType]


def test_tier_one_base_and_duplicate_override_paths() -> None:
    class Base(AbstractCapability[Any]):
        @tier_one_durable_operation
        async def operation(self, ctx: RunContext[Any]) -> str:
            return 'base'

        sentinel = True

    assert collect_capability_operations(Base()) == {}

    class Override(Base):
        async def operation(self, ctx: RunContext[Any]) -> str:
            return 'override'

    assert set(collect_capability_operations(Override())) == {'operation'}

    class Duplicate(Base):
        id = 'duplicate'

        async def operation(self, ctx: RunContext[Any]) -> str:
            return 'override'

        def get_durable_operations(self) -> dict[str, Callable[..., Awaitable[Any]]]:
            return {'operation': self.operation}

    with pytest.raises(UserError, match="Duplicate durable operation name 'operation'"):
        collect_capability_operations(Duplicate())


async def test_run_context_is_located_from_the_schema() -> None:
    capability = ContextPositions()
    agent = Agent(TestModel(), name='context_positions', capabilities=[capability, RecordingDurability()])

    await agent.run('test')

    assert capability.results == [
        'first:test',
        'last:test',
        'keyword:test',
        'none',
        'previous:2:test',
    ]
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    operation_names = [name for name, _ in durability.calls if '__capability__' in name]
    assert operation_names == [
        'context_positions__capability__context_positions.ctx_first',
        'context_positions__capability__context_positions.ctx_last',
        'context_positions__capability__context_positions.ctx_keyword_only',
        'context_positions__capability__context_positions.no_ctx',
        'context_positions__capability__context_positions.summarize',
    ]


def test_before_model_request_context_parameter_can_have_any_name() -> None:
    declaration = collect_capability_operations(RenamedContextBeforeModelRequest())['before_model_request']

    assert declaration.ctx_parameter == 'run_context'


def test_dynamic_operation_without_run_context_is_supported() -> None:
    class MissingContext(AbstractCapability[Any]):
        async def operation(self, value: int) -> int:
            return value

        def get_durable_operations(self) -> dict[str, Callable[..., Awaitable[Any]]]:
            return {'operation': self.operation}

    declaration = collect_capability_operations(MissingContext())['operation']
    assert declaration.ctx_parameter is None


def test_non_callable_dynamic_operation_is_rejected() -> None:
    class Invalid(AbstractCapability[Any]):
        def get_durable_operations(self) -> Mapping[str, object]:
            return {'invalid': object()}

    with pytest.raises(UserError, match="Durable operation 'invalid' must be an async callable"):
        collect_capability_operations(Invalid())


def test_sync_dynamic_operation_is_rejected_at_bind() -> None:
    class SyncOperation(AbstractCapability[Any]):
        id = 'sync_capability'

        def operation(self) -> None:
            pass

        def get_durable_operations(self) -> Mapping[str, object]:
            return {'sync_operation': self.operation}

    with pytest.raises(
        UserError,
        match="Durable operation 'sync_operation' on capability 'sync_capability' must be an async callable",
    ):
        Agent(TestModel(), name='sync_operation', capabilities=[SyncOperation(), RecordingDurability()])


def test_prepare_run_context_without_agent_marks_durability_bound() -> None:
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    RecordingDurability()._prepare_run_context(ctx)  # pyright: ignore[reportPrivateUsage]
    assert ctx.__dict__['_durability_bound'] is True


def test_two_run_context_parameters_are_rejected_at_bind() -> None:
    class DuplicateContext(AbstractCapability[Any]):
        id = 'duplicate_context'

        @durable_operation
        async def operation(self, first: RunContext[Any], second: RunContext[Any]) -> None:
            pass

    with pytest.raises(
        UserError,
        match=r"Durable operation '.*operation' cannot take more than one `RunContext` parameter\.",
    ):
        Agent(TestModel(), name='duplicate_context', capabilities=[DuplicateContext(), RecordingDurability()])


def test_variadic_run_context_is_rejected_for_durable_operation() -> None:
    class VariadicContext(AbstractCapability[Any]):
        id = 'variadic_context'

        @durable_operation
        async def operation(self, *ctx: RunContext[Any]) -> None:
            pass

    with pytest.raises(UserError, match=r'RunContext cannot be used as a variadic positional parameter'):
        Agent(TestModel(), name='variadic_context', capabilities=[VariadicContext(), RecordingDurability()])


async def test_defensive_capability_operation_paths() -> None:
    capability = Operations()
    declaration = collect_capability_operations(capability)['calculate']
    projection_declaration = collect_capability_operations(DurableBeforeModelRequest())['before_model_request']
    ctx = capability.calls[0][0] if capability.calls else RunContext(deps=None, model=TestModel(), usage=RunUsage())

    with pytest.raises(RuntimeError, match='require the durability model scope'):
        await call_declaration(projection_declaration, capability, CapabilityOperationParams(ctx, {}))
    with pytest.raises(RuntimeError, match='requires the worker agent'):
        recover_capability(ctx, 'missing')
    plain_agent = Agent(TestModel())
    ctx.agent = plain_agent
    with pytest.raises(RuntimeError, match='found 0'):
        recover_capability(ctx, 'missing')

    assert (
        await capability._calculate(  # pyright: ignore[reportPrivateUsage]
            RunContext(deps=None, model=TestModel(), usage=RunUsage())
        )
        == 2
    )

    assert isinstance(
        await _model_request_schema(
            ctx,
            ModelRequestContextProjection([], None, ModelRequestParameters(), None, False),
        ),
        ModelRequestContextProjection,
    )
    assert declaration.result_type is int


async def test_bound_dispatch_defensively_rejects_missing_capability_id() -> None:
    agent = Agent(TestModel(), name='defensive', capabilities=[RecordingDurability()])
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    ctx.agent = agent
    with pytest.raises(RuntimeError, match='must have an explicit `id`'):
        await durability._invoke_capability_operation(  # pyright: ignore[reportPrivateUsage]
            AbstractCapability(), 'missing', ctx, (), {}
        )


async def test_capability_operation_rejects_realtime_context_model() -> None:
    capability = Operations()
    agent = Agent(TestModel(), name='realtime_context_model', capabilities=[capability, RecordingDurability()])
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    ctx = RunContext(deps=None, agent=agent, model=cast(Any, object()), usage=RunUsage())

    with pytest.raises(UserError, match='require a non-realtime `Model` on `RunContext`'):
        await durability._invoke_capability_operation(  # pyright: ignore[reportPrivateUsage]
            capability, 'calculate', ctx, (ctx,), {}
        )


async def test_capability_operation_rejects_unregistered_context_model() -> None:
    capability = Operations()
    agent = Agent(TestModel(), name='unregistered_context_model', capabilities=[capability, RecordingDurability()])
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    ctx = RunContext(deps=None, agent=agent, model=TestModel(), usage=RunUsage())

    with pytest.raises(
        UserError,
        match=r'was not registered with `RecordingDurability`.*cannot be used inside a journal',
    ):
        await durability._invoke_capability_operation(  # pyright: ignore[reportPrivateUsage]
            capability, 'calculate', ctx, (ctx,), {}
        )


def test_usage_delta_ignores_non_numeric_extension_values() -> None:
    before = RunUsage()
    after = RunUsage()
    before.__dict__['opaque'] = 'before'
    after.__dict__['opaque'] = 'after'
    after.details['opaque'] = cast(Any, 'after')

    delta = _delta(before, after)

    assert 'opaque' not in delta.__dict__
    assert 'opaque' not in delta.details


def test_usage_delta_preserves_numeric_extension_fields() -> None:
    before = RunUsage()
    after = RunUsage()
    before.__dict__['custom_units'] = 2
    after.__dict__['custom_units'] = 9

    delta = _delta(before, after)

    assert delta.__dict__['custom_units'] == 7


async def test_usage_snapshot_copies_details_before_in_place_handler_mutation() -> None:
    """`RunUsage.__copy__` isolates its only mutable field before worker-side accounting."""
    usage = RunUsage(details={'existing': 2})
    before = copy.copy(usage)
    ctx = RunContext(deps=None, model=TestModel(), usage=usage)

    async def handler(ctx: RunContext[None]) -> None:
        ctx.usage.details['existing'] += 3

    await handler(ctx)

    assert before.details == {'existing': 2}
    assert before.details is not usage.details
    assert _delta(before, usage).details == {'existing': 3}


async def test_temporal_capability_transport_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    capability = Operations()
    agent = Agent(TestModel(), name='temporal_transport', capabilities=[capability, TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    declaration = durability._capability_declarations[('operations', 'calculate')]  # pyright: ignore[reportPrivateUsage]
    transport = _CapabilityOperationTransport(durability, declaration)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    ctx.agent = agent
    params = CapabilityOperationParams(ctx, {'value': 2, 'extra': [], 'scale': 1})
    wire, deps = transport.dump(params)
    assert isinstance(wire, _CapabilityOperationParams)
    loaded = transport.load((wire, deps), runtime=durability)
    assert loaded.arguments == params.arguments

    summaries: list[str] = []

    async def execute_activity(*, activity: Any, args: Any, **config: Any) -> int:
        summaries.append(config['summary'])
        return 2

    monkeypatch.setattr('pydantic_ai.durable_exec.temporal._operation_backend.execute_activity', execute_activity)
    bound = durability._bound_capability_operations[('operations', 'calculate')]  # pyright: ignore[reportPrivateUsage]
    assert await bound(params) == 2
    assert summaries == ['capability: operations.calculate']


async def test_temporal_capability_operation_resolves_ctx_model_worker_side() -> None:
    model = TestModel()
    capability = ModelReadingOperation(model)
    agent = Agent(model, name='temporal_model_reader', capabilities=[capability, TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    declaration = durability._capability_declarations[('model_reader', 'read_model')]  # pyright: ignore[reportPrivateUsage]
    transport = _CapabilityOperationTransport(durability, declaration)
    ctx = RunContext(deps=None, agent=agent, model=model, usage=RunUsage())
    wire, deps = transport.dump(CapabilityOperationParams(ctx, {}, None))
    registration = next(
        activity
        for activity in durability.temporal_activities
        if ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        == 'agent__temporal_model_reader__capability__model_reader__read_model'
    )

    assert await registration(wire, deps)


@pytest.fixture
def dbos(tmp_path: Any) -> Generator[DBOS, None, None]:
    config: DBOSConfig = {
        'name': 'capability_durable_operations',
        'system_database_url': f'sqlite:///{tmp_path / "dbos.sqlite"}',
        'run_admin_server': False,
    }
    instance = DBOS(config=config)
    DBOS.launch()
    try:
        yield instance
    finally:
        DBOS.destroy()


async def test_dbos_capability_operation_end_to_end(dbos: DBOS) -> None:
    model = TestModel()
    capability = Operations()
    model_reader = ModelReadingOperation(model)
    agent = Agent(model, name='dbos_operations', capabilities=[capability, model_reader, DBOSDurability()])
    workflow_id = str(uuid.uuid4())

    @DBOS.workflow(name=f'capability_operations_{workflow_id}')
    async def workflow() -> int:
        await agent.run('test')
        assert capability.result is not None
        return capability.result

    with SetWorkflowID(workflow_id):
        assert await workflow() == 2
    assert model_reader.result

    steps = await dbos.list_workflow_steps_async(workflow_id)
    assert 'dbos_operations__capability__operations.calculate' in [step['function_name'] for step in steps]


async def test_dbos_capability_usage_delta_is_stable_on_replay(dbos: DBOS) -> None:
    capability = UsageOperation()
    agent = Agent(TestModel(), name='dbos_usage', capabilities=[capability, DBOSDurability()])
    workflow_id = str(uuid.uuid4())

    @DBOS.workflow(name=f'capability_usage_{workflow_id}')
    async def workflow() -> tuple[int, int, dict[str, int], Decimal | None, int]:
        result = await agent.run('test')
        usage = result.usage
        return usage.requests, usage.tool_calls, usage.details, usage.cost, cast(int, usage.__dict__['custom_units'])

    with SetWorkflowID(workflow_id):
        first = await workflow()
    with SetWorkflowID(workflow_id):
        replayed = await workflow()

    assert first == replayed == (2, 2, {'summary_tokens': 3}, Decimal('0.25'), 7)
    assert capability.calls == 1


async def test_prefect_capability_operation_end_to_end() -> None:
    capability = Operations()
    agent = Agent(TestModel(), name='prefect_operations', capabilities=[capability, PrefectDurability()])

    @flow
    async def run() -> int:
        await agent.run('test')
        assert capability.result is not None
        return capability.result

    assert await run() == 2


async def test_prefect_capability_operation_cache_identity_includes_context_and_model() -> None:
    class CacheIdentityOperation(AbstractCapability[str]):
        id = 'cache_identity'

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def before_run(self, ctx: RunContext[str]) -> None:
            await self.read_context(ctx, 1)

        @durable_operation
        async def read_context(self, ctx: RunContext[str], value: int) -> None:
            self.calls.append((ctx.deps, ctx.model.model_name))

    capability = CacheIdentityOperation()
    alternative_model = TestModel(custom_output_text='alternative', model_name='alternative')
    agent = Agent[str, str](
        TestModel(),
        name='prefect_capability_cache_identity',
        deps_type=str,
        capabilities=[capability, PrefectDurability(models={'alternative': alternative_model})],
    )

    @flow
    async def run() -> None:
        await agent.run('same', deps='tenant-a')
        await agent.run('same', deps='tenant-a')
        await agent.run('same', deps='tenant-b')
        await agent.run('same', deps='tenant-b', model='alternative')

    await run()

    assert capability.calls == [
        ('tenant-a', 'test'),
        ('tenant-b', 'test'),
        ('tenant-b', 'alternative'),
    ]
