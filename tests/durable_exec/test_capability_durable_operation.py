from __future__ import annotations

import copy
import gc
import re
import uuid
import weakref
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Mapping
from contextlib import asynccontextmanager
from decimal import Decimal
from typing import TYPE_CHECKING, Any, cast

import pytest

from pydantic_ai import Agent, ModelMessage, ModelSettings
from pydantic_ai.capabilities import AbstractCapability, ResolveModelId, WrapperCapability, durable_operation
from pydantic_ai.durable_exec import DurabilityEngineSpec
from pydantic_ai.durable_exec._base import BaseDurabilityCapability
from pydantic_ai.durable_exec._capability_operation import (
    CapabilityOperationParams,
    CapabilityOperationResult,
    ModelRequestContextProjection,
    base_hook_durable_operation,
    call_declaration,
    collect_capability_operations,
    recover_capability,
)
from pydantic_ai.durable_exec._codec import JSON_CODEC
from pydantic_ai.durable_exec._operation import CapabilityOperationId, DurableOperationId, OperationConfigRole
from pydantic_ai.durable_exec._operation_backend import CallableOperationBackend
from pydantic_ai.durable_exec._operation_names import JournalOperationNamer
from pydantic_ai.durable_exec._toolset import ToolConfig
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import (
    ModelRequestContext,
    ModelRequestParameters,
    ModelResolutionContext,
    StreamedResponse,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from ..model_lifecycle_utils import LifecycleTrackingModel

if TYPE_CHECKING:
    from dbos import DBOS, DBOSConfig, SetWorkflowID
    from prefect import flow
    from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]

    from pydantic_ai.durable_exec.dbos import DBOSDurability
    from pydantic_ai.durable_exec.prefect import PrefectDurability
    from pydantic_ai.durable_exec.temporal import TemporalDurability
    from pydantic_ai.durable_exec.temporal._transports import _CapabilityOperationParams, _CapabilityOperationTransport

    dbos_available = prefect_available = temporal_available = False
else:
    try:
        from dbos import DBOS, DBOSConfig, SetWorkflowID

        from pydantic_ai.durable_exec.dbos import DBOSDurability

        dbos_available = True
    except ImportError:  # pragma: lax no cover
        DBOS = DBOSConfig = SetWorkflowID = DBOSDurability = cast(Any, None)
        dbos_available = False

    try:
        from prefect import flow

        from pydantic_ai.durable_exec.prefect import PrefectDurability

        prefect_available = True
    except ImportError:  # pragma: lax no cover
        flow = PrefectDurability = cast(Any, None)
        prefect_available = False

    try:
        from temporalio.activity import _Definition as ActivityDefinition  # pyright: ignore[reportPrivateUsage]

        from pydantic_ai.durable_exec.temporal import TemporalDurability
        from pydantic_ai.durable_exec.temporal._transports import (  # pyright: ignore[reportPrivateUsage]
            _CapabilityOperationParams,
            _CapabilityOperationTransport,
        )

        temporal_available = True
    except ImportError:  # pragma: lax no cover
        ActivityDefinition = TemporalDurability = cast(Any, None)
        temporal_available = False

pytestmark = pytest.mark.anyio

requires_dbos = pytest.mark.skipif(not dbos_available, reason='DBOS is not installed')
requires_prefect = pytest.mark.skipif(not prefect_available, reason='Prefect is not installed')
requires_temporal = pytest.mark.skipif(not temporal_available, reason='Temporal is not installed')


@pytest.fixture(autouse=True)
def blockbuster_enabled() -> bool:
    return False


class _RecordingConfig:
    def base(self, role: OperationConfigRole, operation_id: DurableOperationId) -> ToolConfig:
        return {}

    def for_tool(
        self, role: OperationConfigRole, operation_id: DurableOperationId, tool: object | None, tool_name: str
    ) -> ToolConfig:
        return {}


class _RecordingBackend(CallableOperationBackend[ToolConfig]):
    def __init__(self, durability: RecordingDurability) -> None:
        super().__init__(namer=JournalOperationNamer(durability.name), config=_RecordingConfig())
        self._durability = durability

    async def execute(
        self,
        *,
        operation_id: DurableOperationId,
        name: str,
        body: Callable[[], Awaitable[object]],
        cache_key: tuple[object, ...],
        config: object,
    ) -> object:
        durability = self._durability
        durability.calls.append((name, cache_key))
        if not durability.replay_capability_operations or '__capability__' not in name:
            return await body()
        if name not in durability.recorded_results:
            durability.recorded_results[name] = await body()
        return durability.recorded_results[name]


class RecordingDurability(BaseDurabilityCapability[Any]):
    engine_spec = DurabilityEngineSpec(
        engine_name='recording',
        durable_unit_noun='unit',
        durable_container_noun='journal',
        codec=JSON_CODEC,
    )

    replay_capability_operations = False

    def __init__(self, *, models: Mapping[str, TestModel] | None = None) -> None:
        super().__init__(models=models)
        self.calls: list[tuple[str, tuple[object, ...]]] = []
        self.recorded_results: dict[str, Any] = {}

    @property
    def in_durable_context(self) -> bool:
        return True

    def get_durable_operation_backend(self) -> CallableOperationBackend[ToolConfig]:
        return _RecordingBackend(self)


class ReplayingDurability(RecordingDurability):
    replay_capability_operations = True


class TransparentDurability(RecordingDurability):
    @property
    def in_durable_context(self) -> bool:
        return False


class ModelIdReplacingBeforeModelRequest(AbstractCapability[Any]):
    id = 'model_id_replacing_before_model'

    @durable_operation('before_model_request')
    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context.model_id = 'tracked'
        return request_context


class LifecycleModel(LifecycleTrackingModel):
    def __init__(self, events: list[str], **kwargs: Any) -> None:
        super().__init__(events, event_prefix='model-', **kwargs)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        self.events.append('stream-enter')
        try:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as streamed:
                yield streamed
        finally:
            self.events.append('stream-exit')


class RepeatingLifecycleModel(LifecycleTrackingModel):
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        self.events.append('request')
        return ModelResponse(parts=[TextPart('from-tracked')])


def _tracked_model_resolver(
    built: list[LifecycleTrackingModel],
    *,
    events: list[str] | None = None,
    event_prefix: str = '',
    model_type: type[LifecycleTrackingModel] = LifecycleTrackingModel,
) -> Callable[[ModelResolutionContext[Any], str], LifecycleTrackingModel]:
    """Build a fresh model for the one id these tests swap to, recording every instance.

    Shared across the tests below rather than repeated inside each: the outside-a-container test
    asserts this is never consulted, so a resolver of its own would leave a body nothing executes.
    Each call gets its own event list unless the caller supplies one to share.
    """

    def resolve(_ctx: ModelResolutionContext[Any], model_id: str) -> LifecycleTrackingModel:
        del model_id
        model = model_type(
            events if events is not None else [],
            event_prefix=event_prefix,
            custom_output_text='from-tracked',
        )
        built.append(model)
        return model

    return resolve


async def test_capability_operation_is_direct_outside_durable_context() -> None:
    events: list[str] = []
    built_models: list[LifecycleTrackingModel] = []
    resolve_model = _tracked_model_resolver(built_models, events=events, event_prefix='tracked-')

    durability = TransparentDurability()
    agent = Agent(
        TestModel(custom_output_text='original'),
        name='direct_capability_operation',
        capabilities=[ModelIdReplacingBeforeModelRequest(), ResolveModelId(resolve_model), durability],
    )

    assert (await agent.run('test')).output == 'original'
    assert events == []
    assert built_models == []
    assert not any('__capability__' in name for name, _ in durability.calls)


async def test_capability_operation_model_id_swap_resolves_and_manages_model() -> None:
    events: list[str] = []
    built_models: list[LifecycleTrackingModel] = []
    resolve_model = _tracked_model_resolver(built_models, events=events, event_prefix='tracked-')

    durability = RecordingDurability()
    agent = Agent(
        TestModel(custom_output_text='original'),
        name='durable_capability_operation',
        capabilities=[ModelIdReplacingBeforeModelRequest(), ResolveModelId(resolve_model), durability],
    )

    assert (await agent.run('test')).output == 'from-tracked'

    assert events == [
        'tracked-enter',
        'tracked-enter',
        'request',
        'tracked-exit:none',
        'tracked-exit:none',
    ]
    assert len(built_models) == 2
    assert any('__capability__' in name for name, _ in durability.calls)


async def test_capability_operation_registered_model_id_swap_does_not_manage_model() -> None:
    events: list[str] = []
    registered = LifecycleTrackingModel(events, event_prefix='registered-', custom_output_text='from-registered')
    durability = RecordingDurability(models={'tracked': registered})
    agent = Agent(
        TestModel(custom_output_text='original'),
        name='registered_capability_operation_model',
        capabilities=[ModelIdReplacingBeforeModelRequest(), durability],
    )

    assert (await agent.run('test')).output == 'from-registered'
    assert events == ['request']


async def test_resolved_request_model_records_are_released_with_their_models() -> None:
    built_models: list[LifecycleTrackingModel] = []
    durability = RecordingDurability()
    agent = Agent(
        TestModel(custom_output_text='original'),
        name='released_capability_operation_model',
        capabilities=[
            ModelIdReplacingBeforeModelRequest(),
            ResolveModelId(_tracked_model_resolver(built_models)),
            durability,
        ],
    )

    await agent.run('test')

    bound = RecordingDurability.from_agent(agent)
    assert bound is not None
    records = bound._resolved_request_models  # pyright: ignore[reportPrivateUsage]
    assert records

    # The record outlives the request it was made for, so it has to be released with its model
    # rather than accumulating one entry per swapped request for the life of the agent.
    built_models.clear()
    gc.collect()

    assert records == {}


async def test_repeated_capability_operation_model_id_swaps_close_each_model() -> None:
    built_models: list[LifecycleTrackingModel] = []
    resolve_model = _tracked_model_resolver(built_models, model_type=RepeatingLifecycleModel)

    agent = Agent(
        TestModel(custom_output_text='original'),
        name='repeated_capability_operation_model',
        capabilities=[ModelIdReplacingBeforeModelRequest(), ResolveModelId(resolve_model), RecordingDurability()],
    )

    attempts = 0

    @agent.output_validator
    def retry_once(output: str) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ModelRetry('retry once')
        return output

    assert (await agent.run('test')).output == 'from-tracked'
    assert [model.events for model in built_models] == [
        ['enter', 'exit:none'],
        ['enter', 'request', 'exit:none'],
        ['enter', 'exit:none'],
        ['enter', 'request', 'exit:none'],
    ]


@pytest.mark.parametrize(
    ('stream', 'fail', 'expected_events'),
    [
        (False, False, ['model-enter', 'request', 'model-exit:none']),
        (True, False, ['model-enter', 'stream-enter', 'stream-exit', 'model-exit:none']),
        (False, True, ['model-enter', 'request', 'model-exit:RuntimeError']),
    ],
)
async def test_durable_model_scope_manages_rebuilt_model_lifecycle(
    stream: bool, fail: bool, expected_events: list[str]
) -> None:
    """A worker-built model stays entered for its whole operation and exits on failure."""
    events: list[str] = []

    def resolve_model(_ctx: ModelResolutionContext[Any], _model_id: str) -> LifecycleModel:
        return LifecycleModel(events, fail=fail)

    agent = Agent(
        'lifecycle',
        name=f'rebuilt_model_lifecycle_{stream}_{fail}',
        capabilities=[ResolveModelId(resolve_model), RecordingDurability()],
    )

    if fail:
        with pytest.raises(RuntimeError, match='request failed'):
            await agent.run('test')
    elif stream:
        async with agent.run_stream('test') as result:
            assert await result.get_output() == 'ok'
    else:
        assert (await agent.run('test')).output == 'ok'

    assert events == expected_events


async def test_durable_model_scope_does_not_suppress_body_error() -> None:
    events: list[str] = []
    model = LifecycleModel(events, fail=True, suppress_exit=True)
    agent = Agent(
        'lifecycle',
        name='rebuilt_model_suppressed_exit',
        capabilities=[ResolveModelId(lambda ctx, model_id: model), RecordingDurability()],
    )

    with pytest.raises(RuntimeError, match='request failed'):
        await agent.run('test')

    assert events == ['model-enter', 'request', 'model-exit:RuntimeError']


async def test_durable_model_scope_surfaces_teardown_error() -> None:
    events: list[str] = []
    model = LifecycleModel(events, fail=True, fail_exit=True)
    agent = Agent(
        'lifecycle',
        name='rebuilt_model_failed_exit',
        capabilities=[ResolveModelId(lambda ctx, model_id: model), RecordingDurability()],
    )

    with pytest.raises(ValueError, match='exit failed'):
        await agent.run('test')

    assert events == ['model-enter', 'request', 'model-exit:RuntimeError']


async def test_durable_model_scope_does_not_manage_registered_models() -> None:
    """The agent owner remains responsible for default and `models=` instances."""
    default_events: list[str] = []
    registered_events: list[str] = []
    default = LifecycleModel(default_events, model_name='default')
    registered = LifecycleModel(registered_events, model_name='registered')
    agent = Agent(
        default,
        name='registered_model_lifecycle',
        capabilities=[RecordingDurability(models={'registered': registered})],
    )

    assert (await agent.run('test')).output == 'ok'
    assert (await agent.run('test', model='registered')).output == 'ok'

    async with agent.run_stream('test', model='registered') as result:
        assert await result.get_output() == 'ok'

    assert default_events == ['request']
    assert registered_events == ['request', 'stream-enter', 'stream-exit']


async def test_durable_model_scope_manages_inferred_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model inferred inside the worker has the same owned lifecycle as a resolver-built model."""
    events: list[str] = []

    def infer_lifecycle_model(_model_id: str) -> LifecycleModel:
        return LifecycleModel(events)

    monkeypatch.setattr('pydantic_ai.durable_exec._base.infer_model', infer_lifecycle_model)
    agent = Agent('test', name='inferred_model_lifecycle', capabilities=[RecordingDurability()])

    assert (await agent.run('test')).output == 'ok'
    assert events == ['model-enter', 'request', 'model-exit:none']


class Operations(AbstractCapability[Any]):
    id = 'operations'

    def __init__(self) -> None:
        self.calls: list[tuple[RunContext[Any], object]] = []
        self.result: int | None = None
        self.arguments: tuple[tuple[Any, ...], dict[str, Any]] = ((), {})

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.result = await self._calculate(ctx, *self.arguments[0], **self.arguments[1])

    @durable_operation('calculate')
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

    @durable_operation('before_model_request')
    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context.messages = [ModelRequest(parts=[UserPromptPart('replaced')])]
        return request_context


class CustomModelRequestOperation(AbstractCapability[Any]):
    id = 'custom_model_request'

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        return await self.rewrite_request(ctx, request_context)

    @durable_operation('rewrite_request')
    async def rewrite_request(self, ctx: RunContext[Any], request: ModelRequestContext) -> ModelRequestContext:
        request.messages = [ModelRequest(parts=[UserPromptPart('custom replacement')])]
        return request


class ModelReplacingBeforeModelRequest(AbstractCapability[Any]):
    id = 'model_replacing_before_model'

    def __init__(self, model: TestModel, *, model_id: str | None = None) -> None:
        self.model = model
        self.model_id = model_id

    @durable_operation('before_model_request')
    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context.model = self.model
        request_context.model_id = self.model_id
        return request_context


class RenamedContextBeforeModelRequest(AbstractCapability[Any]):
    id = 'renamed_before_model'

    # Only the declared context parameter name is inspected; dispatch would duplicate other hook tests.
    @durable_operation('before_model_request')
    async def before_model_request(  # pyright: ignore[reportIncompatibleMethodOverride] # pragma: no cover
        self, run_context: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext: ...


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

    @durable_operation('ctx_first')
    async def ctx_first(self, ctx: RunContext[Any], value: str) -> str:
        return f'{value}:{ctx.model.model_name}'

    @durable_operation('ctx_last')
    async def ctx_last(self, value: str, ctx: RunContext[Any]) -> str:
        return f'{value}:{ctx.model.model_name}'

    @durable_operation('ctx_keyword_only')
    async def ctx_keyword_only(self, value: str, *, ctx: RunContext[Any]) -> str:
        return f'{value}:{ctx.model.model_name}'

    @durable_operation('no_ctx')
    async def no_ctx(self, value: str) -> str:
        return value

    @durable_operation('summarize')
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

    @durable_operation('operation')
    async def operation(self, ctx: RunContext[Any]) -> None:
        self.calls += 1


class TenantScopedOperation(AbstractCapability[str]):
    id = 'tenant_scoped_operation'

    def __init__(self, tenant: str = 'unrestricted', observations: list[str] | None = None) -> None:
        self.tenant = tenant
        self.observations = observations if observations is not None else []

    async def for_run(self, ctx: RunContext[str]) -> AbstractCapability[str]:
        return TenantScopedOperation(ctx.deps, self.observations)

    async def before_run(self, ctx: RunContext[str]) -> None:
        self.observations.append(await self.read_tenant(ctx))

    @durable_operation('read_tenant')
    async def read_tenant(self, ctx: RunContext[str]) -> str:
        return self.tenant


class ModelReadingOperation(AbstractCapability[Any]):
    id = 'model_reader'

    def __init__(self, expected: TestModel) -> None:
        self.expected = expected
        self.result = False

    async def before_run(self, ctx: RunContext[Any]) -> None:
        self.result = await self.read_model(ctx)

    @durable_operation('read_model')
    async def read_model(self, ctx: RunContext[Any]) -> bool:
        return ctx.model is self.expected


class UsageOperation(AbstractCapability[Any]):
    id = 'usage_operation'

    def __init__(self) -> None:
        self.calls = 0

    async def before_run(self, ctx: RunContext[Any]) -> None:
        await self.record_nested_usage(ctx)

    @durable_operation('record_nested_usage')
    async def record_nested_usage(self, ctx: RunContext[Any]) -> None:
        self.calls += 1
        await Agent(TestModel(custom_output_text='summary')).run('summarize', usage=ctx.usage)
        ctx.usage.tool_calls += 2
        ctx.usage.details['summary_tokens'] = ctx.usage.details.get('summary_tokens', 0) + 3
        ctx.usage.details['custom_units'] = ctx.usage.details.get('custom_units', 0) + 7
        ctx.usage.cost = (ctx.usage.cost or 0) + Decimal('0.25')


async def test_non_durable_call_is_direct_and_preserves_identity() -> None:
    capability = Operations()
    agent = Agent(TestModel(), capabilities=[capability])

    await agent.run('test')

    assert capability.result == 2
    assert capability.calls[0][0].agent is agent


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


@requires_temporal
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
            usage.details['custom_units'],
        ) == (2, 2, {'summary_tokens': 3, 'custom_units': 7}, Decimal('0.25'), 7)
    assert capability.calls == 1


def test_decorated_capability_requires_explicit_stable_id() -> None:
    class MissingId(AbstractCapability[Any]):
        @durable_operation('operation')
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

    decorated = durable_operation(hook)(invalid)
    capability_type = type('Invalid', (AbstractCapability,), {'id': 'invalid', hook: decorated})
    with pytest.raises(UserError, match=f'`{hook}`'):
        Agent(TestModel(), name='invalid', capabilities=[capability_type(), RecordingDurability()])


def test_base_hook_override_is_automatically_registered() -> None:
    class TierOneBase(AbstractCapability[Any]):
        # Registration, not execution, is the behavior under test.
        @base_hook_durable_operation('provision')
        async def provision(self, ctx: RunContext[Any]) -> str: ...  # pragma: no branch

    class TierOne(TierOneBase):
        id = 'base_hook'

        # Automatic registration inspects this override without dispatching it.
        async def provision(self, ctx: RunContext[Any]) -> str: ...  # pragma: no branch

    agent = Agent(TestModel(), name='base_hook', capabilities=[TierOne(), RecordingDurability()])
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    assert ('base_hook', 'provision') in durability._bound_capability_operations  # pyright: ignore[reportPrivateUsage]


async def test_inherited_base_hook_hook_is_not_registered_or_dispatched() -> None:
    class TierOneBase(AbstractCapability[Any]):
        def __init__(self) -> None:
            self.provisioned = False

        @base_hook_durable_operation('provision')
        async def provision(self, ctx: RunContext[Any]) -> None:
            self.provisioned = True

    class TierOne(TierOneBase):
        id = 'base_hook'

        async def before_run(self, ctx: RunContext[Any]) -> None:
            await self.provision(ctx)

    capability = TierOne()
    assert collect_capability_operations(capability) == {}

    agent = Agent(TestModel(), name='base_hook', capabilities=[capability, RecordingDurability()])
    await agent.run('test')

    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    assert capability.provisioned
    assert not any('__capability__' in name for name, _ in durability.calls)


@requires_temporal
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
    assert definition.ret_type == CapabilityOperationResult[int]


def test_unannotated_parameter_is_rejected_at_bind() -> None:
    class Unannotated(AbstractCapability[Any]):
        id = 'unannotated'

        # Binding rejects the missing annotation before this handler can be dispatched.
        @durable_operation('operation')
        async def operation(  # pragma: no cover
            self,
            ctx: RunContext[Any],
            value,  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
        ) -> str: ...

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


async def test_custom_model_request_operation_round_trips_projection() -> None:
    agent = Agent(
        TestModel(call_tools=[]),
        name='custom_model_request',
        capabilities=[CustomModelRequestOperation(), RecordingDurability()],
    )

    result = await agent.run('original')

    requests = [message for message in result.all_messages() if isinstance(message, ModelRequest)]
    assert isinstance(requests[-1].parts[0], UserPromptPart)
    assert requests[-1].parts[0].content == 'custom replacement'
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    assert any(
        name == 'custom_model_request__capability__custom_model_request.rewrite_request' for name, _ in durability.calls
    )


def test_durable_operation_bindings_do_not_retain_agents() -> None:
    capability = Operations()
    agents = [
        Agent(TestModel(), name=f'weak_binding_{index}', capabilities=[capability, RecordingDurability()])
        for index in range(3)
    ]
    bindings = capability._get_durable_operation_bindings()  # pyright: ignore[reportPrivateUsage]
    references = [weakref.ref(agent) for agent in agents]
    assert len(bindings) == 3

    agents.clear()
    gc.collect()

    assert not any(reference() is not None for reference in references)
    assert len(bindings) == 0


async def test_decorated_model_request_hook_round_trips_registered_model_replacement() -> None:
    original = TestModel(custom_output_text='original')
    restricted = TestModel(custom_output_text='restricted', model_name='restricted')
    agent = Agent(
        original,
        name='model_replacement',
        capabilities=[
            ModelReplacingBeforeModelRequest(restricted),
            RecordingDurability(models={'restricted': restricted}),
        ],
    )

    result = await agent.run('test')

    assert result.output == 'restricted'


async def test_decorated_model_request_hook_keeps_unchanged_model_resolution_cheap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TestModel(custom_output_text='unchanged')
    agent = Agent(model, name='unchanged_model', capabilities=[DurableBeforeModelRequest(), RecordingDurability()])
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    resolutions = 0
    resolve = durability._resolve_model_for_request  # pyright: ignore[reportPrivateUsage]

    async def count_resolutions(model_id: str | None, ctx: RunContext[Any]) -> Any:
        nonlocal resolutions
        resolutions += 1
        return await resolve(model_id, ctx)

    monkeypatch.setattr(durability, '_resolve_model_for_request', count_resolutions)

    result = await agent.run('test')

    assert result.output == 'unchanged'
    assert resolutions == 2


async def test_decorated_model_request_hook_rejects_unregistered_model_replacement() -> None:
    replacement = TestModel(model_name='unregistered')
    agent = Agent(
        TestModel(),
        name='unregistered_model_replacement',
        capabilities=[ModelReplacingBeforeModelRequest(replacement), RecordingDurability()],
    )

    with pytest.raises(
        UserError,
        match=(
            r'A durable `before_model_request` hook replaced `request_context.model` with the unregistered model '
            r"instance 'test:unregistered'\. A live `Model` instance cannot be transported across the unit boundary\. "
            r'Register it in `models=` on `RecordingDurability` and select that registered model by ID\.'
        ),
    ):
        await agent.run('test')


async def test_decorated_model_request_hook_rejects_unknown_replacement_model_id() -> None:
    model = TestModel()
    agent = Agent(
        model,
        name='unknown_model_replacement',
        capabilities=[ModelReplacingBeforeModelRequest(model, model_id='restricted'), RecordingDurability()],
    )

    with pytest.raises(
        UserError,
        match=(
            r"The model 'restricted' could not be rebuilt on the recording worker: it is not a model name "
            r'`infer_model` can build, and no `resolve_model_id` capability claimed it\.'
        ),
    ):
        await agent.run('test')


async def test_registered_model_replacement_is_stable_on_journal_replay() -> None:
    restricted = TestModel(custom_output_text='restricted', model_name='restricted')
    capability = ModelReplacingBeforeModelRequest(restricted)
    durability = ReplayingDurability(models={'restricted': restricted})
    agent = Agent(
        TestModel(custom_output_text='original'),
        name='replayed_model_replacement',
        capabilities=[capability, durability],
    )

    results = [await agent.run('test'), await agent.run('test')]

    assert [result.output for result in results] == ['restricted', 'restricted']


def test_durable_operation_requires_explicit_name() -> None:
    async def operation() -> None:
        pass

    message = (
        '`durable_operation` requires an explicit operation name because it becomes persisted compatibility data '
        "and must not change when the function is renamed. Use `@durable_operation(name='operation_name')`."
    )
    with pytest.raises(TypeError, match='^' + re.escape(message) + '$'):
        durable_operation(operation)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize('name', ['', None])
def test_durable_operation_rejects_invalid_name(name: Any) -> None:
    if name == '':
        with pytest.raises(ValueError, match=re.escape('`durable_operation` name must not be empty')):
            durable_operation(name)
    else:
        with pytest.raises(TypeError, match=re.escape('`durable_operation` name must be a string, got NoneType')):
            durable_operation(name)


def test_durable_operation_name_is_independent_of_function_name() -> None:
    class Renamed(AbstractCapability[Any]):
        id = 'renamed'

        @durable_operation(name='operation')
        async def renamed_function(self, ctx: RunContext[Any]) -> None:
            pass

    declarations = collect_capability_operations(Renamed())
    assert set(declarations) == {'operation'}
    assert (
        JournalOperationNamer('agent').operation_name(
            CapabilityOperationId('renamed', operation=next(iter(declarations)))
        )
        == 'agent__capability__renamed.operation'
    )


def test_sync_non_hook_operation_is_rejected_by_decorator() -> None:
    def operation() -> None:
        pass

    with pytest.raises(TypeError, match='can only decorate async methods'):
        durable_operation('operation')(operation)  # pyright: ignore[reportArgumentType]


def test_base_hook_base_and_duplicate_override_paths() -> None:
    class Base(AbstractCapability[Any]):
        # Collection behavior is tested without dispatching any of these declarations.
        @base_hook_durable_operation('operation')
        async def operation(self, ctx: RunContext[Any]) -> str: ...  # pragma: no branch

        sentinel = True

    assert collect_capability_operations(Base()) == {}

    class Override(Base):
        # Collection inspects this override without dispatching it.
        async def operation(self, ctx: RunContext[Any]) -> str: ...  # pragma: no branch

    assert set(collect_capability_operations(Override())) == {'operation'}


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
    assert declaration.model_request_hook
    assert not collect_capability_operations(Operations())['calculate'].model_request_hook


def test_recording_config_has_no_per_tool_override() -> None:
    assert (
        _RecordingConfig().for_tool(
            'tool', CapabilityOperationId('capability', operation='operation'), None, 'operation'
        )
        == {}
    )


def test_two_run_context_parameters_are_rejected_at_bind() -> None:
    class DuplicateContext(AbstractCapability[Any]):
        id = 'duplicate_context'

        @durable_operation('operation')
        async def operation(self, first: RunContext[Any], second: RunContext[Any]) -> None:
            pass

    with pytest.raises(
        UserError,
        match=r"Durable operation '.*operation' cannot take more than one `RunContext` parameter\.",
    ):
        Agent(TestModel(), name='duplicate_context', capabilities=[DuplicateContext(), RecordingDurability()])


async def test_two_model_request_context_parameters_are_rejected_at_bind() -> None:
    class DuplicateModelRequestContext(AbstractCapability[Any]):
        id = 'duplicate_model_request_context'

        @durable_operation('operation')
        async def operation(self, first: ModelRequestContext, second: ModelRequestContext) -> ModelRequestContext:
            return first

    capability = DuplicateModelRequestContext()
    request_context = ModelRequestContext(
        model=TestModel(), messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    assert await capability.operation(request_context, request_context) is request_context

    with pytest.raises(
        UserError,
        match=r"Durable operation '.*operation' cannot take more than one `ModelRequestContext` parameter\.",
    ):
        Agent(
            TestModel(),
            name='duplicate_model_request_context',
            capabilities=[capability, RecordingDurability()],
        )


def test_variadic_run_context_is_rejected_for_durable_operation() -> None:
    class VariadicContext(AbstractCapability[Any]):
        id = 'variadic_context'

        @durable_operation('operation')
        async def operation(self, *ctx: RunContext[Any]) -> None:
            pass

    with pytest.raises(UserError, match=r'RunContext cannot be used as a variadic positional parameter'):
        Agent(TestModel(), name='variadic_context', capabilities=[VariadicContext(), RecordingDurability()])


async def test_variadic_model_request_context_is_rejected_for_durable_operation() -> None:
    class VariadicModelRequestContext(AbstractCapability[Any]):
        id = 'variadic_model_request_context'

        @durable_operation('operation')
        async def operation(self, *request_context: ModelRequestContext) -> ModelRequestContext:
            return request_context[0]

    capability = VariadicModelRequestContext()
    request_context = ModelRequestContext(
        model=TestModel(), messages=[], model_settings=None, model_request_parameters=ModelRequestParameters()
    )
    assert await capability.operation(request_context) is request_context

    with pytest.raises(UserError, match=r'ModelRequestContext cannot be used as a variadic positional parameter'):
        Agent(
            TestModel(),
            name='variadic_model_request_context',
            capabilities=[capability, RecordingDurability()],
        )


async def test_defensive_capability_operation_paths() -> None:
    capability = Operations()
    declaration = collect_capability_operations(capability)['calculate']
    projection_declaration = collect_capability_operations(DurableBeforeModelRequest())['before_model_request']
    ctx = capability.calls[0][0] if capability.calls else RunContext(deps=None, model=TestModel(), usage=RunUsage())

    with pytest.raises(AssertionError, match='called without its model scope'):
        await call_declaration(
            projection_declaration,
            capability,
            params=CapabilityOperationParams(ctx, arguments={}),
        )
    with pytest.raises(RuntimeError, match='requires the worker agent'):
        await recover_capability(ctx, capability_id='missing')
    plain_agent = Agent(TestModel())
    ctx.agent = plain_agent
    with pytest.raises(RuntimeError, match='found 0'):
        await recover_capability(ctx, capability_id='missing')

    assert (
        await capability._calculate(  # pyright: ignore[reportPrivateUsage]
            RunContext(deps=None, model=TestModel(), usage=RunUsage())
        )
        == 2
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
            AbstractCapability(), 'missing', ctx=ctx, args=(), kwargs={}
        )


async def test_capability_operation_rejects_realtime_context_model() -> None:
    capability = Operations()
    agent = Agent(TestModel(), name='realtime_context_model', capabilities=[capability, RecordingDurability()])
    durability = RecordingDurability.from_agent(agent)
    assert durability is not None
    ctx = RunContext(deps=None, agent=agent, model=cast(Any, object()), usage=RunUsage())

    with pytest.raises(UserError, match='require a non-realtime `Model` on `RunContext`'):
        await durability._invoke_capability_operation(  # pyright: ignore[reportPrivateUsage]
            capability, 'calculate', ctx=ctx, args=(ctx,), kwargs={}
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
            capability, 'calculate', ctx=ctx, args=(ctx,), kwargs={}
        )


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
    assert (usage - before).details == {'existing': 3}


@requires_temporal
async def test_temporal_capability_transport_and_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    capability = Operations()
    agent = Agent(TestModel(), name='temporal_transport', capabilities=[capability, TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    declaration = durability._capability_declarations[('operations', 'calculate')]  # pyright: ignore[reportPrivateUsage]
    transport = _CapabilityOperationTransport(durability, declaration)
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    ctx.agent = agent
    params = CapabilityOperationParams(ctx, arguments={'value': 2, 'extra': [], 'scale': 1})
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


@requires_temporal
async def test_temporal_capability_operation_resolves_ctx_model_worker_side() -> None:
    model = TestModel()
    capability = ModelReadingOperation(model)
    agent = Agent(model, name='temporal_model_reader', capabilities=[capability, TemporalDurability()])
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    declaration = durability._capability_declarations[('model_reader', 'read_model')]  # pyright: ignore[reportPrivateUsage]
    transport = _CapabilityOperationTransport(durability, declaration)
    ctx = RunContext(deps=None, agent=agent, model=model, usage=RunUsage())
    wire, deps = transport.dump(CapabilityOperationParams(ctx, arguments={}, model_id=None))
    registration = next(
        activity
        for activity in durability.temporal_activities
        if ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        == 'agent__temporal_model_reader__capability__model_reader__read_model'
    )

    assert await registration(wire, deps)


@requires_temporal
async def test_temporal_capability_operation_rederives_for_run_instance_worker_side() -> None:
    capability = TenantScopedOperation()
    agent = Agent(
        TestModel(),
        name='temporal_tenant_scope',
        deps_type=str,
        capabilities=[capability, TemporalDurability()],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    declaration = durability._capability_declarations[  # pyright: ignore[reportPrivateUsage]
        ('tenant_scoped_operation', 'read_tenant')
    ]
    transport = _CapabilityOperationTransport(durability, declaration)
    ctx = RunContext(deps='tenant-a', agent=agent, model=TestModel(), usage=RunUsage())
    wire, deps = transport.dump(CapabilityOperationParams(ctx, arguments={}))
    registration = next(
        activity
        for activity in durability.temporal_activities
        if ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        == 'agent__temporal_tenant_scope__capability__tenant_scoped_operation__read_tenant'
    )

    result = await registration(wire, deps)

    assert result.value == 'tenant-a'
    assert capability.tenant == 'unrestricted'


@requires_temporal
async def test_temporal_capability_operation_projects_registered_model_replacement() -> None:
    original = TestModel(model_name='original')
    restricted = TestModel(model_name='restricted')
    capability = ModelReplacingBeforeModelRequest(restricted)
    agent = Agent(
        original,
        name='temporal_model_replacement',
        capabilities=[capability, TemporalDurability(models={'restricted': restricted})],
    )
    durability = TemporalDurability.from_agent(agent)
    assert durability is not None
    declaration = durability._capability_declarations[  # pyright: ignore[reportPrivateUsage]
        ('model_replacing_before_model', 'before_model_request')
    ]
    transport = _CapabilityOperationTransport(durability, declaration)
    ctx = RunContext(deps=None, agent=agent, model=original, usage=RunUsage())
    projection = ModelRequestContextProjection(
        [],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
        model_id=None,
        streaming=False,
    )
    wire, deps = transport.dump(CapabilityOperationParams(ctx, arguments={'request_context': projection}))
    registration = next(
        activity
        for activity in durability.temporal_activities
        if ActivityDefinition.must_from_callable(activity).name  # pyright: ignore[reportUnknownMemberType]
        == 'agent__temporal_model_replacement__capability__model_replacing_before_model__before_model_request'
    )

    result = await registration(wire, deps)

    assert result.value.model_id == 'restricted'


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


@requires_dbos
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


@requires_dbos
async def test_dbos_capability_operation_uses_for_run_instance_in_registered_step(dbos: DBOS) -> None:
    observations: list[str] = []
    capability = TenantScopedOperation(observations=observations)
    agent = Agent(
        TestModel(),
        name='dbos_tenant_scope',
        deps_type=str,
        capabilities=[capability, DBOSDurability()],
    )
    workflow_id = str(uuid.uuid4())

    @DBOS.workflow(name=f'capability_tenant_scope_{workflow_id}')
    async def workflow() -> None:
        await agent.run('test', deps='tenant-a')

    with SetWorkflowID(workflow_id):
        await workflow()

    assert observations == ['tenant-a']
    assert capability.tenant == 'unrestricted'


@requires_dbos
async def test_dbos_capability_usage_delta_is_stable_on_replay(dbos: DBOS) -> None:
    capability = UsageOperation()
    agent = Agent(TestModel(), name='dbos_usage', capabilities=[capability, DBOSDurability()])
    workflow_id = str(uuid.uuid4())

    @DBOS.workflow(name=f'capability_usage_{workflow_id}')
    async def workflow() -> tuple[int, int, dict[str, int], Decimal | None, int]:
        result = await agent.run('test')
        usage = result.usage
        return usage.requests, usage.tool_calls, usage.details, usage.cost, usage.details['custom_units']

    with SetWorkflowID(workflow_id):
        first = await workflow()
    with SetWorkflowID(workflow_id):
        replayed = await workflow()

    assert first == replayed == (2, 2, {'summary_tokens': 3, 'custom_units': 7}, Decimal('0.25'), 7)
    assert capability.calls == 1


@pytest.fixture(scope='module')
def prefect_test_server() -> Generator[None, None, None]:
    """Run the module's Prefect flow tests against an isolated test server.

    The implicit ephemeral server uses the shared default PREFECT_HOME and a short
    connect timeout that flakes on slow CI runners; the test harness gives an isolated
    database and a 60s startup budget, mirroring tests/durable_exec/test_prefect.py.

    Ordering constraint: these tests share the 'prefect' xdist group with
    tests/durable_exec/test_prefect.py, whose harness fixture is session-scoped and
    autouse, so once entered it stays entered for the rest of the worker. Today this
    module collects first purely because it sorts alphabetically ahead of test_prefect.py,
    so this module-scoped harness enters and exits before that one starts. Renaming either
    module so this one sorts after test_prefect.py would nest two Prefect harnesses.
    """
    from prefect.testing.utilities import prefect_test_harness

    with prefect_test_harness(server_startup_timeout=60):
        yield


@requires_prefect
@pytest.mark.xdist_group(name='prefect')
async def test_prefect_capability_operation_end_to_end(prefect_test_server: None) -> None:
    capability = Operations()
    agent = Agent(TestModel(), name='prefect_operations', capabilities=[capability, PrefectDurability()])

    @flow
    async def run() -> int:
        await agent.run('test')
        assert capability.result is not None
        return capability.result

    assert await run() == 2


@requires_prefect
@pytest.mark.xdist_group(name='prefect')
async def test_prefect_capability_operation_cache_identity_includes_context_and_model(
    prefect_test_server: None,
) -> None:
    class CacheIdentityOperation(AbstractCapability[str]):
        id = 'cache_identity'

        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        async def before_run(self, ctx: RunContext[str]) -> None:
            await self.read_context(ctx, 1)

        @durable_operation('read_context')
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
        ('tenant-a', 'test'),
        ('tenant-b', 'test'),
        ('tenant-b', 'alternative'),
    ]
