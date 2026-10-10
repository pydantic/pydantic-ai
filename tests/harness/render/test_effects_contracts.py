"""Black-box contracts for effects returned by Render child tasks."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, MutableMapping
from dataclasses import dataclass
from typing import ParamSpec, TypeGuard, TypeVar

import anyio
import pytest
from pydantic import TypeAdapter
from render.workflows import TaskDefinition, Workflows

from pydantic_ai import Agent, CapabilityEvent, CustomEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.exceptions import ModelRetry, UsageLimitExceeded, UserError
from pydantic_ai.messages import ModelMessage, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext, ToolTaskConcurrency, json_round_trip, run_agent_in_task

P = ParamSpec('P')
R = TypeVar('R')

_JSON_OBJECT = TypeAdapter(dict[str, object])
_RUN_USAGE = TypeAdapter(RunUsage)


def _is_json_envelope(value: object) -> TypeGuard[MutableMapping[str, object]]:
    """Whether a task argument or result is the JSON object Render operations cross as."""
    return isinstance(value, dict)


def _refresh_through_json(envelope: MutableMapping[str, object]) -> dict[str, object]:
    """Replace an envelope's contents with a real JSON encode/decode of them.

    The envelope object is reused rather than replaced because `TaskContext.run` is
    generic over each task's own argument and result types, so a decoded substitute
    cannot be returned in place of a value typed as the task's own result. Every value
    inside the envelope is a freshly decoded copy, and a payload Render could not
    serialize raises here instead of crossing.
    """
    decoded = _JSON_OBJECT.validate_json(json.dumps(envelope))
    envelope.clear()
    envelope.update(decoded)
    return decoded


@dataclass(kw_only=True)
class ChildEffectEvent(CustomEvent, name='render_effects_contract.child'):
    """An ordered event emitted by an application function tool in a child task.

    Application code may only emit `CustomEvent`: Pydantic AI refuses a `CapabilityEvent`
    from a plain `@agent.tool`, so capability-event transfer is covered separately by
    `OwnedEffectEvent`, which a capability-contributed tool is allowed to emit.
    """

    child: str
    sequence: int


@dataclass(kw_only=True)
class OwnedEffectEvent(CapabilityEvent, namespace='render_effects_contract', name='owned'):
    """An ordered capability event emitted by a capability-contributed tool."""

    child: str
    sequence: int


@dataclass(kw_only=True)
class ImmediateDecisionEvent(
    CapabilityEvent,
    namespace='render_effects_contract',
    name='decision',
    dispatch='immediate',
):
    """A synchronous decision event that must not be buffered across a child task."""

    operation: str


@dataclass
class ProtectedDecision:
    """Mutable authorization decision shared by an immediate listener and its capability."""

    allowed: bool = True


class EmittingCapability(AbstractCapability[None]):
    """A capability whose own tool emits capability events across the boundary."""

    id = 'owned_effects'

    def __init__(self) -> None:
        async def owned_emit(ctx: RunContext[None]) -> str:
            ctx.usage.incr(RunUsage(details={'owned_marker': 4}))
            await ctx.emit(OwnedEffectEvent(child='owned', sequence=1))
            await ctx.emit(OwnedEffectEvent(child='owned', sequence=2))
            return 'owned'

        self.toolset = FunctionToolset[None]([owned_emit], id='owned-effects')

    def get_toolset(self) -> AbstractToolset[None]:
        return self.toolset


class ImmediateDecisionCapability(AbstractCapability[None]):
    """A capability that must receive a synchronous decision before acting."""

    id = 'immediate_decision'

    def __init__(self, decision: ProtectedDecision, protected_actions: list[str]) -> None:
        async def protected_operation(ctx: RunContext[None]) -> str:
            await ctx.emit(ImmediateDecisionEvent(operation='publish'))
            if decision.allowed:
                protected_actions.append('published')
            return 'checked'

        self.toolset = FunctionToolset[None]([protected_operation], id='immediate-decision')

    def get_toolset(self) -> AbstractToolset[None]:
        return self.toolset


class JsonRecordingTaskContext(RecordingTaskContext):
    """Capture the JSON requests and results passed through the shared task fixture."""

    def __init__(self) -> None:
        super().__init__()
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.results: list[tuple[str, dict[str, object]]] = []

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        for argument in args:
            assert _is_json_envelope(argument)
            self.requests.append((task.name, json_round_trip(dict(argument))))
        result = await super().run(task, *args, **kwargs)
        assert _is_json_envelope(result)
        self.results.append((task.name, json_round_trip(dict(result))))
        return result


class SiblingCallToolTaskContext(JsonRecordingTaskContext):
    """Hold each sibling `call_tool` task until both runs are in flight.

    The barrier is in `run`, before `TaskDefinition.func`, so neither operation body
    proceeds until overlap is observable. Model and other tasks are not gated.
    """

    def __init__(self, *, siblings: int = 2) -> None:
        super().__init__()
        self._siblings = siblings
        self.concurrency = ToolTaskConcurrency()
        self._overlap = anyio.Event()

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        with self.concurrency.track(task.name) as is_tool_call:
            if is_tool_call:
                if self.concurrency.active == self._siblings:
                    self._overlap.set()
                await self._overlap.wait()
            return await super().run(task, *args, **kwargs)


class MalformedEffectsTaskContext(JsonRecordingTaskContext):
    """Replace one child task's effects with malformed JSON-compatible data."""

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        result = await super().run(task, *args, **kwargs)
        if task.name.endswith('.call_tool') and _is_json_envelope(result) and result.get('status') == 'ok':
            result['effects'] = 'not-an-effects-object'
            _refresh_through_json(result)
        return result


class NegativeUsageEffectsTaskContext(JsonRecordingTaskContext):
    """Inject negative usage into an otherwise successful child result."""

    def __init__(self, caller_usage: RunUsage) -> None:
        super().__init__()
        self.caller_usage = caller_usage
        self.usage_before_effects: RunUsage | None = None

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        result = await super().run(task, *args, **kwargs)
        if task.name.endswith('.call_tool') and _is_json_envelope(result) and result.get('status') == 'ok':
            self.usage_before_effects = _RUN_USAGE.validate_json(_RUN_USAGE.dump_json(self.caller_usage))
            result['effects'] = {
                'usage': {
                    'requests': -1_000,
                    'input_tokens': -2_000,
                    'details': {'injected_negative_count': -3_000},
                }
            }
            _refresh_through_json(result)
        return result


def _event_hooks(seen: list[ChildEffectEvent]) -> Hooks[None]:
    hooks = Hooks[None]()

    @hooks.on.event(ChildEffectEvent)
    async def record(ctx: RunContext[None], event: ChildEffectEvent) -> None:
        del ctx
        seen.append(event)

    return hooks


def _owned_event_hooks(seen: list[OwnedEffectEvent]) -> Hooks[None]:
    hooks = Hooks[None]()

    @hooks.on.event(OwnedEffectEvent)
    async def record(ctx: RunContext[None], event: OwnedEffectEvent) -> None:
        del ctx
        seen.append(event)

    return hooks


def _immediate_decision_hooks(decision: ProtectedDecision) -> Hooks[None]:
    hooks = Hooks[None]()

    @hooks.on.event(ImmediateDecisionEvent)
    async def deny(ctx: RunContext[None], event: ImmediateDecisionEvent) -> None:
        del ctx, event
        decision.allowed = False

    return hooks


async def test_child_usage_delta_is_applied_to_the_original_usage_once() -> None:
    usage = RunUsage(details={'caller_marker': 11})
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['account']),
        name='usage-effects',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def account(ctx: RunContext[None]) -> str:
        ctx.usage.incr(RunUsage(details={'child_marker': 7}))
        return 'accounted'

    context = JsonRecordingTaskContext()
    assert isinstance(await run_agent_in_task(agent, runtime, context, usage=usage), str)

    assert usage.details['caller_marker'] == 11
    assert usage.details['child_marker'] == 7
    assert context.task_names.count('usage-effects__function_toolset__<agent>.call_tool') == 1


async def test_child_events_are_replayed_to_the_caller_in_order_once() -> None:
    seen: list[ChildEffectEvent] = []
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['publish']),
        name='event-effects',
        deps_type=type(None),
        capabilities=[_event_hooks(seen), runtime],
    )

    @agent.tool
    async def publish(ctx: RunContext[None]) -> str:
        await ctx.emit(ChildEffectEvent(child='only', sequence=1))
        await ctx.emit(ChildEffectEvent(child='only', sequence=2))
        return 'published'

    assert isinstance(
        await run_agent_in_task(agent, runtime, JsonRecordingTaskContext(), usage=RunUsage()),
        str,
    )
    assert [(event.child, event.sequence) for event in seen] == [('only', 1), ('only', 2)]


async def test_capability_owned_tool_transfers_its_capability_events_and_usage() -> None:
    seen: list[OwnedEffectEvent] = []
    usage = RunUsage()
    capability = EmittingCapability()
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['owned_emit']),
        name='owned-effects',
        deps_type=type(None),
        capabilities=[capability, _owned_event_hooks(seen), runtime],
    )

    context = JsonRecordingTaskContext()
    assert isinstance(await run_agent_in_task(agent, runtime, context, usage=usage), str)

    assert usage.details['owned_marker'] == 4
    assert [(event.child, event.sequence) for event in seen] == [('owned', 1), ('owned', 2)]
    assert context.task_names.count('owned-effects__function_toolset__owned-effects.call_tool') == 1


async def test_concurrent_child_effects_are_additive_and_ordered_per_child() -> None:
    seen: list[ChildEffectEvent] = []
    usage = RunUsage()
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['alpha', 'beta']),
        name='sibling-effects',
        deps_type=type(None),
        capabilities=[_event_hooks(seen), runtime],
    )

    async def apply_effects(ctx: RunContext[None], child: str, amount: int) -> str:
        ctx.usage.incr(RunUsage(details={f'{child}_marker': amount}))
        await ctx.emit(ChildEffectEvent(child=child, sequence=1))
        await ctx.emit(ChildEffectEvent(child=child, sequence=2))
        return child

    @agent.tool
    async def alpha(ctx: RunContext[None]) -> str:
        return await apply_effects(ctx, 'alpha', 2)

    @agent.tool
    async def beta(ctx: RunContext[None]) -> str:
        return await apply_effects(ctx, 'beta', 5)

    context = SiblingCallToolTaskContext()
    with anyio.fail_after(5):
        assert isinstance(await run_agent_in_task(agent, runtime, context, usage=usage), str)

    assert context.concurrency.maximum == 2
    assert context.concurrency.active == 0
    assert usage.details['alpha_marker'] == 2
    assert usage.details['beta_marker'] == 5
    assert [(event.child, event.sequence) for event in seen if event.child == 'alpha'] == [('alpha', 1), ('alpha', 2)]
    assert [(event.child, event.sequence) for event in seen if event.child == 'beta'] == [('beta', 1), ('beta', 2)]
    assert context.task_names.count('sibling-effects__function_toolset__<agent>.call_tool') == 2


async def _retry_then_finish_stream(
    messages: list[ModelMessage],
    info: AgentInfo,
) -> AsyncIterator[DeltaToolCalls | str]:
    del info
    parts = [part for message in messages for part in message.parts]
    if any(isinstance(part, ToolReturnPart) for part in parts):
        yield 'done'
    else:
        yield {0: DeltaToolCall(name='retrying', json_args='{}', tool_call_id='retrying')}


async def test_failed_attempt_effects_are_discarded_and_success_is_applied_once() -> None:
    seen: list[ChildEffectEvent] = []
    usage = RunUsage()
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        FunctionModel(stream_function=_retry_then_finish_stream),
        name='retry-effects',
        deps_type=type(None),
        retries=1,
        capabilities=[_event_hooks(seen), runtime],
    )

    @agent.tool
    async def retrying(ctx: RunContext[None]) -> str:
        attempt = ctx.retry + 1
        ctx.usage.incr(RunUsage(details={f'attempt_{attempt}': 1}))
        await ctx.emit(ChildEffectEvent(child=f'attempt-{attempt}', sequence=1))
        if ctx.retry == 0:
            raise ModelRetry('retry once')
        return 'recovered'

    assert await run_agent_in_task(agent, runtime, JsonRecordingTaskContext(), usage=usage) == 'done'

    assert 'attempt_1' not in usage.details
    assert usage.details['attempt_2'] == 1
    assert [(event.child, event.sequence) for event in seen] == [('attempt-2', 1)]


async def test_malformed_effects_fail_closed() -> None:
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['account']),
        name='malformed-effects',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def account(ctx: RunContext[None]) -> str:
        ctx.usage.incr(RunUsage(details={'must_not_apply': 1}))
        return 'accounted'

    with pytest.raises(ValueError, match='Operation effects must be a JSON object'):
        await run_agent_in_task(agent, runtime, MalformedEffectsTaskContext(), usage=RunUsage())


async def test_negative_usage_effects_fail_closed_without_mutating_caller_usage() -> None:
    usage = RunUsage(details={'caller_marker': 13}, requests=2, input_tokens=17)
    context = NegativeUsageEffectsTaskContext(usage)
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['account']),
        name='negative-usage-effects',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def account(ctx: RunContext[None]) -> str:
        del ctx
        return 'accounted'

    with pytest.raises(ValueError, match='Operation effects usage deltas cannot contain negative counts'):
        await run_agent_in_task(agent, runtime, context, usage=usage)

    assert context.usage_before_effects is not None
    assert usage == context.usage_before_effects


@pytest.mark.parametrize('denied', [False, True])
async def test_immediate_capability_event_inline_control_respects_decision(denied: bool) -> None:
    decision = ProtectedDecision()
    protected_actions: list[str] = []
    capability = ImmediateDecisionCapability(decision, protected_actions)
    agent = Agent[None, str](
        TestModel(call_tools=['protected_operation']),
        name='immediate-effects-inline-control',
        deps_type=type(None),
        capabilities=[capability, _immediate_decision_hooks(decision)] if denied else [capability],
    )

    assert isinstance((await agent.run('check before publishing')).output, str)
    assert decision.allowed is (not denied)
    assert protected_actions == ([] if denied else ['published'])


async def test_immediate_capability_event_fails_before_protected_action() -> None:
    decision = ProtectedDecision()
    protected_actions: list[str] = []
    capability = ImmediateDecisionCapability(decision, protected_actions)
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['protected_operation']),
        name='immediate-effects',
        deps_type=type(None),
        capabilities=[capability, _immediate_decision_hooks(decision), runtime],
    )

    with pytest.raises(
        UserError,
        match='Immediate capability events are unsupported inside a Render child task',
    ):
        await run_agent_in_task(agent, runtime, JsonRecordingTaskContext(), usage=RunUsage())

    assert protected_actions == []


async def test_new_operation_requests_and_results_use_protocol_v2() -> None:
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['account']),
        name='protocol-v2',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def account(ctx: RunContext[None]) -> str:
        ctx.usage.incr(RunUsage(details={'v2_marker': 1}))
        return 'accounted'

    context = JsonRecordingTaskContext()
    assert isinstance(await run_agent_in_task(agent, runtime, context, usage=RunUsage()), str)

    request_versions = [request.get('version') for _, request in context.requests]
    result_versions = [result.get('version') for _, result in context.results]
    assert (request_versions, result_versions) == ([2, 2, 2], [2, 2, 2])


async def test_nested_agent_model_usage_matches_an_inline_run() -> None:
    """A model called inside a tool contributes tokens beyond the parent's own responses."""
    child = Agent(TestModel(call_tools=[], custom_output_text='child result'), name='usage-child')
    runtime = RenderWorkflows[None](Workflows())
    parent = Agent(
        TestModel(call_tools=['delegate']), name='usage-parent', deps_type=type(None), capabilities=[runtime]
    )

    @parent.tool
    async def delegate(ctx: RunContext[None]) -> str:
        return (await child.run('child', usage=ctx.usage)).output

    inline = await parent.run('parent')
    with runtime.activate(JsonRecordingTaskContext()):
        distributed = await parent.run('parent')
    assert inline.usage.requests == distributed.usage.requests == 3
    assert inline.usage.input_tokens == distributed.usage.input_tokens
    assert inline.usage.output_tokens == distributed.usage.output_tokens


async def test_nested_agent_usage_counts_toward_parent_request_limit() -> None:
    """A model request inside a tool must count before the parent requests again."""
    child = Agent(TestModel(call_tools=[], custom_output_text='child result'), name='limited-child')
    runtime = RenderWorkflows[None](Workflows())
    parent = Agent(
        TestModel(call_tools=['delegate']), name='limited-parent', deps_type=type(None), capabilities=[runtime]
    )

    @parent.tool
    async def delegate(ctx: RunContext[None]) -> str:
        return (await child.run('child', usage=ctx.usage)).output

    inline_usage = RunUsage()
    with pytest.raises(UsageLimitExceeded, match='request_limit of 2'):
        await parent.run('parent', usage=inline_usage, usage_limits=UsageLimits(request_limit=2))

    distributed_usage = RunUsage()
    with runtime.activate(JsonRecordingTaskContext()):
        with pytest.raises(UsageLimitExceeded, match='request_limit of 2'):
            await parent.run('parent', usage=distributed_usage, usage_limits=UsageLimits(request_limit=2))

    assert inline_usage.requests == distributed_usage.requests == 2


@pytest.mark.parametrize('capability_owned', [False, True])
async def test_worker_rejects_events_from_the_wrong_owner(capability_owned: bool) -> None:
    async def emit_wrong_event(ctx: RunContext[None]) -> str:
        event = (
            ChildEffectEvent(child='wrong', sequence=1)
            if capability_owned
            else OwnedEffectEvent(child='wrong', sequence=1)
        )
        await ctx.emit(event)
        pytest.fail('the worker must reject an event from the wrong owner')  # pragma: no cover

    class Owner(AbstractCapability[None]):
        id = 'owner'

        def get_toolset(self) -> AbstractToolset[None]:
            return FunctionToolset([emit_wrong_event], id='owner')

    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(
        TestModel(),
        name='wrong-owner',
        deps_type=type(None),
        tools=[] if capability_owned else [emit_wrong_event],
        capabilities=[Owner(), runtime] if capability_owned else [runtime],
    )
    with pytest.raises(UserError, match=r'must emit|cannot be emitted'):
        await run_agent_in_task(agent, runtime, JsonRecordingTaskContext())


async def test_worker_preserves_explicit_event_ownership_and_tool_call() -> None:
    seen: list[OwnedEffectEvent] = []

    class Owner(AbstractCapability[None]):
        id = 'owner'

        def get_toolset(self) -> AbstractToolset[None]:
            async def emit_owned(ctx: RunContext[None]) -> str:
                await ctx.emit(
                    OwnedEffectEvent(
                        child='owned',
                        sequence=1,
                        capability_id='owner',
                        tool_call_id='original',
                        tool_name='original_tool',
                    )
                )
                return 'emitted'

            return FunctionToolset([emit_owned], id='owner')

    runtime = RenderWorkflows[None](Workflows())
    agent = Agent(
        TestModel(),
        name='explicit-event',
        deps_type=type(None),
        capabilities=[Owner(), _owned_event_hooks(seen), runtime],
    )
    await run_agent_in_task(agent, runtime, JsonRecordingTaskContext())
    assert len(seen) == 1
    assert (seen[0].capability_id, seen[0].tool_call_id, seen[0].tool_name) == ('owner', 'original', 'original_tool')
