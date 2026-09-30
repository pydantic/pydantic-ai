"""Black-box contracts for effects returned by Render child tasks."""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator, MutableMapping
from dataclasses import dataclass
from typing import ParamSpec, TypeGuard, TypeVar

import anyio
import pytest
from pydantic import TypeAdapter
from render.workflows import TaskContext, TaskDefinition, Workflows

from pydantic_ai import Agent, CapabilityEvent, CustomEvent, RunContext
from pydantic_ai.capabilities import AbstractCapability, Hooks
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import AbstractToolset, FunctionToolset
from pydantic_ai.usage import RunUsage
from pydantic_ai_harness import RenderWorkflows

from .conftest import RecordingTaskContext

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
    """Run public task definitions across a real JSON encode/decode boundary.

    This is Level 2/4 boundary evidence: it invokes the public `TaskDefinition.func`
    locally and does not claim process isolation or Render service execution.
    """

    def __init__(self) -> None:
        self.task_names: list[str] = []
        self.requests: list[tuple[str, dict[str, object]]] = []
        self.results: list[tuple[str, dict[str, object]]] = []

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        self.task_names.append(task.name)
        for argument in args:
            if _is_json_envelope(argument):
                self.requests.append((task.name, _refresh_through_json(argument)))
        pending = task.func(self, *args, **kwargs)
        if inspect.isawaitable(pending):
            return self._record(task.name, await pending)
        return self._record(task.name, pending)

    def _record(self, task_name: str, result: R) -> R:
        """Round-trip and keep one task result, leaving non-JSON results untouched."""
        if _is_json_envelope(result):
            self.results.append((task_name, _refresh_through_json(result)))
        return result


class SiblingCallToolTaskContext(JsonRecordingTaskContext):
    """Hold each sibling `call_tool` task until both runs are in flight.

    The barrier is in `run`, before `TaskDefinition.func`, so neither operation body
    proceeds until overlap is observable. Model and other tasks are not gated.
    """

    def __init__(self, *, siblings: int = 2) -> None:
        super().__init__()
        self._siblings = siblings
        self.active_tool_tasks = 0
        self.max_active_tool_tasks = 0
        self._overlap = anyio.Event()

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        is_tool_call = task.name.endswith('.call_tool')
        if is_tool_call:
            self.active_tool_tasks += 1
            self.max_active_tool_tasks = max(self.max_active_tool_tasks, self.active_tool_tasks)
            if self.active_tool_tasks == self._siblings:
                self._overlap.set()
            await self._overlap.wait()
        try:
            return await super().run(task, *args, **kwargs)
        finally:
            if is_tool_call:
                self.active_tool_tasks -= 1


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


class V1CompatibilityTaskContext(JsonRecordingTaskContext):
    """Send v1 requests to the child and record its compatibility responses."""

    def __init__(self) -> None:
        super().__init__()
        self.new_request_versions: list[object] = []
        self.compatibility_results: list[dict[str, object]] = []

    async def run(self, task: TaskDefinition[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        for argument in args:
            if _is_json_envelope(argument):
                self.new_request_versions.append(argument.get('version'))
                argument['version'] = 1
        result = await super().run(task, *args, **kwargs)
        if _is_json_envelope(result):
            self.compatibility_results.append(_JSON_OBJECT.validate_json(json.dumps(result)))
        return result


async def _run_agent(
    agent: Agent[None, str],
    runtime: RenderWorkflows[None],
    context: TaskContext,
    *,
    usage: RunUsage,
) -> str:
    @runtime.task
    async def root(ctx: TaskContext) -> str:
        del ctx
        return (await agent.run('exercise effects', usage=usage)).output

    pending = root.func(context)
    assert inspect.isawaitable(pending)
    return await pending


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
    assert isinstance(await _run_agent(agent, runtime, context, usage=usage), str)

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
        await _run_agent(agent, runtime, JsonRecordingTaskContext(), usage=RunUsage()),
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
    assert isinstance(await _run_agent(agent, runtime, context, usage=usage), str)

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
        assert isinstance(await _run_agent(agent, runtime, context, usage=usage), str)

    assert context.max_active_tool_tasks == 2
    assert usage.details['alpha_marker'] == 2
    assert usage.details['beta_marker'] == 5
    assert [(event.child, event.sequence) for event in seen if event.child == 'alpha'] == [('alpha', 1), ('alpha', 2)]
    assert [(event.child, event.sequence) for event in seen if event.child == 'beta'] == [('beta', 1), ('beta', 2)]
    assert context.task_names.count('sibling-effects__function_toolset__<agent>.call_tool') == 2


def _retry_then_finish(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del info
    parts = [part for message in messages for part in message.parts]
    if any(isinstance(part, ToolReturnPart) for part in parts):
        return ModelResponse(parts=[TextPart('done')])
    return ModelResponse(parts=[ToolCallPart('retrying', {}, tool_call_id='retrying')])


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
        FunctionModel(_retry_then_finish, stream_function=_retry_then_finish_stream),
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

    assert await _run_agent(agent, runtime, JsonRecordingTaskContext(), usage=usage) == 'done'

    assert 'attempt_1' not in usage.details
    assert usage.details['attempt_2'] == 1
    assert [(event.child, event.sequence) for event in seen] == [('attempt-2', 1)]


async def test_effects_share_the_successful_operation_result_json_envelope() -> None:
    usage = RunUsage()
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['account']),
        name='effect-envelope',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def account(ctx: RunContext[None]) -> str:
        ctx.usage.incr(RunUsage(details={'envelope_marker': 3}))
        return 'accounted'

    context = JsonRecordingTaskContext()
    assert isinstance(await _run_agent(agent, runtime, context, usage=usage), str)
    tool_results = [result for name, result in context.results if name.endswith('.call_tool')]

    assert len(tool_results) == 1
    assert set(tool_results[0]) == {'version', 'status', 'payload', 'effects'}
    assert isinstance(tool_results[0]['effects'], dict)


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
        await _run_agent(agent, runtime, MalformedEffectsTaskContext(), usage=RunUsage())


@pytest.mark.skip(reason='unsupported: the public result contract defines no effects result-size limit')
def test_effect_result_size_limit_is_unsupported() -> None:
    """Document the unsupported size-limit row without inventing a policy."""


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
        await _run_agent(agent, runtime, context, usage=usage)

    assert context.usage_before_effects is not None
    assert usage == context.usage_before_effects


async def test_immediate_capability_event_inline_control_denies_protected_action() -> None:
    decision = ProtectedDecision()
    protected_actions: list[str] = []
    capability = ImmediateDecisionCapability(decision, protected_actions)
    agent = Agent[None, str](
        TestModel(call_tools=['protected_operation']),
        name='immediate-effects-inline-control',
        deps_type=type(None),
        capabilities=[capability, _immediate_decision_hooks(decision)],
    )

    assert isinstance((await agent.run('check before publishing')).output, str)
    assert decision.allowed is False
    assert protected_actions == []


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
        await _run_agent(agent, runtime, JsonRecordingTaskContext(), usage=RunUsage())

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
    assert isinstance(await _run_agent(agent, runtime, context, usage=RunUsage()), str)

    request_versions = [request.get('version') for _, request in context.requests]
    result_versions = [result.get('version') for _, result in context.results]
    assert (request_versions, result_versions) == ([2, 2, 2], [2, 2, 2])


async def test_new_caller_accepts_effect_free_v1_compatibility_results() -> None:
    usage = RunUsage(details={'caller_marker': 19})
    runtime = RenderWorkflows[None](Workflows(), deps_type=type(None))
    agent = Agent[None, str](
        TestModel(call_tools=['account']),
        name='protocol-v1-compatibility',
        deps_type=type(None),
        capabilities=[runtime],
    )

    @agent.tool
    async def account(ctx: RunContext[None]) -> str:
        ctx.usage.incr(RunUsage(details={'must_not_cross_v1': 5}))
        return 'accounted'

    context = V1CompatibilityTaskContext()
    assert isinstance(await _run_agent(agent, runtime, context, usage=usage), str)

    observed = {
        'new_request_versions': context.new_request_versions,
        'compatibility_result_versions': [result.get('version') for result in context.compatibility_results],
        'compatibility_result_has_effects': ['effects' in result for result in context.compatibility_results],
        'caller_usage_details': usage.details,
    }
    assert observed == {
        'new_request_versions': [2, 2, 2],
        'compatibility_result_versions': [1, 1, 1],
        'compatibility_result_has_effects': [False, False, False],
        'caller_usage_details': {'caller_marker': 19},
    }
