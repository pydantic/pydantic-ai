"""Tests for the `AbsurdDurability` capability.

Behavior is driven through `Agent(..., capabilities=[AbsurdDurability()])` inside an in-memory
`FakeAsyncTaskContext` (see `_helpers.py`) so there is no Postgres or Docker dependency. The two
production behaviors the capability relies on -- encounter-order step-name disambiguation and a
replay that serves stored checkpoints without re-running `fn` -- are reproduced faithfully by the
fake. `test_pydantic_ai_absurd_compat.py` checks the same format against a real Absurd schema.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractContextManager

import anyio
import pytest

pytest.importorskip('absurd_sdk')

from absurd_sdk import JsonValue
from inline_snapshot import snapshot

from pydantic_ai import Agent, ToolReturn
from pydantic_ai.agent import ParallelExecutionMode
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import ExternalToolset, FunctionToolset
from pydantic_ai_harness.absurd import AbsurdDurability

from ._helpers import FakeAsyncTaskContext, FakeSyncTaskContext, absurd_task_context

pytestmark = pytest.mark.anyio


def _text_model(counter: dict[str, int] | None = None) -> FunctionModel:
    tally = counter if counter is not None else {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tally['calls'] += 1
        return ModelResponse(parts=[TextPart(content='ok')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        tally['calls'] += 1
        yield 'ok'

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


def _tool_then_done_model(tool_name: str, args: dict[str, JsonValue]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(
            isinstance(part, (ToolReturnPart, RetryPromptPart)) for message in messages for part in message.parts
        )
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(fn, model_name='fn')


class TestTransparency:
    async def test_run_outside_task_is_transparent(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])
        result = await agent.run('hi')
        assert result.output == 'ok'
        assert counter['calls'] == 1


class TestCapabilityOperation:
    async def test_operation_is_checkpointed_and_replayed(self) -> None:
        calls: list[str] = []

        class Recorder(AbstractCapability[object]):
            id = 'recorder'

            async def before_run(self, ctx: RunContext[object]) -> None:
                await self.record(ctx, 'started')

            @durable_operation('record')
            async def record(self, ctx: RunContext[object], value: str) -> None:
                del ctx
                calls.append(value)

        agent = Agent(_text_model(), name='cap', capabilities=[Recorder(), AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')

        operation_name = 'cap__capability__recorder.record'
        assert operation_name in ctx.stored
        assert calls == ['started']

        replay = ctx.replay()
        with absurd_task_context(replay):
            await agent.run('hi')

        assert calls == ['started']
        assert replay.invoked == []


class TestModelRequestCheckpoint:
    async def test_request_checkpointed_and_replay_serves_cache(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('hi')
        assert 'a__model.request' in ctx.stored
        assert ctx.invoked == ['a__model.request']

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('hi')

        assert counter['calls'] == 1
        assert first.output == second.output == 'ok'
        assert replay.invoked == []


class TestStreaming:
    async def test_stream_checkpointed_and_replayed(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='stream', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            async with agent.run_stream('hi') as result:
                first_out = await result.get_output()
        assert first_out == 'ok'
        assert 'stream__model.request_stream' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            async with agent.run_stream('hi') as result:
                replay_out = await result.get_output()

        assert replay_out == 'ok'
        assert counter['calls'] == 1
        assert replay.invoked == []

    async def test_handler_sees_live_events_and_stream_replays_equal(self) -> None:
        live_events: list[AgentStreamEvent] = []

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                live_events.append(event)

        counter = {'calls': 0}
        agent = Agent(
            _text_model(counter), name='stream', capabilities=[AbsurdDurability(event_stream_handler=handler)]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            async with agent.run_stream_events('hi') as stream:
                first_events = [event async for event in stream]

        # The model-stream events were delivered live to the handler inside the request_stream step.
        assert any(isinstance(event, (PartStartEvent, PartDeltaEvent)) for event in live_events)
        assert 'stream__model.request_stream' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            async with agent.run_stream_events('hi') as stream:
                replay_events = [event async for event in stream]

        assert counter['calls'] == 1
        assert replay_events == first_events


class TestFunctionTool:
    async def test_tool_checkpointed_exactly_once_across_replay(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='billing')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:
            calls['n'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_then_done_model('charge_card', {'amount': 7}),
            name='pay',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('charge it')
        # The raw tool return value, as `pydantic-ai-absurd` stores it.
        assert ctx.stored['pay__function_toolset__billing.call_tool:charge_card'] == 'charged 7'

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('charge it')

        assert calls['n'] == 1
        assert first.output == second.output == 'done'
        assert replay.invoked == []


class TestModelRetry:
    async def test_model_retry_is_not_checkpointed_and_the_tool_reruns_on_replay(self) -> None:
        calls = {'model': 0, 'tool': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            calls['tool'] += 1
            raise ModelRetry('nope, try again')

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            calls['model'] += 1
            if any(isinstance(p, RetryPromptPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='flaky', args={})])

        agent = Agent(FunctionModel(model_fn), name='retry', toolsets=[toolset], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')
        tool_step = 'retry__function_toolset__tools.call_tool:flaky'
        # As in `pydantic-ai-absurd`, the `ModelRetry` propagates out of the step, so nothing is stored.
        assert sorted(ctx.stored) == ['retry__model.request', 'retry__model.request#2']
        assert first.output == 'done'

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        # Both model responses come from their checkpoints; only the tool runs again.
        assert second.output == 'done'
        assert calls == {'model': 2, 'tool': 2}
        assert replay.invoked == [tool_step]


class TestToolResultFormat:
    async def test_tool_return_object_round_trips_through_replay(self) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def lookup() -> ToolReturn:
            return ToolReturn(return_value='value', content='extra context', metadata={'source': 'db'})

        agent = Agent(
            _tool_then_done_model('lookup', {}), name='tr', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')
        # A `ToolReturn` has no raw form in `pydantic-ai-absurd`, so it is stored under a reserved key.
        assert ctx.stored['tr__function_toolset__tools.call_tool:lookup'] == snapshot(
            {
                '__pydantic_ai_harness_absurd_tool_result__': {
                    'result': {
                        'return_value': 'value',
                        'content': 'extra context',
                        'metadata': {'source': 'db'},
                        'tools': None,
                        'kind': 'tool-return',
                    },
                    'kind': 'tool_return',
                }
            }
        )

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        def returned(messages: list[ModelMessage]) -> list[tuple[object, object]]:
            return [(p.content, p.metadata) for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]

        assert replay.invoked == []
        assert returned(second.all_messages()) == returned(first.all_messages()) == [('value', {'source': 'db'})]

    @pytest.mark.parametrize(
        'value',
        [{'kind': 'tool-return', 'rows': 3}, {'__pydantic_ai_harness_absurd_tool_result__': 'hello'}],
        ids=['tool-return-kind', 'reserved-key'],
    )
    async def test_raw_dict_that_looks_encoded_round_trips(self, value: dict[str, object]) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def query() -> dict[str, object]:
            return value

        agent = Agent(
            _tool_then_done_model('query', {}), name='raw', toolsets=[toolset], capabilities=[AbsurdDurability()]
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('go')
        assert ctx.stored['raw__function_toolset__tools.call_tool:query'] == value

        replay = ctx.replay()
        with absurd_task_context(replay):
            result = await agent.run('go')

        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert [r.content for r in returns] == [value]
        assert replay.invoked == []


class TestCrashMidRunRetry:
    async def test_model_step_served_from_checkpoint_while_failed_tool_reruns(self) -> None:
        # The core value prop: the model step completes and is checkpointed, then a tool raises a
        # real (non-`ModelRetry`) error that fails the task. On retry, Absurd replays: the model
        # step is served from its checkpoint (model not called again) while the tool re-runs.
        model_calls = {'n': 0}
        tool_attempts = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            tool_attempts['n'] += 1
            if tool_attempts['n'] == 1:
                raise RuntimeError('worker died mid-tool')
            return 'recovered'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            model_calls['n'] += 1
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='flaky', args={})])

        agent = Agent(FunctionModel(model_fn), name='crash', toolsets=[toolset], capabilities=[AbsurdDurability()])

        model_step = 'crash__model.request'
        tool_step = 'crash__function_toolset__tools.call_tool:flaky'

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(RuntimeError, match='worker died mid-tool'):
                await agent.run('go')

        # The model step checkpointed before the tool ran; the failed tool step did not.
        assert model_step in ctx.stored
        assert tool_step not in ctx.stored
        assert model_calls['n'] == 1
        assert tool_attempts['n'] == 1

        replay = ctx.replay()
        with absurd_task_context(replay):
            result = await agent.run('go')

        assert result.output == 'done'
        # The first model request was served from its checkpoint (not re-invoked); the tool re-ran
        # and the second model turn is a fresh step.
        assert model_step not in replay.invoked
        assert tool_step in replay.invoked
        assert f'{model_step}#2' in replay.invoked
        assert model_calls['n'] == 2
        assert tool_attempts['n'] == 2


class TestModelSelection:
    async def test_registered_model_folds_id_into_step_and_replays(self) -> None:
        primary = {'n': 0}
        cheap = {'n': 0}

        def primary_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            primary['n'] += 1
            return ModelResponse(parts=[TextPart(content='primary')])

        def cheap_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            cheap['n'] += 1
            return ModelResponse(parts=[TextPart(content='cheap')])

        agent = Agent(
            FunctionModel(primary_fn, model_name='primary'),
            name='sw',
            capabilities=[AbsurdDurability(models={'cheap': FunctionModel(cheap_fn, model_name='cheap')})],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            default_result = await agent.run('hi')
            cheap_result = await agent.run('hi', model='cheap')

        assert default_result.output == 'primary'
        assert cheap_result.output == 'cheap'
        assert 'sw__model.request' in ctx.stored
        assert 'sw__model.request.cheap' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            replayed = await agent.run('hi', model='cheap')

        assert replayed.output == 'cheap'
        assert cheap['n'] == 1
        assert replay.invoked == []

    async def test_string_default_model_gets_unsuffixed_step_name(self) -> None:
        agent = Agent('test', name='strdef', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('hi')

        assert 'strdef__model.request' in ctx.stored
        assert not any(name.endswith('.test') for name in ctx.stored)


class TestRuntimeToolsets:
    async def test_runtime_executing_toolset_rejected_inside_task(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        late = FunctionToolset(id='late')

        # Rejected before it can run.
        @late.tool_plain
        def echo(value: str) -> str:  # pragma: no cover
            return value

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match=r'cannot be added at runtime with Absurd'):
                await agent.run('hi', toolsets=[late])

    async def test_non_executing_runtime_toolset_allowed_inside_task(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi', toolsets=[ExternalToolset(tool_defs=[])])
        assert result.output == 'ok'

    async def test_toolset_decorated_after_construction_rejected_inside_task(self) -> None:
        agent = Agent(_text_model(), name='decorated', capabilities=[AbsurdDurability()])

        # Rejected first.
        @agent.toolset(id='decorated-tools')
        def build(ctx: RunContext[object]) -> FunctionToolset[object]:  # pragma: no cover
            return FunctionToolset[object](id='inner')

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(UserError, match=r'cannot be added at runtime with Absurd'):
                await agent.run('hi')


class TestNonWrappedLeaf:
    async def test_external_toolset_passes_through_unwrapped(self) -> None:
        external = ExternalToolset(tool_defs=[])
        agent = Agent(_text_model(), name='a', toolsets=[external], capabilities=[AbsurdDurability()])
        assert any(leaf is external for leaf in agent.toolsets)


class TestBindingErrors:
    async def test_unnamed_agent_raises(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(_text_model(), capabilities=[AbsurdDurability()])

    async def test_duplicate_toolset_ids_raise(self) -> None:
        first = FunctionToolset(id='dup')

        # Never invoked, only the wrap check runs.
        @first.tool_plain
        def echo(value: str) -> str:  # pragma: no cover
            return value

        second = FunctionToolset(id='dup')

        # Never invoked, only the wrap check runs.
        @second.tool_plain
        def shout(value: str) -> str:  # pragma: no cover
            return value.upper()

        with pytest.raises(UserError, match='same `id`'):
            Agent(_text_model(), name='a', toolsets=[first, second], capabilities=[AbsurdDurability()])


class TestIdLessToolset:
    async def test_id_less_function_toolset_uses_pydantic_ai_absurd_step_names(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset()

        @toolset.tool_plain
        def charge(amount: int) -> str:
            calls['n'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_then_done_model('charge', {'amount': 3}),
            name='idless',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')
        assert ctx.stored['idless__function_toolset.call_tool:charge'] == 'charged 3'

        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')
        assert first.output == second.output == 'done'
        assert calls['n'] == 1
        assert replay.invoked == []

    async def test_id_less_toolset_mounted_twice_shares_one_wrapper(self) -> None:
        calls: list[str] = []
        toolset = FunctionToolset()

        @toolset.tool_plain
        def ping(caller: str) -> str:
            calls.append(caller)
            return f'pong {caller}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='a_ping', args={'caller': 'a'}, tool_call_id='a'),
                    ToolCallPart(tool_name='b_ping', args={'caller': 'b'}, tool_call_id='b'),
                ]
            )

        agent = Agent(
            FunctionModel(model_fn),
            name='shared',
            toolsets=[toolset.prefixed('a'), toolset.prefixed('b')],
            capabilities=[AbsurdDurability()],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            await agent.run('go')
        assert {name: value for name, value in ctx.stored.items() if 'call_tool' in name} == snapshot(
            {'shared__function_toolset.call_tool:ping': 'pong a', 'shared__function_toolset.call_tool:ping#2': 'pong b'}
        )

        with absurd_task_context(ctx.replay()):
            await agent.run('go')
        assert calls == ['a', 'b']


class TestSyncContext:
    async def test_sync_task_context_raises(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[AbsurdDurability()])
        with absurd_task_context(FakeSyncTaskContext()):
            with pytest.raises(UserError, match='requires an async Absurd task context'):
                await agent.run('hi')


class TestParallelExecutionMode:
    async def test_mode_applied_inside_and_outside_a_task(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = Agent(
            _text_model(), name='a', capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')]
        )
        recorded: list[ParallelExecutionMode] = []
        real = agent.parallel_tool_call_execution_mode

        def spy(mode: ParallelExecutionMode = 'parallel') -> AbstractContextManager[None]:
            recorded.append(mode)
            return real(mode)

        monkeypatch.setattr(agent, 'parallel_tool_call_execution_mode', spy)

        with absurd_task_context(FakeAsyncTaskContext()):
            await agent.run('hi')
        await agent.run('hi')

        # `pydantic-ai-absurd` applies the configured mode to every run.
        assert recorded == ['parallel_ordered_events', 'parallel_ordered_events']


class TestRepeatedStepNames:
    async def test_two_runs_in_one_task_disambiguate_by_encounter_order(self) -> None:
        # A single task handler that runs the agent twice: the second run's model step reuses the
        # same step name, so Absurd's encounter-order counter records it under a `#2` suffix.
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('hi')
            second = await agent.run('hi again')

        assert first.output == second.output == 'ok'
        assert 'a__model.request' in ctx.stored
        assert 'a__model.request#2' in ctx.stored

        replay = ctx.replay()
        with absurd_task_context(replay):
            await agent.run('hi')
            await agent.run('hi again')

        assert counter['calls'] == 2
        assert replay.invoked == []

    async def test_same_tool_called_twice_in_one_response(self) -> None:
        calls: list[int] = []
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def charge(amount: int) -> str:
            calls.append(amount)
            return f'charged {amount}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='charge', args={'amount': 1}, tool_call_id='c1'),
                    ToolCallPart(tool_name='charge', args={'amount': 2}, tool_call_id='c2'),
                ]
            )

        agent = Agent(FunctionModel(model_fn), name='pay', toolsets=[toolset], capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('charge both')

        assert result.output == 'done'
        assert calls == [1, 2]
        step = 'pay__function_toolset__tools.call_tool:charge'
        assert step in ctx.stored
        assert f'{step}#2' in ctx.stored


class TestParallelOrderedEventsDeterminism:
    async def test_name_assignment_follows_scheduling_order_not_completion(self) -> None:
        # The adversarial case behind excluding `'parallel'` but keeping `'parallel_ordered_events'`:
        # two concurrent calls of the SAME tool, where the first-scheduled call completes LAST.
        # Absurd assigns the `#1`/`#2` checkpoint slot at `ctx.step(...)` entry, which happens before
        # the tool body runs, so assignment follows call-scheduling order (the model's tool-call
        # order), not completion order. If it followed completion order the slots -- and the cached
        # results served on replay -- would swap.
        toolset = FunctionToolset(id='tools')

        second_done = anyio.Event()

        @toolset.tool_plain
        async def record(marker: str) -> str:
            if marker == 'first':
                await second_done.wait()
            else:
                second_done.set()
            return marker

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            answered = any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts)
            if answered:
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    # The first-scheduled call waits for the second, so it completes last.
                    ToolCallPart(tool_name='record', args={'marker': 'first'}, tool_call_id='r1'),
                    ToolCallPart(tool_name='record', args={'marker': 'second'}, tool_call_id='r2'),
                ]
            )

        agent = Agent(
            FunctionModel(model_fn),
            name='par',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')],
        )

        step = 'par__function_toolset__tools.call_tool:record'

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            first = await agent.run('go')

        assert first.output == 'done'
        # `#1` is the first-scheduled call ('first'), even though it completed last.
        assert ctx.stored[step] == 'first'
        assert ctx.stored[f'{step}#2'] == 'second'

        # On replay the slots must map the same way: results are served from the checkpoint, not
        # swapped, regardless of which call finishes first.
        replay = ctx.replay()
        with absurd_task_context(replay):
            second = await agent.run('go')

        assert second.output == 'done'
        assert replay.invoked == []


class TestEventStreamHandler:
    async def test_handler_events_are_checkpointed(self) -> None:
        events: list[AgentStreamEvent] = []

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events.append(event)

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='greet', json_args='{}')}
            else:
                yield 'done'

        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def greet() -> str:
            return 'hello'

        agent = Agent(
            FunctionModel(stream_function=stream_fn, model_name='fn'),
            name='ev',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(event_stream_handler=handler)],
        )

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'done'
        assert any(isinstance(event, FunctionToolCallEvent) for event in events)
        assert 'ev__event_stream_handler' in ctx.stored


class TestCancelSuspendedResponse:
    async def test_cancel_suspended_response_is_checkpointed(self) -> None:
        # Drive the cancel through a real run: the model first returns a `'suspended'` response, so
        # the agent re-issues it as a continuation; the continuation request then fails, and the
        # graph tears down the suspended job via `cancel_suspended_response`, which Absurd checkpoints.
        cancelled: list[ModelResponse] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response)

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(m, ModelResponse) and m.state == 'suspended' for m in messages):
                return ModelResponse(parts=[TextPart(content='partial')], state='suspended')
            raise RuntimeError('continuation failed')

        agent = Agent(CancellableModel(fn, model_name='fn'), name='a', capabilities=[AbsurdDurability()])

        ctx = FakeAsyncTaskContext()
        with absurd_task_context(ctx):
            with pytest.raises(RuntimeError, match='continuation failed'):
                await agent.run('hi')

        assert len(cancelled) == 1
        assert cancelled[0].state == 'suspended'
        assert 'a__model.cancel_suspended_response' in ctx.stored


class TestCheckpointFormat:
    async def test_hand_written_model_request_payload_replays(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='gold', capabilities=[AbsurdDurability()])

        # A hand-authored checkpoint payload pins the persistence format for a model request.
        payload: JsonValue = {
            'parts': [{'content': 'golden-response', 'part_kind': 'text'}],
            'model_name': 'fn',
            'kind': 'response',
        }
        ctx = FakeAsyncTaskContext(store={'gold__model.request': payload})
        with absurd_task_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'golden-response'
        assert counter['calls'] == 0

    async def test_hand_written_stream_payload_replays(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='gold', capabilities=[AbsurdDurability()])

        # A hand-authored stream checkpoint pins the `{response, events}` payload shape.
        payload: JsonValue = {
            'response': {
                'parts': [{'content': 'golden-stream', 'part_kind': 'text'}],
                'model_name': 'fn',
                'kind': 'response',
            },
            'events': [
                {'index': 0, 'part': {'content': 'golden-stream', 'part_kind': 'text'}, 'event_kind': 'part_start'}
            ],
        }
        ctx = FakeAsyncTaskContext(store={'gold__model.request_stream': payload})
        with absurd_task_context(ctx):
            async with agent.run_stream('hi') as result:
                out = await result.get_output()

        assert out == 'golden-stream'
        assert counter['calls'] == 0
