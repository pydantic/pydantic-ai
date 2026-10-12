"""`AbsurdDurability` against a real Absurd schema on PostgreSQL.

Each test enters a real task context (`_task.running_task_context`), and a replay fails the run
and re-claims the task (`_task.reenter_running_task`), so step naming, encounter-order
disambiguation and checkpoint storage are Absurd's own.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

import anyio
import pytest

pytest.importorskip('absurd_sdk')
pytest.importorskip('fastmcp')

from absurd_sdk import (
    AsyncAbsurd,
    JsonValue,
    TaskContext,
    _current_task_context,  # pyright: ignore[reportPrivateUsage]
)
from fastmcp import FastMCP
from inline_snapshot import snapshot
from pydantic import TypeAdapter

from pydantic_ai import Agent, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.durable_exec._toolset import CallToolResult
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import (
    AgentStreamEvent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelRequest,
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
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, ExternalToolset, FunctionToolset
from pydantic_ai_harness.absurd import AbsurdDurability
from pydantic_ai_harness.absurd._operation_backend import (
    _CONTROL_FLOW_KINDS,  # pyright: ignore[reportPrivateUsage]
    _RAW_RESULT_KINDS,  # pyright: ignore[reportPrivateUsage]
)

from ._task import checkpoints, reenter_running_task, running_task_context


def _make_model(counter: dict[str, int] | None = None, text: str = 'ok') -> FunctionModel:
    tally = counter if counter is not None else {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tally['calls'] += 1
        return ModelResponse(parts=[TextPart(content=text)])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        tally['calls'] += 1
        yield text

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


def _tool_calling_model(*calls: ToolCallPart, counter: dict[str, int] | None = None) -> FunctionModel:
    """Makes `calls` in one response, then answers `'done'` once they have returned or asked to retry."""
    tally = counter if counter is not None else {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tally['calls'] += 1
        if any(isinstance(p, ToolReturnPart | RetryPromptPart) for m in messages for p in m.parts):
            return ModelResponse(parts=[TextPart(content='done')])
        return ModelResponse(parts=list(calls))

    return FunctionModel(fn, model_name='fn')


def _late_toolset(calls: dict[str, int]) -> FunctionToolset[object]:
    toolset = FunctionToolset[object](id='late')

    @toolset.tool_plain
    def late() -> str:
        calls['calls'] += 1
        return 'late result'

    return toolset


def _dynamic_toolset(tool_calls: dict[str, int], *, id: str | None) -> DynamicToolset[object]:
    def build(ctx: RunContext[object]) -> FunctionToolset[object]:
        inner: FunctionToolset[object] = FunctionToolset(id='inner')

        @inner.tool_plain
        def greet(name: str) -> str:
            tool_calls['calls'] += 1
            return f'hi {name}'

        return inner

    return DynamicToolset(build, id=id)


_RUNTIME_TOOLSET_ERROR = 'cannot be added at runtime with Absurd'


class TestDurability:
    async def test_requires_name(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(_make_model(), capabilities=[AbsurdDurability()])

    async def test_requires_model(self) -> None:
        with pytest.raises(UserError, match='needs to have a `model`'):
            Agent(name='a', capabilities=[AbsurdDurability()])

    async def test_reserved_default_model_id_raises(self) -> None:
        with pytest.raises(UserError, match="'default' is reserved"):
            Agent(_make_model(), name='a', capabilities=[AbsurdDurability(models={'default': _make_model()})])

    @pytest.mark.parametrize('kind', ['function', 'mcp', 'dynamic'])
    async def test_leaf_toolset_without_id_raises(self, kind: str) -> None:
        toolsets: dict[str, AbstractToolset[object]] = {
            'function': FunctionToolset[object](),
            'mcp': MCPToolset[object](FastMCP(name='calc')),
            'dynamic': _dynamic_toolset({'calls': 0}, id=None),
        }
        toolset = toolsets[kind]
        with pytest.raises(UserError, match='need to have a unique `id`'):
            Agent(_make_model(), name='a', toolsets=[toolset], capabilities=[AbsurdDurability()])

    async def test_same_toolset_instance_in_two_places_is_wrapped_once(self, absurd: AsyncAbsurd) -> None:
        # Both mounts checkpoint through the one wrapper, so the two calls take the same step name in
        # encounter order.
        toolset = FunctionToolset[object](id='shared')

        @toolset.tool_plain
        def echo(value: str) -> str:
            return value

        agent = Agent(
            _tool_calling_model(ToolCallPart('a_echo', {'value': 'a'}), ToolCallPart('b_echo', {'value': 'b'})),
            name='a',
            toolsets=[toolset.prefixed('a'), toolset.prefixed('b')],
            capabilities=[AbsurdDurability()],
        )
        async with running_task_context(absurd) as ctx:
            await agent.run('hi')

        stored = await checkpoints(absurd, ctx.task_id)
        step = 'a__function_toolset__shared.call_tool:echo'
        assert {k: v for k, v in stored.items() if 'call_tool' in k} == {step: 'a', f'{step}#2': 'b'}

    async def test_duplicate_toolset_id_raises(self) -> None:
        toolsets = [FunctionToolset[object](id='tools'), FunctionToolset[object](id='tools')]
        with pytest.raises(UserError, match='same `id`'):
            Agent(_make_model(), name='a', toolsets=toolsets, capabilities=[AbsurdDurability()])

    async def test_name_from_capability(self) -> None:
        agent = Agent(_make_model(), capabilities=[AbsurdDurability(name='custom')])
        bound = AbsurdDurability.from_agent(agent)
        assert bound is not None
        assert bound.name == 'custom'

    async def test_from_agent_without_capability_returns_none(self) -> None:
        assert AbsurdDurability.from_agent(Agent(_make_model(), name='a')) is None

    async def test_a_second_engine_is_refused(self) -> None:
        with pytest.raises(UserError, match='can have only one durable execution engine'):
            Agent(_make_model(), name='a', capabilities=[AbsurdDurability(), AbsurdDurability()])

    async def test_run_outside_task_is_transparent(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[AbsurdDurability()])
        result = await agent.run('hi')
        assert result.output == 'ok'
        assert counter['calls'] == 1

    async def test_replay_serves_cached_model_response(self, absurd: AsyncAbsurd) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='crash', capabilities=[AbsurdDurability()])

        async with running_task_context(absurd, 'crash') as ctx:
            first = await agent.run('hi')
        async with reenter_running_task(absurd, ctx.task_id):
            replayed = await agent.run('hi')

        assert counter['calls'] == 1
        assert replayed.output == first.output == 'ok'

    async def test_replay_does_not_rerun_function_tool(self, absurd: AsyncAbsurd) -> None:
        tool_calls = {'calls': 0}
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:
            tool_calls['calls'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_calling_model(ToolCallPart('charge_card', {'amount': 42})),
            name='billing',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd, 'billing') as ctx:
            first = await agent.run('charge it')
        # The raw tool return value is what is stored.
        stored = await checkpoints(absurd, ctx.task_id)
        assert stored['billing__function_toolset__tools.call_tool:charge_card'] == 'charged 42'
        async with reenter_running_task(absurd, ctx.task_id):
            replayed = await agent.run('charge it')

        assert tool_calls['calls'] == 1
        assert replayed.output == first.output == 'done'

    async def test_registered_model_selected_per_run(self, absurd: AsyncAbsurd) -> None:
        # The selected model checkpoints under its own id-scoped step name, and a replay serves it.
        primary = {'calls': 0}
        cheap = {'calls': 0}
        agent = Agent(
            _make_model(primary, text='primary'),
            name='a',
            capabilities=[AbsurdDurability(models={'cheap': _make_model(cheap, text='cheap')})],
        )

        async with running_task_context(absurd) as ctx:
            default_result = await agent.run('hi')
            cheap_result = await agent.run('hi', model='cheap')
        async with reenter_running_task(absurd, ctx.task_id):
            await agent.run('hi')
            replayed = await agent.run('hi', model='cheap')

        assert default_result.output == 'primary'
        assert cheap_result.output == replayed.output == 'cheap'
        assert primary['calls'] == cheap['calls'] == 1
        assert list(await checkpoints(absurd, ctx.task_id)) == ['a__model.request', 'a__model.request.cheap']

    @pytest.mark.parametrize('kind', ['function', 'dynamic'])
    async def test_runtime_toolset_rejected(self, absurd: AsyncAbsurd, kind: str) -> None:
        calls = {'calls': 0}
        late = _late_toolset(calls) if kind == 'function' else _dynamic_toolset(calls, id='late')
        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('hi', toolsets=[late])
        assert calls == {'calls': 0}

    async def test_override_toolsets_rejected_inside_task(self, absurd: AsyncAbsurd) -> None:
        calls = {'calls': 0}
        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('late')), name='a', capabilities=[AbsurdDurability()]
        )
        async with running_task_context(absurd):
            with agent.override(toolsets=[_late_toolset(calls)]):
                with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                    await agent.run('hi')
        assert calls['calls'] == 0

    async def test_override_toolsets_respected_outside_task(self) -> None:
        calls = {'calls': 0}
        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('late')), name='a', capabilities=[AbsurdDurability()]
        )
        with agent.override(toolsets=[_late_toolset(calls)]):
            result = await agent.run('hi')
        assert result.output == 'done'
        assert calls['calls'] == 1

    async def test_override_tools_rejected_inside_task(self, absurd: AsyncAbsurd) -> None:
        calls = {'calls': 0}

        def late() -> str:  # pragma: no cover - rejected before it can run
            calls['calls'] += 1
            return 'late result'

        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('late')), name='a', capabilities=[AbsurdDurability()]
        )
        async with running_task_context(absurd):
            with agent.override(tools=[late]):
                with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                    await agent.run('hi')
        assert calls['calls'] == 0

    async def test_override_tools_respected_outside_task(self) -> None:
        # The overriding toolset shares the agent's own toolset id, and must not be swapped for its wrapper.
        calls = {'calls': 0}

        def late() -> str:
            calls['calls'] += 1
            return 'late result'

        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('late')), name='a', capabilities=[AbsurdDurability()]
        )
        with agent.override(tools=[late]):
            result = await agent.run('hi')
        assert result.output == 'done'
        assert calls['calls'] == 1

    async def test_runtime_toolset_still_rejected_alongside_capability_toolset(self, absurd: AsyncAbsurd) -> None:
        # Skipping capability-owned wrappers must not let a genuine runtime toolset through.
        owned = FunctionToolset[object](id='owned')

        class DemoCapability(AbstractCapability[object]):
            def get_toolset(self) -> FunctionToolset[object]:
                return owned

        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[DemoCapability(), AbsurdDurability()])
        async with running_task_context(absurd):
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('hi', toolsets=[_late_toolset({'calls': 0})])

    async def test_capability_owned_toolset_is_durable(self, absurd: AsyncAbsurd) -> None:
        tool_calls = {'calls': 0}
        toolset = FunctionToolset[object](id='owned')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:
            tool_calls['calls'] += 1
            return f'charged {amount}'

        class DemoCapability(AbstractCapability[object]):
            def get_toolset(self) -> FunctionToolset[object]:
                return toolset

        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('charge_card', {'amount': 5})),
            name='owner',
            capabilities=[DemoCapability(), AbsurdDurability()],
        )

        async with running_task_context(absurd, 'owner') as ctx:
            first = await agent.run('charge it')
        async with reenter_running_task(absurd, ctx.task_id):
            replayed = await agent.run('charge it')

        assert tool_calls['calls'] == 1
        assert replayed.output == first.output == 'done'

    async def test_runtime_external_toolset_allowed(self, absurd: AsyncAbsurd) -> None:
        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            result = await agent.run('hi', toolsets=[ExternalToolset[object](tool_defs=[])])
        assert result.output == 'ok'

    async def test_construction_external_toolset_passes_through_unwrapped(self) -> None:
        external = ExternalToolset[object](tool_defs=[])
        agent = Agent(_make_model(), name='a', toolsets=[external], capabilities=[AbsurdDurability()])
        assert any(t is external for t in agent.toolsets)

    async def test_event_stream_handler_receives_events(self, absurd: AsyncAbsurd) -> None:
        events: list[AgentStreamEvent] = []

        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                events.append(event)

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='greet', json_args='{}')}
            else:
                yield 'done'

        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def greet() -> str:
            return 'hello'

        agent = Agent(
            FunctionModel(stream_function=stream_fn, model_name='fn'),
            name='a',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(event_stream_handler=handler)],
        )

        async with running_task_context(absurd) as ctx:
            result = await agent.run('hi')

        assert result.output == 'done'
        assert any(isinstance(e, PartStartEvent | PartDeltaEvent) for e in events)
        assert any(isinstance(e, FunctionToolCallEvent) for e in events)
        assert 'a__event_stream_handler' in await checkpoints(absurd, ctx.task_id)

    async def test_run_stream_events_inside_task(self, absurd: AsyncAbsurd) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd) as ctx:
            async with agent.run_stream_events('hi') as stream:
                events = [event async for event in stream]
        async with reenter_running_task(absurd, ctx.task_id):
            async with agent.run_stream_events('hi') as stream:
                replayed = [event async for event in stream]
        assert any(isinstance(e, PartStartEvent) for e in events)
        assert replayed == events
        assert counter['calls'] == 1

    async def test_run_stream_inside_task_replays_buffered_stream(self, absurd: AsyncAbsurd) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd) as ctx:
            async with agent.run_stream('hi') as result:
                assert await result.get_output() == 'ok'
        async with reenter_running_task(absurd, ctx.task_id):
            async with agent.run_stream('hi') as result:
                assert await result.get_output() == 'ok'
        assert counter['calls'] == 1

    async def test_iter_inside_task(self, absurd: AsyncAbsurd) -> None:
        agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd) as ctx:
            async with agent.iter('hi') as run:
                async for _ in run:
                    pass
        assert run.result is not None
        assert run.result.output == 'ok'
        assert list(await checkpoints(absurd, ctx.task_id)) == ['a__model.request']

    async def test_cancel_suspended_response_is_checkpointed(self, absurd: AsyncAbsurd) -> None:
        # The model returns a `'suspended'` response, the continuation fails, and the graph tears the
        # suspended job down via `cancel_suspended_response`.
        cancelled: list[ModelResponse] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response)

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(isinstance(m, ModelResponse) and m.state == 'suspended' for m in messages):
                return ModelResponse(parts=[TextPart(content='partial')], state='suspended')
            raise RuntimeError('continuation failed')

        agent = Agent(CancellableModel(fn, model_name='fn'), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd) as ctx:
            with pytest.raises(RuntimeError, match='continuation failed'):
                await agent.run('hi')

        assert [response.state for response in cancelled] == ['suspended']
        assert 'a__model.cancel_suspended_response' in await checkpoints(absurd, ctx.task_id)

    async def test_sync_context_raises(self) -> None:
        agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
        sync_ctx: Any = object.__new__(TaskContext)
        token = _current_task_context.set(sync_ctx)
        try:
            with pytest.raises(UserError, match='requires an async Absurd task context'):
                await agent.run('hi')
        finally:
            _current_task_context.reset(token)


class TestCapabilityOperation:
    async def test_operation_is_checkpointed_and_replayed(self, absurd: AsyncAbsurd) -> None:
        calls: list[str] = []

        class Recorder(AbstractCapability[object]):
            id = 'recorder'

            async def before_run(self, ctx: RunContext[object]) -> None:
                await self.record(ctx, 'started')

            @durable_operation('record')
            async def record(self, ctx: RunContext[object], value: str) -> None:
                del ctx
                calls.append(value)

        agent = Agent(_make_model(), name='cap', capabilities=[Recorder(), AbsurdDurability()])

        async with running_task_context(absurd) as ctx:
            await agent.run('hi')
        assert 'cap__capability__recorder.record' in await checkpoints(absurd, ctx.task_id)
        async with reenter_running_task(absurd, ctx.task_id):
            await agent.run('hi')

        assert calls == ['started']


class TestToolResults:
    def test_every_core_result_kind_is_classified(self) -> None:
        # A result kind core adds must be sorted into checkpointed or not before it can reach a task.
        kinds = set(TypeAdapter(CallToolResult).json_schema()['discriminator']['mapping'])
        assert kinds == _RAW_RESULT_KINDS | _CONTROL_FLOW_KINDS

    async def test_model_retry_is_not_checkpointed_and_the_tool_reruns_on_replay(self, absurd: AsyncAbsurd) -> None:
        model_calls = {'calls': 0}
        tool_calls = {'calls': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            tool_calls['calls'] += 1
            raise ModelRetry('nope, try again')

        model = _tool_calling_model(ToolCallPart('flaky'), counter=model_calls)
        agent = Agent(model, name='retry', toolsets=[toolset], capabilities=[AbsurdDurability()])

        async with running_task_context(absurd) as ctx:
            first = await agent.run('go')
        # The `ModelRetry` propagates out of the step, so nothing is stored.
        assert list(await checkpoints(absurd, ctx.task_id)) == ['retry__model.request', 'retry__model.request#2']
        async with reenter_running_task(absurd, ctx.task_id):
            second = await agent.run('go')

        # Both model responses come from their checkpoints; only the tool runs again.
        assert first.output == second.output == 'done'
        assert model_calls['calls'] == tool_calls['calls'] == 2

    async def test_tool_return_object_round_trips_through_replay(self, absurd: AsyncAbsurd) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def lookup() -> ToolReturn:
            return ToolReturn(return_value='value', content='extra context', metadata={'source': 'db'})

        agent = Agent(
            _tool_calling_model(ToolCallPart('lookup')),
            name='tr',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd) as ctx:
            first = await agent.run('go')
        # A `ToolReturn` has no raw form, so it is stored under a reserved key.
        stored = await checkpoints(absurd, ctx.task_id)
        assert stored['tr__function_toolset__tools.call_tool:lookup'] == snapshot(
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
        async with reenter_running_task(absurd, ctx.task_id):
            second = await agent.run('go')

        def returned(messages: list[ModelMessage]) -> list[tuple[object, object]]:
            return [(p.content, p.metadata) for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]

        assert returned(second.all_messages()) == returned(first.all_messages()) == [('value', {'source': 'db'})]

    @pytest.mark.parametrize(
        'value',
        [
            {'kind': 'tool-return', 'rows': 3},
            {'__pydantic_ai_harness_absurd_tool_result__': 'hello'},
            {
                '__pydantic_ai_harness_absurd_tool_result__': {
                    'kind': 'tool_return',
                    'result': {'kind': 'tool-return', 'return_value': 'x', 'content': 'y'},
                }
            },
        ],
        ids=['tool-return-kind', 'reserved-key', 'envelope-shaped'],
    )
    async def test_raw_dict_that_looks_encoded_round_trips(self, absurd: AsyncAbsurd, value: dict[str, object]) -> None:
        calls = {'calls': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def query() -> dict[str, object]:
            calls['calls'] += 1
            return value

        agent = Agent(
            _tool_calling_model(ToolCallPart('query')),
            name='raw',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd) as ctx:
            first = await agent.run('go')
        async with reenter_running_task(absurd, ctx.task_id):
            second = await agent.run('go')

        def returned(messages: list[ModelMessage]) -> list[object]:
            return [p.content for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]

        assert returned(second.all_messages()) == returned(first.all_messages()) == [value]
        assert calls['calls'] == 1


class TestCrashMidRun:
    async def test_model_step_served_from_checkpoint_while_failed_tool_reruns(self, absurd: AsyncAbsurd) -> None:
        # The model step completes and is checkpointed, then a tool raises a real (non-`ModelRetry`)
        # error that fails the attempt. On retry the model step is served from its checkpoint while
        # the tool re-runs.
        model_calls = {'calls': 0}
        tool_attempts = {'calls': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            tool_attempts['calls'] += 1
            if tool_attempts['calls'] == 1:
                raise RuntimeError('worker died mid-tool')
            return 'recovered'

        model = _tool_calling_model(ToolCallPart('flaky'), counter=model_calls)
        agent = Agent(model, name='crash', toolsets=[toolset], capabilities=[AbsurdDurability()])

        async with running_task_context(absurd) as ctx:
            with pytest.raises(RuntimeError, match='worker died mid-tool'):
                await agent.run('go')
        assert list(await checkpoints(absurd, ctx.task_id)) == ['crash__model.request']
        async with reenter_running_task(absurd, ctx.task_id):
            result = await agent.run('go')

        assert result.output == 'done'
        assert model_calls['calls'] == tool_attempts['calls'] == 2
        assert list(await checkpoints(absurd, ctx.task_id)) == [
            'crash__model.request',
            'crash__function_toolset__tools.call_tool:flaky',
            'crash__model.request#2',
        ]


class TestStepNames:
    async def test_string_default_model_gets_unsuffixed_step_name(self, absurd: AsyncAbsurd) -> None:
        agent = Agent('test', name='strdef', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd) as ctx:
            await agent.run('hi')
        assert list(await checkpoints(absurd, ctx.task_id)) == ['strdef__model.request']

    async def test_two_runs_in_one_task_disambiguate_by_encounter_order(self, absurd: AsyncAbsurd) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[AbsurdDurability()])

        async with running_task_context(absurd) as ctx:
            await agent.run('hi')
            await agent.run('hi again')
        assert list(await checkpoints(absurd, ctx.task_id)) == ['a__model.request', 'a__model.request#2']
        async with reenter_running_task(absurd, ctx.task_id):
            await agent.run('hi')
            await agent.run('hi again')

        assert counter['calls'] == 2


class TestParallelExecutionMode:
    async def test_step_slots_follow_scheduling_order_not_completion(self, absurd: AsyncAbsurd) -> None:
        # Two concurrent calls of the same tool, where the first-scheduled call completes last.
        # Absurd assigns the `#1`/`#2` slot when the step begins (`begin_step`), before the tool body
        # runs, so slots follow the model's tool-call order and a replay serves each call its own
        # result.
        toolset = FunctionToolset(id='tools')
        second_done = anyio.Event()

        @toolset.tool_plain
        async def record(marker: str) -> str:
            if marker == 'first':
                await second_done.wait()
            else:
                second_done.set()
            return marker

        agent = Agent(
            _tool_calling_model(
                ToolCallPart('record', {'marker': 'first'}, 'r1'), ToolCallPart('record', {'marker': 'second'}, 'r2')
            ),
            name='par',
            toolsets=[toolset],
            capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')],
        )
        step = 'par__function_toolset__tools.call_tool:record'

        async with running_task_context(absurd) as ctx:
            first = await agent.run('go')
        stored = await checkpoints(absurd, ctx.task_id)
        assert (stored[step], stored[f'{step}#2']) == ('first', 'second')
        async with reenter_running_task(absurd, ctx.task_id):
            second = await agent.run('go')

        def returned(messages: list[ModelMessage]) -> list[tuple[str, object]]:
            return [(p.tool_call_id, p.content) for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]

        assert returned(second.all_messages()) == returned(first.all_messages()) == [('r1', 'first'), ('r2', 'second')]


class TestMcpSessions:
    async def test_replay_opens_no_mcp_session(self, absurd: AsyncAbsurd) -> None:
        # The wrapper does not enter the server itself, so a replay served entirely from checkpoints
        # never connects to it.
        from tests.durable_exec.counting_mcp import counting_mcp_server

        server, counts = counting_mcp_server(instructions='Echo things.')
        agent = Agent(
            _tool_calling_model(ToolCallPart('echo', {'text': 'hi'})),
            name='echo',
            toolsets=[MCPToolset[object](server, id='echo', include_instructions=True)],
            capabilities=[AbsurdDurability()],
        )
        async with running_task_context(absurd) as ctx:
            await agent.run('go')
        after_first_run = dict(counts)
        async with reenter_running_task(absurd, ctx.task_id):
            result = await agent.run('go')

        assert result.output == 'done'
        assert counts == after_first_run
        # The replayed run gets the server's instructions from their checkpoint.
        requests = [m for m in result.all_messages() if isinstance(m, ModelRequest)]
        assert requests[0].instructions is not None and 'Echo things.' in requests[0].instructions


class TestCheckpointFormat:
    async def test_stream_checkpoint_replays(self, absurd: AsyncAbsurd) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='gold', capabilities=[AbsurdDurability()])
        # Pins the `{response, events}` payload shape of a stream checkpoint.
        payload: JsonValue = {
            'response': {
                'parts': [{'content': 'from-checkpoint', 'part_kind': 'text'}],
                'model_name': 'fn',
                'kind': 'response',
            },
            'events': [
                {'index': 0, 'part': {'content': 'from-checkpoint', 'part_kind': 'text'}, 'event_kind': 'part_start'}
            ],
        }

        async def write() -> JsonValue:
            return payload

        async with running_task_context(absurd) as ctx:
            await ctx.step('gold__model.request_stream', write)
        async with reenter_running_task(absurd, ctx.task_id):
            async with agent.run_stream('hi') as result:
                assert await result.get_output() == 'from-checkpoint'

        assert counter['calls'] == 0


class TestDynamicToolset:
    """Function and MCP toolsets are checkpointed; a construction-time `DynamicToolset` runs as-is."""

    async def test_runs_uncheckpointed_inside_a_task(self, absurd: AsyncAbsurd) -> None:
        tool_calls = {'calls': 0}
        agent = Agent(
            _tool_calling_model(ToolCallPart('greet', {'name': 'ada'})),
            name='d',
            toolsets=[_dynamic_toolset(tool_calls, id='dyn')],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd) as ctx:
            first = await agent.run('greet ada')
        assert list(await checkpoints(absurd, ctx.task_id)) == ['d__model.request', 'd__model.request#2']
        async with reenter_running_task(absurd, ctx.task_id):
            second = await agent.run('greet ada')

        # The model responses replay from their checkpoints; the dynamic tool runs again.
        assert first.output == second.output == 'done'
        assert tool_calls['calls'] == 2

    async def test_toolset_decorated_after_construction_rejected_inside_task(self, absurd: AsyncAbsurd) -> None:
        agent = Agent(_make_model(), name='decorated', capabilities=[AbsurdDurability()])

        @agent.toolset(id='decorated-tools')
        def build(ctx: RunContext[object]) -> FunctionToolset[object]:  # pragma: no cover
            return FunctionToolset[object](id='inner')

        async with running_task_context(absurd):
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('hi')


class TestCodeMode:
    async def test_tool_call_inside_run_code_is_checkpointed(self, absurd: AsyncAbsurd) -> None:
        pytest.importorskip('pydantic_monty')
        from pydantic_ai_harness import CodeMode

        calls = {'calls': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def search(query: str) -> str:
            calls['calls'] += 1
            return f'results for {query}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for part in (p for m in messages for p in m.parts):
                if isinstance(part, ToolReturnPart) and part.tool_name == 'run_code':
                    return ModelResponse(parts=[TextPart(content=f'done: {part.content}')])
            code = "result = await search(query='x')\nresult"
            return ModelResponse(parts=[ToolCallPart('run_code', {'code': code}, 'tc1')])

        agent = Agent(
            FunctionModel(model_fn), name='composed', toolsets=[toolset], capabilities=[CodeMode(), AbsurdDurability()]
        )

        async with running_task_context(absurd) as ctx:
            first = await agent.run('go')
        stored = await checkpoints(absurd, ctx.task_id)
        assert stored['composed__function_toolset__tools.call_tool:search'] == 'results for x'
        async with reenter_running_task(absurd, ctx.task_id):
            second = await agent.run('go')

        # The `run_code` body re-runs on replay, but its `search` call is served from the checkpoint.
        assert first.output == second.output == 'done: results for x'
        assert calls['calls'] == 1
