"""`AbsurdDurability` against a real Absurd schema on PostgreSQL.

Each test enters a real task context (`_task.running_task_context`), and a replay fails the run
and re-claims the task (`_task.reenter_running_task`), so step naming, encounter-order
disambiguation and checkpoint storage are Absurd's own.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from contextlib import AbstractContextManager
from typing import Any

import anyio
import pytest

pytest.importorskip('absurd_sdk')
pytest.importorskip('fastmcp')

from absurd_sdk import (
    AsyncAbsurd,
    AsyncTaskContext,
    JsonValue,
    TaskContext,
    _current_task_context,  # pyright: ignore[reportPrivateUsage]
)
from fastmcp import FastMCP
from inline_snapshot import snapshot
from pydantic import TypeAdapter

from pydantic_ai import Agent, ToolReturn
from pydantic_ai.agent import ParallelExecutionMode
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import ModelRetry, UserError
from pydantic_ai.mcp import MCPToolset
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
from pydantic_ai.toolsets import DynamicToolset, ExternalToolset, FunctionToolset
from pydantic_ai_harness.absurd import AbsurdDurability

from ._task import checkpoints, reenter_running_task, running_task_context


def _make_model(counter: dict[str, int] | None = None) -> FunctionModel:
    tally = counter if counter is not None else {'calls': 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tally['calls'] += 1
        return ModelResponse(parts=[TextPart(content='ok')])

    async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
        tally['calls'] += 1
        yield 'ok'

    return FunctionModel(fn, stream_function=stream_fn, model_name='fn')


def _tool_calling_model(tool_name: str, args: dict[str, Any] | None = None) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(isinstance(p, (ToolReturnPart, RetryPromptPart)) for m in messages for p in m.parts)
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args or {})])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(fn, model_name='fn')


def _late_toolset(calls: dict[str, int]) -> FunctionToolset[object]:
    toolset = FunctionToolset[object](id='late')

    @toolset.tool_plain
    def late() -> str:
        calls['calls'] += 1
        return 'late result'

    return toolset


def _calculator(calls: list[tuple[int, int]]) -> FastMCP[None]:
    server: FastMCP[None] = FastMCP(name='calc', instructions='Use the calculator.')

    @server.tool
    def add(a: int, b: int) -> int:
        calls.append((a, b))
        return a + b

    return server


_RUNTIME_TOOLSET_ERROR = 'cannot be added at runtime with Absurd'
_response_adapter: TypeAdapter[ModelResponse] = TypeAdapter(ModelResponse)


class TestDurability:
    async def test_requires_name(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(_make_model(), capabilities=[AbsurdDurability()])

    async def test_name_from_capability(self) -> None:
        agent = Agent(_make_model(), capabilities=[AbsurdDurability(name='custom')])
        bound = AbsurdDurability.from_agent(agent)
        assert bound is not None
        assert bound.name == 'custom'

    async def test_requires_model(self) -> None:
        with pytest.raises(UserError, match='needs to have a `model`'):
            Agent(name='a', capabilities=[AbsurdDurability()])

    async def test_reserved_default_model_id_raises(self) -> None:
        with pytest.raises(UserError, match="'default' is reserved"):
            Agent(_make_model(), name='a', capabilities=[AbsurdDurability(models={'default': _make_model()})])

    async def test_leaf_toolset_without_id_is_durable(self, absurd: AsyncAbsurd) -> None:
        tool_calls = {'calls': 0}
        toolset = FunctionToolset[object]()

        @toolset.tool_plain
        def charge_card(amount: int) -> str:
            tool_calls['calls'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_calling_model('charge_card', {'amount': 7}),
            name='idless',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd, 'idless') as ctx:
            first = await agent.run('charge it')
        async with reenter_running_task(absurd, ctx.task_id):
            replayed = await agent.run('charge it')

        assert tool_calls['calls'] == 1
        assert replayed.output == first.output == 'done'

    @pytest.mark.parametrize(
        ('toolset_id', 'step'),
        [('shared', 'a__function_toolset__shared.call_tool:echo'), (None, 'a__function_toolset.call_tool:echo')],
    )
    async def test_same_toolset_instance_in_two_places_is_wrapped_once(
        self, absurd: AsyncAbsurd, toolset_id: str | None, step: str
    ) -> None:
        # Both mounts checkpoint through the one wrapper, so the two calls take the same step name in
        # encounter order.
        toolset = FunctionToolset[object](id=toolset_id)

        @toolset.tool_plain
        def echo(value: str) -> str:
            return value

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    ToolCallPart('a_echo', {'value': 'a'}, 'c1'),
                    ToolCallPart('b_echo', {'value': 'b'}, 'c2'),
                ]
            )

        agent = Agent(
            FunctionModel(fn, model_name='fn'),
            name='a',
            toolsets=[toolset.prefixed('a'), toolset.prefixed('b')],
            capabilities=[AbsurdDurability()],
        )
        async with running_task_context(absurd) as ctx:
            await agent.run('hi')

        stored = await checkpoints(absurd, ctx.task_id)
        assert {k: v for k, v in stored.items() if 'call_tool' in k} == {step: 'a', f'{step}#2': 'b'}

    async def test_duplicate_toolset_id_raises(self) -> None:
        first = FunctionToolset[object](id='tools')

        # Never invoked, only the wrap check runs.
        @first.tool_plain
        def echo(value: str) -> str:  # pragma: no cover
            return value

        second = FunctionToolset[object](id='tools')

        # Never invoked, only the wrap check runs.
        @second.tool_plain
        def shout(value: str) -> str:  # pragma: no cover
            return value.upper()

        with pytest.raises(UserError, match='same `id`'):
            Agent(_make_model(), name='a', toolsets=[first, second], capabilities=[AbsurdDurability()])

    async def test_from_agent_without_capability_returns_none(self) -> None:
        assert AbsurdDurability.from_agent(Agent(_make_model(), name='a')) is None

    async def test_from_agent_multiple_raises(self) -> None:
        agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability(), AbsurdDurability()])
        with pytest.raises(UserError, match='at most one'):
            AbsurdDurability.from_agent(agent)

    async def test_run_outside_task_is_transparent(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[AbsurdDurability()])
        result = await agent.run('hi')
        assert result.output == 'ok'
        assert counter['calls'] == 1

    async def test_run_inside_task_completes(self, absurd: AsyncAbsurd) -> None:
        agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            result = await agent.run('hi')
        assert result.output == 'ok'

    async def test_run_inside_authored_task_is_durable(self, absurd: AsyncAbsurd) -> None:
        agent = Agent(_make_model(), name='analyst', capabilities=[AbsurdDurability()])

        async def analyse(params: JsonValue, ctx: AsyncTaskContext) -> JsonValue:
            assert isinstance(params, dict)
            prompt = params['prompt']
            assert isinstance(prompt, str)
            result = await agent.run(prompt)
            return {'output': result.output}

        absurd.register_task(name='analyse')(analyse)

        spawned = await absurd.spawn('analyse', {'prompt': 'go'})
        await absurd.work_batch(batch_size=1)
        result = await absurd.fetch_task_result(spawned['task_id'])
        assert result is not None and result.state == 'completed'
        assert result.result == {'output': 'ok'}

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
            _tool_calling_model('charge_card', {'amount': 42}),
            name='billing',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd, 'billing') as ctx:
            first = await agent.run('charge it')
        async with reenter_running_task(absurd, ctx.task_id):
            replayed = await agent.run('charge it')

        assert tool_calls['calls'] == 1
        assert replayed.output == first.output == 'done'

    async def test_registered_model_selected_per_run(self, absurd: AsyncAbsurd) -> None:
        # The selected model checkpoints under its own id-scoped step name, and a replay serves it.
        primary = {'calls': 0}
        cheap = {'calls': 0}

        def primary_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            primary['calls'] += 1
            return ModelResponse(parts=[TextPart(content='primary')])

        def cheap_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            cheap['calls'] += 1
            return ModelResponse(parts=[TextPart(content='cheap')])

        agent = Agent(
            FunctionModel(primary_fn, model_name='primary'),
            name='a',
            capabilities=[AbsurdDurability(models={'cheap': FunctionModel(cheap_fn, model_name='cheap')})],
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

    async def test_runtime_function_toolset_rejected(self, absurd: AsyncAbsurd) -> None:
        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('hi', toolsets=[_late_toolset({'calls': 0})])

    async def test_override_toolsets_rejected_inside_task(self, absurd: AsyncAbsurd) -> None:
        calls = {'calls': 0}
        agent: Agent[object, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            with agent.override(toolsets=[_late_toolset(calls)]):
                with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                    await agent.run('hi')
        assert calls['calls'] == 0

    async def test_override_toolsets_respected_outside_task(self) -> None:
        calls = {'calls': 0}
        agent: Agent[object, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[AbsurdDurability()])
        with agent.override(toolsets=[_late_toolset(calls)]):
            result = await agent.run('hi')
        assert result.output == 'done'
        assert calls['calls'] == 1

    async def test_override_tools_rejected_inside_task(self, absurd: AsyncAbsurd) -> None:
        calls = {'calls': 0}

        # Rejected before it can run.
        def late() -> str:  # pragma: no cover
            calls['calls'] += 1
            return 'late result'

        agent: Agent[object, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            with agent.override(tools=[late]):
                with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                    await agent.run('hi')
        assert calls['calls'] == 0

    async def test_override_tools_respected_outside_task(self) -> None:
        calls = {'calls': 0}

        def late() -> str:
            calls['calls'] += 1
            return 'late result'

        agent: Agent[object, str] = Agent(_tool_calling_model('late'), name='a', capabilities=[AbsurdDurability()])
        with agent.override(tools=[late]):
            result = await agent.run('hi')
        assert result.output == 'done'
        assert calls['calls'] == 1

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
            _tool_calling_model('charge_card', {'amount': 5}),
            name='owner',
            capabilities=[DemoCapability(), AbsurdDurability()],
        )

        async with running_task_context(absurd, 'owner') as ctx:
            first = await agent.run('charge it')
        async with reenter_running_task(absurd, ctx.task_id):
            replayed = await agent.run('charge it')

        assert tool_calls['calls'] == 1
        assert replayed.output == first.output == 'done'

    async def test_runtime_toolset_still_rejected_alongside_capability_toolset(self, absurd: AsyncAbsurd) -> None:
        owned = FunctionToolset[object](id='owned')

        # Never invoked.
        @owned.tool_plain
        def greet() -> str:  # pragma: no cover
            return 'hello'

        class DemoCapability(AbstractCapability[object]):
            def get_toolset(self) -> FunctionToolset[object]:
                return owned

        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[DemoCapability(), AbsurdDurability()])
        async with running_task_context(absurd):
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('hi', toolsets=[_late_toolset({'calls': 0})])

    async def test_runtime_external_toolset_allowed(self, absurd: AsyncAbsurd) -> None:
        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            result = await agent.run('hi', toolsets=[ExternalToolset[object](tool_defs=[])])
        assert result.output == 'ok'

    async def test_construction_external_toolset_passes_through_unwrapped(self) -> None:
        external = ExternalToolset[object](tool_defs=[])
        agent = Agent(_make_model(), name='a', toolsets=[external], capabilities=[AbsurdDurability()])
        assert any(t is external for t in agent.toolsets)

    async def test_mcp_tool_call_inside_task(self, absurd: AsyncAbsurd) -> None:
        agent = Agent(
            _tool_calling_model('add', {'a': 2, 'b': 3}),
            name='calc',
            toolsets=[MCPToolset[object](_calculator([]), id='calc')],
            capabilities=[AbsurdDurability()],
        )
        async with running_task_context(absurd) as ctx:
            result = await agent.run('add 2 and 3')
        assert result.output == 'done'
        assert (await checkpoints(absurd, ctx.task_id))['calc__mcp_server__calc.call_tool'] == 5

    async def test_mcp_get_instructions_inside_context_with_include(self, absurd: AsyncAbsurd) -> None:
        server = MCPToolset[object](_calculator([]), id='calc', include_instructions=True)
        agent = Agent(_make_model(), name='calc', toolsets=[server], capabilities=[AbsurdDurability()])
        async with running_task_context(absurd) as ctx:
            result = await agent.run('hi')
        assert 'Use the calculator.' in str(result.all_messages()[0])
        assert 'calc__mcp_server__calc.get_instructions' in await checkpoints(absurd, ctx.task_id)

    async def test_mcp_get_tools_without_cache(self, absurd: AsyncAbsurd) -> None:
        # Without the tool cache, each listing is its own step.
        server = MCPToolset[object](_calculator([]), id='calc', cache_tools=False)
        agent = Agent(
            _tool_calling_model('add', {'a': 4, 'b': 5}),
            name='calc',
            toolsets=[server],
            capabilities=[AbsurdDurability()],
        )
        async with running_task_context(absurd) as ctx:
            await agent.run('add')
        stored = await checkpoints(absurd, ctx.task_id)
        assert [name for name in stored if name.endswith('.get_tools') or '.get_tools#' in name] == [
            'calc__mcp_server__calc.get_tools',
            'calc__mcp_server__calc.get_tools#2',
        ]

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

    async def test_iter_inside_task(self, absurd: AsyncAbsurd) -> None:
        agent = Agent(_make_model(), name='a', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            async with agent.iter('hi') as run:
                async for _ in run:
                    pass
        assert run.result is not None
        assert run.result.output == 'ok'

    async def test_wrapper_written_stream_checkpoint_replays_under_capability(self, absurd: AsyncAbsurd) -> None:
        """A `request_stream` checkpoint written by the older `AbsurdAgent` wrapper (a bare
        `ModelResponse`) replays under the capability."""
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='legacy', capabilities=[AbsurdDurability()])
        legacy_payload = _response_adapter.dump_python(
            ModelResponse(parts=[TextPart(content='from-wrapper')]), mode='json'
        )

        async def write_legacy() -> JsonValue:
            return legacy_payload

        async with running_task_context(absurd, 'legacy') as ctx:
            await ctx.step('legacy__model.request_stream', write_legacy)
        async with reenter_running_task(absurd, ctx.task_id):
            async with agent.run_stream('hi') as result:
                assert await result.get_output() == 'from-wrapper'

        assert counter['calls'] == 0

    async def test_string_default_model_replays_wrapper_checkpoint(self, absurd: AsyncAbsurd) -> None:
        agent = Agent('test', name='strdef', capabilities=[AbsurdDurability()])
        legacy_payload = _response_adapter.dump_python(
            ModelResponse(parts=[TextPart(content='from-wrapper')]), mode='json'
        )

        async def write_legacy() -> JsonValue:
            return legacy_payload

        async with running_task_context(absurd, 'strdef') as ctx:
            await ctx.step('strdef__model.request', write_legacy)
        async with reenter_running_task(absurd, ctx.task_id):
            replayed = await agent.run('hi')

        assert replayed.output == 'from-wrapper'

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
    async def test_raw_return_value_is_stored(self, absurd: AsyncAbsurd) -> None:
        toolset = FunctionToolset(id='billing')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:
            return f'charged {amount}'

        agent = Agent(
            _tool_calling_model('charge_card', {'amount': 7}),
            name='pay',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )
        async with running_task_context(absurd) as ctx:
            await agent.run('charge it')

        # The raw tool return value is what is stored.
        stored = await checkpoints(absurd, ctx.task_id)
        assert stored['pay__function_toolset__billing.call_tool:charge_card'] == 'charged 7'

    async def test_model_retry_is_not_checkpointed_and_the_tool_reruns_on_replay(self, absurd: AsyncAbsurd) -> None:
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

        async with running_task_context(absurd) as ctx:
            first = await agent.run('go')
        # The `ModelRetry` propagates out of the step, so nothing is stored.
        assert list(await checkpoints(absurd, ctx.task_id)) == ['retry__model.request', 'retry__model.request#2']
        async with reenter_running_task(absurd, ctx.task_id):
            second = await agent.run('go')

        # Both model responses come from their checkpoints; only the tool runs again.
        assert first.output == second.output == 'done'
        assert calls == {'model': 2, 'tool': 2}

    async def test_tool_return_object_round_trips_through_replay(self, absurd: AsyncAbsurd) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def lookup() -> ToolReturn:
            return ToolReturn(return_value='value', content='extra context', metadata={'source': 'db'})

        agent = Agent(_tool_calling_model('lookup'), name='tr', toolsets=[toolset], capabilities=[AbsurdDurability()])

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
        [{'kind': 'tool-return', 'rows': 3}, {'__pydantic_ai_harness_absurd_tool_result__': 'hello'}],
        ids=['tool-return-kind', 'reserved-key'],
    )
    async def test_raw_dict_that_looks_encoded_round_trips(self, absurd: AsyncAbsurd, value: dict[str, object]) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def query() -> dict[str, object]:
            calls['n'] += 1
            return value

        agent = Agent(_tool_calling_model('query'), name='raw', toolsets=[toolset], capabilities=[AbsurdDurability()])

        async with running_task_context(absurd) as ctx:
            await agent.run('go')
        assert (await checkpoints(absurd, ctx.task_id))['raw__function_toolset__tools.call_tool:query'] == value
        async with reenter_running_task(absurd, ctx.task_id):
            result = await agent.run('go')

        returns = [p for m in result.all_messages() for p in m.parts if isinstance(p, ToolReturnPart)]
        assert [r.content for r in returns] == [value]
        assert calls['n'] == 1


class TestCrashMidRun:
    async def test_model_step_served_from_checkpoint_while_failed_tool_reruns(self, absurd: AsyncAbsurd) -> None:
        # The model step completes and is checkpointed, then a tool raises a real (non-`ModelRetry`)
        # error that fails the attempt. On retry the model step is served from its checkpoint while
        # the tool re-runs.
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
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(parts=[ToolCallPart(tool_name='flaky', args={})])

        agent = Agent(FunctionModel(model_fn), name='crash', toolsets=[toolset], capabilities=[AbsurdDurability()])

        async with running_task_context(absurd) as ctx:
            with pytest.raises(RuntimeError, match='worker died mid-tool'):
                await agent.run('go')
        assert list(await checkpoints(absurd, ctx.task_id)) == ['crash__model.request']
        async with reenter_running_task(absurd, ctx.task_id):
            result = await agent.run('go')

        assert result.output == 'done'
        assert model_calls['n'] == tool_attempts['n'] == 2
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

    async def test_id_less_function_toolset_drops_the_id_segment(self, absurd: AsyncAbsurd) -> None:
        toolset = FunctionToolset()

        @toolset.tool_plain
        def charge(amount: int) -> str:
            return f'charged {amount}'

        agent = Agent(
            _tool_calling_model('charge', {'amount': 3}),
            name='idless',
            toolsets=[toolset],
            capabilities=[AbsurdDurability()],
        )
        async with running_task_context(absurd) as ctx:
            await agent.run('go')
        assert (await checkpoints(absurd, ctx.task_id))['idless__function_toolset.call_tool:charge'] == 'charged 3'

    async def test_id_less_mcp_server_drops_the_id_segment(self, absurd: AsyncAbsurd) -> None:
        # An MCP toolset constructed without an `id` (for example an in-process server) is still
        # checkpointed, under step names without the `__<id>` segment.
        calls: list[tuple[int, int]] = []
        agent = Agent(
            _tool_calling_model('add', {'a': 2, 'b': 3}),
            name='calc',
            toolsets=[MCPToolset[object](_calculator(calls), include_instructions=True)],
            capabilities=[AbsurdDurability()],
        )
        async with running_task_context(absurd) as ctx:
            await agent.run('add 2 and 3')
        stored = await checkpoints(absurd, ctx.task_id)
        assert [name for name in stored if '__mcp_server' in name] == snapshot(
            [
                'calc__mcp_server.get_tools',
                'calc__mcp_server.get_instructions',
                'calc__mcp_server.call_tool',
                'calc__mcp_server.get_instructions#2',
            ]
        )
        assert stored['calc__mcp_server.call_tool'] == 5

        async with reenter_running_task(absurd, ctx.task_id):
            await agent.run('add 2 and 3')
        assert calls == [(2, 3)]

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

    async def test_same_tool_called_twice_in_one_response(self, absurd: AsyncAbsurd) -> None:
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def charge(amount: int) -> str:
            return f'charged {amount}'

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
                    ToolCallPart(tool_name='charge', args={'amount': 1}, tool_call_id='c1'),
                    ToolCallPart(tool_name='charge', args={'amount': 2}, tool_call_id='c2'),
                ]
            )

        agent = Agent(FunctionModel(model_fn), name='pay', toolsets=[toolset], capabilities=[AbsurdDurability()])
        async with running_task_context(absurd) as ctx:
            await agent.run('charge both')

        stored = await checkpoints(absurd, ctx.task_id)
        step = 'pay__function_toolset__tools.call_tool:charge'
        assert (stored[step], stored[f'{step}#2']) == ('charged 1', 'charged 2')


class TestParallelExecutionMode:
    async def test_mode_applied_inside_and_outside_a_task(
        self, absurd: AsyncAbsurd, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        agent = Agent(
            _make_model(), name='a', capabilities=[AbsurdDurability(parallel_execution_mode='parallel_ordered_events')]
        )
        recorded: list[ParallelExecutionMode] = []
        real = agent.parallel_tool_call_execution_mode

        def spy(mode: ParallelExecutionMode = 'parallel') -> AbstractContextManager[None]:
            recorded.append(mode)
            return real(mode)

        monkeypatch.setattr(agent, 'parallel_tool_call_execution_mode', spy)

        async with running_task_context(absurd):
            await agent.run('hi')
        await agent.run('hi')

        assert recorded == ['parallel_ordered_events', 'parallel_ordered_events']

    async def test_step_slots_follow_scheduling_order_not_completion(self, absurd: AsyncAbsurd) -> None:
        # Two concurrent calls of the same tool, where the first-scheduled call completes last.
        # Absurd assigns the `#1`/`#2` slot when `ctx.step(...)` is entered, before the tool body
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

        def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart(content='done')])
            return ModelResponse(
                parts=[
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
            _tool_calling_model('echo', {'text': 'hi'}),
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


class TestCheckpointFormat:
    async def test_hand_written_stream_payload_replays(self, absurd: AsyncAbsurd) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='gold', capabilities=[AbsurdDurability()])
        # Pins the `{response, events}` payload shape of a stream checkpoint.
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

        async def write() -> JsonValue:
            return payload

        async with running_task_context(absurd) as ctx:
            await ctx.step('gold__model.request_stream', write)
        async with reenter_running_task(absurd, ctx.task_id):
            async with agent.run_stream('hi') as result:
                assert await result.get_output() == 'golden-stream'

        assert counter['calls'] == 0


def _dynamic_toolset(tool_calls: dict[str, int], *, id: str | None) -> DynamicToolset[object]:
    def build(ctx: RunContext[object]) -> FunctionToolset[object]:
        inner: FunctionToolset[object] = FunctionToolset(id='inner')

        @inner.tool_plain
        def greet(name: str) -> str:
            tool_calls['n'] += 1
            return f'hi {name}'

        return inner

    return DynamicToolset(build, id=id)


class TestDynamicToolset:
    """Function and MCP toolsets are checkpointed; a construction-time `DynamicToolset` runs as-is."""

    @pytest.mark.parametrize('toolset_id', ['dyn', None])
    async def test_runs_uncheckpointed_inside_a_task(self, absurd: AsyncAbsurd, toolset_id: str | None) -> None:
        tool_calls = {'n': 0}
        agent = Agent(
            _tool_calling_model('greet', {'name': 'ada'}),
            name='d',
            toolsets=[_dynamic_toolset(tool_calls, id=toolset_id)],
            capabilities=[AbsurdDurability()],
        )

        async with running_task_context(absurd) as ctx:
            first = await agent.run('greet ada')
        assert list(await checkpoints(absurd, ctx.task_id)) == ['d__model.request', 'd__model.request#2']
        async with reenter_running_task(absurd, ctx.task_id):
            second = await agent.run('greet ada')

        # The model responses replay from their checkpoints; the dynamic tool runs again.
        assert first.output == second.output == 'done'
        assert tool_calls['n'] == 2

    async def test_runtime_dynamic_toolset_rejected_inside_task(self, absurd: AsyncAbsurd) -> None:
        tool_calls = {'n': 0}
        agent = Agent(_tool_calling_model('greet', {'name': 'ada'}), name='d', capabilities=[AbsurdDurability()])
        async with running_task_context(absurd):
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('greet ada', toolsets=[_dynamic_toolset(tool_calls, id='late')])
        assert tool_calls['n'] == 0

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

        calls = {'n': 0}
        toolset = FunctionToolset(id='tools')

        @toolset.tool_plain
        def search(query: str) -> str:
            calls['n'] += 1
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
        assert calls['n'] == 1
