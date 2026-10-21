"""`PGTaskDurability` against a real pgtask schema on PostgreSQL.

Each test runs its agent inside a task worked by a real pgtask `Worker` (`_task.run_in_task`); a
replay is pgtask's own retry after a simulated crash, so step naming, occurrence numbering and
checkpoint storage are pgtask's.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import AsyncIterable, AsyncIterator

import anyio
import pytest

pytest.importorskip('pgtask')
pytest.importorskip('fastmcp')

from fastmcp import FastMCP
from inline_snapshot import snapshot
from pgtask import Task, get_current_task

from pydantic_ai import Agent, ToolReturn
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import ModelRetry, ToolFailed, UserError
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
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, ExternalToolset, FunctionToolset
from pydantic_ai_harness.pgtask import CheckpointDecodeError, PGTaskDurability
from pydantic_ai_harness.pgtask._context import _occurrences  # pyright: ignore[reportPrivateUsage]
from tests.durable_exec.counting_mcp import counting_mcp_server

from ...conftest import try_import
from ._task import PGTaskDatabase, checkpoints, run_in_task

with try_import() as code_mode_available:
    from pydantic_ai_harness import CodeMode


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


_RUNTIME_TOOLSET_ERROR = 'cannot be added at runtime with pgtask'


class TestConstruction:
    async def test_requires_name(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(_make_model(), capabilities=[PGTaskDurability()])

    async def test_requires_model(self) -> None:
        with pytest.raises(UserError, match='needs to have a `model`'):
            Agent(name='a', capabilities=[PGTaskDurability()])

    async def test_reserved_default_model_id_raises(self) -> None:
        with pytest.raises(UserError, match="'default' is reserved"):
            Agent(_make_model(), name='a', capabilities=[PGTaskDurability(models={'default': _make_model()})])

    @pytest.mark.parametrize('kind', ['function', 'mcp', 'dynamic'])
    async def test_leaf_toolset_without_id_raises(self, kind: str) -> None:
        toolsets: dict[str, AbstractToolset[object]] = {
            'function': FunctionToolset[object](),
            'mcp': MCPToolset[object](FastMCP(name='calc')),
            'dynamic': _dynamic_toolset({'calls': 0}, id=None),
        }
        with pytest.raises(UserError, match='need to have a unique `id`'):
            Agent(_make_model(), name='a', toolsets=[toolsets[kind]], capabilities=[PGTaskDurability()])

    async def test_duplicate_toolset_id_raises(self) -> None:
        toolsets = [FunctionToolset[object](id='tools'), FunctionToolset[object](id='tools')]
        with pytest.raises(UserError, match='same `id`'):
            Agent(_make_model(), name='a', toolsets=toolsets, capabilities=[PGTaskDurability()])

    async def test_name_from_capability(self) -> None:
        agent = Agent(_make_model(), capabilities=[PGTaskDurability(name='custom')])
        bound = PGTaskDurability.from_agent(agent)
        assert bound is not None
        assert bound.name == 'custom'

    async def test_from_agent_without_capability_returns_none(self) -> None:
        assert PGTaskDurability.from_agent(Agent(_make_model(), name='a')) is None

    async def test_a_second_engine_is_refused(self) -> None:
        with pytest.raises(UserError, match='can have only one durable execution engine'):
            Agent(_make_model(), name='a', capabilities=[PGTaskDurability(), PGTaskDurability()])

    async def test_construction_external_toolset_passes_through_unwrapped(self) -> None:
        external = ExternalToolset[object](tool_defs=[])
        agent = Agent(_make_model(), name='a', toolsets=[external], capabilities=[PGTaskDurability()])
        assert any(t is external for t in agent.toolsets)


class TestOutsideTask:
    async def test_run_is_transparent(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[PGTaskDurability()])
        result = await agent.run('hi')
        assert result.output == 'ok'
        assert counter['calls'] == 1

    async def test_override_toolsets_respected(self) -> None:
        calls = {'calls': 0}
        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('late')), name='a', capabilities=[PGTaskDurability()]
        )
        with agent.override(toolsets=[_late_toolset(calls)]):
            result = await agent.run('hi')
        assert result.output == 'done'
        assert calls['calls'] == 1

    async def test_override_tools_respected(self) -> None:
        # The overriding toolset shares the agent's own toolset id, and must not be swapped for its wrapper.
        calls = {'calls': 0}

        def late() -> str:
            calls['calls'] += 1
            return 'late result'

        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('late')), name='a', capabilities=[PGTaskDurability()]
        )
        with agent.override(tools=[late]):
            result = await agent.run('hi')
        assert result.output == 'done'
        assert calls['calls'] == 1


class TestReplay:
    async def test_model_response_served_from_checkpoint(self, pgtask_db: PGTaskDatabase) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='crash', capabilities=[PGTaskDurability()])
        outputs: list[str] = []

        async def body(attempt: int) -> None:
            outputs.append((await agent.run('hi')).output)

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert outputs == ['ok', 'ok']
        assert counter['calls'] == 1
        assert list(await checkpoints(pgtask_db, task_id)) == [('crash__model.request', 0)]

    async def test_function_tool_runs_once(self, pgtask_db: PGTaskDatabase) -> None:
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
            capabilities=[PGTaskDurability()],
        )
        outputs: list[str] = []

        async def body(attempt: int) -> None:
            outputs.append((await agent.run('charge it')).output)

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert outputs == ['done', 'done']
        assert tool_calls['calls'] == 1
        stored = await checkpoints(pgtask_db, task_id)
        assert stored[('billing__function_toolset__tools.call_tool:charge_card', 0)] == snapshot(
            {'result': 'charged 42', 'kind': 'tool_return'}
        )

    async def test_agent_level_tool_step_name_is_normalized(self, pgtask_db: PGTaskDatabase) -> None:
        # Tools registered with `@agent.tool_plain` live on the agent's own toolset, whose id is the
        # literal `<agent>`: a step name pgtask rejects unless it is normalized.
        tool_calls = {'calls': 0}
        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('charge_card')), name='billing', capabilities=[PGTaskDurability()]
        )

        @agent.tool_plain
        def charge_card() -> str:
            tool_calls['calls'] += 1
            return 'charged'

        async def body(attempt: int) -> None:
            assert (await agent.run('charge it')).output == 'done'

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert tool_calls['calls'] == 1
        assert ('billing__function_toolset___agent_.call_tool:charge_card', 0) in await checkpoints(pgtask_db, task_id)

    async def test_capability_owned_toolset_is_durable(self, pgtask_db: PGTaskDatabase) -> None:
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
            capabilities=[DemoCapability(), PGTaskDurability()],
        )

        async def body(attempt: int) -> None:
            assert (await agent.run('charge it')).output == 'done'

        await run_in_task(pgtask_db, body, attempts=2)
        assert tool_calls['calls'] == 1

    async def test_dynamic_toolset_is_durable(self, pgtask_db: PGTaskDatabase) -> None:
        tool_calls = {'calls': 0}
        agent = Agent(
            _tool_calling_model(ToolCallPart('greet', {'name': 'ada'})),
            name='d',
            toolsets=[_dynamic_toolset(tool_calls, id='dyn')],
            capabilities=[PGTaskDurability()],
        )

        async def body(attempt: int) -> None:
            assert (await agent.run('greet ada')).output == 'done'

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert tool_calls['calls'] == 1
        assert list(await checkpoints(pgtask_db, task_id)) == snapshot(
            [
                ('d__dynamic_toolset__dyn.get_tools', 0),
                ('d__model.request', 0),
                ('d__dynamic_toolset__dyn.call_tool:greet', 0),
                ('d__dynamic_toolset__dyn.get_tools', 1),
                ('d__model.request', 1),
            ]
        )

    async def test_registered_model_selected_per_run(self, pgtask_db: PGTaskDatabase) -> None:
        # The selected model checkpoints under its own id-scoped step name, and a replay serves it.
        primary = {'calls': 0}
        cheap = {'calls': 0}
        agent = Agent(
            _make_model(primary, text='primary'),
            name='a',
            capabilities=[PGTaskDurability(models={'cheap': _make_model(cheap, text='cheap')})],
        )
        outputs: list[str] = []

        async def body(attempt: int) -> None:
            outputs.append((await agent.run('hi')).output)
            outputs.append((await agent.run('hi', model='cheap')).output)

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert outputs == ['primary', 'cheap'] * 2
        assert primary['calls'] == cheap['calls'] == 1
        assert list(await checkpoints(pgtask_db, task_id)) == [('a__model.request', 0), ('a__model.request.cheap', 0)]

    async def test_string_default_model_gets_unsuffixed_step_name(self, pgtask_db: PGTaskDatabase) -> None:
        agent = Agent('test', name='strdef', capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            await agent.run('hi')

        task_id = await run_in_task(pgtask_db, body)
        assert list(await checkpoints(pgtask_db, task_id)) == [('strdef__model.request', 0)]

    async def test_two_runs_in_one_task_take_successive_occurrences(self, pgtask_db: PGTaskDatabase) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            await agent.run('hi')
            await agent.run('hi again')

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert counter['calls'] == 2
        assert list(await checkpoints(pgtask_db, task_id)) == [('a__model.request', 0), ('a__model.request', 1)]

    async def test_same_toolset_instance_in_two_places_is_wrapped_once(self, pgtask_db: PGTaskDatabase) -> None:
        # Both mounts checkpoint through the one wrapper, so the two calls take successive occurrences
        # of the same step name.
        toolset = FunctionToolset[object](id='shared')

        @toolset.tool_plain
        def echo(value: str) -> str:
            return value

        agent = Agent(
            _tool_calling_model(ToolCallPart('a_echo', {'value': 'a'}), ToolCallPart('b_echo', {'value': 'b'})),
            name='a',
            toolsets=[toolset.prefixed('a'), toolset.prefixed('b')],
            capabilities=[PGTaskDurability()],
        )

        async def body(attempt: int) -> None:
            await agent.run('hi')

        task_id = await run_in_task(pgtask_db, body)

        step = 'a__function_toolset__shared.call_tool:echo'
        stored = await checkpoints(pgtask_db, task_id)
        assert [(stored[(step, 0)], stored[(step, 1)])] == [
            ({'result': 'a', 'kind': 'tool_return'}, {'result': 'b', 'kind': 'tool_return'})
        ]


class TestTaskLifetime:
    async def test_counters_are_dropped_with_their_task(self, pgtask_db: PGTaskDatabase) -> None:
        # Counters are keyed by `id(task)`, and Python reuses the id of a freed task for a later one, so
        # a counter left behind would make the later task's steps start past occurrence 0.
        agent = Agent(_make_model(), name='a', capabilities=[PGTaskDurability()])
        tasks: list[weakref.ref[Task]] = []
        keys: list[int] = []

        async def body(attempt: int) -> None:
            task = get_current_task()
            assert task is not None
            tasks.append(weakref.ref(task))
            await agent.run('hi')
            assert id(task) in _occurrences
            keys.append(id(task))

        await run_in_task(pgtask_db, body, attempts=2)
        gc.collect()

        assert len(tasks) == 2
        assert [ref() for ref in tasks] == [None, None]
        assert [key in _occurrences for key in keys] == [False, False]


class TestRuntimeToolsets:
    @pytest.mark.parametrize('kind', ['function', 'dynamic'])
    async def test_runtime_toolset_rejected(self, pgtask_db: PGTaskDatabase, kind: str) -> None:
        calls = {'calls': 0}
        late = _late_toolset(calls) if kind == 'function' else _dynamic_toolset(calls, id='late')
        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('hi', toolsets=[late])

        await run_in_task(pgtask_db, body)
        assert calls == {'calls': 0}

    async def test_override_toolsets_rejected(self, pgtask_db: PGTaskDatabase) -> None:
        calls = {'calls': 0}
        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('late')), name='a', capabilities=[PGTaskDurability()]
        )

        async def body(attempt: int) -> None:
            with agent.override(toolsets=[_late_toolset(calls)]):
                with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                    await agent.run('hi')

        await run_in_task(pgtask_db, body)
        assert calls['calls'] == 0

    async def test_override_tools_rejected(self, pgtask_db: PGTaskDatabase) -> None:
        # Rejected before it can run.
        def late() -> str:  # pragma: no cover
            return 'late result'

        agent: Agent[object, str] = Agent(
            _tool_calling_model(ToolCallPart('late')), name='a', capabilities=[PGTaskDurability()]
        )

        async def body(attempt: int) -> None:
            with agent.override(tools=[late]):
                with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                    await agent.run('hi')

        await run_in_task(pgtask_db, body)

    async def test_still_rejected_alongside_capability_toolset(self, pgtask_db: PGTaskDatabase) -> None:
        # Skipping capability-owned wrappers must not let a genuine runtime toolset through.
        owned = FunctionToolset[object](id='owned')

        class DemoCapability(AbstractCapability[object]):
            def get_toolset(self) -> FunctionToolset[object]:
                return owned

        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[DemoCapability(), PGTaskDurability()])

        async def body(attempt: int) -> None:
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('hi', toolsets=[_late_toolset({'calls': 0})])

        await run_in_task(pgtask_db, body)

    async def test_toolset_decorated_after_construction_rejected(self, pgtask_db: PGTaskDatabase) -> None:
        agent = Agent(_make_model(), name='decorated', capabilities=[PGTaskDurability()])

        @agent.toolset(id='decorated-tools')
        def build(ctx: RunContext[object]) -> FunctionToolset[object]:  # pragma: no cover
            return FunctionToolset[object](id='inner')

        async def body(attempt: int) -> None:
            with pytest.raises(UserError, match=_RUNTIME_TOOLSET_ERROR):
                await agent.run('hi')

        await run_in_task(pgtask_db, body)

    async def test_external_toolset_allowed(self, pgtask_db: PGTaskDatabase) -> None:
        agent: Agent[object, str] = Agent(_make_model(), name='a', capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            assert (await agent.run('hi', toolsets=[ExternalToolset[object](tool_defs=[])])).output == 'ok'

        await run_in_task(pgtask_db, body)


class TestStreaming:
    async def test_event_stream_handler_receives_events(self, pgtask_db: PGTaskDatabase) -> None:
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
            capabilities=[PGTaskDurability(event_stream_handler=handler)],
        )

        async def body(attempt: int) -> None:
            assert (await agent.run('hi')).output == 'done'

        task_id = await run_in_task(pgtask_db, body)

        assert any(isinstance(e, PartStartEvent | PartDeltaEvent) for e in events)
        assert any(isinstance(e, FunctionToolCallEvent) for e in events)
        assert ('a__event_stream_handler', 0) in await checkpoints(pgtask_db, task_id)

    async def test_run_stream_events_replays(self, pgtask_db: PGTaskDatabase) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[PGTaskDurability()])
        runs: list[list[AgentStreamEvent | AgentRunResultEvent[str]]] = []

        async def body(attempt: int) -> None:
            async with agent.run_stream_events('hi') as stream:
                runs.append([event async for event in stream])

        await run_in_task(pgtask_db, body, attempts=2)

        first, replayed = runs
        assert any(isinstance(e, PartStartEvent) for e in first)
        assert replayed == first
        assert counter['calls'] == 1

    async def test_run_stream_replays_buffered_stream(self, pgtask_db: PGTaskDatabase) -> None:
        counter = {'calls': 0}
        agent = Agent(_make_model(counter), name='a', capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            async with agent.run_stream('hi') as result:
                assert await result.get_output() == 'ok'

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert counter['calls'] == 1
        assert list(await checkpoints(pgtask_db, task_id)) == [('a__model.request_stream', 0)]

    async def test_iter(self, pgtask_db: PGTaskDatabase) -> None:
        agent = Agent(_make_model(), name='a', capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            async with agent.iter('hi') as run:
                async for _ in run:
                    pass
            assert run.result is not None
            assert run.result.output == 'ok'

        task_id = await run_in_task(pgtask_db, body)
        assert list(await checkpoints(pgtask_db, task_id)) == [('a__model.request', 0)]


class TestToolResults:
    @pytest.mark.parametrize(
        ('error', 'stored'),
        [
            (ModelRetry('nope, try again'), snapshot({'kind': 'model_retry', 'message': 'nope, try again'})),
            (ToolFailed('no such record'), snapshot({'kind': 'tool_failed', 'message': 'no such record'})),
        ],
        ids=['model_retry', 'tool_failed'],
    )
    async def test_failed_tool_result_is_checkpointed_and_replayed(
        self, pgtask_db: PGTaskDatabase, error: Exception, stored: object
    ) -> None:
        # A failure the model sees is stored like any other tool result, so a replay serves it
        # without running the tool again.
        model_calls = {'calls': 0}
        tool_calls = {'calls': 0}
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            tool_calls['calls'] += 1
            raise error

        model = _tool_calling_model(ToolCallPart('flaky'), counter=model_calls)
        agent = Agent(model, name='retry', toolsets=[toolset], capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            assert (await agent.run('go')).output == 'done'

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert model_calls['calls'] == 2
        assert tool_calls['calls'] == 1
        assert (await checkpoints(pgtask_db, task_id))[('retry__function_toolset__tools.call_tool:flaky', 0)] == stored

    async def test_tool_return_object_round_trips_through_replay(self, pgtask_db: PGTaskDatabase) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def lookup() -> ToolReturn:
            return ToolReturn(return_value='value', content='extra context', metadata={'source': 'db'})

        agent = Agent(
            _tool_calling_model(ToolCallPart('lookup')),
            name='tr',
            toolsets=[toolset],
            capabilities=[PGTaskDurability()],
        )
        returned: list[list[tuple[object, object]]] = []

        async def body(attempt: int) -> None:
            messages = (await agent.run('go')).all_messages()
            returned.append(
                [(p.content, p.metadata) for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
            )

        await run_in_task(pgtask_db, body, attempts=2)
        assert returned == [[('value', {'source': 'db'})]] * 2

    async def test_undecodable_tool_checkpoint_fails_the_run(self, pgtask_db: PGTaskDatabase) -> None:
        # A checkpoint that does not decode as a tool result must not read as bad model arguments:
        # the model would be asked to retry, and the retry would run the tool again.
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        # Never re-run: the stored checkpoint is used instead.
        def charge_card(amount: int) -> str:  # pragma: no cover
            return 'charged again'

        agent = Agent(
            _tool_calling_model(ToolCallPart('charge_card', {'amount': 42})),
            name='billing',
            toolsets=[toolset],
            capabilities=[PGTaskDurability()],
        )

        async def body(attempt: int) -> None:
            with pytest.raises(CheckpointDecodeError):
                await agent.run('charge it')

        await run_in_task(
            pgtask_db, body, seed={('billing__function_toolset__tools.call_tool:charge_card', 0): 'charged 42'}
        )


class TestCrashMidRun:
    async def test_model_step_served_from_checkpoint_while_failed_tool_reruns(self, pgtask_db: PGTaskDatabase) -> None:
        # The model step completes and is checkpointed, then a tool raises a real (non-`ModelRetry`)
        # error that fails the attempt. On retry the model step is served from its checkpoint while
        # the tool runs again.
        model_calls = {'calls': 0}
        tool_attempts = {'calls': 0}
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            tool_attempts['calls'] += 1
            if tool_attempts['calls'] == 1:
                raise RuntimeError('worker died mid-tool')
            return 'recovered'

        model = _tool_calling_model(ToolCallPart('flaky'), counter=model_calls)
        agent = Agent(model, name='crash', toolsets=[toolset], capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            if attempt == 1:
                with pytest.raises(RuntimeError, match='worker died mid-tool'):
                    await agent.run('go')
            else:
                assert (await agent.run('go')).output == 'done'

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert model_calls['calls'] == tool_attempts['calls'] == 2
        assert list(await checkpoints(pgtask_db, task_id)) == [
            ('crash__model.request', 0),
            ('crash__function_toolset__tools.call_tool:flaky', 0),
            ('crash__model.request', 1),
        ]

    async def test_cancel_suspended_response_is_checkpointed(self, pgtask_db: PGTaskDatabase) -> None:
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

        agent = Agent(CancellableModel(fn, model_name='fn'), name='a', capabilities=[PGTaskDurability()])

        async def body(attempt: int) -> None:
            with pytest.raises(RuntimeError, match='continuation failed'):
                await agent.run('hi')

        task_id = await run_in_task(pgtask_db, body)

        assert [response.state for response in cancelled] == ['suspended']
        assert ('a__model.cancel_suspended_response', 0) in await checkpoints(pgtask_db, task_id)


class TestCapabilityOperation:
    async def test_operation_is_checkpointed_and_replayed(self, pgtask_db: PGTaskDatabase) -> None:
        calls: list[str] = []

        class Recorder(AbstractCapability[object]):
            id = 'recorder'

            async def before_run(self, ctx: RunContext[object]) -> None:
                await self.record(ctx, 'started')

            @durable_operation('record')
            async def record(self, ctx: RunContext[object], value: str) -> None:
                del ctx
                calls.append(value)

        agent = Agent(_make_model(), name='cap', capabilities=[Recorder(), PGTaskDurability()])

        async def body(attempt: int) -> None:
            await agent.run('hi')

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert calls == ['started']
        assert ('cap__capability__recorder.record', 0) in await checkpoints(pgtask_db, task_id)


class TestParallelExecutionMode:
    async def test_occurrences_follow_scheduling_order_not_completion(self, pgtask_db: PGTaskDatabase) -> None:
        # Two concurrent calls of the same tool, where the first-scheduled call completes last. The
        # occurrence is taken when the step starts, before the tool body runs, so occurrences follow
        # the model's tool-call order and a replay serves each call its own result.
        toolset = FunctionToolset[object](id='tools')
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
            capabilities=[PGTaskDurability(parallel_execution_mode='parallel_ordered_events')],
        )
        returned: list[list[tuple[str, object]]] = []

        async def body(attempt: int) -> None:
            messages = (await agent.run('go')).all_messages()
            returned.append(
                [(p.tool_call_id, p.content) for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
            )

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        assert returned == [[('r1', 'first'), ('r2', 'second')]] * 2
        step = 'par__function_toolset__tools.call_tool:record'
        stored = await checkpoints(pgtask_db, task_id)
        assert (stored[(step, 0)], stored[(step, 1)]) == (
            {'result': 'first', 'kind': 'tool_return'},
            {'result': 'second', 'kind': 'tool_return'},
        )


class TestMcp:
    async def test_step_names_and_replay_without_connecting(self, pgtask_db: PGTaskDatabase) -> None:
        server: FastMCP[None] = FastMCP(name='calc')

        @server.tool
        def add(a: int, b: int) -> int:
            return a + b

        def make_agent(toolset: MCPToolset[object]) -> Agent[object, str]:
            return Agent(
                _tool_calling_model(ToolCallPart('add', {'a': 2, 'b': 3})),
                name='calc',
                toolsets=[toolset],
                capabilities=[PGTaskDurability()],
            )

        live = make_agent(MCPToolset[object](server, id='calc'))
        # A replay served entirely from checkpoints does not connect, so an unreachable server is fine.
        unreachable = make_agent(MCPToolset[object]('http://127.0.0.1:9/mcp', id='calc'))

        async def body(attempt: int) -> None:
            assert (await (live if attempt == 1 else unreachable).run('add 2 and 3')).output == 'done'

        task_id = await run_in_task(pgtask_db, body, attempts=2)

        # Step names are what a replay looks checkpoints up by, so they must not drift across releases.
        assert list(await checkpoints(pgtask_db, task_id)) == [
            ('calc__mcp_server__calc.get_tools', 0),
            ('calc__model.request', 0),
            ('calc__mcp_server__calc.call_tool', 0),
            ('calc__model.request', 1),
        ]

    async def test_replay_opens_no_mcp_session(self, pgtask_db: PGTaskDatabase) -> None:
        server, counts = counting_mcp_server(instructions='Echo things.')
        agent = Agent(
            _tool_calling_model(ToolCallPart('echo', {'text': 'hi'})),
            name='echo',
            toolsets=[MCPToolset[object](server, id='echo', include_instructions=True)],
            capabilities=[PGTaskDurability()],
        )
        after_first_run: dict[str, int] = {}
        replayed: list[ModelMessage] = []

        async def body(attempt: int) -> None:
            result = await agent.run('go')
            if attempt == 1:
                after_first_run.update(counts)
            else:
                replayed.extend(result.all_messages())

        await run_in_task(pgtask_db, body, attempts=2)

        assert counts == after_first_run
        # The replayed run gets the server's instructions from their checkpoint.
        requests = [m for m in replayed if isinstance(m, ModelRequest)]
        assert requests[0].instructions is not None and 'Echo things.' in requests[0].instructions


@pytest.mark.skipif(not code_mode_available(), reason='pydantic-monty not installed')
class TestCodeMode:
    async def test_tool_call_inside_run_code_is_checkpointed(self, pgtask_db: PGTaskDatabase) -> None:
        calls = {'calls': 0}
        toolset = FunctionToolset[object](id='tools')

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
            FunctionModel(model_fn), name='composed', toolsets=[toolset], capabilities=[CodeMode(), PGTaskDurability()]
        )

        async def body(attempt: int) -> None:
            # The `run_code` body runs again on replay, but its `search` call is served from the checkpoint.
            assert (await agent.run('go')).output == 'done: results for x'

        await run_in_task(pgtask_db, body, attempts=2)
        assert calls['calls'] == 1
