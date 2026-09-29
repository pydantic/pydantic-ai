"""Tests for the `RestateDurability` capability.

Behavior is driven through `Agent(..., capabilities=[RestateDurability()])` inside an in-memory
`FakeRestateContext` (see `conftest.py`) so there is no network or Docker dependency. The two
production behaviors the capability relies on -- a positional journal and a replay that serves
stored entries without re-running the action -- are reproduced faithfully by the fake.
"""

from __future__ import annotations

import builtins
import runpy
from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from pathlib import Path

import pytest

pytest.importorskip('restate')

from restate.exceptions import TerminalError

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import AbstractCapability, durable_operation
from pydantic_ai.exceptions import ApprovalRequired, CallDeferred, ModelRetry, UserError
from pydantic_ai.messages import (
    AgentStreamEvent,
    BinaryContent,
    FunctionToolCallEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.toolsets import DynamicToolset, ExternalToolset, FunctionToolset
from pydantic_ai_harness.restate import RestateDurability

from .conftest import Entry, FakeRestateContext, restate_context
from .test_restate_mcp import FakeMCPToolset

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


def _tool_then_done_model(tool_name: str, args: dict[str, object]) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        answered = any(
            isinstance(part, (ToolReturnPart, RetryPromptPart)) for message in messages for part in message.parts
        )
        if not answered:
            return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=args)])
        return ModelResponse(parts=[TextPart(content='done')])

    return FunctionModel(fn, model_name='fn')


class _CountingOperation(AbstractCapability[object]):
    id = 'counter'

    def __init__(self, calls: list[int]) -> None:
        self.calls = calls

    async def before_run(self, ctx: RunContext[object]) -> None:
        await self.increment(ctx)

    @durable_operation('increment')
    async def increment(self, ctx: RunContext[object]) -> int:
        self.calls.append(1)
        return len(self.calls)


class TestTransparency:
    def test_missing_restate_dependency_has_install_guidance(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_import = builtins.__import__

        def import_without_restate(
            name: str,
            globals: Mapping[str, object] | None = None,
            locals: Mapping[str, object] | None = None,
            fromlist: Sequence[str] = (),
            level: int = 0,
        ) -> object:
            if name == 'restate':
                raise ImportError
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, '__import__', import_without_restate)
        capability_path = (
            Path(__file__).parents[3]
            / 'src'
            / 'pydantic_ai_harness'
            / 'pydantic_ai_harness'
            / 'restate'
            / '_capability.py'
        )
        with pytest.raises(ImportError, match=r'pydantic-ai-harness\[restate\]'):
            runpy.run_path(str(capability_path))

    async def test_run_outside_context_is_transparent(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[RestateDurability()])
        result = await agent.run('hi')
        assert result.output == 'ok'
        assert counter['calls'] == 1


class TestModelRequestCheckpoint:
    async def test_steps_have_a_finite_retry_default(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[RestateDurability()])
        ctx = FakeRestateContext()

        with restate_context(ctx):
            await agent.run('hello')

        assert ctx.max_attempts == [3]

    async def test_max_attempts_is_configurable(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[RestateDurability(max_attempts=None)])
        ctx = FakeRestateContext()

        with restate_context(ctx):
            await agent.run('hello')

        assert ctx.max_attempts == [None]

    @pytest.mark.parametrize('max_attempts', [0, -1, True])
    def test_max_attempts_must_be_positive(self, max_attempts: int) -> None:
        with pytest.raises(UserError, match='positive integer or `None`'):
            RestateDurability(max_attempts=max_attempts)

    async def test_request_journaled_and_replay_serves_entry(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            first = await agent.run('hi')
        assert ctx.step_names == ['a__model.request']
        assert ctx.invoked == ['a__model.request']

        replay = ctx.replay()
        with restate_context(replay):
            second = await agent.run('hi')

        assert counter['calls'] == 1
        assert first.output == second.output == 'ok'
        assert replay.invoked == []


class TestStreaming:
    async def test_stream_journaled_and_replayed(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='stream', capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            async with agent.run_stream('hi') as result:
                first_out = await result.get_output()
        assert first_out == 'ok'
        assert 'stream__model.request_stream' in ctx.step_names

        replay = ctx.replay()
        with restate_context(replay):
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
            _text_model(counter), name='stream', capabilities=[RestateDurability(event_stream_handler=handler)]
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            async with agent.run_stream_events('hi') as stream:
                first_events = [event async for event in stream]

        # The model-stream events were delivered live to the handler inside the request_stream step.
        assert any(isinstance(event, (PartStartEvent, PartDeltaEvent)) for event in live_events)
        assert 'stream__model.request_stream' in ctx.step_names

        replay = ctx.replay()
        with restate_context(replay):
            async with agent.run_stream_events('hi') as stream:
                replay_events = [event async for event in stream]

        assert counter['calls'] == 1
        assert replay_events == first_events


class TestFunctionTool:
    async def test_tool_journaled_exactly_once_across_replay(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset[object](id='billing')

        @toolset.tool_plain
        def charge_card(amount: int) -> str:
            calls['n'] += 1
            return f'charged {amount}'

        agent = Agent(
            _tool_then_done_model('charge_card', {'amount': 7}),
            name='pay',
            toolsets=[toolset],
            capabilities=[RestateDurability()],
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            first = await agent.run('charge it')
        assert 'pay__function_toolset__billing.call_tool:charge_card' in ctx.step_names

        replay = ctx.replay()
        with restate_context(replay):
            second = await agent.run('charge it')

        assert calls['n'] == 1
        assert first.output == second.output == 'done'
        assert replay.invoked == []


class TestControlFlowSignals:
    """Every control-flow signal must cross the journal as a value, not as a step failure."""

    async def test_model_retry_crosses_journal_and_replays(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def flaky() -> str:
            calls['n'] += 1
            raise ModelRetry('nope, try again')

        agent = Agent(
            _tool_then_done_model('flaky', {}), name='retry', toolsets=[toolset], capabilities=[RestateDurability()]
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            first = await agent.run('go')
        step = 'retry__function_toolset__tools.call_tool:flaky'
        # The raised `ModelRetry` crossed the journal as a serialized value, not an exception.
        assert ctx.stored(step) == {'message': 'nope, try again', 'kind': 'model_retry'}
        assert first.output == 'done'
        assert calls['n'] == 1

        replay = ctx.replay()
        with restate_context(replay):
            second = await agent.run('go')

        assert calls['n'] == 1
        assert second.output == 'done'
        assert replay.invoked == []

    async def test_approval_required_preserves_metadata(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def act() -> str:
            raise ApprovalRequired(metadata={'reason': 'needs sign-off'})

        agent = Agent(
            _tool_then_done_model('act', {}),
            name='ap',
            toolsets=[toolset],
            output_type=[str, DeferredToolRequests],
            capabilities=[RestateDurability()],
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('go')

        assert isinstance(result.output, DeferredToolRequests)
        assert len(result.output.approvals) == 1
        step = 'ap__function_toolset__tools.call_tool:act'
        assert ctx.stored(step) == {'metadata': {'reason': 'needs sign-off'}, 'kind': 'approval_required'}

    async def test_call_deferred_preserves_metadata(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def act() -> str:
            raise CallDeferred(metadata={'ticket': 'ABC-1'})

        agent = Agent(
            _tool_then_done_model('act', {}),
            name='cd',
            toolsets=[toolset],
            output_type=[str, DeferredToolRequests],
            capabilities=[RestateDurability()],
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('go')

        assert isinstance(result.output, DeferredToolRequests)
        assert len(result.output.calls) == 1
        step = 'cd__function_toolset__tools.call_tool:act'
        assert ctx.stored(step) == {'metadata': {'ticket': 'ABC-1'}, 'kind': 'call_deferred'}


class TestToolResultSerialization:
    async def test_non_serializable_tool_return_is_terminal(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def act() -> object:
            return object()

        agent = Agent(
            _tool_then_done_model('act', {}), name='bad', toolsets=[toolset], capabilities=[RestateDurability()]
        )

        ctx = FakeRestateContext()
        with restate_context(ctx), pytest.raises(TerminalError):
            await agent.run('go')

    async def test_tool_return_round_trips(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def act() -> ToolReturn:
            return ToolReturn(return_value='ok', content='extra')

        agent = Agent(
            _tool_then_done_model('act', {}), name='tr', toolsets=[toolset], capabilities=[RestateDurability()]
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('go')

        assert result.output == 'done'

    async def test_binary_content_round_trips(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def act() -> BinaryContent:
            return BinaryContent(data=b'\x89PNG', media_type='image/png')

        agent = Agent(
            _tool_then_done_model('act', {}), name='bc', toolsets=[toolset], capabilities=[RestateDurability()]
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('go')

        assert result.output == 'done'


class TestCrashMidRunRetry:
    async def test_model_step_served_from_journal_while_failed_tool_reruns(self) -> None:
        # The core value prop: the model step completes and is journaled, then a tool raises a real
        # (non-`ModelRetry`) error that fails the run. On retry, Restate replays: the model step is
        # served from its journal entry (model not called again) while the tool re-runs.
        model_calls = {'n': 0}
        tool_attempts = {'n': 0}
        toolset = FunctionToolset[object](id='tools')

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

        agent = Agent(FunctionModel(model_fn), name='crash', toolsets=[toolset], capabilities=[RestateDurability()])

        model_step = 'crash__model.request'
        tool_step = 'crash__function_toolset__tools.call_tool:flaky'

        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(RuntimeError, match='worker died mid-tool'):
                await agent.run('go')

        # The model step was journaled before the tool ran; the failed tool step was not.
        assert model_step in ctx.step_names
        assert tool_step not in ctx.step_names
        assert model_calls['n'] == 1
        assert tool_attempts['n'] == 1

        replay = ctx.replay()
        with restate_context(replay):
            result = await agent.run('go')

        assert result.output == 'done'
        # The first model request was served from its journal entry (not re-invoked); the tool
        # re-ran, and the second model turn recurs under the same label as a fresh journal slot.
        assert replay.invoked == [tool_step, model_step]
        assert model_calls['n'] == 2
        assert tool_attempts['n'] == 2


class TestPerToolOptOut:
    async def test_metadata_false_runs_inline_unjournaled(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain(metadata={'restate': False})
        def ping() -> str:
            calls['n'] += 1
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='inline', toolsets=[toolset], capabilities=[RestateDurability()]
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('ping it')

        assert result.output == 'done'
        assert calls['n'] == 1
        assert not any('call_tool:ping' in name for name in ctx.step_names)

    async def test_empty_dict_config_is_allowed(self) -> None:
        calls = {'n': 0}
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain(metadata={'restate': {}})
        def ping() -> str:
            calls['n'] += 1
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='cfg', toolsets=[toolset], capabilities=[RestateDurability()]
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('ping it')

        assert result.output == 'done'
        assert calls['n'] == 1
        assert 'cfg__function_toolset__tools.call_tool:ping' in ctx.step_names

    async def test_non_empty_dict_config_raises(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain(metadata={'restate': {'retries': 3}})
        def ping() -> str:  # pragma: no cover - rejected before it can run
            return 'pong'

        agent = Agent(
            _tool_then_done_model('ping', {}), name='cfg', toolsets=[toolset], capabilities=[RestateDurability()]
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(UserError, match='take no per-tool options'):
                await agent.run('ping it')


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
            capabilities=[RestateDurability(models={'cheap': FunctionModel(cheap_fn, model_name='cheap')})],
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            default_result = await agent.run('hi')
            cheap_result = await agent.run('hi', model='cheap')

        assert default_result.output == 'primary'
        assert cheap_result.output == 'cheap'
        assert 'sw__model.request' in ctx.step_names
        assert 'sw__model.request.cheap' in ctx.step_names

        # The positional journal replays the same step sequence, so re-run both turns in order.
        replay = ctx.replay()
        with restate_context(replay):
            replayed_default = await agent.run('hi')
            replayed_cheap = await agent.run('hi', model='cheap')

        assert replayed_default.output == 'primary'
        assert replayed_cheap.output == 'cheap'
        assert primary['n'] == 1
        assert cheap['n'] == 1
        assert replay.invoked == []

    async def test_string_default_model_gets_unsuffixed_step_name(self) -> None:
        agent = Agent('test', name='strdef', capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            await agent.run('hi')

        assert 'strdef__model.request' in ctx.step_names
        assert not any(name.endswith('.test') for name in ctx.step_names)


class TestRuntimeToolsets:
    @pytest.mark.parametrize('kind', ['function', 'mcp', 'dynamic'])
    async def test_runtime_executing_toolset_rejected_inside_handler(self, kind: str) -> None:
        toolset: object
        if kind == 'function':
            late = FunctionToolset[object](id='late')

            @late.tool_plain
            def echo(value: str) -> str:  # pragma: no cover - rejected before it can run
                return value

            toolset = late
        elif kind == 'dynamic':
            toolset = DynamicToolset[object](lambda ctx: FunctionToolset[object](id='inner'), id='late')
        else:
            toolset = FakeMCPToolset(id='late')

        agent = Agent(_text_model(), name='a', capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(UserError, match=r'cannot be passed to `run\(toolsets=...\)` at runtime'):
                await agent.run('hi', toolsets=[toolset])

    async def test_non_executing_runtime_toolset_allowed_inside_handler(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[RestateDurability()])
        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('hi', toolsets=[ExternalToolset[object](tool_defs=[])])
        assert result.output == 'ok'


class TestNonWrappedLeaf:
    async def test_external_toolset_passes_through_unwrapped(self) -> None:
        external = ExternalToolset[object](tool_defs=[])
        agent = Agent(_text_model(), name='a', toolsets=[external], capabilities=[RestateDurability()])
        assert any(leaf is external for leaf in agent.toolsets)


class TestBindingErrors:
    async def test_unnamed_agent_raises(self) -> None:
        with pytest.raises(UserError, match='unique `name`'):
            Agent(_text_model(), capabilities=[RestateDurability()])

    async def test_duplicate_toolset_ids_raise(self) -> None:
        first = FunctionToolset[object](id='dup')

        @first.tool_plain
        def echo(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value

        second = FunctionToolset[object](id='dup')

        @second.tool_plain
        def shout(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value.upper()

        with pytest.raises(UserError, match='same `id`'):
            Agent(_text_model(), name='a', toolsets=[first, second], capabilities=[RestateDurability()])

    async def test_idless_leaf_function_toolset_raises(self) -> None:
        toolset = FunctionToolset[object]()

        @toolset.tool_plain
        def echo(value: str) -> str:  # pragma: no cover - never invoked, only the wrap check runs
            return value

        with pytest.raises(UserError, match='need to have a unique `id`'):
            Agent(_text_model(), name='a', toolsets=[toolset], capabilities=[RestateDurability()])

    async def test_capability_name_overrides_agent_name(self) -> None:
        agent = Agent(_text_model(), name='a', capabilities=[RestateDurability(name='custom')])
        ctx = FakeRestateContext()
        with restate_context(ctx):
            await agent.run('hi')
        assert ctx.step_names[0] == 'custom__model.request'


class TestEnqueueGuard:
    async def test_enqueue_inside_a_journaled_tool_raises(self) -> None:
        toolset = FunctionToolset[object](id='tools')

        @toolset.tool
        def act(run_ctx: RunContext[object]) -> str:
            run_ctx.enqueue('later')
            return 'ok'  # pragma: no cover - `enqueue` raises first

        agent = Agent(
            _tool_then_done_model('act', {}), name='a', toolsets=[toolset], capabilities=[RestateDurability()]
        )
        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(UserError, match='enqueue'):
                await agent.run('go')

    async def test_enqueue_from_model_event_handler_raises(self) -> None:
        # Model events are delivered to the handler inside the model-request step, so an enqueue
        # there would be dropped on replay and must raise.
        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for _ in stream:
                pass
            run_ctx.enqueue('later')

        agent = Agent(_text_model(), name='a', capabilities=[RestateDurability(event_stream_handler=handler)])
        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(UserError, match='enqueue'):
                await agent.run('hi')

    async def test_enqueue_on_agent_event_raises(self) -> None:
        # Agent-level events (here a `FunctionToolCallEvent`) are dispatched to the handler in their
        # own journaled step, so an enqueue there must raise too.
        async def handler(run_ctx: RunContext[object], stream: AsyncIterable[AgentStreamEvent]) -> None:
            async for event in stream:
                if isinstance(event, FunctionToolCallEvent):
                    run_ctx.enqueue('later')

        async def stream_fn(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str | DeltaToolCalls]:
            if len(messages) == 1:
                yield {0: DeltaToolCall(name='greet', json_args='{}')}
            else:  # pragma: no cover - the enqueue fails the run before a second turn
                yield 'done'

        toolset = FunctionToolset[object](id='tools')

        @toolset.tool_plain
        def greet() -> str:  # pragma: no cover - the enqueue fails the run before the tool runs
            return 'hello'

        agent = Agent(
            FunctionModel(stream_function=stream_fn, model_name='fn'),
            name='a',
            toolsets=[toolset],
            capabilities=[RestateDurability(event_stream_handler=handler)],
        )
        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(UserError, match='enqueue'):
                await agent.run('hi')


class TestEventStreamHandler:
    async def test_handler_events_are_journaled(self) -> None:
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
            name='ev',
            toolsets=[toolset],
            capabilities=[RestateDurability(event_stream_handler=handler)],
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'done'
        assert any(isinstance(event, FunctionToolCallEvent) for event in events)
        assert 'ev__event_stream_handler' in ctx.step_names


class TestCancelSuspendedResponse:
    async def test_cancel_suspended_response_is_journaled(self) -> None:
        cancelled: list[ModelResponse] = []

        class CancellableModel(FunctionModel):
            async def cancel_suspended_response(self, response: ModelResponse) -> None:
                cancelled.append(response)

        def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if not any(m.parts and getattr(m, 'state', None) == 'suspended' for m in messages):
                return ModelResponse(parts=[TextPart(content='partial')], state='suspended')
            raise RuntimeError('continuation failed')

        agent = Agent(CancellableModel(fn, model_name='fn'), name='a', capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            with pytest.raises(RuntimeError, match='continuation failed'):
                await agent.run('hi')

        assert len(cancelled) == 1
        assert cancelled[0].state == 'suspended'
        assert 'a__model.cancel_suspended_response' in ctx.step_names


class TestRepeatedStepNames:
    async def test_two_runs_in_one_handler_keep_separate_journal_slots(self) -> None:
        # A single handler that runs the agent twice: the second run's model step reuses the same
        # label, and the positional journal keeps a separate slot for it by encounter order.
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='a', capabilities=[RestateDurability()])

        ctx = FakeRestateContext()
        with restate_context(ctx):
            first = await agent.run('hi')
            second = await agent.run('hi again')

        assert first.output == second.output == 'ok'
        assert ctx.step_names == ['a__model.request', 'a__model.request']

        replay = ctx.replay()
        with restate_context(replay):
            await agent.run('hi')
            await agent.run('hi again')

        assert counter['calls'] == 2
        assert replay.invoked == []


class TestCheckpointFormat:
    async def test_capability_operation_name_and_replay_are_pinned(self) -> None:
        operation_calls: list[int] = []
        model_calls = {'calls': 0}
        agent = Agent(
            _text_model(model_calls),
            name='compat',
            capabilities=[_CountingOperation(operation_calls), RestateDurability()],
        )

        ctx = FakeRestateContext()
        with restate_context(ctx):
            await agent.run('hi')

        assert ctx.step_names == ['compat__capability__counter.increment', 'compat__model.request']
        assert operation_calls == [1]

        replay = ctx.replay()
        with restate_context(replay):
            await agent.run('hi')

        assert replay.step_names == ctx.step_names
        assert replay.invoked == []
        assert operation_calls == [1]
        assert model_calls['calls'] == 1

    async def test_hand_written_model_request_payload_replays(self) -> None:
        counter = {'calls': 0}
        agent = Agent(_text_model(counter), name='gold', capabilities=[RestateDurability()])

        # A hand-authored journal entry pins the persistence format for a model request.
        payload = (
            b'{"parts": [{"content": "golden-response", "part_kind": "text"}], "model_name": "fn", "kind": "response"}'
        )
        ctx = FakeRestateContext(journal=[Entry('gold__model.request', payload)])
        with restate_context(ctx):
            result = await agent.run('hi')

        assert result.output == 'golden-response'
        assert counter['calls'] == 0
