"""Session ownership tests: transport recordings cannot assert resource/task lifetimes or races."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, Literal, assert_type

import anyio
import pytest

from pydantic_ai import Agent, Conversation, RunCancelled, SessionStateTypeAdapter, UserError
from pydantic_ai.agent import WrapperAgent
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelResolutionContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRun, AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, RunContext

READINESS_WAIT_TIMEOUT = 10


async def test_session_continues_history_and_usage_without_mutating_snapshots():
    agent = Agent(TestModel())
    original = Conversation()
    async with agent.session(conversation=original) as session:
        first = await session.run('first')
        checkpoint = session.state
        second = await session.run('second')
        assert first.run_id != second.run_id
        assert first.conversation_id == second.conversation_id == original.conversation_id
        assert len(second.all_messages()) == 4
        assert session.conversation.usage.requests == 2
        assert checkpoint.conversation.usage.requests == 1
        assert len(checkpoint.conversation.messages) == 2
        assert first.usage.requests == 1
    assert original.messages == []
    assert original.usage.requests == 0


async def test_session_idle_input_round_trip_and_stale_run_handle():
    agent = Agent(TestModel())
    async with agent.session() as session:
        enqueue_id = session.enqueue('idle input')
        state = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(session.state))
    async with agent.session(state=state) as restored:
        async with restored.iter('start') as run:
            async for _ in run:
                pass
        assert run.result is not None
        assert restored.state.pending == []
        assert any(
            isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) and part.content == 'idle input' for part in message.parts)
            for message in run.result.all_messages()
        )
        with pytest.raises(UserError, match='run has ended'):
            run.enqueue('stale')
        restored.enqueue('next')
        await restored.run()
    assert state.pending[0].enqueue_id == enqueue_id
    assert len(state.pending) == 1


async def test_session_cancelled_run_keeps_pending_input_for_next_run():
    agent = Agent(TestModel())
    async with agent.session() as session:
        with pytest.raises(RunCancelled):
            async with session.iter('first') as run:
                run.enqueue('later', priority='when_idle')
                run.cancel()
                async for _ in run:
                    pass
        assert len(session.state.pending) == 1
        result = await session.run('continue')
        assert session.state.pending == []
        assert result.output == 'success (no tool calls)'


async def test_session_rejects_concurrent_writer_and_active_checkpoint():
    agent = Agent(TestModel())
    async with agent.session() as session:
        async with session.iter('first'):
            with pytest.raises(UserError, match='one run at a time'):
                await session.run('second')
            with pytest.raises(UserError, match='unfinished run'):
                agent.session(state=session.state)
        await session.run('third')


async def test_session_streaming_and_events_use_shared_history():
    agent = Agent(TestModel(custom_output_text='hello'))
    async with agent.session() as session:
        async with session.run_stream('first') as result:
            assert await result.get_output() == 'hello'
        async with session.run_stream_events('second') as events:
            async for _ in events:
                pass
        third = await session.run('third')
        assert len(third.all_messages()) == 6
        assert session.conversation.usage.requests == 3


async def test_session_nested_agent_does_not_inherit_session():
    nested = Agent(TestModel())
    agent = Agent(TestModel())

    @agent.tool_plain
    async def nested_run() -> str:
        result = await nested.run('nested')
        assert result.conversation_id != session.conversation.conversation_id
        return result.output

    async with agent.session() as session:
        await session.run('outer')
        assert session.conversation.usage.requests == 2
        assert len(session.conversation.messages) == 4


async def test_session_deferred_tools_survive_checkpoint():
    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def privileged() -> str:
        return 'approved'

    async with agent.session() as session:
        first = await session.run('do it')
        assert isinstance(first.output, DeferredToolRequests)
        state = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(session.state))
    assert state.conversation.deferred_tool_requests is not None
    results = state.conversation.deferred_tool_requests.build_results(approve_all=True)
    async with agent.session(state=state) as session:
        await session.run(deferred_tool_results=results)
        assert session.conversation.deferred_tool_requests is None


async def test_session_requires_entry_and_rejects_reentry_and_state_conflicts():
    agent = Agent(TestModel())
    session = agent.session()
    with pytest.raises(UserError, match='async with'):
        await session.run('no entry')
    async with session:
        with pytest.raises(UserError, match='session owns'):
            await session.run('conflict', conversation=Conversation())
        assert session.enqueue() is None
    with pytest.raises(UserError, match='only be entered once'):
        async with session:
            pass
    with pytest.raises(UserError, match='session has closed'):
        session.enqueue('closed')
    with pytest.raises(UserError, match='not both'):
        agent.session(state=session.state, conversation=Conversation())


async def test_session_failed_model_does_not_poison_next_run():
    async def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise ValueError('model failed')

    agent = Agent(TestModel())
    async with agent.session() as session:
        with pytest.raises(ValueError, match='model failed'):
            await session.run('first', model=FunctionModel(fail))
        assert len(session.conversation.messages) == 1
        result = await session.run('recover')
        assert result.output == 'success (no tool calls)'


async def test_session_model_resource_lives_across_runs_in_one_owner_task():
    events: list[tuple[str, int]] = []

    class ResourceModel(TestModel):
        async def __aenter__(self):
            self.context = self.resource()
            await self.context.__aenter__()
            return self

        async def __aexit__(self, *args: Any):
            return await self.context.__aexit__(*args)

        @asynccontextmanager
        async def resource(self) -> AsyncGenerator[None]:
            with anyio.CancelScope():
                events.append(('enter', anyio.get_current_task().id))
                try:
                    yield
                finally:
                    events.append(('exit', anyio.get_current_task().id))

    agent = Agent(ResourceModel())
    async with agent.session() as session:
        await session.run('first')
        async with session.run_stream_events('second') as stream:
            async for _ in stream:
                pass
        assert len(events) == 1
    assert events[0][0] == 'enter'
    assert events[1] == ('exit', events[0][1])


async def test_session_close_cancels_and_drains_running_child():
    started = anyio.Event()
    finished = anyio.Event()

    async def blocked(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            finished.set()
        return ModelResponse([TextPart('unreachable')])

    async def run() -> None:
        try:
            await session.run('wait')
        except anyio.get_cancelled_exc_class():
            pass

    async with anyio.create_task_group() as group:
        async with Agent(FunctionModel(blocked)).session() as session:
            group.start_soon(run)
            with anyio.fail_after(10):
                await started.wait()
        assert finished.is_set()
        assert session.state.active_run_id is None


async def test_session_cancel_targets_only_current_run():
    agent = Agent(TestModel())
    async with agent.session() as session:
        session.cancel()
        with pytest.raises(RunCancelled):
            async with session.iter('cancel') as run:
                session.cancel()
                async for _ in run:
                    pass
        session.cancel()
        assert (await session.run('continue')).output == 'success (no tool calls)'


async def test_session_failed_setup_retains_input_and_releases_writer():
    class BrokenModel(TestModel):
        async def __aenter__(self):
            raise ValueError('entry failed')

    agent = Agent(TestModel())
    async with agent.session() as session:
        enqueue_id = session.enqueue('keep this')
        with pytest.raises(ValueError, match='entry failed'):
            await session.run('first', model=BrokenModel())
        assert session.state.pending[0].enqueue_id == enqueue_id
        await session.run('second')
        assert session.state.pending == []


async def test_session_body_exception_is_not_wrapped_in_exception_group():
    with pytest.raises(ValueError, match='body error'):
        async with Agent(TestModel()).session() as session:
            await session.run('first')
            raise ValueError('body error')


async def test_session_submission_after_final_drain_belongs_to_next_run():
    class FollowUp(AbstractCapability[object]):
        async def after_run(self, ctx: RunContext[object], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
            if ctx.usage.requests == 1:
                session.enqueue('next run, not this one')
            return result

    agent = Agent(TestModel(), capabilities=[FollowUp()])
    async with agent.session() as session:
        first = await session.run('first')
        assert first.usage.requests == 1
        assert len(session.state.pending) == 1
        second = await session.run('second')
        assert second.usage.requests == 2
        assert session.state.pending == []


def test_session_pending_messages_use_normal_history_serialization():
    session = Agent(TestModel()).session()
    session.enqueue(ToolReturnPart('tool', b'\xff', tool_call_id='call'))
    encoded = json.loads(SessionStateTypeAdapter.dump_json(session.state))
    assert encoded['pending'][0]['messages'][0]['parts'][0]['content'] == '_w=='
    redacted = json.loads(
        SessionStateTypeAdapter.dump_json(
            session.state,
            exclude={'pending': {'__all__': {'messages': {'__all__': {'parts': {'__all__': {'content'}}}}}}},
        )
    )
    assert 'content' not in redacted['pending'][0]['messages'][0]['parts'][0]


async def test_session_reuses_inferred_model_but_rechecks_dependency_based_resolution(monkeypatch: pytest.MonkeyPatch):
    inferred: list[str] = []
    resolutions: list[str] = []

    def infer(model_id: str):
        inferred.append(model_id)
        return TestModel(custom_output_text='default')

    class Resolve(AbstractCapability[str]):
        async def resolve_model_id(self, ctx: ModelResolutionContext[str], *, model_id: str) -> TestModel | None:
            resolutions.append(ctx.deps)
            if ctx.deps == 'override':
                return TestModel(custom_output_text='override')
            return None

    monkeypatch.setattr('pydantic_ai.models.infer_model', infer)
    agent = Agent(deps_type=str, capabilities=[Resolve()])
    async with agent.session(model='test', deps='default') as session:
        assert (await session.run('first')).output == 'default'
        assert (await session.run('second')).output == 'default'
        assert (await session.run('third', deps='override')).output == 'override'
    assert inferred == ['test']
    assert resolutions == ['default', 'default', 'override']


async def test_session_external_cancellation_finishes_request_cleanup_before_model_close():
    tasks_before = asyncio.all_tasks()
    started = anyio.Event()
    cleanup_started = anyio.Event()
    allow_cleanup = anyio.Event()
    order: list[str] = []

    async def blocked(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        try:
            await anyio.sleep_forever()
        finally:
            with anyio.CancelScope(shield=True):
                cleanup_started.set()
                await allow_cleanup.wait()
                order.append('request cleaned up')
        return ModelResponse([TextPart('unreachable')])

    class ResourceModel(FunctionModel):
        async def __aexit__(self, *args: Any):
            order.append('model closed')

    async def cancel() -> None:
        await started.wait()
        outer.cancel()
        with anyio.CancelScope(shield=True):
            try:
                await cleanup_started.wait()
                assert order == []
            finally:
                allow_cleanup.set()

    session = Agent(ResourceModel(blocked)).session()
    with anyio.fail_after(READINESS_WAIT_TIMEOUT):
        with anyio.CancelScope() as outer:
            async with anyio.create_task_group() as group:
                group.start_soon(cancel)
                async with session:
                    await session.run('wait')
    assert order == ['request cleaned up', 'model closed']
    assert session.state.active_run_id is None
    assert asyncio.all_tasks() == tasks_before


async def test_session_wrapper_helper_cannot_consume_the_outer_session_binding():
    helper = Agent(TestModel())
    helper_results: list[AgentRunResult[str]] = []

    class HelpfulWrapper(WrapperAgent[None, str]):
        @asynccontextmanager
        async def iter(self, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentRun[None, str]]:
            helper_results.append(await helper.run('helper'))
            async with self.wrapped.iter(*args, **kwargs) as run:
                yield run

    agent = HelpfulWrapper(Agent(TestModel()))
    async with agent.session() as session:
        session.enqueue('only for outer')
        async with session.iter('outer') as run:
            # Checking the live checkpoint proves the core bound, not just the final history copy.
            assert session.state.active_run_id == run.ctx.state.run_id
            async for _ in run:
                pass
        assert session.state.pending == []
        assert session.conversation.usage.requests == 1
        prompts = [
            part.content
            for message in session.conversation.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, UserPromptPart)
        ]
        assert prompts == ['outer', 'only for outer']
        assert helper_results[0].conversation_id != session.conversation.conversation_id
        helper_request = helper_results[0].new_messages()[0]
        assert isinstance(helper_request, ModelRequest)
        assert len(helper_request.parts) == 1
        assert isinstance(helper_request.parts[0], UserPromptPart)
        assert helper_request.parts[0].content == 'helper'


async def test_session_failed_entry_preserves_deferred_approval_for_checkpoint_resume():
    calls: list[str] = []

    class BrokenModel(TestModel):
        async def __aenter__(self):
            raise ValueError('entry failed')

    agent = Agent(TestModel(), output_type=[str, DeferredToolRequests])

    @agent.tool_plain(requires_approval=True)
    def privileged() -> str:
        calls.append('executed')
        return 'approved'

    async with agent.session() as session:
        first = await session.run('request approval')
        assert isinstance(first.output, DeferredToolRequests)
        with pytest.raises(ValueError, match='entry failed'):
            await session.run(deferred_tool_results=first.output.build_results(approve_all=True), model=BrokenModel())
        state = SessionStateTypeAdapter.validate_json(SessionStateTypeAdapter.dump_json(session.state))
        assert state.conversation.deferred_tool_requests == first.output
        assert calls == []
    async with agent.session(state=state) as restored:
        assert restored.conversation.deferred_tool_requests is not None
        await restored.run(
            deferred_tool_results=restored.conversation.deferred_tool_requests.build_results(approve_all=True)
        )
        assert calls == ['executed']
        assert restored.conversation.deferred_tool_requests is None


@pytest.mark.parametrize('entry', ['run', 'iter', 'stream', 'events'])
async def test_session_bound_dependencies_and_explicit_none_override(entry: Literal['run', 'iter', 'stream', 'events']):
    seen: list[str | None] = []
    agent = Agent(TestModel(), deps_type=str | None)

    @agent.instructions
    def record_deps(ctx: RunContext[str | None]) -> str:
        seen.append(ctx.deps)
        return 'Be brief.'

    async with agent.session(deps='bound') as session:
        if entry == 'run':
            assert_type((await session.run('bound')).output, str)
            assert_type((await session.run('override', deps=None, output_type=int)).output, int)
        elif entry == 'iter':
            async with session.iter('bound') as run:
                assert_type(run, AgentRun[str | None, str])
                async for _ in run:
                    pass
            async with session.iter('override', deps=None, output_type=int) as run_int:
                assert_type(run_int, AgentRun[str | None, int])
                async for _ in run_int:
                    pass
        elif entry == 'stream':
            async with session.run_stream('bound') as stream:
                assert_type(await stream.get_output(), str)
            async with session.run_stream('override', deps=None, output_type=int) as stream_int:
                assert_type(await stream_int.get_output(), int)
        else:
            async with session.run_stream_events('bound') as events:
                async for _ in events:
                    pass
                assert events.result is not None
                assert_type(events.result.output, str)
            async with session.run_stream_events('override', deps=None, output_type=int) as events_int:
                async for _ in events_int:
                    pass
                assert events_int.result is not None
                assert_type(events_int.result.output, int)
    assert seen == ['bound', None]


async def test_session_rejects_wrapper_that_discards_session_conversation():
    class DetachedWrapper(WrapperAgent[None, str]):
        @asynccontextmanager
        async def iter(self, *args: Any, **kwargs: Any) -> AsyncGenerator[AgentRun[None, str]]:
            async with self.wrapped.iter('detached') as run:
                yield run

    async with DetachedWrapper(Agent(TestModel())).session() as session:
        session.enqueue('not lost')
        with pytest.raises(UserError, match='did not delegate'):
            await session.run('outer')
        assert session.state.active_run_id is None
        assert len(session.state.pending) == 1
        assert session.conversation.messages == []


async def test_completed_session_result_cannot_mutate_history_or_operation_ledger():
    agent = Agent(TestModel())

    @agent.tool_plain
    def value() -> dict[str, str]:
        return {'value': 'original'}

    async with agent.session() as session:
        result = await session.run('first')
        checkpoint = session.state
        for message in result.all_messages():
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    part.tool_call_id = 'changed after completion'
        assert session.state == checkpoint
