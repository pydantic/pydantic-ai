"""Session ownership tests: transport recordings cannot assert resource/task lifetimes or races."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import anyio
import pytest

from pydantic_ai import Agent, Conversation, RunCancelled, SessionStateTypeAdapter, UserError
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelResolutionContext
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, RunContext


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
