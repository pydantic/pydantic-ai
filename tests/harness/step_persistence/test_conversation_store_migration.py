"""The documented recipe for moving `StepPersistence` sessions to a core `ConversationStore`."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from pydantic_ai import Agent, DeferredToolRequests, RunContext
from pydantic_ai.conversation_store import InMemoryConversationStore
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteSessionStore
from tests.harness._docs_examples import python_blocks


def _recipe(path: str) -> str:
    (block,) = [example for example in python_blocks(path) if 'async def move_heads(' in example.source]
    return block.source


async def _call(name: str, *args: object, **kwargs: object) -> None:
    namespace: dict[str, object] = {}
    exec(_recipe('docs/harness/step-persistence.md'), namespace)
    function = namespace[name]
    assert callable(function)
    result = function(*args, **kwargs)
    assert inspect.isawaitable(result)
    await result


class TestStepPersistenceSessionMigration:
    def test_docs_and_readme_share_the_recipe(self) -> None:
        assert _recipe('docs/harness/step-persistence.md') == _recipe(
            'src/pydantic_ai_harness/pydantic_ai_harness/step_persistence/README.md'
        )

    async def test_move_conversation_copies_the_newest_snapshot(self) -> None:
        steps = InMemoryStepStore()
        agent = Agent(TestModel(), capabilities=[StepPersistence(store=steps)])
        first = await agent.run('hello', conversation_id='chat')
        second = await agent.run('again', message_history=first.all_messages(), conversation_id='chat')
        conversations = InMemoryConversationStore()

        await _call('move_conversation', steps, conversations, 'chat')
        moved = await conversations.load('chat')
        assert moved is not None
        assert moved.messages == second.all_messages()

        third = await agent.run('and again', conversation=moved)
        await conversations.save(third)
        await _call('move_conversation', steps, conversations, 'chat')
        reloaded = await conversations.load('chat')
        assert reloaded is not None
        assert reloaded.messages == third.all_messages()

    async def test_move_conversation_skips_delegate_runs_sharing_the_id(self) -> None:
        steps = InMemoryStepStore()
        delegate = Agent(TestModel(), capabilities=[StepPersistence(store=steps, agent_name='delegate')])
        orchestrator = Agent(TestModel(), capabilities=[StepPersistence(store=steps, agent_name='orchestrator')])

        @orchestrator.tool
        async def ask(ctx: RunContext[object], question: str) -> str:
            return (await delegate.run(question, conversation_id=ctx.conversation_id)).output

        result = await orchestrator.run('investigate', conversation_id='chat')
        assert len(await steps.list_runs(conversation_id='chat')) == 2
        conversations = InMemoryConversationStore()

        await _call('move_conversation', steps, conversations, 'chat')
        moved = await conversations.load('chat')
        assert moved is not None
        assert moved.messages == result.all_messages()

    async def test_move_conversation_refuses_a_session_waiting_on_an_approval(self) -> None:
        def ask_to_delete(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart('delete_file', {'path': 'notes.txt'})])

        steps = InMemoryStepStore()
        agent = Agent(
            FunctionModel(ask_to_delete),
            output_type=[str, DeferredToolRequests],
            capabilities=[StepPersistence(store=steps)],
        )

        @agent.tool_plain(requires_approval=True)
        def delete_file(path: str) -> str:
            return f'deleted {path}'  # pragma: no cover

        plain = Agent(TestModel(), capabilities=[StepPersistence(store=steps)])
        first = await plain.run('list files', conversation_id='chat')
        paused = await agent.run('clean up', message_history=first.all_messages(), conversation_id='chat')
        assert isinstance(paused.output, DeferredToolRequests)
        conversations = InMemoryConversationStore()

        with pytest.raises(ValueError, match='no complete snapshot holds'):
            await _call('move_conversation', steps, conversations, 'chat')
        assert await conversations.load('chat') is None

    async def test_move_conversation_records_the_scope(self) -> None:
        steps = InMemoryStepStore()
        result = await Agent(TestModel(), capabilities=[StepPersistence(store=steps)]).run('hi', conversation_id='c')
        conversations = InMemoryConversationStore()

        await _call('move_conversation', steps, conversations, 'c', scope='user-1')

        assert await conversations.load('c', scope='user-2') is None
        moved = await conversations.load('c', scope='user-1')
        assert moved is not None
        assert moved.messages == result.all_messages()

    async def test_move_heads_copies_every_head_under_the_scope(self, tmp_path: Path) -> None:
        agent = Agent(TestModel())
        heads = SqliteSessionStore(database=tmp_path / 'sessions.db')
        runs = {name: await agent.run(name) for name in ('one', 'two')}
        for name, result in runs.items():
            await heads.save(summary=ConversationSummary(id=name, workspace='.'), messages=result.all_messages())
        conversations = InMemoryConversationStore()

        await _call('move_heads', heads, conversations, scope='user-1')
        await _call('move_heads', heads, conversations, scope='user-1')

        for name, result in runs.items():
            assert await conversations.load(name, scope='user-2') is None
            moved = await conversations.load(name, scope='user-1')
            assert moved is not None
            assert moved.messages == result.all_messages()

        moved = await conversations.load('one', scope='user-1')
        assert moved is not None
        continued = await agent.run('and again', conversation=moved)
        await conversations.save(continued, scope='user-1')
        reloaded = await conversations.load('one', scope='user-1')
        assert reloaded is not None
        assert reloaded.messages == continued.all_messages()
        assert len(reloaded.messages) > len(runs['one'].all_messages())
