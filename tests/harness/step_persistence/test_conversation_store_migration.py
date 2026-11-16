"""The documented recipe for moving `StepPersistence` sessions to a core `ConversationStore`."""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.conversation_store import InMemoryConversationStore
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence
from pydantic_ai_harness.step_persistence.conversations import ConversationSummary, SqliteConversationStore

_ROOT = Path(__file__).parents[3]
_SURFACES = (
    _ROOT / 'docs' / 'harness' / 'step-persistence.md',
    _ROOT / 'src' / 'pydantic_ai_harness' / 'pydantic_ai_harness' / 'step_persistence' / 'README.md',
)


def _recipe(path: Path) -> str:
    section = path.read_text(encoding='utf-8').split('## Moving sessions to a `ConversationStore`', 1)[1]
    match = re.search(r'```python\n(.*?)```', section, flags=re.DOTALL)
    assert match
    return match.group(1)


def _recipe_function(name: str) -> Callable[..., object]:
    namespace: dict[str, object] = {}
    exec(_recipe(_SURFACES[0]), namespace)
    function = namespace[name]
    assert callable(function)
    return function


async def _call(function: Callable[..., object], *args: object) -> None:
    result = function(*args)
    assert inspect.isawaitable(result)
    await result


class TestStepPersistenceSessionMigration:
    def test_docs_and_readme_share_the_recipe(self) -> None:
        assert _recipe(_SURFACES[0]) == _recipe(_SURFACES[1])

    async def test_move_conversation_copies_the_newest_snapshot(self) -> None:
        steps = InMemoryStepStore()
        agent = Agent(TestModel(), capabilities=[StepPersistence(store=steps)])
        first = await agent.run('hello', conversation_id='chat')
        second = await agent.run('again', message_history=first.all_messages(), conversation_id='chat')
        move_conversation = _recipe_function('move_conversation')
        conversations = InMemoryConversationStore()

        await _call(move_conversation, steps, conversations, 'chat')
        moved = await conversations.load('chat')
        assert moved is not None
        assert moved.messages == second.all_messages()

        third = await agent.run('and again', conversation=moved)
        await conversations.save(third)
        await _call(move_conversation, steps, conversations, 'chat')
        reloaded = await conversations.load('chat')
        assert reloaded is not None
        assert reloaded.messages == third.all_messages()

    async def test_move_heads_copies_every_head_and_resumes(self, tmp_path: Path) -> None:
        agent = Agent(TestModel())
        heads = SqliteConversationStore(database=tmp_path / 'sessions.db')
        runs = {name: await agent.run(name) for name in ('one', 'two')}
        for name, result in runs.items():
            await heads.save(summary=ConversationSummary(id=name, workspace='.'), messages=result.all_messages())
        move_heads = _recipe_function('move_heads')
        conversations = InMemoryConversationStore()

        await _call(move_heads, heads, conversations)
        await _call(move_heads, heads, conversations)

        for name, result in runs.items():
            moved = await conversations.load(name)
            assert moved is not None
            assert moved.messages == result.all_messages()
