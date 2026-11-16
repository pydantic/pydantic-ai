"""The documented recipe for moving `StepPersistence` sessions to a core `ConversationStore`."""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from pydantic_ai import Agent
from pydantic_ai.conversation_store import ConversationConflict, InMemoryConversationStore
from pydantic_ai.models.test import TestModel
from pydantic_ai_harness.step_persistence import InMemoryStepStore, StepPersistence

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


class TestStepPersistenceSessionMigration:
    def test_docs_and_readme_share_the_recipe(self) -> None:
        assert _recipe(_SURFACES[0]) == _recipe(_SURFACES[1])

    async def test_recipe_moves_each_conversation(self) -> None:
        steps = InMemoryStepStore()
        agent = Agent(TestModel(), capabilities=[StepPersistence(store=steps)])
        first = await agent.run('hello', conversation_id='chat')
        second = await agent.run('again', message_history=first.all_messages(), conversation_id='chat')
        other = await agent.run('elsewhere', conversation_id='other')

        namespace: dict[str, object] = {}
        exec(_recipe(_SURFACES[0]), namespace)
        move_all = namespace['move_all']
        assert callable(move_all)
        conversations = InMemoryConversationStore()
        moving = move_all(steps, conversations)
        assert inspect.isawaitable(moving)
        await moving

        moved = await conversations.load('chat')
        assert moved is not None
        assert moved.messages == second.all_messages()
        other_moved = await conversations.load('other')
        assert other_moved is not None
        assert other_moved.messages == other.all_messages()

        third = await agent.run('and again', conversation=moved)
        await conversations.save(third)
        reloaded = await conversations.load('chat')
        assert reloaded is not None
        assert reloaded.messages == third.all_messages()

        again = move_all(steps, conversations)
        assert inspect.isawaitable(again)
        with pytest.raises(ConversationConflict):
            await again
