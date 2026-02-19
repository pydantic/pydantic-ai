from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from typing_extensions import TypedDict

from pydantic_ai.ext.mem0 import Mem0Toolset

from .._inline_snapshot import snapshot


class AddCall(TypedDict):
    messages: list[dict[str, str]]
    user_id: str | None
    agent_id: str | None
    metadata: dict[str, Any] | None


class SearchCall(TypedDict):
    query: str
    user_id: str | None
    agent_id: str | None
    limit: int


def _add_calls_factory() -> list[AddCall]:
    return []


def _search_calls_factory() -> list[SearchCall]:
    return []


@dataclass
class SimulatedMem0Client:
    add_calls: list[AddCall] = field(default_factory=_add_calls_factory)
    search_calls: list[SearchCall] = field(default_factory=_search_calls_factory)

    def add(
        self,
        messages: list[dict[str, str]],
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        self.add_calls.append({'messages': messages, 'user_id': user_id, 'agent_id': agent_id, 'metadata': metadata})
        return {'id': 'mem_1', 'stored': messages, 'metadata': metadata}

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        limit: int = 5,
    ) -> Any:
        self.search_calls.append({'query': query, 'user_id': user_id, 'agent_id': agent_id, 'limit': limit})
        return [{'id': f'mem_{i}', 'memory': f'match:{query}:{i}'} for i in range(limit)]


def test_mem0_toolset_has_tools():
    """
    Function that checks if the toolset created has the correct tools.
    """
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client)

    assert sorted(toolset.tools.keys()) == snapshot(['save_memory', 'search_memory'])


def test_mem0_save_memory_calls_add():
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client, user_id='u1', agent_id='a1')

    save_tool = toolset.tools['save_memory']
    result = save_tool.function(content='hello', metadata={'k': 'v'})  # type: ignore

    assert result == snapshot(
        {
            'result': {
                'id': 'mem_1',
                'stored': [{'role': 'user', 'content': 'hello'}],
                'metadata': {'k': 'v'},
            }
        }
    )
    assert client.add_calls == snapshot(
        [
            {
                'messages': [{'role': 'user', 'content': 'hello'}],
                'user_id': 'u1',
                'agent_id': 'a1',
                'metadata': {'k': 'v'},
            }
        ]
    )


def test_mem0_search_memory_calls_search_default_limit():
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client, user_id='u2', agent_id='a2')

    search_tool = toolset.tools['search_memory']
    result = search_tool.function(query='postgres')  # type: ignore

    assert result == snapshot(
        {
            'results': [
                {'id': 'mem_0', 'memory': 'match:postgres:0'},
                {'id': 'mem_1', 'memory': 'match:postgres:1'},
                {'id': 'mem_2', 'memory': 'match:postgres:2'},
                {'id': 'mem_3', 'memory': 'match:postgres:3'},
                {'id': 'mem_4', 'memory': 'match:postgres:4'},
            ]
        }
    )
    assert client.search_calls == snapshot([{'query': 'postgres', 'user_id': 'u2', 'agent_id': 'a2', 'limit': 5}])


def test_mem0_search_memory_calls_search_custom_limit():
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client, user_id='u3', agent_id='a3')

    search_tool = toolset.tools['search_memory']
    result = search_tool.function(query='cats', limit=2)  # type: ignore

    assert result == snapshot(
        {'results': [{'id': 'mem_0', 'memory': 'match:cats:0'}, {'id': 'mem_1', 'memory': 'match:cats:1'}]}
    )
    assert client.search_calls == snapshot([{'query': 'cats', 'user_id': 'u3', 'agent_id': 'a3', 'limit': 2}])
