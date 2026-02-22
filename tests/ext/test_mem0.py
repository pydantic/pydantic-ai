from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from typing_extensions import TypedDict

from pydantic_ai import Agent
from pydantic_ai.exceptions import UserError
from pydantic_ai.ext.mem0 import Mem0Toolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from .._inline_snapshot import snapshot


class AddCall(TypedDict):
    messages: list[dict[str, str]]
    user_id: str | None
    agent_id: str | None
    run_id: str | None
    metadata: dict[str, Any] | None


class SearchCall(TypedDict):
    query: str
    user_id: str | None
    agent_id: str | None
    run_id: str | None
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
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        self.add_calls.append(
            {
                'messages': messages,
                'user_id': user_id,
                'agent_id': agent_id,
                'run_id': run_id,
                'metadata': metadata,
            }
        )
        return {'id': 'mem_1', 'stored': messages, 'metadata': metadata}

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 5,
    ) -> Any:
        self.search_calls.append(
            {
                'query': query,
                'user_id': user_id,
                'agent_id': agent_id,
                'run_id': run_id,
                'limit': limit,
            }
        )
        return [{'id': f'mem_{i}', 'memory': f'match:{query}:{i}'} for i in range(limit)]


def test_mem0_toolset_has_tools():
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client, user_id='u0')

    assert sorted(toolset.tools.keys()) == snapshot(['save_memory', 'search_memory'])


def test_mem0_toolset_requires_identifier():
    client = SimulatedMem0Client()
    with pytest.raises(UserError, match='requires at least one of user_id, agent_id, or run_id'):
        Mem0Toolset(client)


def test_mem0_save_memory_via_agent_runtime_calls_add():
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client, user_id='u1', agent_id='a1')

    called = False

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal called
        if not called:
            called = True
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='save_memory',
                        args={'content': 'hello', 'metadata': {'k': 'v'}},
                        tool_call_id='tool_call_1',
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent('test', toolsets=[toolset])
    result = agent.run_sync('store this', model=FunctionModel(model_fn))

    assert result.output == snapshot('done')
    assert client.add_calls == snapshot(
        [
            {
                'messages': [{'role': 'user', 'content': 'hello'}],
                'user_id': 'u1',
                'agent_id': 'a1',
                'run_id': None,
                'metadata': {'k': 'v'},
            }
        ]
    )


def test_mem0_search_memory_via_agent_runtime_calls_search_default_limit():
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client, user_id='u2', agent_id='a2')

    called = False

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal called
        if not called:
            called = True
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='search_memory',
                        args={'query': 'postgres'},
                        tool_call_id='tool_call_1',
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent('test', toolsets=[toolset])
    result = agent.run_sync('search', model=FunctionModel(model_fn))

    assert result.output == snapshot('done')
    assert client.search_calls == snapshot(
        [{'query': 'postgres', 'user_id': 'u2', 'agent_id': 'a2', 'run_id': None, 'limit': 5}]
    )


def test_mem0_search_memory_via_agent_runtime_calls_search_custom_limit():
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client, user_id='u3', agent_id='a3')

    called = False

    def model_fn(_messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        nonlocal called
        if not called:
            called = True
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name='search_memory',
                        args={'query': 'cats', 'limit': 2},
                        tool_call_id='tool_call_1',
                    )
                ]
            )
        return ModelResponse(parts=[TextPart('done')])

    agent = Agent('test', toolsets=[toolset])
    result = agent.run_sync('search', model=FunctionModel(model_fn))

    assert result.output == snapshot('done')
    assert client.search_calls == snapshot(
        [{'query': 'cats', 'user_id': 'u3', 'agent_id': 'a3', 'run_id': None, 'limit': 2}]
    )


def test_mem0_tools_schema_is_exposed_to_model():
    client = SimulatedMem0Client()
    toolset = Mem0Toolset(client, user_id='u0')

    test_model = TestModel()
    agent = Agent('test', toolsets=[toolset])

    agent.run_sync('hi', model=test_model)

    params = test_model.last_model_request_parameters
    assert params is not None  # makes Pylance/mypy happy

    tool_names = sorted(t.name for t in params.function_tools)
    assert tool_names == snapshot(['save_memory', 'search_memory'])

    tools = {t.name: t.parameters_json_schema for t in params.function_tools}
    assert tools['search_memory']['properties']['limit'] == snapshot(
        {'default': 5, 'description': 'Max number of results to return.', 'type': 'integer'}
    )
