from __future__ import annotations

import typing
from typing import Protocol, cast

import pydantic_ai
import pydantic_ai.agent
from pydantic_ai import _agent_graph
from pydantic_ai._agent_graph import graph as _graph


class _NodeClass(Protocol):
    run: object


def test_agent_graph_compatibility_surface() -> None:
    node_classes = (
        _agent_graph.UserPromptNode,
        _agent_graph.ModelRequestNode,
        _agent_graph.CallToolsNode,
    )

    assert node_classes == (
        pydantic_ai.UserPromptNode,
        pydantic_ai.ModelRequestNode,
        pydantic_ai.CallToolsNode,
    )
    assert node_classes == (
        pydantic_ai.agent.UserPromptNode,
        pydantic_ai.agent.ModelRequestNode,
        pydantic_ai.agent.CallToolsNode,
    )
    assert pydantic_ai.EndStrategy is _agent_graph.EndStrategy
    assert pydantic_ai.agent.EndStrategy is _agent_graph.EndStrategy
    assert pydantic_ai.capture_run_messages is _agent_graph.capture_run_messages
    assert pydantic_ai.agent.capture_run_messages is _agent_graph.capture_run_messages
    assert [node_class.get_node_id() for node_class in node_classes] == [
        'UserPromptNode',
        'ModelRequestNode',
        'CallToolsNode',
    ]


def test_agent_graph_node_type_hints_are_resolvable() -> None:
    def assert_type_hints_resolve(obj: object) -> None:
        assert typing.get_type_hints(obj)

    for node_class in (
        _agent_graph.UserPromptNode,
        _agent_graph.ModelRequestNode,
        _agent_graph.CallToolsNode,
    ):
        assert_type_hints_resolve(cast(_NodeClass, node_class).run)

    assert_type_hints_resolve(_graph.build_agent_graph)
