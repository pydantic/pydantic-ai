"""Tests for the Exa search tools."""

from __future__ import annotations

from exa_py import AsyncExa

from pydantic_ai.common_tools.exa import (
    ExaAnswerTool,
    ExaFindSimilarTool,
    ExaGetContentsTool,
    ExaSearchTool,
    ExaToolset,
    exa_answer_tool,
    exa_find_similar_tool,
    exa_get_contents_tool,
    exa_search_tool,
)


def _get_client(tool_obj: object) -> AsyncExa:
    """Pull the AsyncExa client out of the underlying tool dataclass captured by `Tool.function`."""
    function = getattr(tool_obj, 'function', None)
    if function is not None:
        bound_self = getattr(function, '__self__', None)
        assert isinstance(bound_self, (ExaSearchTool, ExaFindSimilarTool, ExaGetContentsTool, ExaAnswerTool))
        return bound_self.client
    assert isinstance(tool_obj, (ExaSearchTool, ExaFindSimilarTool, ExaGetContentsTool, ExaAnswerTool))
    return tool_obj.client


def test_factory_sets_integration_header(exa_api_key: str):
    """Each factory attaches the pydantic-ai attribution header when it builds the client."""
    for tool in (
        exa_search_tool(exa_api_key),
        exa_find_similar_tool(exa_api_key),
        exa_get_contents_tool(exa_api_key),
        exa_answer_tool(exa_api_key),
    ):
        client = _get_client(tool)
        assert client.headers['x-exa-integration'] == 'pydantic-ai'


def test_factory_preserves_user_provided_client(exa_api_key: str):
    """A user-supplied client is used as-is; the factory does not mutate its headers."""
    client = AsyncExa(api_key=exa_api_key)
    assert 'x-exa-integration' not in client.headers

    tool = exa_search_tool(client=client)
    assert _get_client(tool) is client
    assert 'x-exa-integration' not in client.headers


def test_toolset_sets_integration_header(exa_api_key: str):
    """The shared client inside ExaToolset is tagged with the pydantic-ai attribution header."""
    toolset = ExaToolset(exa_api_key)
    tools = list(toolset.tools.values())
    assert tools, 'ExaToolset should expose at least one tool by default'
    for tool in tools:
        assert _get_client(tool).headers['x-exa-integration'] == 'pydantic-ai'
