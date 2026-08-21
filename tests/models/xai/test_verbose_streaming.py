"""Tests for xAI's `verbose_streaming` include option."""

from __future__ import annotations as _annotations

from typing import Any

import pytest

from pydantic_ai import (
    Agent,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    PartStartEvent,
    WebSearchTool,
)
from pydantic_ai.capabilities import NativeTool

from ..._inline_snapshot import snapshot
from ...conftest import try_import

with try_import() as imports_successful:
    from pydantic_ai.models.xai import XaiModel, XaiModelSettings
    from pydantic_ai.providers.xai import XaiProvider


pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='xai_sdk not installed'),
    pytest.mark.anyio,
    pytest.mark.vcr,
]

XAI_MODEL = 'grok-4-fast-reasoning'


async def test_xai_verbose_streaming_native_tool_lifecycle(allow_model_requests: None, xai_provider: XaiProvider):
    """Verbose streaming surfaces each server-side tool call as xAI's agentic loop makes it.

    The recording has the loop run `web_search` three times before answering, and the event order is
    the evidence the option does what it claims: the second call is announced before the first search
    has returned, so the caller sees the loop's progress rather than a batch of finished calls at the
    end. The final `ModelResponse` carries the same three call/return pairs, so opting in adds
    progress events without changing the resulting message.

    Cassette replay does not match request bodies, so this does not prove the include option went out
    on the wire; `test_xai_include_settings` asserts that separately.
    """
    agent = Agent(
        XaiModel(XAI_MODEL, provider=xai_provider),
        capabilities=[NativeTool(WebSearchTool())],
        model_settings=XaiModelSettings(xai_include_verbose_streaming=True),
    )

    events: list[Any] = []
    async with agent.iter(
        user_prompt='Search the web for the latest Pydantic AI release. Reply with just the version.'
    ) as agent_run:
        async for node in agent_run:
            if Agent.is_model_request_node(node) or Agent.is_call_tools_node(node):
                async with node.stream(agent_run.ctx) as request_stream:
                    async for event in request_stream:
                        events.append(event)

    assert agent_run.result is not None
    assert agent_run.result.output == snapshot('2.33.0')

    native_tool_events = [
        (event.part.part_kind, event.part.tool_name)
        for event in events
        if isinstance(event, PartStartEvent) and isinstance(event.part, (NativeToolCallPart, NativeToolReturnPart))
    ]
    assert native_tool_events == snapshot(
        [
            ('builtin-tool-call', 'web_search'),
            ('builtin-tool-call', 'web_search'),
            ('builtin-tool-return', 'web_search'),
            ('builtin-tool-call', 'web_search'),
            ('builtin-tool-return', 'web_search'),
            ('builtin-tool-return', 'web_search'),
        ]
    )

    response = agent_run.result.all_messages()[-1]
    assert isinstance(response, ModelResponse)
    assert [(call.tool_name, call.args) for call, _ in response.native_tool_calls] == snapshot(
        [
            ('web_search', {'query': 'latest Pydantic AI release version', 'num_results': '10'}),
            ('web_search', {'query': 'pydantic-ai PyPI', 'num_results': '5'}),
            ('web_search', {'query': 'pydantic-ai GitHub releases', 'num_results': '5'}),
        ]
    )
    assert response.usage.details == snapshot({'reasoning_tokens': 311, 'server_side_tools_web_search': 3})
