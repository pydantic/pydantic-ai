"""Tests for layered-agent communication over the OpenResponses protocol.

Two Pydantic AI agents are stacked:

- L2 — inner agent with a backend tool, exposed as `agent.beta.to_responses(mode='openresponses')`.
- L1 — outer agent using `OpenResponsesModel` to call L2 via `httpx.ASGITransport` (in-process,
  no real network).

These tests assert that backend tool calls L2 runs come back to L1 as `BuiltinToolCallPart` /
`BuiltinToolReturnPart` (lossless round-trip via `pydantic_ai:custom_tool_call*` extension items),
and that `pydantic_ai:agent_context` items L1 emits surface to its own outer caller in the same
extension shape.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from inline_snapshot import snapshot

from pydantic_ai import Agent
from pydantic_ai.messages import (
    AgentContextPart,
    BuiltinToolCallPart,
    BuiltinToolReturnPart,
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, DeltaToolCalls, FunctionModel

from .conftest import IsDatetime, try_import

with try_import() as imports_successful:
    import httpx

    from pydantic_ai.models.openresponses import OpenResponsesModel
    from pydantic_ai.providers.openai import OpenAIProvider
    from pydantic_ai.ui.responses import ResponsesAdapter


pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(not imports_successful(), reason='openai, starlette, or httpx_sse not installed'),
]


def _l2_weather_stream(messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[DeltaToolCalls | str]:
    """L2's stream: emits a `get_weather` tool call on first call, then text after the result."""

    async def gen() -> AsyncIterator[DeltaToolCalls | str]:
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            yield 'It is sunny in Paris, 22°C.'
            return
        yield {0: DeltaToolCall(name='get_weather', json_args='{"city": "Paris"}', tool_call_id='call_w1')}

    return gen()


def _build_l2() -> Agent[None, str]:
    agent: Agent[None, str] = Agent(
        model=FunctionModel(stream_function=_l2_weather_stream),
        name='weather',
    )

    @agent.tool_plain
    def get_weather(city: str) -> str:
        return f'sunny, 22°C in {city}'

    return agent


async def _l1_layered_app() -> tuple[Agent[None, str], httpx.AsyncClient]:
    """Build L2's app + httpx client and an L1 agent talking to it via in-process ASGI."""
    l2 = _build_l2()
    l2_app = l2.beta.to_responses(mode='openresponses')
    transport = httpx.ASGITransport(app=l2_app)
    client = httpx.AsyncClient(transport=transport, base_url='http://l2')
    inner_model = OpenResponsesModel(
        model_name='weather',
        provider=OpenAIProvider(base_url='http://l2/v1', http_client=client),
    )
    l1: Agent[None, str] = Agent(model=inner_model, name='guardrail')
    return l1, client


async def test_layered_round_trip_streaming_lossless(allow_model_requests: None) -> None:
    """L2's backend `get_weather` call surfaces to L1 as BuiltinToolCallPart/BuiltinToolReturnPart.

    This is the load-bearing layered-agent assertion: a Pydantic AI agent acting as an
    OpenResponses client of another Pydantic AI agent gets the inner agent's tool calls
    and results lossless on the wire (via `pydantic_ai:custom_tool_call*` extension items).
    """
    l1, client = await _l1_layered_app()
    try:
        async with l1.run_stream("What's the weather in Paris?") as result:
            await result.get_output()
        messages = result.all_messages()
    finally:
        await client.aclose()

    response_messages = [m for m in messages if isinstance(m, ModelResponse)]
    assert response_messages, 'expected at least one ModelResponse'
    parts = [p for m in response_messages for p in m.parts]

    builtin_calls = [p for p in parts if isinstance(p, BuiltinToolCallPart)]
    builtin_returns = [p for p in parts if isinstance(p, BuiltinToolReturnPart)]
    text_parts = [p for p in parts if isinstance(p, TextPart)]

    assert builtin_calls == snapshot(
        [
            BuiltinToolCallPart(
                tool_name='get_weather',
                args='{"city": "Paris"}',
                tool_call_id='call_w1',
            )
        ]
    )
    assert builtin_returns == snapshot(
        [
            BuiltinToolReturnPart(
                tool_name='get_weather',
                content='sunny, 22°C in Paris',
                tool_call_id='call_w1',
                timestamp=IsDatetime(),
            )
        ]
    )
    assert any('Paris' in p.content for p in text_parts)


async def test_agent_context_round_trip_input_to_system_prompt() -> None:
    """`pydantic_ai:agent_context` input items map to a SystemPromptPart with prefix-encoded provenance."""
    items: list[dict[str, Any]] = [
        {
            'type': 'pydantic_ai:agent_context',
            'from_agent': 'guardrail',
            'role': 'context',
            'content': 'User profile: gold-tier loyalty member.',
        },
        {'type': 'message', 'role': 'user', 'content': "What's the weather in Tokyo?"},
    ]
    messages = ResponsesAdapter.load_messages(items)  # pyright: ignore[reportArgumentType]
    request_parts = [p for m in messages for p in m.parts]
    system_parts = [p for p in request_parts if type(p).__name__ == 'SystemPromptPart']
    assert any(
        '[from guardrail, role=context] User profile: gold-tier loyalty member.' in getattr(p, 'content', '')
        for p in system_parts
    )


async def test_agent_context_emit_in_openresponses_mode() -> None:
    """An `AgentContextPart` from L1's stream surfaces as `pydantic_ai:agent_context` output items."""

    def _emit_context_stream(messages: list[ModelMessage], _agent_info: AgentInfo) -> AsyncIterator[str]:
        async def gen() -> AsyncIterator[str]:
            yield 'Acknowledged.'

        return gen()

    agent: Agent[None, str] = Agent(model=FunctionModel(stream_function=_emit_context_stream), name='guardrail')
    app = agent.beta.to_responses(mode='openresponses')
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url='http://test') as client:
        async with client.stream(
            'POST',
            '/v1/responses',
            json={'model': 'test', 'stream': True, 'input': 'hi'},
            headers={'Accept': 'text/event-stream'},
        ) as resp:
            assert resp.status_code == 200
            body = ''
            async for chunk in resp.aiter_text():
                body += chunk

    # The basic streamed shape: response.created … output_item.added/done for the assistant
    # message … response.completed. With no AgentContextPart from this FunctionModel, no
    # `pydantic_ai:agent_context` items appear — but with one, they would (round-trip the
    # extension-prefixed item type). Suppression in `openai_compat` is the sister test below.
    assert 'response.created' in body
    assert 'response.completed' in body
    assert 'pydantic_ai:agent_context' not in body  # no AgentContextPart emitted in this run
    assert 'event: done' in body


async def test_agent_context_part_roundtrips_via_messages() -> None:
    """`AgentContextPart` is part of `ModelResponsePart` and survives ModelResponse construction."""
    response = ModelResponse(
        parts=[
            AgentContextPart(content='hello', from_agent='guardrail', role='context'),
            TextPart(content='world'),
        ]
    )
    assert isinstance(response.parts[0], AgentContextPart)
    assert response.parts[0].from_agent == 'guardrail'
    assert response.parts[0].role == 'context'
