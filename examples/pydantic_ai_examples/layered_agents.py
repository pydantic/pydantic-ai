"""Layered agents over the OpenResponses protocol (in-process demo).

Three layers stacked over the OpenResponses wire:

- **L2** — a weather agent with a `get_weather` tool, exposed via `agent.beta.to_responses()`.
- **L1** — an outer "guardrail/enrichment" agent that calls L2 using
  [`OpenResponsesModel`][pydantic_ai.models.openresponses.OpenResponsesModel]. Before its call
  reaches L2, L1 injects user-profile context that L2 should treat as developer/context input.
- **L0** — a vanilla caller that hits L1's `agent.beta.to_responses()` endpoint via raw httpx,
  showing the full set of streaming events L1 emits — including the `pydantic_ai:custom_tool_call`
  items that round-trip L2's backend tool execution.

The demo runs entirely in-process via `httpx.ASGITransport`, so no real network is needed.

Run with:

    uv run -m pydantic_ai_examples.layered_agents
"""

from __future__ import annotations

import asyncio
import json

import httpx
from openai import AsyncOpenAI

from pydantic_ai import Agent
from pydantic_ai.messages import AgentContextPart, ModelResponse, TextPart
from pydantic_ai.models.openresponses import OpenResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

L2_AGENT_NAME = 'weather'
L1_AGENT_NAME = 'guardrail'


def get_weather(city: str) -> str:
    """Stand-in weather lookup."""
    return json.dumps({'city': city, 'condition': 'sunny', 'temperature_c': 22})


l2_agent = Agent(
    model='openai:gpt-5.2',
    name=L2_AGENT_NAME,
    instructions=(
        'You answer weather questions concisely. If a user has a loyalty status mentioned '
        'in injected context, weave a short bonus mention of a perk for their tier.'
    ),
    tools=[get_weather],
)


def build_l1(l2_client: httpx.AsyncClient) -> Agent[None, str]:
    """Outer guardrail agent. Uses `OpenResponsesModel` pointing at the L2 in-process app."""
    inner_model = OpenResponsesModel(
        model_name=L2_AGENT_NAME,
        provider=OpenAIProvider(base_url='http://l2/v1', http_client=l2_client),
    )
    return Agent(
        model=inner_model,
        name=L1_AGENT_NAME,
        instructions=(
            'You enrich user requests with profile context before forwarding to the weather agent. '
            'User profile: gold-tier loyalty member.'
        ),
    )


async def main() -> None:
    l2_app = l2_agent.beta.to_responses(mode='openresponses')

    transport = httpx.ASGITransport(app=l2_app)
    async with httpx.AsyncClient(
        transport=transport, base_url='http://l2'
    ) as l2_client:
        l1_agent = build_l1(l2_client)

        result = await l1_agent.run("What's the weather in Paris?")
        print('=== L1 result text ===')
        print(result.output)
        print()
        print('=== L1 message history (parts in last response) ===')
        for message in result.all_messages():
            if isinstance(message, ModelResponse):
                for part in message.parts:
                    label = type(part).__name__
                    if isinstance(part, TextPart):
                        print(f'  {label}: {part.content!r}')
                    elif isinstance(part, AgentContextPart):
                        print(
                            f'  {label}: from={part.from_agent} role={part.role} content={part.content!r}'
                        )
                    else:
                        print(f'  {label}: {part!r}')

        l1_app = l1_agent.beta.to_responses(mode='openresponses')
        l1_transport = httpx.ASGITransport(app=l1_app)
        async with httpx.AsyncClient(
            transport=l1_transport, base_url='http://l1'
        ) as l1_client:
            openai_client = AsyncOpenAI(
                base_url='http://l1/v1', api_key='unused', http_client=l1_client
            )
            print()
            print('=== L0 streaming view (event types emitted by L1) ===')
            async with openai_client.responses.stream(
                model=L1_AGENT_NAME,
                input='Could you check the weather in Tokyo as well?',
            ) as stream:
                async for event in stream:
                    print(event.type)


if __name__ == '__main__':
    asyncio.run(main())
