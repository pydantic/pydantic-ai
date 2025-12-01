"""Example of a durable Pydantic AI agent using Temporal for fault-tolerant execution.

This example demonstrates:
1. Wrapping an agent with TemporalAgent for durable execution
2. Injecting non-serializable dependencies (httpx client) via activity_deps
3. Using both serializable deps (user context) and activity deps (HTTP client)

The key difference from regular deps:
- `deps` (serializable): Passed per-run, serialized by Temporal (e.g., user IDs, config)
- `activity_deps` (non-serializable): Initialized at worker startup (e.g., clients, pools)

Prerequisites:
    - Start Temporal server: `temporal server start-dev`
    - Set OPENAI_API_KEY environment variable

Run with:
    uv run -m pydantic_ai_examples.temporal_durable_agent
"""

from __future__ import annotations as _annotations

import asyncio
import uuid
from dataclasses import dataclass

import httpx
from temporalio import workflow
from temporalio.client import Client
from temporalio.worker import Worker

from pydantic_ai import Agent, RunContext
from pydantic_ai.durable_exec.temporal import (
    AgentPlugin,
    PydanticAIPlugin,
    TemporalAgent,
    get_activity_deps,
)


# Serializable deps - passed per agent run
@dataclass
class Deps:
    user_id: str


# Non-serializable activity deps - initialized at worker startup
@dataclass
class ActivityDeps:
    http_client: httpx.AsyncClient


# Create the agent
weather_agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=Deps,
    name='weather_agent',
    instructions='You help users check the weather. Use the get_weather tool.',
)


@weather_agent.tool
async def get_weather(ctx: RunContext[Deps], city: str) -> str:
    """Get weather for a city using the injected HTTP client.

    Args:
        ctx: The run context with user info.
        city: The city to get weather for.
    """
    # Get the activity deps (httpx client) injected at worker startup
    activity_deps: ActivityDeps | None = get_activity_deps()
    if activity_deps is None:
        return 'Weather service unavailable'

    # Use the injected httpx client to make a request
    response = await activity_deps.http_client.get(
        'https://demo-endpoints.pydantic.workers.dev/weather',
        params={'city': city},
    )
    response.raise_for_status()

    return f'Weather in {city} for user {ctx.deps.user_id}: {response.text}'


# Wrap for Temporal - this freezes the model and tools
temporal_agent = TemporalAgent(weather_agent)


@workflow.defn
class WeatherWorkflow:
    """Durable workflow that fetches weather using the agent."""

    @workflow.run
    async def run(self, city: str, user_id: str) -> str:
        deps = Deps(user_id=user_id)
        result = await temporal_agent.run(f'What is the weather in {city}?', deps=deps)
        return result.output


async def main():
    print('Connecting to Temporal server...')
    client = await Client.connect('localhost:7233', plugins=[PydanticAIPlugin()])

    # Create the HTTP client at worker startup - this is NOT serialized
    async with httpx.AsyncClient() as http_client:
        activity_deps = ActivityDeps(http_client=http_client)

        print('Starting worker with injected httpx client...')
        async with Worker(
            client,
            task_queue='weather-queue',
            workflows=[WeatherWorkflow],
            plugins=[AgentPlugin(temporal_agent, activity_deps=activity_deps)],
        ):
            print('Executing workflow...\n')
            result = await client.execute_workflow(
                WeatherWorkflow.run,
                args=['London', 'user-123'],
                id=f'weather-{uuid.uuid4()}',
                task_queue='weather-queue',
            )
            print(f'Result: {result}')


if __name__ == '__main__':
    asyncio.run(main())
