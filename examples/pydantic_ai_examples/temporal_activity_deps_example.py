"""Example script demonstrating activity-level dependency injection with TemporalAgent.

This script shows how to inject non-serializable dependencies (like httpx clients)
into Temporal activities at worker startup.

Prerequisites:
    - Temporal server running locally: `temporal server start-dev`
    - Install dependencies: `pip install pydantic-ai[temporal] httpx`

Run with:
    python -m pydantic_ai_examples.temporal_activity_deps_example
"""

from __future__ import annotations

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
from pydantic_ai.models.test import TestModel


@dataclass
class ActivityDeps:
    """Non-serializable dependencies injected at worker startup."""

    http_client: httpx.AsyncClient


# Create agent with a tool that uses the injected httpx client
agent = Agent(
    TestModel(
        call_tools=['check_httpbin']
    ),  # Use TestModel for demo (no API key needed)
    name='http_demo_agent',
)


@agent.tool
async def check_httpbin(ctx: RunContext[None]) -> str:
    """Check httpbin.org using the injected httpx client."""
    # Get the activity deps - this works because we're inside a Temporal activity
    deps = get_activity_deps()

    if deps is None:
        return 'Error: No activity deps available'

    # Use the injected httpx client to make a real HTTP request
    url = 'https://httpbin.org/get'
    response = await deps.http_client.get(url)
    return f'Fetched {url} - Status: {response.status_code}, Content length: {len(response.content)} bytes'


# Wrap the agent for Temporal
temporal_agent = TemporalAgent(agent)


@workflow.defn
class HttpDemoWorkflow:
    """Workflow that uses the agent to check httpbin."""

    @workflow.run
    async def run(self) -> str:
        result = await temporal_agent.run('Check httpbin')
        return result.output


async def main():
    print('Connecting to Temporal server at localhost:7233...')

    # Connect to Temporal server
    client = await Client.connect(
        'localhost:7233',
        plugins=[PydanticAIPlugin()],
    )

    print('Connected! Creating httpx client and worker...')

    # Create the httpx client that will be injected into activities
    async with httpx.AsyncClient() as http_client:
        # Create activity deps with the httpx client
        activity_deps = ActivityDeps(http_client=http_client)

        # Start the worker with the agent plugin and activity deps
        async with Worker(
            client,
            task_queue='http-demo-queue',
            workflows=[HttpDemoWorkflow],
            plugins=[
                AgentPlugin(temporal_agent, activity_deps=activity_deps),
            ],
        ):
            print('Worker started! Executing workflow...\n')

            # Execute the workflow
            result = await client.execute_workflow(
                HttpDemoWorkflow.run,
                id=f'http-demo-{uuid.uuid4()}',
                task_queue='http-demo-queue',
            )

            print(f'Workflow result: {result}')
            print(
                '\nSuccess! The httpx client was injected into the activity and used to fetch the URL.'
            )


if __name__ == '__main__':
    asyncio.run(main())
