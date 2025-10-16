"""Temporal workflow for streaming agent execution.

This module defines the Temporal workflow that orchestrates the agent execution
with streaming capabilities, using signals and queries for event communication.
"""

import asyncio
import os
from collections import deque
from datetime import timedelta
from typing import Any

from pydantic_ai import UsageLimits
from temporalio import activity, workflow

from .agents import build_agent
from .datamodels import AgentDependencies, EventKind, EventStream
from .streaming_handler import streaming_handler
from .utils import read_config_yml


@workflow.defn
class YahooFinanceSearchWorkflow:
    """
    Temporal workflow for executing the Yahoo Finance search agent with streaming.

    This workflow manages the agent execution, collects streaming events via signals,
    and exposes them through queries for consumption by external systems.
    """

    def __init__(self):
        """Initialize the workflow with an empty event queue."""
        self.events: deque[EventStream] = deque()

    @workflow.run
    async def run(self, user_prompt: str):
        """
        Execute the agent with the given user prompt.

        Args:
            user_prompt: The user's question or request.

        Returns:
            The agent's final output.
        """
        # Retrieve environment variables from configuration
        wf_vars = await workflow.execute_activity(
            activity='retrieve_env_vars',
            start_to_close_timeout=timedelta(seconds=10),
            result_type=dict[str, Any],
        )

        # Create dependencies with workflow identification for signal routing
        deps = AgentDependencies(workflow_id=workflow.info().workflow_id, run_id=workflow.info().run_id)

        # Build and run the agent
        agent = await build_agent(streaming_handler, **wf_vars)
        result = await agent.run(
            user_prompt=user_prompt, usage_limits=UsageLimits(request_limit=50), deps=deps
        )

        # Signal the final result
        await self.append_event(event_stream=EventStream(kind=EventKind.RESULT, content=result.output))

        # Signal completion
        await self.append_event(event_stream=EventStream(kind=EventKind.CONTINUE_CHAT, content=''))

        # Wait for events to be consumed before completing
        try:
            await workflow.wait_condition(
                lambda: len(self.events) == 0,
                timeout=timedelta(seconds=10),
                timeout_summary='Waiting for events to be consumed',
            )
            return result.output
        except asyncio.TimeoutError:
            return result.output

    @staticmethod
    @activity.defn(name='retrieve_env_vars')
    async def retrieve_env_vars():
        """
        Retrieve environment variables from configuration file.

        Returns:
            Dictionary containing API keys and other configuration.
        """
        config_path = os.getenv('APP_CONFIG_PATH', './app_conf.yml')
        configs = read_config_yml(config_path)
        return {'anthropic_api_key': configs['llm']['anthropic_api_key']}

    @workflow.query
    def event_stream(self) -> EventStream | None:
        """
        Query to retrieve the next event from the stream.

        Returns:
            The next event if available, None otherwise.
        """
        if self.events:
            return self.events.popleft()
        return None

    @workflow.signal
    async def append_event(self, event_stream: EventStream):
        """
        Signal to append a new event to the stream.

        This is called by the streaming handler running in activities.

        Args:
            event_stream: The event to append.
        """
        self.events.append(event_stream)
