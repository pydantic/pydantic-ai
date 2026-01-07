"""Main entry point for the Temporal streaming example.

This module sets up the Temporal client and worker, executes the workflow,
and polls for streaming events to display to the user.
"""

import asyncio
import os
import uuid
from typing import Any

from temporalio.client import Client, WorkflowHandle
from temporalio.worker import Worker

from pydantic_ai.durable_exec.temporal import AgentPlugin, PydanticAIPlugin
from .agents import build_agent
from .datamodels import EventKind, EventStream
from .streaming_handler import streaming_handler
from .utils import read_config_yml
from .workflow import YahooFinanceSearchWorkflow


async def poll_events(workflow_handle: WorkflowHandle[Any, str]) -> None:
    """
    Poll for events from the workflow and print them.

    Args:
        workflow_handle: Handle to the running workflow.
    """
    while True:
        event: EventStream | None = await workflow_handle.query('event_stream',
                                                                result_type=EventStream | None)  # type: ignore[misc]
        if event is None:
            await asyncio.sleep(0.1)
            continue

        if event.kind == EventKind.CONTINUE_CHAT:
            print('\n--- Workflow completed ---')
            break
        elif event.kind == EventKind.RESULT:
            print(f'\n=== Final Result ===\n{event.content}\n')
        elif event.kind == EventKind.EVENT:
            print(f'\n--- Event ---\n{event.content}\n')


async def main() -> None:
    """
    Main function to set up and run the Temporal workflow.

    This function:
    1. Connects to the Temporal server
    2. Builds the agent and registers activities
    3. Starts a worker
    4. Executes the workflow
    5. Polls for streaming events
    """
    # Connect to Temporal server
    client = await Client.connect(
        # target_host='localhost:7233',
        target_host='localhost:7233',
        plugins=[PydanticAIPlugin()],
    )
    config_path = os.getenv('APP_CONFIG_PATH', './app_conf.yml')
    confs = read_config_yml(config_path)

    # Build the agent with streaming handler
    temporal_agent = await build_agent(streaming_handler, **confs['llm'])

    # Define the task queue
    task_queue = 'yahoo-finance-search'

    # Start the worker
    async with Worker(
            client,
            task_queue=task_queue,
            workflows=[YahooFinanceSearchWorkflow],
            activities=[YahooFinanceSearchWorkflow.retrieve_env_vars],
            plugins=[AgentPlugin(temporal_agent)],
    ):
        # Execute the workflow
        workflow_id = f'yahoo-finance-search-{uuid.uuid4()}'
        workflow_handle: WorkflowHandle[Any, str] = await client.start_workflow(  # type: ignore[misc]
            'YahooFinanceSearchWorkflow',
            arg='What are the latest financial metrics for Apple (AAPL)?',
            id=workflow_id,
            task_queue=task_queue,
            result_type=str
        )

        print(f'Started workflow with ID: {workflow_id}')
        print('Polling for events...\n')

        # Poll for events in the background
        event_task = asyncio.create_task(poll_events(workflow_handle))

        # Wait for workflow to complete
        result = await workflow_handle.result()

        # Ensure event polling is done
        await event_task

        print(f'\nWorkflow completed with result: {result}')


if __name__ == '__main__':
    asyncio.run(main())
