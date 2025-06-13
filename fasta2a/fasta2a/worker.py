from __future__ import annotations as _annotations

from abc import ABC
from contextlib import asynccontextmanager

from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater


class Worker(AgentExecutor, ABC):
    """
    An abstract class for implementing the core logic of an A2A agent.

    This class inherits from the `a2a.server.agent_execution.AgentExecutor`
    and must be subclassed to define the agent's behavior.
    """

    @asynccontextmanager
    async def task_updater(
        self, context: RequestContext, event_queue: EventQueue
    ) -> TaskUpdater:
        """
        A convenience context manager to get a `TaskUpdater` for the current task.

        Args:
            context: The `RequestContext` for the current execution.
            event_queue: The `EventQueue` to publish updates to.

        Yields:
            A `TaskUpdater` instance for the current task.
        """
        if not context.task_id or not context.context_id:
            raise ValueError(
                "RequestContext must have a task_id and context_id to create a TaskUpdater."
            )
        yield TaskUpdater(event_queue, context.task_id, context.context_id)
