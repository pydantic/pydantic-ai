from __future__ import annotations as _annotations

from typing import TYPE_CHECKING, Any, Sequence

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps.jsonrpc import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import (
    AgentCard,
    Capabilities,
    InvalidParamsError,
    MessageSendParams,
    TaskIdParams,
    TaskState,
)
from a2a.utils.errors import ServerError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.types import ExceptionHandler, Lifespan

from .storage import Storage
from .worker import Worker

if TYPE_CHECKING:
    from .schema import Provider, Skill


class _WorkerExecutor(AgentExecutor):
    """An adapter to make a fasta2a.Worker compatible with a2a.AgentExecutor."""

    def __init__(self, worker: Worker, storage: Storage):
        self.worker = worker
        self.storage = storage

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.server.tasks import TaskUpdater

        self.worker.storage = self.storage

        if not (context.task_id and context.context_id and context.message):
            raise ServerError(
                InvalidParamsError(
                    message="task_id, context_id, and message are required for execution"
                )
            )

        params = MessageSendParams(
            message=context.message, configuration=context.configuration
        )

        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await self.worker.run_task(params, updater)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.server.tasks import TaskUpdater

        self.worker.storage = self.storage

        if not context.task_id or not context.context_id:
            raise ServerError(
                InvalidParamsError(
                    message="task_id and context_id are required for cancellation"
                )
            )

        params = TaskIdParams(id=context.task_id)
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await self.worker.cancel_task(params, updater)
        await updater.update_status(TaskState.canceled, final=True)


class FastA2A:
    """The main class for the FastA2A library."""

    def __init__(
        self,
        *,
        storage: Storage,
        worker: Worker,
        # Agent card
        name: str | None = None,
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        description: str | None = None,
        provider: Provider | None = None,
        skills: list[Skill] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[Route] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: dict[Any, ExceptionHandler] | None = None,
        lifespan: Lifespan | None = None,
    ):
        agent_executor = _WorkerExecutor(worker, storage)

        request_handler = DefaultRequestHandler(
            agent_executor=agent_executor, task_store=storage
        )

        agent_card = AgentCard(
            name=name or "Agent",
            url=url,
            version=version,
            description=description,
            provider=provider,
            skills=skills or [],
            defaultInputModes=["application/json"],
            defaultOutputModes=["application/json"],
            capabilities=Capabilities(
                streaming=True, pushNotifications=False, stateTransitionHistory=True
            ),
        )

        app_builder = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )
        self.app: Starlette = app_builder.build(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            lifespan=lifespan,
        )

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)
