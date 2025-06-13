from __future__ import annotations as _annotations

from typing import Any

import httpx
from a2a.server.apps.jsonrpc.starlette_app import A2AStarletteApplication
from a2a.server.request_handlers.default_request_handler import DefaultRequestHandler
from a2a.server.tasks.inmemory_push_notifier import InMemoryPushNotifier
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentProvider
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.types import ExceptionHandler, Lifespan, Receive, Scope, Send

from .schema import Skill
from .storage import Storage
from .worker import Worker


class FastA2A:
    """
    The main class for the FastA2A library. It provides a simple way to create
    an A2A server by wrapping the Google A2A SDK.
    """

    def __init__(
        self,
        *,
        worker: Worker,
        storage: Storage | None = None,
        # Agent card
        name: str = "Agent",
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        description: str | None = None,
        provider: AgentProvider | None = None,
        skills: list[Skill] | None = None,
        # Starlette
        debug: bool = False,
        routes: list[Route] | None = None,
        middleware: list[Middleware] | None = None,
        exception_handlers: dict[Any, ExceptionHandler] | None = None,
        lifespan: Lifespan | None = None,
    ):
        """
        Initializes the FastA2A application.

        Args:
            worker: An implementation of `fasta2a.Worker` (which is an `a2a.server.agent_execution.AgentExecutor`).
            storage: An implementation of `fasta2a.Storage` (which is an `a2a.server.tasks.TaskStore`).
                Defaults to `InMemoryTaskStore`.
            name: The human-readable name of the agent.
            url: The URL where the agent is hosted.
            version: The version of the agent.
            description: A human-readable description of the agent.
            provider: The service provider of the agent.
            skills: A list of skills the agent can perform.
            debug: Starlette's debug flag.
            routes: A list of additional Starlette routes.
            middleware: A list of Starlette middleware.
            exception_handlers: A dictionary of Starlette exception handlers.
            lifespan: A Starlette lifespan context manager.
        """
        self.agent_card = AgentCard(
            name=name,
            url=url,
            version=version,
            description=description or "A FastA2A Agent",
            provider=provider,
            skills=skills or [],
            capabilities=AgentCapabilities(
                streaming=True, pushNotifications=True, stateTransitionHistory=True
            ),
            defaultInputModes=["application/json"],
            defaultOutputModes=["application/json"],
            securitySchemes={},
        )

        self.storage = storage or InMemoryTaskStore()
        self.worker = worker

        # The SDK's DefaultRequestHandler uses httpx to send push notifications
        http_client = httpx.AsyncClient()
        push_notifier = InMemoryPushNotifier(httpx_client)

        request_handler = DefaultRequestHandler(
            agent_executor=self.worker,
            task_store=self.storage,
            push_notifier=push_notifier,
        )

        a2a_app = A2AStarletteApplication(
            agent_card=self.agent_card,
            http_handler=request_handler,
        )

        self.app = a2a_app.build(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            lifespan=lifespan,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app(scope, receive, send)
