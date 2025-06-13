from __future__ import annotations as _annotations

from typing import Any

from a2a.server.agent_execution import AgentExecutor
from a2a.server.apps.jsonrpc import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import TaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentProvider, AgentSkill as Skill
from starlette.middleware import Middleware
from starlette.routing import Route
from starlette.types import Receive, Scope, Send


class FastA2A:
    """The main class for the FastA2A library."""

    def __init__(
        self,
        *,
        worker: AgentExecutor,
        storage: TaskStore,
        # Agent card
        name: str | None = None,
        url: str = "http://localhost:8000",
        version: str = "1.0.0",
        description: str | None = None,
        provider: AgentProvider | None = None,
        skills: list[Skill] | None = None,
        # Starlette
        routes: list[Route] | None = None,
        middleware: list[Middleware] | None = None,
        **starlette_kwargs: Any,
    ):
        self.worker = worker
        self.storage = storage
        self.name = name or "Agent"
        self.url = url
        self.version = version
        self.description = description
        self.provider = provider
        self.skills = skills or []
        self.default_input_modes = ["application/json"]
        self.default_output_modes = ["application/json"]
        self.capabilities = AgentCapabilities(
            streaming=True,
            pushNotifications=False,
            stateTransitionHistory=False,
        )

        agent_card = AgentCard(
            name=self.name,
            url=self.url,
            version=self.version,
            description=self.description or "",
            provider=self.provider,
            skills=self.skills,
            defaultInputModes=self.default_input_modes,
            defaultOutputModes=self.default_output_modes,
            capabilities=self.capabilities,
        )

        handler = DefaultRequestHandler(
            agent_executor=self.worker,
            task_store=self.storage,
        )

        self._app = A2AStarletteApplication(
            agent_card=agent_card, http_handler=handler
        ).build(routes=routes, middleware=middleware, **starlette_kwargs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self._app(scope, receive, send)
