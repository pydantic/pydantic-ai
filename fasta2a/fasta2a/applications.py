from __future__ import annotations as _annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from pydantic import TypeAdapter
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import ExceptionHandler, Lifespan, Receive, Scope, Send

from .schema import A2ARequest, A2AResponse, AgentCard, Provider, Skill, agent_card_ta
from .storage import Storage
from .task_manager import TaskManager
from .worker import Worker

a2a_request_ta: TypeAdapter[A2ARequest] = TypeAdapter(A2ARequest)
a2a_response_ta: TypeAdapter[A2AResponse] = TypeAdapter(A2AResponse)


class FastA2A(Starlette):
    """The main class for the FastA2A library."""

    def __init__(
        self,
        *,
        storage: Storage,
        worker: Worker,
        # Agent card
        name: str | None = None,
        url: str = 'http://localhost:8000',
        version: str = '1.0.0',
        description: str | None = None,
        provider: Provider | None = None,
        skills: list[Skill] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[Route] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: dict[Any, ExceptionHandler] | None = None,
        lifespan: Lifespan[FastA2A] | None = None,
    ):
        if lifespan is None:
            lifespan = _default_lifespan

        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            lifespan=lifespan,
        )

        self.name = name or 'Agent'
        self.url = url
        self.version = version
        self.description = description
        self.provider = provider
        self.skills = skills or []
        # NOTE: For now, I don't think there's any reason to support any other input/output modes.
        self.default_input_modes = ['application/json']
        self.default_output_modes = ['application/json']

        self.task_manager = TaskManager(worker=worker, storage=storage)

        # Setup
        self._agent_card_json_schema: bytes | None = None
        self.router.add_route('/.well-known/agent.json', self._agent_card_endpoint, methods=['HEAD', 'GET', 'OPTIONS'])
        self.router.add_route('/', self._agent_run_endpoint, methods=['POST'])

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await super().__call__(scope, receive, send)

    async def _agent_card_endpoint(self, request: Request) -> Response:
        if self._agent_card_json_schema is None:
            agent_card = AgentCard(
                name=self.name,
                url=self.url,
                version=self.version,
                skills=self.skills,
                default_input_modes=self.default_input_modes,
                default_output_modes=self.default_output_modes,
            )
            if self.description is not None:
                agent_card['description'] = self.description
            if self.provider is not None:
                agent_card['provider'] = self.provider
            self._agent_card_json_schema = agent_card_ta.dump_json(agent_card)
        return Response(content=self._agent_card_json_schema, media_type='application/json')

    async def _agent_run_endpoint(self, request: Request) -> Response:
        """This is the main endpoint for the A2A server.

        Although the specification allows freedom of choice and implementation, I'm pretty sure about some decisions.

        1. The server will always either send a "submitted" or a "failed" on `tasks/send`.
            Never a "completed" on the first message.
        2. There are three possible ends for the task:
            2.1. The task was "completed" successfully.
            2.2. The task was "canceled".
            2.3. The task "failed".
        3. The server will send a "working" on the first chunk on `tasks/pushNotification/get`.
        """
        data = await request.body()
        a2a_request = a2a_request_ta.validate_json(data)

        if a2a_request['method'] == 'tasks/send':
            jsonrpc_response = await self.task_manager.send_task(a2a_request)
        elif a2a_request['method'] == 'tasks/get':
            jsonrpc_response = await self.task_manager.get_task(a2a_request)
        elif a2a_request['method'] == 'tasks/cancel':
            jsonrpc_response = await self.task_manager.cancel_task(a2a_request)
        else:
            raise NotImplementedError(f'Method {a2a_request["method"]} not implemented.')
        return Response(content=a2a_response_ta.dump_json(jsonrpc_response), media_type='application/json')


@asynccontextmanager
async def _default_lifespan(app: FastA2A) -> AsyncIterator[None]:
    async with app.task_manager:
        yield
