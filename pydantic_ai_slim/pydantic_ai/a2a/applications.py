from __future__ import annotations as _annotations

from collections.abc import Sequence
from typing import Any

from pydantic import TypeAdapter
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route
from starlette.types import ExceptionHandler, Lifespan

from pydantic_ai import Agent
from pydantic_ai.a2a.history_manager import HistoryManager
from pydantic_ai.a2a.task_manager import InMemoryTaskManager, TaskManager

from .schema import A2ARequest, Skill, agent_card_ta

a2a_request_ta: TypeAdapter[A2ARequest] = TypeAdapter(A2ARequest)

default_task_manager = InMemoryTaskManager()


class FastA2A(Starlette):
    """The main class for the FastA2A library."""

    def __init__(
        self,
        agent: Agent,
        *,
        task_manager: TaskManager = default_task_manager,
        history_manager: HistoryManager,
        description: str | None = None,
        skills: list[Skill] | None = None,
        # Starlette
        debug: bool = False,
        routes: Sequence[Route] | None = None,
        middleware: Sequence[Middleware] | None = None,
        exception_handlers: dict[Any, ExceptionHandler] | None = None,
        lifespan: Lifespan[FastA2A] | None = None,
    ):
        super().__init__(
            debug=debug,
            routes=routes,
            middleware=middleware,
            exception_handlers=exception_handlers,
            lifespan=lifespan,
        )

        self.agent = agent
        assert self.agent.name is not None, 'The agent needs to have a name to create an A2A server.'
        self.name = self.agent.name
        self.description = description
        self.skills = skills or []
        # NOTE: For now, I don't think there's any reason to support any other input/output modes.
        self.default_input_modes = ['application/json']
        self.default_output_modes = ['application/json']

        self.task_manager = task_manager
        self.history_manager = history_manager

        # Setup
        self._agent_card_json_schema: bytes | None = None
        self.router.add_route('/.well-known/agent.json', self.agent_card_endpoint, methods=['HEAD', 'GET', 'OPTIONS'])
        self.router.add_route('/', self.agent_run_endpoint, methods=['POST'])

    async def agent_card_endpoint(self, request: Request) -> Response:
        if self._agent_card_json_schema is None:
            self._agent_card_json_schema = agent_card_ta.dump_json(
                {
                    'name': self.name,
                    'skills': self.skills,
                    'default_input_modes': self.default_input_modes,
                    'default_output_modes': self.default_output_modes,
                }
            )
        return Response(content=self._agent_card_json_schema, media_type='application/json')

    async def agent_run_endpoint(self, request: Request) -> Response:
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
            await self.task_manager.send_task(a2a_request)
        elif a2a_request['method'] == 'tasks/get':
            await self.task_manager.get_task(a2a_request)
        elif a2a_request['method'] == 'tasks/cancel':
            ...
        else:
            raise NotImplementedError(f'Method {a2a_request["method"]} not implemented.')
