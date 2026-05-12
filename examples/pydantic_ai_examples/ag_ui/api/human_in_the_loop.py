"""Human in the Loop Feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from textwrap import dedent

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent(
    'openai:gpt-5-mini',
    instructions=dedent(
        """
        When planning tasks use tools only, without any other messages.
        IMPORTANT:
        - Use the `generate_task_steps` tool to display the suggested steps to the user
        - Never repeat the plan, or send a message detailing steps
        - If accepted, confirm the creation of the plan and the number of selected (enabled) steps only
        - If not accepted, ask the user for more information, DO NOT use the `generate_task_steps` tool again
        """
    ),
)


async def run_agent(request: Request) -> Response:
    return await AGUIAdapter.dispatch_request(request, agent=agent)


app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])
