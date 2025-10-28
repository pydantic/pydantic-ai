"""Vercel UI protocol integration for Pydantic AI agents."""

from typing import Any

from pydantic_ai.agent import AbstractAgent
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT

from ..app import UIApp
from ._adapter import VercelAIAdapter


class VercelAIApp(UIApp[AgentDepsT, OutputDataT]):
    """ASGI application for running Pydantic AI agents with Vercel UI protocol support."""

    def __init__(self, agent: AbstractAgent[AgentDepsT, OutputDataT], **kwargs: Any):
        super().__init__(VercelAIAdapter[AgentDepsT, OutputDataT], agent, **kwargs)
