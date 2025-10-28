"""AG-UI protocol integration for Pydantic AI agents."""

from typing import Any

from pydantic_ai.agent import AbstractAgent
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT

from ..app import UIApp
from ._adapter import AGUIAdapter


class AGUIApp(UIApp[AgentDepsT, OutputDataT]):
    """ASGI application for running Pydantic AI agents with AG-UI protocol support."""

    def __init__(self, agent: AbstractAgent[AgentDepsT, OutputDataT], **kwargs: Any):
        super().__init__(AGUIAdapter[AgentDepsT, OutputDataT], agent, **kwargs)
